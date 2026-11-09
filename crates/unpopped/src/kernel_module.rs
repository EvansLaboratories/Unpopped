// SPDX-License-Identifier: MIT OR Apache-2.0
//! [`check_module`]: a [`Module`] of kernels in which any may [`Stmt::Call`] another
//! (`docs/options-u3-calls.md`, Q3 and Q5), split from `kernel_check.rs` so neither outgrows the size the
//! analysers accept.

use std::collections::BTreeSet;

use crate::exec::MemorySpace;
use crate::kernel::{Block, CollectiveKind, ExternDecl, KernelError, KernelFn, Module, Stmt};
use crate::kernel_check::{CalleeSig, ExternSig, check_kernel_with};

/// Check every function of `m`, with `entry` the function a launch starts at.
///
/// A `Call` resolves against the module: the callee exists, arity and argument kinds and types match, a
/// buffer is passed by alias with a role the caller has, and the result local has the callee's result type.
/// The call graph must be acyclic. A function that is called is a plain function: no launch, no
/// `Shared` or `Constant` allocation, no barrier (a barrier through a call needs the transitive
/// divergence check, a later slice).
///
/// # Errors
///
/// The first [`KernelError`], functions in index order.
pub fn check_module(m: &Module, entry: u32) -> Result<(), KernelError> {
    if m.fns.get(entry as usize).is_none() {
        return Err(KernelError::BadCallee { callee: entry });
    }
    crate::kernel_extern::check_extern_decls(m)?;
    let edges = call_edges(m)?;
    no_cycle(&edges)?;
    let called: BTreeSet<u32> = edges.iter().flatten().copied().collect();
    for f in called {
        plain_function(&m.fns[f as usize])?;
    }
    let sigs = signatures(m, &edges)?;
    let externs = extern_sigs(m);
    let reach = reachable(&edges, entry);
    let launch = m.fns[entry as usize].launch;
    for (i, f) in m.fns.iter().enumerate() {
        let inside = u32::try_from(i).is_ok_and(|i| i != entry && reach.contains(&i));
        if inside && launch.is_some() {
            // a callee runs inside the entry's launch: its ids and barriers are the entry's
            let mut g = f.clone();
            g.launch = launch;
            check_kernel_with(&g, sigs.clone(), externs.clone())?;
        } else {
            check_kernel_with(f, sigs.clone(), externs.clone())?;
        }
    }
    Ok(())
}

/// The functions `entry` can reach through calls.
fn reachable(edges: &[Vec<u32>], entry: u32) -> BTreeSet<u32> {
    let mut seen = BTreeSet::new();
    let mut todo = vec![entry];
    while let Some(f) = todo.pop() {
        if seen.insert(f) {
            todo.extend(edges[f as usize].iter().copied());
        }
    }
    seen
}

/// The callees of each function, by index; a callee outside the module is `BadCallee`.
fn call_edges(m: &Module) -> Result<Vec<Vec<u32>>, KernelError> {
    let mut edges = Vec::with_capacity(m.fns.len());
    for f in &m.fns {
        let mut out = Vec::new();
        collect_calls(&f.body, &mut out);
        if let Some(bad) = out.iter().find(|c| (**c as usize) >= m.fns.len()) {
            return Err(KernelError::BadCallee { callee: *bad });
        }
        edges.push(out);
    }
    Ok(edges)
}

fn collect_calls(b: &Block, out: &mut Vec<u32>) {
    for st in &b.0 {
        match st {
            Stmt::Call { callee, .. } | Stmt::CallTuple { callee, .. } => out.push(*callee),
            // a combining collective calls its combine function once per lane
            Stmt::Collective {
                kind: CollectiveKind::ReduceWith { combine } | CollectiveKind::ScanWith { combine },
                ..
            }
            | Stmt::CollectiveTuple {
                kind: CollectiveKind::ReduceWith { combine } | CollectiveKind::ScanWith { combine },
                ..
            } => out.push(*combine),
            Stmt::If { then, els, .. } => {
                collect_calls(then, out);
                collect_calls(els, out);
            }
            Stmt::For { body, .. } | Stmt::While { body, .. } => collect_calls(body, out),
            _ => {}
        }
    }
}

/// A depth-first search with a visiting set; a back edge is a cycle.
fn no_cycle(edges: &[Vec<u32>]) -> Result<(), KernelError> {
    // 0 = unvisited, 1 = on the stack, 2 = done
    let mut state = vec![0u8; edges.len()];
    for start in 0..edges.len() {
        visit(start, edges, &mut state)?;
    }
    Ok(())
}

fn visit(f: usize, edges: &[Vec<u32>], state: &mut [u8]) -> Result<(), KernelError> {
    match state[f] {
        1 => {
            return Err(KernelError::CallCycle {
                callee: u32::try_from(f).unwrap_or(u32::MAX),
            });
        }
        2 => return Ok(()),
        _ => {}
    }
    state[f] = 1;
    for c in &edges[f] {
        visit(*c as usize, edges, state)?;
    }
    state[f] = 2;
    Ok(())
}

/// A callee runs on the calling thread: no launch of its own and only private allocations (a barrier is
/// allowed: a call that reaches one is a barrier site, checked by the uniformity rule).
fn plain_function(f: &KernelFn) -> Result<(), KernelError> {
    let refuse = |what| Err(KernelError::CallUnsupported { what });
    if f.launch.is_some() {
        return refuse("a callee with a launch shape (a callee runs on the calling thread)");
    }
    if f.allocs.iter().any(|a| a.space != MemorySpace::Private) {
        return refuse(
            "a callee that declares a Shared or Constant allocation (the kernel owns those)",
        );
    }
    Ok(())
}

/// Whether `b` holds a rendezvous: a `Barrier`, a `Collective`, an `Mma` or an `ExternCall` declared `sync` (a call
/// that reaches one is a barrier site).
fn holds_barrier(b: &Block, externs: &[ExternDecl]) -> bool {
    b.0.iter().any(|st| match st {
        Stmt::Barrier { .. }
        | Stmt::Collective { .. }
        | Stmt::CollectiveTuple { .. }
        | Stmt::Mma { .. } => true,
        Stmt::ExternCall { decl, .. } => externs
            .get(*decl as usize)
            .is_some_and(|d| d.effects.sync.is_some()),
        Stmt::If { then, els, .. } => holds_barrier(then, externs) || holds_barrier(els, externs),
        Stmt::For { body, .. } | Stmt::While { body, .. } => holds_barrier(body, externs),
        _ => false,
    })
}

/// The extern declarations as the checker reads them.
fn extern_sigs(m: &Module) -> Vec<ExternSig> {
    m.externs
        .iter()
        .map(|d| ExternSig {
            params: d.params.clone(),
            results: d.results.clone(),
            pure: d.effects.pure,
            sync: d.effects.sync,
        })
        .collect()
}

/// Whether each function reaches a barrier, directly or through its calls (the graph is acyclic here).
fn reaches_barrier(m: &Module, edges: &[Vec<u32>]) -> Vec<bool> {
    fn go(f: usize, m: &Module, edges: &[Vec<u32>], memo: &mut [Option<bool>]) -> bool {
        if let Some(v) = memo[f] {
            return v;
        }
        let mut v = holds_barrier(&m.fns[f].body, &m.externs);
        for c in &edges[f] {
            v |= go(*c as usize, m, edges, memo);
        }
        memo[f] = Some(v);
        v
    }
    let mut memo = vec![None; m.fns.len()];
    (0..m.fns.len())
        .map(|f| go(f, m, edges, &mut memo))
        .collect()
}

fn signatures(m: &Module, edges: &[Vec<u32>]) -> Result<Vec<CalleeSig>, KernelError> {
    let barrier = reaches_barrier(m, edges);
    m.fns
        .iter()
        .zip(barrier)
        .map(|(f, barrier)| {
            let result = match f.result {
                None => None,
                Some(l) => match f.locals.get(l as usize) {
                    Some(d) => Some(d.ty()),
                    None => return Err(KernelError::UnknownLocal { local: l }),
                },
            };
            let extra = f
                .results
                .iter()
                .map(|l| match f.locals.get(*l as usize) {
                    Some(d) => Ok(d.ty()),
                    None => Err(KernelError::UnknownLocal { local: *l }),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(CalleeSig {
                params: f.params.iter().map(|p| p.kind.clone()).collect(),
                result,
                extra,
                barrier,
            })
        })
        .collect()
}
