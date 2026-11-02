// SPDX-License-Identifier: MIT OR Apache-2.0
//! [`check_module`]: a [`Module`] of kernels in which any may [`Stmt::Call`] another
//! (`docs/options-u3-calls.md`, Q3 and Q5), split from `kernel_check.rs` so neither outgrows the size the
//! analysers accept.

use std::collections::BTreeSet;

use crate::exec::MemorySpace;
use crate::kernel::{Block, KernelError, KernelFn, Module, Stmt};
use crate::kernel_check::{CalleeSig, check_kernel_with};

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
    let edges = call_edges(m)?;
    no_cycle(&edges)?;
    let called: BTreeSet<u32> = edges.iter().flatten().copied().collect();
    for f in called {
        plain_function(&m.fns[f as usize])?;
    }
    let sigs = signatures(m)?;
    let reach = reachable(&edges, entry);
    let launch = m.fns[entry as usize].launch;
    for (i, f) in m.fns.iter().enumerate() {
        let inside = u32::try_from(i).is_ok_and(|i| i != entry && reach.contains(&i));
        if inside && launch.is_some() {
            // a callee runs inside the entry's launch: its ids and barriers are the entry's
            let mut g = f.clone();
            g.launch = launch;
            check_kernel_with(&g, sigs.clone())?;
        } else {
            check_kernel_with(f, sigs.clone())?;
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
            Stmt::Call { callee, .. } => out.push(*callee),
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

/// A callee runs on the calling thread: no launch of its own, only private allocations, no barrier.
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
    if holds_barrier(&f.body) {
        return refuse(
            "a callee holding a Barrier (the transitive divergence check has not landed)",
        );
    }
    Ok(())
}

fn holds_barrier(b: &Block) -> bool {
    b.0.iter().any(|st| match st {
        Stmt::Barrier { .. } => true,
        Stmt::If { then, els, .. } => holds_barrier(then) || holds_barrier(els),
        Stmt::For { body, .. } | Stmt::While { body, .. } => holds_barrier(body),
        _ => false,
    })
}

fn signatures(m: &Module) -> Result<Vec<CalleeSig>, KernelError> {
    m.fns
        .iter()
        .map(|f| {
            let result = match f.result {
                None => None,
                Some(l) => match f.locals.get(l as usize) {
                    Some(d) => Some(d.ty()),
                    None => return Err(KernelError::UnknownLocal { local: l }),
                },
            };
            Ok(CalleeSig {
                params: f.params.iter().map(|p| p.kind.clone()).collect(),
                result,
            })
        })
        .collect()
}
