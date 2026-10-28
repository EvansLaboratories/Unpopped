// SPDX-License-Identifier: MIT OR Apache-2.0
//! Source-language → IR converters (IR-hub Phase 4, `convert` feature) — the
//! source-language side of the translation hub, built on **tree-sitter**.
//!
//! Instead of a hand-written parser (the pilot in `lift.rs`), these frontends
//! reuse mature tree-sitter grammars (here `tree-sitter-slang`; CUDA's lives in
//! Baracuda's `baracuda-cuda-parse`, see below): parse
//! source into an error-tolerant CST, recognize the expressible idioms
//! (grid-stride / dispatch elementwise, …) into the neutral IR (`OpDef`), and
//! keep whatever doesn't map as a typed [`LiftError`] residue
//! ([`LiftError::Unrecognized`] / [`LiftError::Inexpressible`]).
//! Error tolerance is a feature — unrecognized constructs stay as intact subtrees
//! we refuse rather than mis-lift; portability scales with the lift fraction.
//!
//! Because `tree-sitter-slang` is built on `tree-sitter-cpp`, a Slang elementwise
//! store parses into the SAME `assignment_expression` / `subscript_expression` /
//! `binary_expression` node shapes as CUDA (`tree-sitter-cuda` is built on it
//! too) — so ONE CST walk serves every C-family language, differing only in
//! naming conventions (`out`/`in` are HLSL keywords, so Slang uses
//! `output`/`input{k}`) and the kernel/residue markers.
//!
//! # The shape: one walk, many [`Frontend`]s
//!
//! That sharing used to be implicit — eight public wrappers, one per
//! (language × op-class), each repeating the same four-step preamble before
//! calling a shared walker. It is now explicit: a [`Frontend`] value carries
//! everything a language contributes (kernel marker, residue list, buffer
//! naming, and a `parse` function pointer), and [`lift_elementwise`],
//! [`lift_reduction`], [`lift_scan`] and [`lift`] take one.
//!
//! [`SLANG`] is the one supplied here. A caller defines any other, and
//! `a_caller_defined_frontend_lifts_through_the_shared_walkers` proves the walkers
//! read the descriptor rather than hardcoding this crate's.
//!
//! # CUDA is not here (0.30.0)
//!
//! `parse_cuda`, the CUDA residue list, the `CUDA` descriptor and the
//! `tree-sitter-cuda` dependency moved out: Baracuda owns CUDA in this ecosystem
//! and supplies the parser as `baracuda-cuda-parse` (and the emitter as
//! `baracuda-cuda-emit`). That crate depends on this one for [`Frontend`] and the
//! shared walkers, so this crate cannot depend back (a cycle) and carries no CUDA
//! tests; the CUDA golden vectors live in `baracuda-cuda-parse/tests/convert.rs`.
//! `tests/neutral_core_names_no_cuda_grammar.rs` guards the absence.
//!
//! The grammar crate is named in exactly one place ([`parse_slang`]).

use crate::ir::{BinaryOp, Expr, OpDef, ReduceOp, ScalarExpr, UnaryOp};
use crate::lift::{LiftError, Lifted, binary_fn, unary_fn};
use tree_sitter::{Node, Parser, Tree};
use unpopped_vocab::ElementKind;

/// Parse Slang source into a tree-sitter CST.
pub fn parse_slang(src: &str) -> Option<Tree> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_slang::LANGUAGE_SLANG.into())
        .ok()?;
    parser.parse(src, None)
}

/// Slang constructs that aren't IR-expressible — group-shared memory, barriers, atomics.
const SLANG_RESIDUE: &[&str] = &[
    "groupshared",
    "GroupMemoryBarrier",
    "DeviceMemoryBarrier",
    "AllMemoryBarrier",
    "InterlockedAdd",
    "InterlockedCompareExchange",
    "RWByteAddressBuffer",
];

/// What a source language contributes to lifting — everything that is NOT the
/// CST walk.
///
/// # Why this type exists
///
/// Measured before writing it: this module is ~620 non-test lines, of which the
/// language-specific surface is **two grammar references, two residue lists, and
/// eight thin wrappers** that repeat the same four-step preamble (kernel marker
/// → residue check → parse → walk) three times per language. Eleven private
/// walkers do the actual work and **both languages share all eleven** —
/// `tree-sitter-slang` is built on `tree-sitter-cpp`, so a Slang elementwise
/// store parses into the same `assignment_expression` / `subscript_expression`
/// node shapes as CUDA.
///
/// So the parser in the neutral crate is really a **C-family CST walker** that a
/// language feeds, and the language-specific part is a grammar binding, not
/// parsing logic. Naming the seam has two payoffs: the duplicated preamble
/// collapses to one function, and the grammar dependency is isolated behind a
/// `fn` pointer — which is what made moving CUDA's into a per-target crate
/// (`baracuda-cuda-parse`, which supplies its own [`Frontend`]) mechanical
/// rather than a rewrite.
///
/// # Ingest is not the same coupling as emit
///
/// Worth stating, because the two look alike and are not. What a generator
/// **emits** is its vendor coupling: a kernel spelling `__half2float` only runs
/// on CUDA. What it **reads** is not — a C compiler that parses C is not thereby
/// a C program. Reading CUDA source into neutral IR is a frontend, and the
/// reason it moved out is narrower than neutrality: it is the grammar
/// *dependency*, not the act of parsing.
#[derive(Clone, Copy, Debug)]
pub struct Frontend {
    /// Language name, for diagnostics.
    pub name: &'static str,
    /// A substring whose presence marks source as a kernel in this language.
    /// Absent ⇒ [`LiftError::NotAKernel`] before any parsing is attempted.
    pub kernel_marker: &'static str,
    /// Constructs this frontend refuses as residue — hand-optimized machinery
    /// (shared memory, atomics, barriers, library calls) that belongs in the
    /// source language rather than the neutral IR.
    pub residue: &'static [&'static str],
    /// The output buffer's parameter name.
    pub out_name: &'static str,
    /// The input buffer prefix, so `{in_prefix}{K}` names operand `K`.
    pub in_prefix: &'static str,
    /// Parse source into an error-tolerant CST.
    ///
    /// A function pointer rather than a generic or a trait object: it is the
    /// **only** member that needs a grammar crate, so keeping it a plain pointer
    /// is what lets a target-specific crate hand one in without this crate
    /// depending on any grammar at all.
    pub parse: fn(&str) -> Option<Tree>,
}

/// Slang: `numthreads` compute kernels, `output`/`input{K}` buffers.
///
/// The buffer names differ from CUDA's because `out` and `in` are **HLSL
/// keywords** — a language constraint, not a style choice.
pub const SLANG: Frontend = Frontend {
    name: "slang",
    kernel_marker: "numthreads",
    residue: SLANG_RESIDUE,
    out_name: "output",
    in_prefix: "input",
    parse: parse_slang,
};

/// The four-step preamble every lifter shares: kernel marker, residue refusal,
/// parse, then hand the CST to a walker.
///
/// Ordering is deliberate. The marker check is a substring test and comes first
/// so non-kernel input costs nothing. The residue check comes **before** parsing
/// so an `atomicAdd` kernel is refused as *inexpressible* rather than parsed and
/// then found unrecognizable — different diagnoses for different facts, and the
/// caller acts on them differently.
fn prepared(fe: &Frontend, src: &str) -> Result<Tree, LiftError> {
    if !src.contains(fe.kernel_marker) {
        return Err(LiftError::NotAKernel);
    }
    reject_markers(src, fe.residue)?;
    (fe.parse)(src).ok_or(LiftError::NotAKernel)
}

/// Lift a grid-stride / dispatch **elementwise** kernel into an [`OpDef`] by
/// walking its tree-sitter CST.
///
/// Recognizes `{out_name}[i] = <expr>;` where the body is arithmetic over
/// `{in_prefix}K[i]`, literals, and math intrinsics; refuses anything else as a
/// typed [`LiftError`] residue.
pub fn lift_elementwise(
    fe: &Frontend,
    src: &str,
    name: &str,
    dtypes: &[ElementKind],
) -> Result<Lifted, LiftError> {
    let tree = prepared(fe, src)?;
    lift_store(&tree, src, name, dtypes, fe.out_name, fe.in_prefix)
}

/// Lift a naive accumulator-loop full **reduction** into an [`OpDef`].
///
/// Reduce-op mapping: `acc += e` / `acc = acc + e` → [`ReduceOp::Sum`];
/// `acc *= e` / `acc = acc * e` → [`ReduceOp::Prod`]; `acc = fmaxf(acc, e)` →
/// [`ReduceOp::Max`]; `fminf` → [`ReduceOp::Min`]. This recognizes the IDIOM,
/// not the loop bounds — the reduced extent is the caller's `StructureKey`, set
/// at generate time.
pub fn lift_reduction(
    fe: &Frontend,
    src: &str,
    name: &str,
    dtypes: &[ElementKind],
) -> Result<Lifted, LiftError> {
    let tree = prepared(fe, src)?;
    lift_reduce_store(&tree, src, name, dtypes, fe.out_name, fe.in_prefix)
}

/// Lift a naive inclusive-forward **scan** (prefix cumulative) into an
/// [`OpDef`].
///
/// Recognizes `T acc = init; for(...) { acc <reduce> <pre>; out[i] = acc; }` —
/// the running accumulator stored INSIDE the loop at `out[i]` (vs a reduction's
/// `out[0]` after it). Monoid from the update (`+=` → cumsum, `*=` → cumprod,
/// `fmaxf` → cummax, `fminf` → cummin); `<pre>` is the per-element pre-map
/// (reuses the elementwise walk). Lifts as an inclusive forward scan on
/// `axis = 0` (the 1-D reading of the flat loop); a higher-rank row-scan sets
/// the axis via a matching-rank `StructureKey`. `Mean` is not a monoid, and
/// `fmaxf`/`fminf`/`+=`/`*=` never produce it, so it cannot arise here.
pub fn lift_scan(
    fe: &Frontend,
    src: &str,
    name: &str,
    dtypes: &[ElementKind],
) -> Result<Lifted, LiftError> {
    let tree = prepared(fe, src)?;
    lift_scan_store(&tree, src, name, dtypes, fe.out_name, fe.in_prefix)
}

/// Lift a kernel of unknown op-class: elementwise, then reduction, then scan
/// (the three are mutually exclusive by store shape).
pub fn lift(
    fe: &Frontend,
    src: &str,
    name: &str,
    dtypes: &[ElementKind],
) -> Result<Lifted, LiftError> {
    lift_elementwise(fe, src, name, dtypes)
        .or_else(|_| lift_reduction(fe, src, name, dtypes))
        .or_else(|_| lift_scan(fe, src, name, dtypes))
}

/// Refuse the first construct we cannot express in the IR — it belongs in the
/// source language (residue).
fn reject_markers(src: &str, markers: &[&str]) -> Result<(), LiftError> {
    for m in markers {
        if src.contains(m) {
            return Err(LiftError::Inexpressible((*m).to_string()));
        }
    }
    Ok(())
}

/// Shared tail: find the `{out_name}[i] = <rhs>` store, walk the RHS into a
/// `ScalarExpr` (`{in_prefix}{K}[i]` → `Input(K)`), and build the [`OpDef`].
fn lift_store(
    tree: &Tree,
    src: &str,
    name: &str,
    dtypes: &[ElementKind],
    out_name: &str,
    in_prefix: &str,
) -> Result<Lifted, LiftError> {
    let (idx_var, rhs) =
        find_out_store(tree.root_node(), src, out_name).ok_or(LiftError::NotElementwise)?;
    let mut w = Walk {
        src,
        idx_var,
        in_prefix,
        max_input: None,
        locals: local_bindings(tree.root_node(), src),
        resolving: Vec::new(),
    };
    let body = w.expr(rhs)?;
    let n_inputs = w.max_input.map_or(0, |m| m + 1);
    Ok(Lifted {
        op: OpDef::elementwise(name, n_inputs, dtypes, Expr(body)),
        n_inputs,
    })
}

/// Shared scan tail: find the in-loop `{out_name}[i] = acc` running store, find
/// the reduce update of `acc`, walk its per-element `pre`, and build an inclusive
/// forward [`OpDef::scan`] on axis 0 (`post = reduced(0)`, the plain prefix).
fn lift_scan_store(
    tree: &Tree,
    src: &str,
    name: &str,
    dtypes: &[ElementKind],
    out_name: &str,
    in_prefix: &str,
) -> Result<Lifted, LiftError> {
    let acc = find_running_out_store(tree.root_node(), src, out_name).ok_or_else(|| {
        LiftError::Unrecognized(format!("no running `{out_name}[i] = acc` scan store"))
    })?;
    let (op, pre_node) = find_accumulation(tree.root_node(), src, &acc).ok_or_else(|| {
        LiftError::Unrecognized(format!("no recognizable scan update of `{acc}`"))
    })?;
    let idx_var = first_input_index(pre_node, src, in_prefix)
        .ok_or_else(|| LiftError::Unrecognized("scan body reads no inK[idx]".into()))?;
    let mut w = Walk {
        src,
        idx_var,
        in_prefix,
        max_input: None,
        locals: local_bindings(tree.root_node(), src),
        resolving: Vec::new(),
    };
    let pre = w.expr(pre_node)?;
    let n_inputs = w.max_input.map_or(0, |m| m + 1);
    Ok(Lifted {
        op: OpDef::scan(
            name,
            n_inputs,
            dtypes,
            op,
            0,     // axis — 1-D reading; caller sets higher rank via the key
            false, // reverse
            false, // exclusive — naive `acc op= in; out[i] = acc` is inclusive
            Expr(pre),
            crate::ir::reduced(0),
        ),
        n_inputs,
    })
}

/// Find an in-loop `{out_name}[<identifier>] = <identifier>` store (the running
/// accumulator written at the scanned axis) and return the accumulator ident.
/// The identifier index (vs. a numeric literal) distinguishes a scan's running
/// store from a reduction's scalar `out[0] = acc`; the bare-identifier RHS
/// distinguishes it from an elementwise `out[i] = <expr>`.
fn find_running_out_store(node: Node, src: &str, out_name: &str) -> Option<String> {
    if node.kind() == "assignment_expression"
        && let (Some(lhs), Some(rhs)) = (
            node.child_by_field_name("left"),
            node.child_by_field_name("right"),
        )
        && lhs.kind() == "subscript_expression"
        && rhs.kind() == "identifier"
        && lhs
            .child_by_field_name("argument")
            .and_then(|a| a.utf8_text(src.as_bytes()).ok())
            == Some(out_name)
        && subscript_index(lhs, src).is_some()
    {
        return Some(rhs.utf8_text(src.as_bytes()).ok()?.to_string());
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if let Some(hit) = find_running_out_store(child, src, out_name) {
            return Some(hit);
        }
    }
    None
}

/// Shared reduction tail: find `{out_name}[<lit>] = acc`, find the reduce update
/// of `acc`, walk the per-element body, and build a last-axis
/// [`OpDef::reduction`].
fn lift_reduce_store(
    tree: &Tree,
    src: &str,
    name: &str,
    dtypes: &[ElementKind],
    out_name: &str,
    in_prefix: &str,
) -> Result<Lifted, LiftError> {
    let acc = find_scalar_out_store(tree.root_node(), src, out_name).ok_or_else(|| {
        LiftError::Unrecognized(format!("no scalar `{out_name}[..] = acc` store"))
    })?;
    let (op, body_node) = find_accumulation(tree.root_node(), src, &acc).ok_or_else(|| {
        LiftError::Unrecognized(format!("no recognizable reduce update of `{acc}`"))
    })?;
    let idx_var = first_input_index(body_node, src, in_prefix)
        .ok_or_else(|| LiftError::Unrecognized("reduction body reads no inK[idx]".into()))?;
    let mut w = Walk {
        src,
        idx_var,
        in_prefix,
        max_input: None,
        locals: local_bindings(tree.root_node(), src),
        resolving: Vec::new(),
    };
    let body = w.expr(body_node)?;
    let n_inputs = w.max_input.map_or(0, |m| m + 1);
    Ok(Lifted {
        op: OpDef::reduction(name, n_inputs, dtypes, Expr(body), op),
        n_inputs,
    })
}

/// Find a `{out_name}[<number>] = <ident>` store and return the stored
/// accumulator identifier. The numeric index (vs. an identifier) is what
/// distinguishes a scalar reduction store from an elementwise `out[i] = ...`.
fn find_scalar_out_store(node: Node, src: &str, out_name: &str) -> Option<String> {
    if node.kind() == "assignment_expression"
        && let (Some(lhs), Some(rhs)) = (
            node.child_by_field_name("left"),
            node.child_by_field_name("right"),
        )
        && lhs.kind() == "subscript_expression"
        && rhs.kind() == "identifier"
        && lhs
            .child_by_field_name("argument")
            .and_then(|a| a.utf8_text(src.as_bytes()).ok())
            == Some(out_name)
        && let Some(indices) = lhs.child_by_field_name("indices")
    {
        let mut c = indices.walk();
        let kids: Vec<Node> = indices.named_children(&mut c).collect();
        if kids.len() == 1 && kids[0].kind() == "number_literal" {
            return Some(rhs.utf8_text(src.as_bytes()).ok()?.to_string());
        }
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if let Some(hit) = find_scalar_out_store(child, src, out_name) {
            return Some(hit);
        }
    }
    None
}

/// Find the reduce-update assignment of `acc` — `acc += e`, `acc *= e`,
/// `acc = acc <+|*> e`, or `acc = fmaxf/fminf(acc, e)` — returning the reduce op
/// and the per-element body node `e`.
fn find_accumulation<'t>(node: Node<'t>, src: &str, acc: &str) -> Option<(ReduceOp, Node<'t>)> {
    if node.kind() == "assignment_expression"
        && let (Some(lhs), Some(op), Some(rhs)) = (
            node.child_by_field_name("left"),
            node.child_by_field_name("operator"),
            node.child_by_field_name("right"),
        )
        && lhs.kind() == "identifier"
        && lhs.utf8_text(src.as_bytes()).ok() == Some(acc)
    {
        match op.utf8_text(src.as_bytes()).unwrap_or("") {
            "+=" => return Some((ReduceOp::Sum, rhs)),
            "*=" => return Some((ReduceOp::Prod, rhs)),
            "=" => {
                if let Some(hit) = classify_reduce_rhs(rhs, src, acc) {
                    return Some(hit);
                }
            }
            _ => {}
        }
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if let Some(hit) = find_accumulation(child, src, acc) {
            return Some(hit);
        }
    }
    None
}

/// Classify the RHS of `acc = <rhs>` as a reduce update: `acc <+|*> e` (acc on
/// either side) or `fmaxf/fminf(acc, e)`. Returns the op + the body `e`.
fn classify_reduce_rhs<'t>(rhs: Node<'t>, src: &str, acc: &str) -> Option<(ReduceOp, Node<'t>)> {
    let is_acc =
        |n: Node| n.kind() == "identifier" && n.utf8_text(src.as_bytes()).ok() == Some(acc);
    match rhs.kind() {
        "parenthesized_expression" => classify_reduce_rhs(rhs.named_child(0)?, src, acc),
        "binary_expression" => {
            let l = rhs.child_by_field_name("left")?;
            let r = rhs.child_by_field_name("right")?;
            let other = if is_acc(l) {
                r
            } else if is_acc(r) {
                l
            } else {
                return None;
            };
            match rhs
                .child_by_field_name("operator")?
                .utf8_text(src.as_bytes())
                .unwrap_or("")
            {
                "+" => Some((ReduceOp::Sum, other)),
                "*" => Some((ReduceOp::Prod, other)),
                _ => None,
            }
        }
        "call_expression" => {
            let fname = rhs
                .child_by_field_name("function")?
                .utf8_text(src.as_bytes())
                .ok()?;
            let args_node = rhs.child_by_field_name("arguments")?;
            let mut c = args_node.walk();
            let args: Vec<Node> = args_node.named_children(&mut c).collect();
            if args.len() != 2 {
                return None;
            }
            let other = if is_acc(args[0]) {
                args[1]
            } else if is_acc(args[1]) {
                args[0]
            } else {
                return None;
            };
            let op = match fname {
                "fmaxf" | "fmax" | "max" => ReduceOp::Max,
                "fminf" | "fmin" | "min" => ReduceOp::Min,
                _ => return None,
            };
            Some((op, other))
        }
        _ => None,
    }
}

/// The index identifier of the first `{in_prefix}{K}[idx]` read in `node` — the
/// reduced-axis loop variable the body iterates over.
fn first_input_index(node: Node, src: &str, in_prefix: &str) -> Option<String> {
    if node.kind() == "subscript_expression"
        && let Some(base) = node
            .child_by_field_name("argument")
            .and_then(|a| a.utf8_text(src.as_bytes()).ok())
        && base
            .strip_prefix(in_prefix)
            .and_then(|d| d.parse::<u8>().ok())
            .is_some()
    {
        return subscript_index(node, src);
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if let Some(idx) = first_input_index(child, src, in_prefix) {
            return Some(idx);
        }
    }
    None
}

/// Find the `{out_name}[<idx>] = <rhs>` store (an `assignment_expression` whose
/// LHS is a `subscript_expression` on `out_name`) and return `(idx_var, rhs)`.
fn find_out_store<'t>(node: Node<'t>, src: &str, out_name: &str) -> Option<(String, Node<'t>)> {
    if node.kind() == "assignment_expression"
        && let (Some(lhs), Some(rhs)) = (
            node.child_by_field_name("left"),
            node.child_by_field_name("right"),
        )
        && lhs.kind() == "subscript_expression"
        && lhs
            .child_by_field_name("argument")
            .and_then(|a| a.utf8_text(src.as_bytes()).ok())
            == Some(out_name)
        && let Some(idx) = subscript_index(lhs, src)
    {
        return Some((idx, rhs));
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if let Some(hit) = find_out_store(child, src, out_name) {
            return Some(hit);
        }
    }
    None
}

/// The single index identifier inside a `subscript_expression`'s `indices`, or
/// `None` if it isn't a lone identifier.
fn subscript_index(sub: Node, src: &str) -> Option<String> {
    let indices = sub.child_by_field_name("indices")?;
    let mut cursor = indices.walk();
    let kids: Vec<Node> = indices.named_children(&mut cursor).collect();
    if kids.len() != 1 || kids[0].kind() != "identifier" {
        return None;
    }
    Some(kids[0].utf8_text(src.as_bytes()).ok()?.to_string())
}

/// Names assigned exactly once in the whole tree, mapped to the single
/// expression node that value came from: either a declaration's initializer
/// (`TYPE x = <expr>;`) or the sole plain reassignment of an
/// already-declared name (`x = <expr>;`, LHS a bare identifier, never a
/// buffer subscript). A name assigned more than once — declared then
/// reassigned, or reassigned more than once — is deliberately absent: this
/// scan does not choose among competing definitions, it only recognizes the
/// case where there is exactly one.
///
/// # Why this is safe where a blind text substitution was not
///
/// This is consulted in exactly one place — [`Walk::expr`]'s `identifier`
/// arm, evaluating a name that appears as a VALUE inside the store's RHS
/// expression. It never touches the store-shape scanners
/// (`find_out_store`/`find_scalar_out_store`/`find_running_out_store`,
/// `find_accumulation`) that run on the raw tree BEFORE a [`Walk`] exists —
/// those look for `{out_name}[i] = <ident>` and `acc <op>= <expr>` by
/// structural shape, never through this map, so a scan/reduction
/// accumulator's own name cannot be silently resolved out from under them
/// the way a global text substitution did (`docs/slang-second-blocker-sizing.md`).
fn local_bindings<'t>(root: Node<'t>, src: &str) -> Vec<(String, Node<'t>)> {
    let mut sites: Vec<(String, Node<'t>)> = Vec::new();
    let mut mutations: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    collect_binding_sites(root, src, &mut sites, &mut mutations);

    sites
        .into_iter()
        .filter(|(name, _)| mutations.get(name) == Some(&1))
        .collect()
}

/// Every `(name, value-node)` a bare identifier is assigned at via a
/// declaration's initializer or a plain (`=`) reassignment — candidate
/// substitution values. `mutations` separately counts EVERY way a bare
/// identifier's value changes, plain assignment included, but ALSO
/// compound assignment (`+=`/`-=`/…) and increment/decrement (`++`/`--`),
/// none of which contribute a candidate value but all of which must still
/// disqualify a name from the single-assignment fast path.
///
/// Counting compound assignment is not optional: `float acc = 0.0f; ... acc
/// += in0[i]; out[i] = acc;` declares `acc` exactly once by the narrower
/// (declaration-or-`=`) count, so a scanner that only watched `=` would
/// treat `acc` as "safely bound to 0.0" and let a scan's own running
/// accumulator get folded into a bogus constant elementwise store — caught
/// by `scan_reduction_elementwise_are_disjoint` failing exactly that way
/// before this counter existed.
fn collect_binding_sites<'t>(
    n: Node<'t>,
    src: &str,
    sites: &mut Vec<(String, Node<'t>)>,
    mutations: &mut std::collections::HashMap<String, u32>,
) {
    if n.kind() == "init_declarator"
        && let (Some(decl), Some(value)) = (
            n.child_by_field_name("declarator"),
            n.child_by_field_name("value"),
        )
        && decl.kind() == "identifier"
        && let Ok(name) = decl.utf8_text(src.as_bytes())
    {
        sites.push((name.to_string(), value));
        *mutations.entry(name.to_string()).or_insert(0) += 1;
    } else if n.kind() == "assignment_expression"
        && let (Some(left), Some(op), Some(right)) = (
            n.child_by_field_name("left"),
            n.child_by_field_name("operator"),
            n.child_by_field_name("right"),
        )
        && left.kind() == "identifier"
        && let Ok(name) = left.utf8_text(src.as_bytes())
    {
        *mutations.entry(name.to_string()).or_insert(0) += 1;
        if op.utf8_text(src.as_bytes()) == Ok("=") {
            sites.push((name.to_string(), right));
        }
    } else if matches!(n.kind(), "update_expression")
        && let Some(arg) = n.named_child(0)
        && arg.kind() == "identifier"
        && let Ok(name) = arg.utf8_text(src.as_bytes())
    {
        // `x++` / `--x` — no substitutable value, but still a mutation.
        *mutations.entry(name.to_string()).or_insert(0) += 1;
    }
    let mut cursor = n.walk();
    for child in n.named_children(&mut cursor) {
        collect_binding_sites(child, src, sites, mutations);
    }
}

struct Walk<'a> {
    src: &'a str,
    idx_var: String,
    in_prefix: &'a str,
    max_input: Option<u8>,
    /// See [`local_bindings`]. Consulted only when [`Walk::expr`] hits a bare
    /// `identifier` it cannot otherwise place — never used to rewrite source,
    /// never touching the store/accumulator scanners.
    locals: Vec<(String, Node<'a>)>,
    /// Names currently being resolved through `locals`, so a self-referential
    /// or mutually-referential binding declines as a typed miss instead of
    /// recursing forever.
    resolving: Vec<String>,
}

impl<'a> Walk<'a> {
    fn text(&self, n: Node) -> &'a str {
        n.utf8_text(self.src.as_bytes()).unwrap_or("")
    }

    fn expr(&mut self, n: Node) -> Result<ScalarExpr, LiftError> {
        match n.kind() {
            "parenthesized_expression" => {
                let inner = n
                    .named_child(0)
                    .ok_or_else(|| LiftError::Unrecognized("()".into()))?;
                self.expr(inner)
            }
            "binary_expression" => {
                let l = self.field(n, "left")?;
                let r = self.field(n, "right")?;
                let op = self.field(n, "operator")?;
                let (le, re) = (self.expr(l)?, self.expr(r)?);
                Ok(match self.text(op) {
                    "+" => ScalarExpr::Add(Box::new(le), Box::new(re)),
                    "-" => ScalarExpr::Sub(Box::new(le), Box::new(re)),
                    "*" => ScalarExpr::Mul(Box::new(le), Box::new(re)),
                    "/" => ScalarExpr::Div(Box::new(le), Box::new(re)),
                    // Comparisons — never reachable as a store's own value
                    // without `conditional_expression` below, which is the
                    // only place a Cmp* result gets consumed (as a Select
                    // condition). Kept here rather than duplicated: the
                    // grammar parses `a < b` as an ordinary binary_expression
                    // whether it sits inside `? :` or not.
                    "==" => ScalarExpr::Binary(BinaryOp::CmpEq, Box::new(le), Box::new(re)),
                    "!=" => ScalarExpr::Binary(BinaryOp::CmpNe, Box::new(le), Box::new(re)),
                    "<" => ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(le), Box::new(re)),
                    "<=" => ScalarExpr::Binary(BinaryOp::CmpLe, Box::new(le), Box::new(re)),
                    ">" => ScalarExpr::Binary(BinaryOp::CmpGt, Box::new(le), Box::new(re)),
                    ">=" => ScalarExpr::Binary(BinaryOp::CmpGe, Box::new(le), Box::new(re)),
                    other => return Err(LiftError::Unrecognized(format!("operator '{other}'"))),
                })
            }
            // `cond ? a : b` -> the IR's own ternary, ScalarExpr::Select —
            // already spec'd (KISS-CLASSIFY, the dropout-scaling use) and
            // already lowered by unpopped-cpu-c (F32/F32Strict/F64) and
            // unpopped-slang; no Backend/Lowering change needed here.
            "conditional_expression" => {
                let cond = self.field(n, "condition")?;
                let cons = self.field(n, "consequence")?;
                let alt = self.field(n, "alternative")?;
                Ok(ScalarExpr::Select(
                    Box::new(self.expr(cond)?),
                    Box::new(self.expr(cons)?),
                    Box::new(self.expr(alt)?),
                ))
            }
            "unary_expression" => {
                let op = self.field(n, "operator")?;
                let a = self.field(n, "argument")?;
                match self.text(op) {
                    "-" => Ok(ScalarExpr::Unary(UnaryOp::Neg, Box::new(self.expr(a)?))),
                    other => Err(LiftError::Unrecognized(format!("unary '{other}'"))),
                }
            }
            "subscript_expression" => {
                let base = self.text(self.field(n, "argument")?);
                let k: u8 = base
                    .strip_prefix(self.in_prefix)
                    .and_then(|d| d.parse().ok())
                    .ok_or_else(|| {
                        LiftError::Unrecognized(format!(
                            "read '{base}[..]' (not {}K)",
                            self.in_prefix
                        ))
                    })?;
                let idx = subscript_index(n, self.src)
                    .ok_or_else(|| LiftError::Unrecognized("index".into()))?;
                if idx != self.idx_var {
                    return Err(LiftError::Unrecognized(format!(
                        "non-elementwise index [{idx}] (expected [{}])",
                        self.idx_var
                    )));
                }
                self.max_input = Some(self.max_input.map_or(k, |m| m.max(k)));
                Ok(ScalarExpr::Input(k))
            }
            "number_literal" => {
                let t = self
                    .text(n)
                    .trim_end_matches(['f', 'F', 'l', 'L', 'u', 'U']);
                t.parse::<f64>()
                    .map(ScalarExpr::Const)
                    .map_err(|_| LiftError::Unrecognized(format!("literal {t}")))
            }
            "call_expression" => {
                let fname = self.text(self.field(n, "function")?);
                let args_node = self.field(n, "arguments")?;
                let mut cursor = args_node.walk();
                let args: Vec<Node> = args_node.named_children(&mut cursor).collect();
                match args.len() {
                    1 => {
                        let op = unary_fn(fname)
                            .ok_or_else(|| LiftError::Unrecognized(format!("call '{fname}(_)'")))?;
                        Ok(ScalarExpr::Unary(op, Box::new(self.expr(args[0])?)))
                    }
                    2 => {
                        let op = binary_fn(fname).ok_or_else(|| {
                            LiftError::Unrecognized(format!("call '{fname}(_,_)'"))
                        })?;
                        Ok(ScalarExpr::Binary(
                            op,
                            Box::new(self.expr(args[0])?),
                            Box::new(self.expr(args[1])?),
                        ))
                    }
                    n => Err(LiftError::Unrecognized(format!(
                        "call '{fname}' with {n} args"
                    ))),
                }
            }
            "identifier" => {
                let name = self.text(n).to_string();
                let bound = self
                    .locals
                    .iter()
                    .find(|(bn, _)| *bn == name)
                    .map(|(_, v)| *v);
                match bound {
                    Some(_) if self.resolving.contains(&name) => Err(LiftError::Unrecognized(
                        format!("identifier '{name}' (self-referential binding)"),
                    )),
                    Some(value) => {
                        self.resolving.push(name.clone());
                        let result = self.expr(value);
                        self.resolving.pop();
                        result
                    }
                    None => Err(LiftError::Unrecognized(format!("identifier '{name}'"))),
                }
            }
            other => Err(LiftError::Unrecognized(format!("node '{other}'"))),
        }
    }

    fn field<'t>(&self, n: Node<'t>, name: &str) -> Result<Node<'t>, LiftError> {
        n.child_by_field_name(name)
            .ok_or_else(|| LiftError::Unrecognized(format!("missing {name} in {}", n.kind())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const F32: &[ElementKind] = &[ElementKind::F32];

    /// A [`Frontend`] this module does not know about lifts through the same
    /// walkers — which is the whole claim.
    ///
    /// Every other test here uses [`SLANG`], so all of them would
    /// still pass if the walkers secretly hardcoded `"out"`/`"in"` and ignored
    /// the descriptor. This one supplies different values for **every** field
    /// and checks the lift still works, so the descriptor is proven to be read
    /// rather than decorative.
    ///
    /// It is also the shape a per-target crate will use once the grammar
    /// bindings move out: hand in a `Frontend`, get the neutral walk.
    #[test]
    fn a_caller_defined_frontend_lifts_through_the_shared_walkers() {
        const MINE: Frontend = Frontend {
            name: "toy",
            kernel_marker: "TOY_KERNEL",
            residue: &["forbidden_thing"],
            out_name: "dst",
            in_prefix: "src",
            // Reuses the Slang grammar because the CST shapes are C-family; the
            // point under test is the descriptor, not the grammar.
            parse: parse_slang,
        };
        let src = "TOY_KERNEL void k(float* dst, const float* src0, const float* src1) {\
                   int i = 0; dst[i] = src0[i] + src1[i]; }";
        let lifted = lift_elementwise(&MINE, src, "toy", F32).expect("caller's frontend lifts");
        assert_eq!(
            lifted.op.body,
            ScalarExpr::Add(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(1))
            ),
            "the walk must honour dst/src naming, not out/in"
        );

        // The marker gates entry: the same body without it is not a kernel.
        assert!(matches!(
            lift_elementwise(&MINE, &src.replace("TOY_KERNEL ", ""), "toy", F32),
            Err(LiftError::NotAKernel)
        ));
        // And the caller's residue list is the one enforced — `groupshared` is
        // Slang's residue, not this frontend's, so it must NOT be refused here.
        let with_slang_residue = src.replace("int i = 0;", "groupshared float s; int i = 0;");
        assert!(
            lift_elementwise(&MINE, &with_slang_residue, "toy", F32).is_ok(),
            "Slang's residue list must not leak into a frontend that did not declare it"
        );
        let with_own_residue = src.replace("int i = 0;", "forbidden_thing(); int i = 0;");
        assert!(matches!(
            lift_elementwise(&MINE, &with_own_residue, "toy", F32),
            Err(LiftError::Inexpressible(_))
        ));
    }

    const SLANG_MUL: &str = "StructuredBuffer<float> input0;\n\
        StructuredBuffer<float> input1;\n\
        RWStructuredBuffer<float> output;\n\
        [numthreads(256, 1, 1)]\n\
        void mul(uint3 tid : SV_DispatchThreadID) {\n\
            uint i = tid.x;\n\
            output[i] = input0[i] * input1[i];\n\
        }";

    fn reduce_op(l: &Lifted) -> Option<ReduceOp> {
        if let crate::ir::Access::Reduction { op, .. } = &l.op.access {
            Some(*op)
        } else {
            None
        }
    }

    fn scan_info(l: &Lifted) -> Option<(ReduceOp, &ScalarExpr, bool, bool)> {
        if let crate::ir::Access::Scan {
            op,
            pre,
            reverse,
            exclusive,
            ..
        } = &l.op.access
        {
            Some((*op, pre, *reverse, *exclusive))
        } else {
            None
        }
    }

    #[test]
    fn grammars_load_and_parse() {
        assert!(parse_slang("float4 f(float x) { return x; }").is_some());
    }

    #[test]
    fn slang_lifts_multiply() {
        let lifted = lift_elementwise(&SLANG, SLANG_MUL, "slang_mul", F32).unwrap();
        assert_eq!(lifted.n_inputs, 2);
        assert_eq!(
            lifted.op.body,
            ScalarExpr::Mul(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(1)),
            )
        );
    }

    /// Slang's `groupshared` is the same inexpressible-residue class as CUDA's
    /// `__shared__` (`baracuda-cuda-parse` tests that side), and must be reported as such
    /// — the two source languages spell it differently but the neutral answer
    /// is identical, which is what makes the refusal category portable.
    #[test]
    fn slang_refuses_groupshared() {
        let src = "groupshared float s[256];\n\
            [numthreads(256,1,1)]\n\
            void k(uint3 tid : SV_DispatchThreadID) { output[tid.x] = s[0]; }";
        assert!(
            matches!(
                lift_elementwise(&SLANG, src, "x", F32),
                Err(LiftError::Inexpressible(ref w)) if w == "groupshared"
            ),
            "got {:?}",
            lift_elementwise(&SLANG, src, "x", F32)
        );
    }

    #[test]
    fn slang_lifts_sum_reduction() {
        let src = "StructuredBuffer<float> input0;\n\
            RWStructuredBuffer<float> output;\n\
            [numthreads(256,1,1)]\n\
            void rsum(uint3 tid : SV_DispatchThreadID){ float acc = 0.0f; for (uint i=0;i<256;i++) acc += input0[i]; output[0] = acc; }";
        let lifted = lift_reduction(&SLANG, src, "rsum", F32).unwrap();
        assert_eq!(lifted.op.body, ScalarExpr::Input(0));
        assert_eq!(reduce_op(&lifted), Some(ReduceOp::Sum));
    }

    #[test]
    fn slang_lifts_cumsum() {
        let src = "StructuredBuffer<float> input0;\n\
            RWStructuredBuffer<float> output;\n\
            [numthreads(256,1,1)]\n\
            void csum(uint3 tid : SV_DispatchThreadID){ float acc = 0.0f; for (uint i=0;i<256;i++){ acc += input0[i]; output[i] = acc; } }";
        let lifted = lift_scan(&SLANG, src, "csum", F32).unwrap();
        assert_eq!(scan_info(&lifted).unwrap().0, ReduceOp::Sum);
    }

    const SLANG_COPY: &str = "StructuredBuffer<float> input0;\n\
        RWStructuredBuffer<float> output;\n\
        [numthreads(256, 1, 1)]\n\
        void k(uint3 tid : SV_DispatchThreadID) { uint i = tid.x; output[i] = input0[i]; }";
    const SLANG_SUM: &str = "StructuredBuffer<float> input0;\n\
        RWStructuredBuffer<float> output;\n\
        [numthreads(256,1,1)]\n\
        void rsum(uint3 tid : SV_DispatchThreadID){ float acc = 0.0f; for (uint i=0;i<256;i++) acc += input0[i]; output[0] = acc; }";
    const SLANG_CSUM: &str = "StructuredBuffer<float> input0;\n\
        RWStructuredBuffer<float> output;\n\
        [numthreads(256,1,1)]\n\
        void csum(uint3 tid : SV_DispatchThreadID){ float acc = 0.0f; for (uint i=0;i<256;i++){ acc += input0[i]; output[i] = acc; } }";

    // The walker-behaviour tests below were CUDA-source tests until the CUDA front
    // end moved to `baracuda-cuda-parse`. The walkers are shared (both grammars give
    // the same C-family node shapes), so they are re-expressed on Slang here and keep
    // their coverage in the neutral core.

    #[test]
    fn elementwise_and_reduction_lifters_dont_cross() {
        // The elementwise lifter refuses a reduction (scalar store), and vice versa.
        assert!(lift_elementwise(&SLANG, SLANG_SUM, "x", F32).is_err());
        assert!(lift_reduction(&SLANG, SLANG_COPY, "x", F32).is_err());
        // The unified entry point handles either class.
        assert!(lift(&SLANG, SLANG_SUM, "x", F32).is_ok());
        assert!(lift(&SLANG, SLANG_COPY, "x", F32).is_ok());
    }

    #[test]
    fn scan_reduction_elementwise_are_disjoint() {
        // out[i]=acc (running) -> scan; out[0]=acc -> reduction; out[i]=expr -> elementwise.
        assert!(lift_scan(&SLANG, SLANG_CSUM, "x", F32).is_ok());
        assert!(lift_reduction(&SLANG, SLANG_CSUM, "x", F32).is_err());
        assert!(lift_elementwise(&SLANG, SLANG_CSUM, "x", F32).is_err());
        assert!(matches!(
            lift(&SLANG, SLANG_CSUM, "x", F32).unwrap().op.access,
            crate::ir::Access::Scan { .. }
        ));
    }

    #[test]
    fn a_single_assignment_local_is_resolved_before_the_store() {
        // The shape a real Fuel kernel uses (cast_f32_to_f16.slang): declare,
        // assign once, read once at the store. Before local-variable
        // resolution this refused as `Unrecognized("identifier 'x'")`.
        let src = "StructuredBuffer<float> input0;\n\
            RWStructuredBuffer<float> output;\n\
            [numthreads(256,1,1)]\n\
            void cast(uint3 tid : SV_DispatchThreadID){\n\
                uint i = tid.x;\n\
                float x = input0[i];\n\
                output[i] = x + 1.0f;\n\
            }";
        let lifted = lift_elementwise(&SLANG, src, "cast", F32)
            .expect("a single-assignment local must resolve, not refuse as an unknown identifier");
        assert_eq!(
            lifted.op.body,
            ScalarExpr::Add(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Const(1.0)),
            )
        );
    }

    #[test]
    fn a_reassigned_local_is_not_resolved() {
        // `x` is assigned twice (declaration, then a plain `=` reassignment) with no
        // branch involved at all — the simplest case a single-assignment resolver
        // must still decline, because picking either value would be a guess.
        // Distinct from `scan_reduction_elementwise_are_disjoint`, which covers the
        // COMPOUND-assignment (`+=`) shape of the same hazard.
        let src = "StructuredBuffer<float> input0;\n\
            RWStructuredBuffer<float> output;\n\
            [numthreads(256,1,1)]\n\
            void two(uint3 tid : SV_DispatchThreadID){\n\
                uint i = tid.x;\n\
                float x = input0[i];\n\
                x = x + 1.0f;\n\
                output[i] = x;\n\
            }";
        assert!(
            lift_elementwise(&SLANG, src, "two", F32).is_err(),
            "a name assigned twice must not be silently resolved to either value"
        );
    }

    #[test]
    fn a_ternary_lifts_to_select() {
        let src = "StructuredBuffer<float> input0;\n\
            RWStructuredBuffer<float> output;\n\
            [numthreads(256,1,1)]\n\
            void relu(uint3 tid : SV_DispatchThreadID){\n\
                uint i = tid.x;\n\
                output[i] = input0[i] > 0.0f ? input0[i] : 0.0f;\n\
            }";
        let lifted = lift_elementwise(&SLANG, src, "relu", F32)
            .expect("a ternary must lift to Select, not refuse as an unknown node");
        assert_eq!(
            lifted.op.body,
            ScalarExpr::Select(
                Box::new(ScalarExpr::Binary(
                    crate::ir::BinaryOp::CmpGt,
                    Box::new(ScalarExpr::Input(0)),
                    Box::new(ScalarExpr::Const(0.0)),
                )),
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Const(0.0)),
            )
        );
    }
}
