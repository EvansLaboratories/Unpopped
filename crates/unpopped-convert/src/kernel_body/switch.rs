// SPDX-License-Identifier: MIT OR Apache-2.0
//! `switch` as an `if` chain (ROADMAP U19 census slice: the nf4 codebook, 11 real instantiations). A `switch (e)` whose cases each
//! end in `break`, `return` or `continue` (no fall-through into a later group of statements) and whose `default` comes last is
//! rewritten, in the source text before the walk, to `{ auto __sw = (e); if (__sw == a || __sw == b) { ... } else if ... else { ... } }`:
//! the selector is evaluated once, stacked labels (`case 1: case 2:`) share a body, the trailing `break` goes, and a `return` stays
//! where the inlined-function machinery (`returns.rs`) reads it. A `break` that leaves the switch from anywhere but the end of its
//! case, a fall-through and a `default` that is not last leave the `switch` in place, so it declines by name as before.

use super::{KernelLiftError, text};
use tree_sitter::{Node, Tree};

/// How many rounds of rewriting (an inner `switch` goes first, then the one around it).
const MAX_ROUNDS: usize = 8;

/// One group of labels with the statements they share.
struct Group<'t> {
    /// The `case` values; empty for `default`.
    labels: Vec<Node<'t>>,
    is_default: bool,
    stmts: Vec<Node<'t>>,
}

/// `src` with every rewritable `switch` replaced by its `if` chain.
pub(super) fn expand_switches(
    parse: fn(&str) -> Option<Tree>,
    mut src: String,
) -> Result<String, KernelLiftError> {
    for _ in 0..MAX_ROUNDS {
        if !src.contains("switch") {
            return Ok(src);
        }
        let tree = parse(&src).ok_or(KernelLiftError::SyntaxError { line: 0 })?;
        let mut edits: Vec<(usize, usize, String)> = Vec::new();
        let mut stack = vec![tree.root_node()];
        while let Some(n) = stack.pop() {
            if n.kind() == "switch_statement" && !contains_switch(n) {
                if let Some(text) = rewrite(n, &src) {
                    edits.push((n.start_byte(), n.end_byte(), text));
                }
                continue;
            }
            let mut c = n.walk();
            stack.extend(n.children(&mut c));
        }
        if edits.is_empty() {
            return Ok(src);
        }
        edits.sort_by_key(|e| e.0);
        for (start, end, new) in edits.into_iter().rev() {
            src.replace_range(start..end, &new);
        }
    }
    Ok(src)
}

/// Whether a `switch` lies strictly inside `n`.
fn contains_switch(n: Node<'_>) -> bool {
    let mut c = n.walk();
    n.named_children(&mut c)
        .any(|k| k.kind() == "switch_statement" || contains_switch(k))
}

/// A `break` that leaves the enclosing `switch`: any not inside a nested loop.
fn leaves_switch(n: Node<'_>) -> bool {
    match n.kind() {
        "break_statement" => true,
        "for_statement" | "while_statement" | "do_statement" | "switch_statement" => false,
        _ => {
            let mut c = n.walk();
            n.named_children(&mut c).any(leaves_switch)
        }
    }
}

fn rewrite(sw: Node<'_>, src: &str) -> Option<String> {
    let cond = sw
        .child_by_field_name("condition")?
        .child_by_field_name("value")?;
    let body = sw.child_by_field_name("body")?;
    let groups = collect_groups(body)?;
    check_groups(&groups)?;
    emit_chain(sw, cond, &groups, src)
}

/// The statements a group runs: its tail `break` is the switch's own exit and is dropped.
fn kept_stmts<'a, 'b>(g: &'b Group<'a>) -> Option<&'b [Node<'a>]> {
    let tail = g.stmts.last()?;
    Some(if tail.kind() == "break_statement" {
        &g.stmts[..g.stmts.len() - 1]
    } else {
        &g.stmts[..]
    })
}

/// The `case` groups of a switch body, fallthrough labels merged; `None` for a shape the rewrite does not handle.
fn collect_groups(body: Node<'_>) -> Option<Vec<Group<'_>>> {
    let mut c = body.walk();
    let cases: Vec<_> = body.named_children(&mut c).collect();
    let mut groups: Vec<Group<'_>> = Vec::new();
    let mut pending: Vec<Node<'_>> = Vec::new();
    let mut pending_default = false;
    for case in cases {
        if case.kind() == "comment" {
            continue;
        }
        if case.kind() != "case_statement" {
            return None;
        }
        let value = case.child_by_field_name("value");
        match value {
            Some(v) => pending.push(v),
            None => pending_default = true,
        }
        let mut k = case.walk();
        let mut stmts: Vec<Node<'_>> = case
            .named_children(&mut k)
            .filter(|s| Some(*s) != value && s.kind() != "comment")
            .collect();
        // `case K: { ...; break; }` is the block's own statements
        if let [only] = stmts.as_slice()
            && only.kind() == "compound_statement"
        {
            let mut b = only.walk();
            stmts = only
                .named_children(&mut b)
                .filter(|s| s.kind() != "comment")
                .collect();
        }
        if stmts.is_empty() {
            continue;
        }
        if pending_default && !pending.is_empty() {
            // `case 1: default:` shares one body between a value and the default: leave it
            return None;
        }
        groups.push(Group {
            labels: std::mem::take(&mut pending),
            is_default: std::mem::take(&mut pending_default),
            stmts,
        });
    }
    if !pending.is_empty() || pending_default || groups.is_empty() {
        return None;
    }
    Some(groups)
}

/// Every `default` group is last and every other group ends the switch (or is last).
fn check_groups(groups: &[Group<'_>]) -> Option<()> {
    // the `default` group is the last, and every other group ends the switch
    let last = groups.len() - 1;
    for (i, g) in groups.iter().enumerate() {
        if g.is_default && i != last {
            return None;
        }
        let tail = g.stmts.last()?;
        let ends = matches!(
            tail.kind(),
            "break_statement" | "return_statement" | "continue_statement"
        );
        if !ends && i != last {
            return None;
        }
        let kept = kept_stmts(g)?;
        if kept.iter().any(|s| leaves_switch(*s)) {
            return None;
        }
    }
    Some(())
}

/// The `if`/`else if` chain over a selector temporary that replaces the switch.
fn emit_chain(sw: Node<'_>, cond: Node<'_>, groups: &[Group<'_>], src: &str) -> Option<String> {
    let sel = format!("__sw{}", sw.start_byte());
    let mut out = format!("{{ auto {sel} = ({}); ", text(cond, src));
    let mut first = true;
    for g in groups {
        let kept = kept_stmts(g)?;
        let body: Vec<&str> = kept.iter().map(|s| text(*s, src)).collect();
        if g.is_default {
            if first {
                out.push_str(&format!("{{ {} }} ", body.join("\n")));
            } else {
                out.push_str(&format!("else {{ {} }} ", body.join("\n")));
            }
        } else {
            let cond = g
                .labels
                .iter()
                .map(|l| format!("{sel} == ({})", text(*l, src)))
                .collect::<Vec<_>>()
                .join(" || ");
            out.push_str(&format!(
                "{}if ({cond}) {{ {} }} ",
                if first { "" } else { "else " },
                body.join("\n")
            ));
        }
        first = false;
    }
    out.push('}');
    Some(out)
}
