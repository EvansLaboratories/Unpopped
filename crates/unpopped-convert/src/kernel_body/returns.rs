// SPDX-License-Identifier: MIT OR Apache-2.0
//! `return` inside an inlined function (ROADMAP U19 slice 4e). A callee's `return`s must all be in **tail position**: the last
//! statement of the body, or the last statement of the block of a top-level `if (c) { ...; return; }` with no `else`. Such an
//! `if` becomes an `If` whose `else` branch is **the rest of the function**, so control never leaves by any path but the
//! structured one; the value of a `return e;` is assigned to one result local, declared (zero) before the body. A final
//! `if (c) { ... } else { ... }` that returns lifts as an `If`, each arm held to the same rules (so `else if` chains and nested
//! tail forms lift). A `return` inside a loop or a nested block, and one followed by more statements, decline by name.

use super::{R, Walker};
use std::collections::HashMap;
use tree_sitter::Node;
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, Stmt};
use unpopped_vocab::ElementKind as E;

/// The statements of a block (or the one statement standing in for it).
fn stmts_of(n: Node<'_>) -> Vec<Node<'_>> {
    if n.kind() == "compound_statement" {
        let mut c = n.walk();
        n.named_children(&mut c).collect()
    } else {
        vec![n]
    }
}

/// `if (c) { ...; return ...; }` with no `else`.
fn is_tail_return_if(n: Node<'_>) -> bool {
    if n.kind() != "if_statement" || n.child_by_field_name("alternative").is_some() {
        return false;
    }
    let Some(cons) = n.child_by_field_name("consequence") else {
        return false;
    };
    stmts_of(cons)
        .last()
        .is_some_and(|l| l.kind() == "return_statement")
}

/// An `if` with an `else` that returns somewhere, standing as the last statement of its block.
fn is_final_branch(n: Node<'_>) -> bool {
    n.kind() == "if_statement"
        && n.child_by_field_name("alternative").is_some()
        && contains_return(n)
}

/// The statements of an `else` clause (the clause wraps one statement or block).
fn else_stmts(n: Node<'_>) -> Vec<Node<'_>> {
    n.child_by_field_name("alternative")
        .and_then(|a| a.named_child(0))
        .map(stmts_of)
        .unwrap_or_default()
}

fn contains_return(n: Node<'_>) -> bool {
    if n.kind() == "return_statement" {
        return true;
    }
    let mut c = n.walk();
    n.named_children(&mut c).any(contains_return)
}

impl<'s, 't> Walker<'s, 't> {
    /// Every `return` of the callee is in tail position and carries a value iff the function is not `void`.
    pub(super) fn check_returns(&self, body: Node<'_>, has_value: bool) -> R<()> {
        self.check_seq(&stmts_of(body), has_value)
    }

    fn check_seq(&self, seq: &[Node<'_>], has_value: bool) -> R<()> {
        for (i, n) in seq.iter().enumerate() {
            match n.kind() {
                "return_statement" => {
                    if i + 1 != seq.len() {
                        return self.not_yet(
                            *n,
                            "a `return` followed by more statements in an inlined function",
                        );
                    }
                    if has_value != (n.named_child_count() > 0) {
                        return self.not_yet(
                            *n,
                            "a `return` whose value does not match the function's type",
                        );
                    }
                }
                _ if is_tail_return_if(*n) => {
                    let cons = n
                        .child_by_field_name("consequence")
                        .expect("checked by is_tail_return_if");
                    self.check_seq(&stmts_of(cons), has_value)?;
                    return self.check_seq(&seq[i + 1..], has_value);
                }
                _ if i + 1 == seq.len() && is_final_branch(*n) => {
                    let cons = n
                        .child_by_field_name("consequence")
                        .expect("an `if` has a consequence");
                    self.check_seq(&stmts_of(cons), has_value)?;
                    return self.check_seq(&else_stmts(*n), has_value);
                }
                _ if contains_return(*n) => {
                    return self.not_yet(
                        *n,
                        "a `return` inside a loop, a nested block or an `if`/`else` of an inlined function (only a final `return` or a tail `if (c) { ...; return; }` lifts)",
                    );
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// The callee's statements into `out`; its `return` value, if any, as the result local.
    pub(super) fn callee_body(
        &mut self,
        body: Node<'_>,
        ret_ty: Option<E>,
        out: &mut Vec<Stmt>,
    ) -> R<Option<(ScalarExpr, E)>> {
        let res = match ret_ty {
            Some(t) => {
                let l = self.new_local(t);
                out.push(Stmt::Let {
                    local: l,
                    init: self.zero_of(t, body)?,
                });
                Some((l, t))
            }
            None => None,
        };
        self.scopes.push(HashMap::new());
        let r = self.callee_seq(&stmts_of(body), res, out);
        self.scopes.pop();
        r.map(|()| res.map(|(l, t)| (ScalarExpr::Local(l), t)))
    }

    fn callee_seq(
        &mut self,
        seq: &[Node<'_>],
        res: Option<(u32, E)>,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        for (i, n) in seq.iter().enumerate() {
            if n.kind() == "return_statement" {
                return self.callee_return(*n, res, out);
            }
            if is_tail_return_if(*n) {
                return self.callee_tail_if(*n, &seq[i + 1..], res, out);
            }
            if i + 1 == seq.len() && is_final_branch(*n) {
                return self.callee_if_else(*n, res, out);
            }
            self.stmt(*n, out)?;
        }
        Ok(())
    }

    /// `if (c) { A; return; } rest` as `if (c) { A } else { rest }`.
    fn callee_tail_if(
        &mut self,
        n: Node<'_>,
        rest: &[Node<'_>],
        res: Option<(u32, E)>,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        let (Some(c), Some(cons)) = (
            n.child_by_field_name("condition"),
            n.child_by_field_name("consequence"),
        ) else {
            return self.not_yet(n, "an `if` of an unexpected shape");
        };
        let cond = self.condition(c)?;
        self.flush(out);
        let mut then = self.callee_block(&stmts_of(cons), res)?;
        if self.callee_exits_kernel && res.is_none() {
            // `if (c) { A; return; } rest` as `if (c) { A; return; }` then `rest`: the return ends the kernel, so `rest` (a warp
            // shuffle may follow) is not inside a branch
            then.push(Stmt::Return);
            out.push(Stmt::If {
                cond,
                then: Block(then),
                els: Block(Vec::new()),
            });
            return self.callee_seq(rest, res, out);
        }
        let els = self.callee_block(rest, res)?;
        out.push(Stmt::If {
            cond,
            then: Block(then),
            els: Block(els),
        });
        Ok(())
    }

    /// A final `if (c) { A } else { B }` that returns: an `If` whose arms are the callee's tail forms.
    fn callee_if_else(&mut self, n: Node<'_>, res: Option<(u32, E)>, out: &mut Vec<Stmt>) -> R<()> {
        let (Some(c), Some(cons)) = (
            n.child_by_field_name("condition"),
            n.child_by_field_name("consequence"),
        ) else {
            return self.not_yet(n, "an `if` of an unexpected shape");
        };
        let cond = self.condition(c)?;
        self.flush(out);
        let then = self.callee_block(&stmts_of(cons), res)?;
        let els = self.callee_block(&else_stmts(n), res)?;
        out.push(Stmt::If {
            cond,
            then: Block(then),
            els: Block(els),
        });
        Ok(())
    }

    fn callee_block(&mut self, seq: &[Node<'_>], res: Option<(u32, E)>) -> R<Vec<Stmt>> {
        self.scopes.push(HashMap::new());
        let mut v = Vec::new();
        let r = self.callee_seq(seq, res, &mut v);
        self.scopes.pop();
        r.map(|()| v)
    }

    /// `return e;` as an assignment to the result local (a bare `return;` has no effect: control reaches the end).
    fn callee_return(&mut self, r: Node<'_>, res: Option<(u32, E)>, out: &mut Vec<Stmt>) -> R<()> {
        let (Some((local, ty)), Some(v)) = (res, r.named_child(0)) else {
            return Ok(());
        };
        let (e, et) = self.expr(v)?;
        let value = self.convert(e, et, ty, v)?;
        out.append(&mut self.hoisted);
        out.push(Stmt::Assign { local, value });
        Ok(())
    }
}
