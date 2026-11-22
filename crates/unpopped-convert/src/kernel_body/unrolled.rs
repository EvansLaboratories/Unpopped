// SPDX-License-Identifier: MIT OR Apache-2.0
//! A loop whose trip count is a compile-time constant but is not the counted form `for (T i = a; i < b; i += s)` (ROADMAP U19
//! slice 5: the decode kernel's butterfly `for (uint32_t offset = BDX / 2; offset > 0; offset /= 2)`). The walker **runs the
//! loop over its induction variable** (start, condition and update all constant) and emits the body once per value, the
//! variable a constant local in each copy. At most `MAX_TRIPS` trips: a longer loop declines rather than bloating the IR.

use super::{Fixed, R, Sym, Walker, text};
use std::collections::HashMap;
use tree_sitter::Node;
use unpopped::kernel::Stmt;

use super::globals::fold;
use unpopped::ir::ScalarExpr;

/// The most trips the walker unrolls.
const MAX_TRIPS: usize = 64;

/// How the induction variable moves each trip.
#[derive(Clone, Copy)]
enum Update {
    Add(i128),
    Mul(i128),
    Div(i128),
    Shl(i128),
    Shr(i128),
}

impl Update {
    fn apply(self, v: i128) -> Option<i128> {
        match self {
            Self::Add(k) => v.checked_add(k),
            Self::Mul(k) => v.checked_mul(k),
            Self::Div(k) if k != 0 => Some(v / k),
            Self::Shl(k) if (0..64).contains(&k) => Some(v << k),
            Self::Shr(k) if (0..64).contains(&k) => Some(v >> k),
            _ => None,
        }
    }
}

impl<'s, 't> Walker<'s, 't> {
    /// A constant expression's value (its loads, if any, are dropped: a constant has none).
    pub(super) fn const_value(&mut self, n: Node<'_>) -> R<Option<i128>> {
        let mark = self.hoisted.len();
        let r = self.expr(n);
        self.hoisted.truncate(mark);
        Ok(fold(&r?.0))
    }

    fn loop_update(&mut self, upd: Node<'_>, name: &str) -> R<Option<Update>> {
        let is_var = |n: Option<Node<'_>>| n.is_some_and(|a| text(a, self.src) == name);
        match upd.kind() {
            "update_expression" => {
                let op = upd
                    .child_by_field_name("operator")
                    .map(|o| text(o, self.src));
                if !is_var(upd.child_by_field_name("argument")) {
                    return Ok(None);
                }
                Ok(match op {
                    Some("++") => Some(Update::Add(1)),
                    Some("--") => Some(Update::Add(-1)),
                    _ => None,
                })
            }
            "assignment_expression" => {
                let (Some(l), Some(op), Some(r)) = (
                    upd.child_by_field_name("left"),
                    upd.child_by_field_name("operator"),
                    upd.child_by_field_name("right"),
                ) else {
                    return Ok(None);
                };
                if !is_var(Some(l)) {
                    return Ok(None);
                }
                let Some(k) = self.const_value(r)? else {
                    return Ok(None);
                };
                Ok(match text(op, self.src) {
                    "+=" => Some(Update::Add(k)),
                    "-=" => Some(Update::Add(-k)),
                    "*=" => Some(Update::Mul(k)),
                    "/=" => Some(Update::Div(k)),
                    "<<=" => Some(Update::Shl(k)),
                    ">>=" => Some(Update::Shr(k)),
                    _ => None,
                })
            }
            _ => Ok(None),
        }
    }

    /// `for (T v = C; v OP K; update) body` with every piece constant: the body once per value of `v`. `false` if the loop is
    /// not of that shape (nothing is emitted then).
    pub(super) fn constant_for(&mut self, n: Node<'_>, out: &mut Vec<Stmt>) -> R<bool> {
        let (Some(init), Some(cond), Some(upd), Some(body)) = (
            n.child_by_field_name("initializer"),
            n.child_by_field_name("condition"),
            n.child_by_field_name("update"),
            n.child_by_field_name("body"),
        ) else {
            return Ok(false);
        };
        let Some((name, ty, a)) = self.for_init(init)? else {
            return Ok(false);
        };
        let Some(start) = fold(&a) else {
            return Ok(false);
        };
        let (Some(l), Some(op), Some(r)) = (
            cond.child_by_field_name("left"),
            cond.child_by_field_name("operator"),
            cond.child_by_field_name("right"),
        ) else {
            return Ok(false);
        };
        if cond.kind() != "binary_expression" || text(l, self.src) != name {
            return Ok(false);
        }
        let (Some(bound), Some(update)) = (self.const_value(r)?, self.loop_update(upd, &name)?)
        else {
            return Ok(false);
        };
        // the counter has its type's range: an unsigned counter that steps below zero wraps (C), so `for (unsigned i = 3; i >= 0;
        // --i)` never ends and is refused by the trip limit below, not run as four trips; a bound outside the type's range is
        // compared after C's conversion, which is not modelled
        let range = super::int_range(ty);
        if range.is_some_and(|(lo, hi)| bound < lo || bound > hi) {
            return self.not_yet(n, "a loop bound outside the range of the counter's type");
        }
        let wrap = |x: i128| match range {
            Some((lo, hi)) => (x - lo).rem_euclid(hi - lo + 1) + lo,
            None => x,
        };
        let holds = |v: i128| match text(op, self.src) {
            "<" => Some(v < bound),
            "<=" => Some(v <= bound),
            ">" => Some(v > bound),
            ">=" => Some(v >= bound),
            "!=" => Some(v != bound),
            _ => None,
        };
        let mut values = Vec::new();
        let mut v = start;
        while holds(v) == Some(true) {
            values.push(v);
            let next = update.apply(v).map(wrap);
            match next {
                Some(x) if values.len() < MAX_TRIPS && x != v => v = x,
                _ => {
                    return self.not_yet(
                        n,
                        format!("a constant loop of more than {MAX_TRIPS} trips (or one that does not advance)"),
                    );
                }
            }
        }
        if holds(v).is_none() {
            return Ok(false);
        }
        for value in values {
            let Ok(c) = i64::try_from(value) else {
                return self.not_yet(n, "a loop variable outside the 64-bit range");
            };
            self.flush(out);
            let local = self.new_local(ty);
            out.push(Stmt::Let {
                local,
                init: ScalarExpr::ConstInt(c, ty),
            });
            self.scopes.push(HashMap::new());
            self.scopes
                .last_mut()
                .expect("a scope is open")
                .insert(name.clone(), Sym::Scalar(local, ty, Fixed::Counter));
            let b = self.block_of(body);
            self.scopes.pop();
            out.extend(b?.0);
        }
        Ok(true)
    }
}
