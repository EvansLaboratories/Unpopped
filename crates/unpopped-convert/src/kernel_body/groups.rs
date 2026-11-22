// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cooperative groups, the block-wide subset: `cg::thread_block b = cg::this_thread_block();` (or `auto b = ...`), then `b.sync()`,
//! `b.thread_rank()`, `b.size()`. A group handle carries no state of its own: it is a name for the launch's block, so the
//! declaration emits nothing and each method is the construct it stands for (`__syncthreads()`, the linear thread index, the
//! block's thread count). Tiles, `cg::reduce`, `this_grid` and the rest decline by name.

use super::*;

/// The spelling of the block type, with or without its namespace alias.
fn is_block_type(t: &str) -> bool {
    matches!(
        t,
        "cg::thread_block" | "cooperative_groups::thread_block" | "thread_block"
    )
}

/// Whether `v` is a call of `this_thread_block` (any namespace spelling).
fn is_this_block(v: &str) -> bool {
    let v = v.replace(' ', "");
    matches!(
        v.as_str(),
        "cg::this_thread_block()"
            | "cooperative_groups::this_thread_block()"
            | "this_thread_block()"
    )
}

impl<'s, 't> Walker<'s, 't> {
    /// A declaration of a block group (`cg::thread_block b = cg::this_thread_block();`): remembered by name, no IR. `None` when the
    /// declaration is not one.
    pub(super) fn group_declaration(&mut self, n: Node<'_>) -> Option<R<()>> {
        let ty = n
            .child_by_field_name("type")
            .map(|t| text(t, self.src).trim().to_string())?;
        let mut c = n.walk();
        let decls: Vec<_> = n.children_by_field_name("declarator", &mut c).collect();
        let [d] = decls.as_slice() else {
            return None;
        };
        if d.kind() != "init_declarator" {
            return None;
        }
        let (id, value) = (
            d.child_by_field_name("declarator")?,
            d.child_by_field_name("value")?,
        );
        let this_block = is_this_block(text(value, self.src));
        if id.kind() != "identifier" || !(this_block && (ty == "auto" || is_block_type(&ty))) {
            return None;
        }
        if self.launch.is_none() {
            return Some(self.not_yet(
                n,
                "a cooperative group without a launch (use `lift_kernel_launched`)",
            ));
        }
        self.groups.insert(text(id, self.src).to_string());
        Some(Ok(()))
    }

    /// Whether `obj` names a block group in scope.
    fn is_group(&self, obj: Node<'_>) -> bool {
        obj.kind() == "identifier" && self.groups.contains(text(obj, self.src))
    }

    /// `b.sync()` as a statement: the block barrier. `None` when the callee is not a group's method.
    pub(super) fn group_statement(
        &mut self,
        call: Node<'_>,
        callee: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> Option<R<()>> {
        let obj = callee.child_by_field_name("argument")?;
        if !self.is_group(obj) {
            return None;
        }
        let m = text(callee.child_by_field_name("field")?, self.src);
        if m == "sync" && self.call_args(call).is_empty() {
            self.flush(out);
            out.push(Stmt::Barrier {
                scope: Scope::Block,
                kind: BarrierKind::Full,
            });
            return Some(Ok(()));
        }
        Some(self.not_yet(call, format!("the cooperative-group method `{m}`")))
    }

    /// `b.thread_rank()` / `b.size()` as a value (`U32`). `None` when the callee is not a group's method.
    pub(super) fn group_value(
        &self,
        call: Node<'_>,
        callee: Node<'_>,
    ) -> Option<R<(ScalarExpr, E)>> {
        let obj = callee.child_by_field_name("argument")?;
        if !self.is_group(obj) {
            return None;
        }
        let m = text(callee.child_by_field_name("field")?, self.src);
        let id = |i: fn(u8) -> ExecId, a: u8| ScalarExpr::Exec(i(a));
        let mul = |a: ScalarExpr, b: ScalarExpr| ScalarExpr::Mul(Box::new(a), Box::new(b));
        let add = |a: ScalarExpr, b: ScalarExpr| ScalarExpr::Add(Box::new(a), Box::new(b));
        if !self.call_args(call).is_empty() {
            return Some(self.not_yet(
                call,
                format!("the cooperative-group method `{m}` with arguments"),
            ));
        }
        let v = match m {
            // x + y * dim.x + z * dim.x * dim.y
            "thread_rank" => add(
                id(ExecId::ThreadIdx, 0),
                add(
                    mul(id(ExecId::ThreadIdx, 1), id(ExecId::BlockDim, 0)),
                    mul(
                        id(ExecId::ThreadIdx, 2),
                        mul(id(ExecId::BlockDim, 0), id(ExecId::BlockDim, 1)),
                    ),
                ),
            ),
            "size" | "num_threads" => mul(
                id(ExecId::BlockDim, 0),
                mul(id(ExecId::BlockDim, 1), id(ExecId::BlockDim, 2)),
            ),
            _ => return Some(self.not_yet(call, format!("the cooperative-group method `{m}`"))),
        };
        Some(Ok((v, E::U32)))
    }
}
