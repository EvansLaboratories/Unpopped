// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cooperative groups, the block-wide subset: `cg::thread_block b = cg::this_thread_block();` (or `auto b = ...`), then `b.sync()`,
//! `b.thread_rank()`, `b.size()`. A group handle carries no state of its own: it is a name for the launch's block, so the
//! declaration emits nothing and each method is the construct it stands for (`__syncthreads()`, the linear thread index, the
//! block's thread count). Tiles, `cg::reduce`, `this_grid` and the rest decline by name.

use super::*;
use unpopped::ir::ReduceOp;
use unpopped::kernel::{CollectiveKind, ShuffleMode};

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

/// `(width, name, group argument)` of `cg::thread_block_tile<W> name = cg::tiled_partition<W>(arg)`, or `auto name = ...`, read
/// from a statement's text: tree-sitter-cpp parses the `thread_block_tile<32> name = ...` spelling as a comparison chain, so the
/// declaration is recognised by its text and its tokens, never by its tree.
fn parse_tile_declaration(t: &str) -> Option<(u32, String, String)> {
    let t = t.trim().trim_end_matches(';').trim();
    let (lhs, rhs) = t.split_once('=')?;
    let lhs = lhs.trim();
    let name = lhs
        .rsplit(|c: char| !(c.is_alphanumeric() || c == '_'))
        .next()?;
    let ty = lhs[..lhs.len() - name.len()].trim();
    let ty_width = if ty == "auto" {
        None
    } else {
        let inner = ty
            .strip_prefix("cg::thread_block_tile<")
            .or_else(|| ty.strip_prefix("cooperative_groups::thread_block_tile<"))?
            .strip_suffix('>')?;
        Some(inner.trim().parse::<u32>().ok()?)
    };
    let rhs = rhs.trim();
    let rest = rhs
        .strip_prefix("cg::tiled_partition<")
        .or_else(|| rhs.strip_prefix("cooperative_groups::tiled_partition<"))?;
    let (w, call) = rest.split_once('>')?;
    let width = w.trim().parse::<u32>().ok()?;
    if ty_width.is_some_and(|tw| tw != width) {
        return None;
    }
    let arg = call.trim().strip_prefix('(')?.strip_suffix(')')?.trim();
    Some((width, name.to_string(), arg.to_string()))
}

impl<'s, 't> Walker<'s, 't> {
    /// A tile declaration met as a statement (the comparison-chain parse) or as a declaration: remembered by name, no IR. The tile
    /// is a name for the launch's subgroup, so the launch must name a subgroup of exactly that width.
    pub(super) fn tile_declaration(&mut self, n: Node<'_>) -> Option<R<()>> {
        let (width, name, arg) = parse_tile_declaration(text(n, self.src))?;
        if !(self.groups.contains(&arg) || is_this_block(&arg)) {
            return Some(self.not_yet(
                n,
                format!("a tile partitioned from `{arg}`, which is not a thread block"),
            ));
        }
        match self.launch.and_then(|l| l.subgroup) {
            Some(s) if s.min == width && s.max == width => {}
            _ => {
                return Some(self.not_yet(
                    n,
                    format!(
                        "a `thread_block_tile<{width}>` needs a launch whose subgroup is exactly {width} (`LaunchShape::with_subgroup({width}, {width})`)"
                    ),
                ));
            }
        }
        self.tiles.insert(name, width);
        Some(Ok(()))
    }

    fn tile_of(&self, obj: Node<'_>) -> Option<u32> {
        (obj.kind() == "identifier")
            .then(|| self.tiles.get(text(obj, self.src)).copied())
            .flatten()
    }

    /// `w.sync()` as a statement: the subgroup barrier. `None` when the callee is not a tile's method.
    pub(super) fn tile_statement(
        &mut self,
        call: Node<'_>,
        callee: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> Option<R<()>> {
        let obj = callee.child_by_field_name("argument")?;
        self.tile_of(obj)?;
        let m = text(callee.child_by_field_name("field")?, self.src);
        if m == "sync" && self.call_args(call).is_empty() {
            self.flush(out);
            out.push(Stmt::Barrier {
                scope: Scope::Subgroup,
                kind: BarrierKind::Full,
            });
            return Some(Ok(()));
        }
        // a value method used as a statement has no effect the walker models
        Some(self.not_yet(call, format!("the tile method `{m}` as a statement")))
    }

    /// `w.thread_rank()`, `w.size()`, `w.shfl(v, l)`, `w.shfl_down(v, d)`, `w.shfl_up(v, d)`, `w.shfl_xor(v, m)` as values.
    pub(super) fn tile_value(
        &mut self,
        call: Node<'_>,
        callee: Node<'_>,
    ) -> Option<R<(ScalarExpr, E)>> {
        let obj = callee.child_by_field_name("argument")?;
        let width = self.tile_of(obj)?;
        let m = text(callee.child_by_field_name("field")?, self.src).to_string();
        let args = self.call_args(call);
        Some((|| match (m.as_str(), args.as_slice()) {
            ("thread_rank", []) => Ok((ScalarExpr::Exec(ExecId::LaneId), E::U32)),
            ("size" | "num_threads", []) => {
                Ok((ScalarExpr::ConstInt(i64::from(width), E::U32), E::U32))
            }
            ("shfl" | "shfl_down" | "shfl_up" | "shfl_xor", [v, operand]) => {
                let mode = match m.as_str() {
                    "shfl" => ShuffleMode::Idx,
                    "shfl_down" => ShuffleMode::Down,
                    "shfl_up" => ShuffleMode::Up,
                    _ => ShuffleMode::Xor,
                };
                let (value, vt) = self.expr(*v)?;
                let (op, ot) = self.expr(*operand)?;
                let operand = self.convert(op, ot, E::U32, *operand)?;
                let result = self.new_local(vt);
                self.hoisted.push(Stmt::Collective {
                    kind: CollectiveKind::Shuffle { mode, operand },
                    value,
                    result,
                    scope: Scope::Subgroup,
                });
                Ok((ScalarExpr::Local(result), vt))
            }
            _ => self.not_yet(call, format!("the tile method `{m}`")),
        })())
    }

    /// `cg::reduce(tile, v, cg::plus<T>())` (also `cg::greater`, `cg::less`): every lane receives the fold over the tile. `None`
    /// when `name` is not `cg::reduce`.
    pub(super) fn tile_reduce(&mut self, call: Node<'_>, name: &str) -> Option<R<(ScalarExpr, E)>> {
        if !matches!(name, "cg::reduce" | "cooperative_groups::reduce") {
            return None;
        }
        let args = self.call_args(call);
        let [tile, v, op] = args.as_slice() else {
            return Some(self.not_yet(
                call,
                format!("`cg::reduce` with {} argument(s)", args.len()),
            ));
        };
        if self.tile_of(*tile).is_none() {
            return Some(self.not_yet(
                *tile,
                "`cg::reduce` over a group that is not a warp tile (`thread_block_tile`)",
            ));
        }
        let spelled = text(*op, self.src).replace(' ', "");
        let rop = if spelled.starts_with("cg::plus<") {
            ReduceOp::Sum
        } else if spelled.starts_with("cg::greater<") {
            ReduceOp::Max
        } else if spelled.starts_with("cg::less<") {
            ReduceOp::Min
        } else {
            return Some(self.not_yet(*op, format!("the reduction operator `{spelled}`")));
        };
        Some((|| {
            let (value, vt) = self.expr(*v)?;
            let result = self.new_local(vt);
            self.hoisted.push(Stmt::Collective {
                kind: CollectiveKind::Reduce(rop),
                value,
                result,
                scope: Scope::Subgroup,
            });
            Ok((ScalarExpr::Local(result), vt))
        })())
    }
}
