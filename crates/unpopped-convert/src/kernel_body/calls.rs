// SPDX-License-Identifier: MIT OR Apache-2.0
//! Calls: math functions through the frontend's tables, atomics and call statements (ROADMAP U19).

use super::*;

impl<'s, 't> Walker<'s, 't> {
    /// A call statement: only `__syncthreads()` (a block barrier) in slice 2.
    pub(super) fn call_statement(&mut self, e: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        let callee = e.child_by_field_name("function").map(|f| text(f, self.src));
        let nargs = e
            .child_by_field_name("arguments")
            .map_or(0, |a| a.named_child_count());
        match (callee, nargs) {
            (Some("__syncthreads"), 0) if self.launch.is_some() => push(
                out,
                Stmt::Barrier {
                    scope: Scope::Block,
                    kind: BarrierKind::Full,
                },
            ),
            (Some(f), _) if self.defines(f) => {
                self.flush(out);
                self.inline_call(e, f, out)?;
                Ok(())
            }
            (Some(f), _) if f.starts_with("atomic") => {
                self.atomic(e, f, false)?;
                self.flush(out);
                Ok(())
            }
            (Some("__syncthreads"), _) => self.not_yet(
                e,
                "`__syncthreads` without a launch (use `lift_kernel_launched`) or with arguments",
            ),
            (Some(f), _) => self.not_yet(e, format!("a call to `{f}`")),
            (None, _) => self.not_yet(e, "a call"),
        }
    }

    pub(super) fn call_args<'n>(&self, n: Node<'n>) -> Vec<Node<'n>> {
        n.child_by_field_name("arguments")
            .map_or_else(Vec::new, |a| {
                let mut c = a.walk();
                a.named_children(&mut c).collect()
            })
    }

    /// A math call through the frontend's tables, or an atomic that returns the old value (hoisted as a statement).
    pub(super) fn call_expr(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let Some(f) = n.child_by_field_name("function") else {
            return self.not_yet(n, "a call of an unexpected shape");
        };
        let name = text(f, self.src);
        if self.defines(name) {
            let mut body = Vec::new();
            let r = self.inline_call(n, name, &mut body)?;
            self.hoisted.append(&mut body);
            return match r {
                Some(v) => Ok(v),
                None => self.not_yet(n, format!("the result of `{name}`, which returns void")),
            };
        }
        if name.starts_with("atomic") {
            return match self.atomic(n, name, true)? {
                Some(v) => Ok(v),
                None => self.not_yet(n, "an atomic without a result"),
            };
        }
        let args = self.call_args(n);
        if name == "__ldg" && args.len() == 1 {
            // `__ldg(&a[i])` is a read through the read-only cache: a plain load (the cache path is a hint, never a meaning)
            let (place, elem, _) = self.address_of(args[0])?;
            let local = self.new_local(elem);
            self.hoisted.push(Stmt::Load { local, place });
            return Ok((ScalarExpr::Local(local), elem));
        }
        match args.len() {
            1 if (self.unary_fn)(name).is_some() => self.math_unary(n, name, args[0]),
            2 if (self.binary_fn)(name).is_some() => self.math_binary(n, name, args[0], args[1]),
            _ => self.not_yet(
                n,
                format!("a call to `{name}` with {} argument(s)", args.len()),
            ),
        }
    }

    /// The type a math function computes at: the `f`-suffixed C names are `float`, the others take their argument's type.
    pub(super) fn math_type(name: &str, arg: E) -> E {
        let single = name.ends_with('f') && !matches!(name, "fabs" | "abs");
        if single && name != "__fsqrt_rn" {
            E::F32
        } else {
            arg
        }
    }

    pub(super) fn math_unary(
        &mut self,
        n: Node<'_>,
        name: &str,
        a: Node<'_>,
    ) -> R<(ScalarExpr, E)> {
        let Some(op) = (self.unary_fn)(name) else {
            return self.not_yet(n, format!("a call to `{name}`"));
        };
        let (v, t) = self.expr(a)?;
        let at = Self::math_type(name, t);
        let v = self.convert(v, t, at, a)?;
        let e = ScalarExpr::Unary(op, Box::new(v));
        let ty = self.ty_of(&e, n)?;
        Ok((e, ty))
    }

    pub(super) fn math_binary(
        &mut self,
        n: Node<'_>,
        name: &str,
        a: Node<'_>,
        b: Node<'_>,
    ) -> R<(ScalarExpr, E)> {
        let Some(op) = (self.binary_fn)(name) else {
            return self.not_yet(n, format!("a call to `{name}`"));
        };
        let (x, xt) = self.expr(a)?;
        let (y, yt) = self.expr(b)?;
        let common = self.common_type(xt, yt, n)?;
        let at = Self::math_type(name, common);
        let x = self.convert(x, xt, at, a)?;
        let y = self.convert(y, yt, at, b)?;
        let e = ScalarExpr::Binary(op, Box::new(x), Box::new(y));
        let ty = self.ty_of(&e, n)?;
        Ok((e, ty))
    }

    /// `atomicAdd(&a[i], v)` and its kin: one `AtomicRmw` (or `AtomicCas`), relaxed, at device scope (block scope on a
    /// `__shared__` array), as CUDA's are. With `want_old` the old value is a fresh local and the result.
    pub(super) fn atomic(
        &mut self,
        n: Node<'_>,
        name: &str,
        want_old: bool,
    ) -> R<Option<(ScalarExpr, E)>> {
        let args = self.call_args(n);
        let op = match name {
            "atomicAdd" => Some(AtomicOp::Add),
            "atomicMax" => Some(AtomicOp::Max),
            "atomicMin" => Some(AtomicOp::Min),
            "atomicAnd" => Some(AtomicOp::And),
            "atomicOr" => Some(AtomicOp::Or),
            "atomicXor" => Some(AtomicOp::Xor),
            "atomicExch" => Some(AtomicOp::Exchange),
            "atomicCAS" => None,
            other => return self.not_yet(n, format!("the atomic `{other}`")),
        };
        let want = if op.is_some() { 2 } else { 3 };
        if args.len() != want {
            return self.not_yet(n, format!("`{name}` with {} argument(s)", args.len()));
        }
        let (place, elem, scope) = self.address_of(args[0])?;
        if matches!(place, Place::Raw { .. } | Place::AllocRaw { .. }) {
            return self.not_yet(
                n,
                "an atomic through a pointer (the IR's atomics take no raw place)",
            );
        }
        let old = want_old.then(|| self.new_local(elem));
        let first = self.expr(args[1])?;
        let first = self.convert(first.0, first.1, elem, args[1])?;
        let stmt = match op {
            Some(op) => Stmt::AtomicRmw {
                op,
                place,
                value: first,
                old,
                order: MemoryOrder::Relaxed,
                scope,
            },
            None => {
                let d = self.expr(args[2])?;
                let desired = self.convert(d.0, d.1, elem, args[2])?;
                Stmt::AtomicCas {
                    place,
                    expected: first,
                    desired,
                    old,
                    order: MemoryOrder::Relaxed,
                    scope,
                }
            }
        };
        self.hoisted.push(stmt);
        Ok(old.map(|l| (ScalarExpr::Local(l), elem)))
    }
}
