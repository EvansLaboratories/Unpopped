// SPDX-License-Identifier: MIT OR Apache-2.0
//! Calls: math functions through the frontend's tables, atomics and call statements (ROADMAP U19).

use super::*;
use unpopped::ir::QualOp;

impl<'s, 't> Walker<'s, 't> {
    /// A call statement: only `__syncthreads()` (a block barrier) in slice 2.
    pub(super) fn call_statement(&mut self, e: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        if let Some(f) = e
            .child_by_field_name("function")
            .filter(|f| f.kind() == "field_expression")
        {
            if let Some(r) = self.group_statement(e, f, out) {
                return r;
            }
            if let Some(r) = self.tile_statement(e, f, out) {
                return r;
            }
            return self.method_statement(e, f, out);
        }
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
            (Some(f), _) => self.not_yet(e, self.unexpanded_or(f, format!("a call to `{f}`"))),
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

    /// `__float2int_rn(x)` and the rest of the CUDA conversion intrinsics `__{float,double}2{int,uint,ll,ull}_{rn,rz,rd,ru}` and
    /// `__{int,uint,ll,ull}2{float,double}_{rn,rz,rd,ru}`: one [`ScalarExpr::Cast`] with the named rounding. Float to integer
    /// clamps to the integer's range (NaN gives 0), as the intrinsic does. `None` when `name` is not one.
    fn convert_intrinsic(&mut self, call: Node<'_>, name: &str) -> Option<R<(ScalarExpr, E)>> {
        let (from, to, round) = conversion_intrinsic(name)?;
        let args = self.call_args(call);
        let [x] = args.as_slice() else {
            return Some(self.not_yet(call, format!("`{name}` with {} argument(s)", args.len())));
        };
        Some((|| {
            let (e, t) = self.expr(*x)?;
            let e = self.convert(e, t, from, *x)?;
            let sat = if int_range(to).is_some() {
                Saturation::Clamp
            } else {
                Saturation::Wrap
            };
            Ok((ScalarExpr::Cast(Box::new(e), CastSpec { to, round, sat }), to))
        })())
    }

    /// `__fadd_rn(a, b)` and the other rounding-named arithmetic intrinsics (`__f{add,sub,mul,div,rcp,sqrt,maf}_{rn,rz,rd,ru}`,
    /// single-precision only): the plain operation wrapped in `Qualified` with that rounding, which also keeps
    /// a compiler from contracting it. A direction the operation does not take (`__frcp_rz`) declines. `None` when `name` is not one.
    fn rounded_arithmetic(&mut self, call: Node<'_>, name: &str) -> Option<R<(ScalarExpr, E)>> {
        let (op, ty, round) = rounded_intrinsic(name)?;
        if round != RoundMode::NearestEven && !op.takes_directed_round() {
            return Some(self.not_yet(call, format!("`{name}`: that operation takes no rounding but to nearest")));
        }
        let args = self.call_args(call);
        if args.len() != op.arity() {
            return Some(self.not_yet(call, format!("`{name}` with {} argument(s)", args.len())));
        }
        Some((|| {
            let mut v = Vec::new();
            for a in &args {
                let (e, t) = self.expr(*a)?;
                v.push(Box::new(self.convert(e, t, ty, *a)?));
            }
            let mut it = v.into_iter();
            let (a, b, c) = (it.next(), it.next(), it.next());
            let plain = match (op, a, b, c) {
                (QualOp::Add, Some(a), Some(b), _) => ScalarExpr::Add(a, b),
                (QualOp::Sub, Some(a), Some(b), _) => ScalarExpr::Sub(a, b),
                (QualOp::Mul, Some(a), Some(b), _) => ScalarExpr::Mul(a, b),
                (QualOp::Div, Some(a), Some(b), _) => ScalarExpr::Div(a, b),
                (QualOp::Fma, Some(a), Some(b), Some(c)) => ScalarExpr::Fma(a, b, c),
                (QualOp::Sqrt, Some(x), _, _) => ScalarExpr::Unary(UnaryOp::Sqrt, x),
                (QualOp::Recip, Some(x), _, _) => ScalarExpr::Unary(UnaryOp::Recip, x),
                _ => return self.not_yet(call, format!("`{name}`")),
            };
            let n = unpopped::ir::Numerics {
                round,
                ftz: false,
                approx: false,
            };
            Ok((ScalarExpr::Qualified(Box::new(plain), n), ty))
        })())
    }

    /// A math call through the frontend's tables, or an atomic that returns the old value (hoisted as a statement).
    pub(super) fn call_expr(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let Some(f) = n.child_by_field_name("function") else {
            return self.not_yet(n, "a call of an unexpected shape");
        };
        if f.kind() == "field_expression" {
            if let Some(r) = self.group_value(n, f) {
                return r;
            }
            if let Some(r) = self.tile_value(n, f) {
                return r;
            }
            return self.method_value(n, f);
        }
        let name = text(f, self.src);
        if let Some(r) = self.tile_reduce(n, name) {
            return r;
        }
        if let Some(r) = self.sync_shuffle(n, name) {
            return r;
        }
        if let Some(r) = self.convert_intrinsic(n, name) {
            return r;
        }
        if let Some(r) = self.rounded_arithmetic(n, name) {
            return r;
        }
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
        // a C++ functional cast `uint64_t(x)` or `static_cast<uint64_t>(x)`: the same conversion as `(uint64_t)x`
        let cast_to = name
            .strip_prefix("static_cast<")
            .and_then(|r| r.strip_suffix('>'))
            .map_or(name, str::trim);
        if let (Ok(to), 1) = (self.ctype(cast_to), args.len()) {
            let (v, from) = self.expr(args[0])?;
            return Ok((self.convert(v, from, to, n)?, to));
        }
        if matches!(name, "min" | "max") && args.len() == 2 {
            return self.min_max(n, name == "min", args[0], args[1]);
        }
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
                self.unexpanded_or(
                    name,
                    format!("a call to `{name}` with {} argument(s)", args.len()),
                ),
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
            return self.not_yet(n, self.unexpanded_or(name, format!("a call to `{name}`")));
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
            return self.not_yet(n, self.unexpanded_or(name, format!("a call to `{name}`")));
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

    /// The element place a raw (byte-offset) place names, when it is an element of the base's own type: an atomic's place is an
    /// element, and a pointer into a buffer or array (`T* p = &a[i]`, a helper's `T* addr`) is the byte offset `i * sizeof(T)`.
    /// A place at another type (a cast pointer) stays raw and the atomic declines.
    fn unraw(&self, place: Place) -> Place {
        match place {
            Place::Raw {
                buffer,
                byte_offset,
                elem,
            } if matches!(
                self.params.get(buffer as usize).map(|p| &p.kind),
                Some(unpopped::kernel::ParamKind::Buffer { elem: be, rank: Some(1), .. }) if *be == elem
            ) =>
            {
                Place::Logical {
                    buffer,
                    coords: vec![element_index(byte_offset, elem)],
                }
            }
            Place::AllocRaw {
                alloc,
                byte_offset,
                elem,
            } if self
                .allocs
                .get(alloc as usize)
                .is_some_and(|a| a.elem == elem) =>
            {
                Place::Alloc {
                    alloc,
                    index: element_index(byte_offset, elem),
                }
            }
            other => other,
        }
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
        let Some(op) = atomic_op(name) else {
            return self.not_yet(n, format!("the atomic `{name}`"));
        };
        let want = if op.is_some() { 2 } else { 3 };
        if args.len() != want {
            return self.not_yet(n, format!("`{name}` with {} argument(s)", args.len()));
        }
        let (place, elem, scope) = self.address_of(args[0])?;
        let place = self.unraw(place);
        if matches!(place, Place::Raw { .. } | Place::AllocRaw { .. }) {
            return self.not_yet(
                n,
                "an atomic through a pointer (the IR's atomics take no raw place)",
            );
        }
        let old = want_old.then(|| self.new_local(elem));
        let stmt = self.atomic_stmt(op, &args, (place, elem, scope), old)?;
        self.hoisted.push(stmt);
        Ok(old.map(|l| (ScalarExpr::Local(l), elem)))
    }

    /// The `AtomicRmw` (or, for `atomicCAS`, the `AtomicCas`) of `args` on `target`.
    fn atomic_stmt(
        &mut self,
        op: Option<AtomicOp>,
        args: &[Node<'_>],
        (place, elem, scope): (Place, E, Scope),
        old: Option<u32>,
    ) -> R<Stmt> {
        let first = self.expr(args[1])?;
        let first = self.convert(first.0, first.1, elem, args[1])?;
        let order = MemoryOrder::Relaxed;
        Ok(match op {
            Some(op) => Stmt::AtomicRmw {
                op,
                place,
                value: first,
                old,
                order,
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
                    order,
                    scope,
                }
            }
        })
    }

    /// CUDA `min(a, b)` and `max(a, b)`: the plain integer minimum or maximum, and on floats `fminf`/`fmaxf` (the IEEE form that
    /// returns the other operand when one is NaN). Both operands convert to their common type first.
    fn min_max(
        &mut self,
        n: Node<'_>,
        is_min: bool,
        a: Node<'_>,
        b: Node<'_>,
    ) -> R<(ScalarExpr, E)> {
        let (x, xt) = self.expr(a)?;
        let (y, yt) = self.expr(b)?;
        let common = self.common_type(xt, yt, n)?;
        let x = self.convert(x, xt, common, a)?;
        let y = self.convert(y, yt, common, b)?;
        let op = match (is_min, int_range(common).is_some()) {
            (true, true) => BinaryOp::Min,
            (false, true) => BinaryOp::Max,
            (true, false) => BinaryOp::FminIeee,
            (false, false) => BinaryOp::FmaxIeee,
        };
        let e = ScalarExpr::Binary(op, Box::new(x), Box::new(y));
        let ty = self.ty_of(&e, n)?;
        Ok((e, ty))
    }
}

/// The IR operation of a CUDA atomic: `Some(Some(op))` for a read-modify-write, `Some(None)` for `atomicCAS`, `None` for any
/// other name.
fn atomic_op(name: &str) -> Option<Option<AtomicOp>> {
    Some(match name {
        "atomicAdd" => Some(AtomicOp::Add),
        "atomicMax" => Some(AtomicOp::Max),
        "atomicMin" => Some(AtomicOp::Min),
        "atomicAnd" => Some(AtomicOp::And),
        "atomicOr" => Some(AtomicOp::Or),
        "atomicXor" => Some(AtomicOp::Xor),
        "atomicExch" => Some(AtomicOp::Exchange),
        "atomicCAS" => None,
        _ => return None,
    })
}

/// The element index of a byte offset into storage of `elem`-typed elements (`elem` has a whole byte size).
fn element_index(byte_offset: ScalarExpr, elem: E) -> ScalarExpr {
    let size = super::pointer::byte_size(elem).unwrap_or(1);
    ScalarExpr::Div(
        Box::new(byte_offset),
        Box::new(ScalarExpr::ConstInt(size, E::I64)),
    )
}

/// The source type, destination type and rounding of a CUDA conversion intrinsic name.
fn conversion_intrinsic(name: &str) -> Option<(E, E, RoundMode)> {
    let rest = name.strip_prefix("__")?;
    let (kinds, mode) = rest.rsplit_once('_')?;
    let round = match mode {
        "rn" => RoundMode::NearestEven,
        "rz" => RoundMode::TowardZero,
        "rd" => RoundMode::TowardNegInf,
        "ru" => RoundMode::TowardPosInf,
        _ => return None,
    };
    let (a, b) = kinds.split_once('2')?;
    let ty = |s: &str| {
        Some(match s {
            "float" => E::F32,
            "double" => E::F64,
            "int" => E::I32,
            "uint" => E::U32,
            "ll" => E::I64,
            "ull" => E::U64,
            _ => return None,
        })
    };
    let (from, to) = (ty(a)?, ty(b)?);
    // exactly one side is a float
    (matches!(from, E::F32 | E::F64) != matches!(to, E::F32 | E::F64)).then_some((from, to, round))
}

/// The operation, element type and rounding of a rounding-named arithmetic intrinsic (`__fadd_rn`, `__fmul_rz`, `__fmaf_rd`).
fn rounded_intrinsic(name: &str) -> Option<(QualOp, E, RoundMode)> {
    let rest = name.strip_prefix("__")?;
    let (stem, mode) = rest.rsplit_once('_')?;
    let round = match mode {
        "rn" => RoundMode::NearestEven,
        "rz" => RoundMode::TowardZero,
        "rd" => RoundMode::TowardNegInf,
        "ru" => RoundMode::TowardPosInf,
        _ => return None,
    };
    let (op, ty) = match stem {
        "fadd" => (QualOp::Add, E::F32),
        "fsub" => (QualOp::Sub, E::F32),
        "fmul" => (QualOp::Mul, E::F32),
        "fdiv" => (QualOp::Div, E::F32),
        "frcp" => (QualOp::Recip, E::F32),
        "fsqrt" => (QualOp::Sqrt, E::F32),
        "fmaf" => (QualOp::Fma, E::F32),
        _ => return None,
    };
    Some((op, ty, round))
}
