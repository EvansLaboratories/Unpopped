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
        let callee = e
            .child_by_field_name("function")
            .map(|f| self.short_callee(text(f, self.src)));
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
            Ok((
                ScalarExpr::Cast(Box::new(e), CastSpec { to, round, sat }),
                to,
            ))
        })())
    }

    /// `__fadd_rn(a, b)` and the other rounding-named arithmetic intrinsics (`__f{add,sub,mul,div,rcp,sqrt,maf}_{rn,rz,rd,ru}`,
    /// single-precision only): the plain operation wrapped in `Qualified` with that rounding, which also keeps
    /// a compiler from contracting it. A direction the operation does not take (`__frcp_rz`) declines. `None` when `name` is not one.
    fn rounded_arithmetic(&mut self, call: Node<'_>, name: &str) -> Option<R<(ScalarExpr, E)>> {
        let (op, ty, round) = rounded_intrinsic(name)?;
        if round != RoundMode::NearestEven && !op.takes_directed_round() {
            return Some(self.not_yet(
                call,
                format!("`{name}`: that operation takes no rounding but to nearest"),
            ));
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

    /// The CUDA fast-math intrinsics, written as the hardware sequences they are: `__expf(x)` is `ex2.approx(x * log2 e)`,
    /// `__exp10f` the same with `log2 10`, `__log2f` is `lg2.approx(x)`, `__logf` and `__log10f` scale that by `ln 2` and
    /// `log10 2`, `__fdividef(a, b)` an approximate divide. Each result is the oracle's `approx` node (within a few ulp of exact),
    /// which is the contract of the intrinsic. `None` when `name` is not one.
    fn fast_math(&mut self, call: Node<'_>, name: &str) -> Option<R<(ScalarExpr, E)>> {
        use std::f32::consts as k;
        let arity = match name {
            "__expf" | "__exp10f" | "__log2f" | "__logf" | "__log10f" => 1,
            "__fdividef" => 2,
            _ => return None,
        };
        let args = self.call_args(call);
        if args.len() != arity {
            return Some(self.not_yet(call, format!("`{name}` with {} argument(s)", args.len())));
        }
        Some((|| {
            let mut v = Vec::new();
            for a in &args {
                let (e, t) = self.expr(*a)?;
                v.push(self.convert(e, t, E::F32, *a)?);
            }
            let approx = || unpopped::ir::Numerics {
                round: RoundMode::NearestEven,
                ftz: false,
                approx: true,
            };
            let c = |x: f32| Box::new(ScalarExpr::Const(f64::from(x)));
            let un = |op, x: ScalarExpr| {
                ScalarExpr::Qualified(Box::new(ScalarExpr::Unary(op, Box::new(x))), approx())
            };
            let mut it = v.into_iter();
            let (a, b) = (it.next(), it.next());
            let e = match (name, a, b) {
                ("__expf", Some(x), _) => {
                    un(UnaryOp::Exp2, ScalarExpr::Mul(Box::new(x), c(k::LOG2_E)))
                }
                ("__exp10f", Some(x), _) => {
                    un(UnaryOp::Exp2, ScalarExpr::Mul(Box::new(x), c(k::LOG2_10)))
                }
                ("__log2f", Some(x), _) => un(UnaryOp::Log2, x),
                ("__logf", Some(x), _) => {
                    ScalarExpr::Mul(Box::new(un(UnaryOp::Log2, x)), c(k::LN_2))
                }
                ("__log10f", Some(x), _) => {
                    ScalarExpr::Mul(Box::new(un(UnaryOp::Log2, x)), c(k::LOG10_2))
                }
                ("__fdividef", Some(x), Some(y)) => ScalarExpr::Qualified(
                    Box::new(ScalarExpr::Div(Box::new(x), Box::new(y))),
                    approx(),
                ),
                _ => return self.not_yet(call, format!("`{name}`")),
            };
            Ok((e, E::F32))
        })())
    }

    /// `__half2float(h)`, `__bfloat162float(b)` (exact widening) and `__float2half[_rn](f)`, `__float2bfloat16[_rn](f)` (round to
    /// nearest even): one `Cast`. The widening takes only an argument of the 16-bit type it names (C would not compile any
    /// other); a narrowing overflows to infinity, as the intrinsic does. `None` when `name` is not one.
    fn half_conversion(&mut self, call: Node<'_>, name: &str) -> Option<R<(ScalarExpr, E)>> {
        let (from, to) = match name {
            "__half2float" => (Some(E::F16), E::F32),
            "__bfloat162float" => (Some(E::Bf16), E::F32),
            "__float2half" | "__float2half_rn" => (None, E::F16),
            "__float2bfloat16" | "__float2bfloat16_rn" => (None, E::Bf16),
            _ => return None,
        };
        let args = self.call_args(call);
        let [x] = args.as_slice() else {
            return Some(self.not_yet(call, format!("`{name}` with {} argument(s)", args.len())));
        };
        Some((|| {
            let (e, t) = self.expr(*x)?;
            if let Some(want) = from
                && t != want
            {
                return self.not_yet(*x, format!("`{name}` of a {t:?} value (it takes {want:?})"));
            }
            let e = if from.is_none() {
                self.convert(e, t, E::F32, *x)?
            } else {
                e
            };
            let spec = CastSpec {
                to,
                round: RoundMode::NearestEven,
                sat: Saturation::Wrap,
            };
            Ok((ScalarExpr::Cast(Box::new(e), spec), to))
        })())
    }

    /// `__low2half(h)`, `__high2half(h)` (a lane of a `half2`, as `half`) and `__low2float(h)`, `__high2float(h)` (as `float`).
    /// `None` when `name` is not one.
    fn half2_lane(&mut self, call: Node<'_>, name: &str) -> Option<R<(ScalarExpr, E)>> {
        let (lane, to) = match name {
            "__low2half" => (0, E::F16),
            "__high2half" => (1, E::F16),
            "__low2float" => (0, E::F32),
            "__high2float" => (1, E::F32),
            _ => return None,
        };
        let args = self.call_args(call);
        let [x] = args.as_slice() else {
            return Some(self.not_yet(call, format!("`{name}` with {} argument(s)", args.len())));
        };
        Some((|| {
            let mut lanes = self.vector_value(*x, (E::F16, 2))?;
            let v = lanes.swap_remove(lane);
            if to == E::F16 {
                return Ok((v, E::F16));
            }
            let spec = CastSpec {
                to,
                round: RoundMode::NearestEven,
                sat: Saturation::Wrap,
            };
            Ok((ScalarExpr::Cast(Box::new(v), spec), to))
        })())
    }

    /// `isnan(x)`, `isinf(x)`, `isfinite(x)` on a float: `x != x`; `|x| == inf`; `x - x == 0` (NaN and infinity both make
    /// `x - x` not equal to zero). `None` when `name` is not one.
    fn float_class(&mut self, call: Node<'_>, name: &str) -> Option<R<(ScalarExpr, E)>> {
        if !matches!(name, "isnan" | "isinf" | "isfinite") {
            return None;
        }
        let args = self.call_args(call);
        let [x] = args.as_slice() else {
            return Some(self.not_yet(call, format!("`{name}` with {} argument(s)", args.len())));
        };
        Some((|| {
            let (e, t) = self.expr(*x)?;
            if !matches!(t, E::F32 | E::F64) {
                return self.not_yet(
                    *x,
                    format!("`{name}` of a {t:?} value (it takes a float or double)"),
                );
            }
            let x2 = || (e.clone(), t);
            match name {
                "isnan" => self.binary("!=", x2(), x2(), call),
                "isinf" => {
                    let abs = ScalarExpr::Unary(UnaryOp::Abs, Box::new(e.clone()));
                    self.binary("==", (abs, t), (ScalarExpr::Const(f64::INFINITY), t), call)
                }
                _ => {
                    let diff = self.binary("-", x2(), x2(), call)?;
                    self.binary("==", diff, (ScalarExpr::Const(0.0), t), call)
                }
            }
        })())
    }

    /// `(Acc)(x)` where the grammar read `(Acc)` as a parenthesised expression because it did not know `Acc` is a type: when the
    /// name is a type alias (or scalar type) and no variable of that name is in scope, it is the C cast to it. `None` otherwise.
    fn parenthesised_type_cast(
        &mut self,
        call: Node<'_>,
        f: Node<'_>,
    ) -> Option<R<(ScalarExpr, E)>> {
        if f.kind() != "parenthesized_expression" {
            return None;
        }
        let inner = f
            .named_child(0)
            .filter(|i| matches!(i.kind(), "identifier" | "type_identifier"))?;
        let name = text(inner, self.src);
        if self.lookup(name).is_some() {
            return None;
        }
        let to = self.ctype(name).ok()?;
        let args = self.call_args(call);
        let [x] = args.as_slice() else {
            return Some(
                self.not_yet(call, format!("a cast to `{name}` of {} values", args.len())),
            );
        };
        Some((|| {
            let (e, from) = self.expr(*x)?;
            Ok((self.convert(e, from, to, *x)?, to))
        })())
    }

    /// A namespace-qualified call (`baracuda::block_reduce_max_f32(..)`) of a function this source defines is that function by its
    /// short name (the walker is namespace-blind, as for qualified template calls). Any other name is returned as written.
    fn short_callee<'a>(&self, written: &'a str) -> &'a str {
        // `(f)(x)`: a callee in parentheses (a function substituted for a template parameter, once per level) is `f`, but only
        // when `f` is a function this source defines: `(Idx)(x)` with a local `Idx` is not a call of it
        let mut name = written.trim();
        while let Some(inner) = name
            .strip_prefix('(')
            .and_then(|r| r.strip_suffix(')'))
            .map(str::trim)
            .filter(|n| {
                !n.is_empty()
                    && n.chars().all(|c| {
                        c.is_ascii_alphanumeric() || c == '_' || c == ':' || c == '(' || c == ')'
                    })
            })
        {
            name = inner;
        }
        let known = |n: &str| {
            self.defines(n)
                || n.rsplit_once("::")
                    .is_some_and(|(_, short)| self.defines(short))
        };
        if name.len() != written.trim().len() && !known(name) {
            return written;
        }
        if self.defines(name) {
            return name;
        }
        match name.rsplit_once("::") {
            Some((_, short)) if self.defines(short) => short,
            _ => name,
        }
    }

    /// A math call through the frontend's tables, or an atomic that returns the old value (hoisted as a statement).
    pub(super) fn call_expr(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let Some(f) = n.child_by_field_name("function") else {
            return self.not_yet(n, "a call of an unexpected shape");
        };
        if let Some(r) = self.parenthesised_type_cast(n, f) {
            return r;
        }
        if f.kind() == "field_expression" {
            if let Some(r) = self.group_value(n, f) {
                return r;
            }
            if let Some(r) = self.tile_value(n, f) {
                return r;
            }
            return self.method_value(n, f);
        }
        let name = self.short_callee(text(f, self.src));
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
        if let Some(r) = self.fast_math(n, name) {
            return r;
        }
        if let Some(r) = self.half_conversion(n, name) {
            return r;
        }
        if let Some(r) = self.half2_lane(n, name) {
            return r;
        }
        if let Some(r) = self.float_class(n, name) {
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
