// SPDX-License-Identifier: MIT OR Apache-2.0
//! The cpu_c emitter for a kernel-layer [`KernelFn`] (`docs/e1-emit-from-the-kernel-layer.md`,
//! stage E1a): **serial C**, one logical thread, the statements spelled as C statements.
//!
//! This is a correctness reference, not a fast kernel: nothing here maps a loop to threads, and
//! its output is never a performance or a device-timing claim (PM ruling 2026-10-30).
//!
//! # The C function
//!
//! `int NAME(args)`, returning `0` when the kernel finished, `1` when an access fell outside its
//! buffer (the interpreter's `OutOfBounds` trap) and `2` for a zero loop step (its `ZeroStep`).
//! Arguments, in `KernelFn::params` order:
//!
//! - a buffer `i` is four arguments, `T* bI, const long long* bI_shape, const long long* bI_stride,
//!   long long bI_base` (`const T*` for a `Read` buffer): the storage, the per-axis extents and
//!   signed element strides, and the element offset added before address math, exactly the
//!   fields of the oracle's `TypedBuffer`;
//! - a scalar `i` is `T pI`, bound to its local.
//!
//! A kernel the emitter has not been taught (a packed local, an unsupported dtype, a form of launch
//! `launch.rs` does not spell) is [`LowerError::UnsupportedKernel`]; nothing is spelled by default.
//!
//! A kernel with a [`unpopped::exec::LaunchShape`] is spelled by `launch.rs` as a **serialized thread
//! loop**: the blocks one after another and, in a block, the threads one after another, the loop split at
//! each barrier. Everything here that is not about launches is shared with it.

use std::fmt::Write as _;

use unpopped::backend::{Decline, DeclinedOp, GeneratedKernel, LowerError};
use unpopped::backend::{Lowering, Spelling, const_lit, lower_expr};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    Block, BufferRole, CallArg, KernelFn, ParamKind, Place, Stmt, check_kernel,
};
use unpopped::types::Ty;
use unpopped_vocab::ElementKind;

use crate::launch::LaunchInfo;
use crate::{cpu_binary, cpu_fma, cpu_select, cpu_unary, qualified, typed};

pub(crate) fn refuse(detail: impl Into<String>) -> LowerError {
    LowerError::UnsupportedKernel {
        detail: format!("cpu_c: {}", detail.into()),
    }
}

/// The C type of a scalar the emitter handles, or the refusal.
pub(crate) fn ctype(t: ElementKind, what: &str) -> Result<&'static str, LowerError> {
    typed::c_type(t).ok_or_else(|| {
        refuse(format!(
            "{what} has dtype {t:?}, which this emitter does not spell"
        ))
    })
}

/// Everything about `k` the emitter cannot do, refused before any text is written.
fn admit(k: &KernelFn, in_module: bool) -> Result<(), LowerError> {
    if !in_module {
        check_kernel(k).map_err(|e| refuse(format!("kernel `{}` is ill-formed: {e:?}", k.name)))?;
    }
    if k.launch.is_some() {
        crate::launch::admit_launch(k)?;
    } else if !k.allocs.is_empty() {
        return Err(refuse("an allocation without a launch shape"));
    }
    if !matches!(
        k.compute,
        ElementKind::F32
            | ElementKind::F64
            | ElementKind::I32
            | ElementKind::I64
            | ElementKind::U32
            | ElementKind::U64
    ) {
        return Err(refuse(format!(
            "compute dtype {:?}; only F32, F64, I32, I64, U32 and U64",
            k.compute
        )));
    }
    for p in &k.params {
        match &p.kind {
            ParamKind::Buffer { elem, .. } => ctype(*elem, &format!("buffer `{}`", p.name))?,
            ParamKind::Scalar { ty, .. } => ctype(*ty, &format!("scalar `{}`", p.name))?,
            _ => return Err(refuse("a parameter kind this emitter has not been taught")),
        };
    }
    for (i, d) in k.locals.iter().enumerate() {
        if d.lanes != 1 {
            return Err(refuse(format!("local {i} is packed ({} lanes)", d.lanes)));
        }
        ctype(d.elem, &format!("local {i}"))?;
    }
    Ok(())
}

/// `x` as a C identifier fragment.
pub(crate) fn ident(x: &str) -> String {
    x.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

pub(crate) struct Emitter<'a> {
    pub(crate) k: &'a KernelFn,
    pub(crate) lo: &'a Lowering<'a>,
    pub(crate) out: String,
    pub(crate) depth: usize,
    /// Fresh-name counter for the offset and loop temporaries.
    pub(crate) fresh: u32,
    /// The launch, when the kernel has one: locals become per-thread arrays (`launch.rs`).
    pub(crate) launch: Option<LaunchInfo>,
    /// Thread-level C loops open inside the current thread run (a `Break` needs one).
    pub(crate) thread_loops: u32,
    /// The module a `Call` resolves in (`module.rs`); `None` for a lone kernel.
    pub(crate) module: Option<&'a ModuleCtx<'a>>,
    /// In a callee: the result local, declared up front and stored through `res` on every exit.
    pub(crate) result: Option<u32>,
}

/// The functions of the module being lowered and the C name of each.
pub(crate) struct ModuleCtx<'a> {
    pub(crate) fns: &'a [KernelFn],
    pub(crate) names: Vec<String>,
    /// The entry's launch: when present, a callee takes the thread's ids as six extra C arguments.
    pub(crate) launch: Option<LaunchInfo>,
}

impl Emitter<'_> {
    pub(crate) fn line(&mut self, text: &str) {
        for _ in 0..self.depth {
            self.out.push_str("    ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    pub(crate) fn expr(&self, e: &ScalarExpr) -> Result<String, LowerError> {
        lower_expr(e, self.lo)?.spelled().map_err(LowerError::from)
    }

    pub(crate) fn local_ctype(&self, id: u32) -> Result<&'static str, LowerError> {
        ctype(self.k.locals[id as usize].elem, "a local")
    }

    pub(crate) fn block(&mut self, b: &Block) -> Result<(), LowerError> {
        for s in &b.0 {
            self.stmt(s)?;
        }
        Ok(())
    }

    /// `local` as C reads and writes it: a plain variable, or this thread's element of a per-thread array.
    pub(crate) fn local_ref(&self, local: u32) -> String {
        if self.launch.is_some() {
            format!("l{local}[t]")
        } else {
            format!("l{local}")
        }
    }

    /// `local = (T)(value);`, declaring it when the kernel is serial and assigning its array element
    /// when it is launched (the array is declared once, up front).
    fn bind_local(&mut self, local: u32, value: &str) -> Result<(), LowerError> {
        let t = self.local_ctype(local)?;
        if self.launch.is_some() || self.result == Some(local) {
            let r = self.local_ref(local);
            self.line(&format!("{r} = ({t})({value});"));
        } else {
            self.line(&format!("{t} l{local} = ({t})({value});"));
        }
        Ok(())
    }

    pub(crate) fn stmt(&mut self, s: &Stmt) -> Result<(), LowerError> {
        match s {
            Stmt::Let { local, init } => {
                let v = self.expr(init)?;
                self.bind_local(*local, &v)?;
            }
            Stmt::Assign { local, value } => {
                let (t, v) = (self.local_ctype(*local)?, self.expr(value)?);
                let r = self.local_ref(*local);
                self.line(&format!("{r} = ({t})({v});"));
            }
            Stmt::Load { local, place } => {
                let at = self.address(place)?;
                self.bind_local(*local, &at)?;
            }
            Stmt::Store { place, value } => {
                let (at, v) = (self.address(place)?, self.expr(value)?);
                self.line(&format!("{at} = {v};"));
            }
            Stmt::If { cond, then, els } => self.if_stmt(cond, then, els)?,
            Stmt::For {
                local,
                start,
                end,
                step,
                body,
            } => {
                self.for_stmt(*local, [start, end, step], body)?;
            }
            Stmt::While { cond, body } => {
                let c = self.expr(cond)?;
                self.line(&format!("while (({c}) != 0) {{"));
                self.nested_loop(body)?;
                self.line("}");
            }
            Stmt::Break | Stmt::Continue => self.escape(matches!(s, Stmt::Break))?,
            Stmt::Return => self.ret()?,
            Stmt::Barrier { .. } => return Err(refuse("a Barrier needs a launch (U3 row 4)")),
            Stmt::AtomicRmw { .. } => self.atomic_rmw(s)?,
            Stmt::AtomicCas { .. } => self.atomic_cas(s)?,
            // the serial thread loop is one valid schedule: a fence orders nothing more there
            Stmt::Fence { .. } => {}
            Stmt::Collective { .. } => return Err(refuse("a Collective is block-level only")),
            Stmt::Call {
                callee,
                args,
                result,
            } => self.call_stmt(*callee, args, *result)?,
        }
        Ok(())
    }

    /// `Break` or `Continue`. In a launch it must target a loop inside the current thread run: one that
    /// targets a block-level loop (a loop holding a barrier) cannot be spelled as a thread loop.
    fn escape(&mut self, is_break: bool) -> Result<(), LowerError> {
        if self.launch.is_some() && self.thread_loops == 0 {
            return Err(refuse(
                "a Break or Continue that leaves a loop holding a Barrier (the thread loop cannot spell it)",
            ));
        }
        self.line(if is_break { "break;" } else { "continue;" });
        Ok(())
    }

    /// `Return`: leave the kernel, or in a launch end this thread's run (a kernel with a barrier
    /// refuses it before this is reached).
    fn ret(&mut self) -> Result<(), LowerError> {
        if self.launch.is_some() {
            self.line("goto thread_end;");
        } else if let Some(r) = self.result {
            self.line(&format!("{{ *res = l{r}; return 0; }}"));
        } else {
            self.line("return 0;");
        }
        Ok(())
    }

    /// `result = callee(args...)` as a C call: the arguments forwarded or cast, a nonzero status returned at
    /// once, the callee's result bound to the caller's local.
    fn call_stmt(
        &mut self,
        callee: u32,
        args: &[CallArg],
        result: Option<u32>,
    ) -> Result<(), LowerError> {
        let Some(m) = self.module else {
            return Err(refuse("a Call outside a module (use lower_module)"));
        };
        let f = &m.fns[callee as usize];
        let mut actuals = self.call_actuals(f, args)?;
        actuals.extend(self.id_actuals(m)?);
        let n = self.fresh;
        self.fresh += 1;
        let rtype = match f.result {
            Some(r) => Some(ctype(f.locals[r as usize].elem, "a callee result")?),
            None => None,
        };
        if let Some(t) = rtype {
            self.line(&format!("{t} r{n} = 0;"));
            actuals.push(format!("&r{n}"));
        }
        self.line(&format!(
            "int st{n} = {}({});",
            m.names[callee as usize],
            actuals.join(", ")
        ));
        self.line(&format!("if (st{n}) return st{n};"));
        if let (Some(dst), Some(_)) = (result, rtype) {
            self.bind_local(dst, &format!("r{n}"))?;
        }
        Ok(())
    }

    /// The thread's ids handed to a callee when the module is launched: `ThreadIdx` then `BlockIdx`, each of
    /// the three axes. The launched entry spells them from its thread loop; a callee forwards its own.
    fn id_actuals(&self, m: &ModuleCtx<'_>) -> Result<Vec<String>, LowerError> {
        if m.launch.is_none() {
            return Ok(Vec::new());
        }
        let Some(info) = self.launch else {
            return Ok(["ti0", "ti1", "ti2", "bi0", "bi1", "bi2"]
                .map(String::from)
                .to_vec());
        };
        let ids = [0u8, 1, 2]
            .map(unpopped::exec::ExecId::ThreadIdx)
            .into_iter();
        let blocks = [0u8, 1, 2].map(unpopped::exec::ExecId::BlockIdx);
        ids.chain(blocks)
            .map(|id| crate::launch::exec_c(info, id))
            .collect()
    }

    /// The C arguments of a call, in the callee's parameter order.
    fn call_actuals(&self, f: &KernelFn, args: &[CallArg]) -> Result<Vec<String>, LowerError> {
        let mut out = Vec::new();
        for (a, p) in args.iter().zip(&f.params) {
            match (a, &p.kind) {
                (CallArg::Scalar(e), ParamKind::Scalar { ty, .. }) => {
                    out.push(format!("({})({})", ctype(*ty, "a scalar")?, self.expr(e)?));
                }
                (CallArg::Buffer(b), ParamKind::Buffer { .. }) => {
                    out.push(format!("b{b}, b{b}_shape, b{b}_stride, b{b}_base"));
                }
                (CallArg::Alloc(a), ParamKind::Buffer { .. }) if self.launch.is_some() => {
                    // the block's `Shared` array, with the one-dimensional shape and stride it declares
                    out.push(format!("sh{a}, ash{a}, ast{a}, 0LL"));
                }
                (CallArg::Alloc(_), _) => return Err(refuse("an allocation passed to a callee")),
                _ => return Err(refuse("a call argument and its parameter differ in kind")),
            }
        }
        Ok(out)
    }

    fn nested(&mut self, b: &Block) -> Result<(), LowerError> {
        self.depth += 1;
        let r = self.block(b);
        self.depth -= 1;
        r
    }

    /// [`Self::nested`] for the body of a C loop inside a thread run, where `Break`/`Continue` are legal.
    fn nested_loop(&mut self, b: &Block) -> Result<(), LowerError> {
        self.thread_loops += 1;
        let r = self.nested(b);
        self.thread_loops -= 1;
        r
    }

    fn if_stmt(&mut self, cond: &ScalarExpr, then: &Block, els: &Block) -> Result<(), LowerError> {
        let c = self.expr(cond)?;
        self.line(&format!("if (({c}) != 0) {{"));
        self.nested(then)?;
        if els.0.is_empty() {
            self.line("}");
        } else {
            self.line("} else {");
            self.nested(els)?;
            self.line("}");
        }
        Ok(())
    }

    /// A counted loop. The counter is a 64-bit temporary the body never sees; the declared local
    /// is a fresh copy per pass, so a narrow counter type cannot overflow the loop itself.
    fn for_stmt(
        &mut self,
        local: u32,
        [start, end, step]: [&ScalarExpr; 3],
        body: &Block,
    ) -> Result<(), LowerError> {
        self.local_ctype(local)?;
        let n = self.fresh;
        self.fresh += 1;
        let (s, e, st) = (self.expr(start)?, self.expr(end)?, self.expr(step)?);
        self.line("{");
        self.depth += 1;
        self.line(&format!(
            "long long s{n} = (long long)({s}), e{n} = (long long)({e}), t{n} = (long long)({st});"
        ));
        self.line(&format!("if (t{n} == 0) return 2;"));
        self.line(&format!(
            "for (long long k{n} = s{n}; t{n} > 0 ? k{n} < e{n} : k{n} > e{n}; k{n} += t{n}) {{"
        ));
        self.depth += 1;
        self.bind_local(local, &format!("k{n}"))?;
        self.thread_loops += 1;
        let r = self.block(body);
        self.thread_loops -= 1;
        r?;
        self.depth -= 1;
        self.line("}");
        self.depth -= 1;
        self.line("}");
        Ok(())
    }

    /// An lvalue for `place`, after the statements that bounds-check its coordinates.
    pub(crate) fn address(&mut self, place: &Place) -> Result<String, LowerError> {
        let Place::Logical { buffer, coords } = place else {
            return self.alloc_address(place);
        };
        let n = self.fresh;
        self.fresh += 1;
        self.line(&format!("long long o{n} = b{buffer}_base;"));
        for (d, c) in coords.iter().enumerate() {
            let v = self.expr(c)?;
            self.line(&format!(
                "{{ long long c = (long long)({v}); if (c < 0 || c >= b{buffer}_shape[{d}]) return 1; o{n} += c * b{buffer}_stride[{d}]; }}"
            ));
        }
        Ok(format!("b{buffer}[o{n}]"))
    }
}

/// The signature, with one line per C argument.
fn signature(
    k: &KernelFn,
    name: &str,
    (ids, result): (bool, Option<&'static str>),
) -> Result<String, LowerError> {
    let mut args = Vec::new();
    for (i, p) in k.params.iter().enumerate() {
        match &p.kind {
            ParamKind::Buffer { elem, role, .. } => {
                let t = ctype(*elem, "a buffer")?;
                let q = if matches!(role, BufferRole::Write | BufferRole::ReadWrite) {
                    ""
                } else {
                    "const "
                };
                args.push(format!("{q}{t}* b{i}"));
                args.push(format!("const long long* b{i}_shape"));
                args.push(format!("const long long* b{i}_stride"));
                args.push(format!("long long b{i}_base"));
            }
            ParamKind::Scalar { ty, .. } => args.push(format!("{} p{i}", ctype(*ty, "a scalar")?)),
            _ => return Err(refuse("a parameter kind this emitter has not been taught")),
        }
    }
    if ids {
        for n in ["ti0", "ti1", "ti2", "bi0", "bi1", "bi2"] {
            args.push(format!("unsigned int {n}"));
        }
    }
    if let Some(t) = result {
        args.push(format!("{t}* res"));
    }
    Ok(format!("int {name}(\n    {})", args.join(",\n    ")))
}

/// The leaf seam: a kernel has no `Input` leaves (the checker rejects them).
fn no_input_leaf(i: u8) -> Result<Spelling, LowerError> {
    Ok(Spelling::Declined(Decline::UnsupportedOp {
        op: DeclinedOp::Kernel,
        why: format!("an Input({i}) leaf inside a kernel; the checker rejects it"),
    }))
}

/// A literal, cast to the compute type: a bare C literal is a `double`, and a `float` expression
/// containing one would be evaluated in double and rounded once at the end.
fn literal(dtype: ElementKind, v: f64) -> Result<Spelling, LowerError> {
    Ok(Spelling::Spelled(match dtype {
        ElementKind::F32 => format!("((float){})", const_lit(v)),
        _ => const_lit(v),
    }))
}

/// Infix arithmetic: a plain C operator on a float compute dtype, a wrapping integer form on an
/// integer one (C's signed overflow is undefined).
fn infix(
    dtype: ElementKind,
    op: unpopped::ir::ArithOp,
    a: String,
    b: String,
) -> Result<Spelling, LowerError> {
    if typed::is_int_compute(dtype) {
        return typed::cpu_int_arith(op, a, b, dtype);
    }
    Ok(Spelling::Spelled(format!("({a} {} {b})", op.c_operator())))
}

/// Lower `k` to a C function. See the module docs for the contract.
pub(crate) fn lower_kernel(k: &KernelFn) -> Result<GeneratedKernel, LowerError> {
    let name = format!("unpopped_cpu_kernel_{}", ident(&k.name));
    let text = function_text(k, (None, &name, false))?;
    Ok(GeneratedKernel::new(name, with_prelude(&k.name, &text)))
}

/// The file header and the shared helpers, then `body` (one or more C functions).
pub(crate) fn with_prelude(kernel_name: &str, body: &str) -> String {
    let mut src =
        String::from("// Generated by unpopped (cpu_c kernel emitter), serial C: do not edit.\n");
    let _ = writeln!(src, "// kernel: {kernel_name}\n#include <math.h>\n");
    src.push_str(typed::TYPED_HELPERS);
    src.push('\n');
    src.push_str(qualified::FTZ_HELPER);
    src.push('\n');
    src.push_str(body);
    src
}

/// The C function for `k` alone (signature and body), with the seams of its compute dtype. `module` is
/// the module its `Call`s resolve in; `callee` makes the function return its result through `res`.
pub(crate) fn function_text(
    k: &KernelFn,
    (module, name, callee): (Option<&ModuleCtx<'_>>, &str, bool),
) -> Result<String, LowerError> {
    admit(k, module.is_some())?;
    let dtype = k.compute;
    let tys: Vec<Ty> = k.locals.iter().map(|d| d.ty()).collect();
    let un = |op, x: String| cpu_unary(op, x, dtype);
    let bin = |op, a: String, b: String| {
        if typed::is_int_compute(dtype) {
            typed::cpu_int_binary(op, a, b, dtype)
        } else {
            cpu_binary(op, a, b, dtype)
        }
    };
    let ar = |op, a: String, b: String| infix(dtype, op, a, b);
    let sel = |c, a, b| cpu_select(c, a, b, dtype);
    let fm = |a, b, c| cpu_fma(a, b, c, dtype);
    let cs = |x: String, from, spec| typed::cpu_cast(x, from, spec);
    let bc = |x: String, from, to| typed::cpu_bitcast(x, from, to);
    let tn = |t| typed::c_type(t).map(str::to_string);
    let ql = |op, operands, n| qualified::cpu_qualified(op, operands, n, dtype);
    let cst = |v| literal(dtype, v);
    let ci = |v, t| typed::cpu_const_int(v, t);
    let io = |op, t, operands: Vec<String>| typed::cpu_int_op(op, t, &operands, dtype);
    let launch = LaunchInfo::of(k);
    let local = |i: u32| crate::launch::local_c(launch, i);
    let ids = module.and_then(|m| m.launch).filter(|_| callee);
    let exec = |id| crate::launch::exec_spelling((launch, ids), id);
    let lowering = Lowering::builder(&no_input_leaf, &un, &bin)
        .exec(&exec)
        .arith(&ar)
        .select(&sel)
        .fma(&fm)
        .cast(&cs)
        .bitcast(&bc)
        .qualified(&ql)
        .compute_dtype(dtype)
        .type_name(&tn)
        .constant(&cst)
        .const_int(&ci)
        .int_op(&io)
        .local(&local)
        .local_types(&tys)
        .build();
    emit(k, &lowering, (module, name, callee))
}

/// Spell the whole function with the seams `lo` carries.
fn emit(
    k: &KernelFn,
    lo: &Lowering<'_>,
    (module, name, callee): (Option<&ModuleCtx<'_>>, &str, bool),
) -> Result<String, LowerError> {
    let result = if callee { k.result } else { None };
    let mut e = Emitter {
        k,
        lo,
        out: String::new(),
        depth: 1,
        fresh: 0,
        launch: LaunchInfo::of(k),
        thread_loops: 0,
        module,
        result,
    };
    let res_type = match result {
        Some(r) => Some(e.local_ctype(r)?),
        None => None,
    };
    if let (Some(r), Some(t)) = (result, res_type) {
        e.line(&format!("{t} l{r} = 0;"));
    }
    match e.launch {
        Some(info) => e.launched_body(info)?,
        None => e.body()?,
    }
    let store = result.map_or(String::new(), |r| format!("    *res = l{r};\n"));
    Ok(format!(
        "{} {{\n{}{store}    return 0;\n}}\n",
        signature(
            k,
            name,
            (
                callee && module.is_some_and(|m| m.launch.is_some()),
                res_type
            )
        )?,
        e.out
    ))
}

impl Emitter<'_> {
    /// The scalar-parameter bindings, then the program.
    fn body(&mut self) -> Result<(), LowerError> {
        for (i, p) in self.k.params.iter().enumerate() {
            if let ParamKind::Scalar { ty, local } = &p.kind {
                let t = ctype(*ty, "a scalar")?;
                self.line(&format!("{t} l{local} = p{i};"));
            }
        }
        let body = &self.k.body;
        self.block(body)
    }
}
