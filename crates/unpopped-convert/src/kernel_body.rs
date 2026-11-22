// SPDX-License-Identifier: MIT OR Apache-2.0
//! The C-family **kernel-body frontend** (ROADMAP U19, `docs/options-u19-cfamily-body-frontend.md`), slice 1: the statement
//! spine. [`lift_kernel`] walks the CST of one function and builds a [`KernelFn`], or declines the **whole kernel** with a
//! typed [`KernelLiftError`]; it never returns a partial kernel and never guesses.
//!
//! Two declines, by who owns the gap: [`KernelLiftError::NotYetLifted`] (the IR can say it, the walker does not yet, so
//! it is a later slice) and [`KernelLiftError::OutsideIr`] (the IR's own type rules refuse it, so it is an IR backlog row).
//! The result is always run through [`unpopped::kernel::check_kernel`], so a walker bug surfaces as
//! [`KernelLiftError::Invalid`] rather than as a wrong kernel.
//!
//! **Slice 1 covers:** scalar and `T*` / `const T*` parameters; `T x = e;`; assignment and compound assignment to locals and
//! to buffer elements; `i++`/`i--` as statements; `if`/`else`; `while`; counted `for` loops; `break`, `continue`, bare
//! `return`; arithmetic, comparisons, `&&`, `||`, `!`, `?:`, casts and `a[i]` reads (hoisted into `Load` statements).
//! **Not yet:** calls and intrinsics, thread builtins, `__shared__`, atomics, structs, vector types, templates, pointer
//! arithmetic, multi-dimensional indexing. Each is a typed decline naming the construct.
//!
//! Types follow C's usual arithmetic conversions, made explicit with `Cast`s: integer types narrower than `int` promote to
//! `int`, a mixed pair converts to the wider (the unsigned one at equal width), an integer meeting a float converts to the
//! float. A comparison has the IR's type (the kernel's compute type, 0 or 1), not C's `int`. The kernel's **compute type** is
//! the first floating-point parameter type, else the first buffer's element type, else `I32`; a floating-point literal must
//! have that type.

use crate::Frontend;
use std::collections::{HashMap, HashSet};
use tree_sitter::Node;
use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr, UnaryOp};
use unpopped::kernel::{
    AtomicOp, BarrierKind, Block, BufferRole, KernelError, KernelFn, LocalDecl, MemoryOrder, Param,
    Place, Stmt, Unroll, check_kernel,
};
use unpopped::types::{Ty, infer_ty_in, int_range};
use unpopped_vocab::ElementKind as E;

/// Why [`lift_kernel`] declined a kernel. Every variant declines the whole kernel.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum KernelLiftError {
    /// The frontend's kernel marker is absent, or the source did not parse.
    NotAKernel,
    /// No function of that name is defined in the source.
    NoSuchFunction(String),
    /// The CST holds an error node: the source is not valid in the frontend's grammar.
    SyntaxError {
        /// The 1-based source line.
        line: usize,
    },
    /// The IR can express this; the walker does not lift it yet (a later U19 slice).
    NotYetLifted {
        /// The construct, e.g. `"a call"`.
        construct: String,
        /// The 1-based source line.
        line: usize,
    },
    /// The IR's own rules refuse this (an IR backlog row, not a walker slice).
    OutsideIr {
        /// What was refused, with the IR's reason.
        construct: String,
        /// The 1-based source line.
        line: usize,
    },
    /// The lifted kernel failed [`check_kernel`]: a walker bug or a source error the walker did not see.
    Invalid(KernelError),
}

impl std::fmt::Display for KernelLiftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAKernel => f.write_str("not a kernel in this frontend"),
            Self::NoSuchFunction(n) => write!(f, "no function `{n}`"),
            Self::SyntaxError { line } => write!(f, "syntax error at line {line}"),
            Self::NotYetLifted { construct, line } => {
                write!(f, "line {line}: {construct} is not lifted yet")
            }
            Self::OutsideIr { construct, line } => {
                write!(f, "line {line}: {construct} is outside the IR")
            }
            Self::Invalid(e) => write!(f, "the lifted kernel is invalid: {e:?}"),
        }
    }
}

impl std::error::Error for KernelLiftError {}

mod asm;
mod context;
pub use context::LiftContext;
mod calls;
mod cpasync;
mod declarations;
mod expressions;
mod globals;
mod inline;
mod methods;
mod places;
mod pointer;
mod returns;
mod roles;
mod statements;
mod structs;
mod templates;
mod unrolled;
mod vecmem;

type R<T> = Result<T, KernelLiftError>;

/// Lift the function `name` of `src` into a [`KernelFn`] (ROADMAP U19 slice 1).
///
/// # Errors
/// [`KernelLiftError`]: the kernel is declined whole, naming the first construct that stopped it.
pub fn lift_kernel(fe: &Frontend, src: &str, name: &str) -> Result<KernelFn, KernelLiftError> {
    lift_in(fe, src, name, &LiftContext::without_launch())
}

/// [`lift_kernel`] under a full [`LiftContext`]: the launch shape, the dynamic shared-memory size and the macro definitions the
/// source's `#if` and `#ifdef` read (ROADMAP U19 slice 5).
///
/// # Errors
/// As [`lift_kernel`].
pub fn lift_kernel_in(
    fe: &Frontend,
    src: &str,
    name: &str,
    ctx: &LiftContext,
) -> Result<KernelFn, KernelLiftError> {
    lift_in(fe, src, name, ctx)
}

/// [`lift_kernel`] for a kernel that runs under `launch` (ROADMAP U19 slice 2). The launch shape is the **caller's**: a CUDA
/// source does not carry its block and grid (the call site does), and `__launch_bounds__` gives only a limit. With it the walker
/// lifts the thread builtins (`threadIdx.x`, `blockIdx`, `blockDim`, `gridDim`), `__syncthreads()` and static
/// `__shared__ T name[N];` arrays, none of which has a meaning without one.
///
/// # Errors
/// As [`lift_kernel`].
pub fn lift_kernel_launched(
    fe: &Frontend,
    src: &str,
    name: &str,
    launch: LaunchShape,
) -> Result<KernelFn, KernelLiftError> {
    lift_in(fe, src, name, &LiftContext::new(launch))
}

fn lift_in(
    fe: &Frontend,
    src: &str,
    name: &str,
    ctx: &LiftContext,
) -> Result<KernelFn, KernelLiftError> {
    let (pre, unexpanded) = context::preprocess(src, &ctx.defines)
        .map_err(|construct| KernelLiftError::NotYetLifted { construct, line: 0 })?;
    let src = pre.as_str();
    // Not `prepared`: that applies the frontend's RESIDUE list (`__shared__`, `atomicAdd`, `__syncthreads`, `asm`, ...), the legacy
    // elementwise lifters' "not expressible" gate, which maps a hit to `NotAKernel`. The kernel-body walker lifts those and declines
    // each construct it cannot by name, so only the marker is checked. Found by running the census over Baracuda's real CUDA
    // frontend (2026-11-22): every kernel containing one of those words was refused as `NotAKernel`.
    if !src.contains(fe.kernel_marker) {
        return Err(KernelLiftError::NotAKernel);
    }
    let mut tree = (fe.parse)(src).ok_or(KernelLiftError::SyntaxError { line: 0 })?;
    let first = find_function(tree.root_node(), src, name)
        .ok_or_else(|| KernelLiftError::NoSuchFunction(name.to_string()))?;
    // a template kernel is instantiated from the caller's arguments and re-parsed (census slice 1)
    let instantiated = templates::instantiate(src, first, name, ctx)?;
    let src = match &instantiated {
        Some(s) => {
            tree = (fe.parse)(s).ok_or(KernelLiftError::SyntaxError { line: 0 })?;
            s.as_str()
        }
        None => src,
    };
    // template helpers called with explicit arguments become plain functions (census slice 2)
    let expanded = templates::expand_calls(fe.parse, src.to_string())?;
    let src = expanded.as_str();
    tree = (fe.parse)(src).ok_or(KernelLiftError::SyntaxError { line: 0 })?;
    let root = tree.root_node();
    let func = find_function(root, src, name)
        .ok_or_else(|| KernelLiftError::NoSuchFunction(name.to_string()))?;
    // only an error INSIDE the kernel stops it: a grammar error in another function of the file is not this kernel's
    if let Some(bad) = first_error(func) {
        return Err(KernelLiftError::SyntaxError {
            line: bad.start_position().row + 1,
        });
    }
    let mut w = Walker::new(src);
    w.functions = all_functions(root, src);
    w.structs = structs::collect_structs(root, src);
    w.methods = methods::collect_methods(root, src);
    w.launch = ctx.launch;
    w.dyn_shared = ctx.dynamic_shared_bytes;
    w.null_buffers = ctx.null_buffers.iter().cloned().collect();
    w.consts = collect_consts(root, src);
    w.unexpanded = unexpanded;
    w.unary_fn = fe.unary_fn;
    w.binary_fn = fe.binary_fn;
    let k = w.function(func, name)?;
    check_kernel(&k).map_err(KernelLiftError::Invalid)?;
    Ok(k)
}

/// Every function defined in the source, by name.
fn all_functions<'t>(root: Node<'t>, src: &str) -> HashMap<String, Node<'t>> {
    let mut out = HashMap::new();
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if n.kind() == "function_definition" {
            let id = n
                .child_by_field_name("declarator")
                .and_then(|d| d.child_by_field_name("declarator"));
            if let Some(i) = id.filter(|i| i.kind() == "identifier") {
                out.insert(text(i, src).to_string(), n);
            }
        }
        let mut c = n.walk();
        stack.extend(n.children(&mut c));
    }
    out
}

/// The file-scope `constexpr` and `const` constants (namespace members included) with an initialiser, by name.
fn collect_consts<'t>(root: Node<'t>, src: &str) -> HashMap<String, (String, Node<'t>)> {
    let mut out = HashMap::new();
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if n.kind() == "declaration" && at_file_scope(n) {
            let quals: Vec<&str> = {
                let mut c = n.walk();
                n.children(&mut c)
                    .filter(|k| k.kind() == "type_qualifier")
                    .map(|k| text(k, src))
                    .collect()
            };
            let ty = n.child_by_field_name("type").map(|t| {
                text(t, src)
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            });
            let is_const = quals.iter().any(|q| matches!(*q, "constexpr" | "const"));
            if let (true, Some(ty)) = (is_const, ty) {
                let mut c = n.walk();
                for d in n.children_by_field_name("declarator", &mut c) {
                    let id = d
                        .child_by_field_name("declarator")
                        .filter(|i| i.kind() == "identifier");
                    if let (Some(id), Some(v)) = (id, d.child_by_field_name("value")) {
                        out.insert(text(id, src).to_string(), (ty.clone(), v));
                    }
                }
            }
        }
        if !matches!(n.kind(), "function_definition" | "compound_statement") {
            let mut c = n.walk();
            stack.extend(n.children(&mut c));
        }
    }
    out
}

/// Whether `n` sits directly in the translation unit or a namespace body (not in a function).
fn at_file_scope(n: Node<'_>) -> bool {
    n.parent()
        .is_some_and(|p| matches!(p.kind(), "translation_unit" | "declaration_list"))
}

fn first_error(n: Node<'_>) -> Option<Node<'_>> {
    if n.is_error() || n.is_missing() {
        return Some(n);
    }
    let mut c = n.walk();
    n.children(&mut c).find_map(first_error)
}

fn find_function<'t>(root: Node<'t>, src: &str, name: &str) -> Option<Node<'t>> {
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if n.kind() == "function_definition" {
            let id = n
                .child_by_field_name("declarator")
                .and_then(|d| d.child_by_field_name("declarator"));
            if id.is_some_and(|i| text(i, src) == name) {
                return Some(n);
            }
        }
        let mut c = n.walk();
        stack.extend(n.children(&mut c));
    }
    None
}

fn text<'a>(n: Node<'_>, src: &'a str) -> &'a str {
    &src[n.byte_range()]
}

/// A name in scope.
#[derive(Clone)]
enum Sym {
    /// A struct instance (slice 4d): its fields scalarized, each a scalar local, a private array or a nested instance.
    Struct(std::rc::Rc<structs::StructInst>),
    /// A pointer local or parameter (slice 4c-4): a base, a pointee type and a byte offset, resolved in the walker.
    Pointer(pointer::PtrVal),
    /// A buffer parameter: its index in `KernelFn::params`, element type and whether the source made it `const`.
    Buffer(u32, E),
    /// A local of a CUDA vector type (`float4`, `int2`, ...), **scalarized**: one scalar local per lane (the first `lanes` of
    /// the array), the lane type and the lane count. Not an IR packed value: the oracle cannot run a packed local
    /// (`tests/kernel_packed.rs`, the pinned gap), and a scalarized vector is exact.
    Vector([u32; 4], E, u8),
    /// A vector local whose address is taken (`&tmp`): backed by a `Private` allocation of `lanes` elements so a pointer can
    /// reinterpret it (`(half2*)&tmp`); the allocation's index, lane type and lane count.
    VecAlloc(u32, E, u8),
    /// A static `__shared__` array: its index in `KernelFn::allocs` and element type.
    Alloc(u32, E),
    /// A scalar local (a scalar parameter is one): its id, type and whether the IR forbids assigning it.
    Scalar(u32, E, Fixed),
}

/// Why a scalar cannot be assigned.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fixed {
    /// An ordinary local.
    No,
    /// A scalar parameter (`ParamAssigned` in the checker).
    Param,
    /// A `For` counter (`CounterAssigned`).
    Counter,
}

struct Walker<'s, 't> {
    /// Every function defined in the source, by name (the kernel included), for inlining (slice 4c-2).
    functions: HashMap<String, Node<'t>>,
    /// Every struct defined in the source (slice 4d).
    structs: HashMap<String, Vec<structs::FieldDef>>,
    /// The methods of those structs, by struct then method name (slice 4e).
    methods: HashMap<String, HashMap<String, Node<'t>>>,
    /// The functions being inlined now, outermost first.
    inline_stack: Vec<String>,
    src: &'s str,
    params: Vec<Param>,
    locals: Vec<LocalDecl>,
    scopes: Vec<HashMap<String, Sym>>,
    compute: E,
    /// Names assigned anywhere in the function: a local not among them is immutable, so a `For` bound may read it.
    assigned: HashSet<String>,
    /// `Load`s the current expression needs, in evaluation order; the statement emits them first.
    hoisted: Vec<Stmt>,
    /// The caller's launch shape, if any (slice 2).
    launch: Option<LaunchShape>,
    /// Bytes of dynamic shared memory the caller gave (`extern __shared__ T a[];`).
    dyn_shared: u32,
    /// Buffer parameters the caller declared null.
    null_buffers: HashSet<String>,
    /// Macros the preprocessor could not expand, with why: a use declines naming the macro (`Walker::unexpanded_or`).
    unexpanded: context::Unexpanded,
    /// File-scope `constexpr`/`const` constants by name: the declared type text and the initialiser.
    consts: HashMap<String, (String, Node<'t>)>,
    allocs: Vec<AllocDecl>,
    /// A `#pragma unroll` read and not yet given to a loop (it applies to the very next statement only).
    pending_unroll: Unroll,
    /// The frontend's math-call tables (`Frontend::unary_fn`, `binary_fn`).
    unary_fn: fn(&str) -> Option<UnaryOp>,
    binary_fn: fn(&str) -> Option<BinaryOp>,
}

impl<'s, 't> Walker<'s, 't> {
    fn new(src: &'s str) -> Self {
        Self {
            functions: HashMap::new(),
            structs: HashMap::new(),
            methods: HashMap::new(),
            inline_stack: Vec::new(),
            src,
            params: Vec::new(),
            locals: Vec::new(),
            scopes: vec![HashMap::new()],
            compute: E::I32,
            assigned: HashSet::new(),
            hoisted: Vec::new(),
            launch: None,
            dyn_shared: 0,
            null_buffers: HashSet::new(),
            unexpanded: Vec::new(),
            consts: HashMap::new(),
            allocs: Vec::new(),
            pending_unroll: Unroll::None,
            unary_fn: |_| None,
            binary_fn: |_| None,
        }
    }

    fn line(n: Node<'_>) -> usize {
        n.start_position().row + 1
    }

    fn not_yet<T>(&self, n: Node<'_>, what: impl Into<String>) -> R<T> {
        Err(KernelLiftError::NotYetLifted {
            construct: what.into(),
            line: Self::line(n),
        })
    }

    fn outside<T>(&self, n: Node<'_>, what: impl Into<String>) -> R<T> {
        Err(KernelLiftError::OutsideIr {
            construct: what.into(),
            line: Self::line(n),
        })
    }

    fn function(&mut self, f: Node<'_>, name: &str) -> R<KernelFn> {
        let decl = f.child_by_field_name("declarator");
        let plist = decl.and_then(|d| d.child_by_field_name("parameters"));
        let (Some(plist), Some(body)) = (plist, f.child_by_field_name("body")) else {
            return self.not_yet(f, "a function without a parameter list or body");
        };
        collect_assigned(f.parent().unwrap_or(f), self.src, &mut self.assigned);
        let mut raw = Vec::new();
        let mut c = plist.walk();
        for p in plist.named_children(&mut c) {
            // a comment between parameters is not one (the census over Baracuda's kernels: 94 kernels stopped on it)
            if p.kind() == "comment" {
                continue;
            }
            if p.kind() != "parameter_declaration" {
                return self.not_yet(p, format!("a `{}` parameter", p.kind()));
            }
            raw.push(self.parameter(p)?);
        }
        self.compute = pick_compute(&raw);
        for (i, (pname, shape)) in raw.into_iter().enumerate() {
            self.bind_param(i, pname, shape, f)?;
        }
        let block = self.block_of(body)?;
        let (params, locals, compute) = (
            std::mem::take(&mut self.params),
            std::mem::take(&mut self.locals),
            self.compute,
        );
        let mut k = KernelFn::new(name, compute, params, locals, block);
        roles::refine_roles(&mut k);
        if let Some(l) = self.launch.take() {
            k = k.with_launch(l);
        }
        Ok(k.with_allocs(std::mem::take(&mut self.allocs)))
    }

    fn parameter(&self, p: Node<'_>) -> R<(String, PShape)> {
        if let Some(r) = self.struct_parameter(p) {
            return r;
        }
        // `void*` is an untyped byte buffer (census slice 3): its element is a byte, and the kernel reads it through a cast
        let is_void = p
            .child_by_field_name("type")
            .is_some_and(|t| text(t, self.src).trim() == "void");
        let ty = if is_void {
            E::U8
        } else {
            self.type_of(p.child_by_field_name("type"), p)?
        };
        let mut d = p.child_by_field_name("declarator");
        let mut ptr = false;
        while let Some(x) = d {
            match x.kind() {
                "pointer_declarator" => {
                    if ptr {
                        return self.not_yet(x, "a pointer to a pointer");
                    }
                    ptr = true;
                    d = x.child_by_field_name("declarator");
                }
                "identifier" => {
                    let is_const = has_const(p, self.src);
                    if is_void && !ptr {
                        return self.not_yet(x, "a parameter of type `void`");
                    }
                    let shape = if ptr {
                        PShape::Buffer(ty, is_const)
                    } else {
                        PShape::Scalar(ty)
                    };
                    return Ok((text(x, self.src).to_string(), shape));
                }
                other => return self.not_yet(x, format!("a `{other}` parameter declarator")),
            }
        }
        self.not_yet(p, "an unnamed parameter")
    }

    fn bind_param(&mut self, index: usize, name: String, shape: PShape, at: Node<'_>) -> R<()> {
        let sym = match shape {
            PShape::Buffer(elem, is_const) => {
                let role = if is_const {
                    BufferRole::Read
                } else {
                    BufferRole::ReadWrite
                };
                self.params
                    .push(Param::buffer(name.clone(), elem, role, Some(1)));
                Sym::Buffer(u32::try_from(index).unwrap_or(u32::MAX), elem)
            }
            PShape::Scalar(ty) => {
                let local = self.new_local(ty);
                self.params.push(Param::scalar(name.clone(), ty, local));
                Sym::Scalar(local, ty, Fixed::Param)
            }
            PShape::Struct(sname) => self.bind_struct_param(&name, &sname, at)?,
        };
        self.scopes[0].insert(name, sym);
        Ok(())
    }

    /// A parameter of struct type passed by value (`const S p`), or `None` if the parameter is not one.
    fn struct_parameter(&self, p: Node<'_>) -> Option<R<(String, PShape)>> {
        let ty = p
            .child_by_field_name("type")
            .filter(|t| t.kind() == "type_identifier")?;
        let sname = text(ty, self.src);
        if !self.is_struct(sname) {
            return None;
        }
        let d = p.child_by_field_name("declarator")?;
        if d.kind() != "identifier" {
            return Some(self.not_yet(
                d,
                format!("a `{}` parameter of struct type (only by value)", d.kind()),
            ));
        }
        Some(Ok((
            text(d, self.src).to_string(),
            PShape::Struct(sname.to_string()),
        )))
    }

    /// A defect in the walker's own bookkeeping, reported as a decline rather than a panic.
    pub(super) fn invalid(&self, what: &str) -> KernelLiftError {
        KernelLiftError::NotYetLifted {
            construct: what.to_string(),
            line: 0,
        }
    }

    fn new_local(&mut self, ty: E) -> u32 {
        self.locals.push(LocalDecl::scalar(ty));
        u32::try_from(self.locals.len() - 1).unwrap_or(u32::MAX)
    }

    fn lookup(&self, name: &str) -> Option<Sym> {
        self.scopes.iter().rev().find_map(|s| s.get(name).cloned())
    }

    fn type_of(&self, ty: Option<Node<'_>>, at: Node<'_>) -> R<E> {
        let Some(ty) = ty else {
            return self.not_yet(at, "a declaration without a type");
        };
        let t = text(ty, self.src)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        match c_type(&t) {
            Ok(e) => Ok(e),
            Err(why) => self.not_yet(ty, why),
        }
    }

    // ---- statements ----
}

/// The lane type and count of a CUDA vector type name (`float4`, `int2`, ...).
fn vector_type(t: &str) -> Option<(E, u8)> {
    let (base, n) = t.split_at(t.len().checked_sub(1)?);
    let lanes: u8 = n.parse().ok().filter(|k| matches!(k, 2 | 4))?;
    let elem = match base {
        "float" => E::F32,
        "double" => E::F64,
        "int" => E::I32,
        "uint" => E::U32,
        "half" => E::F16,
        _ => return None,
    };
    Some((elem, lanes))
}

/// The lane a member name selects in a `lanes`-lane vector (`x`, `y`, then `z`, `w` for four).
fn lane_of(member: &str, lanes: u8) -> Option<u8> {
    let i = match member {
        "x" => 0,
        "y" => 1,
        "z" => 2,
        "w" => 3,
        _ => return None,
    };
    (i < lanes).then_some(i)
}

/// Whether a declaration declares a pointer (`T* p = ...`).
fn declares_pointer(n: Node<'_>) -> bool {
    let mut c = n.walk();
    n.children_by_field_name("declarator", &mut c).any(|d| {
        let inner = if d.kind() == "init_declarator" {
            d.child_by_field_name("declarator")
        } else {
            Some(d)
        };
        inner.is_some_and(|i| i.kind() == "pointer_declarator")
    })
}

/// Whether a declaration declares an array (`T a[N]`, with or without an initializer).
fn declares_array(n: Node<'_>) -> bool {
    let mut c = n.walk();
    n.children_by_field_name("declarator", &mut c).any(|d| {
        let inner = if d.kind() == "init_declarator" {
            d.child_by_field_name("declarator")
        } else {
            Some(d)
        };
        inner.is_some_and(|i| i.kind() == "array_declarator")
    })
}

fn push(out: &mut Vec<Stmt>, s: Stmt) -> R<()> {
    out.push(s);
    Ok(())
}

enum PShape {
    Buffer(E, bool),
    Scalar(E),
    /// A struct passed by value (its type name).
    Struct(String),
}

fn pick_compute(raw: &[(String, PShape)]) -> E {
    let types = raw.iter().filter_map(|(_, s)| match s {
        PShape::Buffer(t, _) | PShape::Scalar(t) => Some(*t),
        PShape::Struct(_) => None,
    });
    // a compute float is `F32` or `F64`; a narrow float (`half`) is an element type, never the compute type
    let mut types = types.collect::<Vec<_>>();
    if let Some(f) = types.iter().find(|t| matches!(t, E::F32 | E::F64)) {
        return *f;
    }
    let first_buffer = raw.iter().find_map(|(_, s)| match s {
        PShape::Buffer(t, _) => Some(*t),
        PShape::Scalar(_) | PShape::Struct(_) => None,
    });
    types.clear();
    match first_buffer {
        Some(E::F16 | E::Bf16) => E::F32,
        Some(t) => t,
        None => E::I32,
    }
}

fn has_const(p: Node<'_>, src: &str) -> bool {
    let mut c = p.walk();
    p.children(&mut c)
        .any(|k| k.kind() == "type_qualifier" && text(k, src) == "const")
}

/// Every name that is the target of an assignment or an increment in `n`.
fn collect_assigned(n: Node<'_>, src: &str, out: &mut HashSet<String>) {
    let target = match n.kind() {
        "assignment_expression" => n.child_by_field_name("left"),
        "update_expression" => n.child_by_field_name("argument"),
        _ => None,
    };
    if let Some(t) = target.filter(|t| t.kind() == "identifier") {
        out.insert(text(t, src).to_string());
    }
    let mut c = n.walk();
    for k in n.named_children(&mut c) {
        collect_assigned(k, src, out);
    }
}

/// The element type of a C type name, or why there is none.
fn c_type(t: &str) -> Result<E, String> {
    Ok(match t {
        "float" => E::F32,
        "double" => E::F64,
        "int" | "signed" | "signed int" | "int32_t" => E::I32,
        "unsigned" | "unsigned int" | "uint32_t" => E::U32,
        "short" | "signed short" | "int16_t" => E::I16,
        "unsigned short" | "uint16_t" => E::U16,
        "signed char" | "int8_t" => E::I8,
        "unsigned char" | "uint8_t" => E::U8,
        "long long" | "long long int" | "int64_t" => E::I64,
        "unsigned long long" | "uint64_t" | "size_t" => E::U64,
        // a C `bool` is a byte holding 0 or 1 (an initialiser is normalised: any non-zero value is 1)
        "bool" => E::U8,
        "half" | "__half" => E::F16,
        "__nv_bfloat16" | "nv_bfloat16" => E::Bf16,
        "char" => return Err("`char` (its signedness is implementation-defined)".into()),
        "long" | "unsigned long" => {
            return Err(format!(
                "`{t}` (its width differs by ABI; spell it `long long` or `int64_t`)"
            ));
        }
        other => return Err(format!("the type `{other}`")),
    })
}

enum Lit {
    Int(i64, E),
    Float(f64, E),
}

fn parse_literal(s: &str) -> Option<Lit> {
    let lower = s.to_ascii_lowercase();
    if lower.starts_with("0x") || lower.starts_with("-0x") {
        return None;
    }
    if lower.contains('.') || lower.contains('e') {
        let (digits, ty) = match lower.strip_suffix('f') {
            Some(d) => (d, E::F32),
            None => (lower.as_str(), E::F64),
        };
        let v: f64 = digits.parse().ok()?;
        // an `f` literal is the nearest f32: keep exactly that value
        return Some(Lit::Float(
            if ty == E::F32 { f64::from(v as f32) } else { v },
            ty,
        ));
    }
    let digits = lower.trim_end_matches(['u', 'l']);
    let suffix = &lower[digits.len()..];
    let v: i64 = digits.parse().ok()?;
    let ty = match suffix {
        "" if i32::try_from(v).is_ok() => E::I32,
        "" => E::I64,
        "u" if u32::try_from(v).is_ok() => E::U32,
        "u" => E::U64,
        "ll" | "l" => E::I64,
        "ull" | "ul" | "lu" | "llu" => E::U64,
        _ => return None,
    };
    Some(Lit::Int(v, ty))
}

/// C's integer promotion: anything narrower than `int` becomes `int`.
fn promote(t: E) -> E {
    match t {
        E::I8 | E::I16 | E::U8 | E::U16 | E::Bool => E::I32,
        other => other,
    }
}

fn int_rank(t: E) -> u32 {
    match t {
        E::I32 | E::U32 => 32,
        E::I64 | E::U64 => 64,
        _ => 0,
    }
}

fn is_unsigned(t: E) -> bool {
    matches!(t, E::U8 | E::U16 | E::U32 | E::U64)
}
