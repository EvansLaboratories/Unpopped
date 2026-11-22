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

use crate::{Frontend, prepared};
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
mod inline;
mod pointer;

type R<T> = Result<T, KernelLiftError>;

/// Lift the function `name` of `src` into a [`KernelFn`] (ROADMAP U19 slice 1).
///
/// # Errors
/// [`KernelLiftError`]: the kernel is declined whole, naming the first construct that stopped it.
pub fn lift_kernel(fe: &Frontend, src: &str, name: &str) -> Result<KernelFn, KernelLiftError> {
    lift_with_launch(fe, src, name, None)
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
    lift_with_launch(fe, src, name, Some(launch))
}

fn lift_with_launch(
    fe: &Frontend,
    src: &str,
    name: &str,
    launch: Option<LaunchShape>,
) -> Result<KernelFn, KernelLiftError> {
    let tree = prepared(fe, src).map_err(|_| KernelLiftError::NotAKernel)?;
    let root = tree.root_node();
    if let Some(bad) = first_error(root) {
        return Err(KernelLiftError::SyntaxError {
            line: bad.start_position().row + 1,
        });
    }
    let func = find_function(root, src, name)
        .ok_or_else(|| KernelLiftError::NoSuchFunction(name.to_string()))?;
    let mut w = Walker::new(src);
    w.functions = all_functions(root, src);
    w.launch = launch;
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
    /// A pointer local or parameter (slice 4c-4): a base, a pointee type and a byte offset, resolved in the walker.
    Pointer(pointer::PtrVal),
    /// A buffer parameter: its index in `KernelFn::params`, element type and whether the source made it `const`.
    Buffer(u32, E),
    /// A local of a CUDA vector type (`float4`, `int2`, ...), **scalarized**: one scalar local per lane (the first `lanes` of
    /// the array), the lane type and the lane count. Not an IR packed value: the oracle cannot run a packed local
    /// (`tests/kernel_packed.rs`, the pinned gap), and a scalarized vector is exact.
    Vector([u32; 4], E, u8),
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
            inline_stack: Vec::new(),
            src,
            params: Vec::new(),
            locals: Vec::new(),
            scopes: vec![HashMap::new()],
            compute: E::I32,
            assigned: HashSet::new(),
            hoisted: Vec::new(),
            launch: None,
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
            if p.kind() != "parameter_declaration" {
                return self.not_yet(p, format!("a `{}` parameter", p.kind()));
            }
            raw.push(self.parameter(p)?);
        }
        self.compute = pick_compute(&raw);
        for (i, (pname, shape)) in raw.into_iter().enumerate() {
            self.bind_param(i, pname, shape);
        }
        let block = self.block_of(body)?;
        let (params, locals, compute) = (
            std::mem::take(&mut self.params),
            std::mem::take(&mut self.locals),
            self.compute,
        );
        let mut k = KernelFn::new(name, compute, params, locals, block);
        if let Some(l) = self.launch.take() {
            k = k.with_launch(l);
        }
        Ok(k.with_allocs(std::mem::take(&mut self.allocs)))
    }

    fn parameter(&self, p: Node<'_>) -> R<(String, PShape)> {
        let ty = self.type_of(p.child_by_field_name("type"), p)?;
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

    fn bind_param(&mut self, index: usize, name: String, shape: PShape) {
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
        };
        self.scopes[0].insert(name, sym);
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

    fn block_of(&mut self, n: Node<'_>) -> R<Block> {
        self.scopes.push(HashMap::new());
        let mut out = Vec::new();
        let r = if n.kind() == "compound_statement" {
            let mut c = n.walk();
            let kids: Vec<_> = n.named_children(&mut c).collect();
            kids.into_iter().try_for_each(|k| self.stmt(k, &mut out))
        } else {
            self.stmt(n, &mut out)
        };
        self.scopes.pop();
        r.map(|()| Block(out))
    }

    fn stmt(&mut self, n: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        if n.kind() == "preproc_call" {
            return self.pragma(n);
        }
        // a hint binds the statement right after it: any other statement drops it, a `for` takes it
        let unroll = std::mem::take(&mut self.pending_unroll);
        match n.kind() {
            "compound_statement" => {
                let b = self.block_of(n)?;
                // a nested `{ }` is a scope, not a statement: splice its statements (the IR scopes `Let` by block, so
                // keep them in a one-armed `If` would change nothing, and splicing is exact).
                out.extend(b.0);
                Ok(())
            }
            "declaration" => self.declaration(n, out),
            "expression_statement" => self.expression_statement(n, out),
            "if_statement" => self.if_statement(n, out),
            "while_statement" => self.while_statement(n, out),
            "for_statement" => self.for_statement(n, out, unroll),
            "break_statement" => push(out, Stmt::Break),
            "continue_statement" => push(out, Stmt::Continue),
            "return_statement" => {
                if n.named_child_count() > 0 {
                    return self.not_yet(n, "a `return` with a value");
                }
                push(out, Stmt::Return)
            }
            "comment" | ";" => Ok(()),
            other => self.not_yet(n, format!("a `{other}` statement")),
        }
    }

    /// The zero of `ty`, as an expression.
    fn zero_of(&self, ty: E, at: Node<'_>) -> R<ScalarExpr> {
        if int_range(ty).is_some() {
            return Ok(ScalarExpr::ConstInt(0, ty));
        }
        self.convert(ScalarExpr::Const(0.0), self.compute, ty, at)
    }

    fn flush(&mut self, out: &mut Vec<Stmt>) {
        out.append(&mut self.hoisted);
    }

    fn declaration(&mut self, n: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        let tname = n.child_by_field_name("type").map(|t| {
            text(t, self.src)
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        });
        if let Some((elem, lanes)) = tname.as_deref().and_then(vector_type) {
            return self.vector_declaration(n, (elem, lanes), out);
        }
        let ty = self.type_of(n.child_by_field_name("type"), n)?;
        let head = text(n, self.src).trim_start();
        if declares_pointer(n) {
            return self.pointer_declaration(n, ty, out);
        }
        if head.starts_with("__shared__") {
            return self.array_declaration(n, ty, MemorySpace::Shared);
        }
        let has_array = declares_array(n);
        if has_array && !head.starts_with("extern") {
            return self.array_declaration(n, ty, MemorySpace::Private);
        }
        if head.starts_with("extern")
            || head.starts_with("__constant__")
            || head.starts_with("__device__")
        {
            return self.not_yet(
                n,
                "a declaration with a storage class other than `__shared__`",
            );
        }
        let mut c = n.walk();
        let decls: Vec<_> = n.children_by_field_name("declarator", &mut c).collect();
        for d in decls {
            let (id, value) = match d.kind() {
                "init_declarator" => (
                    d.child_by_field_name("declarator"),
                    d.child_by_field_name("value"),
                ),
                _ => (Some(d), None),
            };
            let Some(id) = id else {
                return self.not_yet(d, "a declarator of an unexpected shape");
            };
            if id.kind() != "identifier" {
                return self.not_yet(id, format!("a `{}` declarator", id.kind()));
            }
            // `T x;` is indeterminate in C until it is written; the IR needs a value, so it starts at zero (a refinement: a
            // read before the write is undefined in C)
            let e = match value {
                Some(v) => {
                    let (e, et) = self.expr(v)?;
                    self.convert(e, et, ty, v)?
                }
                None => self.zero_of(ty, d)?,
            };
            self.flush(out);
            let local = self.new_local(ty);
            self.scopes.last_mut().expect("a scope is open").insert(
                text(id, self.src).to_string(),
                Sym::Scalar(local, ty, Fixed::No),
            );
            out.push(Stmt::Let { local, init: e });
        }
        Ok(())
    }

    /// `float4 v = make_float4(a, b, c, d);` or `float4 v = w;`: one scalar local per lane.
    fn vector_declaration(
        &mut self,
        n: Node<'_>,
        (elem, lanes): (E, u8),
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        let mut c = n.walk();
        let decls: Vec<_> = n.children_by_field_name("declarator", &mut c).collect();
        for d in decls {
            let (id, value) = match d.kind() {
                "init_declarator" => (
                    d.child_by_field_name("declarator"),
                    d.child_by_field_name("value"),
                ),
                _ => (Some(d), None),
            };
            let (Some(id), Some(value)) = (id, value) else {
                return self.not_yet(d, "a vector local without an initializer");
            };
            if id.kind() != "identifier" {
                return self.not_yet(id, format!("a `{}` declarator", id.kind()));
            }
            let init = self.vector_value(value, (elem, lanes))?;
            self.flush(out);
            let mut ls = [0u32; 4];
            for (k, v) in init.into_iter().enumerate() {
                ls[k] = self.new_local(elem);
                out.push(Stmt::Let {
                    local: ls[k],
                    init: v,
                });
            }
            self.scopes
                .last_mut()
                .expect("a scope is open")
                .insert(text(id, self.src).to_string(), Sym::Vector(ls, elem, lanes));
        }
        Ok(())
    }

    /// A vector-typed value as its lane expressions: `make_<type>(lanes...)` or another vector local of the same type.
    fn vector_value(&mut self, n: Node<'_>, (elem, lanes): (E, u8)) -> R<Vec<ScalarExpr>> {
        if n.kind() == "identifier" {
            return match self.lookup(text(n, self.src)) {
                Some(Sym::Vector(ls, e, k)) if (e, k) == (elem, lanes) => Ok(ls[..usize::from(k)]
                    .iter()
                    .map(|l| ScalarExpr::Local(*l))
                    .collect()),
                _ => self.not_yet(
                    n,
                    "a vector value that is not a vector local of the same type",
                ),
            };
        }
        let callee = (n.kind() == "call_expression")
            .then(|| n.child_by_field_name("function").map(|f| text(f, self.src)))
            .flatten();
        let Some(callee) = callee
            .filter(|c| c.starts_with("make_") && vector_type(&c[5..]) == Some((elem, lanes)))
        else {
            return self.not_yet(
                n,
                "a vector value that is neither `make_<type>(...)` nor a vector local",
            );
        };
        let args = self.call_args(n);
        if args.len() != usize::from(lanes) {
            return self.not_yet(n, format!("`{callee}` with {} argument(s)", args.len()));
        }
        let mut xs = Vec::new();
        for a in args {
            let (v, t) = self.expr(a)?;
            xs.push(self.convert(v, t, elem, a)?);
        }
        Ok(xs)
    }

    /// `v = make_float4(...)` or `v = w`. Every lane value is computed into a temporary first, so `v = make_float2(v.y, v.x)`
    /// reads the old lanes.
    fn assign_vector(
        &mut self,
        ls: [u32; 4],
        shape: (E, u8),
        op: &str,
        rhs: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        if !op.is_empty() {
            return self.not_yet(
                rhs,
                "a compound assignment to a whole vector (operators on vector types are not C)",
            );
        }
        let values = self.vector_value(rhs, shape)?;
        self.flush(out);
        let mut temps = Vec::new();
        for v in values {
            let t = self.new_local(shape.0);
            out.push(Stmt::Let { local: t, init: v });
            temps.push(t);
        }
        for (k, t) in temps.into_iter().enumerate() {
            out.push(Stmt::Assign {
                local: ls[k],
                value: ScalarExpr::Local(t),
            });
        }
        Ok(())
    }

    /// `v.x = e` (and `v.x += e`): an assignment to that lane's local.
    fn assign_member(
        &mut self,
        lhs: Node<'_>,
        op: &str,
        rhs: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        let (Some(a), Some(f)) = (
            lhs.child_by_field_name("argument"),
            lhs.child_by_field_name("field"),
        ) else {
            return self.not_yet(lhs, "a member assignment of an unexpected shape");
        };
        let Some(Sym::Vector(ls, elem, lanes)) = self.lookup(text(a, self.src)) else {
            return self.not_yet(
                lhs,
                "an assignment to a member of something that is not a vector local",
            );
        };
        let Some(i) = lane_of(text(f, self.src), lanes) else {
            return self.not_yet(
                f,
                format!(
                    "the member `{}` of a {lanes}-lane vector",
                    text(f, self.src)
                ),
            );
        };
        let local = ls[usize::from(i)];
        let (r, rt) = self.expr(rhs)?;
        let value = self.compound(ScalarExpr::Local(local), elem, op, (r, rt), lhs)?;
        self.flush(out);
        out.push(Stmt::Assign { local, value });
        Ok(())
    }

    /// `__shared__ T name[N];` with a literal `N`: one `Shared` allocation, named by the array.
    fn array_declaration(&mut self, n: Node<'_>, ty: E, space: MemorySpace) -> R<()> {
        let what = if space == MemorySpace::Shared {
            "`__shared__`"
        } else {
            "a local array"
        };
        if self.launch.is_none() {
            return self.not_yet(
                n,
                format!("{what} without a launch (use `lift_kernel_launched`)"),
            );
        }
        let mut c = n.walk();
        let decls: Vec<_> = n.children_by_field_name("declarator", &mut c).collect();
        for d in decls {
            let (Some(id), Some(size)) = (
                d.child_by_field_name("declarator"),
                d.child_by_field_name("size"),
            ) else {
                return self.not_yet(d, format!("{what} declarator that is not a sized array `T name[N]` (an initializer is a later slice)"));
            };
            let count = match parse_literal(text(size, self.src)) {
                Some(Lit::Int(v, _)) if v > 0 => u32::try_from(v).ok(),
                _ => None,
            };
            let Some(count) = count else {
                return self.not_yet(
                    size,
                    format!("{what} whose size is not a positive integer literal"),
                );
            };
            if id.kind() != "identifier" {
                return self.not_yet(id, format!("{what} with more than one dimension"));
            }
            let name = text(id, self.src).to_string();
            self.allocs.push(AllocDecl::new(
                name.clone(),
                space,
                ty,
                Extent::Static(count),
            ));
            let index = u32::try_from(self.allocs.len() - 1).unwrap_or(u32::MAX);
            self.scopes
                .last_mut()
                .expect("a scope is open")
                .insert(name, Sym::Alloc(index, ty));
        }
        Ok(())
    }

    /// A call statement: only `__syncthreads()` (a block barrier) in slice 2.
    fn call_statement(&mut self, e: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
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

    /// `threadIdx.x`, `blockIdx.y`, `blockDim.z`, `gridDim.x`: the launch's execution ids (`U32`).
    fn builtin(&self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let (Some(a), Some(f)) = (
            n.child_by_field_name("argument"),
            n.child_by_field_name("field"),
        ) else {
            return self.not_yet(n, "a member access of an unexpected shape");
        };
        let base = text(a, self.src);
        if let (true, Some(Sym::Vector(ls, elem, lanes))) =
            (a.kind() == "identifier", self.lookup(base))
        {
            let lane = lane_of(text(f, self.src), lanes);
            let Some(i) = lane else {
                return self.not_yet(
                    f,
                    format!(
                        "the member `{}` of a {lanes}-lane vector",
                        text(f, self.src)
                    ),
                );
            };
            return Ok((ScalarExpr::Local(ls[usize::from(i)]), elem));
        }
        if a.kind() != "identifier" || self.lookup(base).is_some() {
            return self.not_yet(
                n,
                "a member access on a value (structs and vector types are a later slice)",
            );
        }
        let axis = match text(f, self.src) {
            "x" => 0,
            "y" => 1,
            "z" => 2,
            other => return self.not_yet(f, format!("the member `{other}`")),
        };
        let id = match base {
            "threadIdx" => ExecId::ThreadIdx(axis),
            "blockIdx" => ExecId::BlockIdx(axis),
            "blockDim" => ExecId::BlockDim(axis),
            "gridDim" => ExecId::GridDim(axis),
            other => {
                return self.not_yet(
                    a,
                    format!(
                        "the name `{other}`, which is not a parameter, local or thread builtin"
                    ),
                );
            }
        };
        if self.launch.is_none() {
            return self.not_yet(
                n,
                format!("`{base}` without a launch (use `lift_kernel_launched`)"),
            );
        }
        Ok((ScalarExpr::Exec(id), E::U32))
    }

    fn call_args<'n>(&self, n: Node<'n>) -> Vec<Node<'n>> {
        n.child_by_field_name("arguments")
            .map_or_else(Vec::new, |a| {
                let mut c = a.walk();
                a.named_children(&mut c).collect()
            })
    }

    /// A math call through the frontend's tables, or an atomic that returns the old value (hoisted as a statement).
    fn call_expr(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
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
    fn math_type(name: &str, arg: E) -> E {
        let single = name.ends_with('f') && !matches!(name, "fabs" | "abs");
        if single && name != "__fsqrt_rn" {
            E::F32
        } else {
            arg
        }
    }

    fn math_unary(&mut self, n: Node<'_>, name: &str, a: Node<'_>) -> R<(ScalarExpr, E)> {
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

    fn math_binary(
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
    fn atomic(&mut self, n: Node<'_>, name: &str, want_old: bool) -> R<Option<(ScalarExpr, E)>> {
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

    /// `&a[i]`: the place, its element type and the scope an atomic on it has.
    fn address_of(&mut self, n: Node<'_>) -> R<(Place, E, Scope)> {
        if !self.is_plain_address(n) {
            if let Some(p) = self.ptr_expr(n)? {
                let (place, elem) = self.pointer_place(&p, None, n)?;
                let scope = if matches!(place, Place::AllocRaw { .. }) {
                    Scope::Block
                } else {
                    Scope::Device
                };
                return Ok((place, elem, scope));
            }
        }
        let inner = if n.kind() == "parenthesized_expression" {
            n.named_child(0)
        } else {
            Some(n)
        };
        let Some(inner) = inner.filter(|i| i.kind() == "pointer_expression") else {
            return self.not_yet(
                n,
                "an atomic or `__ldg` whose address is not a pointer expression (`&a[i]`, `p + i`, ...)",
            );
        };
        let op = inner
            .child_by_field_name("operator")
            .map(|o| text(o, self.src));
        let arg = inner
            .child_by_field_name("argument")
            .filter(|a| a.kind() == "subscript_expression");
        let (Some("&"), Some(arg)) = (op, arg) else {
            return self.not_yet(n, "an atomic or `__ldg` whose address is not `&p[i]`");
        };
        let (place, elem) = self.element_place(arg)?;
        let scope = if matches!(place, Place::Alloc { .. }) {
            Scope::Block
        } else {
            Scope::Device
        };
        Ok((place, elem, scope))
    }

    fn expression_statement(&mut self, n: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        let Some(e) = n.named_child(0) else {
            return Ok(());
        };
        match e.kind() {
            "assignment_expression" => self.assignment(e, out),
            "update_expression" => self.update(e, out),
            "call_expression" => self.call_statement(e, out),
            "gnu_asm_expression" => self.asm_statement(e, out),
            other => self.not_yet(e, format!("a `{other}` expression statement")),
        }
    }

    fn assignment(&mut self, n: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        let (Some(lhs), Some(rhs), Some(op)) = (
            n.child_by_field_name("left"),
            n.child_by_field_name("right"),
            n.child_by_field_name("operator"),
        ) else {
            return self.not_yet(n, "an assignment of an unexpected shape");
        };
        let op = text(op, self.src).trim_end_matches('=').to_string();
        match lhs.kind() {
            "identifier" => self.assign_local(lhs, &op, rhs, out),
            "subscript_expression" | "pointer_expression" => {
                self.assign_element(lhs, &op, rhs, out)
            }
            "field_expression" => self.assign_member(lhs, &op, rhs, out),
            other => self.not_yet(lhs, format!("an assignment to a `{other}`")),
        }
    }

    fn assign_local(
        &mut self,
        lhs: Node<'_>,
        op: &str,
        rhs: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        let name = text(lhs, self.src);
        if let Some(Sym::Vector(ls, elem, lanes)) = self.lookup(name) {
            return self.assign_vector(ls, (elem, lanes), op, rhs, out);
        }
        let Some(Sym::Scalar(local, ty, fixed)) = self.lookup(name) else {
            return self.not_yet(
                lhs,
                format!("an assignment to `{name}`, which is not a scalar local"),
            );
        };
        match fixed {
            Fixed::Param => {
                return self.not_yet(lhs, format!("an assignment to the parameter `{name}`"));
            }
            Fixed::Counter => {
                return self.not_yet(lhs, format!("an assignment to the loop counter `{name}`"));
            }
            Fixed::No => {}
        }
        let (r, rt) = self.expr(rhs)?;
        let value = self.compound(ScalarExpr::Local(local), ty, op, (r, rt), lhs)?;
        self.flush(out);
        out.push(Stmt::Assign { local, value });
        Ok(())
    }

    fn assign_element(
        &mut self,
        lhs: Node<'_>,
        op: &str,
        rhs: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        let (place, elem) = self.element_place(lhs)?;
        let (r, rt) = self.expr(rhs)?;
        let value = if op.is_empty() {
            self.convert(r, rt, elem, rhs)?
        } else {
            let local = self.new_local(elem);
            self.hoisted.push(Stmt::Load {
                local,
                place: place.clone(),
            });
            self.compound(ScalarExpr::Local(local), elem, op, (r, rt), lhs)?
        };
        self.flush(out);
        out.push(Stmt::Store { place, value });
        Ok(())
    }

    /// `old op= rhs` as the value stored back, converted to the target's type; plain `=` converts `rhs` alone.
    fn compound(
        &self,
        old: ScalarExpr,
        ty: E,
        op: &str,
        rhs: (ScalarExpr, E),
        at: Node<'_>,
    ) -> R<ScalarExpr> {
        if op.is_empty() {
            return self.convert(rhs.0, rhs.1, ty, at);
        }
        let (v, vt) = self.binary(op, (old, ty), rhs, at)?;
        self.convert(v, vt, ty, at)
    }

    fn update(&mut self, n: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        let (Some(arg), Some(op)) = (
            n.child_by_field_name("argument"),
            n.child_by_field_name("operator"),
        ) else {
            return self.not_yet(n, "an increment of an unexpected shape");
        };
        let one = Self::lit_one_for(self.lookup(text(arg, self.src)));
        let Some(one) = one else {
            return self.not_yet(
                arg,
                "an increment of something that is not an integer local",
            );
        };
        let sign = if text(op, self.src) == "++" { "+" } else { "-" };
        self.assign_local_value(arg, sign, one, out)
    }

    fn lit_one_for(sym: Option<Sym>) -> Option<(ScalarExpr, E)> {
        match sym {
            Some(Sym::Scalar(_, t, _)) if int_range(t).is_some() => {
                Some((ScalarExpr::ConstInt(1, t), t))
            }
            _ => None,
        }
    }

    fn assign_local_value(
        &mut self,
        lhs: Node<'_>,
        op: &str,
        v: (ScalarExpr, E),
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        let name = text(lhs, self.src);
        let Some(Sym::Scalar(local, ty, fixed)) = self.lookup(name) else {
            return self.not_yet(lhs, format!("an update of `{name}`"));
        };
        if fixed != Fixed::No {
            return self.not_yet(
                lhs,
                format!("an update of the parameter or loop counter `{name}`"),
            );
        }
        let value = self.compound(ScalarExpr::Local(local), ty, op, v, lhs)?;
        out.push(Stmt::Assign { local, value });
        Ok(())
    }

    fn condition(&mut self, n: Node<'_>) -> R<ScalarExpr> {
        let inner = if n.kind() == "condition_clause" {
            n.child_by_field_name("value")
        } else {
            Some(n)
        };
        let Some(inner) = inner else {
            return self.not_yet(n, "a condition with a declaration");
        };
        Ok(self.expr(inner)?.0)
    }

    fn if_statement(&mut self, n: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        let (Some(c), Some(then)) = (
            n.child_by_field_name("condition"),
            n.child_by_field_name("consequence"),
        ) else {
            return self.not_yet(n, "an `if` of an unexpected shape");
        };
        let cond = self.condition(c)?;
        self.flush(out);
        let then = self.block_of(then)?;
        let els = match n.child_by_field_name("alternative") {
            Some(a) => match a.named_child(0) {
                Some(s) => self.block_of(s)?,
                None => Block::default(),
            },
            None => Block::default(),
        };
        out.push(Stmt::If { cond, then, els });
        Ok(())
    }

    fn while_statement(&mut self, n: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        let (Some(c), Some(body)) = (
            n.child_by_field_name("condition"),
            n.child_by_field_name("body"),
        ) else {
            return self.not_yet(n, "a `while` of an unexpected shape");
        };
        let cond = self.condition(c)?;
        // taken now: the body's own statements flush `hoisted`, and must not flush the condition's loads
        let mut lead = std::mem::take(&mut self.hoisted);
        let mut body = self.block_of(body)?;
        if lead.is_empty() {
            out.push(Stmt::While { cond, body });
            return Ok(());
        }
        // The condition reads memory: it must be re-read every iteration, so `while (1) { loads; if (!c) break; body }`.
        let (zero, _) = self.constant(0);
        lead.push(Stmt::If {
            cond: ScalarExpr::Select(
                Box::new(cond),
                Box::new(zero.clone()),
                Box::new(self.constant(1).0),
            ),
            then: Block(vec![Stmt::Break]),
            els: Block::default(),
        });
        lead.append(&mut body.0);
        out.push(Stmt::While {
            cond: ScalarExpr::ConstInt(1, E::I32),
            body: Block(lead),
        });
        Ok(())
    }

    /// `#pragma unroll` and `#pragma unroll N`: a request to the emitter, never a meaning (`Stmt::For::unroll`). Any other
    /// directive is a typed decline.
    fn pragma(&mut self, n: Node<'_>) -> R<()> {
        let directive = n
            .child_by_field_name("directive")
            .map(|d| text(d, self.src));
        let arg = n
            .child_by_field_name("argument")
            .map_or("", |a| text(a, self.src).trim());
        let mut words = arg.split_whitespace();
        if directive != Some("#pragma") || words.next() != Some("unroll") {
            return self.not_yet(n, format!("the directive `{}`", text(n, self.src).trim()));
        }
        self.pending_unroll = match (words.next(), words.next()) {
            (None, _) => Unroll::Full,
            (Some(c), None) => match c.parse::<u32>() {
                Ok(0) | Err(_) => {
                    return self
                        .not_yet(n, "a `#pragma unroll` count that is not a positive literal");
                }
                Ok(1) => Unroll::None,
                Ok(k) => Unroll::Count(k),
            },
            _ => return self.not_yet(n, "a `#pragma unroll` with trailing text"),
        };
        Ok(())
    }

    fn for_statement(&mut self, n: Node<'_>, out: &mut Vec<Stmt>, unroll: Unroll) -> R<()> {
        let Some(shape) = self.counted_for(n)? else {
            return self.not_yet(
                n,
                "a `for` that is not a counted loop `for (T i = a; i < b; i += s)`",
            );
        };
        let (name, ty, start, end, step, body) = shape;
        // the start is evaluated once, before the loop: its loads go before the `For`
        self.flush(out);
        let local = self.new_local(ty);
        self.scopes.push(HashMap::new());
        self.scopes
            .last_mut()
            .expect("a scope is open")
            .insert(name, Sym::Scalar(local, ty, Fixed::Counter));
        let body = self.block_of(body);
        self.scopes.pop();
        out.push(Stmt::For {
            local,
            start,
            end,
            step,
            unroll,
            body: body?,
        });
        Ok(())
    }

    /// The pieces of `for (T i = a; i < b; i += s) body`, or `None` for any other shape.
    #[allow(clippy::type_complexity)]
    fn counted_for<'n>(
        &mut self,
        n: Node<'n>,
    ) -> R<Option<(String, E, ScalarExpr, ScalarExpr, ScalarExpr, Node<'n>)>> {
        let (Some(init), Some(cond), Some(upd), Some(body)) = (
            n.child_by_field_name("initializer"),
            n.child_by_field_name("condition"),
            n.child_by_field_name("update"),
            n.child_by_field_name("body"),
        ) else {
            return Ok(None);
        };
        let Some((name, ty, a)) = self.for_init(init)? else {
            return Ok(None);
        };
        let (Some(b), Some(op)) = (
            cond.child_by_field_name("right"),
            cond.child_by_field_name("operator"),
        ) else {
            return Ok(None);
        };
        let lhs_is_counter = cond
            .child_by_field_name("left")
            .is_some_and(|l| text(l, self.src) == name);
        if cond.kind() != "binary_expression" || text(op, self.src) != "<" || !lhs_is_counter {
            return Ok(None);
        }
        let Some(step) = self.for_step(upd, &name, ty)? else {
            return Ok(None);
        };
        self.require_invariant(b)?;
        let (be, bt) = self.expr(b)?;
        let be = self.convert(be, bt, ty, b)?;
        Ok(Some((name, ty, a, be, step, body)))
    }

    fn for_init(&mut self, init: Node<'_>) -> R<Option<(String, E, ScalarExpr)>> {
        if init.kind() != "declaration" {
            return Ok(None);
        }
        let ty = self.type_of(init.child_by_field_name("type"), init)?;
        let d = init.child_by_field_name("declarator");
        let (Some(d), true) = (d, int_range(ty).is_some()) else {
            return Ok(None);
        };
        let (Some(id), Some(v)) = (
            d.child_by_field_name("declarator"),
            d.child_by_field_name("value"),
        ) else {
            return Ok(None);
        };
        let (a, at) = self.expr(v)?;
        let a = self.convert(a, at, ty, v)?;
        Ok(Some((text(id, self.src).to_string(), ty, a)))
    }

    fn for_step(&mut self, upd: Node<'_>, counter: &str, ty: E) -> R<Option<ScalarExpr>> {
        match upd.kind() {
            "update_expression" => {
                let ok = upd
                    .child_by_field_name("argument")
                    .is_some_and(|a| text(a, self.src) == counter)
                    && upd
                        .child_by_field_name("operator")
                        .is_some_and(|o| text(o, self.src) == "++");
                Ok(ok.then_some(ScalarExpr::ConstInt(1, ty)))
            }
            "assignment_expression" => {
                let l = upd.child_by_field_name("left");
                let op = upd.child_by_field_name("operator");
                let r = upd.child_by_field_name("right");
                let (Some(l), Some(op), Some(r)) = (l, op, r) else {
                    return Ok(None);
                };
                if text(l, self.src) != counter || text(op, self.src) != "+=" {
                    return Ok(None);
                }
                self.require_invariant(r)?;
                let (s, st) = self.expr(r)?;
                Ok(Some(self.convert(s, st, ty, r)?))
            }
            _ => Ok(None),
        }
    }

    /// A `For` bound is read once by the IR and every iteration by C: they agree only if nothing it names is assigned and
    /// nothing is loaded from memory.
    fn require_invariant(&self, n: Node<'_>) -> R<()> {
        if n.kind() == "subscript_expression" || n.kind() == "call_expression" {
            return self.not_yet(n, "a loop bound that reads memory or calls a function");
        }
        if n.kind() == "identifier" && self.assigned.contains(text(n, self.src)) {
            return self.not_yet(
                n,
                "a loop bound or step that names a variable the function assigns",
            );
        }
        let mut c = n.walk();
        n.named_children(&mut c)
            .try_for_each(|k| self.require_invariant(k))
    }

    // ---- expressions ----

    fn constant(&self, v: i64) -> (ScalarExpr, E) {
        if int_range(self.compute).is_some() {
            (ScalarExpr::ConstInt(v, self.compute), self.compute)
        } else {
            (ScalarExpr::Const(v as f64), self.compute)
        }
    }

    fn ty_of(&self, e: &ScalarExpr, at: Node<'_>) -> R<E> {
        let tys: Vec<Ty> = self.locals.iter().map(|l| l.ty()).collect();
        match infer_ty_in(e, self.compute, &tys) {
            Ok(Ty::Scalar(t)) => Ok(t),
            Ok(_) => self.not_yet(at, "a packed value"),
            Err(why) => self.outside(at, format!("`{}`: {why:?}", text(at, self.src))),
        }
    }

    fn expr(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let (e, _) = self.expr_raw(n)?;
        let t = self.ty_of(&e, n)?;
        Ok((e, t))
    }

    fn expr_raw(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        match n.kind() {
            "parenthesized_expression" => match n.named_child(0) {
                Some(i) => self.expr_raw(i),
                None => self.not_yet(n, "an empty parenthesis"),
            },
            "identifier" => self.identifier(n),
            "number_literal" => self.number(n),
            "true" | "false" => self.not_yet(n, "a boolean literal"),
            "binary_expression" => self.binary_node(n),
            "unary_expression" => self.unary_node(n),
            "conditional_expression" => self.conditional(n),
            "cast_expression" => self.cast_node(n),
            "subscript_expression" | "pointer_expression" => self.read_element(n),
            "field_expression" => self.builtin(n),
            "call_expression" => self.call_expr(n),
            other => self.not_yet(n, format!("a `{other}` expression")),
        }
    }

    fn identifier(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let name = text(n, self.src);
        match self.lookup(name) {
            Some(Sym::Scalar(l, t, _)) => Ok((ScalarExpr::Local(l), t)),
            Some(Sym::Buffer(..) | Sym::Alloc(..) | Sym::Pointer(..)) => {
                self.not_yet(n, format!("the array or pointer `{name}` used as a value"))
            }
            Some(Sym::Vector(..)) => self.not_yet(
                n,
                format!("the vector `{name}` used as a scalar (read a member, or assign it whole)"),
            ),
            None => self.not_yet(
                n,
                format!("the name `{name}`, which is not a parameter or local"),
            ),
        }
    }

    fn number(&self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        match parse_literal(text(n, self.src)) {
            Some(Lit::Int(v, t)) => Ok((ScalarExpr::ConstInt(v, t), t)),
            Some(Lit::Float(v, t)) if t == self.compute => Ok((ScalarExpr::Const(v), t)),
            Some(Lit::Float(_, t)) => self.not_yet(
                n,
                format!(
                    "a {t:?} literal in a kernel whose compute type is {:?}",
                    self.compute
                ),
            ),
            None => self.not_yet(n, format!("the literal `{}`", text(n, self.src))),
        }
    }

    fn binary_node(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let (Some(l), Some(r), Some(op)) = (
            n.child_by_field_name("left"),
            n.child_by_field_name("right"),
            n.child_by_field_name("operator"),
        ) else {
            return self.not_yet(n, "a binary expression of an unexpected shape");
        };
        let op = text(op, self.src);
        let lv = self.expr(l)?;
        let before = self.hoisted.len();
        let rv = self.expr(r)?;
        if matches!(op, "&&" | "||") {
            if self.hoisted.len() != before {
                return self.not_yet(r, "a memory read in the right operand of `&&`/`||` (it is conditionally evaluated)");
            }
            return Ok(self.logical(op, lv.0, rv.0));
        }
        self.binary(op, lv, rv, n)
    }

    fn logical(&self, op: &str, l: ScalarExpr, r: ScalarExpr) -> (ScalarExpr, E) {
        let (one, _) = self.constant(1);
        let (zero, _) = self.constant(0);
        let truth = ScalarExpr::Select(Box::new(r), Box::new(one.clone()), Box::new(zero.clone()));
        let e = if op == "&&" {
            ScalarExpr::Select(Box::new(l), Box::new(truth), Box::new(zero))
        } else {
            ScalarExpr::Select(Box::new(l), Box::new(one), Box::new(truth))
        };
        (e, self.compute)
    }

    fn binary(
        &self,
        op: &str,
        l: (ScalarExpr, E),
        r: (ScalarExpr, E),
        at: Node<'_>,
    ) -> R<(ScalarExpr, E)> {
        if matches!(op, "<<" | ">>") {
            return self.shift(op, l, r, at);
        }
        let common = self.common_type(l.1, r.1, at)?;
        let a = Box::new(self.convert(l.0, l.1, common, at)?);
        let b = Box::new(self.convert(r.0, r.1, common, at)?);
        let e = match op {
            "+" => ScalarExpr::Add(a, b),
            "-" => ScalarExpr::Sub(a, b),
            "*" => ScalarExpr::Mul(a, b),
            "/" => ScalarExpr::Div(a, b),
            "%" => return self.remainder(a, b, common, at),
            "&" => ScalarExpr::Binary(BinaryOp::BitAnd, a, b),
            "|" => ScalarExpr::Binary(BinaryOp::BitOr, a, b),
            "^" => ScalarExpr::Binary(BinaryOp::BitXor, a, b),
            "<" => ScalarExpr::Binary(BinaryOp::CmpLt, a, b),
            "<=" => ScalarExpr::Binary(BinaryOp::CmpLe, a, b),
            ">" => ScalarExpr::Binary(BinaryOp::CmpGt, a, b),
            ">=" => ScalarExpr::Binary(BinaryOp::CmpGe, a, b),
            "==" => ScalarExpr::Binary(BinaryOp::CmpEq, a, b),
            "!=" => ScalarExpr::Binary(BinaryOp::CmpNe, a, b),
            other => return self.not_yet(at, format!("the operator `{other}`")),
        };
        let t = self.ty_of(&e, at)?;
        Ok((e, t))
    }

    /// C `a % b` on integers: truncating, sign of the dividend, and `0` for a zero divisor (and for `MIN % -1`), as the
    /// total `Div` composes it (`docs/int-div-shift-semantics.md`, the PM ruling 2026-10-31): `Select(b == 0, 0, a - (a / b) * b)`.
    /// A zero divisor is undefined in C, so any total result is a refinement. Both operands are pure (a memory read is
    /// already a local), so naming each twice repeats no effect.
    fn remainder(
        &self,
        a: Box<ScalarExpr>,
        b: Box<ScalarExpr>,
        ty: E,
        at: Node<'_>,
    ) -> R<(ScalarExpr, E)> {
        if int_range(ty).is_none() {
            return self.outside(at, "`%` on a floating-point operand (not valid C)");
        }
        let quotient = ScalarExpr::Div(a.clone(), b.clone());
        let product = ScalarExpr::Mul(Box::new(quotient), b.clone());
        let rem = ScalarExpr::Sub(a, Box::new(product));
        let is_zero = ScalarExpr::Binary(BinaryOp::CmpEq, b, Box::new(ScalarExpr::ConstInt(0, ty)));
        let e = ScalarExpr::Select(
            Box::new(is_zero),
            Box::new(ScalarExpr::ConstInt(0, ty)),
            Box::new(rem),
        );
        let t = self.ty_of(&e, at)?;
        Ok((e, t))
    }

    /// A shift's result has the type of its **left** operand alone (the right is not converted to it).
    fn shift(
        &self,
        op: &str,
        l: (ScalarExpr, E),
        r: (ScalarExpr, E),
        at: Node<'_>,
    ) -> R<(ScalarExpr, E)> {
        let lt = promote(l.1);
        let a = self.convert(l.0, l.1, lt, at)?;
        let b = self.convert(r.0, r.1, lt, at)?;
        let bop = if op == "<<" {
            BinaryOp::Shl
        } else {
            BinaryOp::Shr
        };
        let e = ScalarExpr::Binary(bop, Box::new(a), Box::new(b));
        let t = self.ty_of(&e, at)?;
        Ok((e, t))
    }

    fn common_type(&self, a: E, b: E, at: Node<'_>) -> R<E> {
        let (fa, fb) = (int_range(a).is_none(), int_range(b).is_none());
        if fa || fb {
            return match (fa, fb) {
                (true, true) if a == b => Ok(a),
                (true, true) if matches!((a, b), (E::F64, _) | (_, E::F64)) => Ok(E::F64),
                (true, false) => Ok(a),
                (false, true) => Ok(b),
                _ => self.not_yet(at, format!("an operation on {a:?} and {b:?}")),
            };
        }
        let (a, b) = (promote(a), promote(b));
        if a == b {
            return Ok(a);
        }
        let (ra, rb) = (int_rank(a), int_rank(b));
        Ok(if ra == rb {
            // equal width, different signedness: the unsigned one
            if is_unsigned(a) { a } else { b }
        } else if ra > rb {
            a
        } else {
            b
        })
    }

    /// `e` of type `from` as type `to`, by a `Cast` when they differ (C's implicit conversion made explicit).
    fn convert(&self, e: ScalarExpr, from: E, to: E, at: Node<'_>) -> R<ScalarExpr> {
        if from == to {
            return Ok(e);
        }
        let float_to_int = int_range(from).is_none() && int_range(to).is_some();
        let spec = CastSpec {
            to,
            round: if float_to_int {
                RoundMode::TowardZero
            } else {
                RoundMode::NearestEven
            },
            sat: if float_to_int {
                Saturation::Clamp
            } else {
                Saturation::Wrap
            },
        };
        let out = ScalarExpr::Cast(Box::new(e), spec);
        self.ty_of(&out, at)?;
        Ok(out)
    }

    fn unary_node(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let (Some(arg), Some(op)) = (
            n.child_by_field_name("argument"),
            n.child_by_field_name("operator"),
        ) else {
            return self.not_yet(n, "a unary expression of an unexpected shape");
        };
        let (v, t) = self.expr(arg)?;
        match text(op, self.src) {
            "+" => Ok((v, t)),
            "-" if int_range(t).is_some() => {
                let wide = promote(t);
                let v = self.convert(v, t, wide, arg)?;
                Ok((
                    ScalarExpr::Sub(Box::new(ScalarExpr::ConstInt(0, wide)), Box::new(v)),
                    wide,
                ))
            }
            "-" => Ok((ScalarExpr::Unary(UnaryOp::Neg, Box::new(v)), t)),
            "!" => {
                let (one, _) = self.constant(1);
                let (zero, _) = self.constant(0);
                Ok((
                    ScalarExpr::Select(Box::new(v), Box::new(zero), Box::new(one)),
                    self.compute,
                ))
            }
            other => self.not_yet(n, format!("the unary operator `{other}`")),
        }
    }

    fn conditional(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let (Some(c), Some(a), Some(b)) = (
            n.child_by_field_name("condition"),
            n.child_by_field_name("consequence"),
            n.child_by_field_name("alternative"),
        ) else {
            return self.not_yet(n, "a conditional of an unexpected shape");
        };
        let cond = self.expr(c)?.0;
        let before = self.hoisted.len();
        let (av, at) = self.expr(a)?;
        let (bv, bt) = self.expr(b)?;
        if self.hoisted.len() != before {
            return self.not_yet(n, "a memory read inside `?:` (only one arm is evaluated)");
        }
        let common = self.common_type(at, bt, n)?;
        let av = self.convert(av, at, common, n)?;
        let bv = self.convert(bv, bt, common, n)?;
        Ok((
            ScalarExpr::Select(Box::new(cond), Box::new(av), Box::new(bv)),
            common,
        ))
    }

    fn cast_node(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let (Some(ty), Some(v)) = (
            n.child_by_field_name("type"),
            n.child_by_field_name("value"),
        ) else {
            return self.not_yet(n, "a cast of an unexpected shape");
        };
        let to = self.type_of(ty.child_by_field_name("type"), ty)?;
        let (e, from) = self.expr(v)?;
        Ok((self.convert(e, from, to, n)?, to))
    }

    fn element_place(&mut self, n: Node<'_>) -> R<(Place, E)> {
        if n.kind() == "pointer_expression" {
            return self.deref_place(n);
        }
        let (Some(arg), Some(idx)) = (
            n.child_by_field_name("argument"),
            n.child_by_field_name("indices"),
        ) else {
            return self.not_yet(n, "an index of an unexpected shape");
        };
        let sym = (arg.kind() == "identifier")
            .then(|| self.lookup(text(arg, self.src)))
            .flatten();
        let Some(Sym::Buffer(_, elem) | Sym::Alloc(_, elem)) = sym else {
            // not a plain buffer or array: a pointer expression (`p[i]`, `(T*)p`, `(p + k)[i]`), read through a raw place
            if let Some(p) = self.ptr_expr(arg)? {
                let one = idx.named_child(0).filter(|_| idx.named_child_count() == 1);
                let Some(one) = one else {
                    return self.not_yet(idx, "an index with other than one subscript");
                };
                return self.pointer_place(&p, Some(one), n);
            }
            return self.not_yet(
                arg,
                "indexing something that is not a buffer parameter, a `__shared__` array or a pointer",
            );
        };
        if idx.named_child_count() != 1 {
            return self.not_yet(idx, "an index with other than one subscript");
        }
        let i = idx.named_child(0).expect("one subscript");
        let (ie, it) = self.expr(i)?;
        if int_range(it).is_none() {
            return self.outside(i, "a floating-point index");
        }
        let place = match sym {
            Some(Sym::Alloc(alloc, _)) => Place::Alloc { alloc, index: ie },
            Some(Sym::Buffer(buffer, _)) => Place::Logical {
                buffer,
                coords: vec![ie],
            },
            _ => unreachable!("matched above"),
        };
        Ok((place, elem))
    }

    /// `*p`: the place a pointer expression points at.
    fn deref_place(&mut self, n: Node<'_>) -> R<(Place, E)> {
        let op = n.child_by_field_name("operator").map(|o| text(o, self.src));
        let arg = n.child_by_field_name("argument");
        let (Some("*"), Some(arg)) = (op, arg) else {
            return self.not_yet(n, "the address-of operator used as a value (only `&p[i]` as an atomic or `__ldg` address, or a pointer initialiser)");
        };
        match self.ptr_expr(arg)? {
            Some(p) => self.pointer_place(&p, None, n),
            None => self.not_yet(n, "a dereference of something that is not a pointer"),
        }
    }

    /// Whether `n` is `&a[i]` of a plain buffer parameter or `__shared__` array (which keeps its typed `Logical`/`Alloc` place).
    fn is_plain_address(&self, n: Node<'_>) -> bool {
        let inner = if n.kind() == "parenthesized_expression" {
            n.named_child(0)
        } else {
            Some(n)
        };
        let Some(inner) = inner.filter(|i| i.kind() == "pointer_expression") else {
            return false;
        };
        let is_amp = inner
            .child_by_field_name("operator")
            .is_some_and(|o| text(o, self.src) == "&");
        let base = inner
            .child_by_field_name("argument")
            .filter(|a| a.kind() == "subscript_expression")
            .and_then(|a| a.child_by_field_name("argument"))
            .filter(|b| b.kind() == "identifier");
        is_amp
            && base.is_some_and(|b| {
                matches!(
                    self.lookup(text(b, self.src)),
                    Some(Sym::Buffer(..) | Sym::Alloc(..))
                )
            })
    }

    fn read_element(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let (place, elem) = self.element_place(n)?;
        let local = self.new_local(elem);
        self.hoisted.push(Stmt::Load { local, place });
        Ok((ScalarExpr::Local(local), elem))
    }
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
}

fn pick_compute(raw: &[(String, PShape)]) -> E {
    let float = |t: E| int_range(t).is_none();
    let all = || {
        raw.iter().map(|(_, s)| match s {
            PShape::Buffer(t, _) | PShape::Scalar(t) => *t,
        })
    };
    all()
        .find(|t| float(*t))
        .or_else(|| {
            raw.iter().find_map(|(_, s)| match s {
                PShape::Buffer(t, _) => Some(*t),
                PShape::Scalar(_) => None,
            })
        })
        .unwrap_or(E::I32)
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
        "bool" => E::Bool,
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
