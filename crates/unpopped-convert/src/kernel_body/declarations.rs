// SPDX-License-Identifier: MIT OR Apache-2.0
//! Declarations: scalar, vector, array and `__shared__` locals (ROADMAP U19).

use super::pointer::{PBase, PtrVal};
use super::*;

impl<'s, 't> Walker<'s, 't> {
    pub(super) fn declaration(&mut self, n: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        if let Some(r) = self.group_declaration(n) {
            return r;
        }
        if let Some(r) = self.tile_declaration(n) {
            return r;
        }
        let struct_type = n
            .child_by_field_name("type")
            .and_then(|t| self.struct_named(t));
        if let Some(sname) = struct_type {
            return self.struct_declaration(n, sname, out);
        }
        let tname = n.child_by_field_name("type").map(|t| {
            text(t, self.src)
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        });
        let vector = tname
            .as_deref()
            .map(|t| self.resolve_alias_text(t))
            .and_then(|t| vector_type(&t));
        if let Some((elem, lanes)) = vector {
            if declares_pointer(n) {
                return self.pointer_declaration(n, elem, lanes, out);
            }
            if declares_array(n) {
                // `float4 d[N];`: N vectors, held as one allocation of N * lanes scalars read and written as vectors
                let space = if text(n, self.src).trim_start().starts_with("__shared__") {
                    MemorySpace::Shared
                } else {
                    MemorySpace::Private
                };
                return self.array_declaration(n, elem, space, lanes, out);
            }
            return self.vector_declaration(n, (elem, lanes), out);
        }
        if declares_pointer(n)
            && let Some(sname) = tname.as_deref().and_then(|t| self.layout_name(t))
        {
            return self.struct_pointer_declaration(n, &sname, out);
        }
        let ty = self.type_of(n.child_by_field_name("type"), n)?;
        let head = text(n, self.src).trim_start();
        if declares_pointer(n) {
            return self.pointer_declaration(n, ty, 1, out);
        }
        if head.starts_with("__shared__") {
            return self.array_declaration(n, ty, MemorySpace::Shared, 1, out);
        }
        let has_array = declares_array(n);
        if has_array && head.starts_with("extern") && head.contains("__shared__") {
            return self.dynamic_shared_declaration(n, ty);
        }
        if has_array && !head.starts_with("extern") {
            return self.array_declaration(n, ty, MemorySpace::Private, 1, out);
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
        self.scalar_declaration(n, ty, out)
    }

    /// `T x = e;` and `T x;` (a scalar local, zero when not initialised).
    pub(super) fn scalar_declaration(&mut self, n: Node<'_>, ty: E, out: &mut Vec<Stmt>) -> R<()> {
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
            if let Some(v) = value
                && let Some(mut p) = self.shared_address(v)?
            {
                // the integer a `cp.async` takes: a symbolic shared pointer, fixed now
                self.flush(out);
                p.off = self.pin_offset(p.off, out);
                self.scopes
                    .last_mut()
                    .expect("a scope is open")
                    .insert(text(id, self.src).to_string(), Sym::Pointer(p));
                continue;
            }
            let is_bool = n
                .child_by_field_name("type")
                .is_some_and(|t| text(t, self.src) == "bool");
            let e = match value {
                // `bool b = e;` is `e != 0`: normalised, so a value other than 0 or 1 still gives 1
                Some(v) if is_bool => {
                    // `e != 0` is the compute type's 1 or 0 (a comparison), cast to a byte. Not a `Select` of two byte
                    // constants: an integer `Select` with a float-typed condition passes `check_kernel` but the oracle
                    // refuses it ("comparison of a non-integer operand").
                    let (e, et) = self.expr(v)?;
                    let zero = self.zero_of(et, v)?;
                    let nonzero = ScalarExpr::Binary(BinaryOp::CmpNe, Box::new(e), Box::new(zero));
                    ScalarExpr::Cast(
                        Box::new(nonzero),
                        CastSpec {
                            to: E::U8,
                            round: RoundMode::TowardZero,
                            sat: Saturation::Clamp,
                        },
                    )
                }
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
            // a `const` integer local with a known initialiser is a constant a loop can unroll over
            if value.is_some()
                && int_range(ty).is_some()
                && (0..n.child_count()).filter_map(|i| n.child(i)).any(|k| {
                    k.kind() == "type_qualifier"
                        && matches!(text(k, self.src), "const" | "constexpr")
                })
                && let Some(v) = self.launch_fold(&e)
            {
                self.known.insert(local, v);
            }
            out.push(Stmt::Let { local, init: e });
        }
        Ok(())
    }

    /// The zero of `ty`, as an expression.
    pub(super) fn zero_of(&self, ty: E, at: Node<'_>) -> R<ScalarExpr> {
        if int_range(ty).is_some() {
            // an integer leaf is I32, I64, U32 or U64 only: a narrower zero is an `I32` zero converted
            if matches!(ty, E::I32 | E::I64 | E::U32 | E::U64) {
                return Ok(ScalarExpr::ConstInt(0, ty));
            }
            return self.convert(ScalarExpr::ConstInt(0, E::I32), E::I32, ty, at);
        }
        self.convert(ScalarExpr::Const(0.0), self.compute, ty, at)
    }

    /// `float4 v = make_float4(a, b, c, d);` or `float4 v = w;`: one scalar local per lane.
    pub(super) fn vector_declaration(
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
            let Some(id) = id else {
                return self.not_yet(d, "a vector local declarator of an unexpected shape");
            };
            if id.kind() != "identifier" {
                return self.not_yet(id, format!("a `{}` declarator", id.kind()));
            }
            let init = match value {
                Some(v) => self.vector_value(v, (elem, lanes))?,
                // no initializer: only for a vector whose address is taken (it is then filled through the pointer); zero
                // no initializer: only for a vector filled later, through its address or as an argument (a reference parameter);
                // it starts at zero
                None if self.address_taken(text(id, self.src))
                    || self.passed_as_argument(text(id, self.src)) =>
                {
                    let zero = self.zero_of(elem, id)?;
                    vec![zero; usize::from(lanes)]
                }
                None => return self.not_yet(d, "a vector local without an initializer"),
            };
            self.flush(out);
            if self.address_taken(text(id, self.src)) {
                self.vector_alloc_declaration(text(id, self.src), (elem, lanes), init, out)?;
                continue;
            }
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
    pub(super) fn vector_value(
        &mut self,
        n: Node<'_>,
        (elem, lanes): (E, u8),
    ) -> R<Vec<ScalarExpr>> {
        if n.kind() == "identifier"
            && let Some(Sym::VecAlloc(i, e, k)) = self.lookup(text(n, self.src))
            && (e, k) == (elem, lanes)
        {
            return Ok(self.vecalloc_lanes(i, (e, k)));
        }
        if let Some(v) = self.vector_convert(n, (elem, lanes))? {
            return Ok(v);
        }
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
        if let Some(v) = self.vector_load(n, (elem, lanes))? {
            return Ok(v);
        }
        if let Some(v) = self.layout_vector(n, (elem, lanes))? {
            return Ok(v);
        }
        if n.kind() == "initializer_list" {
            // `float4 s = {a, b}`: the lanes in order, the rest zero (aggregate initialisation)
            let mut c = n.walk();
            let els: Vec<_> = n
                .named_children(&mut c)
                .filter(|e| e.kind() != "comment")
                .collect();
            if els.len() > usize::from(lanes) {
                return self.not_yet(
                    n,
                    format!("{} initialisers for a vector of {lanes} lanes", els.len()),
                );
            }
            let mut xs = Vec::new();
            for e in els {
                let (v, t) = self.expr(e)?;
                xs.push(self.convert(v, t, elem, e)?);
            }
            while xs.len() < usize::from(lanes) {
                xs.push(self.zero_of(elem, n)?);
            }
            return Ok(xs);
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
    pub(super) fn assign_vector(
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
    pub(super) fn assign_member(
        &mut self,
        lhs: Node<'_>,
        op: &str,
        rhs: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        if let Some(m) = self.resolve_member(lhs)? {
            return self.assign_struct_member(m, lhs, op, rhs, out);
        }
        let (Some(a), Some(f)) = (
            lhs.child_by_field_name("argument"),
            lhs.child_by_field_name("field"),
        ) else {
            return self.not_yet(lhs, "a member assignment of an unexpected shape");
        };
        if let Some((place, elem)) = self.vector_lane_place(a, f)? {
            // `d[i].x = e` (and `+=`): the lane read, combined, written back
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
            return Ok(());
        }
        // the vector may itself be a field of a struct (`v.head.y = e`)
        let vector = if a.kind() == "field_expression" {
            self.resolve_member(a)?
        } else {
            self.lookup(text(a, self.src))
        };
        let Some(Sym::Vector(ls, elem, lanes)) = vector else {
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

    /// `T name[N] = { e0, e1, ... };`: the elements stored in order, the rest zero (C zero-fills a short brace list). A `__shared__`
    /// array has no initialiser, a nested brace or a designator declines (an element is an ordinary expression).
    fn array_initialiser(
        &mut self,
        (alloc, ty, count): (u32, E, u32),
        list: Node<'_>,
        space: MemorySpace,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        if space == MemorySpace::Shared {
            return self.not_yet(list, "a `__shared__` array with an initialiser");
        }
        if list.kind() != "initializer_list" {
            return self.not_yet(list, "an array initialiser that is not a brace list");
        }
        let mut c = list.walk();
        let els: Vec<_> = list
            .named_children(&mut c)
            .filter(|e| e.kind() != "comment")
            .collect();
        if els.len() > count as usize {
            return self.not_yet(
                list,
                format!(
                    "{} initialisers for an array of {count} elements",
                    els.len()
                ),
            );
        }
        if els.len() < count as usize {
            self.zero_fill(alloc, ty, count, list, out)?;
        }
        for (i, el) in els.into_iter().enumerate() {
            let (v, vt) = self.expr(el)?;
            let value = self.convert(v, vt, ty, el)?;
            self.flush(out);
            out.push(Stmt::Store {
                place: Place::Alloc {
                    alloc,
                    index: ScalarExpr::ConstInt(i64::try_from(i).unwrap_or(i64::MAX), E::I32),
                },
                value,
            });
        }
        Ok(())
    }

    /// `__shared__ T name[N];` with a literal `N`: one `Shared` allocation, named by the array.
    pub(super) fn array_declaration(
        &mut self,
        n: Node<'_>,
        ty: E,
        space: MemorySpace,
        lanes: u8,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
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
            // `T name[N] = { ... }`: a private array with an initialiser list
            let (d, init) = if d.kind() == "init_declarator" {
                let inner = d.child_by_field_name("declarator").unwrap_or(d);
                (inner, d.child_by_field_name("value"))
            } else {
                (d, None)
            };
            let (Some(id), Some(size)) = (
                d.child_by_field_name("declarator"),
                d.child_by_field_name("size"),
            ) else {
                return self.not_yet(
                    d,
                    format!("{what} declarator that is not a sized array `T name[N]`"),
                );
            };
            // `T name[M][K]`: nested array declarators, the innermost written first; the allocation is the row-major product
            let (mut id, mut sizes) = (id, vec![size]);
            while id.kind() == "array_declarator" {
                let (Some(inner), Some(s)) = (
                    id.child_by_field_name("declarator"),
                    id.child_by_field_name("size"),
                ) else {
                    return self.not_yet(id, format!("{what} declarator of an unexpected shape"));
                };
                sizes.push(s);
                id = inner;
            }
            sizes.reverse();
            let mut dims = Vec::new();
            for s in &sizes {
                let Some(extent) = self.fold_size(*s)? else {
                    return self.not_yet(
                        *s,
                        format!("{what} whose size is not a positive integer literal"),
                    );
                };
                dims.push(extent);
            }
            let Some(count) = dims
                .iter()
                .try_fold(u32::from(lanes), |a, &d| a.checked_mul(d))
            else {
                return self.not_yet(d, format!("{what} with more than u32::MAX elements"));
            };
            if lanes > 1 && (dims.len() > 1 || init.is_some()) {
                return self.not_yet(
                    d,
                    format!("{what} of vectors with more than one dimension or an initialiser"),
                );
            }
            if dims.len() > 1 && init.is_some() {
                return self.not_yet(
                    d,
                    format!("{what} with more than one dimension and an initialiser"),
                );
            }
            if id.kind() != "identifier" {
                return self.not_yet(id, format!("{what} declarator that is not a plain name"));
            }
            let name = text(id, self.src).to_string();
            self.allocs.push(AllocDecl::new(
                name.clone(),
                space,
                ty,
                Extent::Static(count),
            ));
            let index = u32::try_from(self.allocs.len() - 1).unwrap_or(u32::MAX);
            if dims.len() > 1 {
                self.alloc_dims.insert(index, dims);
            }
            let sym = if lanes > 1 {
                // an array of vectors is a pointer whose pointee is the vector: `d[i]` is then a vector read or write
                Sym::Pointer(PtrVal {
                    base: PBase::Alloc(index),
                    pointee: ty,
                    lanes,
                    off: ScalarExpr::ConstInt(0, E::I64),
                    strukt: None,
                })
            } else {
                Sym::Alloc(index, ty)
            };
            self.scopes
                .last_mut()
                .expect("a scope is open")
                .insert(name, sym);
            if let Some(list) = init {
                self.array_initialiser((index, ty, count), list, space, out)?;
            }
        }
        Ok(())
    }
}
