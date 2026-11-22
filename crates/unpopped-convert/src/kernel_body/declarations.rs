// SPDX-License-Identifier: MIT OR Apache-2.0
//! Declarations: scalar, vector, array and `__shared__` locals (ROADMAP U19).

use super::*;

impl<'s, 't> Walker<'s, 't> {
    pub(super) fn declaration(&mut self, n: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        if let Some(r) = self.group_declaration(n) {
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
        if let Some((elem, lanes)) = tname.as_deref().and_then(vector_type) {
            return self.vector_declaration(n, (elem, lanes), out);
        }
        let ty = self.type_of(n.child_by_field_name("type"), n)?;
        let head = text(n, self.src).trim_start();
        if declares_pointer(n) {
            return self.pointer_declaration(n, ty, out);
        }
        if head.starts_with("__shared__") {
            return self.array_declaration(n, ty, MemorySpace::Shared, out);
        }
        let has_array = declares_array(n);
        if has_array && head.starts_with("extern") && head.contains("__shared__") {
            return self.dynamic_shared_declaration(n, ty);
        }
        if has_array && !head.starts_with("extern") {
            return self.array_declaration(n, ty, MemorySpace::Private, out);
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
            out.push(Stmt::Let { local, init: e });
        }
        Ok(())
    }

    /// The zero of `ty`, as an expression.
    pub(super) fn zero_of(&self, ty: E, at: Node<'_>) -> R<ScalarExpr> {
        if int_range(ty).is_some() {
            return Ok(ScalarExpr::ConstInt(0, ty));
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
                None if self.address_taken(text(id, self.src)) => {
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
            let Some(count) = dims.iter().try_fold(1u32, |a, &d| a.checked_mul(d)) else {
                return self.not_yet(d, format!("{what} with more than u32::MAX elements"));
            };
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
            self.scopes
                .last_mut()
                .expect("a scope is open")
                .insert(name, Sym::Alloc(index, ty));
            if let Some(list) = init {
                self.array_initialiser((index, ty, count), list, space, out)?;
            }
        }
        Ok(())
    }
}
