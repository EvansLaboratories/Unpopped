// SPDX-License-Identifier: MIT OR Apache-2.0
//! Structs (ROADMAP U19 slice 4d, `docs/options-u19-slice5-decode-census.md`): a C `struct` of scalar fields, fixed-size arrays and
//! nested structs is **scalarized**: an instance is one scalar local per scalar field, one `Private` allocation per array
//! field, and the nested instances likewise. A kernel parameter of struct type becomes one `ParamKind::Struct` (scalar fields
//! only; nested fields are flattened with dotted names). Member functions, pointer fields, bit-fields and brace initialisers
//! decline by name. An uninitialised field starts at zero (C leaves it indeterminate until written).

use super::{Fixed, R, Sym, Walker, text};
use std::collections::HashMap;
use std::rc::Rc;
use tree_sitter::Node;
use unpopped::exec::{AllocDecl, Extent, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, Param, Place, Stmt, StructField, Unroll};
use unpopped_vocab::ElementKind as E;

/// The most elements of a struct parameter's array field that are flattened to scalar fields (`Select` ladders grow with it).
const MAX_PARAM_ARRAY: u32 = 64;

/// The size of an array field, as written.
#[derive(Clone, Debug)]
pub(super) enum ArrSize {
    /// `T name[8]`.
    Lit(u32),
    /// `T name[VEC_SIZE]`: a file-scope `constexpr`.
    Named(String),
}

/// One field of a struct definition, as written.
#[derive(Clone, Debug)]
pub(super) struct FieldDef {
    pub(super) name: String,
    /// The type text (`float`, `uint32_t`, another struct's name).
    pub(super) ty: String,
    /// The element count of an array field: a literal or the name of a file-scope constant.
    pub(super) array: Option<ArrSize>,
    /// A pointer field (`T* name`): held in an instance as a symbolic pointer, so only a brace list or a copy can build it.
    pub(super) pointer: bool,
    /// Why this field cannot be scalarized, if it cannot (a bit-field, ...).
    pub(super) unsupported: Option<String>,
}

/// An instance of a struct: its fields, each a scalar local, a private array or a nested instance.
pub(super) struct StructInst {
    /// The struct's type name (its methods are looked up by it).
    pub(super) sname: String,
    pub(super) fields: Vec<(String, Sym)>,
}

/// The struct definitions of the source, by name, in field order.
pub(super) fn collect_structs(root: Node<'_>, src: &str) -> HashMap<String, Vec<FieldDef>> {
    let mut out = HashMap::new();
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if n.kind() == "struct_specifier" {
            if let (Some(name), Some(body)) =
                (n.child_by_field_name("name"), n.child_by_field_name("body"))
            {
                out.insert(text(name, src).to_string(), struct_fields(body, src));
            }
        }
        let mut c = n.walk();
        stack.extend(n.children(&mut c));
    }
    out
}

fn struct_fields(body: Node<'_>, src: &str) -> Vec<FieldDef> {
    let mut out = Vec::new();
    let mut c = body.walk();
    for f in body
        .named_children(&mut c)
        .filter(|f| f.kind() == "field_declaration")
    {
        let ty = f
            .child_by_field_name("type")
            .map(|t| {
                text(t, src)
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_default();
        let mut dc = f.walk();
        for d in f.children_by_field_name("declarator", &mut dc) {
            out.push(field_of(d, &ty, src));
        }
    }
    out
}

fn field_of(d: Node<'_>, ty: &str, src: &str) -> FieldDef {
    let bad = |name: &str, why: &str| FieldDef {
        name: name.to_string(),
        ty: ty.to_string(),
        array: None,
        pointer: false,
        unsupported: Some(why.to_string()),
    };
    match d.kind() {
        "field_identifier" => FieldDef {
            name: text(d, src).to_string(),
            ty: ty.to_string(),
            array: None,
            pointer: false,
            unsupported: None,
        },
        "array_declarator" => {
            let id = d
                .child_by_field_name("declarator")
                .filter(|i| i.kind() == "field_identifier");
            let n = d.child_by_field_name("size").and_then(|s| {
                // an extent substituted from a template argument arrives parenthesised (`data[(4)]`)
                let mut t = text(s, src).trim();
                while let Some(inner) = t.strip_prefix('(').and_then(|r| r.strip_suffix(')')) {
                    t = inner.trim();
                }
                let ident = !t.is_empty()
                    && !t.starts_with(|c: char| c.is_ascii_digit())
                    && t.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
                match t.parse::<u32>() {
                    Ok(n) if n > 0 => Some(ArrSize::Lit(n)),
                    _ if ident => Some(ArrSize::Named(t.to_string())),
                    // an extent that is arithmetic on literals (`(128) / 4`, from `vec_size / 4`)
                    _ => super::context::literal_expression(text(s, src))
                        .and_then(|v| u32::try_from(v).ok())
                        .filter(|v| *v > 0)
                        .map(ArrSize::Lit),
                }
            });
            match (id, n) {
                (Some(id), Some(n)) => FieldDef {
                    name: text(id, src).to_string(),
                    ty: ty.to_string(),
                    array: Some(n),
                    pointer: false,
                    unsupported: None,
                },
                _ => bad(
                    text(d, src),
                    "an array field that is not `T name[N]` with a positive literal or constant `N`",
                ),
            }
        }
        "pointer_declarator" => {
            let id = d
                .child_by_field_name("declarator")
                .filter(|i| i.kind() == "field_identifier");
            match id {
                Some(i) => FieldDef {
                    name: text(i, src).to_string(),
                    ty: ty.to_string(),
                    array: None,
                    pointer: true,
                    unsupported: None,
                },
                None => bad(text(d, src), "a pointer field that is not `T* name`"),
            }
        }
        other => bad(text(d, src), &format!("a `{other}` field")),
    }
}

impl<'s, 't> Walker<'s, 't> {
    /// Whether `name` is a struct defined in this source.
    pub(super) fn is_struct(&self, name: &str) -> bool {
        self.structs.contains_key(name)
    }

    /// The struct a type node names, plain (`S`) or namespace-qualified (`ns::inner::S`, resolved by its last segment, since
    /// structs are collected by short name across namespaces), if it is one of this source's structs.
    pub(super) fn struct_named(&self, t: Node<'_>) -> Option<&'s str> {
        let id = match t.kind() {
            "type_identifier" => t,
            "qualified_identifier" => {
                let mut cur = t;
                while let Some(inner) = cur
                    .child_by_field_name("name")
                    .filter(|n| n.kind() == "qualified_identifier")
                {
                    cur = inner;
                }
                cur.child_by_field_name("name")
                    .filter(|n| n.kind() == "type_identifier")?
            }
            _ => return None,
        };
        let name = text(id, self.src);
        self.is_struct(name).then_some(name)
    }

    /// A fresh instance of `sname`: every field zero, every array a private allocation; its statements go to `out`.
    pub(super) fn new_struct(
        &mut self,
        sname: &str,
        label: &str,
        at: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<Rc<StructInst>> {
        let Some(defs) = self.structs.get(sname).cloned() else {
            return self.not_yet(at, format!("the struct `{sname}`"));
        };
        let mut fields = Vec::new();
        for f in defs {
            if let Some(why) = &f.unsupported {
                return self.not_yet(at, format!("the field `{}` of `{sname}`: {why}", f.name));
            }
            let path = format!("{label}.{}", f.name);
            if f.pointer {
                return self.not_yet(at, format!("the pointer field `{path}` of `{sname}`: build the struct from a brace list or a copy of another instance"));
            }
            fields.push((f.name.clone(), self.new_field(&f, &path, at, out)?));
        }
        Ok(Rc::new(StructInst {
            sname: sname.to_string(),
            fields,
        }))
    }

    fn new_field(&mut self, f: &FieldDef, path: &str, at: Node<'_>, out: &mut Vec<Stmt>) -> R<Sym> {
        if self.is_struct(&f.ty) {
            if f.array.is_some() {
                return self.not_yet(at, format!("an array of structs (`{path}`)"));
            }
            return Ok(Sym::Struct(self.new_struct(&f.ty, path, at, out)?));
        }
        if let Some(shape) = super::vector_type(&f.ty) {
            return self.new_vector_field(f, path, shape, at, out);
        }
        let e = match self.ctype(&f.ty) {
            Ok(e) => e,
            Err(why) => return self.not_yet(at, format!("the field `{path}`: {why}")),
        };
        match f.array.clone() {
            None => {
                let local = self.new_local(e);
                out.push(Stmt::Let {
                    local,
                    init: self.zero_of(e, at)?,
                });
                Ok(Sym::Scalar(local, e, Fixed::No))
            }
            Some(sz) => {
                let n = self.arr_count(&sz, at)?;
                if self.launch.is_none() {
                    return self.not_yet(
                        at,
                        format!(
                            "the array field `{path}` without a launch (use `lift_kernel_launched`)"
                        ),
                    );
                }
                self.allocs.push(AllocDecl::new(
                    path,
                    MemorySpace::Private,
                    e,
                    Extent::Static(n),
                ));
                let alloc = u32::try_from(self.allocs.len() - 1).unwrap_or(u32::MAX);
                self.zero_fill(alloc, e, n, at, out)?;
                Ok(Sym::Alloc(alloc, e))
            }
        }
    }

    /// The element count an array field's size names.
    fn arr_count(&mut self, sz: &ArrSize, at: Node<'_>) -> R<u32> {
        match sz {
            ArrSize::Lit(n) => Ok(*n),
            ArrSize::Named(name) => {
                let Some((e, _)) = self.global_const(name, at)? else {
                    return self.not_yet(
                        at,
                        format!("the array size `{name}`, which is not a file-scope constant"),
                    );
                };
                match super::globals::fold(&e)
                    .and_then(|v| u32::try_from(v).ok())
                    .filter(|v| *v > 0)
                {
                    Some(n) => Ok(n),
                    None => self.not_yet(
                        at,
                        format!(
                            "the array size `{name}`, which is not a positive integer constant"
                        ),
                    ),
                }
            }
        }
    }

    /// A loop storing zero into every element of a private array (the oracle traps a read of an unwritten one; C leaves it
    /// indeterminate).
    /// A field of CUDA vector type: `float4 v;` is one scalar local per lane; `float4 data[N];` is an allocation of `N * lanes`
    /// scalars held as a pointer whose pointee is the vector (so `data[i]`, `data[i].x` and `((float*)data)[k]` work as for a
    /// local vector array), zero at construction.
    fn new_vector_field(
        &mut self,
        f: &FieldDef,
        path: &str,
        (elem, lanes): (E, u8),
        at: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<Sym> {
        let Some(sz) = f.array.clone() else {
            let mut ls = [0u32; 4];
            for l in ls.iter_mut().take(usize::from(lanes)) {
                *l = self.new_local(elem);
                out.push(Stmt::Let {
                    local: *l,
                    init: self.zero_of(elem, at)?,
                });
            }
            return Ok(Sym::Vector(ls, elem, lanes));
        };
        let n = self.arr_count(&sz, at)?;
        if self.launch.is_none() {
            return self.not_yet(
                at,
                format!("the array field `{path}` without a launch (use `lift_kernel_launched`)"),
            );
        }
        let count = n * u32::from(lanes);
        self.allocs.push(AllocDecl::new(
            path,
            MemorySpace::Private,
            elem,
            Extent::Static(count),
        ));
        let alloc = u32::try_from(self.allocs.len() - 1).unwrap_or(u32::MAX);
        self.zero_fill(alloc, elem, count, at, out)?;
        Ok(Sym::Pointer(super::pointer::PtrVal {
            base: super::pointer::PBase::Alloc(alloc),
            pointee: elem,
            lanes,
            off: ScalarExpr::ConstInt(0, E::I64),
        }))
    }

    pub(super) fn zero_fill(
        &mut self,
        alloc: u32,
        e: E,
        n: u32,
        at: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        let counter = self.new_local(E::I32);
        let store = Stmt::Store {
            place: Place::Alloc {
                alloc,
                index: ScalarExpr::Local(counter),
            },
            value: self.zero_of(e, at)?,
        };
        out.push(Stmt::For {
            local: counter,
            start: ScalarExpr::ConstInt(0, E::I32),
            end: ScalarExpr::ConstInt(i64::from(n), E::I32),
            step: ScalarExpr::ConstInt(1, E::I32),
            unroll: Unroll::None,
            body: Block(vec![store]),
        });
        Ok(())
    }

    /// `S x = { e0, e1, ... };`: one element per field, in order. A scalar field is a fresh local, a nested struct a copy of the
    /// struct value, a pointer field the pointer expression (fixed now). An array field and a short list decline.
    pub(super) fn struct_init(
        &mut self,
        sname: &str,
        label: &str,
        list: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<Rc<StructInst>> {
        let Some(defs) = self.structs.get(sname).cloned() else {
            return self.not_yet(list, format!("the struct `{sname}`"));
        };
        let mut c = list.walk();
        let elems: Vec<_> = list.named_children(&mut c).collect();
        if elems.len() != defs.len() {
            return self.not_yet(list, format!("a brace list of {} element(s) for `{sname}`, which has {} field(s) (every field must be given)", elems.len(), defs.len()));
        }
        let mut fields = Vec::new();
        for (f, e) in defs.iter().zip(elems) {
            let path = format!("{label}.{}", f.name);
            fields.push((f.name.clone(), self.init_field(f, &path, e, out)?));
        }
        Ok(Rc::new(StructInst {
            sname: sname.to_string(),
            fields,
        }))
    }

    fn init_field(&mut self, f: &FieldDef, path: &str, e: Node<'_>, out: &mut Vec<Stmt>) -> R<Sym> {
        if f.pointer {
            let Some(mut p) = self.ptr_expr(e)? else {
                return self.not_yet(e, format!("the pointer field `{path}` initialised from something that is not a pointer expression"));
            };
            let want = match self.ctype(&f.ty) {
                Ok(t) => t,
                Err(why) => return self.not_yet(e, format!("the pointer field `{path}`: {why}")),
            };
            if p.pointee != want {
                return self.not_yet(e, format!("the pointer field `{path}` of type {want:?}* initialised from a {:?}* (cast it)", p.pointee));
            }
            out.append(&mut self.hoisted);
            p.off = self.pin_offset(p.off, out);
            return Ok(Sym::Pointer(p));
        }
        if self.is_struct(&f.ty) && f.array.is_none() {
            let from = self.struct_value(e, &f.ty)?;
            return Ok(Sym::Struct(self.clone_struct(&from, path, e, out)?));
        }
        let (Ok(t), None, None) = (self.ctype(&f.ty), &f.array, &f.unsupported) else {
            return self.not_yet(e, format!("the field `{path}`: only scalar, pointer and struct fields take a brace element"));
        };
        let (v, vt) = self.expr(e)?;
        let v = self.convert(v, vt, t, e)?;
        out.append(&mut self.hoisted);
        let local = self.new_local(t);
        out.push(Stmt::Let { local, init: v });
        Ok(Sym::Scalar(local, t, Fixed::No))
    }

    /// A fresh instance holding `src`'s values: scalars copied into new locals, arrays copied into new allocations, nested
    /// structs cloned, pointers shared (a pointer value is immutable).
    pub(super) fn clone_struct(
        &mut self,
        src: &StructInst,
        label: &str,
        at: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<Rc<StructInst>> {
        let mut fields = Vec::new();
        for (name, s) in &src.fields {
            let path = format!("{label}.{name}");
            let copy = match s {
                Sym::Scalar(l, t, _) => {
                    let local = self.new_local(*t);
                    out.push(Stmt::Let {
                        local,
                        init: ScalarExpr::Local(*l),
                    });
                    Sym::Scalar(local, *t, Fixed::No)
                }
                Sym::Alloc(a, e) => {
                    let Extent::Static(n) = self.allocs[*a as usize].extent else {
                        return Err(self.invalid("a struct array field with a dynamic size"));
                    };
                    self.allocs.push(AllocDecl::new(
                        path,
                        MemorySpace::Private,
                        *e,
                        Extent::Static(n),
                    ));
                    let alloc = u32::try_from(self.allocs.len() - 1).unwrap_or(u32::MAX);
                    self.copy_array(alloc, *a, *e, out)?;
                    Sym::Alloc(alloc, *e)
                }
                Sym::Struct(inner) => Sym::Struct(self.clone_struct(inner, &path, at, out)?),
                Sym::Pointer(p)
                    if p.lanes > 1 && matches!(p.base, super::pointer::PBase::Alloc(_)) =>
                {
                    return self.not_yet(
                        at,
                        format!(
                            "the field `{path}`, an array of vectors, copied (copy its elements)"
                        ),
                    );
                }
                Sym::Pointer(p) => Sym::Pointer(p.clone()),
                // a struct parameter's array is a row of immutable locals (an element is never assigned): the copy shares them
                Sym::ParamArray(ls, e) => Sym::ParamArray(ls.clone(), *e),
                _ => {
                    return self.not_yet(at, format!("the field `{path}`, which cannot be copied"));
                }
            };
            fields.push((name.clone(), copy));
        }
        Ok(Rc::new(StructInst {
            sname: src.sname.clone(),
            fields,
        }))
    }

    /// `S name;`, `S name = other;`, `S a, b;`: instances, copied from another struct value when initialised.
    pub(super) fn struct_declaration(
        &mut self,
        n: Node<'_>,
        sname: &str,
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
            let Some(id) = id.filter(|i| i.kind() == "identifier") else {
                return self.not_yet(d, "a struct declarator that is not a plain name");
            };
            let name = text(id, self.src).to_string();
            let inst = match value {
                None => self.new_struct(sname, &name, d, out)?,
                Some(v) if v.kind() == "initializer_list" => {
                    self.struct_init(sname, &name, v, out)?
                }
                Some(v) => {
                    let from = self.struct_value(v, sname)?;
                    self.clone_struct(&from, &name, v, out)?
                }
            };
            self.scopes
                .last_mut()
                .expect("a scope is open")
                .insert(name, Sym::Struct(inst));
        }
        Ok(())
    }

    /// The struct a value expression names (a struct local, parameter or member), which must be of type `sname`.
    pub(super) fn struct_value(&self, v: Node<'_>, sname: &str) -> R<Rc<StructInst>> {
        let sym = match v.kind() {
            "identifier" => self.lookup(text(v, self.src)),
            "field_expression" => self.resolve_member(v)?,
            _ => None,
        };
        match sym {
            Some(Sym::Struct(i)) if self.shape_of(&i) == self.shape_named(sname) => Ok(i),
            _ => self.not_yet(v, format!("a value of struct `{sname}` that is not a struct local, parameter or member of that type")),
        }
    }

    /// The names of an instance's fields (a cheap structural type check between instance and definition).
    fn shape_of(&self, i: &StructInst) -> Vec<String> {
        i.fields.iter().map(|(n, _)| n.clone()).collect()
    }

    fn shape_named(&self, sname: &str) -> Vec<String> {
        self.structs
            .get(sname)
            .map_or_else(Vec::new, |d| d.iter().map(|f| f.name.clone()).collect())
    }

    /// `dst = src` field by field: scalars assigned, arrays copied element by element, nested structs recursively.
    pub(super) fn copy_struct(
        &mut self,
        dst: &StructInst,
        src: &StructInst,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        for ((_, d), (_, s)) in dst.fields.iter().zip(&src.fields) {
            match (d, s) {
                (Sym::Scalar(dl, ..), Sym::Scalar(sl, ..)) => {
                    out.push(Stmt::Assign {
                        local: *dl,
                        value: ScalarExpr::Local(*sl),
                    });
                }
                (Sym::Alloc(da, e), Sym::Alloc(sa, _)) => self.copy_array(*da, *sa, *e, out)?,
                (Sym::Struct(di), Sym::Struct(si)) => self.copy_struct(di, si, out)?,
                _ => return Err(self.invalid("a struct copy between instances of different shape")),
            }
        }
        Ok(())
    }

    fn copy_array(&mut self, dst: u32, src: u32, elem: E, out: &mut Vec<Stmt>) -> R<()> {
        let Extent::Static(len) = self.allocs[src as usize].extent else {
            return Err(self.invalid("a struct array field with a dynamic size"));
        };
        for k in 0..len {
            let at = |alloc| Place::Alloc {
                alloc,
                index: ScalarExpr::ConstInt(i64::from(k), E::I32),
            };
            let tmp = self.new_local(elem);
            out.push(Stmt::Load {
                local: tmp,
                place: at(src),
            });
            out.push(Stmt::Store {
                place: at(dst),
                value: ScalarExpr::Local(tmp),
            });
        }
        Ok(())
    }

    /// `a.b.c`: the member a field-access chain names, if its root is a struct instance (`None` otherwise, so `threadIdx.x`
    /// and a vector member fall through to their own handling).
    pub(super) fn resolve_member(&self, n: Node<'_>) -> R<Option<Sym>> {
        let (Some(a), Some(f)) = (
            n.child_by_field_name("argument"),
            n.child_by_field_name("field"),
        ) else {
            return Ok(None);
        };
        let base = match a.kind() {
            "identifier" => self.lookup(text(a, self.src)),
            "field_expression" => self.resolve_member(a)?,
            _ => None,
        };
        let Some(Sym::Struct(inst)) = base else {
            return Ok(None);
        };
        let name = text(f, self.src);
        match inst.fields.iter().find(|(n, _)| n == name) {
            Some((_, s)) => Ok(Some(s.clone())),
            None => self.not_yet(
                f,
                format!("the member `{name}`, which the struct does not have"),
            ),
        }
    }

    /// A struct-typed kernel parameter: one `ParamKind::Struct` over the flattened scalar fields, bound as a struct instance
    /// whose fields cannot be assigned.
    pub(super) fn bind_struct_param(&mut self, pname: &str, sname: &str, at: Node<'_>) -> R<Sym> {
        let mut flat = Vec::new();
        let inst = self.param_struct(sname, pname, at, &mut flat)?;
        self.params.push(Param::structure(pname.to_string(), flat));
        Ok(Sym::Struct(inst))
    }

    fn param_struct(
        &mut self,
        sname: &str,
        label: &str,
        at: Node<'_>,
        flat: &mut Vec<StructField>,
    ) -> R<Rc<StructInst>> {
        let Some(defs) = self.structs.get(sname).cloned() else {
            return self.not_yet(at, format!("the struct `{sname}`"));
        };
        let mut fields = Vec::new();
        for f in defs {
            let path = format!("{label}.{}", f.name);
            let sym = if self.is_struct(&f.ty) && f.array.is_none() {
                Sym::Struct(self.param_struct(&f.ty, &path, at, flat)?)
            } else if let (Ok(e), None, None, false) =
                (self.ctype(&f.ty), &f.array, &f.unsupported, f.pointer)
            {
                let local = self.new_local(e);
                flat.push(StructField::new(path, e, local));
                Sym::Scalar(local, e, Fixed::Param)
            } else if let (Ok(e), Some(size), None, false) =
                (self.ctype(&f.ty), &f.array, &f.unsupported, f.pointer)
            {
                self.param_array(&path, e, size, at, flat)?
            } else {
                return self.not_yet(at, format!("the field `{path}` of a struct parameter: only scalar fields, arrays of scalars and nested structs of them are allowed"));
            };
            fields.push((f.name, sym));
        }
        Ok(Rc::new(StructInst {
            sname: sname.to_string(),
            fields,
        }))
    }

    /// An array field of a struct parameter, **flattened** to `n` scalar fields `path[0]` .. `path[n-1]` (option A of
    /// `docs/options-u23b-struct-array-fields.md`: no IR change). The flattened fields lie consecutively with the element's
    /// alignment, which is the C layout of the array byte for byte. An array larger than [`MAX_PARAM_ARRAY`] declines.
    fn param_array(
        &mut self,
        path: &str,
        e: E,
        size: &ArrSize,
        at: Node<'_>,
        flat: &mut Vec<StructField>,
    ) -> R<Sym> {
        let n = self.arr_count(size, at)?;
        if n > MAX_PARAM_ARRAY {
            return self.not_yet(
                at,
                format!(
                    "the array field `{path}` of {n} elements in a struct parameter (at most {MAX_PARAM_ARRAY} are flattened)"
                ),
            );
        }
        let mut locals = Vec::new();
        for k in 0..n {
            let local = self.new_local(e);
            flat.push(StructField::new(format!("{path}[{k}]"), e, local));
            locals.push(local);
        }
        Ok(Sym::ParamArray(locals, e))
    }

    /// `p.v[i]` of a struct parameter's array field: the element as an expression. A constant index is the element itself; a
    /// runtime index is `Select(i == 0, v0, Select(i == 1, v1, ... 0))`. **An index outside `0..n` reads `0`** (C leaves it
    /// undefined; a total result is the same refinement the integer rules make, `docs/int-div-shift-semantics.md`), and a
    /// negative index of an unsigned type is a huge one, so also `0`. `None` when `n` is not such an array.
    pub(super) fn param_array_read(&mut self, n: Node<'_>) -> R<Option<(ScalarExpr, E)>> {
        let (Some(arg), Some(idx)) = (
            n.child_by_field_name("argument"),
            n.child_by_field_name("indices"),
        ) else {
            return Ok(None);
        };
        if arg.kind() != "field_expression" {
            return Ok(None);
        }
        let Some(Sym::ParamArray(locals, e)) = self.resolve_member(arg)? else {
            return Ok(None);
        };
        let Some(i) = idx.named_child(0).filter(|_| idx.named_child_count() == 1) else {
            return self.not_yet(idx, "an index with other than one subscript");
        };
        let (ie, it) = self.expr(i)?;
        if super::int_range(it).is_none() {
            return self.outside(i, "a floating-point index");
        }
        let zero = if super::int_range(e).is_some() {
            ScalarExpr::ConstInt(0, e)
        } else {
            self.convert(ScalarExpr::Const(0.0), self.compute, e, n)?
        };
        if let Some(k) = super::globals::fold(&ie) {
            let at = usize::try_from(k).ok().and_then(|k| locals.get(k));
            return Ok(Some((at.map_or(zero, |l| ScalarExpr::Local(*l)), e)));
        }
        // the index is read once per rung: bind a compound one to a local first
        let ie = if matches!(ie, ScalarExpr::Local(_)) {
            ie
        } else {
            let t = self.new_local(it);
            self.hoisted.push(Stmt::Let { local: t, init: ie });
            ScalarExpr::Local(t)
        };
        let mut ladder = zero;
        for (k, l) in locals.iter().enumerate().rev() {
            let hit = ScalarExpr::Binary(
                unpopped::ir::BinaryOp::CmpEq,
                Box::new(ie.clone()),
                Box::new(ScalarExpr::ConstInt(
                    i64::try_from(k).unwrap_or(i64::MAX),
                    it,
                )),
            );
            ladder = ScalarExpr::Select(
                Box::new(hit),
                Box::new(ScalarExpr::Local(*l)),
                Box::new(ladder),
            );
        }
        Ok(Some((ladder, e)))
    }

    /// `s.x = e` and `s.x += e`: the member's scalar local.
    pub(super) fn assign_struct_member(
        &mut self,
        m: Sym,
        lhs: Node<'_>,
        op: &str,
        rhs: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        let Sym::Scalar(local, ty, fixed) = m else {
            return self.not_yet(lhs, "an assignment to a struct or array member as a whole (assign its scalar members, or an array element)");
        };
        if fixed != Fixed::No {
            return self.not_yet(lhs, "an assignment to a member of a struct parameter");
        }
        let (r, rt) = self.expr(rhs)?;
        let value = self.compound(ScalarExpr::Local(local), ty, op, (r, rt), lhs)?;
        self.flush(out);
        out.push(Stmt::Assign { local, value });
        Ok(())
    }

    /// `a = b` of two struct values of one type: a field-by-field copy.
    pub(super) fn assign_struct(
        &mut self,
        dst: &StructInst,
        op: &str,
        rhs: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        if !op.is_empty() {
            return self.not_yet(rhs, "a compound assignment to a whole struct");
        }
        let sname = self
            .structs
            .iter()
            .find(|(_, d)| {
                d.iter()
                    .map(|f| f.name.as_str())
                    .eq(dst.fields.iter().map(|(n, _)| n.as_str()))
            })
            .map(|(n, _)| n.clone());
        let Some(sname) = sname else {
            return self.not_yet(rhs, "an assignment to a struct of an unknown type");
        };
        let src = self.struct_value(rhs, &sname)?;
        self.copy_struct(dst, &src, out)
    }
}
