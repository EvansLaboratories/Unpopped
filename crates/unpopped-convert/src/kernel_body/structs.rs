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

/// One field of a struct definition, as written.
#[derive(Clone, Debug)]
pub(super) struct FieldDef {
    pub(super) name: String,
    /// The type text (`float`, `uint32_t`, another struct's name).
    pub(super) ty: String,
    /// The element count of an array field.
    pub(super) array: Option<u32>,
    /// Why this field cannot be scalarized, if it cannot (a pointer, a bit-field, ...).
    pub(super) unsupported: Option<String>,
}

/// An instance of a struct: its fields, each a scalar local, a private array or a nested instance.
pub(super) struct StructInst {
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
        unsupported: Some(why.to_string()),
    };
    match d.kind() {
        "field_identifier" => FieldDef {
            name: text(d, src).to_string(),
            ty: ty.to_string(),
            array: None,
            unsupported: None,
        },
        "array_declarator" => {
            let id = d
                .child_by_field_name("declarator")
                .filter(|i| i.kind() == "field_identifier");
            let n = d
                .child_by_field_name("size")
                .and_then(|s| text(s, src).trim().parse::<u32>().ok())
                .filter(|n| *n > 0);
            match (id, n) {
                (Some(id), Some(n)) => FieldDef {
                    name: text(id, src).to_string(),
                    ty: ty.to_string(),
                    array: Some(n),
                    unsupported: None,
                },
                _ => bad(
                    text(d, src),
                    "an array field that is not `T name[N]` with a positive literal `N`",
                ),
            }
        }
        "pointer_declarator" => {
            let id = d
                .child_by_field_name("declarator")
                .map_or("?", |i| text(i, src));
            bad(
                id,
                "a pointer field (a pointer inside a struct is not modelled)",
            )
        }
        other => bad(text(d, src), &format!("a `{other}` field")),
    }
}

impl<'s, 't> Walker<'s, 't> {
    /// Whether `name` is a struct defined in this source.
    pub(super) fn is_struct(&self, name: &str) -> bool {
        self.structs.contains_key(name)
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
            fields.push((f.name.clone(), self.new_field(&f, &path, at, out)?));
        }
        Ok(Rc::new(StructInst { fields }))
    }

    fn new_field(&mut self, f: &FieldDef, path: &str, at: Node<'_>, out: &mut Vec<Stmt>) -> R<Sym> {
        if self.is_struct(&f.ty) {
            if f.array.is_some() {
                return self.not_yet(at, format!("an array of structs (`{path}`)"));
            }
            return Ok(Sym::Struct(self.new_struct(&f.ty, path, at, out)?));
        }
        let e = match super::c_type(&f.ty) {
            Ok(e) => e,
            Err(why) => return self.not_yet(at, format!("the field `{path}`: {why}")),
        };
        match f.array {
            None => {
                let local = self.new_local(e);
                out.push(Stmt::Let {
                    local,
                    init: self.zero_of(e, at)?,
                });
                Ok(Sym::Scalar(local, e, Fixed::No))
            }
            Some(n) => {
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

    /// A loop storing zero into every element of a private array (the oracle traps a read of an unwritten one; C leaves it
    /// indeterminate).
    fn zero_fill(&mut self, alloc: u32, e: E, n: u32, at: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
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
            let inst = self.new_struct(sname, &name, d, out)?;
            if let Some(v) = value {
                let from = self.struct_value(v, sname)?;
                self.copy_struct(&inst, &from, out)?;
            }
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
            } else if let (Ok(e), None, None) = (super::c_type(&f.ty), f.array, &f.unsupported) {
                let local = self.new_local(e);
                flat.push(StructField::new(path, e, local));
                Sym::Scalar(local, e, Fixed::Param)
            } else {
                return self.not_yet(at, format!("the field `{path}` of a struct parameter: only scalar fields and nested structs of them are allowed"));
            };
            fields.push((f.name, sym));
        }
        Ok(Rc::new(StructInst { fields }))
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
