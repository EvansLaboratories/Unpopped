// SPDX-License-Identifier: MIT OR Apache-2.0
//! A struct read through a pointer into a byte buffer, with the layout the CALLER stated (ROADMAP U38d,
//! `docs/options-u38-struct-buffers.md`; `LiftContext::with_struct_layout`). `(const block_q4_0*)vx` is a byte pointer whose stride
//! is the layout's size; `x[i].d` and `x[i].qs[k]` read the byte-offset place `i * size + offset(field) + k * width`. **The walker
//! derives no layout**: a struct with none supplied, a field the layout lacks, and a struct read whole all decline by name.

use super::pointer::{PBase, PtrVal, byte_size};
use super::{R, Walker, text};
use tree_sitter::Node;
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Place, Stmt};
use unpopped_vocab::ElementKind as E;

impl<'s, 't> Walker<'s, 't> {
    /// The name of the struct with a supplied layout that the type text `t` (qualifiers and `*` allowed) names, if any.
    pub(super) fn layout_name(&self, t: &str) -> Option<String> {
        let name = t
            .split_whitespace()
            .filter(|w| {
                !matches!(
                    *w,
                    "const" | "volatile" | "struct" | "__restrict__" | "restrict"
                )
            })
            .collect::<Vec<_>>()
            .join("")
            .trim_end_matches(['*', '&'])
            .to_string();
        self.struct_layouts.contains_key(&name).then_some(name)
    }

    /// The stride of a pointer to the laid-out struct `name`, in bytes.
    pub(super) fn layout_size(&self, name: &str) -> i64 {
        self.struct_layouts
            .get(name)
            .map_or(0, |l| i64::from(l.size))
    }

    /// `p[i].f` or `p->f` (a scalar field) and `p[i].f[k]` (an array field) of a pointer to a laid-out struct, as a load. `None`
    /// when `n` is not a member access through such a pointer.
    pub(super) fn layout_member(&mut self, n: Node<'_>) -> R<Option<(ScalarExpr, E)>> {
        let (fe, index) = match n.kind() {
            "subscript_expression" => {
                let (Some(a), Some(idx)) = (
                    n.child_by_field_name("argument"),
                    n.child_by_field_name("indices")
                        .and_then(|i| i.named_child(0)),
                ) else {
                    return Ok(None);
                };
                if a.kind() != "field_expression" {
                    return Ok(None);
                }
                (a, Some(idx))
            }
            "field_expression" => (n, None),
            _ => return Ok(None),
        };
        let (Some(obj), Some(field)) = (
            fe.child_by_field_name("argument"),
            fe.child_by_field_name("field"),
        ) else {
            return Ok(None);
        };
        let Some((p, element)) = self.struct_object(obj)? else {
            return Ok(None);
        };
        self.load_field(n, &p, element, text(field, self.src), index)
            .map(Some)
    }

    /// The pointer a member access reads through, and the element index when the object is `x[i]`; `None` if the object is not a
    /// pointer to a laid-out struct.
    fn struct_object<'n>(&mut self, obj: Node<'n>) -> R<Option<(PtrVal, Option<Node<'n>>)>> {
        let (base, element) = if obj.kind() == "subscript_expression" {
            let (Some(a), Some(i)) = (
                obj.child_by_field_name("argument"),
                obj.child_by_field_name("indices")
                    .and_then(|i| i.named_child(0)),
            ) else {
                return Ok(None);
            };
            (a, Some(i))
        } else {
            (obj, None)
        };
        Ok(self
            .ptr_expr(base)?
            .filter(|p| p.strukt.is_some())
            .map(|p| (p, element)))
    }

    fn load_field(
        &mut self,
        n: Node<'_>,
        p: &PtrVal,
        element: Option<Node<'_>>,
        field: &str,
        index: Option<Node<'_>>,
    ) -> R<(ScalarExpr, E)> {
        let sname = p.strukt.clone().unwrap_or_default();
        let layout = self
            .struct_layouts
            .get(&sname)
            .cloned()
            .expect("a laid-out struct has its layout");
        let Some(f) = layout.fields.iter().find(|f| f.name == field) else {
            return self.not_yet(
                n,
                format!("the field `{field}`, which the layout of `{sname}` does not give"),
            );
        };
        let elem = match self.ctype(&f.elem) {
            Ok(e) => e,
            Err(why) => return self.not_yet(n, why),
        };
        let width = byte_size(elem).unwrap_or(1);
        let mut off = p.off.clone();
        if let Some(e) = element {
            off = ScalarExpr::Add(
                Box::new(off),
                Box::new(self.scaled_by(e, i64::from(layout.size), n)?),
            );
        }
        off = ScalarExpr::Add(
            Box::new(off),
            Box::new(ScalarExpr::ConstInt(i64::from(f.offset), E::I64)),
        );
        match (f.count, index) {
            (1, None) => {}
            (c, Some(k)) if c > 1 => {
                off = ScalarExpr::Add(Box::new(off), Box::new(self.scaled_by(k, width, n)?));
            }
            (1, Some(_)) => {
                return self.not_yet(
                    n,
                    format!("the scalar field `{field}` of `{sname}` subscripted"),
                );
            }
            _ => {
                return self.not_yet(
                    n,
                    format!("the array field `{field}` of `{sname}` used as a value (index it)"),
                );
            }
        }
        let place = match p.base {
            PBase::Buffer(buffer) => Place::Raw {
                buffer,
                byte_offset: off,
                elem,
            },
            PBase::Alloc(alloc) => Place::AllocRaw {
                alloc,
                byte_offset: off,
                elem,
            },
        };
        let local = self.new_local(elem);
        self.hoisted.push(Stmt::Load { local, place });
        Ok((ScalarExpr::Local(local), elem))
    }

    /// `const S* x = <pointer expression>;` for a laid-out struct `S`: a pointer local with the struct's stride.
    pub(super) fn struct_pointer_declaration(
        &mut self,
        n: Node<'_>,
        sname: &str,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        let mut c = n.walk();
        let decls: Vec<_> = n.children_by_field_name("declarator", &mut c).collect();
        for d in decls {
            let (id, value) = (
                d.child_by_field_name("declarator"),
                d.child_by_field_name("value"),
            );
            let id = id
                .filter(|i| i.kind() == "pointer_declarator")
                .and_then(|i| i.child_by_field_name("declarator"))
                .filter(|i| i.kind() == "identifier");
            let (Some(id), Some(value)) = (id, value) else {
                return self.not_yet(
                    d,
                    "a struct pointer that is not `S* name = <pointer expression>;`",
                );
            };
            let Some(mut p) = self.ptr_expr(value)? else {
                return self.not_yet(
                    value,
                    "a struct pointer initialised from something that is not a pointer expression",
                );
            };
            (p.pointee, p.lanes, p.strukt) = (E::U8, 1, Some(sname.to_string()));
            self.flush(out);
            p.off = self.pin_offset(p.off, out);
            self.scopes
                .last_mut()
                .expect("a scope is open")
                .insert(text(id, self.src).to_string(), super::Sym::Pointer(p));
        }
        Ok(())
    }
}
