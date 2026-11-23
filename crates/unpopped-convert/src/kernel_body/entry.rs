// SPDX-License-Identifier: MIT OR Apache-2.0
//! An entry point read through the frontend's [`EntryModel`](crate::EntryModel) (ROADMAP U19 census slice 4,
//! `docs/options-u19-slang-entry-model.md`): the file-scope resources become the kernel's parameters in slot order, and the entry's
//! semantic parameters become locals computed from the execution ids. Everything here maps to IR that exists.

use super::{PShape, R, Sym, Walker, text};
use crate::{EntryModel, ResourceKind, Semantic};
use tree_sitter::Node;
use unpopped::exec::ExecId;
use unpopped::ir::ScalarExpr;
use unpopped::kernel::Stmt;
use unpopped_vocab::ElementKind as E;

fn exec(id: fn(u8) -> ExecId, axis: u8) -> ScalarExpr {
    ScalarExpr::Exec(id(axis))
}

/// The value of a semantic along one axis (`U32`).
fn axis_value(s: Semantic, axis: u8, block: [u32; 3]) -> ScalarExpr {
    match s {
        Semantic::DispatchThreadId => ScalarExpr::Add(
            Box::new(ScalarExpr::Mul(
                Box::new(exec(ExecId::BlockIdx, axis)),
                Box::new(ScalarExpr::Exec(ExecId::BlockDim(axis))),
            )),
            Box::new(exec(ExecId::ThreadIdx, axis)),
        ),
        Semantic::GroupId => exec(ExecId::BlockIdx, axis),
        Semantic::GroupThreadId => exec(ExecId::ThreadIdx, axis),
        // `x + y * bx + z * bx * by`: the linear thread index in the block (a `uint`, so only axis 0 is asked for)
        _ => {
            let (bx, by) = (i64::from(block[0]), i64::from(block[1]));
            let term = |a: u8, scale: i64| {
                ScalarExpr::Mul(
                    Box::new(exec(ExecId::ThreadIdx, a)),
                    Box::new(ScalarExpr::ConstInt(scale, E::U32)),
                )
            };
            ScalarExpr::Add(
                Box::new(ScalarExpr::Add(
                    Box::new(exec(ExecId::ThreadIdx, 0)),
                    Box::new(term(1, bx)),
                )),
                Box::new(term(2, bx * by)),
            )
        }
    }
}

impl<'s, 't> Walker<'s, 't> {
    /// The model's resources as the kernel's parameters, in slot order. The slots must be dense from 0 with no repeat.
    pub(super) fn entry_resources(&self, m: &EntryModel, at: Node<'_>) -> R<Vec<(String, PShape)>> {
        let mut rs: Vec<&crate::Resource> = m.resources.iter().collect();
        rs.sort_by_key(|r| r.slot);
        let slots: Vec<u32> = rs.iter().map(|r| r.slot).collect();
        if slots
            .iter()
            .enumerate()
            .any(|(i, s)| u32::try_from(i) != Ok(*s))
        {
            return self.not_yet(
                at,
                format!("resource binding slots {slots:?} (the referenced resources must be dense from 0, each slot once)"),
            );
        }
        rs.into_iter()
            .map(|r| match r.kind {
                ResourceKind::StructuredRO | ResourceKind::StructuredRW => {
                    match self.ctype(&r.elem) {
                        Ok(e) => Ok((
                            r.name.clone(),
                            PShape::Buffer(e, r.kind == ResourceKind::StructuredRO),
                        )),
                        Err(why) => self.not_yet(at, format!("the buffer `{}`: {why}", r.name)),
                    }
                }
                ResourceKind::ConstantBuffer if self.is_struct(&r.elem) => {
                    Ok((r.name.clone(), PShape::Struct(r.elem.clone())))
                }
                ResourceKind::ConstantBuffer => self.not_yet(
                    at,
                    format!(
                        "the constant buffer `{}` of the unknown struct `{}`",
                        r.name, r.elem
                    ),
                ),
            })
            .collect()
    }

    /// Every entry parameter must be a semantic one: bind each as a local (a `uint3` as three), computed in the returned prologue.
    pub(super) fn bind_entry_semantics(
        &mut self,
        m: &EntryModel,
        plist: Node<'_>,
        block: [u32; 3],
    ) -> R<Vec<Stmt>> {
        let mut prologue = Vec::new();
        let mut c = plist.walk();
        for p in plist.named_children(&mut c) {
            if p.kind() == "comment" {
                continue;
            }
            let Some(name) = param_name(p, self.src) else {
                return self.not_yet(p, "an entry parameter without a name");
            };
            let Some((_, sem)) = m.semantics.iter().find(|(n, _)| *n == name) else {
                return self.not_yet(
                    p,
                    format!("the entry parameter `{name}`, which carries no execution-id semantic"),
                );
            };
            let ty = p
                .child_by_field_name("type")
                .map_or(String::new(), |t| text(t, self.src).trim().to_string());
            let sym = match (ty.as_str(), sem) {
                (
                    "uint3",
                    Semantic::DispatchThreadId | Semantic::GroupId | Semantic::GroupThreadId,
                ) => {
                    let mut ls = [0u32; 4];
                    for axis in 0u8..3 {
                        let l = self.new_local(E::U32);
                        prologue.push(Stmt::Let {
                            local: l,
                            init: axis_value(*sem, axis, block),
                        });
                        ls[usize::from(axis)] = l;
                    }
                    ls[3] = ls[2];
                    Sym::Vector(ls, E::U32, 3)
                }
                ("uint", Semantic::GroupIndex) => {
                    let l = self.new_local(E::U32);
                    prologue.push(Stmt::Let {
                        local: l,
                        init: axis_value(*sem, 0, block),
                    });
                    Sym::Scalar(l, E::U32, super::Fixed::Param)
                }
                (other, s) => {
                    return self.not_yet(
                        p,
                        format!("the entry parameter `{name}` of type `{other}` with the semantic {s:?} (a `uint3` takes an id, a `uint` the group index)"),
                    );
                }
            };
            self.scopes[0].insert(name, sym);
        }
        Ok(prologue)
    }
}

/// The name of a parameter declaration: its declarator's identifier, however the grammar nests the semantic around it.
fn param_name(p: Node<'_>, src: &str) -> Option<String> {
    fn first_identifier<'a>(n: Node<'a>) -> Option<Node<'a>> {
        if n.kind() == "identifier" {
            return Some(n);
        }
        let mut c = n.walk();
        n.named_children(&mut c).find_map(first_identifier)
    }
    let d = p.child_by_field_name("declarator").or(Some(p))?;
    first_identifier(d).map(|i| text(i, src).to_string())
}
