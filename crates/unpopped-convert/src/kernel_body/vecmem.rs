// SPDX-License-Identifier: MIT OR Apache-2.0
//! Vectors in memory (ROADMAP U19 slice 5, `docs/options-u19-slice5-decode-census.md`): a pointer whose pointee is a CUDA vector
//! type (`(const int4*)p`, `(uint4*)q`) reads and writes **lane by lane**, each lane its own scalar raw access at
//! `byte offset + lane * sizeof(lane type)`, matching the scalarized vector locals (slice 4b): `int4 t = ((const int4*)p)[0];`
//! is four `Load`s, `*((uint4*)q) = make_uint4(...)` is four `Store`s. Only whole-vector accesses lift; a lane read through
//! the pointer (`->x`) and arithmetic on vector values are not C.

use super::pointer::{PtrVal, byte_size};
use super::{R, Walker, text};
use tree_sitter::Node;
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Place, Stmt};
use unpopped_vocab::ElementKind as E;

impl<'s, 't> Walker<'s, 't> {
    /// `p[i]` or `*p` where `p` points at a vector type: the pointer and the index node (`None` for `*p`).
    fn vector_target<'n>(&mut self, n: Node<'n>) -> R<Option<(PtrVal, Option<Node<'n>>)>> {
        let (base, index) = match n.kind() {
            "subscript_expression" => {
                let idx = n
                    .child_by_field_name("indices")
                    .and_then(|i| i.named_child(0));
                (n.child_by_field_name("argument"), idx)
            }
            "pointer_expression"
                if n.child_by_field_name("operator")
                    .is_some_and(|o| text(o, self.src) == "*") =>
            {
                (n.child_by_field_name("argument"), None)
            }
            _ => return Ok(None),
        };
        let Some(base) = base else {
            return Ok(None);
        };
        Ok(self
            .ptr_expr(base)?
            .filter(|p| p.lanes > 1)
            .map(|p| (p, index)))
    }

    fn lane_places(&mut self, p: &PtrVal, index: Option<Node<'_>>, at: Node<'_>) -> R<Vec<Place>> {
        let Some(size) = byte_size(p.pointee) else {
            return self.not_yet(
                at,
                format!("a vector of {:?} (no whole-byte size)", p.pointee),
            );
        };
        let mut base = p.off.clone();
        if let Some(i) = index {
            let step = self.scaled_by(i, size * i64::from(p.lanes), at)?;
            base = ScalarExpr::Add(Box::new(base), Box::new(step));
        }
        Ok((0..i64::from(p.lanes))
            .map(|k| {
                let byte_offset = ScalarExpr::Add(
                    Box::new(base.clone()),
                    Box::new(ScalarExpr::ConstInt(k * size, E::I64)),
                );
                match p.base {
                    super::pointer::PBase::Buffer(buffer) => Place::Raw {
                        buffer,
                        byte_offset,
                        elem: p.pointee,
                    },
                    super::pointer::PBase::Alloc(alloc) => Place::AllocRaw {
                        alloc,
                        byte_offset,
                        elem: p.pointee,
                    },
                }
            })
            .collect())
    }

    /// A vector read through a pointer, as its lane expressions (the loads are hoisted); `None` if `n` is not one.
    pub(super) fn vector_load(
        &mut self,
        n: Node<'_>,
        (elem, lanes): (E, u8),
    ) -> R<Option<Vec<ScalarExpr>>> {
        let Some((p, index)) = self.vector_target(n)? else {
            return Ok(None);
        };
        if (p.pointee, p.lanes) != (elem, lanes) {
            return self.not_yet(
                n,
                format!(
                    "a read of a {}-lane {:?} vector as a {lanes}-lane {elem:?} vector",
                    p.lanes, p.pointee
                ),
            );
        }
        let mut out = Vec::new();
        for place in self.lane_places(&p, index, n)? {
            let local = self.new_local(elem);
            self.hoisted.push(Stmt::Load { local, place });
            out.push(ScalarExpr::Local(local));
        }
        Ok(Some(out))
    }

    /// `*((T4*)p) = v;` and `((T4*)p)[i] = v;`: one `Store` per lane. `false` if `lhs` is not a vector in memory.
    pub(super) fn vector_store(
        &mut self,
        lhs: Node<'_>,
        op: &str,
        rhs: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<bool> {
        let Some((p, index)) = self.vector_target(lhs)? else {
            return Ok(false);
        };
        if !op.is_empty() {
            return self.not_yet(
                lhs,
                "a compound assignment to a vector in memory (operators on vector types are not C)",
            );
        }
        let values = self.vector_value(rhs, (p.pointee, p.lanes))?;
        let places = self.lane_places(&p, index, lhs)?;
        self.flush(out);
        for (place, value) in places.into_iter().zip(values) {
            out.push(Stmt::Store { place, value });
        }
        Ok(true)
    }
}
