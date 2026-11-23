// SPDX-License-Identifier: MIT OR Apache-2.0
//! Vectors in memory (ROADMAP U19 slice 5, `docs/options-u19-slice5-decode-census.md`): a pointer whose pointee is a CUDA vector
//! type (`(const int4*)p`, `(uint4*)q`) reads and writes **lane by lane**, each lane its own scalar raw access at
//! `byte offset + lane * sizeof(lane type)`, matching the scalarized vector locals (slice 4b): `int4 t = ((const int4*)p)[0];`
//! is four `Load`s, `*((uint4*)q) = make_uint4(...)` is four `Store`s. Only whole-vector accesses lift; a lane read through
//! the pointer (`->x`) and arithmetic on vector values are not C.

use super::pointer::{PtrVal, byte_size};
use super::{R, Sym, Walker, text, vector_type};
use tree_sitter::Node;
use unpopped::exec::{AllocDecl, Extent, MemorySpace};
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

    /// The place of lane `member` (`x`, `y`, ...) of the vector element `a` names (`d[i]`, `p[i]`, `*p` of a pointer to a vector),
    /// and its element type; `None` if `a` is not a vector in memory.
    pub(super) fn vector_lane_place(
        &mut self,
        a: Node<'_>,
        member: Node<'_>,
    ) -> R<Option<(Place, E)>> {
        let Some((p, index)) = self.vector_target(a)? else {
            return Ok(None);
        };
        let Some(lane) = super::lane_of(text(member, self.src), p.lanes) else {
            return self.not_yet(
                member,
                format!(
                    "the member `{}` of a {}-lane vector",
                    text(member, self.src),
                    p.lanes
                ),
            );
        };
        let place = self
            .lane_places(&p, index, a)?
            .swap_remove(usize::from(lane));
        Ok(Some((place, p.pointee)))
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

    /// Whether the source takes the address of `name` anywhere (`&name`, not followed by more identifier characters).
    /// [`Walker::address_taken`] for an address used by anything but `memcpy(&name, ...)`, which a scalar local serves without living in
    /// memory (`memcpy.rs` assembles the bytes into the local).
    pub(super) fn address_escapes(&self, name: &str) -> bool {
        let pat = format!("&{name}");
        self.src.match_indices(&pat).any(|(i, _)| {
            let after = self.src[i + pat.len()..].chars().next();
            let before = self.src[..i].chars().next_back();
            !after.is_some_and(|c| c.is_alphanumeric() || c == '_')
                && before != Some('&')
                && !self.src[..i].trim_end().ends_with("memcpy(")
        })
    }

    pub(super) fn address_taken(&self, name: &str) -> bool {
        let pat = format!("&{name}");
        self.src.match_indices(&pat).any(|(i, _)| {
            let after = self.src[i + pat.len()..].chars().next();
            let before = self.src[..i].chars().next_back();
            !after.is_some_and(|c| c.is_alphanumeric() || c == '_') && before != Some('&')
        })
    }

    /// A vector local whose address is taken: a `Private` allocation of `lanes` elements holding the initial lanes.
    pub(super) fn vector_alloc_declaration(
        &mut self,
        name: &str,
        (elem, lanes): (E, u8),
        init: Vec<ScalarExpr>,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        self.allocs.push(AllocDecl::new(
            name.to_string(),
            MemorySpace::Private,
            elem,
            Extent::Static(u32::from(lanes)),
        ));
        let alloc = u32::try_from(self.allocs.len() - 1).unwrap_or(u32::MAX);
        let size = byte_size(elem).unwrap_or(1);
        for (k, value) in init.into_iter().enumerate() {
            out.push(Stmt::Store {
                place: Place::AllocRaw {
                    alloc,
                    byte_offset: ScalarExpr::ConstInt(k as i64 * size, E::I64),
                    elem,
                },
                value,
            });
        }
        self.scopes
            .last_mut()
            .expect("a scope is open")
            .insert(name.to_string(), Sym::VecAlloc(alloc, elem, lanes));
        Ok(())
    }

    /// The lanes of a memory-backed vector local, each loaded into a fresh local.
    pub(super) fn vecalloc_lanes(&mut self, alloc: u32, (elem, lanes): (E, u8)) -> Vec<ScalarExpr> {
        let size = byte_size(elem).unwrap_or(1);
        (0..i64::from(lanes))
            .map(|k| {
                let local = self.new_local(elem);
                self.hoisted.push(Stmt::Load {
                    local,
                    place: Place::AllocRaw {
                        alloc,
                        byte_offset: ScalarExpr::ConstInt(k * size, E::I64),
                        elem,
                    },
                });
                ScalarExpr::Local(local)
            })
            .collect()
    }

    /// `__half22float2(h)` and `__float22half2_rn(f)`: each lane converted, the vector shape kept.
    pub(super) fn vector_convert(
        &mut self,
        n: Node<'_>,
        (elem, lanes): (E, u8),
    ) -> R<Option<Vec<ScalarExpr>>> {
        if n.kind() != "call_expression" {
            return Ok(None);
        }
        let callee = n.child_by_field_name("function").map(|f| text(f, self.src));
        let from = match (callee, elem) {
            (Some("__half22float2"), E::F32) => E::F16,
            (Some("__float22half2_rn"), E::F16) => E::F32,
            _ => return Ok(None),
        };
        let args = self.call_args(n);
        let [a] = args.as_slice() else {
            return self.not_yet(n, "a vector conversion with other than one argument");
        };
        debug_assert!(vector_type("half2").is_some());
        let lanes_in = self.vector_value(*a, (from, lanes))?;
        let mut out = Vec::new();
        for v in lanes_in {
            out.push(self.convert(v, from, elem, *a)?);
        }
        Ok(Some(out))
    }
}
