// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Symbolic pointers** (ROADMAP U19 slice 4c-4; pm-kernels ruling Q6, 2026-11-21, `docs/options-u19-slice5-decode-census.md`):
//! the IR has no pointer value (U9c option C was declined), so the walker carries a pointer as a *base* (a buffer parameter or an
//! allocation), the pointee type and a **byte offset** (an `I64` expression), and lowers every dereference to
//! `Place::Raw`/`Place::AllocRaw`. A pointer that must survive as a value (stored to memory, compared, returned from a call that
//! is not inlined, reassigned) is not representable, so it declines.
//!
//! A pointer **local** is fixed when it is declared: its offset is evaluated into a fresh immutable local then, so a later
//! change to a variable the expression mentioned cannot change where the pointer points. Atomics through a pointer decline (the
//! IR's atomics take no raw place).

use super::{R, Sym, Walker, text};
use tree_sitter::Node;
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Place, Stmt};
use unpopped_vocab::ElementKind as E;

/// What a pointer points into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PBase {
    /// A buffer parameter (index into `KernelFn::params`).
    Buffer(u32),
    /// An allocation (index into `KernelFn::allocs`).
    Alloc(u32),
}

/// A pointer: where it points into, the type it is read as, and its byte offset from the base's first element.
#[derive(Clone, Debug)]
pub(super) struct PtrVal {
    pub(super) base: PBase,
    pub(super) pointee: E,
    /// `1` for a scalar pointee; `2` or `4` when the pointee is a CUDA vector type (`int4`): then `pointee` is the lane type.
    pub(super) lanes: u8,
    /// Bytes from the start of the base; an `I64` expression.
    pub(super) off: ScalarExpr,
}

/// The byte size of a pointee type (whole bytes only), or `None`.
pub(super) fn byte_size(t: E) -> Option<i64> {
    Some(match t {
        E::I8 | E::U8 | E::Bool | E::Fp8E4M3FN | E::Fp8E5M2 => 1,
        E::F16 | E::Bf16 | E::I16 | E::U16 => 2,
        E::F32 | E::F32Strict | E::I32 | E::U32 => 4,
        E::F64 | E::I64 | E::U64 => 8,
        _ => return None,
    })
}

impl<'s, 't> Walker<'s, 't> {
    /// `Some` if `n` is a pointer expression this walker models (a pointer name, `p + i`, `p - i`, `i + p`, `&p[i]`, a cast to
    /// a pointer type, a parenthesis of one); `None` if it is not a pointer expression; a decline if it is one it cannot model.
    pub(super) fn ptr_expr(&mut self, n: Node<'_>) -> R<Option<PtrVal>> {
        match n.kind() {
            "identifier" => Ok(self.pointer_named(text(n, self.src))),
            "parenthesized_expression" => match n.named_child(0) {
                Some(i) => self.ptr_expr(i),
                None => Ok(None),
            },
            "field_expression" => Ok(match self.resolve_member(n)? {
                Some(Sym::Pointer(p)) => Some(p),
                // an array field of a struct is a private allocation: the member decays to a pointer to it
                Some(Sym::Alloc(i, e)) => Some(PtrVal {
                    base: PBase::Alloc(i),
                    pointee: e,
                    lanes: 1,
                    off: ScalarExpr::ConstInt(0, E::I64),
                }),
                _ => None,
            }),
            "cast_expression" => self.ptr_cast(n),
            "call_expression" => self.ptr_reinterpret(n),
            "binary_expression" => self.ptr_arith(n),
            "pointer_expression" => self.ptr_address_of(n),
            _ => Ok(None),
        }
    }

    fn pointer_named(&self, name: &str) -> Option<PtrVal> {
        let zero = ScalarExpr::ConstInt(0, E::I64);
        match self.lookup(name)? {
            Sym::Buffer(i, e) => Some(PtrVal {
                base: PBase::Buffer(i),
                pointee: e,
                lanes: 1,
                off: zero,
            }),
            Sym::Alloc(i, e) => Some(PtrVal {
                base: PBase::Alloc(i),
                pointee: e,
                lanes: 1,
                off: zero,
            }),
            Sym::Pointer(p) => Some(p),
            _ => None,
        }
    }

    /// `reinterpret_cast<T*>(p)`: the same place, read as `T` (as `(T*)p`).
    fn ptr_reinterpret(&mut self, n: Node<'_>) -> R<Option<PtrVal>> {
        let f = n
            .child_by_field_name("function")
            .map_or("", |f| text(f, self.src));
        // `static_cast<T*>(p)` between pointer types (a `void*` to `T*` is the usual one) reads the same place as `reinterpret_cast`
        let Some(inner) = f
            .strip_prefix("reinterpret_cast<")
            .or_else(|| f.strip_prefix("static_cast<"))
            .and_then(|r| r.strip_suffix('>'))
            .map(|t| t.trim().trim_start_matches("const ").trim())
        else {
            return Ok(None);
        };
        let Some(tname) = inner.strip_suffix('*').map(str::trim) else {
            return Ok(None);
        };
        let args = self.call_args(n);
        let [a] = args.as_slice() else {
            return Ok(None);
        };
        let Some(mut p) = self.ptr_expr(*a)? else {
            return self.not_yet(n, "a `reinterpret_cast` of something that is not a pointer");
        };
        if let Some((elem, lanes)) = super::vector_type(tname) {
            (p.pointee, p.lanes) = (elem, lanes);
        } else {
            match self.ctype(tname) {
                Ok(e) => (p.pointee, p.lanes) = (e, 1),
                Err(why) => {
                    return self.not_yet(n, format!("a `reinterpret_cast` to `{tname}*`: {why}"));
                }
            }
        }
        Ok(Some(p))
    }

    /// `(T*)p`: the same place, read as `T`.
    fn ptr_cast(&mut self, n: Node<'_>) -> R<Option<PtrVal>> {
        let (Some(ty), Some(v)) = (
            n.child_by_field_name("type"),
            n.child_by_field_name("value"),
        ) else {
            return Ok(None);
        };
        let is_ptr = ty
            .child_by_field_name("declarator")
            .is_some_and(|d| d.kind().contains("pointer"));
        if !is_ptr {
            return Ok(None);
        }
        let Some(mut p) = self.ptr_expr(v)? else {
            return self.not_yet(
                n,
                "a cast to a pointer type of something that is not a pointer",
            );
        };
        let tname = ty.child_by_field_name("type").map(|t| {
            text(t, self.src)
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        });
        if let Some((elem, lanes)) = tname.as_deref().and_then(super::vector_type) {
            (p.pointee, p.lanes) = (elem, lanes);
        } else {
            p.pointee = self.type_of(ty.child_by_field_name("type"), ty)?;
            p.lanes = 1;
        }
        Ok(Some(p))
    }

    /// `p + i`, `i + p`, `p - i`.
    fn ptr_arith(&mut self, n: Node<'_>) -> R<Option<PtrVal>> {
        let (Some(l), Some(r), Some(op)) = (
            n.child_by_field_name("left"),
            n.child_by_field_name("right"),
            n.child_by_field_name("operator"),
        ) else {
            return Ok(None);
        };
        let op = text(op, self.src);
        if op != "+" && op != "-" {
            return Ok(None);
        }
        if let Some(p) = self.ptr_expr(l)? {
            if self.ptr_expr(r)?.is_some() {
                return self.not_yet(n, "a pointer difference or a sum of two pointers");
            }
            return self.offset_by(p, r, op == "-", n).map(Some);
        }
        if op == "+" {
            if let Some(p) = self.ptr_expr(r)? {
                return self.offset_by(p, l, false, n).map(Some);
            }
        }
        Ok(None)
    }

    /// `&p[i]` is `p + i`; `&x` of anything else is not modelled.
    fn ptr_address_of(&mut self, n: Node<'_>) -> R<Option<PtrVal>> {
        let op = n.child_by_field_name("operator").map(|o| text(o, self.src));
        if op != Some("&") {
            return Ok(None);
        }
        if let Some(a) = n
            .child_by_field_name("argument")
            .filter(|a| a.kind() == "identifier")
            && let Some(Sym::VecAlloc(i, e, _)) = self.lookup(text(a, self.src))
        {
            return Ok(Some(PtrVal {
                base: PBase::Alloc(i),
                pointee: e,
                lanes: 1,
                off: ScalarExpr::ConstInt(0, E::I64),
            }));
        }
        let arg = n
            .child_by_field_name("argument")
            .filter(|a| a.kind() == "subscript_expression");
        let Some(arg) = arg else {
            return self.not_yet(
                n,
                "the address of something that is not an array or pointer element `&p[i]`",
            );
        };
        // `&a[i][j]` of a multi-dimensional array: the flat element's address (the array is one row-major allocation)
        if let Some((Place::Alloc { alloc, index }, elem)) = self.nd_element_place(arg)? {
            let Some(size) = byte_size(elem) else {
                return self.not_yet(arg, format!("a pointer to {elem:?} (no whole-byte size)"));
            };
            let off = ScalarExpr::Mul(
                Box::new(index),
                Box::new(ScalarExpr::ConstInt(size, E::I64)),
            );
            return Ok(Some(PtrVal {
                base: PBase::Alloc(alloc),
                pointee: elem,
                lanes: 1,
                off,
            }));
        }
        let (Some(base), Some(idx)) = (
            arg.child_by_field_name("argument"),
            arg.child_by_field_name("indices"),
        ) else {
            return self.not_yet(arg, "an index of an unexpected shape");
        };
        let Some(p) = self.ptr_expr(base)? else {
            return self.not_yet(
                arg,
                "the address of an element of something that is not a pointer or array",
            );
        };
        let i = idx.named_child(0).filter(|_| idx.named_child_count() == 1);
        let Some(i) = i else {
            return self.not_yet(idx, "an index with other than one subscript");
        };
        self.offset_by(p, i, false, n).map(Some)
    }

    /// `p` moved by `delta` elements of its pointee (negated for `-`).
    fn offset_by(
        &mut self,
        mut p: PtrVal,
        delta: Node<'_>,
        negate: bool,
        at: Node<'_>,
    ) -> R<PtrVal> {
        let step = self.scaled(delta, p.pointee, at)?;
        p.off = if negate {
            ScalarExpr::Sub(Box::new(p.off), Box::new(step))
        } else {
            ScalarExpr::Add(Box::new(p.off), Box::new(step))
        };
        Ok(p)
    }

    /// `index * sizeof(pointee)` as an `I64` expression.
    fn scaled(&mut self, index: Node<'_>, pointee: E, at: Node<'_>) -> R<ScalarExpr> {
        let Some(size) = byte_size(pointee) else {
            return self.not_yet(at, format!("a pointer to {pointee:?} (no whole-byte size)"));
        };
        self.scaled_by(index, size, at)
    }

    /// `index * size` as an `I64` expression.
    pub(super) fn scaled_by(&mut self, index: Node<'_>, size: i64, at: Node<'_>) -> R<ScalarExpr> {
        let _ = at;
        let (i, it) = self.expr(index)?;
        if unpopped::types::int_range(it).is_none() {
            return self.outside(index, "a floating-point pointer offset");
        }
        let i = self.convert(i, it, E::I64, index)?;
        Ok(ScalarExpr::Mul(
            Box::new(i),
            Box::new(ScalarExpr::ConstInt(size, E::I64)),
        ))
    }

    /// The place `p[index]` (`index` of `None` is `*p`), and its type.
    pub(super) fn pointer_place(
        &mut self,
        p: &PtrVal,
        index: Option<Node<'_>>,
        at: Node<'_>,
    ) -> R<(Place, E)> {
        let off = match index {
            Some(i) => {
                let step = self.scaled(i, p.pointee, at)?;
                ScalarExpr::Add(Box::new(p.off.clone()), Box::new(step))
            }
            None => p.off.clone(),
        };
        let place = match p.base {
            PBase::Buffer(buffer) => Place::Raw {
                buffer,
                byte_offset: off,
                elem: p.pointee,
            },
            PBase::Alloc(alloc) => Place::AllocRaw {
                alloc,
                byte_offset: off,
                elem: p.pointee,
            },
        };
        Ok((place, p.pointee))
    }

    /// `T* name = <pointer expression>;`: a pointer local, fixed now (its offset is evaluated into a local).
    pub(super) fn pointer_declaration(
        &mut self,
        n: Node<'_>,
        ty: E,
        lanes: u8,
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
            let id = id
                .filter(|i| i.kind() == "pointer_declarator")
                .and_then(|i| i.child_by_field_name("declarator"));
            let (Some(id), Some(value)) = (id.filter(|i| i.kind() == "identifier"), value) else {
                return self.not_yet(
                    d,
                    "a pointer local that is not `T* name = <pointer expression>;`",
                );
            };
            let Some(mut p) = self.ptr_expr(value)? else {
                return self.not_yet(
                    value,
                    "a pointer local initialised from something that is not a pointer expression",
                );
            };
            if p.pointee != ty || p.lanes != lanes {
                return self.not_yet(
                    value,
                    format!(
                        "an implicit pointer conversion from {:?}* to {ty:?}* (cast it)",
                        p.pointee
                    ),
                );
            }
            self.flush(out);
            let name = text(id, self.src).to_string();
            p.off = if self.assigned.contains(&name) {
                // the pointer moves (`p += n`): its offset is a local the assignment can update, even when it starts constant
                let local = self.new_local(E::I64);
                out.push(Stmt::Let { local, init: p.off });
                ScalarExpr::Local(local)
            } else {
                self.pin_offset(p.off, out)
            };
            self.scopes
                .last_mut()
                .expect("a scope is open")
                .insert(name, Sym::Pointer(p));
        }
        Ok(())
    }

    /// `p += n` / `p -= n` on a pointer local that was declared as a mover: its offset local moves by `n` elements. Any other
    /// assignment to a pointer local (a new base, `p = q`) declines.
    pub(super) fn move_pointer(
        &mut self,
        lhs: Node<'_>,
        p: &PtrVal,
        op: &str,
        rhs: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        let name = text(lhs, self.src);
        let (ScalarExpr::Local(local), true) = (&p.off, matches!(op, "+" | "-")) else {
            return self.not_yet(
                lhs,
                format!("an assignment to `{name}`, which is not a scalar local (a pointer local only moves with `+=` or `-=`)"),
            );
        };
        let Some(size) = byte_size(p.pointee).map(|s| s * i64::from(p.lanes)) else {
            return self.not_yet(
                lhs,
                format!("moving the pointer `{name}`, whose pointee has no whole-byte size"),
            );
        };
        let (r, rt) = self.expr(rhs)?;
        let n = self.convert(r, rt, E::I64, rhs)?;
        let delta = ScalarExpr::Mul(Box::new(n), Box::new(ScalarExpr::ConstInt(size, E::I64)));
        let cur = Box::new(ScalarExpr::Local(*local));
        let value = if op == "+" {
            ScalarExpr::Add(cur, Box::new(delta))
        } else {
            ScalarExpr::Sub(cur, Box::new(delta))
        };
        self.flush(out);
        out.push(Stmt::Assign {
            local: *local,
            value,
        });
        Ok(())
    }

    /// The offset in a fresh immutable local (so the pointer cannot move when a variable its expression named changes).
    pub(super) fn pin_offset(&mut self, off: ScalarExpr, out: &mut Vec<Stmt>) -> ScalarExpr {
        if matches!(off, ScalarExpr::ConstInt(..)) {
            return off;
        }
        let local = self.new_local(E::I64);
        out.push(Stmt::Let { local, init: off });
        ScalarExpr::Local(local)
    }
}
