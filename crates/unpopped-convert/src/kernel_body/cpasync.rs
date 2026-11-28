// SPDX-License-Identifier: MIT OR Apache-2.0
//! `cp.async` copies written as inline asm (ROADMAP U19 slice 5, the decode kernel's `cp_async_pred_load_128b_*`): the
//! `__cvta_generic_to_shared(p)` integer the asm takes as `"r"` is bound as a **symbolic shared pointer** (never an integer
//! value: any other use of it declines), and the two templates the file uses lift to `Stmt::CopyAsync`.
//!
//! - `cp.async.cg.shared.global.L2::128B [%d], [%s], N, %n` (the **fill-zero** form): `valid` = `%n` bytes / element size, so
//!   a zero count reads nothing and zero-fills.
//! - `{ .reg .pred p; setp.ne.b32 p, %k, 0; @p cp.async... [%d], [%s], N; }` (the **no-fill** form): `If(%k != 0)` around the
//!   copy, so a false predicate leaves the destination untouched.

use super::asm::Operand;
use super::pointer::{PBase, PtrVal, byte_size};
use super::{R, Walker, text};
use tree_sitter::Node;
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, CacheHint, Place, Stmt};
use unpopped_vocab::ElementKind as E;

impl<'s, 't> Walker<'s, 't> {
    /// `static_cast<uint32_t>(__cvta_generic_to_shared(p))` (casts and parentheses allowed): the pointer `p`, which must point
    /// into a `__shared__` or dynamic-shared allocation. `None` when the expression is something else.
    pub(super) fn shared_address(&mut self, v: Node<'_>) -> R<Option<PtrVal>> {
        let mut n = v;
        loop {
            match n.kind() {
                "parenthesized_expression" => match n.named_child(0) {
                    Some(i) => n = i,
                    None => return Ok(None),
                },
                "cast_expression" => match n.child_by_field_name("value") {
                    Some(i) => n = i,
                    None => return Ok(None),
                },
                "call_expression" => {
                    let f = n
                        .child_by_field_name("function")
                        .map_or("", |f| text(f, self.src));
                    let args = self.call_args(n);
                    if f.starts_with("static_cast<") && args.len() == 1 {
                        n = args[0];
                    } else if f == "__cvta_generic_to_shared" && args.len() == 1 {
                        return match self.ptr_expr(args[0])? {
                            Some(p) if matches!(p.base, PBase::Alloc(_)) => Ok(Some(p)),
                            _ => self.not_yet(n, "`__cvta_generic_to_shared` of something that is not a pointer into a shared allocation"),
                        };
                    } else {
                        return Ok(None);
                    }
                }
                _ => return Ok(None),
            }
        }
    }

    /// A `cp.async.{ca,cg}.shared.global` copy, plain or predicated.
    pub(super) fn asm_copy_async(
        &mut self,
        e: Node<'_>,
        words: &[&str],
        ins: &[Operand<'_>],
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        let Some(ci) = words.iter().position(|w| w.starts_with("cp.async.c")) else {
            return self.not_yet(e, "an asm copy of an unexpected shape");
        };
        let opcode = words[ci];
        let cache = if opcode.starts_with("cp.async.cg.") {
            CacheHint::Global
        } else {
            CacheHint::All
        };
        let l2 = opcode
            .split("L2::")
            .nth(1)
            .and_then(|s| s.trim_end_matches('B').parse::<u32>().ok());
        let rest = &words[ci + 1..];
        let copy = self.cp_async_copy(e, words, rest, ins, cache, l2)?;
        let predicated = words.contains(&"@p") && words.contains(&"setp.ne.b32");
        if !predicated {
            out.append(&mut self.hoisted);
            out.push(copy);
            return Ok(());
        }
        let cond = self.cp_async_predicate(e, words, ins)?;
        out.append(&mut self.hoisted);
        out.push(Stmt::If {
            cond,
            then: Block(vec![copy]),
            els: Block(vec![]),
        });
        Ok(())
    }

    /// The unpredicated `CopyAsync` of a `cp.async` template: destination, source, byte count and optional valid-bytes.
    fn cp_async_copy(
        &mut self,
        e: Node<'_>,
        words: &[&str],
        rest: &[&str],
        ins: &[Operand<'_>],
        cache: CacheHint,
        l2: Option<u32>,
    ) -> R<Stmt> {
        let reg = |w: &str| w.trim_matches(['[', ']', '%']).parse::<usize>().ok();
        let operand = |w: &str| reg(w).and_then(|k| ins.get(k));
        let (Some(dst), Some(src), Some(size)) = (
            rest.first().and_then(|w| operand(w)),
            rest.get(1).and_then(|w| operand(w)),
            rest.get(2).and_then(|w| operand(w)),
        ) else {
            return self.not_yet(e, format!("the asm template `{}`", words.join(" ")));
        };
        let (alloc, dp, buffer, sp, bytes) =
            self.cp_async_endpoints(e, dst.value, src.value, size.value)?;
        let (elem, esize) = self.cp_async_elem(e, alloc, bytes)?;
        let scaled = |v: ScalarExpr| -> ScalarExpr {
            if esize == 1 {
                v
            } else {
                ScalarExpr::Div(Box::new(v), Box::new(ScalarExpr::ConstInt(esize, E::I64)))
            }
        };
        let valid = match rest.get(3).and_then(|w| operand(w)) {
            Some(o) => Some(self.cp_async_valid(o.value, esize)?),
            None => None,
        };
        Ok(Stmt::CopyAsync {
            dst: Place::Alloc {
                alloc,
                index: scaled(dp.off),
            },
            src: Place::Raw {
                buffer,
                byte_offset: sp.off,
                elem,
            },
            elems: u32::try_from(bytes / esize).unwrap_or(0),
            valid,
            cache,
            l2_prefetch: l2,
        })
    }

    /// The shared destination, the global source and the literal byte count of a `cp.async`.
    fn cp_async_endpoints(
        &mut self,
        e: Node<'_>,
        dst: Node<'_>,
        src: Node<'_>,
        size: Node<'_>,
    ) -> R<(u32, PtrVal, u32, PtrVal, i64)> {
        let (Some(dp), Some(sp)) = (self.ptr_expr(dst)?, self.ptr_expr(src)?) else {
            return self.not_yet(
                e,
                "a `cp.async` whose operands are not shared/global pointers",
            );
        };
        let Ok(bytes) = text(size, self.src).trim().parse::<i64>() else {
            return self.not_yet(e, "a `cp.async` whose byte count is not a literal");
        };
        let (PBase::Alloc(alloc), PBase::Buffer(buffer)) = (dp.base, sp.base) else {
            return self.not_yet(e, "a `cp.async` that is not shared-from-global");
        };
        Ok((alloc, dp, buffer, sp, bytes))
    }

    /// The element type of the destination allocation and its byte size: the copy is in the allocation's own element type, so a
    /// byte count must be a whole number of those elements.
    fn cp_async_elem(&mut self, e: Node<'_>, alloc: u32, bytes: i64) -> R<(E, i64)> {
        // The copy is in the destination allocation's own element type (the checker wants the two places to agree, and the
        // decode kernel's dynamic shared memory is `uint8_t` under `half*` pointers): byte counts become element counts.
        let Some(elem) = self.allocs.get(alloc as usize).map(|a| a.elem) else {
            return self.not_yet(e, "a `cp.async` into an unknown allocation");
        };
        let Some(esize) = byte_size(elem).filter(|s| *s > 0 && bytes % *s == 0) else {
            return self.not_yet(
                e,
                "a `cp.async` byte count that is not a whole number of destination elements",
            );
        };
        Ok((elem, esize))
    }

    /// The valid-bytes operand of a `cp.async`, in destination elements.
    fn cp_async_valid(&mut self, value: Node<'_>, esize: i64) -> R<ScalarExpr> {
        let (v, vt) = self.expr(value)?;
        let v = self.convert(v, vt, E::I32, value)?;
        Ok(ScalarExpr::Div(
            Box::new(v),
            Box::new(ScalarExpr::ConstInt(esize, E::I32)),
        ))
    }

    /// The condition of a predicated `cp.async`: `setp.ne.b32 p, %k, 0`.
    fn cp_async_predicate(
        &mut self,
        e: Node<'_>,
        words: &[&str],
        ins: &[Operand<'_>],
    ) -> R<ScalarExpr> {
        let reg = |w: &str| w.trim_matches(['[', ']', '%']).parse::<usize>().ok();
        let operand = |w: &str| reg(w).and_then(|k| ins.get(k));
        let Some(k) = words
            .iter()
            .position(|w| *w == "setp.ne.b32")
            .and_then(|i| words.get(i + 2))
            .and_then(|w| operand(w))
        else {
            return self.not_yet(
                e,
                "a predicated `cp.async` whose predicate operand is not `setp.ne.b32 p, %k, 0`",
            );
        };
        let (p, pt) = self.expr(k.value)?;
        let zero = self.zero_of(pt, k.value)?;
        Ok(ScalarExpr::Binary(
            BinaryOp::CmpNe,
            Box::new(p),
            Box::new(zero),
        ))
    }
}
