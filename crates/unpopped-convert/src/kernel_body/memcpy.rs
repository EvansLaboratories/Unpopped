// SPDX-License-Identifier: MIT OR Apache-2.0
//! `memcpy(&x, p, sizeof(x))` of a scalar local from memory (census wall: the gguf `memcpy(&qh, x[ib].qh, sizeof(qh))` that gathers
//! four bytes into a `uint32_t`). It is the load of `x`'s type from the pointer `p`, at `p`'s byte offset (an unaligned or
//! struct-field address included, as `memcpy` allows). Only that form lifts: the destination is the address of a plain scalar
//! local, the size is `sizeof` that local (or its byte count), the source a pointer into a buffer or array. A vector, struct or
//! array destination, a size that is not the local's, or a source that is not a pointer declines by name.

use super::pointer::byte_size;
use super::{Fixed, R, Sym, Walker, text};
use tree_sitter::Node;
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::Stmt;
use unpopped_vocab::ElementKind as E;

impl<'s, 't> Walker<'s, 't> {
    /// `memcpy(&x, src, n);` as a statement. `None` if the callee is not `memcpy`.
    pub(super) fn memcpy_statement(
        &mut self,
        call: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> Option<R<()>> {
        let callee = call
            .child_by_field_name("function")
            .map(|f| text(f, self.src))?;
        if !matches!(callee, "memcpy" | "std::memcpy" | "__builtin_memcpy") {
            return None;
        }
        Some(self.memcpy_scalar(call, out))
    }

    fn memcpy_scalar(&mut self, call: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        let args = self.call_args(call);
        let [dst, src, size] = args.as_slice() else {
            return self.not_yet(call, "a `memcpy` with other than three arguments");
        };
        let target = (dst.kind() == "pointer_expression"
            && dst
                .child_by_field_name("operator")
                .is_some_and(|o| text(o, self.src) == "&"))
        .then(|| dst.child_by_field_name("argument"))
        .flatten()
        .filter(|a| a.kind() == "identifier")
        .map(|a| text(a, self.src));
        let Some(name) = target else {
            return self.not_yet(
                call,
                "a `memcpy` whose destination is not `&x` of a scalar local",
            );
        };
        let Some(Sym::Scalar(local, ty, fixed)) = self.lookup(name) else {
            return self.not_yet(
                call,
                format!("a `memcpy` into `{name}`, which is not a scalar local"),
            );
        };
        if !matches!(fixed, Fixed::No) {
            return self.not_yet(
                call,
                format!("a `memcpy` into the parameter or counter `{name}`"),
            );
        }
        let Some(bytes) = byte_size(ty) else {
            return self.not_yet(
                call,
                format!("a `memcpy` into `{name}`, whose size is unknown"),
            );
        };
        let stated = text(*size, self.src).split_whitespace().collect::<String>();
        let ok = stated == format!("sizeof({name})")
            || stated
                == format!(
                    "sizeof({})",
                    text(
                        dst.child_by_field_name("argument").unwrap_or(*dst),
                        self.src
                    )
                )
            || stated.parse::<i64>().ok() == Some(bytes);
        if !ok {
            return self.not_yet(
                *size,
                format!("a `memcpy` of `{stated}` bytes into `{name}`, which is {bytes}"),
            );
        }
        let Some(mut p) = self.ptr_expr(*src)? else {
            return self.not_yet(*src, "a `memcpy` from something that is not a pointer");
        };
        // the value is assembled from byte loads, little-endian: `memcpy` is allowed an unaligned source (a field at an even offset of
        // a packed block), which a wide aligned load would trap on
        let wide = match bytes {
            4 => E::U32,
            8 => E::U64,
            _ => {
                return self.not_yet(
                    call,
                    format!("a `memcpy` into `{name}` of {bytes} bytes (4 and 8 lift)"),
                );
            }
        };
        (p.pointee, p.lanes) = (E::U8, 1);
        self.flush(out);
        let mut acc: Option<ScalarExpr> = None;
        for k in 0..bytes {
            let mut q = p.clone();
            q.off = ScalarExpr::Add(
                Box::new(p.off.clone()),
                Box::new(ScalarExpr::ConstInt(k, E::I64)),
            );
            let (place, _) = self.pointer_place(&q, None, *src)?;
            let byte = self.new_local(E::U8);
            out.push(Stmt::Load { local: byte, place });
            let widened = self.convert(ScalarExpr::Local(byte), E::U8, wide, *src)?;
            let placed = if k == 0 {
                widened
            } else {
                ScalarExpr::Binary(
                    BinaryOp::Shl,
                    Box::new(widened),
                    Box::new(ScalarExpr::ConstInt(8 * k, wide)),
                )
            };
            acc = Some(match acc {
                None => placed,
                Some(prev) => ScalarExpr::Binary(BinaryOp::BitOr, Box::new(prev), Box::new(placed)),
            });
        }
        let Some(bits) = acc else {
            return self.not_yet(call, "a `memcpy` of no bytes");
        };
        let value = if ty == wide {
            bits
        } else {
            ScalarExpr::Bitcast(Box::new(bits), ty)
        };
        out.push(Stmt::Assign { local, value });
        Ok(())
    }
}
