// SPDX-License-Identifier: MIT OR Apache-2.0
//! File-scope constants, `sizeof` and dynamic shared memory (ROADMAP U19 slice 5, `docs/options-u19-slice5-decode-census.md`).
//! A `constexpr` or `const` global with an initialiser is read where its name is used, by evaluating its initialiser in an empty
//! scope (so it can name other globals and nothing local); a bare literal takes the declared type directly. `sizeof(T)` is a
//! `size_t` (`U64`) constant (a pointer is 8 bytes). `extern __shared__ T a[];` is one `Shared` allocation of the caller's
//! dynamic byte count divided by `sizeof(T)` (`LiftContext::with_dynamic_shared_bytes`).

use super::pointer::byte_size;
use super::{Lit, R, Walker, parse_literal, text};
use std::collections::HashMap;
use tree_sitter::Node;
use unpopped::exec::{AllocDecl, Extent, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped_vocab::ElementKind as E;

impl<'s, 't> Walker<'s, 't> {
    /// The value of the file-scope constant `name`, if there is one.
    pub(super) fn global_const(&mut self, name: &str, at: Node<'_>) -> R<Option<(ScalarExpr, E)>> {
        let Some((tyname, init)) = self.consts.get(name).cloned() else {
            return Ok(None);
        };
        let to = match self.ctype(&tyname) {
            Ok(e) => e,
            Err(why) => return self.not_yet(at, format!("the constant `{name}`: {why}")),
        };
        let (e, t) = if init.kind() == "number_literal" {
            self.bare_literal(init, at)?
        } else {
            let saved = std::mem::replace(&mut self.scopes, vec![HashMap::new()]);
            let saved_loads = std::mem::take(&mut self.hoisted);
            let r = self.expr(init);
            self.scopes = saved;
            self.hoisted = saved_loads;
            r?
        };
        // An integer constant expression is a literal: `1 << 4` stays a literal 16, not a shift the oracle would have to type
        // inside a cast
        let e = match (fold(&e), t) {
            (Some(v), E::I32 | E::I64 | E::U32 | E::U64) => {
                i64::try_from(v).map_or(e, |v| ScalarExpr::ConstInt(v, t))
            }
            _ => e,
        };
        Ok(Some((self.convert(e, t, to, at)?, to)))
    }

    /// A literal as its own C type, with no requirement that it be the kernel's compute type (the declared type converts it).
    fn bare_literal(&self, n: Node<'_>, at: Node<'_>) -> R<(ScalarExpr, E)> {
        match parse_literal(text(n, self.src)) {
            Some(Lit::Int(v, t)) => Ok((ScalarExpr::ConstInt(v, t), t)),
            Some(Lit::Float(v, _)) => Ok((ScalarExpr::Const(v), self.compute)),
            None => self.not_yet(at, format!("the literal `{}`", text(n, self.src))),
        }
    }

    /// An operand tested for truth (`&&`, `||`, `!`, a condition): a **buffer pointer** is a null check, and is `1` (the buffer
    /// is bound) unless the caller declared it null (`LiftContext::with_null_buffer`); anything else is an ordinary expression.
    /// Whether the buffer or pointer `name` is null: declared so by the caller (`LiftContext::with_null_buffer`), or a pointer
    /// local whose base is a buffer parameter so declared (`const bool* m = m_ptr;` is null when `m_ptr` is).
    pub(super) fn pointer_is_null(&self, name: &str) -> bool {
        if self.null_buffers.contains(name) {
            return true;
        }
        match self.lookup(name) {
            Some(super::Sym::Pointer(p)) => match p.base {
                super::pointer::PBase::Buffer(i) => self
                    .params
                    .get(i as usize)
                    .is_some_and(|q| self.null_buffers.contains(&q.name)),
                super::pointer::PBase::Alloc(_) => false,
            },
            _ => false,
        }
    }

    /// `otherwise`, unless `name` is a macro the preprocessor could not expand: then the reason it could not (`the function-like
    /// macro `F``), so a decline names the macro, not an anonymous call or name.
    pub(super) fn unexpanded_or(&self, name: &str, otherwise: String) -> String {
        self.unexpanded
            .iter()
            .find(|(n, _)| n == name)
            .map_or(otherwise, |(_, why)| {
                format!("{why} (expansion is a later slice)")
            })
    }

    /// Whether `n` is the name of a buffer or pointer.
    pub(super) fn is_pointer_name(&self, n: Node<'_>) -> bool {
        n.kind() == "identifier"
            && matches!(
                self.lookup(text(n, self.src)),
                Some(super::Sym::Buffer(..) | super::Sym::Pointer(_))
            )
    }

    pub(super) fn truth_expr(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let operand = if n.kind() == "parenthesized_expression" {
            n.named_child(0).unwrap_or(n)
        } else {
            n
        };
        if operand.kind() == "identifier" {
            let name = text(operand, self.src);
            if matches!(
                self.lookup(name),
                Some(super::Sym::Buffer(..) | super::Sym::Pointer(_))
            ) {
                return Ok(self.constant(i64::from(!self.pointer_is_null(name))));
            }
        }
        self.expr(n)
    }

    /// `sizeof(T)` and `sizeof(T*)`: a `U64` constant. A grammar that does not know `T` as a type (`half`) parses `sizeof(T)` as
    /// `sizeof` of a parenthesised name, so the operand's text is read as a type name either way.
    pub(super) fn sizeof_expr(&self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let operand = n
            .child_by_field_name("type")
            .or_else(|| n.child_by_field_name("value"))
            .map(|o| {
                text(o, self.src)
                    .trim()
                    .trim_start_matches('(')
                    .trim_end_matches(')')
                    .trim()
                    .to_string()
            });
        let Some(name) = operand else {
            return self.not_yet(n, "a `sizeof` of an unexpected shape");
        };
        let size = if let Some(base) = name.strip_suffix('*') {
            let _ = base;
            8
        } else {
            let known = self.ctype(&name).ok().and_then(byte_size);
            match known {
                Some(s) => s,
                None => {
                    return self.not_yet(
                        n,
                        format!(
                            "`sizeof({name})` (only the scalar types and pointers to anything lift)"
                        ),
                    );
                }
            }
        };
        Ok((ScalarExpr::ConstInt(size, E::U64), E::U64))
    }

    /// The element count of a local or shared array: a constant integer expression (a literal, a file-scope constant, `sizeof`,
    /// arithmetic over them), positive.
    pub(super) fn fold_size(&mut self, n: Node<'_>) -> R<Option<u32>> {
        let saved = std::mem::take(&mut self.hoisted);
        let r = self.expr(n);
        self.hoisted = saved;
        let (e, _) = r?;
        Ok(fold(&e)
            .and_then(|v| u32::try_from(v).ok())
            .filter(|v| *v > 0))
    }

    /// `extern __shared__ T name[];`: the caller's dynamic shared memory as one array.
    pub(super) fn dynamic_shared_declaration(&mut self, n: Node<'_>, ty: E) -> R<()> {
        if self.launch.is_none() {
            return self.not_yet(
                n,
                "`extern __shared__` without a launch (use `lift_kernel_in`)",
            );
        }
        let Some(size) = byte_size(ty).filter(|s| *s > 0) else {
            return self.not_yet(
                n,
                format!("`extern __shared__` of {ty:?}, which has no whole-byte size"),
            );
        };
        let count = u32::try_from(i64::from(self.dyn_shared) / size).unwrap_or(0);
        if count == 0 {
            return self.not_yet(n, "`extern __shared__` with no dynamic shared memory given (`LiftContext::with_dynamic_shared_bytes`)");
        }
        let mut c = n.walk();
        let decls: Vec<_> = n.children_by_field_name("declarator", &mut c).collect();
        for d in decls {
            let id = d
                .child_by_field_name("declarator")
                .filter(|i| i.kind() == "identifier");
            let (true, Some(id)) = (d.kind() == "array_declarator", id) else {
                return self.not_yet(
                    d,
                    "an `extern __shared__` declarator that is not `T name[]`",
                );
            };
            let name = text(id, self.src).to_string();
            self.allocs.push(AllocDecl::new(
                name.clone(),
                MemorySpace::Shared,
                ty,
                Extent::Static(count),
            ));
            let index = u32::try_from(self.allocs.len() - 1).unwrap_or(u32::MAX);
            self.scopes
                .last_mut()
                .expect("a scope is open")
                .insert(name, super::Sym::Alloc(index, ty));
        }
        Ok(())
    }
}

/// The value of a constant integer expression (literals, casts, arithmetic, shifts, bit operations, comparisons and a selected
/// arm), or `None` if it depends on anything else.
pub(super) fn fold(e: &ScalarExpr) -> Option<i128> {
    fold_with(e, &|_| None)
}

/// [`fold`] where `leaf` may give a value to an expression the plain fold cannot (a launch extent, a known constant local).
pub(super) fn fold_with(e: &ScalarExpr, leaf: &dyn Fn(&ScalarExpr) -> Option<i128>) -> Option<i128> {
    use unpopped::ir::BinaryOp as B;
    let fold = |x: &ScalarExpr| fold_with(x, leaf);
    let two = |a: &ScalarExpr, b: &ScalarExpr| Some((fold(a)?, fold(b)?));
    Some(match e {
        ScalarExpr::ConstInt(v, _) => i128::from(*v),
        ScalarExpr::Cast(x, _) => fold(x)?,
        ScalarExpr::Add(a, b) => {
            let (x, y) = two(a, b)?;
            x + y
        }
        ScalarExpr::Sub(a, b) => {
            let (x, y) = two(a, b)?;
            x - y
        }
        ScalarExpr::Mul(a, b) => {
            let (x, y) = two(a, b)?;
            x * y
        }
        ScalarExpr::Div(a, b) => {
            let (x, y) = two(a, b)?;
            if y == 0 { return None } else { x / y }
        }
        ScalarExpr::Binary(op, a, b) => {
            let (x, y) = two(a, b)?;
            match op {
                B::Shl if (0..64).contains(&y) => x << y,
                B::Shr if (0..64).contains(&y) => x >> y,
                B::BitAnd => x & y,
                B::BitOr => x | y,
                B::BitXor => x ^ y,
                B::Min => x.min(y),
                B::Max => x.max(y),
                B::CmpEq => i128::from(x == y),
                B::CmpNe => i128::from(x != y),
                B::CmpLt => i128::from(x < y),
                B::CmpLe => i128::from(x <= y),
                B::CmpGt => i128::from(x > y),
                B::CmpGe => i128::from(x >= y),
                _ => return None,
            }
        }
        ScalarExpr::Select(c, a, b) => {
            if fold(c)? != 0 {
                fold(a)?
            } else {
                fold(b)?
            }
        }
        _ => return leaf(e),
    })
}
