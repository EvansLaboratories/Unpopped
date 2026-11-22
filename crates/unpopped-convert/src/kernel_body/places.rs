// SPDX-License-Identifier: MIT OR Apache-2.0
//! Places: element and dereference places, thread builtins and addresses (ROADMAP U19).

use super::*;

impl<'s, 't> Walker<'s, 't> {
    pub(super) fn element_place(&mut self, n: Node<'_>) -> R<(Place, E)> {
        if n.kind() == "pointer_expression" {
            return self.deref_place(n);
        }
        let (Some(arg), Some(idx)) = (
            n.child_by_field_name("argument"),
            n.child_by_field_name("indices"),
        ) else {
            return self.not_yet(n, "an index of an unexpected shape");
        };
        let sym = match arg.kind() {
            "identifier" => self.lookup(text(arg, self.src)),
            "field_expression" => self.resolve_member(arg)?,
            _ => None,
        };
        let Some(Sym::Buffer(_, elem) | Sym::Alloc(_, elem)) = sym else {
            // not a plain buffer or array: a pointer expression (`p[i]`, `(T*)p`, `(p + k)[i]`), read through a raw place
            if let Some(p) = self.ptr_expr(arg)? {
                let one = idx.named_child(0).filter(|_| idx.named_child_count() == 1);
                let Some(one) = one else {
                    return self.not_yet(idx, "an index with other than one subscript");
                };
                return self.pointer_place(&p, Some(one), n);
            }
            return self.not_yet(
                arg,
                "indexing something that is not a buffer parameter, a `__shared__` array or a pointer",
            );
        };
        if idx.named_child_count() != 1 {
            return self.not_yet(idx, "an index with other than one subscript");
        }
        let i = idx.named_child(0).expect("one subscript");
        let (ie, it) = self.expr(i)?;
        if int_range(it).is_none() {
            return self.outside(i, "a floating-point index");
        }
        let place = match sym {
            Some(Sym::Alloc(alloc, _)) => Place::Alloc { alloc, index: ie },
            Some(Sym::Buffer(buffer, _)) => Place::Logical {
                buffer,
                coords: vec![ie],
            },
            _ => unreachable!("matched above"),
        };
        Ok((place, elem))
    }

    /// `*p`: the place a pointer expression points at.
    pub(super) fn deref_place(&mut self, n: Node<'_>) -> R<(Place, E)> {
        let op = n.child_by_field_name("operator").map(|o| text(o, self.src));
        let arg = n.child_by_field_name("argument");
        let (Some("*"), Some(arg)) = (op, arg) else {
            return self.not_yet(n, "the address-of operator used as a value (only `&p[i]` as an atomic or `__ldg` address, or a pointer initialiser)");
        };
        match self.ptr_expr(arg)? {
            Some(p) => self.pointer_place(&p, None, n),
            None => self.not_yet(n, "a dereference of something that is not a pointer"),
        }
    }

    /// Whether `n` is `&a[i]` of a plain buffer parameter or `__shared__` array (which keeps its typed `Logical`/`Alloc` place).
    pub(super) fn is_plain_address(&self, n: Node<'_>) -> bool {
        let inner = if n.kind() == "parenthesized_expression" {
            n.named_child(0)
        } else {
            Some(n)
        };
        let Some(inner) = inner.filter(|i| i.kind() == "pointer_expression") else {
            return false;
        };
        let is_amp = inner
            .child_by_field_name("operator")
            .is_some_and(|o| text(o, self.src) == "&");
        let base = inner
            .child_by_field_name("argument")
            .filter(|a| a.kind() == "subscript_expression")
            .and_then(|a| a.child_by_field_name("argument"))
            .filter(|b| b.kind() == "identifier");
        is_amp
            && base.is_some_and(|b| {
                matches!(
                    self.lookup(text(b, self.src)),
                    Some(Sym::Buffer(..) | Sym::Alloc(..))
                )
            })
    }

    pub(super) fn read_element(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let (place, elem) = self.element_place(n)?;
        let local = self.new_local(elem);
        self.hoisted.push(Stmt::Load { local, place });
        Ok((ScalarExpr::Local(local), elem))
    }

    /// `threadIdx.x`, `blockIdx.y`, `blockDim.z`, `gridDim.x`: the launch's execution ids (`U32`).
    pub(super) fn builtin(&self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        if let Some(m) = self.resolve_member(n)? {
            return match m {
                Sym::Scalar(l, t, _) => Ok((ScalarExpr::Local(l), t)),
                _ => self.not_yet(
                    n,
                    "a struct or array member used as a value (read a scalar member)",
                ),
            };
        }
        let (Some(a), Some(f)) = (
            n.child_by_field_name("argument"),
            n.child_by_field_name("field"),
        ) else {
            return self.not_yet(n, "a member access of an unexpected shape");
        };
        let base = text(a, self.src);
        if let (true, Some(Sym::Vector(ls, elem, lanes))) =
            (a.kind() == "identifier", self.lookup(base))
        {
            return self.vector_member(f, ls, (elem, lanes));
        }
        if a.kind() != "identifier" || self.lookup(base).is_some() {
            return self.not_yet(
                n,
                "a member access on a value (structs and vector types are a later slice)",
            );
        }
        self.exec_builtin(n, a, f)
    }

    /// `v.x` of a vector local: that lane's scalar local.
    fn vector_member(
        &self,
        f: Node<'_>,
        ls: [u32; 4],
        (elem, lanes): (E, u8),
    ) -> R<(ScalarExpr, E)> {
        let Some(i) = lane_of(text(f, self.src), lanes) else {
            return self.not_yet(
                f,
                format!(
                    "the member `{}` of a {lanes}-lane vector",
                    text(f, self.src)
                ),
            );
        };
        Ok((ScalarExpr::Local(ls[usize::from(i)]), elem))
    }

    /// `threadIdx.x` and kin: the launch's execution id (`U32`), which needs the caller's launch.
    fn exec_builtin(&self, n: Node<'_>, a: Node<'_>, f: Node<'_>) -> R<(ScalarExpr, E)> {
        let base = text(a, self.src);
        let axis = match text(f, self.src) {
            "x" => 0,
            "y" => 1,
            "z" => 2,
            other => return self.not_yet(f, format!("the member `{other}`")),
        };
        let id = match base {
            "threadIdx" => ExecId::ThreadIdx(axis),
            "blockIdx" => ExecId::BlockIdx(axis),
            "blockDim" => ExecId::BlockDim(axis),
            "gridDim" => ExecId::GridDim(axis),
            other => {
                return self.not_yet(
                    a,
                    format!(
                        "the name `{other}`, which is not a parameter, local or thread builtin"
                    ),
                );
            }
        };
        if self.launch.is_none() {
            return self.not_yet(
                n,
                format!("`{base}` without a launch (use `lift_kernel_launched`)"),
            );
        }
        Ok((ScalarExpr::Exec(id), E::U32))
    }

    /// `&a[i]`: the place, its element type and the scope an atomic on it has.
    pub(super) fn address_of(&mut self, n: Node<'_>) -> R<(Place, E, Scope)> {
        if !self.is_plain_address(n) {
            if let Some(p) = self.ptr_expr(n)? {
                let (place, elem) = self.pointer_place(&p, None, n)?;
                let scope = if matches!(place, Place::AllocRaw { .. }) {
                    Scope::Block
                } else {
                    Scope::Device
                };
                return Ok((place, elem, scope));
            }
        }
        let inner = if n.kind() == "parenthesized_expression" {
            n.named_child(0)
        } else {
            Some(n)
        };
        let Some(inner) = inner.filter(|i| i.kind() == "pointer_expression") else {
            return self.not_yet(
                n,
                "an atomic or `__ldg` whose address is not a pointer expression (`&a[i]`, `p + i`, ...)",
            );
        };
        let op = inner
            .child_by_field_name("operator")
            .map(|o| text(o, self.src));
        let arg = inner
            .child_by_field_name("argument")
            .filter(|a| a.kind() == "subscript_expression");
        let (Some("&"), Some(arg)) = (op, arg) else {
            return self.not_yet(n, "an atomic or `__ldg` whose address is not `&p[i]`");
        };
        let (place, elem) = self.element_place(arg)?;
        let scope = if matches!(place, Place::Alloc { .. }) {
            Scope::Block
        } else {
            Scope::Device
        };
        Ok((place, elem, scope))
    }
}
