// SPDX-License-Identifier: MIT OR Apache-2.0
//! Inlining of the source's own functions (ROADMAP U19 slice 4c-2, `docs/options-u19-slice5-decode-census.md`): a call to a
//! function defined in the same source is replaced by its body, in the caller's statement list. Scalar parameters are bound
//! by value (a fresh local holding the converted argument) or, for `T&`, to the caller's own local, so an assignment through
//! the reference reaches the caller. A callee may `return` once, as its last statement. Pointer and struct parameters,
//! recursion and a `return` anywhere else decline (4c-4 for pointers).

use super::{Fixed, R, Sym, Walker, text};
use std::collections::HashMap;
use tree_sitter::Node;
use unpopped::ir::ScalarExpr;
use unpopped::kernel::Stmt;
use unpopped_vocab::ElementKind as E;

/// The deepest chain of inlined calls (a guard, not a limit anyone should meet).
const MAX_DEPTH: usize = 16;

/// How a parameter takes its argument.
enum Bind {
    /// A copy: a fresh local initialised from the converted argument.
    Value(E),
    /// `T&`: the caller's own local.
    Reference(E),
}

impl<'s, 't> Walker<'s, 't> {
    /// Whether `name` is a function defined in this source (so a call to it is inlined).
    pub(super) fn defines(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    /// Inline the call `call` of `name`. The callee's statements are appended to `out`; the value of its `return`, if any, is
    /// the result.
    pub(super) fn inline_call(
        &mut self,
        call: Node<'_>,
        name: &str,
        out: &mut Vec<Stmt>,
    ) -> R<Option<(ScalarExpr, E)>> {
        let Some(&func) = self.functions.get(name) else {
            return self.not_yet(call, format!("a call to `{name}`"));
        };
        if self.inline_stack.len() >= MAX_DEPTH || self.inline_stack.iter().any(|n| n == name) {
            return self.not_yet(call, format!("a recursive call to `{name}`"));
        }
        let (Some(plist), Some(body)) = (
            func.child_by_field_name("declarator")
                .and_then(|d| d.child_by_field_name("parameters")),
            func.child_by_field_name("body"),
        ) else {
            return self.not_yet(call, format!("`{name}` without a parameter list or body"));
        };
        let ret_ty = self.return_type(func)?;
        let args = self.call_args(call);
        let mut cursor = plist.walk();
        let params: Vec<_> = plist.named_children(&mut cursor).collect();
        if params.len() != args.len() {
            return self.not_yet(
                call,
                format!(
                    "`{name}` with {} argument(s), it takes {}",
                    args.len(),
                    params.len()
                ),
            );
        }
        let mut bound = HashMap::new();
        for (p, a) in params.into_iter().zip(args) {
            let (pname, sym) = self.bind_argument(p, a, out)?;
            bound.insert(pname, sym);
        }
        self.check_returns(body, ret_ty.is_some())?;
        self.inline_stack.push(name.to_string());
        let saved = std::mem::replace(&mut self.scopes, vec![bound]);
        let r = self.callee_body(body, ret_ty, out);
        self.scopes = saved;
        self.inline_stack.pop();
        r
    }

    /// `void` is `None`; otherwise the element type.
    fn return_type(&self, func: Node<'_>) -> R<Option<E>> {
        let ty = func.child_by_field_name("type");
        if ty.is_some_and(|t| text(t, self.src) == "void") {
            return Ok(None);
        }
        self.type_of(ty, func).map(Some)
    }

    /// One parameter bound to one argument, in the caller's scope; the initialising `Let` goes to `out`.
    fn bind_argument(
        &mut self,
        p: Node<'_>,
        arg: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<(String, Sym)> {
        if let Some(sname) = self.struct_type_of(p) {
            return self.bind_struct_argument(p, arg, &sname, out);
        }
        let ty = self.type_of(p.child_by_field_name("type"), p)?;
        let Some(d) = p.child_by_field_name("declarator") else {
            return self.not_yet(p, "an unnamed parameter");
        };
        if d.kind() == "pointer_declarator" {
            return self.bind_pointer_argument(d, ty, arg, out);
        }
        let (kind, id) = match d.kind() {
            "identifier" => (Bind::Value(ty), d),
            "reference_declarator" => {
                let id = d.named_child(0).filter(|i| i.kind() == "identifier");
                let Some(id) = id else {
                    return self.not_yet(d, "a reference parameter of an unexpected shape");
                };
                if super::has_const(p, self.src) {
                    (Bind::Value(ty), id)
                } else {
                    (Bind::Reference(ty), id)
                }
            }
            other => {
                return self.not_yet(
                    d,
                    format!("a `{other}` parameter of an inlined function (pointer parameters are a later slice)"),
                );
            }
        };
        let pname = text(id, self.src).to_string();
        match kind {
            Bind::Value(t) => self.bind_value(pname, t, arg, out),
            Bind::Reference(t) => self.bind_reference(pname, t, arg),
        }
    }

    /// A by-value parameter: a fresh local holding the converted argument.
    fn bind_value(
        &mut self,
        pname: String,
        t: E,
        arg: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<(String, Sym)> {
        let (v, vt) = self.expr(arg)?;
        let v = self.convert(v, vt, t, arg)?;
        out.append(&mut self.hoisted);
        let local = self.new_local(t);
        out.push(Stmt::Let { local, init: v });
        Ok((pname, Sym::Scalar(local, t, Fixed::No)))
    }

    /// A `T&` parameter: the caller's own local, which must be a plain local of the same type.
    fn bind_reference(&mut self, pname: String, t: E, arg: Node<'_>) -> R<(String, Sym)> {
        let target = (arg.kind() == "identifier")
            .then(|| self.lookup(text(arg, self.src)))
            .flatten();
        match target {
            Some(Sym::Scalar(l, at, Fixed::No)) if at == t => {
                Ok((pname, Sym::Scalar(l, t, Fixed::No)))
            }
            _ => self.not_yet(
                arg,
                format!(
                    "a reference argument that is not a plain local of the parameter's type `{t:?}`"
                ),
            ),
        }
    }

    /// The struct a parameter declaration is typed with, if it is one of this source's structs.
    fn struct_type_of(&self, p: Node<'_>) -> Option<String> {
        let t = p
            .child_by_field_name("type")
            .filter(|t| t.kind() == "type_identifier")?;
        let name = text(t, self.src);
        self.is_struct(name).then(|| name.to_string())
    }

    /// A struct parameter: `S& s` aliases the caller's instance, `const S& s` too (it is read-only), `S s` is a copy.
    fn bind_struct_argument(
        &mut self,
        p: Node<'_>,
        arg: Node<'_>,
        sname: &str,
        out: &mut Vec<Stmt>,
    ) -> R<(String, Sym)> {
        let Some(d) = p.child_by_field_name("declarator") else {
            return self.not_yet(p, "an unnamed parameter");
        };
        let (by_ref, id) = match d.kind() {
            "identifier" => (false, Some(d)),
            "reference_declarator" => (true, d.named_child(0).filter(|i| i.kind() == "identifier")),
            other => return self.not_yet(d, format!("a `{other}` parameter of struct type")),
        };
        let Some(id) = id else {
            return self.not_yet(d, "a struct parameter of an unexpected shape");
        };
        let pname = text(id, self.src).to_string();
        let from = self.struct_value(arg, sname)?;
        if by_ref {
            return Ok((pname, Sym::Struct(from)));
        }
        let copy = self.new_struct(sname, &pname, arg, out)?;
        self.copy_struct(&copy, &from, out)?;
        Ok((pname, Sym::Struct(copy)))
    }

    /// `T* name` bound to a pointer argument: the pointer, its offset pinned in a local (so it cannot move afterwards).
    fn bind_pointer_argument(
        &mut self,
        d: Node<'_>,
        ty: E,
        arg: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<(String, Sym)> {
        let id = d
            .child_by_field_name("declarator")
            .filter(|i| i.kind() == "identifier");
        let Some(id) = id else {
            return self.not_yet(d, "a pointer parameter of an unexpected shape");
        };
        let Some(mut p) = self.ptr_expr(arg)? else {
            return self.not_yet(
                arg,
                "a pointer parameter given something that is not a pointer expression",
            );
        };
        if p.pointee != ty {
            return self.not_yet(
                arg,
                format!(
                    "an implicit pointer conversion from {:?}* to {ty:?}* (cast it)",
                    p.pointee
                ),
            );
        }
        out.append(&mut self.hoisted);
        p.off = self.pin_offset(p.off, out);
        Ok((text(id, self.src).to_string(), Sym::Pointer(p)))
    }

    /// A callee's `return` is allowed only as its last top-level statement (with a value iff it is not `void`).
    fn check_returns(&self, body: Node<'_>, has_value: bool) -> R<()> {
        let last = u32::try_from(body.named_child_count().saturating_sub(1))
            .ok()
            .and_then(|i| body.named_child(i));
        let mut stack = vec![body];
        while let Some(n) = stack.pop() {
            if n.kind() == "return_statement" {
                let is_last = last.is_some_and(|l| l.id() == n.id());
                if !is_last {
                    return self.not_yet(
                        n,
                        "a `return` that is not the last statement of an inlined function",
                    );
                }
                if has_value != (n.named_child_count() > 0) {
                    return self.not_yet(
                        n,
                        "a `return` whose value does not match the function's type",
                    );
                }
            }
            let mut c = n.walk();
            stack.extend(n.named_children(&mut c));
        }
        Ok(())
    }

    /// The callee's statements into `out`; the trailing `return`'s value as a fresh local.
    fn callee_body(
        &mut self,
        body: Node<'_>,
        ret_ty: Option<E>,
        out: &mut Vec<Stmt>,
    ) -> R<Option<(ScalarExpr, E)>> {
        self.scopes.push(HashMap::new());
        let mut cursor = body.walk();
        let kids: Vec<_> = body.named_children(&mut cursor).collect();
        let mut result = Ok(None);
        for k in kids {
            if k.kind() == "return_statement" {
                result = self.callee_return(k, ret_ty, out);
                break;
            }
            if let Err(e) = self.stmt(k, out) {
                result = Err(e);
                break;
            }
        }
        self.scopes.pop();
        result
    }

    fn callee_return(
        &mut self,
        r: Node<'_>,
        ret_ty: Option<E>,
        out: &mut Vec<Stmt>,
    ) -> R<Option<(ScalarExpr, E)>> {
        let (Some(ty), Some(v)) = (ret_ty, r.named_child(0)) else {
            return Ok(None);
        };
        let (e, et) = self.expr(v)?;
        let e = self.convert(e, et, ty, v)?;
        out.append(&mut self.hoisted);
        let local = self.new_local(ty);
        out.push(Stmt::Let { local, init: e });
        Ok(Some((ScalarExpr::Local(local), ty)))
    }
}
