// SPDX-License-Identifier: MIT OR Apache-2.0
//! Inlining of the source's own functions (ROADMAP U19 slice 4c-2, `docs/options-u19-slice5-decode-census.md`): a call to a
//! function defined in the same source is replaced by its body, in the caller's statement list. Scalar parameters are bound
//! by value (a fresh local holding the converted argument) or, for `T&`, to the caller's own local, so an assignment through
//! the reference reaches the caller. A callee may `return` once, as its last statement. Pointer and struct parameters,
//! recursion and a `return` anywhere else decline (4c-4 for pointers).

use super::structs::StructInst;
use super::{Fixed, R, Sym, Walker, text};
use std::collections::HashMap;
use std::rc::Rc;
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

/// Whether the call statement `call` is the last statement of the kernel's own body (not inside a loop, an `if` or a block), so a
/// `return` in the callee ends the kernel too.
fn is_kernel_tail_call(call: Node<'_>) -> bool {
    let Some(stmt) = call.parent().filter(|p| p.kind() == "expression_statement") else {
        return false;
    };
    let Some(block) = stmt.parent().filter(|b| b.kind() == "compound_statement") else {
        return false;
    };
    block
        .parent()
        .is_some_and(|f| f.kind() == "function_definition")
        && u32::try_from(block.named_child_count().saturating_sub(1))
            .ok()
            .and_then(|last| block.named_child(last))
            == Some(stmt)
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
            return self.not_yet(
                call,
                self.unexpanded_or(name, format!("a call to `{name}`")),
            );
        };
        self.inline_function(call, func, name, None, out)
    }

    /// Inline the method `mname` of `this` (a missing method declines by name).
    pub(super) fn inline_method(
        &mut self,
        call: Node<'_>,
        this: &Rc<StructInst>,
        mname: &str,
        out: &mut Vec<Stmt>,
    ) -> R<Option<(ScalarExpr, E)>> {
        let func = self
            .methods
            .get(&this.sname)
            .and_then(|m| m.get(mname))
            .copied();
        let Some(func) = func else {
            return self.not_yet(
                call,
                format!(
                    "the member `{mname}` of struct `{}`, which has no such method",
                    this.sname
                ),
            );
        };
        let name = format!("{}::{mname}", this.sname);
        self.inline_function(call, func, &name, Some(this), out)
    }

    fn inline_function(
        &mut self,
        call: Node<'_>,
        func: Node<'t>,
        name: &str,
        this: Option<&Rc<StructInst>>,
        out: &mut Vec<Stmt>,
    ) -> R<Option<(ScalarExpr, E)>> {
        if self.inline_stack.len() >= MAX_DEPTH || self.inline_stack.iter().any(|n| n == name) {
            return self.not_yet(call, format!("a recursive call to `{name}`"));
        }
        let (Some(plist), Some(body)) = (callee_params(func), func.child_by_field_name("body"))
        else {
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
        if let Some(inst) = this {
            // inside a method the instance's own fields are in scope by name
            for (n, s) in &inst.fields {
                bound.insert(n.clone(), s.clone());
            }
        }
        for (p, a) in params.into_iter().zip(args) {
            let (pname, sym) = self.bind_argument(p, a, out)?;
            bound.insert(pname, sym);
        }
        self.check_returns(body, ret_ty.is_some())?;
        let exits = ret_ty.is_none() && self.inline_stack.is_empty() && is_kernel_tail_call(call);
        self.inline_stack.push(name.to_string());
        let saved = std::mem::replace(&mut self.scopes, vec![bound]);
        let saved_exit = std::mem::replace(&mut self.callee_exits_kernel, exits);
        let r = self.callee_body(body, ret_ty, out);
        self.callee_exits_kernel = saved_exit;
        self.scopes = saved;
        self.inline_stack.pop();
        r
    }

    /// `void` is `None`; otherwise the element type.
    fn return_type(&self, func: Node<'_>) -> R<Option<E>> {
        let conv = func
            .child_by_field_name("declarator")
            .filter(|d| d.kind() == "operator_cast");
        if let Some(c) = conv {
            return self.type_of(c.child_by_field_name("type"), c).map(Some);
        }
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
        let Some(d) = p.child_by_field_name("declarator") else {
            return self.not_yet(p, "an unnamed parameter");
        };
        if let Some(shape) = self.vector_param_shape(p) {
            return self.bind_vector_argument(p, d, shape, arg, out);
        }
        // `void*` takes any pointer argument unchanged (the helper casts it to what it reads; census slice 3)
        let void_ptr = d.kind() == "pointer_declarator"
            && p.child_by_field_name("type")
                .is_some_and(|t| text(t, self.src).trim() == "void");
        if void_ptr {
            return self.bind_pointer_argument(d, None, arg, out);
        }
        let ty = self.type_of(p.child_by_field_name("type"), p)?;
        if d.kind() == "pointer_declarator" {
            return self.bind_pointer_argument(d, Some(ty), arg, out);
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

    /// The shape of a parameter typed as a CUDA vector (`float2`, or an alias of it: `dfloat2`), if it is one.
    fn vector_param_shape(&self, p: Node<'_>) -> Option<(E, u8)> {
        let t = p.child_by_field_name("type")?;
        let mut name = text(t, self.src)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        for _ in 0..8 {
            match self.type_aliases.get(&name) {
                Some(next) => name = next.clone(),
                None => break,
            }
        }
        super::vector_type(&name)
    }

    /// A vector parameter: `V& v` aliases the caller's vector local (it must have the same shape), `V v` and `const V& v` are a
    /// copy of the argument's lanes in fresh locals.
    fn bind_vector_argument(
        &mut self,
        p: Node<'_>,
        d: Node<'_>,
        (elem, lanes): (E, u8),
        arg: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<(String, Sym)> {
        let (by_ref, id) = match d.kind() {
            "identifier" => (false, Some(d)),
            "reference_declarator" => (
                !super::has_const(p, self.src),
                d.named_child(0).filter(|i| i.kind() == "identifier"),
            ),
            other => return self.not_yet(d, format!("a `{other}` parameter of vector type")),
        };
        let Some(id) = id else {
            return self.not_yet(d, "a vector parameter of an unexpected shape");
        };
        let pname = text(id, self.src).to_string();
        if by_ref {
            let target = (arg.kind() == "identifier")
                .then(|| self.lookup(text(arg, self.src)))
                .flatten();
            return match target {
                Some(Sym::Vector(ls, e, k)) if (e, k) == (elem, lanes) => {
                    Ok((pname, Sym::Vector(ls, e, k)))
                }
                _ => self.not_yet(
                    arg,
                    format!(
                        "a reference argument that is not a {lanes}-lane {elem:?} vector local"
                    ),
                ),
            };
        }
        let values = self.vector_value(arg, (elem, lanes))?;
        out.append(&mut self.hoisted);
        let mut ls = [0u32; 4];
        for (k, v) in values.into_iter().enumerate() {
            ls[k] = self.new_local(elem);
            out.push(Stmt::Let {
                local: ls[k],
                init: v,
            });
        }
        Ok((pname, Sym::Vector(ls, elem, lanes)))
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
        let t = p.child_by_field_name("type")?;
        self.struct_named(t).map(str::to_string)
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
        let copy = self.clone_struct(&from, &pname, arg, out)?;
        Ok((pname, Sym::Struct(copy)))
    }

    /// `T* name` bound to a pointer argument: the pointer, its offset pinned in a local (so it cannot move afterwards).
    fn bind_pointer_argument(
        &mut self,
        d: Node<'_>,
        ty: Option<E>,
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
        if let Some(ty) = ty.filter(|t| *t != p.pointee) {
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
}

/// The parameter list of a function or method (a conversion operator keeps it one level down).
pub(super) fn callee_params(func: Node<'_>) -> Option<Node<'_>> {
    let d = func.child_by_field_name("declarator")?;
    let d = if d.kind() == "operator_cast" {
        d.child_by_field_name("declarator")?
    } else {
        d
    };
    d.child_by_field_name("parameters")
}
