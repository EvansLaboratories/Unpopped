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
        let lambda = func.kind() == "lambda_expression";
        let plist = callee_params(func);
        let Some(body) = func.child_by_field_name("body") else {
            return self.not_yet(call, format!("`{name}` without a parameter list or body"));
        };
        // `[&] { ... }` has no parameter list; a function always does
        if plist.is_none() && !lambda {
            return self.not_yet(call, format!("`{name}` without a parameter list or body"));
        }
        if lambda {
            self.check_lambda_env(call, name)?;
        }
        let ret_ty = self.return_type(func)?;
        let args = self.call_args(call);
        // a comment among the parameters (`float /*unused*/`) is not one
        let params: Vec<_> = plist
            .map(|pl| {
                let mut cursor = pl.walk();
                pl.named_children(&mut cursor)
                    .filter(|p| p.kind() != "comment")
                    .collect()
            })
            .unwrap_or_default();
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
            // an unnamed parameter (`float /*p1*/`) is passed and never read: the argument is still evaluated, as C does
            if p.child_by_field_name("declarator").is_none() && self.struct_type_of(p).is_none() {
                self.expr(a)?;
                continue;
            }
            let (pname, sym) = self.bind_argument(p, a, out)?;
            bound.insert(pname, sym);
        }
        let exits = ret_ty.is_none() && self.inline_stack.is_empty() && is_kernel_tail_call(call);
        self.inline_stack.push(name.to_string());
        // a lambda sees the caller's scopes (it captures by reference): its parameters go on top, and the scopes come back as the
        // body left them, so a captured pointer it advanced stays advanced
        let saved = if lambda {
            self.scopes.push(bound);
            Vec::new()
        } else {
            std::mem::replace(&mut self.scopes, vec![bound])
        };
        let saved_exit = std::mem::replace(&mut self.callee_exits_kernel, exits);
        let r = self.callee_body(body, ret_ty, out);
        self.callee_exits_kernel = saved_exit;
        if lambda {
            self.scopes.pop();
        } else {
            self.scopes = saved;
        }
        self.inline_stack.pop();
        r
    }

    /// `auto name = [&](params) -> T { body };` (also `[]`): the lambda becomes an inlinable function under its name. It sees the
    /// caller's locals by reference; an explicit capture list, a capture by value and a deduced non-`void` return type decline.
    /// `None` when the declaration is not of that shape.
    pub(super) fn lambda_declaration(&mut self, n: Node<'_>) -> Option<R<()>> {
        let mut c = n.walk();
        let decls: Vec<_> = n.children_by_field_name("declarator", &mut c).collect();
        let [d] = decls.as_slice() else {
            return None;
        };
        let value = d
            .child_by_field_name("value")
            .filter(|v| d.kind() == "init_declarator" && v.kind() == "lambda_expression")?;
        let id = d.child_by_field_name("declarator")?;
        let lname = text(id, self.src).to_string();
        let caps = value
            .child_by_field_name("captures")
            .map_or("[]", |c| text(c, self.src));
        let caps: String = caps.split_whitespace().collect();
        if caps != "[&]" && caps != "[]" {
            return Some(self.not_yet(
                value,
                "a lambda with an explicit capture list or a capture by value",
            ));
        }
        let body = value.child_by_field_name("body")?;
        let trailing = value.child_by_field_name("declarator").is_some_and(|d| {
            d.named_children(&mut d.walk())
                .any(|k| k.kind() == "trailing_return_type")
        });
        if !trailing && returns_value(body) {
            return Some(self.not_yet(
                value,
                "a lambda whose return type is deduced (write `-> T`)",
            ));
        }
        if self.functions.contains_key(&lname) && !self.lambda_env.contains_key(&lname) {
            return Some(self.not_yet(
                value,
                format!("a lambda named `{lname}`, which is also a function of the source"),
            ));
        }
        // the same node of the tree, with its lifetime: found again by byte range
        let root = self.root?;
        let node = root
            .descendant_for_byte_range(value.start_byte(), value.end_byte())
            .filter(|f| f.kind() == "lambda_expression" && f.start_byte() == value.start_byte())?;
        let env = self
            .scopes
            .iter()
            .flat_map(|s| s.iter())
            .map(|(k, v)| {
                let l = if let Sym::Scalar(l, ..) = v {
                    Some(*l)
                } else {
                    None
                };
                (k.clone(), l)
            })
            .collect();
        self.functions.insert(lname.clone(), node);
        self.lambda_env.insert(lname, env);
        Some(Ok(()))
    }

    /// A call of a lambda declines when a scalar it captured is shadowed at the call (the body would read the wrong one).
    fn check_lambda_env(&self, call: Node<'_>, name: &str) -> R<()> {
        let Some(env) = self.lambda_env.get(name) else {
            return Ok(());
        };
        for (k, id) in env {
            let Some(id) = id else { continue };
            if !matches!(self.lookup(k), Some(Sym::Scalar(l, ..)) if l == *id) {
                return self.not_yet(
                    call,
                    format!(
                        "a call of lambda `{name}` where its captured `{k}` is shadowed or gone"
                    ),
                );
            }
        }
        Ok(())
    }

    /// `void` is `None`; otherwise the element type.
    fn return_type(&self, func: Node<'_>) -> R<Option<E>> {
        if func.kind() == "lambda_expression" {
            let ty = func
                .child_by_field_name("declarator")
                .and_then(|d| {
                    d.named_children(&mut d.walk())
                        .find(|k| k.kind() == "trailing_return_type")
                })
                .and_then(|t| t.named_child(0))
                .and_then(|td| td.child_by_field_name("type"));
            return match ty {
                Some(t) if text(t, self.src) != "void" => self.type_of(Some(t), func).map(Some),
                _ => Ok(None),
            };
        }
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

/// Whether `body` has a `return` that carries a value (a nested lambda's own returns do not count).
fn returns_value(body: Node<'_>) -> bool {
    if body.kind() == "lambda_expression" {
        return false;
    }
    if body.kind() == "return_statement" && body.named_child_count() > 0 {
        return true;
    }
    let mut c = body.walk();
    body.named_children(&mut c).any(returns_value)
}
