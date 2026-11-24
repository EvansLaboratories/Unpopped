// SPDX-License-Identifier: MIT OR Apache-2.0
//! Member functions of structs (ROADMAP U19 slice 4e): `obj.f(args)` inlines the method with `this` bound to the instance, so the
//! method's own fields (`d_`, `m_`) name the instance's scalar locals and an assignment through them writes the instance. A
//! struct's one conversion operator (`operator unsigned int() const`) is inlined where an instance is used as a number.
//! Constructors, destructors, operators other than the conversion, virtual and static members, and calls through a pointer
//! decline by name.

use super::structs::StructInst;
use super::{R, Sym, Walker, text};
use std::collections::HashMap;
use std::rc::Rc;
use tree_sitter::Node;
use unpopped::ir::ScalarExpr;
use unpopped::kernel::Stmt;
use unpopped_vocab::ElementKind as E;

/// The key a struct's conversion operator is stored under.
pub(super) const CONVERSION: &str = "operator";

/// Every method of every struct, by struct name then method name (`CONVERSION` for the conversion operator).
pub(super) fn collect_methods<'t>(
    root: Node<'t>,
    src: &str,
) -> HashMap<String, HashMap<String, Node<'t>>> {
    let mut out: HashMap<String, HashMap<String, Node<'t>>> = HashMap::new();
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if n.kind() == "struct_specifier" {
            if let (Some(name), Some(body)) =
                (n.child_by_field_name("name"), n.child_by_field_name("body"))
            {
                let mut c = body.walk();
                for f in body
                    .named_children(&mut c)
                    .filter(|f| f.kind() == "function_definition")
                {
                    if let Some(key) = method_key(f, src) {
                        out.entry(text(name, src).to_string())
                            .or_default()
                            .insert(key, f);
                    }
                }
            }
        }
        let mut c = n.walk();
        stack.extend(n.children(&mut c));
    }
    out
}

/// The `static` members among `methods`, each as `Struct::name` and its definition (called without an instance).
pub(super) fn static_members<'t>(
    methods: &HashMap<String, HashMap<String, Node<'t>>>,
    src: &str,
) -> Vec<(String, Node<'t>)> {
    let mut out = Vec::new();
    for (sname, ms) in methods {
        for (mname, f) in ms {
            let mut c = f.walk();
            let is_static = f
                .children(&mut c)
                .any(|k| k.kind() == "storage_class_specifier" && text(k, src) == "static");
            if is_static {
                out.push((format!("{sname}::{mname}"), *f));
            }
        }
    }
    out
}

fn method_key(f: Node<'_>, src: &str) -> Option<String> {
    let d = f.child_by_field_name("declarator")?;
    match d.kind() {
        "operator_cast" => Some(CONVERSION.to_string()),
        "function_declarator" => d
            .child_by_field_name("declarator")
            .filter(|i| i.kind() == "field_identifier" || i.kind() == "operator_name")
            .map(|i| text(i, src).split_whitespace().collect::<String>()),
        _ => None,
    }
}

impl<'s, 't> Walker<'s, 't> {
    /// The struct instance a call's object expression names (`a`, `a.b`).
    fn object_of(&self, n: Node<'_>) -> R<Option<Rc<StructInst>>> {
        let sym = match n.kind() {
            "identifier" => self.lookup(text(n, self.src)),
            "field_expression" => self.resolve_member(n)?,
            _ => None,
        };
        Ok(match sym {
            Some(Sym::Struct(i)) => Some(i),
            _ => None,
        })
    }

    /// `op(args)` where `op` is a struct instance (a functor parameter or local) with an `operator()`: the method, inlined. `None`
    /// when `callee` is not such a name.
    pub(super) fn functor_call(
        &mut self,
        call: Node<'_>,
        callee: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> Option<R<Option<(ScalarExpr, E)>>> {
        if callee.kind() != "identifier" {
            return None;
        }
        let Some(Sym::Struct(inst)) = self.lookup(text(callee, self.src)) else {
            return None;
        };
        self.methods
            .get(&inst.sname)
            .is_some_and(|m| m.contains_key("operator()"))
            .then(|| {
                self.flush(out);
                self.inline_method(call, &inst, "operator()", out)
            })
    }

    /// `obj.f(args);` as a statement.
    pub(super) fn method_statement(
        &mut self,
        call: Node<'_>,
        callee: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        self.method_call(call, callee, out).map(|_| ())
    }

    /// `obj.f(args)` as a value: the method's `return`, hoisted.
    pub(super) fn method_value(&mut self, call: Node<'_>, callee: Node<'_>) -> R<(ScalarExpr, E)> {
        let mut body = Vec::new();
        let r = self.method_call(call, callee, &mut body)?;
        self.hoisted.append(&mut body);
        match r {
            Some(v) => Ok(v),
            None => self.not_yet(
                call,
                format!(
                    "the result of `{}`, which returns void",
                    text(callee, self.src)
                ),
            ),
        }
    }

    fn method_call(
        &mut self,
        call: Node<'_>,
        callee: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<Option<(ScalarExpr, E)>> {
        let (Some(obj), Some(m)) = (
            callee.child_by_field_name("argument"),
            callee.child_by_field_name("field"),
        ) else {
            return self.not_yet(call, "a member call of an unexpected shape");
        };
        let Some(inst) = self.object_of(obj)? else {
            return self.not_yet(
                call,
                format!(
                    "a call to `{}`, whose object is not a struct local, parameter or member",
                    text(callee, self.src)
                ),
            );
        };
        self.flush(out);
        self.inline_method(call, &inst, text(m, self.src), out)
    }

    /// A struct instance used as a number: its conversion operator, inlined.
    pub(super) fn struct_as_value(
        &mut self,
        inst: &Rc<StructInst>,
        at: Node<'_>,
    ) -> R<(ScalarExpr, E)> {
        let has_conversion = self
            .methods
            .get(&inst.sname)
            .is_some_and(|m| m.contains_key(CONVERSION));
        if !has_conversion {
            return self.not_yet(
                at,
                format!(
                    "the struct `{}` used as a value (it has no conversion operator)",
                    inst.sname
                ),
            );
        }
        let mut body = Vec::new();
        let r = self.inline_method(at, inst, CONVERSION, &mut body)?;
        self.hoisted.append(&mut body);
        match r {
            Some(v) => Ok(v),
            None => self.not_yet(
                at,
                format!(
                    "the struct `{}` used as a value (it has no conversion operator)",
                    inst.sname
                ),
            ),
        }
    }
}
