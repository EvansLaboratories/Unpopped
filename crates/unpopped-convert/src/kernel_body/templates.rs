// SPDX-License-Identifier: MIT OR Apache-2.0
//! Template instantiation from the caller (ROADMAP U19 census slice 1, `docs/options-u19-census.md`: 391 kernels stopped on
//! "template"). A `template <typename T, int N> __global__ void k(...)` is not a kernel until its parameters have values, and a
//! CUDA source does not carry them (the host launch's instantiation does), so the caller gives them
//! (`LiftContext::with_type`, `with_const`). The substitution is **textual, inside the kernel's own `template` declaration only**
//! (a helper template elsewhere in the file keeps its own `T`), whole identifiers only, line numbers unchanged; the result is
//! re-parsed. A parameter with no argument takes its default, and one with neither declines by name. An argument naming no
//! parameter declines too: a stale instantiation is a bug the caller should hear about, not a value to ignore.

use super::context::LiftContext;
use super::{KernelLiftError, text};
use std::collections::HashMap;
use tree_sitter::{Node, Tree};

type Res<T> = Result<T, KernelLiftError>;

fn decline<T>(what: String) -> Res<T> {
    Err(KernelLiftError::NotYetLifted {
        construct: what,
        line: 0,
    })
}

/// One template parameter of the kernel: its name and, when it has one, its default text.
struct TParam {
    name: String,
    is_type: bool,
    default: Option<String>,
}

/// The kernel's `template<...>` parameters, or `None` for a function that is not directly under a template declaration.
fn params_of<'t>(func: Node<'t>, src: &str) -> Res<Option<(Node<'t>, Vec<TParam>)>> {
    let Some(decl) = func.parent().filter(|p| p.kind() == "template_declaration") else {
        return Ok(None);
    };
    params_of_decl(decl, src)
}

/// The `template<...>` parameters of the template declaration `decl`.
fn params_of_decl<'t>(decl: Node<'t>, src: &str) -> Res<Option<(Node<'t>, Vec<TParam>)>> {
    let Some(list) = decl.child_by_field_name("parameters") else {
        return Ok(None);
    };
    let mut out = Vec::new();
    let mut c = list.walk();
    for p in list.named_children(&mut c) {
        let field = |f: &str| p.child_by_field_name(f).map(|n| text(n, src).to_string());
        let one = match p.kind() {
            "type_parameter_declaration" => {
                let mut k = p.walk();
                let name = p
                    .named_children(&mut k)
                    .find(|n| n.kind() == "type_identifier")
                    .map(|n| text(n, src).to_string());
                name.map(|name| TParam {
                    name,
                    is_type: true,
                    default: None,
                })
            }
            "optional_type_parameter_declaration" => field("name").map(|name| TParam {
                name,
                is_type: true,
                default: field("default_type"),
            }),
            "parameter_declaration" => field("declarator").map(|name| TParam {
                name,
                is_type: false,
                default: None,
            }),
            "optional_parameter_declaration" => field("declarator").map(|name| TParam {
                name,
                is_type: false,
                default: field("default_value"),
            }),
            other => return decline(format!("the template parameter form `{other}`")),
        };
        match one {
            Some(t) => out.push(t),
            None => return decline("an unnamed template parameter".to_string()),
        }
    }
    Ok(Some((decl, out)))
}

/// The text for each parameter, from the caller's arguments, then the defaults.
fn resolve(params: &[TParam], ctx: &LiftContext, kernel: &str) -> Res<Vec<(String, String)>> {
    for (n, _) in &ctx.type_args {
        if !params.iter().any(|p| p.is_type && &p.name == n) {
            return decline(format!(
                "the template type argument `{n}`: `{kernel}` has no type parameter of that name"
            ));
        }
    }
    for (n, _) in &ctx.const_args {
        if !params.iter().any(|p| !p.is_type && &p.name == n) {
            return decline(format!(
                "the template value argument `{n}`: `{kernel}` has no non-type parameter of that name"
            ));
        }
    }
    let mut out = Vec::new();
    for p in params {
        let given = if p.is_type {
            ctx.type_args
                .iter()
                .rev()
                .find(|(n, _)| *n == p.name)
                .map(|(_, t)| t.clone())
        } else {
            ctx.const_args
                .iter()
                .rev()
                .find(|(n, _)| *n == p.name)
                .map(|(_, v)| format!("({v})"))
        };
        match given.or_else(|| p.default.clone()) {
            Some(v) => out.push((p.name.clone(), v)),
            None => {
                let how = if p.is_type { "with_type" } else { "with_const" };
                return decline(format!(
                    "the template parameter `{}` of `{kernel}` has no value (give one with `LiftContext::{how}`)",
                    p.name
                ));
            }
        }
    }
    Ok(out)
}

/// `region` with every whole-identifier occurrence of a name replaced by its value (no newline is added or removed).
fn substitute(region: &str, subs: &[(String, String)]) -> String {
    let word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut out = String::with_capacity(region.len());
    let mut rest = region;
    while let Some(c) = rest.chars().next() {
        if word(c) {
            let end = rest.find(|c: char| !word(c)).unwrap_or(rest.len());
            let (id, tail) = rest.split_at(end);
            match subs.iter().find(|(n, _)| n == id) {
                Some((_, v)) => out.push_str(v),
                None => out.push_str(id),
            }
            rest = tail;
        } else {
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    out
}

/// The source with the kernel's template instantiated, `None` when the kernel is not a template and the caller gave no arguments.
pub(super) fn instantiate(
    src: &str,
    func: Node<'_>,
    kernel: &str,
    ctx: &LiftContext,
) -> Res<Option<String>> {
    let Some((decl, params)) = params_of(func, src)? else {
        if ctx.type_args.is_empty() && ctx.const_args.is_empty() {
            return Ok(None);
        }
        return decline(format!(
            "template arguments were given but `{kernel}` is not a template"
        ));
    };
    let subs = resolve(&params, ctx, kernel)?;
    // the header's own text (`template <typename T, int N = 4>`) is replaced by blanks: it has no meaning once instantiated
    let (start, end) = (decl.start_byte(), decl.end_byte());
    let body_start = func.start_byte();
    let header: String = src[start..body_start]
        .chars()
        .map(|c| if c == '\n' || c == '\r' { c } else { ' ' })
        .collect();
    let mut out = String::with_capacity(src.len());
    out.push_str(&src[..start]);
    out.push_str(&header);
    out.push_str(&substitute(&src[body_start..end], &subs));
    out.push_str(&src[end..]);
    Ok(Some(out))
}

// ---- calls with explicit template arguments (census slice 2) ----

/// Rounds of rewriting before a chain of template helpers is declined (each round resolves one nesting level).
const MAX_ROUNDS: usize = 16;

/// Rounds of class template rewriting: an argument's own parentheses are normalised so an instance key never grows.
const MAX_TYPE_ROUNDS: usize = 16;

/// The instance each distinct `(helper, arguments)` became: a plain function appended to the source, so it is walked and inlined
/// like any other helper.
#[derive(Default)]
struct Instances {
    by_key: HashMap<String, String>,
    text: String,
}

/// Rewrite every call with explicit template arguments (`helper<float, 4>(x)`) to a **plain function** that is the helper
/// instantiated with those arguments (`helper__t0`), appended to the source; repeat until no such call is left (an instance can
/// itself call a template helper). A call whose helper is not a template defined in the file, or whose arguments do not fit its
/// parameters, is left as it is and declines at its use. Calls without explicit arguments (`helper(x)`, deduced) are not modelled.
///
/// # Errors
/// A chain of instantiations deeper than [`MAX_ROUNDS`], or a source that stops parsing.
pub(super) fn expand_calls(parse: fn(&str) -> Option<Tree>, mut src: String) -> Res<String> {
    let mut inst = Instances::default();
    for _ in 0..MAX_ROUNDS {
        let tree = parse(&src).ok_or(KernelLiftError::SyntaxError { line: 0 })?;
        let edits = call_edits(&tree, &src, &mut inst);
        if edits.is_empty() {
            return Ok(src);
        }
        for (start, end, name) in edits.into_iter().rev() {
            src.replace_range(start..end, &name);
        }
        src.push_str(&std::mem::take(&mut inst.text));
    }
    decline(format!(
        "template helper calls nested deeper than {MAX_ROUNDS} levels"
    ))
}

/// The `f<T>` of a callee spelled `f<T>` or `ns::inner::f<T>` (a template helper is found by its last segment, since helpers
/// are collected by short name across namespaces).
fn template_callee(f: Node<'_>) -> Option<Node<'_>> {
    match f.kind() {
        "template_function" => Some(f),
        "qualified_identifier" => f.child_by_field_name("name").and_then(template_callee),
        _ => None,
    }
}

/// The call sites to rewrite this round, as `(start, end, instance name)`, in source order.
fn call_edits(tree: &Tree, src: &str, inst: &mut Instances) -> Vec<(usize, usize, String)> {
    let mut templates: HashMap<String, Node<'_>> = HashMap::new();
    let mut specs: HashMap<String, (Node<'_>, Node<'_>)> = HashMap::new();
    let mut calls: Vec<Node<'_>> = Vec::new();
    let mut stack = vec![tree.root_node()];
    while let Some(n) = stack.pop() {
        match n.kind() {
            "function_definition" => {
                let id = n
                    .child_by_field_name("declarator")
                    .and_then(|d| d.child_by_field_name("declarator"));
                let is_template = n
                    .parent()
                    .is_some_and(|p| p.kind() == "template_declaration");
                if let (true, Some(id)) = (is_template, id) {
                    if id.kind() == "template_function" {
                        // an explicit specialisation (`template <> float f<double>(double)`): kept apart, keyed by its arguments
                        if let Some(key) = specialisation_key(id, src) {
                            specs.insert(key, (n, id));
                        }
                    } else {
                        templates.insert(text(id, src).to_string(), n);
                    }
                    continue;
                }
            }
            "call_expression"
                if n.child_by_field_name("function")
                    .and_then(template_callee)
                    .is_some() =>
            {
                calls.push(n);
            }
            _ => {}
        }
        let mut c = n.walk();
        stack.extend(n.children(&mut c));
    }
    let mut edits = Vec::new();
    for call in calls {
        let Some(f) = call.child_by_field_name("function") else {
            continue;
        };
        // the call's callee, whole (`ns::f<T>` is replaced with its qualifier), and the `f<T>` inside it
        let Some(tf) = template_callee(f) else {
            continue;
        };
        let (Some(name), Some(args)) = (
            tf.child_by_field_name("name"),
            tf.child_by_field_name("arguments"),
        ) else {
            continue;
        };
        let helper = text(name, src);
        let mut c = args.walk();
        let given: Vec<String> = args
            .named_children(&mut c)
            .map(|a| text(a, src).trim().to_string())
            .collect();
        // an explicit specialisation with exactly these arguments wins over the primary template
        let key = format!("{helper}<{}>", given.join(","));
        let instance = match (specs.get(&key), templates.get(helper)) {
            (Some(&(def, id)), _) => specialisation_instance(src, def, id, &key, inst),
            (None, Some(&def)) => instance_of(src, def, helper, &given, inst),
            (None, None) => None,
        };
        if let Some(instance) = instance {
            edits.push((f.start_byte(), f.end_byte(), instance));
        }
    }
    edits.sort_by_key(|e| e.0);
    edits
}

/// `name<a,b>` of an explicit specialisation's declarator, spelled as a call's key is.
fn specialisation_key(id: Node<'_>, src: &str) -> Option<String> {
    let name = text(id.child_by_field_name("name")?, src);
    let args = id.child_by_field_name("arguments")?;
    let mut c = args.walk();
    let parts: Vec<String> = args
        .named_children(&mut c)
        .map(|a| text(a, src).trim().to_string())
        .collect();
    Some(format!("{name}<{}>", parts.join(",")))
}

/// An explicit specialisation as a plain function (its declarator `f<double>` renamed to the instance's name).
fn specialisation_instance(
    src: &str,
    def: Node<'_>,
    id: Node<'_>,
    key: &str,
    inst: &mut Instances,
) -> Option<String> {
    if let Some(n) = inst.by_key.get(key) {
        return Some(n.clone());
    }
    let name = format!(
        "{}__t{}",
        text(id.child_by_field_name("name")?, src),
        inst.by_key.len()
    );
    let (a, b) = (
        id.start_byte() - def.start_byte(),
        id.end_byte() - def.start_byte(),
    );
    let body = &src[def.start_byte()..def.end_byte()];
    inst.text.push('\n');
    inst.text.push_str(&body[..a]);
    inst.text.push_str(&name);
    inst.text.push_str(&body[b..]);
    inst.by_key.insert(key.to_string(), name.clone());
    Some(name)
}

/// The name of `helper` instantiated with `given`, creating it on first use; `None` when the arguments do not fit.
fn instance_of(
    src: &str,
    def: Node<'_>,
    helper: &str,
    given: &[String],
    inst: &mut Instances,
) -> Option<String> {
    let key = format!("{helper}<{}>", given.join(","));
    if let Some(n) = inst.by_key.get(&key) {
        return Some(n.clone());
    }
    let (decl, params) = params_of(def, src).ok().flatten()?;
    let subs = bind_explicit(&params, given)?;
    let id = def
        .child_by_field_name("declarator")?
        .child_by_field_name("declarator")?;
    let name = format!("{helper}__t{}", inst.by_key.len());
    let body = &src[def.start_byte()..decl.end_byte()];
    let (a, b) = (
        id.start_byte() - def.start_byte(),
        id.end_byte() - def.start_byte(),
    );
    inst.text.push('\n');
    inst.text.push_str(&substitute(&body[..a], &subs));
    inst.text.push_str(&name);
    inst.text.push_str(&substitute(&body[b..], &subs));
    inst.by_key.insert(key, name.clone());
    Some(name)
}

// ---- class templates (U38b-1) ----

/// A class template of the file: its primary definition, or a partial or explicit specialisation with its pattern arguments.
struct ClassTemplate<'t> {
    spec: Node<'t>,
    decl: Node<'t>,
    name: Node<'t>,
    pattern: Option<Vec<String>>,
}

/// Replace every use of a class template with explicit arguments (`vec_t<float, 128>`) by a **plain struct** that is the template
/// instantiated with them (`vec_t__t0`), appended to the source: the explicit specialisation or partial specialisation whose
/// pattern matches (`template <size_t n> struct vec_t<float, n>` matches `vec_t<float, 128>` with `n` = 128), else the primary
/// template. The instance is then walked like any struct, its member functions included. Repeats until no use is left (an
/// instance can itself name another instance). A use inside another template's body is left for that template's own
/// instantiation; a use that matches nothing is left and declines at its use.
///
/// # Errors
/// A chain of instantiations deeper than [`MAX_ROUNDS`], or a source that stops parsing.
pub(super) fn expand_types(parse: fn(&str) -> Option<Tree>, mut src: String) -> Res<String> {
    let mut inst = Instances::default();
    for _ in 0..MAX_TYPE_ROUNDS {
        let tree = parse(&src).ok_or(KernelLiftError::SyntaxError { line: 0 })?;
        let edits = type_edits(&tree, &src, &mut inst);
        if edits.is_empty() {
            return Ok(src);
        }
        for (start, end, name) in edits.into_iter().rev() {
            src.replace_range(start..end, &name);
        }
        src.push_str(&std::mem::take(&mut inst.text));
    }
    decline(format!(
        "class templates nested deeper than {MAX_TYPE_ROUNDS} levels"
    ))
}

fn type_edits(tree: &Tree, src: &str, inst: &mut Instances) -> Vec<(usize, usize, String)> {
    let mut primary: HashMap<String, ClassTemplate<'_>> = HashMap::new();
    let mut specs: HashMap<String, Vec<ClassTemplate<'_>>> = HashMap::new();
    let mut uses: Vec<Node<'_>> = Vec::new();
    let mut members: HashMap<String, Vec<MemberDef>> = HashMap::new();
    let mut stack = vec![tree.root_node()];
    while let Some(n) = stack.pop() {
        match n.kind() {
            // generic code (a function or class template) is instantiated as a whole, not rewritten in place
            "template_declaration" => {
                register_class(n, src, &mut primary, &mut specs);
                register_member_specialization(n, src, &mut members);
                continue;
            }
            "template_type" => {
                uses.push(n);
                continue;
            }
            _ => {}
        }
        let mut c = n.walk();
        stack.extend(n.children(&mut c));
    }
    let mut edits: Vec<(usize, usize, String)> = Vec::new();
    for u in uses {
        let (Some(name), Some(args)) = (
            u.child_by_field_name("name"),
            u.child_by_field_name("arguments"),
        ) else {
            continue;
        };
        let class = text(name, src);
        let given = template_args(args, src);
        if let Some(instance) =
            class_instance(src, class, &given, (&primary, &specs, &members), inst)
        {
            edits.push((u.start_byte(), u.end_byte(), instance));
        }
    }
    edits.sort_by_key(|e| e.0);
    edits
}

/// Registers the class (or struct) a template declaration defines, if it defines one: a primary template, or a specialisation
/// keyed by its class name with its pattern arguments.
fn register_class<'t>(
    decl: Node<'t>,
    src: &str,
    primary: &mut HashMap<String, ClassTemplate<'t>>,
    specs: &mut HashMap<String, Vec<ClassTemplate<'t>>>,
) {
    let mut c = decl.walk();
    let mut cands: Vec<Node<'t>> = decl.named_children(&mut c).collect();
    let mut deeper: Vec<Node<'t>> = Vec::new();
    for k in &cands {
        let mut c2 = k.walk();
        deeper.extend(k.named_children(&mut c2));
    }
    cands.extend(deeper);
    for n in cands {
        if !matches!(n.kind(), "struct_specifier" | "class_specifier") {
            continue;
        }
        let (Some(name), Some(_)) = (n.child_by_field_name("name"), n.child_by_field_name("body"))
        else {
            continue;
        };
        if name.kind() == "template_type" {
            let key = name
                .child_by_field_name("name")
                .map(|k| text(k, src).to_string());
            let pattern = name
                .child_by_field_name("arguments")
                .map(|a| template_args(a, src));
            if let (Some(key), Some(pattern)) = (key, pattern) {
                specs.entry(key).or_default().push(ClassTemplate {
                    spec: n,
                    decl,
                    name,
                    pattern: Some(pattern),
                });
            }
        } else {
            primary.insert(
                text(name, src).to_string(),
                ClassTemplate {
                    spec: n,
                    decl,
                    name,
                    pattern: None,
                },
            );
        }
        return;
    }
}

/// An out-of-class definition of a member of an explicit specialisation of a class template, as an in-class one.
struct MemberDef {
    name: String,
    text: String,
}

/// `template <> float AccMin<float>::identity() { ... }`: keyed by the class use it names (`AccMin<float>`), with the qualifier removed.
fn register_member_specialization(
    decl: Node<'_>,
    src: &str,
    out: &mut HashMap<String, Vec<MemberDef>>,
) {
    // only `template <>`: a partial member specialisation is not modelled
    let Ok(Some((_, params))) = params_of_decl(decl, src) else {
        return;
    };
    if !params.is_empty() {
        return;
    }
    let mut c = decl.walk();
    let Some(def) = decl
        .named_children(&mut c)
        .find(|k| k.kind() == "function_definition")
    else {
        return;
    };
    let Some(q) = def
        .child_by_field_name("declarator")
        .filter(|d| d.kind() == "function_declarator")
        .and_then(|fd| fd.child_by_field_name("declarator"))
        .filter(|d| d.kind() == "qualified_identifier")
    else {
        return;
    };
    let (Some(scope), Some(name)) = (
        q.child_by_field_name("scope"),
        q.child_by_field_name("name"),
    ) else {
        return;
    };
    let (Some(cname), Some(args)) = (
        scope
            .child_by_field_name("name")
            .filter(|_| scope.kind() == "template_type"),
        scope.child_by_field_name("arguments"),
    ) else {
        return;
    };
    let key = format!(
        "{}<{}>",
        text(cname, src),
        template_args(args, src)
            .iter()
            .map(|g| bare(g))
            .collect::<Vec<_>>()
            .join(",")
    );
    let body = format!(
        "{}{}{}",
        &src[def.start_byte()..q.start_byte()],
        text(name, src),
        &src[q.end_byte()..def.end_byte()]
    );
    out.entry(key).or_default().push(MemberDef {
        name: text(name, src).to_string(),
        text: body,
    });
}

/// Whether `body` (a class body) declares `name` as a `static` member: the statement holding `name(` mentions `static`.
fn declared_static(body: &str, name: &str) -> bool {
    let needle = format!("{name}(");
    body.match_indices(&needle).any(|(i, _)| {
        let before = &body[..i];
        if before
            .chars()
            .next_back()
            .is_some_and(|c| c.is_alphanumeric() || c == '_')
        {
            return false;
        }
        let start = before.rfind([';', '{', '}']).map_or(0, |p| p + 1);
        before[start..].contains("static")
    })
}

/// The texts of a template argument list, trimmed.
fn template_args(list: Node<'_>, src: &str) -> Vec<String> {
    let mut c = list.walk();
    list.named_children(&mut c)
        .map(|a| text(a, src).trim().to_string())
        .collect()
}

/// `x` without whitespace and without any outer parentheses that enclose the whole text (an argument substituted as `(128)` or
/// `((128))` equals `128`; `(a)+(b)` keeps its parentheses).
fn bare(x: &str) -> String {
    let mut s: String = x.split_whitespace().collect();
    loop {
        let Some(inner) = s.strip_prefix('(').and_then(|r| r.strip_suffix(')')) else {
            return s;
        };
        // the opening parenthesis must close at the very end, not earlier
        let mut depth = 0i32;
        let encloses = inner.chars().all(|c| {
            depth += i32::from(c == '(') - i32::from(c == ')');
            depth >= 0
        }) && depth == 0;
        if !encloses {
            return s;
        }
        s = inner.to_string();
    }
}

/// The bindings a specialisation's pattern gives for `given`, and how many pattern arguments were fixed (not parameters); `None`
/// when it does not match.
fn match_pattern(
    params: &[TParam],
    pattern: &[String],
    given: &[String],
) -> Option<(Vec<(String, String)>, usize)> {
    if pattern.len() != given.len() {
        return None;
    }
    let mut subs: Vec<(String, String)> = Vec::new();
    let mut fixed = 0;
    for (pat, giv) in pattern.iter().zip(given) {
        match params.iter().find(|p| p.name == *pat) {
            Some(p) => {
                let v = if p.is_type {
                    giv.clone()
                } else {
                    format!("({})", bare(giv))
                };
                match subs.iter().find(|(n, _)| n == pat) {
                    Some((_, old)) if bare(old) != bare(&v) => return None,
                    Some(_) => {}
                    None => subs.push((pat.clone(), v)),
                }
            }
            None if bare(pat) == bare(giv) => fixed += 1,
            None => return None,
        }
    }
    Some((subs, fixed))
}

/// The name of the class template `class` instantiated with `given`, creating the plain struct on first use.
fn class_instance(
    src: &str,
    class: &str,
    given: &[String],
    (primary, specs, members): (
        &HashMap<String, ClassTemplate<'_>>,
        &HashMap<String, Vec<ClassTemplate<'_>>>,
        &HashMap<String, Vec<MemberDef>>,
    ),
    inst: &mut Instances,
) -> Option<String> {
    let key = format!(
        "{class}<{}>",
        given.iter().map(|g| bare(g)).collect::<Vec<_>>().join(",")
    );
    if let Some(n) = inst.by_key.get(&key) {
        return Some(n.clone());
    }
    let mut best: Option<(&ClassTemplate<'_>, Vec<(String, String)>, usize)> = None;
    for sp in specs.get(class).into_iter().flatten() {
        let (_, params) = params_of_decl(sp.decl, src).ok().flatten()?;
        let pattern = sp.pattern.as_deref()?;
        if let Some((subs, fixed)) = match_pattern(&params, pattern, given)
            && best.as_ref().is_none_or(|b| fixed > b.2)
        {
            best = Some((sp, subs, fixed));
        }
    }
    let (ct, subs) = match best {
        Some((ct, subs, _)) => (ct, subs),
        None => {
            let ct = primary.get(class)?;
            let (_, params) = params_of_decl(ct.decl, src).ok().flatten()?;
            let subs = bind_explicit(&params, given)?;
            (ct, subs)
        }
    };
    let name = format!("{class}__t{}", inst.by_key.len());
    let (spec, id) = (ct.spec, ct.name);
    inst.text.push('\n');
    inst.text
        .push_str(&substitute(&src[spec.start_byte()..id.start_byte()], &subs));
    inst.text.push_str(&name);
    let mut body = substitute(&src[id.end_byte()..spec.end_byte()], &subs);
    // out-of-class definitions of this instance's members (`template <> float AccMin<float>::identity() { .. }`) go inside it
    if let (Some(defs), Some(close)) = (members.get(&key), body.rfind('}')) {
        let mut extra = String::new();
        for d in defs {
            if declared_static(&body, &d.name) && !d.text.trim_start().starts_with("static") {
                extra.push_str("static ");
            }
            extra.push_str(&d.text);
            extra.push('\n');
        }
        body.insert_str(close, &extra);
    }
    inst.text.push_str(&body);
    inst.text.push(';');
    inst.by_key.insert(key, name.clone());
    Some(name)
}

/// Positional arguments to parameters (a non-type value parenthesised), the rest from defaults; `None` when there are too many
/// arguments or a parameter has neither.
fn bind_explicit(params: &[TParam], given: &[String]) -> Option<Vec<(String, String)>> {
    if given.len() > params.len() {
        return None;
    }
    params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let v = match given.get(i) {
                Some(a) if p.is_type => a.clone(),
                Some(a) => format!("({})", bare(a)),
                None => p.default.clone()?,
            };
            Some((p.name.clone(), v))
        })
        .collect()
}
