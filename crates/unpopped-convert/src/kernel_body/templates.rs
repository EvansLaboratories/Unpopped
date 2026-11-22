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

/// The call sites to rewrite this round, as `(start, end, instance name)`, in source order.
fn call_edits(tree: &Tree, src: &str, inst: &mut Instances) -> Vec<(usize, usize, String)> {
    let mut templates: HashMap<String, Node<'_>> = HashMap::new();
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
                match (is_template, id) {
                    (true, Some(id)) => {
                        templates.insert(text(id, src).to_string(), n);
                        continue;
                    }
                    _ => {}
                }
            }
            "call_expression" => {
                if n.child_by_field_name("function")
                    .is_some_and(|f| f.kind() == "template_function")
                {
                    calls.push(n);
                }
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
        let (Some(name), Some(args)) = (
            f.child_by_field_name("name"),
            f.child_by_field_name("arguments"),
        ) else {
            continue;
        };
        let helper = text(name, src);
        let Some(&def) = templates.get(helper) else {
            continue;
        };
        let mut c = args.walk();
        let given: Vec<String> = args
            .named_children(&mut c)
            .map(|a| text(a, src).trim().to_string())
            .collect();
        if let Some(instance) = instance_of(src, def, helper, &given, inst) {
            edits.push((f.start_byte(), f.end_byte(), instance));
        }
    }
    edits.sort_by_key(|e| e.0);
    edits
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
    inst.text.push_str("\n");
    inst.text.push_str(&substitute(&body[..a], &subs));
    inst.text.push_str(&name);
    inst.text.push_str(&substitute(&body[b..], &subs));
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
                Some(a) => format!("({a})"),
                None => p.default.clone()?,
            };
            Some((p.name.clone(), v))
        })
        .collect()
}
