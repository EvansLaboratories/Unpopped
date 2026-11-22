// SPDX-License-Identifier: MIT OR Apache-2.0
//! Type aliases, `static_assert` and the C library's limit constants (ROADMAP U19 census slices 5/6: `using T = ...;` stopped 27 of
//! Baracuda's kernels, `static_assert` 21, `INFINITY` and kin 14). An alias (`using AccT = float;`, `typedef float floatX;`, at file
//! scope or in the function) is resolved where a type name is read; `static_assert` is a compile-time check the real compiler already
//! passed, so it lifts to nothing unless it evaluates to **false** under the caller's arguments (a bad instantiation, reported);
//! `INFINITY`, `FLT_MAX`, `INT_MAX` and kin are constants.

use super::{R, Walker, text};
use std::collections::HashMap;
use tree_sitter::Node;
use unpopped::ir::ScalarExpr;
use unpopped_vocab::ElementKind as E;

/// The deepest alias chain followed (`using A = B; using B = C; ...`).
const MAX_ALIAS_DEPTH: usize = 8;

/// The file-scope aliases of the source (`using X = T;` and `typedef T X;`, outside every function), name to type text.
pub(super) fn collect_aliases(root: Node<'_>, src: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if let Some((name, ty)) = alias_of(n, src) {
            out.insert(name, ty);
        }
        if !matches!(n.kind(), "function_definition" | "compound_statement") {
            let mut c = n.walk();
            stack.extend(n.children(&mut c));
        }
    }
    out
}

/// The member types of the struct templates of the source, for `typename Trait<Args>::member`: by the struct's name, or by
/// `Name<args>` (spaces removed) for an explicit specialisation (`template <> struct Name<double> { using type = double; };`),
/// then member name to type text.
pub(super) type Traits = HashMap<String, HashMap<String, String>>;

pub(super) fn collect_traits(root: Node<'_>, src: &str) -> Traits {
    let mut out: Traits = HashMap::new();
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if matches!(n.kind(), "struct_specifier" | "class_specifier")
            && let (Some(name), Some(body)) =
                (n.child_by_field_name("name"), n.child_by_field_name("body"))
        {
            let key: String = text(name, src).split_whitespace().collect();
            let mut c = body.walk();
            for m in body.named_children(&mut c) {
                if let Some((member, ty)) = alias_of(m, src) {
                    out.entry(key.clone()).or_default().insert(member, ty);
                } else if let Some((member, entry)) = static_member(m, src) {
                    out.entry(key.clone()).or_default().insert(member, entry);
                }
            }
        }
        if !matches!(n.kind(), "function_definition" | "compound_statement") {
            let mut c = n.walk();
            stack.extend(n.children(&mut c));
        }
    }
    out
}

/// A `static constexpr T name = init;` member: `("#name", "T|init")`, the `#` keeping it apart from the type members.
fn static_member(m: Node<'_>, src: &str) -> Option<(String, String)> {
    if m.kind() != "field_declaration" {
        return None;
    }
    let ty = text(m.child_by_field_name("type")?, src).trim().to_string();
    let d = m.child_by_field_name("declarator")?;
    let (name, init) = match d.kind() {
        "init_declarator" => (
            d.child_by_field_name("declarator")?,
            d.child_by_field_name("value")?,
        ),
        _ => (d, m.child_by_field_name("default_value")?),
    };
    Some((
        format!("#{}", text(name, src).trim()),
        format!("{ty}|{}", text(init, src).trim()),
    ))
}

/// `(name, type text)` of an alias declaration node, if it is one.
fn alias_of(n: Node<'_>, src: &str) -> Option<(String, String)> {
    let (name, ty) = match n.kind() {
        "alias_declaration" => (
            n.child_by_field_name("name")?,
            n.child_by_field_name("type")?,
        ),
        "type_definition" => (
            n.child_by_field_name("declarator")?,
            n.child_by_field_name("type")?,
        ),
        _ => return None,
    };
    if name.kind() != "type_identifier" {
        return None;
    }
    let ty = text(ty, src)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    Some((text(name, src).to_string(), ty))
}

impl<'s, 't> Walker<'s, 't> {
    /// The element type a C type name denotes, through the aliases in scope.
    pub(super) fn ctype(&self, t: &str) -> Result<E, String> {
        let mut t = t.to_string();
        for _ in 0..MAX_ALIAS_DEPTH {
            match self.type_aliases.get(&t) {
                Some(next) => t = next.clone(),
                None => match self.trait_member(&t) {
                    Some(next) => t = next,
                    None => break,
                },
            }
        }
        super::c_type(&t)
    }

    /// `typename Name<args>::member` (or without `typename`): the member type of the explicit specialisation with exactly those
    /// arguments, else of the primary template. `None` for any other text, or a member the struct does not have.
    fn trait_member(&self, t: &str) -> Option<String> {
        let t = t.strip_prefix("typename ").unwrap_or(t);
        let (head, member) = t.rsplit_once("::")?;
        self.trait_entry(head, member.trim())
    }

    /// The entry `member` of the trait `head` (`Name<args>`, the arguments read through the type aliases in scope, so a template
    /// parameter bound to `float` is `float`): the specialisation with exactly those arguments, else the primary template.
    fn trait_entry(&self, head: &str, member: &str) -> Option<String> {
        let lt = head.find('<')?;
        if !head.ends_with('>') {
            return None;
        }
        let name = head[..lt].trim();
        let inner = &head[lt + 1..head.len() - 1];
        let mut args = Vec::new();
        let (mut depth, mut cur) = (0usize, String::new());
        for c in inner.chars() {
            match c {
                '<' | '(' => depth += 1,
                '>' | ')' => depth = depth.saturating_sub(1),
                ',' if depth == 0 => {
                    args.push(std::mem::take(&mut cur));
                    continue;
                }
                _ => {}
            }
            cur.push(c);
        }
        args.push(cur);
        let args: Vec<String> = args
            .iter()
            .map(|a| {
                let mut a = a.split_whitespace().collect::<Vec<_>>().join(" ");
                for _ in 0..MAX_ALIAS_DEPTH {
                    match self.type_aliases.get(&a) {
                        Some(next) => a = next.clone(),
                        None => break,
                    }
                }
                a.split_whitespace().collect::<String>()
            })
            .collect();
        let key = format!("{name}<{}>", args.join(","));
        self.traits
            .get(&key)
            .and_then(|m| m.get(member))
            .or_else(|| self.traits.get(name).and_then(|m| m.get(member)))
            .cloned()
    }

    /// `Trait<Args>::member` as a value: the trait's `static constexpr` integer member.
    pub(super) fn qualified_value(&self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let t = text(n, self.src).to_string();
        let found = t
            .rsplit_once("::")
            .and_then(|(head, member)| self.trait_entry(head, &format!("#{}", member.trim())));
        let Some((ty, init)) = found.as_deref().and_then(|e| e.split_once('|')) else {
            return self.not_yet(
                n,
                format!("a `qualified_identifier` expression `{}`", t.trim()),
            );
        };
        let value = match super::parse_literal(init) {
            Some(super::Lit::Int(v, _)) => v,
            _ => {
                return self.not_yet(
                    n,
                    format!(
                        "the member `{}` with the non-literal initialiser `{init}`",
                        t.trim()
                    ),
                );
            }
        };
        match self.ctype(ty) {
            Ok(e) if super::int_range(e).is_some() => Ok((ScalarExpr::ConstInt(value, e), e)),
            _ => self.not_yet(n, format!("the member `{}` of type `{ty}`", t.trim())),
        }
    }

    /// `using X = T;` or `typedef T X;` inside a function: the alias is in scope for the rest of the function.
    pub(super) fn alias_statement(&mut self, n: Node<'_>) -> R<()> {
        match alias_of(n, self.src) {
            Some((name, ty)) => {
                self.type_aliases.insert(name, ty);
                Ok(())
            }
            None => self.not_yet(n, "a type alias of an unexpected shape"),
        }
    }

    /// `static_assert(cond, "msg")`: nothing, unless its condition is a constant the walker can evaluate and it is **false**.
    pub(super) fn static_assert(&mut self, n: Node<'_>) -> R<()> {
        let Some(cond) = n.child_by_field_name("condition") else {
            return Ok(());
        };
        if self.const_value(cond).ok().flatten() == Some(0) {
            return self.not_yet(
                n,
                format!(
                    "a `static_assert` that does not hold for these arguments: `{}`",
                    text(cond, self.src).trim()
                ),
            );
        }
        Ok(())
    }

    /// `INFINITY`, `FLT_MAX`, `INT_MAX` and kin: a C library limit as a constant, or `None` for any other name.
    pub(super) fn limit_constant(&self, name: &str) -> Option<(ScalarExpr, E)> {
        let float = |v: f32| Some((ScalarExpr::Const(f64::from(v)), self.compute));
        let int = |v: i64, t: E| Some((ScalarExpr::ConstInt(v, t), t));
        match name {
            "INFINITY" if matches!(self.compute, E::F32 | E::F64) => {
                Some((ScalarExpr::Const(f64::INFINITY), self.compute))
            }
            "FLT_MAX" => float(f32::MAX),
            "FLT_MIN" => float(f32::MIN_POSITIVE),
            "FLT_EPSILON" => float(f32::EPSILON),
            // CUDA's `math_constants.h` single-precision names
            "CUDART_INF_F" => float(f32::INFINITY),
            "CUDART_NAN_F" => float(f32::NAN),
            "CUDART_PI_F" => float(std::f32::consts::PI),
            "CUDART_ZERO_F" => float(0.0),
            "CUDART_ONE_F" => float(1.0),
            "INT_MAX" => int(i64::from(i32::MAX), E::I32),
            "INT_MIN" => int(i64::from(i32::MIN), E::I32),
            "UINT_MAX" => int(i64::from(u32::MAX), E::U32),
            "LLONG_MAX" => int(i64::MAX, E::I64),
            "LLONG_MIN" => int(i64::MIN, E::I64),
            _ => None,
        }
    }
}
