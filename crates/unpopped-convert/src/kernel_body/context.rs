// SPDX-License-Identifier: MIT OR Apache-2.0
//! What a CUDA source does not carry, given by the caller, and the preprocessor (ROADMAP U19 slice 5, pm-kernels Q7,
//! `docs/options-u19-slice5-decode-census.md`). A [`LiftContext`] holds the launch shape, the dynamic shared-memory size in bytes
//! (`extern __shared__ T a[];` is sized by the launch) and the integer macro definitions the source's `#if` and `#ifdef` read
//! (`__CUDACC_VER_MAJOR__`, `__CUDA_ARCH__`). Before parsing, [`preprocess`] keeps only the **active** branch of every conditional
//! (evaluating the condition over the caller's definitions and then the file's own `#define`s, in order) and blanks the rest,
//! directive lines included, to spaces: every byte offset and line number is unchanged, so a diagnostic still points at the
//! source line. A source that defines its own feature macros lifts as it would compile. A condition it cannot evaluate declines by name; a
//! function-like macro, or one with a non-integer body, is recorded and declines at its **use** (a file's unused macros never
//! stop a kernel: the census over Baracuda's real corpus, 2026-11-22, found 85% of kernels refused for a macro they never used). Only integer macros are modelled; `#include` is dropped, `#pragma` is kept.

use std::collections::HashMap;
use unpopped::exec::LaunchShape;

/// The launch shape, dynamic shared-memory size and macro definitions a lift assumes.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LiftContext {
    /// The launch the kernel runs under; `None` for [`crate::lift_kernel`] (no builtins, no shared memory).
    pub launch: Option<LaunchShape>,
    /// Bytes of dynamic shared memory (`extern __shared__ T a[];`), `0` when the launch gives none.
    pub dynamic_shared_bytes: u32,
    /// Integer macro definitions, in addition to the source's own `#define`s.
    pub defines: Vec<(String, i64)>,
    /// Buffer parameters that are **null** in this lift (an optional buffer the launch does not bind): a null check on one is false.
    /// Every other buffer is non-null.
    pub null_buffers: Vec<String>,
}

impl LiftContext {
    /// A context for a kernel that runs under `launch`, with no dynamic shared memory and no predefined macros.
    #[must_use]
    pub fn new(launch: LaunchShape) -> Self {
        Self {
            launch: Some(launch),
            dynamic_shared_bytes: 0,
            defines: Vec::new(),
            null_buffers: Vec::new(),
        }
    }

    /// The context of a kernel with no launch.
    pub(crate) fn without_launch() -> Self {
        Self {
            launch: None,
            dynamic_shared_bytes: 0,
            defines: Vec::new(),
            null_buffers: Vec::new(),
        }
    }

    /// With `bytes` of dynamic shared memory.
    #[must_use]
    pub fn with_dynamic_shared_bytes(mut self, bytes: u32) -> Self {
        self.dynamic_shared_bytes = bytes;
        self
    }

    /// With the buffer parameter `name` null (a `if (p && ...)` check on it is false).
    #[must_use]
    pub fn with_null_buffer(mut self, name: impl Into<String>) -> Self {
        self.null_buffers.push(name.into());
        self
    }

    /// With the macro `name` defined as `value` (a flag such as `FOO` is `1`).
    #[must_use]
    pub fn with_define(mut self, name: impl Into<String>, value: i64) -> Self {
        self.defines.push((name.into(), value));
        self
    }
}

type Defs = HashMap<String, i64>;

/// One `#if` level: whether its current branch is taken, whether any branch was, and whether the enclosing code is active.
struct Frame {
    taken: bool,
    any: bool,
    parent_active: bool,
}

/// The source with every inactive region and every directive (but an active `#pragma`) blanked to spaces, or the first thing it
/// could not model. Offsets and line numbers are preserved.
/// The macros the pass could not expand, by name, each with why (`the function-like macro `F``): a definition is recorded, never an
/// error, because most kernels never use most of a file's macros. The walker declines at a **use**.
pub(super) type Unexpanded = Vec<(String, String)>;

pub(super) fn preprocess(
    src: &str,
    predefined: &[(String, i64)],
) -> Result<(String, Unexpanded), String> {
    let mut defs: Defs = predefined.iter().cloned().collect();
    let mut unexpanded: Unexpanded = Vec::new();
    let mut stack: Vec<Frame> = Vec::new();
    let mut out = String::with_capacity(src.len());
    let mut continued = false;
    for line in src.split_inclusive('\n') {
        let body = line.trim_end_matches(['\n', '\r']);
        let active = stack.iter().all(|f| f.taken);
        let directive = continued || body.trim_start().starts_with('#');
        if !directive {
            out.push_str(&if active {
                line.to_string()
            } else {
                blank(line)
            });
            continue;
        }
        let was_continued = continued;
        continued = body.trim_end().ends_with('\\');
        if was_continued {
            out.push_str(&blank(line));
            continue;
        }
        let keep = handle(
            body.trim_start()[1..].trim_start(),
            active,
            &mut stack,
            (&mut defs, &mut unexpanded),
        )?;
        out.push_str(&if keep { line.to_string() } else { blank(line) });
    }
    if stack.is_empty() {
        Ok((out, unexpanded))
    } else {
        Err("an `#if` with no `#endif`".to_string())
    }
}

/// `line` with every character but the line ending replaced by a space.
fn blank(line: &str) -> String {
    line.chars()
        .map(|c| if c == '\n' || c == '\r' { c } else { ' ' })
        .collect()
}

/// One directive (the text after `#`); `true` if the line is kept verbatim (an active `#pragma`).
fn handle(
    d: &str,
    active: bool,
    stack: &mut Vec<Frame>,
    (defs, unexpanded): (&mut Defs, &mut Unexpanded),
) -> Result<bool, String> {
    let kw: String = d.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
    let rest = strip_comment(d[kw.len()..].trim());
    match kw.as_str() {
        "if" | "ifdef" | "ifndef" => {
            let t = active && cond_truth(&kw, rest, defs)?;
            stack.push(Frame {
                taken: t,
                any: t,
                parent_active: active,
            });
        }
        "elif" => {
            let f = stack.last_mut().ok_or("an `#elif` with no `#if`")?;
            let t = f.parent_active && !f.any && Expr::new(rest, defs).value()? != 0;
            f.taken = t;
            f.any |= t;
        }
        "else" => {
            let f = stack.last_mut().ok_or("an `#else` with no `#if`")?;
            f.taken = f.parent_active && !f.any;
            f.any = true;
        }
        "endif" => {
            stack.pop().ok_or("an `#endif` with no `#if`")?;
        }
        "define" if active => define(rest, defs, unexpanded)?,
        "undef" if active => {
            defs.remove(rest.split_whitespace().next().unwrap_or(""));
        }
        "pragma" => return Ok(active),
        "include" | "define" | "undef" => {}
        _ if !active => {}
        other => return Err(format!("the preprocessor directive `#{other}`")),
    }
    Ok(false)
}

fn cond_truth(kw: &str, rest: &str, defs: &Defs) -> Result<bool, String> {
    match kw {
        "ifdef" => Ok(defs.contains_key(rest.split_whitespace().next().unwrap_or(""))),
        "ifndef" => Ok(!defs.contains_key(rest.split_whitespace().next().unwrap_or(""))),
        _ => Ok(Expr::new(rest, defs).value()? != 0),
    }
}

/// `#define NAME`, `#define NAME <integer>`; a macro with parameters or a non-integer body declines.
fn define(rest: &str, defs: &mut Defs, unexpanded: &mut Unexpanded) -> Result<(), String> {
    let name: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    let after = &rest[name.len()..];
    if after.starts_with('(') {
        unexpanded.push((name.clone(), format!("the function-like macro `{name}`")));
        defs.remove(&name);
        return Ok(());
    }
    let v = after.trim();
    let value = if v.is_empty() {
        1
    } else if let Ok(n) = v.parse::<i64>() {
        n
    } else {
        unexpanded.push((
            name.clone(),
            format!("the macro `{name}` with the non-integer body `{v}`"),
        ));
        defs.remove(&name);
        return Ok(());
    };
    unexpanded.retain(|(n, _)| *n != name);
    defs.insert(name, value);
    Ok(())
}

fn strip_comment(s: &str) -> &str {
    let s = s.split("//").next().unwrap_or("");
    s.split("/*").next().unwrap_or("").trim()
}

/// A recursive-descent evaluator of an integer preprocessor expression.
struct Expr<'a> {
    t: Vec<String>,
    i: usize,
    defs: &'a Defs,
}

impl<'a> Expr<'a> {
    fn new(src: &str, defs: &'a Defs) -> Self {
        let mut t = Vec::new();
        let cs: Vec<char> = src.chars().collect();
        let mut i = 0;
        while i < cs.len() {
            let c = cs[i];
            if c.is_whitespace() {
                i += 1;
            } else if c.is_ascii_alphanumeric() || c == '_' {
                let j = (i..cs.len())
                    .find(|&j| !(cs[j].is_ascii_alphanumeric() || cs[j] == '_'))
                    .unwrap_or(cs.len());
                t.push(cs[i..j].iter().collect());
                i = j;
            } else {
                let two: String = cs[i..(i + 2).min(cs.len())].iter().collect();
                if matches!(two.as_str(), "&&" | "||" | "==" | "!=" | "<=" | ">=") {
                    t.push(two);
                    i += 2;
                } else {
                    t.push(c.to_string());
                    i += 1;
                }
            }
        }
        Self { t, i: 0, defs }
    }

    fn peek(&self) -> &str {
        self.t.get(self.i).map_or("", String::as_str)
    }

    fn eat(&mut self, s: &str) -> bool {
        let hit = self.peek() == s;
        if hit {
            self.i += 1;
        }
        hit
    }

    fn value(&mut self) -> Result<i64, String> {
        let v = self.or()?;
        if self.i == self.t.len() {
            Ok(v)
        } else {
            Err(format!("the preprocessor condition `{}`", self.t.join(" ")))
        }
    }

    fn or(&mut self) -> Result<i64, String> {
        let mut v = self.and()?;
        while self.eat("||") {
            let r = self.and()?;
            v = i64::from(v != 0 || r != 0);
        }
        Ok(v)
    }

    fn and(&mut self) -> Result<i64, String> {
        let mut v = self.cmp()?;
        while self.eat("&&") {
            let r = self.cmp()?;
            v = i64::from(v != 0 && r != 0);
        }
        Ok(v)
    }

    fn cmp(&mut self) -> Result<i64, String> {
        let mut v = self.sum()?;
        loop {
            let op = self.peek().to_string();
            if !matches!(op.as_str(), "==" | "!=" | "<" | "<=" | ">" | ">=") {
                return Ok(v);
            }
            self.i += 1;
            let r = self.sum()?;
            v = i64::from(match op.as_str() {
                "==" => v == r,
                "!=" => v != r,
                "<" => v < r,
                "<=" => v <= r,
                ">" => v > r,
                _ => v >= r,
            });
        }
    }

    fn sum(&mut self) -> Result<i64, String> {
        let mut v = self.prod()?;
        loop {
            if self.eat("+") {
                v += self.prod()?;
            } else if self.eat("-") {
                v -= self.prod()?;
            } else {
                return Ok(v);
            }
        }
    }

    fn prod(&mut self) -> Result<i64, String> {
        let mut v = self.unary()?;
        while self.eat("*") {
            v *= self.unary()?;
        }
        Ok(v)
    }

    fn unary(&mut self) -> Result<i64, String> {
        if self.eat("!") {
            return Ok(i64::from(self.unary()? == 0));
        }
        if self.eat("-") {
            return Ok(-self.unary()?);
        }
        if self.eat("(") {
            let v = self.or()?;
            return if self.eat(")") {
                Ok(v)
            } else {
                Err("an unbalanced `(` in a preprocessor condition".to_string())
            };
        }
        let tok = self.peek().to_string();
        self.i += 1;
        if tok == "defined" {
            let paren = self.eat("(");
            let name = self.peek().to_string();
            self.i += 1;
            if paren && !self.eat(")") {
                return Err("an unbalanced `defined(`".to_string());
            }
            return Ok(i64::from(self.defs.contains_key(&name)));
        }
        if let Ok(n) = tok.trim_end_matches(['u', 'U', 'l', 'L']).parse::<i64>() {
            return Ok(n);
        }
        if tok
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        {
            return Ok(self.defs.get(&tok).copied().unwrap_or(0));
        }
        Err(format!("the token `{tok}` in a preprocessor condition"))
    }
}
