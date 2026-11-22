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

use crate::KernelLiftError;
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
    /// Template **type** arguments of the kernel (`T` is `float`), by parameter name (ROADMAP U19 census slice 1).
    pub type_args: Vec<(String, String)>,
    /// Template **non-type** arguments of the kernel (`N` is `128`, `kCausal` is `1`), by parameter name.
    pub const_args: Vec<(String, i64)>,
    /// Include files the caller supplies, by the exact name the source writes (`#include "baracuda_loss.cuh"` is `baracuda_loss.cuh`).
    pub includes: Vec<(String, String)>,
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
            type_args: Vec::new(),
            const_args: Vec::new(),
            includes: Vec::new(),
        }
    }

    /// The context of a kernel with no launch.
    pub(crate) fn without_launch() -> Self {
        Self {
            launch: None,
            dynamic_shared_bytes: 0,
            defines: Vec::new(),
            null_buffers: Vec::new(),
            type_args: Vec::new(),
            const_args: Vec::new(),
            includes: Vec::new(),
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

    /// With the kernel's template type parameter `name` instantiated as the C type `ty` (`with_type("T", "float")`). A parameter
    /// with a default and no argument takes its default; one with neither declines.
    #[must_use]
    pub fn with_type(mut self, name: impl Into<String>, ty: impl Into<String>) -> Self {
        self.type_args.push((name.into(), ty.into()));
        self
    }

    /// With the kernel's template non-type parameter `name` instantiated as `value` (`with_const("N", 128)`; a `bool` is `0` or `1`).
    #[must_use]
    pub fn with_const(mut self, name: impl Into<String>, value: i64) -> Self {
        self.const_args.push((name.into(), value));
        self
    }

    /// With the file `name` available to `#include "name"` (or `<name>`): its `text` is spliced where the directive is, after being
    /// preprocessed itself (so a header's own guards, `#pragma once`, conditionals and nested includes work). **Nothing is read
    /// from disk.** The name is matched exactly; the same name given twice with different text declines. A quoted include
    /// the caller did not supply is named in the decline of anything the file could have defined (a call, a type, a name).
    #[must_use]
    pub fn with_include(mut self, name: impl Into<String>, text: impl Into<String>) -> Self {
        self.includes.push((name.into(), text.into()));
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

/// The macros the pass could not expand, by name, each with why (`the function-like macro `F``): a definition is recorded, never an
/// error, because most kernels never use most of a file's macros. The walker declines at a **use**.
pub(super) type Unexpanded = Vec<(String, String)>;

/// The deepest `#include` nesting spliced (a cycle with no guard reaches it and is named).
const MAX_INCLUDE_DEPTH: usize = 16;

/// Where one line of the preprocessed text came from: the file (`None` is the source being lifted) and its 1-based line there.
pub(super) type Origin = (Option<String>, usize);

/// What the preprocessor pass produced.
pub(super) struct Preprocessed {
    /// The source with every inactive region and directive (but an active `#pragma`) blanked to spaces, each supplied `#include`
    /// replaced by that file's own preprocessed text. For a source with no supplied include the offsets and line numbers equal the input's.
    pub(super) text: String,
    /// The macros it could not expand.
    pub(super) unexpanded: Unexpanded,
    /// Quoted includes (`#include "x.cuh"`) that were active and not supplied: the walker names them where it declines on
    /// something such a file could have defined. A `<system>` include is not listed (the toolchain's headers are modelled by the walker).
    pub(super) missing_includes: Vec<String>,
    /// One entry per line of `text`.
    pub(super) origins: Vec<Origin>,
    /// Object-like integer macros with ONE value over the whole preprocessed text (a `#define NAME <integer expression>` never
    /// redefined to another value, never `#undef`ed, not an empty flag), by name: what a use in code reads as. A caller's
    /// definition counts. A name redefined or removed is absent (a use declines).
    pub(super) macros: HashMap<String, i64>,
}

/// The pass's running state across the source and every file it splices.
struct Pass<'a> {
    includes: &'a [(String, String)],
    defs: Defs,
    unexpanded: Unexpanded,
    missing: Vec<String>,
    /// Files that said `#pragma once`: spliced at most once.
    once: Vec<String>,
    out: String,
    origins: Vec<Origin>,
    /// Every macro's value history: `Some(v)` while one value only, `None` once redefined, removed or empty.
    history: HashMap<String, Option<i64>>,
}

/// Run the preprocessor over `src` with the caller's macro definitions and supplied include files.
///
/// # Errors
/// An `#if` that cannot be evaluated, an unbalanced conditional, or an include nest deeper than [`MAX_INCLUDE_DEPTH`].
pub(super) fn preprocess(
    src: &str,
    predefined: &[(String, i64)],
    includes: &[(String, String)],
) -> Result<Preprocessed, String> {
    let mut pass = Pass {
        includes,
        defs: predefined.iter().cloned().collect(),
        unexpanded: Vec::new(),
        missing: Vec::new(),
        once: Vec::new(),
        out: String::with_capacity(src.len()),
        origins: Vec::new(),
        history: predefined
            .iter()
            .map(|(n, v)| (n.clone(), Some(*v)))
            .collect(),
    };
    pass.run(None, src, &mut Vec::new())?;
    Ok(Preprocessed {
        text: pass.out,
        unexpanded: pass.unexpanded,
        missing_includes: pass.missing,
        origins: pass.origins,
        macros: pass
            .history
            .into_iter()
            .filter_map(|(n, v)| v.map(|v| (n, v)))
            .collect(),
    })
}

impl Pass<'_> {
    fn emit(&mut self, line: &str, origin: (&Option<String>, usize)) {
        self.out.push_str(line);
        self.origins.push((origin.0.clone(), origin.1));
    }

    /// Note the macro a just-handled `#define`/`#undef` line `d` touched in its value history.
    fn record_macro(&mut self, d: &str) {
        let rest = d
            .trim_start_matches(|c: char| c.is_ascii_alphabetic())
            .trim_start();
        let name: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            return;
        }
        let body = strip_comment(&rest[name.len()..]);
        let now = if d.starts_with("define") && !body.is_empty() {
            self.defs.get(&name).copied()
        } else {
            None
        };
        let merged = match (self.history.get(&name), now) {
            (None, v) => v,
            (Some(Some(old)), Some(v)) if *old == v => Some(v),
            _ => None,
        };
        self.history.insert(name, merged);
    }

    /// A line ending, if the text so far does not end in one (a spliced file's last line may lack it).
    fn end_line(&mut self) {
        if !self.out.is_empty() && !self.out.ends_with('\n') {
            self.out.push('\n');
        }
    }

    /// One file's lines. `chain` is the include path that reached it (for a cycle's message). The conditionals must balance within the file.
    fn run(
        &mut self,
        file: Option<&str>,
        src: &str,
        chain: &mut Vec<String>,
    ) -> Result<(), String> {
        let origin = file.map(str::to_string);
        let mut stack: Vec<Frame> = Vec::new();
        let lines: Vec<&str> = src.split_inclusive('\n').collect();
        let mut idx = 0;
        while idx < lines.len() {
            let line = lines[idx];
            let at = (&origin, idx + 1);
            let body = line.trim_end_matches(['\n', '\r']);
            let active = stack.iter().all(|f| f.taken);
            if !body.trim_start().starts_with('#') {
                self.emit(
                    &if active {
                        line.to_string()
                    } else {
                        blank(line)
                    },
                    at,
                );
                idx += 1;
                continue;
            }
            // a directive may continue over lines ending in a backslash (`#if defined(A) && \`): evaluate the joined text and blank
            // every physical line, so the line count is unchanged
            let (joined, last) = join_directive(&lines, idx);
            let d = joined.trim_start()[1..].trim_start();
            let physical: Vec<(usize, &str)> = (idx..=last).map(|i| (i, lines[i])).collect();
            let blank_all = |pass: &mut Self| {
                for (i, l) in &physical {
                    pass.emit(&blank(l), (&origin, i + 1));
                }
            };
            idx = last + 1;
            if active && d.starts_with("include") {
                blank_all(self);
                self.include(d["include".len()..].trim(), chain)?;
                continue;
            }
            if active
                && file.is_some()
                && d.starts_with("pragma")
                && d["pragma".len()..].trim() == "once"
            {
                self.once.push(file.unwrap_or_default().to_string());
                blank_all(self);
                continue;
            }
            let defs_unexp = (&mut self.defs, &mut self.unexpanded);
            let keep = handle(d, active, &mut stack, defs_unexp)?;
            if active && (d.starts_with("define") || d.starts_with("undef")) {
                self.record_macro(d);
            }
            if keep && physical.len() == 1 {
                self.emit(line, at);
            } else {
                blank_all(self);
            }
        }
        if stack.is_empty() {
            Ok(())
        } else {
            Err("an `#if` with no `#endif`".to_string())
        }
    }

    /// `#include "name"` or `<name>`: the supplied file's text spliced here, exactly once for a `#pragma once` file.
    fn include(&mut self, arg: &str, chain: &mut Vec<String>) -> Result<(), String> {
        let (name, quoted) = match (arg.strip_prefix('"'), arg.strip_prefix('<')) {
            (Some(r), _) => (r.split('"').next().unwrap_or(""), true),
            (_, Some(r)) => (r.split('>').next().unwrap_or(""), false),
            _ => return Ok(()),
        };
        let found: Vec<&(String, String)> =
            self.includes.iter().filter(|(n, _)| n == name).collect();
        let Some((_, text)) = found.first().copied() else {
            if quoted && !self.missing.iter().any(|m| m == name) {
                self.missing.push(name.to_string());
            }
            return Ok(());
        };
        if found.iter().any(|(_, t)| t != text) {
            return Err(format!(
                "the include `{name}` was supplied twice with different text"
            ));
        }
        if self.once.iter().any(|o| o == name) {
            return Ok(());
        }
        if chain.len() >= MAX_INCLUDE_DEPTH {
            return Err(format!(
                "the include depth limit ({MAX_INCLUDE_DEPTH}) was exceeded (a cycle with no guard?): {} -> {name}",
                chain.join(" -> ")
            ));
        }
        chain.push(name.to_string());
        let text = text.clone();
        // each file starts on its own line, and the line after it does too, whether or not it ends in a newline
        self.end_line();
        let r = self.run(Some(name), &text, chain);
        self.end_line();
        chain.pop();
        r
    }
}

/// A lift error as the caller should see it: its line is the **source's own** line (a line inside a spliced include is reported
/// at the `#include` that brought it in, with the file and line named in the text), and a decline on something a file the
/// caller did not supply could have defined (a call, a type, a name) names that file.
pub(super) fn report(e: KernelLiftError, pre: &Preprocessed) -> KernelLiftError {
    let place = |line: usize| -> (usize, Option<String>) {
        let Some(row) = line.checked_sub(1).filter(|r| *r < pre.origins.len()) else {
            return (line, None);
        };
        match &pre.origins[row] {
            (None, n) => (*n, None),
            (Some(f), n) => {
                let at = pre.origins[..row]
                    .iter()
                    .rev()
                    .find_map(|(f, n)| f.is_none().then_some(*n))
                    .unwrap_or(0);
                (at, Some(format!(" (in the include `{f}`, line {n})")))
            }
        }
    };
    match e {
        KernelLiftError::NotYetLifted { construct, line } => {
            let (line, within) = place(line);
            let mut construct = construct;
            construct.push_str(&within.unwrap_or_default());
            let unknown = ["a call to ", "the type ", "the name "]
                .iter()
                .any(|p| construct.starts_with(p));
            if unknown && !pre.missing_includes.is_empty() {
                let names: Vec<String> = pre
                    .missing_includes
                    .iter()
                    .map(|n| format!("`{n}`"))
                    .collect();
                construct.push_str(&format!(
                    "; the source includes {}, which {} not supplied (`LiftContext::with_include`)",
                    names.join(", "),
                    if names.len() == 1 { "was" } else { "were" }
                ));
            }
            KernelLiftError::NotYetLifted { construct, line }
        }
        KernelLiftError::SyntaxError { line } => KernelLiftError::SyntaxError {
            line: place(line).0,
        },
        other => other,
    }
}

/// The directive that starts at line `start` of `lines`, with its backslash continuations joined into one, and the index of its last
/// physical line.
fn join_directive(lines: &[&str], start: usize) -> (String, usize) {
    let mut joined = String::new();
    let mut last = start;
    loop {
        let body = lines[last].trim_end_matches(['\n', '\r']);
        let trimmed = body.trim_end();
        if trimmed.ends_with('\\') && last + 1 < lines.len() {
            joined.push_str(trimmed.trim_end_matches('\\'));
            joined.push(' ');
            last += 1;
        } else {
            joined.push_str(body);
            return (joined, last);
        }
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
    } else if let Some(n) = constant_body(v, defs) {
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

/// The value of a macro body that is an integer expression over macros already defined (`(QK4_0 / (4 * QR4_0))`), or `None`.
/// A name that is not a known macro is not read as 0 here (it may be a constant variable, which the preprocessor does not see).
fn constant_body(body: &str, defs: &Defs) -> Option<i64> {
    // only a body that is one parenthesised group: an unparenthesised `1 + 2` would regroup inside `N * 2` where it is used
    let inner = body.trim().strip_prefix('(')?.strip_suffix(')')?;
    let mut depth = 0i32;
    for c in inner.chars() {
        depth += i32::from(c == '(') - i32::from(c == ')');
        if depth < 0 {
            return None;
        }
    }
    if depth != 0 {
        return None;
    }
    let mut word = String::new();
    for c in body.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_alphanumeric() || c == '_' {
            word.push(c);
            continue;
        }
        if !word.is_empty() {
            let starts_digit = word.chars().next().is_some_and(|d| d.is_ascii_digit());
            if !starts_digit && !defs.contains_key(&word) {
                return None;
            }
            word.clear();
        }
    }
    Expr::new(body, defs).value().ok()
}

/// The value of an integer constant expression of literals and operators only (`(128) / 4`, `2 * 8`): no name is read as a macro
/// or a constant. `None` for anything else (a name, a float, an unbalanced parenthesis).
pub(super) fn literal_expression(text: &str) -> Option<i64> {
    constant_body(&format!("({})", text.trim()), &Defs::new())
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
        loop {
            if self.eat("*") {
                v = v
                    .checked_mul(self.unary()?)
                    .ok_or("an overflow in a preprocessor expression")?;
            } else if self.eat("/") {
                let d = self.unary()?;
                v = v
                    .checked_div(d)
                    .ok_or("a division by zero in a preprocessor expression")?;
            } else if self.eat("%") {
                let d = self.unary()?;
                v = v
                    .checked_rem(d)
                    .ok_or("a remainder by zero in a preprocessor expression")?;
            } else {
                break;
            }
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

#[cfg(test)]
mod include_tests {
    use super::*;

    fn pp(src: &str, inc: &[(&str, &str)]) -> Result<Preprocessed, String> {
        let inc: Vec<(String, String)> = inc
            .iter()
            .map(|(n, t)| ((*n).into(), (*t).into()))
            .collect();
        preprocess(src, &[], &inc)
    }

    fn lines(p: &Preprocessed) -> Vec<&str> {
        p.text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect()
    }

    #[test]
    fn a_supplied_include_is_spliced_where_the_directive_is_and_by_exact_name() {
        let p = pp("a\n#include \"h.cuh\"\nb\n", &[("h.cuh", "H1\nH2\n")]).unwrap();
        assert_eq!(lines(&p), ["a", "H1", "H2", "b"]);
        // `<h.cuh>` is the same name; a different name is a different file
        assert_eq!(
            lines(&pp("#include <h.cuh>\n", &[("h.cuh", "H\n")]).unwrap()),
            ["H"]
        );
        let p = pp(
            "#include \"h.cuh\"\n",
            &[("x/h.cuh", "H\n"), ("H.cuh", "H\n"), ("h.cu", "H\n")],
        )
        .unwrap();
        assert!(lines(&p).is_empty(), "no near match resolves: {:?}", p.text);
        assert_eq!(p.missing_includes, ["h.cuh"]);
    }

    #[test]
    fn an_unsupplied_quoted_include_is_listed_and_a_system_include_is_not() {
        let p = pp(
            "#include <cuda_fp16.h>\n#include \"mine.cuh\"\n#include \"mine.cuh\"\n",
            &[],
        )
        .unwrap();
        assert_eq!(
            p.missing_includes,
            ["mine.cuh"],
            "named once, and the system header is not"
        );
        // an include in an inactive region is not needed
        let p = pp("#if 0\n#include \"never.cuh\"\n#endif\n", &[]).unwrap();
        assert!(p.missing_includes.is_empty());
    }

    #[test]
    fn a_guarded_or_pragma_once_header_is_spliced_once() {
        let guarded = "#ifndef G\n#define G\nBODY\n#endif\n";
        let p = pp("#include \"g.h\"\n#include \"g.h\"\n", &[("g.h", guarded)]).unwrap();
        assert_eq!(lines(&p), ["BODY"]);
        let once = "#pragma once\nBODY\n";
        let p = pp("#include \"o.h\"\n#include \"o.h\"\n", &[("o.h", once)]).unwrap();
        assert_eq!(lines(&p), ["BODY"]);
        // control: an unguarded header is spliced each time it is included
        let p = pp("#include \"u.h\"\n#include \"u.h\"\n", &[("u.h", "BODY\n")]).unwrap();
        assert_eq!(lines(&p), ["BODY", "BODY"]);
    }

    #[test]
    fn nested_includes_and_the_headers_own_conditionals_and_defines_work() {
        let outer = "#include \"inner.h\"\n#if INNER_VALUE >= 2\nBIG\n#else\nSMALL\n#endif\n";
        let inner = "#define INNER_VALUE 3\nINNER\n";
        let p = pp(
            "#include \"outer.h\"\nMAIN\n",
            &[("outer.h", outer), ("inner.h", inner)],
        )
        .unwrap();
        assert_eq!(lines(&p), ["INNER", "BIG", "MAIN"]);
    }

    #[test]
    fn a_cycle_with_no_guard_and_a_deep_nest_are_named_not_followed_forever() {
        let e = pp(
            "#include \"a.h\"\n",
            &[("a.h", "#include \"b.h\"\n"), ("b.h", "#include \"a.h\"\n")],
        )
        .err()
        .unwrap();
        assert!(
            e.contains("depth limit (16)") && e.contains("a.h -> b.h -> a.h"),
            "{e}"
        );
        // a guarded cycle is fine: the guard stops it
        let a = "#ifndef A\n#define A\n#include \"b.h\"\nA_BODY\n#endif\n";
        let b = "#ifndef B\n#define B\n#include \"a.h\"\nB_BODY\n#endif\n";
        let p = pp("#include \"a.h\"\n", &[("a.h", a), ("b.h", b)]).unwrap();
        assert_eq!(lines(&p), ["B_BODY", "A_BODY"]);
        // a chain of exactly the limit is accepted, one more is not
        let chain = |n: usize| -> Vec<(String, String)> {
            (0..n)
                .map(|i| {
                    (
                        format!("h{i}.h"),
                        if i + 1 < n {
                            format!("#include \"h{}.h\"\n", i + 1)
                        } else {
                            "LEAF\n".into()
                        },
                    )
                })
                .collect()
        };
        let run = |n: usize| {
            let inc = chain(n);
            preprocess("#include \"h0.h\"\n", &[], &inc)
        };
        assert_eq!(lines(&run(16).unwrap()), ["LEAF"]);
        assert!(run(17).is_err());
    }

    #[test]
    fn the_same_name_with_different_text_is_an_error_and_with_the_same_text_is_not() {
        let e = pp("#include \"h.h\"\n", &[("h.h", "A\n"), ("h.h", "B\n")])
            .err()
            .unwrap();
        assert!(e.contains("supplied twice with different text"), "{e}");
        assert_eq!(
            lines(&pp("#include \"h.h\"\n", &[("h.h", "A\n"), ("h.h", "A\n")]).unwrap()),
            ["A"]
        );
    }

    #[test]
    fn every_output_line_knows_where_it_came_from() {
        let p = pp("m1\n#include \"h.h\"\nm3", &[("h.h", "h1\nh2")]).unwrap();
        let o: Vec<(Option<&str>, usize)> =
            p.origins.iter().map(|(f, n)| (f.as_deref(), *n)).collect();
        // main line 1, the (blanked) directive line 2, the header's two lines, then main line 3
        assert_eq!(
            o,
            [
                (None, 1),
                (None, 2),
                (Some("h.h"), 1),
                (Some("h.h"), 2),
                (None, 3)
            ]
        );
        assert_eq!(p.text.lines().count(), p.origins.len());
        // a header that lacks a final newline does not glue to the next line
        assert_eq!(lines(&p), ["m1", "h1", "h2", "m3"]);
    }

    #[test]
    fn a_directive_continued_with_a_backslash_is_evaluated_whole_and_the_line_count_is_kept() {
        // FlashInfer's shape: a condition over several physical lines (39 kernels declined on the `\` token)
        let src = "#define A\n#define B 3\n#if defined(A) && \\\n    (B >= 2) && \\\n    !defined(C)\nYES\n#else\nNO\n#endif\nEND\n";
        let p = pp(src, &[]).unwrap();
        assert_eq!(lines(&p), ["YES", "END"]);
        assert_eq!(
            p.text.lines().count(),
            src.lines().count(),
            "every physical line is kept"
        );
        assert_eq!(p.origins.len(), src.lines().count());
        // control: with `C` defined the same condition is false
        let p = pp(&format!("#define C\n{src}"), &[]).unwrap();
        assert_eq!(lines(&p), ["NO", "END"]);
        // a continued `#define` takes its whole body, and the lines after it are not swallowed
        let p = pp("#define N 1 + \\\n  2\nX\n", &[]).unwrap();
        assert_eq!(lines(&p), ["X"]);
        // a trailing backslash on the last line does not read past the end
        let p = pp("#define Q \\", &[]).unwrap();
        assert!(lines(&p).is_empty());
    }

    #[test]
    fn a_source_with_no_supplied_include_is_unchanged() {
        let src = "int a;\n#include \"x.h\"\nint b;";
        let p = pp(src, &[]).unwrap();
        assert_eq!(p.text.len(), src.len(), "offsets are preserved");
        assert_eq!(p.text.lines().count(), 3);
    }
}
