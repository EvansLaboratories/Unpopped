// SPDX-License-Identifier: MIT OR Apache-2.0
//! A `Stmt::Collective` as serial C (ROADMAP U5 P5; `docs/options-u5-collectives-and-richer-reductions.md`): a
//! **block-level** construct between two thread loops, like a barrier. The thread loop before it ends with every
//! thread's contribution stored (`cv<n>[t]` the value, `co<n>[t]` the shuffle or broadcast operand); the construct then
//! walks the groups of the scope (the whole block, or consecutive runs of the launch's subgroup size) and writes each
//! lane's result into its local array `l<result>[lane]`; the next thread loop starts from there. The rules are the
//! oracle's (`oracle/kernel_collective.rs`), written out in C so the two can be compared byte for byte.
//!
//! The subgroup size is the launch's required minimum (the oracle's default when the caller does not choose one),
//! or `1` when the launch states none. A ballot over more than 32 lanes traps in the oracle and is not claimed here.

use unpopped::backend::LowerError;
use unpopped::exec::Scope;
use unpopped::ir::{ReduceOp, ScalarExpr};
use unpopped::kernel::{CollectiveKind, ShuffleMode, Stmt, VoteKind};
use unpopped::types::{self, Ty};
use unpopped_vocab::ElementKind;

use crate::base::{Emitter, ctype, refuse};

/// The C function a combining collective folds with, and the extra id arguments it takes.
pub(crate) struct Combine {
    pub(crate) name: String,
    pub(crate) ids: Vec<String>,
}

impl Combine {
    /// The C statements that set `acc = f(acc, rhs)` (the callee returns a status, then its result through a pointer).
    fn step(&self, s: &Spec, acc: &str, rhs: &str) -> Vec<String> {
        let n = s.n;
        let mut actuals = vec![format!("({})({acc})", s.val), format!("({})({rhs})", s.val)];
        actuals.extend(self.ids.iter().cloned());
        actuals.push(format!("&cr{n}"));
        vec![
            format!("{} cr{n} = 0;", s.val),
            format!("int cs{n} = {}({});", self.name, actuals.join(", ")),
            format!("if (cs{n}) return cs{n};"),
            format!("{acc} = cr{n};"),
        ]
    }
}

/// What the C of one collective is spelled from.
struct Spec {
    /// The id making the names unique: `cv<n>`, `co<n>`.
    n: u32,
    /// The C type of the contributed value and of the result local.
    val: &'static str,
    res: &'static str,
    ty: ElementKind,
    /// The result local.
    local: u32,
}

impl Emitter<'_> {
    /// Spell a collective at block level (the caller is between two thread loops).
    pub(crate) fn collective(&mut self, st: &Stmt) -> Result<(), LowerError> {
        let Stmt::Collective {
            kind,
            value,
            result,
            scope,
        } = st
        else {
            unreachable!("collective is called for a Collective");
        };
        let info = self.launch.expect("a launched emitter");
        let tys: Vec<Ty> = self.k.locals.iter().map(|d| d.ty()).collect();
        let Ok(Ty::Scalar(ty)) = types::infer_ty_in(value, self.k.compute, &tys) else {
            return Err(refuse("a collective value this emitter cannot type"));
        };
        let n = self.fresh;
        self.fresh += 1;
        let spec = Spec {
            n,
            val: ctype(ty, "a collective value")?,
            res: self.local_ctype(*result)?,
            ty,
            local: *result,
        };
        let group = if matches!(scope, Scope::Block) {
            info.threads()
        } else {
            info.subgroup.clamp(1, info.threads())
        };
        let nt = info.threads();
        self.contribute(&spec, kind, value, nt)?;
        self.line(&format!(
            "for (int g{n} = 0; g{n} < {nt}; g{n} += {group}) {{"
        ));
        self.depth += 1;
        self.line(&format!(
            "const int ge{n} = g{n} + {group} < {nt} ? g{n} + {group} : {nt}; (void)ge{n};"
        ));
        let comb = match kind {
            CollectiveKind::ReduceWith { combine } | CollectiveKind::ScanWith { combine } => {
                Some(self.combine_call(*combine)?)
            }
            _ => None,
        };
        let body = group_body(&spec, kind, comb.as_ref());
        for l in body? {
            self.line(&l);
        }
        self.depth -= 1;
        self.line("}");
        self.depth -= 1;
        self.line("}");
        Ok(())
    }

    /// The C function of the combine function `callee` and the id actuals it takes: a combine function runs on the
    /// group, not on a thread, so a launched module's id parameters are passed as `0` (a combine that reads an id is
    /// not a combine; the oracle traps on it).
    pub(crate) fn combine_call(&self, callee: u32) -> Result<Combine, LowerError> {
        let Some(m) = self.module else {
            return Err(refuse(
                "a combining collective outside a module (use lower_module)",
            ));
        };
        let ids = if m.launch.is_some() { 6 } else { 0 };
        Ok(Combine {
            name: m.names[callee as usize].clone(),
            ids: vec!["0".to_string(); ids],
        })
    }

    /// Open the construct's block and run the thread loop that stores every thread's contribution.
    fn contribute(
        &mut self,
        s: &Spec,
        kind: &CollectiveKind,
        value: &ScalarExpr,
        nt: u32,
    ) -> Result<(), LowerError> {
        let n = s.n;
        let operand = match kind {
            CollectiveKind::Shuffle { operand, .. } => self.expr(operand)?,
            CollectiveKind::Broadcast { lane } => self.expr(lane)?,
            _ => "0".to_string(),
        };
        let v = self.expr(value)?;
        self.line("{");
        self.depth += 1;
        self.line(&format!("{} cv{n}[{nt}]; long long co{n}[{nt}];", s.val));
        self.line(&format!(
            "for (int t = 0; t < {nt}; ++t) {{ cv{n}[t] = ({})({v}); co{n}[t] = (long long)({operand}); }}",
            s.val
        ));
        Ok(())
    }
}

/// The lines of the per-group code, given `g<n>` (first lane) and `ge<n>` (one past the last).
fn group_body(
    s: &Spec,
    kind: &CollectiveKind,
    comb: Option<&Combine>,
) -> Result<Vec<String>, LowerError> {
    match kind {
        CollectiveKind::ReduceWith { .. } => Ok(reduce_with(s, comb.expect("resolved"))),
        CollectiveKind::ScanWith { .. } => Ok(scan_with(s, comb.expect("resolved"))),
        CollectiveKind::Reduce(op) => reduce(s, *op),
        CollectiveKind::Scan { op, exclusive } => scan(s, *op, *exclusive),
        CollectiveKind::Shuffle { mode, .. } => Ok(shuffle(s, Some(*mode))),
        CollectiveKind::Broadcast { .. } => Ok(shuffle(s, None)),
        CollectiveKind::Vote(v) => Ok(vote(s, *v)),
        _ => Err(refuse("a collective kind this emitter has not been taught")),
    }
}

/// The left fold of the group's values under the combine function, every lane receiving it.
fn reduce_with(s: &Spec, c: &Combine) -> Vec<String> {
    let (n, local) = (s.n, s.local);
    let mut out = vec![
        format!("{} acc = cv{n}[g{n}];", s.val),
        format!("for (int i = g{n} + 1; i < ge{n}; ++i) {{"),
    ];
    out.extend(
        c.step(s, "acc", &format!("cv{n}[i]"))
            .into_iter()
            .map(|l| format!("    {l}")),
    );
    out.push("}".to_string());
    out.push(format!(
        "for (int i = g{n}; i < ge{n}; ++i) l{local}[i] = ({})acc;",
        s.res
    ));
    out
}

/// The inclusive prefix fold under the combine function.
fn scan_with(s: &Spec, c: &Combine) -> Vec<String> {
    let (n, local) = (s.n, s.local);
    let mut out = vec![
        format!("{} acc = cv{n}[g{n}];", s.val),
        format!("for (int i = g{n}; i < ge{n}; ++i) {{"),
        format!("    if (i > g{n}) {{"),
    ];
    out.extend(
        c.step(s, "acc", &format!("cv{n}[i]"))
            .into_iter()
            .map(|l| format!("        {l}")),
    );
    out.push("    }".to_string());
    out.push(format!("    l{local}[i] = ({})acc;", s.res));
    out.push("}".to_string());
    out
}

fn reduce(s: &Spec, op: ReduceOp) -> Result<Vec<String>, LowerError> {
    let (n, local) = (s.n, s.local);
    let step = combine(s, op, "acc", &format!("cv{n}[i]"))?;
    Ok(vec![
        format!("{} acc = cv{n}[g{n}];", s.val),
        format!("for (int i = g{n} + 1; i < ge{n}; ++i) acc = {step};"),
        format!(
            "for (int i = g{n}; i < ge{n}; ++i) l{local}[i] = ({})acc;",
            s.res
        ),
    ])
}

fn scan(s: &Spec, op: ReduceOp, exclusive: bool) -> Result<Vec<String>, LowerError> {
    let (n, local) = (s.n, s.local);
    let step = combine(s, op, "acc", &format!("cv{n}[i]"))?;
    let mut out = vec![
        format!("{} acc = cv{n}[g{n}];", s.val),
        format!("for (int i = g{n}; i < ge{n}; ++i) {{"),
    ];
    if exclusive {
        let id = identity(s, op)?;
        out.push(format!(
            "    l{local}[i] = ({})(i == g{n} ? {id} : acc);",
            s.res
        ));
        out.push(format!("    if (i > g{n}) acc = {step};"));
    } else {
        out.push(format!("    if (i > g{n}) acc = {step};"));
        out.push(format!("    l{local}[i] = ({})acc;", s.res));
    }
    out.push("}".to_string());
    Ok(out)
}

/// A shuffle (`Some(mode)`) or a broadcast (`None`, which is `Idx`): lane `lane` of the group reads lane `src`.
fn shuffle(s: &Spec, mode: Option<ShuffleMode>) -> Vec<String> {
    let (n, local) = (s.n, s.local);
    let size = format!("(ge{n} - g{n})");
    let src = match mode {
        Some(ShuffleMode::Up) => {
            format!("(lane - co{n}[i] >= 0 && lane - co{n}[i] < {size} ? lane - co{n}[i] : lane)")
        }
        Some(ShuffleMode::Down) => {
            format!("(lane + co{n}[i] >= 0 && lane + co{n}[i] < {size} ? lane + co{n}[i] : lane)")
        }
        Some(ShuffleMode::Xor) => format!(
            "((lane ^ co{n}[i]) >= 0 && (lane ^ co{n}[i]) < {size} ? (lane ^ co{n}[i]) : lane)"
        ),
        _ => format!("((co{n}[i] % {size} + {size}) % {size})"),
    };
    vec![
        format!("for (int i = g{n}; i < ge{n}; ++i) {{"),
        format!("    const long long lane = i - g{n};"),
        format!("    l{local}[i] = ({})cv{n}[g{n} + {src}];", s.res),
        "}".to_string(),
    ]
}

fn vote(s: &Spec, kind: VoteKind) -> Vec<String> {
    let (n, local) = (s.n, s.local);
    let result = match kind {
        VoteKind::All => "all",
        VoteKind::Any => "any",
        _ => "mask",
    };
    vec![
        "int all = 1, any = 0; unsigned int mask = 0;".to_string(),
        format!("for (int i = g{n}; i < ge{n}; ++i) {{"),
        format!("    const int f = cv{n}[i] != 0;"),
        format!("    all &= f; any |= f; mask |= (unsigned int)f << (i - g{n});"),
        "}".to_string(),
        format!(
            "for (int i = g{n}; i < ge{n}; ++i) l{local}[i] = ({})({result});",
            s.res
        ),
    ]
}

/// The unsigned C type of the same width as the signed or unsigned integer `ty`.
fn unsigned_of(ty: ElementKind) -> Result<&'static str, LowerError> {
    match ty {
        ElementKind::I32 | ElementKind::U32 => Ok("unsigned int"),
        ElementKind::I64 | ElementKind::U64 => Ok("unsigned long long"),
        _ => Err(refuse("a collective integer outside I32, I64, U32, U64")),
    }
}

/// `a op b` at the value type (an integer wraps through the unsigned type; a float rounds in its own width; a float
/// `Max`/`Min` keeps a NaN and otherwise lets a strictly better value replace, as the oracle does).
fn combine(s: &Spec, op: ReduceOp, a: &str, b: &str) -> Result<String, LowerError> {
    let t = s.val;
    let float = matches!(s.ty, ElementKind::F32 | ElementKind::F64);
    if float {
        return match op {
            ReduceOp::Sum => Ok(format!("({t})({a} + {b})")),
            ReduceOp::Max => Ok(format!(
                "({a} != {a} ? {a} : ({b} != {b} || {b} > {a} ? {b} : {a}))"
            )),
            ReduceOp::Min => Ok(format!(
                "({a} != {a} ? {a} : ({b} != {b} || {b} < {a} ? {b} : {a}))"
            )),
            _ => Err(refuse("a collective float fold outside Sum, Max, Min")),
        };
    }
    let u = unsigned_of(s.ty)?;
    match op {
        ReduceOp::Sum => Ok(format!("({t})(({u}){a} + ({u}){b})")),
        ReduceOp::Max => Ok(format!("({a} > {b} ? {a} : {b})")),
        ReduceOp::Min => Ok(format!("({a} < {b} ? {a} : {b})")),
        ReduceOp::And => Ok(format!("({t})({a} & {b})")),
        ReduceOp::Or => Ok(format!("({t})({a} | {b})")),
        ReduceOp::Xor => Ok(format!("({t})({a} ^ {b})")),
        _ => Err(refuse("a collective integer fold outside the v1 set")),
    }
}

/// The value lane 0 of an exclusive scan receives.
fn identity(s: &Spec, op: ReduceOp) -> Result<String, LowerError> {
    let t = s.val;
    match op {
        ReduceOp::Sum | ReduceOp::Or | ReduceOp::Xor => Ok(format!("({t})0")),
        ReduceOp::And => Ok(format!("({t})(~({})0)", unsigned_of(s.ty)?)),
        _ => Err(refuse("an exclusive scan with no identity")),
    }
}
