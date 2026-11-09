// SPDX-License-Identifier: MIT OR Apache-2.0
//! A `Stmt::CollectiveTuple` as serial C (ROADMAP R5.7 D4b; `docs/options-d4b-tuple-state-combine.md`): a block-level
//! construct between two thread loops, like [`crate::collective`]'s combining collectives. The thread loop before it
//! stores every thread's components (`cv<n>_<k>[t]`); the construct then walks the groups of the scope and folds each
//! group's states left to right through the module's combine function, which takes the two states component by
//! component and returns the new state through one out-pointer per component. A reduce writes the last state into every
//! lane's result locals; a scan writes the running state into each lane. The rules are the oracle's
//! (`oracle/kernel_collective.rs`, `tuple_results`).

use unpopped::backend::LowerError;
use unpopped::exec::Scope;
use unpopped::kernel::{CollectiveKind, Stmt};
use unpopped::types::{self, Ty};

use crate::kernel::{Emitter, ctype, refuse};

/// One component of the state: the C types of its value and its result local, the result local, and the C of its value.
struct Component {
    val: &'static str,
    res: &'static str,
    local: u32,
    expr: String,
}

impl Emitter<'_> {
    /// Spell a tuple collective at block level (the caller is between two thread loops).
    pub(crate) fn collective_tuple(&mut self, st: &Stmt) -> Result<(), LowerError> {
        let Stmt::CollectiveTuple {
            kind,
            values,
            results,
            scope,
        } = st
        else {
            unreachable!("collective_tuple is called for a CollectiveTuple");
        };
        let (combine, scan) = match kind {
            CollectiveKind::ReduceWith { combine } => (*combine, false),
            CollectiveKind::ScanWith { combine } => (*combine, true),
            _ => {
                return Err(refuse(
                    "a tuple collective other than ReduceWith or ScanWith",
                ));
            }
        };
        let info = self.launch.expect("a launched emitter");
        let comps = self.components(values, results)?;
        let n = self.fresh;
        self.fresh += 1;
        let nt = info.threads();
        let group = if matches!(scope, Scope::Block) {
            nt
        } else {
            info.subgroup.clamp(1, nt)
        };
        let comb = self.combine_call(combine)?;
        self.line("{");
        self.depth += 1;
        for l in contribute(n, nt, &comps) {
            self.line(&l);
        }
        self.line(&format!(
            "for (int g{n} = 0; g{n} < {nt}; g{n} += {group}) {{"
        ));
        self.depth += 1;
        self.line(&format!(
            "const int ge{n} = g{n} + {group} < {nt} ? g{n} + {group} : {nt}; (void)ge{n};"
        ));
        for l in group_body(n, &comps, (&comb.name, &comb.ids), scan) {
            self.line(&l);
        }
        self.depth -= 1;
        self.line("}");
        self.depth -= 1;
        self.line("}");
        Ok(())
    }

    /// The components of the state: each value typed and spelled, each result local's C type.
    fn components(
        &self,
        values: &[unpopped::ir::ScalarExpr],
        results: &[u32],
    ) -> Result<Vec<Component>, LowerError> {
        let tys: Vec<Ty> = self.k.locals.iter().map(|d| d.ty()).collect();
        let mut out = Vec::with_capacity(values.len());
        for (v, r) in values.iter().zip(results) {
            let Ok(Ty::Scalar(ty)) = types::infer_ty_in(v, self.k.compute, &tys) else {
                return Err(refuse("a collective value this emitter cannot type"));
            };
            out.push(Component {
                val: ctype(ty, "a collective value")?,
                res: self.local_ctype(*r)?,
                local: *r,
                expr: self.expr(v)?,
            });
        }
        Ok(out)
    }
}

/// The lines that declare the component arrays and run the thread loop storing every thread's contribution.
fn contribute(n: u32, nt: u32, comps: &[Component]) -> Vec<String> {
    let mut out: Vec<String> = comps
        .iter()
        .enumerate()
        .map(|(k, c)| format!("{} cv{n}_{k}[{nt}];", c.val))
        .collect();
    let stores: String = comps
        .iter()
        .enumerate()
        .map(|(k, c)| format!(" cv{n}_{k}[t] = ({})({});", c.val, c.expr))
        .collect();
    out.push(format!("for (int t = 0; t < {nt}; ++t) {{{stores} }}"));
    out
}

/// The per-group code, given `g<n>` (first lane) and `ge<n>` (one past the last): the running state `acc<n>_<k>` starts
/// as lane `g<n>`'s and takes each later lane through the combine function; a reduce stores the last state in every lane,
/// a scan stores the running state in each lane as it goes.
fn group_body(
    n: u32,
    comps: &[Component],
    (name, ids): (&str, &[String]),
    scan: bool,
) -> Vec<String> {
    let mut out: Vec<String> = comps
        .iter()
        .enumerate()
        .map(|(k, c)| format!("{} acc{n}_{k} = cv{n}_{k}[g{n}];", c.val))
        .collect();
    out.push(format!(
        "for (int i = g{n}{}; i < ge{n}; ++i) {{",
        if scan { "" } else { " + 1" }
    ));
    let pad = if scan { "        " } else { "    " };
    if scan {
        out.push(format!("    if (i > g{n}) {{"));
    }
    out.extend(
        step(n, comps, (name, ids))
            .into_iter()
            .map(|l| format!("{pad}{l}")),
    );
    if scan {
        out.push("    }".to_string());
        out.extend(
            comps
                .iter()
                .enumerate()
                .map(|(k, c)| format!("    l{}[i] = ({})acc{n}_{k};", c.local, c.res)),
        );
    }
    out.push("}".to_string());
    if !scan {
        let stores: String = comps
            .iter()
            .enumerate()
            .map(|(k, c)| format!(" l{}[i] = ({})acc{n}_{k};", c.local, c.res))
            .collect();
        out.push(format!("for (int i = g{n}; i < ge{n}; ++i) {{{stores} }}"));
    }
    out
}

/// The C statements that set `acc = f(acc, cv[i])` component by component: the callee returns a status, then its values
/// through one pointer each.
fn step(n: u32, comps: &[Component], (name, ids): (&str, &[String])) -> Vec<String> {
    let mut actuals: Vec<String> = comps
        .iter()
        .enumerate()
        .map(|(k, c)| format!("({})(acc{n}_{k})", c.val))
        .collect();
    actuals.extend(
        comps
            .iter()
            .enumerate()
            .map(|(k, c)| format!("({})(cv{n}_{k}[i])", c.val)),
    );
    actuals.extend(ids.iter().cloned());
    actuals.extend((0..comps.len()).map(|k| format!("&cr{n}_{k}")));
    let mut out: Vec<String> = comps
        .iter()
        .enumerate()
        .map(|(k, c)| format!("{} cr{n}_{k} = 0;", c.val))
        .collect();
    out.push(format!("int cs{n} = {name}({});", actuals.join(", ")));
    out.push(format!("if (cs{n}) return cs{n};"));
    out.extend((0..comps.len()).map(|k| format!("acc{n}_{k} = cr{n}_{k};")));
    out
}
