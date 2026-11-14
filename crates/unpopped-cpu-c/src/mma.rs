// SPDX-License-Identifier: MIT OR Apache-2.0
//! A `Stmt::Mma` as serial C (ROADMAP U7 S2; `docs/options-u7-mma-atoms.md`): a **block-level** construct between two
//! thread loops, like a collective. The thread loop before it has stored every lane's A, B and C fragments in the
//! `Private` arrays (`sh<alloc>[t * n + v]`); the construct then walks the consecutive subgroups of the atom's lane
//! count, gathers each one's fragments into the logical column-major matrices through id tables **baked from the atom**
//! (`MmaAtom::id`, so the C and the oracle read the same layout data), forms `D = A.B + C`, and scatters each lane's D
//! fragment back. The arithmetic is the oracle's (`oracle/kernel_mma.rs`): floats are summed from `C` in ascending `k`
//! order in `double` and rounded once when stored to the accumulator element; integers are exact in `long long` and
//! wrap when stored. An atom with **extra operands** is spelled too: a 2:4 sparse `A` is expanded into a dense `mx`
//! through the metadata (the other two positions of a group are zero), and block scales multiply the factors of
//! each `k` block; the extras' id tables come from the atom like the rest, and lanes that hold one element of a
//! broadcast scale must agree (the oracle traps on a disagreement; the C takes the last, so that is a precondition
//! of the program, as is a metadata nibble that is not `i0 < i1`). Only element types this emitter spells (`float`, `double`, the integers) can be fragments, so
//! the f16 and bf16 atoms are refused with a typed error naming the dtype; an atom with `F32` operands (atoms are
//! data) is spelled.

use unpopped::backend::LowerError;
use unpopped::kernel::Stmt;
use unpopped::mma::{ExtraRole, MmaAtom};
use unpopped_vocab::ElementKind;

use crate::base::{Emitter, ctype, refuse};

/// `static const int <name>[] = {...};` of the logical id of every (lane, value) slot of operand `which`.
fn id_table(atom: &MmaAtom, which: usize, name: &str) -> Result<String, LowerError> {
    let mut ids = Vec::new();
    for lane in 0..atom.lanes() {
        for v in 0..atom.values(which) {
            let id = atom
                .id(which, lane, v)
                .ok_or_else(|| refuse("an MMA fragment slot outside the atom"))?;
            ids.push(id.to_string());
        }
    }
    Ok(format!(
        "static const int {name}[] = {{{}}};",
        ids.join(", ")
    ))
}

/// The C type of an MMA fragment element: a narrow float is its bit pattern. The product and the sum run on `double`, so
/// the gather decodes a narrow operand (ROADMAP U26). A narrow accumulator is refused: no atom has one, so nothing could
/// check its encode against the oracle.
fn fragment_ctype(t: ElementKind, what: &str) -> Result<&'static str, LowerError> {
    ctype(t, what)
}

/// `value` read from a fragment element of type `t` as the `wide` value the product works on.
fn read_element(t: ElementKind, wide: &str, value: &str) -> String {
    match crate::narrow::load_fn(t) {
        Some(dec) => format!("({wide}){dec}({value})"),
        None => format!("({wide}){value}"),
    }
}

fn is_float(t: ElementKind) -> bool {
    matches!(
        t,
        ElementKind::F32 | ElementKind::F32Strict | ElementKind::F64
    )
}

impl Emitter<'_> {
    /// Spell a `Stmt::Mma` at block level (the caller is between two thread loops).
    pub(crate) fn mma(&mut self, st: &Stmt) -> Result<(), LowerError> {
        let Stmt::Mma {
            atom,
            a,
            b,
            c,
            d,
            extra,
        } = st
        else {
            unreachable!("mma is called for an Mma");
        };
        let roles: Vec<(ExtraRole, ElementKind)> = atom.extras().collect();
        if extra.len() != roles.len() {
            return Err(refuse("an MMA whose extra operands are not the atom's"));
        }
        let info = self.launch.expect("a launched emitter");
        let nt = info.threads();
        let lanes = atom.lanes();
        if nt % lanes != 0 {
            return Err(refuse(
                "a block that is not a whole number of the atom's subgroups",
            ));
        }
        let (ta, tb, tacc) = atom.dtypes();
        for (t, what) in [(ta, "an MMA A fragment"), (tb, "an MMA B fragment")] {
            fragment_ctype(t, what)?;
        }
        for (_, t) in &roles {
            fragment_ctype(*t, "an MMA extra fragment")?;
        }
        if crate::narrow::is_narrow(tacc) {
            return Err(refuse(format!(
                "an MMA accumulator of dtype {tacc:?}: no atom has one, so its encode is unchecked (ROADMAP U26)"
            )));
        }
        let td = fragment_ctype(tacc, "an MMA accumulator fragment")?;
        let wide = if is_float(tacc) {
            "double"
        } else {
            "long long"
        };
        let id = self.fresh;
        self.fresh += 1;
        self.line("{");
        self.depth += 1;
        for w in 0..3 + roles.len() {
            let table = id_table(atom, w, &format!("mi{id}o{w}"))?;
            self.line(&table);
        }
        self.line(&format!(
            "for (int g = 0; g < {nt}; g += {lanes}) {{ /* {} */",
            atom.name()
        ));
        self.depth += 1;
        let mut allocs = vec![*a, *b, *c];
        allocs.extend(extra.iter().copied());
        let mut dtypes = vec![ta, tb, tacc];
        dtypes.extend(roles.iter().map(|(_, t)| *t));
        self.mma_gather(atom, &allocs, &dtypes, id, wide);
        let mut a_name = "ma";
        if let Some(w) = atom.extra_which(ExtraRole::Metadata) {
            self.mma_expand_sparse(atom, w, wide);
            a_name = "mx";
        }
        let block = match (
            atom.block(),
            atom.extra_which(ExtraRole::ScaleA),
            atom.extra_which(ExtraRole::ScaleB),
        ) {
            (Some(bk), Some(wa), Some(wb)) => Some((bk as usize, wa, wb)),
            _ => None,
        };
        self.mma_product(atom, wide, a_name, block);
        let vd = atom.values(2);
        let stored = format!("({td})md[mi{id}o2[l * {vd} + v]]");
        self.line(&format!(
            "for (int l = 0; l < {lanes}; ++l) for (int v = 0; v < {vd}; ++v) sh{d}[(long long)(g + l) * {vd} + v] = {stored};"
        ));
        self.depth -= 1;
        self.line("}");
        self.depth -= 1;
        self.line("}");
        Ok(())
    }

    /// Declare the logical matrices of one subgroup and fill every operand from the lanes' fragments.
    fn mma_gather(
        &mut self,
        atom: &MmaAtom,
        allocs: &[u32],
        dtypes: &[ElementKind],
        id: u32,
        wide: &str,
    ) {
        let (m, n, _) = atom.mnk();
        let lanes = atom.lanes();
        self.line(&format!("{wide} md[{}];", m * n));
        for (w, alloc) in allocs.iter().enumerate() {
            let (rows, cols) = atom.dims(w);
            let vals = atom.values(w);
            // the metadata is integer whatever the accumulator is
            let ty = if w >= 3 && atom.extra_which(ExtraRole::Metadata) == Some(w) {
                "long long"
            } else {
                wide
            };
            let nm = ["ma", "mb", "mc"]
                .get(w)
                .map_or_else(|| format!("me{w}"), |s| (*s).to_string());
            self.line(&format!("{ty} {nm}[{}];", rows * cols));
            let read = read_element(
                dtypes[w],
                ty,
                &format!("sh{alloc}[(long long)(g + l) * {vals} + v]"),
            );
            self.line(&format!(
                "for (int l = 0; l < {lanes}; ++l) for (int v = 0; v < {vals}; ++v) {nm}[mi{id}o{w}[l * {vals} + v]] = {read};"
            ));
        }
    }

    /// `mx` = the dense `m x k` `A`: group `g` of row `r` keeps its two compressed values where the metadata says.
    fn mma_expand_sparse(&mut self, atom: &MmaAtom, meta: usize, wide: &str) {
        let (m, _, k) = atom.mnk();
        self.line(&format!("{wide} mx[{}];", m * k));
        self.line(&format!("for (int i = 0; i < {}; ++i) mx[i] = 0;", m * k));
        self.line(&format!(
            "for (int q = 0; q < {}; ++q) for (int r = 0; r < {m}; ++r) {{",
            k / 4
        ));
        self.depth += 1;
        self.line(&format!("int nib = (int)(me{meta}[r + {m} * q] & 15);"));
        self.line(&format!(
            "mx[r + {m} * (4 * q + (nib & 3))] = ma[r + {m} * (2 * q)];"
        ));
        self.line(&format!(
            "mx[r + {m} * (4 * q + (nib >> 2))] = ma[r + {m} * (2 * q + 1)];"
        ));
        self.depth -= 1;
        self.line("}");
    }

    /// `D = A.B + C` over the logical matrices: from `C`, ascending `k`; `(block, sfa, sfb)` scales each factor.
    fn mma_product(
        &mut self,
        atom: &MmaAtom,
        wide: &str,
        a: &str,
        scales: Option<(usize, usize, usize)>,
    ) {
        let (m, n, k) = atom.mnk();
        self.line(&format!(
            "for (int col = 0; col < {n}; ++col) for (int row = 0; row < {m}; ++row) {{"
        ));
        self.depth += 1;
        self.line(&format!("{wide} s = mc[row + {m} * col];"));
        let term = match scales {
            Some((bk, wa, wb)) => format!(
                "({a}[row + {m} * i] * me{wa}[row + {m} * (i / {bk})]) * (mb[i + {k} * col] * me{wb}[i / {bk} + {} * col])",
                k as usize / bk
            ),
            None => format!("{a}[row + {m} * i] * mb[i + {k} * col]"),
        };
        self.line(&format!("for (int i = 0; i < {k}; ++i) s += {term};"));
        self.line(&format!("md[row + {m} * col] = s;"));
        self.depth -= 1;
        self.line("}");
    }
}
