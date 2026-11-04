// SPDX-License-Identifier: MIT OR Apache-2.0
//! A `Stmt::Mma` as serial C (ROADMAP U7 S2; `docs/options-u7-mma-atoms.md`): a **block-level** construct between two
//! thread loops, like a collective. The thread loop before it has stored every lane's A, B and C fragments in the
//! `Private` arrays (`sh<alloc>[t * n + v]`); the construct then walks the consecutive subgroups of the atom's lane
//! count, gathers each one's fragments into the logical column-major matrices through id tables **baked from the atom**
//! (`MmaAtom::id`, so the C and the oracle read the same layout data), forms `D = A.B + C`, and scatters each lane's D
//! fragment back. The arithmetic is the oracle's (`oracle/kernel_mma.rs`): floats are summed from `C` in ascending `k`
//! order in `double` and rounded once when stored to the accumulator element; integers are exact in `long long` and
//! wrap when stored. Only element types this emitter spells (`float`, `double`, the integers) can be fragments, so
//! the f16 and bf16 atoms are refused with a typed error naming the dtype; an atom with `F32` operands (atoms are
//! data) is spelled.

use unpopped::backend::LowerError;
use unpopped::kernel::Stmt;
use unpopped::mma::MmaAtom;
use unpopped_vocab::ElementKind;

use crate::kernel::{Emitter, ctype, refuse};

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

fn is_float(t: ElementKind) -> bool {
    matches!(
        t,
        ElementKind::F32 | ElementKind::F32Strict | ElementKind::F64
    )
}

impl Emitter<'_> {
    /// Spell a `Stmt::Mma` at block level (the caller is between two thread loops).
    pub(crate) fn mma(&mut self, st: &Stmt) -> Result<(), LowerError> {
        let Stmt::Mma { atom, a, b, c, d } = st else {
            unreachable!("mma is called for an Mma");
        };
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
            ctype(t, what)?;
        }
        ctype(tacc, "an MMA accumulator fragment")?;
        let (m, n, k) = atom.mnk();
        let wide = if is_float(tacc) {
            "double"
        } else {
            "long long"
        };
        let id = self.fresh;
        self.fresh += 1;
        self.line("{");
        self.depth += 1;
        for (w, nm) in ["a", "b", "c"].into_iter().enumerate() {
            let table = id_table(atom, w, &format!("mi{id}{nm}"))?;
            self.line(&table);
        }
        self.line(&format!(
            "for (int g = 0; g < {nt}; g += {lanes}) {{ /* {} */",
            atom.name()
        ));
        self.depth += 1;
        self.line(&format!(
            "{wide} ma[{}], mb[{}], mc[{}], md[{}];",
            m * k,
            k * n,
            m * n,
            m * n
        ));
        for (w, (nm, alloc)) in [("a", *a), ("b", *b), ("c", *c)].into_iter().enumerate() {
            let vals = atom.values(w);
            self.line(&format!(
                "for (int l = 0; l < {lanes}; ++l) for (int v = 0; v < {vals}; ++v) m{nm}[mi{id}{nm}[l * {vals} + v]] = ({wide})sh{alloc}[(long long)(g + l) * {vals} + v];"
            ));
        }
        self.line(&format!(
            "for (int col = 0; col < {n}; ++col) for (int row = 0; row < {m}; ++row) {{"
        ));
        self.depth += 1;
        self.line(&format!("{wide} s = mc[row + {m} * col];"));
        self.line(&format!(
            "for (int i = 0; i < {k}; ++i) s += ma[row + {m} * i] * mb[i + {k} * col];"
        ));
        self.line(&format!("md[row + {m} * col] = s;"));
        self.depth -= 1;
        self.line("}");
        let vd = atom.values(2);
        let td = ctype(tacc, "an MMA accumulator fragment")?;
        self.line(&format!(
            "for (int l = 0; l < {lanes}; ++l) for (int v = 0; v < {vd}; ++v) sh{d}[(long long)(g + l) * {vd} + v] = ({td})md[mi{id}c[l * {vd} + v]];"
        ));
        self.depth -= 1;
        self.line("}");
        self.depth -= 1;
        self.line("}");
        Ok(())
    }
}
