// SPDX-License-Identifier: MIT OR Apache-2.0
//! A `Stmt::MatrixLoad` as serial C (ROADMAP U8b, `docs/options-u8-copy-atoms.md` S2): a **block-level** construct between
//! two thread loops, like an MMA. A thread loop first stores, for each lane that supplies a row (lane `8t + r` of each
//! 32-lane warp, `t < count`), the address of its 8 contiguous `Shared` elements; the construct then walks the warps and
//! gives each lane, per tile, row `l / 4` columns `2 * (l % 4)` and `+ 1` (transposed: rows `2 * (l % 4)` and `+ 1`,
//! column `l / 4`), exactly the oracle's distribution (`oracle/kernel_matrix.rs`). Lanes that supply no row never
//! evaluate their address. Bits are moved, so any 16-bit element type is a fragment.

use unpopped::backend::LowerError;
use unpopped::kernel::Stmt;

use crate::base::{Emitter, ctype, refuse};

impl Emitter<'_> {
    /// Spell a `Stmt::MatrixLoad` at block level (the caller is between two thread loops).
    pub(crate) fn matrix_load(&mut self, st: &Stmt) -> Result<(), LowerError> {
        let Stmt::MatrixLoad {
            count,
            trans,
            src,
            dst,
        } = st
        else {
            unreachable!("matrix_load is called for a MatrixLoad");
        };
        let info = self.launch.expect("a launched emitter");
        let nt = info.threads();
        if nt % 32 != 0 {
            return Err(refuse("a matrix load in a block that is not whole warps"));
        }
        let elem = self.k.allocs[*dst as usize].elem;
        let t = ctype(elem, "a matrix fragment")?;
        let n = self.fresh;
        self.fresh += 1;
        let supplied = 8 * count;
        let width = 2 * count;
        self.line("{");
        self.depth += 1;
        self.line(&format!("const {t} *mr{n}[{nt}];"));
        // the address (and its bounds check) is evaluated only by the lanes that supply a row, inside the thread loop
        self.line(&format!("for (int t = 0; t < {nt}; ++t) {{"));
        self.depth += 1;
        self.line(&format!("mr{n}[t] = 0;"));
        self.line(&format!("if ((t % 32) < {supplied}) {{"));
        self.depth += 1;
        let addr = self.address(src)?;
        self.line(&format!("mr{n}[t] = &({addr});"));
        self.depth -= 1;
        self.line("}");
        self.depth -= 1;
        self.line("}");
        let (row, col) = if *trans {
            ("2 * (l % 4) + e", "l / 4")
        } else {
            ("l / 4", "2 * (l % 4) + e")
        };
        self.line(&format!(
            "for (int g = 0; g < {nt}; g += 32) for (int l = 0; l < 32; ++l) for (int tt = 0; tt < {count}; ++tt) for (int e = 0; e < 2; ++e) \
             sh{dst}[(long long)(g + l) * {width} + 2 * tt + e] = mr{n}[g + 8 * tt + ({row})][{col}];"
        ));
        self.depth -= 1;
        self.line("}");
        Ok(())
    }
}
