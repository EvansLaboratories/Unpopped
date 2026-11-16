// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of `Stmt::MatrixStore` agrees with the launch interpreter** (ROADMAP U8b,
//! `docs/options-u8-copy-atoms.md` S2): for 1, 2 and 4 tiles, plain and transposed, the compiled C and `run_launch` write the
//! same bytes, and both equal the tile contents written out from the PTX description.

use unpopped::backend::Backend;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;

#[path = "support/cc.rs"]
mod cc;
#[expect(
    dead_code,
    reason = "the shared driver; this file uses the part that runs a launched kernel"
)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[path = "support/launch_run.rs"]
mod launch_run;
#[path = "support/matrix_store.rs"]
mod matrix_store;
use launch_run::agree_launch;
use matrix_store::{H16, fragment_input, matrix_store_kernel, tiles_expected};

fn buf(v: &[u16]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        H16,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

#[test]
fn the_c_matrix_store_agrees_with_the_oracle() {
    for count in [1, 2, 4] {
        for trans in [false, true] {
            let k = matrix_store_kernel(count, trans, true);
            assert!(CpuC.lower_kernel(&k).is_ok(), "count {count}: it lowers");
            let args = vec![buf(&fragment_input()), buf(&vec![0; 64 * count as usize])];
            agree_launch(&format!("matrix store x{count} trans={trans}"), &k, args);
        }
    }
}

#[test]
fn the_c_output_is_the_ptx_tiles() {
    // the oracle test pins the interpreter to the table; this pins the compiled C to it directly
    for count in [1, 2, 4] {
        for trans in [false, true] {
            let k = matrix_store_kernel(count, trans, true);
            let args = vec![buf(&fragment_input()), buf(&vec![0; 64 * count as usize])];
            let Some(c) =
                kernel_run::run_c(&format!("matrix table x{count} trans={trans}"), &k, &args)
            else {
                return;
            };
            let want: Vec<u8> = tiles_expected(count, trans)
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            assert_eq!(c.status, 0);
            assert_eq!(c.outs[0], want, "count {count}, trans {trans}");
        }
    }
}
