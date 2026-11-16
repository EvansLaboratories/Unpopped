// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of `Stmt::MatrixLoad` agrees with the launch interpreter** (ROADMAP U8b,
//! `docs/options-u8-copy-atoms.md` S2): for 1, 2 and 4 tiles, plain and transposed, the compiled C and `run_launch` write the
//! same bytes, and both equal the lane distribution written out from the PTX description.

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
#[path = "support/matrix_load.rs"]
mod matrix_load;
use launch_run::agree_launch;
use matrix_load::{U16, matrix_expected, matrix_input, matrix_load_kernel};

fn buf(v: &[u16]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        U16,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

#[test]
fn the_c_matrix_load_agrees_with_the_oracle() {
    for count in [1, 2, 4] {
        for trans in [false, true] {
            let k = matrix_load_kernel(count, trans, true);
            assert!(CpuC.lower_kernel(&k).is_ok(), "count {count}: it lowers");
            let args = vec![buf(&matrix_input(count)), buf(&[0; 32 * 8])];
            agree_launch(&format!("matrix load x{count} trans={trans}"), &k, args);
        }
    }
}

#[test]
fn the_c_output_is_the_ptx_distribution() {
    // the oracle test pins the interpreter to the table; this pins the compiled C to it directly
    for count in [1, 2, 4] {
        for trans in [false, true] {
            let k = matrix_load_kernel(count, trans, true);
            let args = vec![buf(&matrix_input(count)), buf(&[0; 32 * 8])];
            let Some(c) =
                kernel_run::run_c(&format!("matrix table x{count} trans={trans}"), &k, &args)
            else {
                return;
            };
            let want: Vec<u8> = matrix_expected(count, trans)
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            assert_eq!(c.status, 0);
            assert_eq!(c.outs[0], want, "count {count}, trans {trans}");
        }
    }
}
