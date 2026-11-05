// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The emitted C agrees with the kernel interpreter on an arg-reduction written as a loop** (ROADMAP U5 P3; PM
//! ruling 2026-11-02 (3)): the kernel layer has no lift of `ArgMax`/`ArgMin`, so the sequential form is the loop of
//! `unpopped/tests/support/arg_loop.rs` (KISS-OPS-6.11-0007's order: NaN greatest, the lower
//! index wins a tie). It is lowered by
//! `CpuC::lower_kernel`, compiled with a host C compiler, run, and its `I64` index bytes compared with `run_kernel`;
//! the interpreter side is held to the oracle in `unpopped/tests/reduce_argmax.rs`, so the three agree.
//!
//! The corpus has ties (the first index must win), a NaN first, a NaN in the middle with a larger number after it,
//! an all-negative row, equal signed zeros and an all-NaN row.

use unpopped::backend::Backend;
use unpopped::kernel::check_kernel;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/arg_loop.rs"]
mod arg_loop;
#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_data.rs"]
mod kernel_data;
#[path = "support/kernel_run.rs"]
mod kernel_run;
use arg_loop::loop_kernel;
use kernel_data::zeroed;
use kernel_run::{agree, run_c_mutated};

const ROWS: i64 = 6;
const COLS: i64 = 6;

fn args() -> Vec<KernelArg> {
    let n = f32::NAN;
    let vals: [f32; 36] = [
        3.0, 9.0, 9.0, 1.0, 1.0, 2.0, //
        n, 1.0, 2.0, 3.0, -4.0, 5.0, //
        1.0, 4.0, n, 7.0, n, 2.0, //
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, //
        0.0, -0.0, 0.0, -0.0, 0.0, -0.0, //
        n, n, n, n, n, n, //
    ];
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    vec![
        KernelArg::Buffer(TypedBuffer::new(
            ElementKind::F32,
            vec![ROWS, COLS],
            vec![COLS, 1],
            bytes,
        )),
        KernelArg::Buffer(zeroed(ElementKind::I64, &[ROWS])),
        KernelArg::Int(i128::from(ROWS)),
        KernelArg::Int(i128::from(COLS)),
    ]
}

#[test]
fn the_arg_loop_agrees_between_the_c_and_the_interpreter() {
    for (name, max) in [("argmax", true), ("argmin", false)] {
        let k = loop_kernel(max);
        assert_eq!(check_kernel(&k), Ok(()));
        agree(&format!("arg_{name}"), &k, args());
    }
}

#[test]
fn the_c_comparison_is_what_the_differential_is_sensitive_to() {
    // a seeded mutant of the generated C: the `>` of the argmax turned into `<` must change an answer
    let k = loop_kernel(true);
    let g = CpuC.lower_kernel(&k).expect("the loop lowers");
    assert!(g.source.contains(" > "), "the argmax comparison is spelled");
    let honest = run_c_mutated("arg_honest", &k, &args(), |s| s).expect("a C compiler");
    let flipped = run_c_mutated("arg_flipped", &k, &args(), |s| s.replace(" > ", " < "));
    if let Some(m) = flipped {
        assert_ne!(
            m.outs, honest.outs,
            "the answer must not survive the comparison being flipped"
        );
    }
}
