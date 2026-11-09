// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The C of the padding-mode window reads agrees with the oracle** (ROADMAP R5.7 C4b; the modes and their hand-worked rows
//! are `crates/unpopped/tests/kernel_padding_modes.rs`): zero, NaN, constant and `-inf` fill (a `Stmt::MaskedLoad`) and
//! replicate and reflect (an index remap), lowered by `unpopped_cpu_c::CpuC::lower_kernel`, compiled and run, and compared
//! byte for byte with `run_kernel` (NaN compared by bits: the pad is the same quiet NaN on both sides).

use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c as _;

#[path = "support/cc.rs"]
mod cc;
// The shared compile-and-run driver; this file uses a subset of it.
#[allow(dead_code)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[path = "support/padding_modes.rs"]
mod padding_modes;
use kernel_run::agree;
use padding_modes::{F32, IN_LEN, Mode, TAPS, window};

fn args() -> Vec<KernelArg> {
    let bytes: Vec<u8> = [10.0f32, 11.0, 12.0, 13.0, 14.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    vec![
        KernelArg::Buffer(TypedBuffer::new(F32, vec![IN_LEN], vec![1], bytes)),
        KernelArg::Buffer(TypedBuffer::new(
            F32,
            vec![TAPS],
            vec![1],
            vec![0; 4 * TAPS as usize],
        )),
    ]
}

#[test]
fn every_padding_mode_agrees_with_the_oracle() {
    for (tag, mode) in [
        ("pad_zero", Mode::Zero),
        ("pad_nan", Mode::Nan),
        ("pad_const", Mode::Constant(7.0)),
        ("pad_ninf", Mode::Constant(f64::NEG_INFINITY)),
        ("pad_replicate", Mode::Replicate),
        ("pad_reflect", Mode::Reflect),
    ] {
        agree(tag, &window(mode), args());
    }
}
