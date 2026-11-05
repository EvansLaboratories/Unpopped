// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of the double-buffered `cp.async` pipeline agrees with the launch interpreter** (ROADMAP U8 S4,
//! `docs/options-u8-copy-atoms.md`). cpu-c copies at issue and spells commit and wait as nothing; the correct pipeline
//! (copy, commit, wait, barrier, read, barrier, over two stages) must write the same bytes as `run_launch` and the sum
//! the oracle test pins. The C is compiled with a host compiler and run.

use unpopped::backend::Backend;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_cpu_c::CpuC;

#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_data.rs"]
mod kernel_data;
#[expect(
    dead_code,
    reason = "the shared driver; this file uses the part that runs a launched kernel"
)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[path = "support/launch_run.rs"]
mod launch_run;
#[path = "support/pipeline.rs"]
mod pipeline;
use kernel_data::zeroed;
use launch_run::agree_launch;
use pipeline::{EXPECTED, Fault, I32, double_buffered};

fn args() -> Vec<KernelArg> {
    let src: Vec<u8> = (0..16).flat_map(|i: i32| (100 + i).to_le_bytes()).collect();
    vec![
        KernelArg::Buffer(TypedBuffer::new(I32, vec![16], vec![1], src)),
        KernelArg::Buffer(zeroed(I32, &[4])),
    ]
}

#[test]
fn the_double_buffered_pipeline_agrees_between_the_c_and_the_interpreter() {
    let k = double_buffered(Fault::None);
    agree_launch("pipeline", &k, args());
    // and the number is the one the oracle test pins, not merely one both sides share
    let mut run = args();
    run_launch(&k, &mut run).expect("the oracle runs it");
    let KernelArg::Buffer(out) = &run[1] else {
        unreachable!("a buffer")
    };
    let sums: Vec<i32> = out
        .bytes
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(sums, EXPECTED);
}

#[test]
fn the_c_has_the_four_copies_and_no_async_machinery() {
    let g = CpuC
        .lower_kernel(&double_buffered(Fault::None))
        .expect("the pipeline lowers");
    // tile 0 plus tiles 1..3: four copy loops, two barriers per iteration
    assert_eq!(g.source.matches("cp_ < 1").count(), 4, "{}", g.source);
    assert!(!g.source.contains("async"));
}
