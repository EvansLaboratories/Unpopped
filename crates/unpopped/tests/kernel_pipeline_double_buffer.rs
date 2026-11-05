// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A double-buffered `cp.async` pipeline in the oracle** (ROADMAP U8 S4, `docs/options-u8-copy-atoms.md`): the
//! conformance kernel for "stages are an allocation with a stage axis plus async groups; no IR node". The correct
//! kernel sums as expected; each way of getting the pipeline wrong is a typed trap, not a wrong number.

use unpopped::kernel::check_kernel;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/pipeline.rs"]
mod pipeline;
use pipeline::{EXPECTED, Fault, I32, double_buffered};

fn buffer(values: &[i32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        I32,
        vec![values.len() as i64],
        vec![1],
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    ))
}

fn run(fault: Fault) -> Result<Vec<i32>, KernelTrap> {
    let k = double_buffered(fault);
    assert_eq!(check_kernel(&k), Ok(()), "{fault:?} is well formed");
    let src: Vec<i32> = (0..16).map(|i| 100 + i).collect();
    let mut args = [buffer(&src), buffer(&[0; 4])];
    run_launch(&k, &mut args)?;
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!("a buffer")
    };
    Ok(out
        .bytes
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

#[test]
fn the_pipeline_sums_every_tile() {
    assert_eq!(run(Fault::None), Ok(EXPECTED.to_vec()));
}

#[test]
fn waiting_for_too_little_reads_a_tile_that_has_not_landed() {
    // thread 0 reads its mirror's element, which nobody has landed yet; a thread reading its OWN in-flight element is
    // the `AsyncRace` of `kernel_async_copy.rs`
    assert_eq!(
        run(Fault::WaitsTooLittle),
        Err(KernelTrap::UninitializedShared { alloc: 0 })
    );
}

#[test]
fn without_the_read_barrier_a_thread_reads_what_has_not_landed_for_it() {
    // thread 0 reads thread 3's element before thread 3 has run its wait
    assert_eq!(
        run(Fault::NoReadBarrier),
        Err(KernelTrap::UninitializedShared { alloc: 0 })
    );
}

#[test]
fn without_the_reuse_barrier_a_refill_races_a_slow_reader() {
    assert_eq!(run(Fault::NoReuseBarrier), Err(KernelTrap::DataRace));
}
