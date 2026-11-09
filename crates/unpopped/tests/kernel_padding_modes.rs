// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Out-of-bounds padding modes at the kernel layer** (ROADMAP R5.7 C4b): zero, NaN, constant, replicate (clamp) and reflect,
//! the modes the grid's C4b row lists, as one window read of five elements at positions `-2..7`. Zero, NaN and constant are a
//! `Stmt::MaskedLoad`; replicate and reflect are an index remap in front of an ordinary load. The expected rows are worked out
//! by hand from the definition of each mode, not by a model of it. No language in the survey has reflect or replicate as a
//! built-in load mode (`docs/ir-union-master-capabilities.md`, C4b), so none is added to the IR: they are expressible, and
//! this test is the evidence.

use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};

#[path = "support/padding_modes.rs"]
mod padding_modes;
use padding_modes::{F32, IN_LEN, Mode, TAPS, window};

const DATA: [f32; IN_LEN as usize] = [10.0, 11.0, 12.0, 13.0, 14.0];

fn run(mode: Mode) -> Vec<f32> {
    let bytes: Vec<u8> = DATA.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::new(F32, vec![IN_LEN], vec![1], bytes)),
        KernelArg::Buffer(TypedBuffer::new(
            F32,
            vec![TAPS],
            vec![1],
            vec![0; 4 * TAPS as usize],
        )),
    ];
    run_kernel(&window(mode), &mut args).expect("the oracle runs it");
    let KernelArg::Buffer(b) = &args[1] else {
        unreachable!()
    };
    b.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn same(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}[{i}]: got {g}, want {w}");
    }
}

#[test]
fn zero_nan_and_constant_fill_the_two_taps_on_each_side() {
    let (a, b, c, d, e) = (10.0, 11.0, 12.0, 13.0, 14.0);
    same(
        &run(Mode::Zero),
        &[0.0, 0.0, a, b, c, d, e, 0.0, 0.0],
        "zero",
    );
    let n = f32::NAN;
    same(&run(Mode::Nan), &[n, n, a, b, c, d, e, n, n], "nan");
    same(
        &run(Mode::Constant(7.0)),
        &[7.0, 7.0, a, b, c, d, e, 7.0, 7.0],
        "constant",
    );
    // a negative-infinity pad, the max-pool convention
    let m = f32::NEG_INFINITY;
    same(
        &run(Mode::Constant(f64::NEG_INFINITY)),
        &[m, m, a, b, c, d, e, m, m],
        "-inf",
    );
}

#[test]
fn replicate_repeats_the_edge_and_reflect_mirrors_about_it() {
    let (a, b, c, d, e) = (10.0, 11.0, 12.0, 13.0, 14.0);
    // positions -2 -1 | 0 1 2 3 4 | 5 6
    same(
        &run(Mode::Replicate),
        &[a, a, a, b, c, d, e, e, e],
        "replicate",
    );
    // reflect101: -2 -> 2, -1 -> 1, 5 -> 3, 6 -> 2 (the edge element is not repeated)
    same(&run(Mode::Reflect), &[c, b, a, b, c, d, e, d, c], "reflect");
}

#[test]
fn the_modes_differ_where_they_should() {
    // the control that the table above is not satisfied by one mode: at the first tap every mode gives a different value
    let first: Vec<u32> = [
        Mode::Zero,
        Mode::Nan,
        Mode::Constant(7.0),
        Mode::Replicate,
        Mode::Reflect,
    ]
    .iter()
    .map(|m| run(*m)[0].to_bits())
    .collect();
    let mut sorted = first.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), first.len(), "{first:?}");
}
