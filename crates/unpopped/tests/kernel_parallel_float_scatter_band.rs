// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A launched F32 Scatter add is inside the reassociation band, and is not asserted bit-equal**
//! (`docs/ir-kernel-layer.md` section 5a; PM ruling 2026-11-01, option A).
//!
//! The order threads reach a destination cell is unspecified, so each cell is compared with the
//! exact sum of its initial value and the updates that hit it, inside
//! `gamma_(n-1) * sum(abs(x_i))` with `n` = hits + 1 (the initial value is a summand) and
//! `gamma_k = k*u / (1 - k*u)`, `u = 2^-24`. A cell no source names keeps its bytes exactly. The
//! data spans 2^-3 to 2^22 with mixed signs so a different order really gives different bytes. The
//! interpreter runs threads in lockstep (one schedule), so the difference is shown with two host
//! orders of the same summands; that is why no test asserts serial bytes.

use unpopped::ir::OpDef;
use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::kernel_lift::{lift_scatter_opdef, lift_scatter_opdef_launched};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;
const GEOMETRIES: [(u32, u32); 5] = [(1, 1), (2, 3), (3, 8), (4, 32), (16, 64)];
const SRC: [i64; 2] = [12, 3];
const DST: [i64; 2] = [4, 3];

/// Magnitudes from 2^-3 to 2^22 with mixed signs and a non-dyadic mantissa, so adding them in a
/// different order rounds differently.
fn update(i: usize) -> f32 {
    let sign = if (i * 7 + i / 5) % 3 == 0 { -1.0 } else { 1.0 };
    sign * 1.0000001_f32 * 2f32.powi(((i * 11) % 26) as i32 - 3)
}

fn target(i: usize) -> i32 {
    [0, 0, 1, 0, 0, 5, 1, 0, -1, 0, 1, 0][(i / SRC[1] as usize) % 12]
}

fn init(i: usize) -> f32 {
    (i as f32) * 0.375 - 2.0
}

fn buffers() -> Vec<KernelArg> {
    let f32s = |n: usize, f: &dyn Fn(usize) -> f32| -> Vec<u8> {
        (0..n).flat_map(|i| f(i).to_le_bytes()).collect()
    };
    let n_src = (SRC[0] * SRC[1]) as usize;
    let n_dst = (DST[0] * DST[1]) as usize;
    let idx: Vec<u8> = (0..n_src).flat_map(|i| target(i).to_le_bytes()).collect();
    vec![
        KernelArg::Buffer(TypedBuffer::new(
            F32,
            SRC.to_vec(),
            vec![SRC[1], 1],
            f32s(n_src, &update),
        )),
        KernelArg::Buffer(TypedBuffer::new(I32, SRC.to_vec(), vec![SRC[1], 1], idx)),
        KernelArg::Buffer(TypedBuffer::new(
            F32,
            DST.to_vec(),
            vec![DST[1], 1],
            f32s(n_dst, &init),
        )),
        KernelArg::Int(i128::from(SRC[0])),
        KernelArg::Int(i128::from(SRC[1])),
        KernelArg::Int(i128::from(DST[0])),
    ]
}

fn dest(k: &KernelFn, launch: bool) -> Vec<f32> {
    let mut args = buffers();
    if launch {
        run_launch(k, &mut args).expect("the launch runs");
    } else {
        run_kernel(k, &mut args).expect("the kernel runs");
    }
    let KernelArg::Buffer(b) = &args[2] else {
        unreachable!()
    };
    b.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// The exact sum (in `f64`) of cell `c`'s initial value and its updates, `sum(abs)`, and `n`.
fn cell_truth(c: usize) -> (f64, f64, usize) {
    let (mut sum, mut abs, mut n) = (f64::from(init(c)), f64::from(init(c)).abs(), 1);
    for i in 0..(SRC[0] * SRC[1]) as usize {
        let t = target(i);
        if t >= 0
            && (t as i64) < DST[0]
            && (t as usize) * DST[1] as usize + i % SRC[1] as usize == c
        {
            sum += f64::from(update(i));
            abs += f64::from(update(i)).abs();
            n += 1;
        }
    }
    (sum, abs, n)
}

fn within_band(got: f32, c: usize) -> bool {
    let (sum, abs, n) = cell_truth(c);
    let u = 2f64.powi(-24);
    let k = (n - 1) as f64 * u;
    let bound = k / (1.0 - k) * abs;
    (f64::from(got) - sum).abs() <= bound
}

fn scatter_add_op() -> OpDef {
    OpDef::scatter_add("s", &[F32], 0, I32)
}

#[test]
fn every_cell_is_inside_the_band_on_every_geometry_and_untouched_cells_are_exact() {
    for (grid, block) in GEOMETRIES {
        let k = lift_scatter_opdef_launched(&scatter_add_op(), F32, 2, block, grid).unwrap();
        assert_eq!(check_kernel(&k), Ok(()));
        for (c, got) in dest(&k, true).into_iter().enumerate() {
            assert!(within_band(got, c), "cell {c} {grid}x{block}: {got}");
            if cell_truth(c).2 == 1 {
                assert_eq!(got.to_bits(), init(c).to_bits(), "untouched cell {c}");
            }
        }
    }
}

/// The `f32` sum of cell `c`'s initial value and updates in ascending or descending source order.
fn host_sum(c: usize, ascending: bool) -> f32 {
    let mut hits: Vec<f32> = (0..(SRC[0] * SRC[1]) as usize)
        .filter(|&i| {
            let t = target(i);
            t >= 0
                && i64::from(t) < DST[0]
                && t as usize * DST[1] as usize + i % SRC[1] as usize == c
        })
        .map(update)
        .collect();
    if !ascending {
        hits.reverse();
    }
    hits.into_iter().fold(init(c), |acc, x| acc + x)
}

/// The interpreter keeps threads in lockstep, so one schedule cannot show a reordering; two host
/// orders of the same summands can. Both are inside the band yet differ, which is why the band is
/// the comparison and serial bytes are never asserted, and a wrong sum (one update dropped) is
/// outside it, so the band is not vacuous.
#[test]
fn two_orders_differ_yet_both_are_in_the_band_and_a_dropped_update_is_not() {
    let cells = (DST[0] * DST[1]) as usize;
    let differing: Vec<usize> = (0..cells)
        .filter(|&c| host_sum(c, true).to_bits() != host_sum(c, false).to_bits())
        .collect();
    assert!(!differing.is_empty(), "the data must make the order matter");
    for c in 0..cells {
        assert!(within_band(host_sum(c, true), c) && within_band(host_sum(c, false), c));
    }
    let c = differing[0];
    let dropped = host_sum(c, true) - update(0).max(update(1)).max(1.0);
    assert!(
        !within_band(dropped, c),
        "a wrong sum must fall outside the band"
    );
    let serial = lift_scatter_opdef(&scatter_add_op(), F32, 2).unwrap();
    assert!(
        dest(&serial, false)
            .iter()
            .enumerate()
            .all(|(c, v)| within_band(*v, c))
    );
}
