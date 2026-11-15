// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A launch grid read from scalar arguments** (ROADMAP U22, baracuda#356 G1): `GridDim::CeilDiv { param, by }` and
//! `GridDim::Param` beside the static grid. The same kernel is run at several argument values; the answer must cover every
//! element each time (a grid of the wrong size leaves a tail untouched or runs blocks past the data), and `gridDim` read
//! inside the kernel must be the resolved value.

use unpopped::exec::GridDim;
use unpopped::kernel::{KernelError, KernelFn, Param, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/grid_dynamic.rs"]
mod grid_dynamic;
use grid_dynamic::{BLOCK, F32, doubler, grid_probe};

fn f32_buf(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn floats(a: &KernelArg) -> Vec<f32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!("a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn run(k: &KernelFn, n: i128, len: usize) -> Result<Vec<f32>, KernelTrap> {
    let input: Vec<f32> = (0..len).map(|i| i as f32 + 1.0).collect();
    let mut args = [
        f32_buf(&input),
        f32_buf(&vec![-1.0; len]),
        KernelArg::Int(n),
    ];
    run_launch(k, &mut args)?;
    Ok(floats(&args[1]))
}

#[test]
fn a_ceil_div_grid_covers_every_element_at_each_size() {
    let k = doubler(GridDim::CeilDiv {
        param: 0,
        by: BLOCK,
    });
    assert_eq!(check_kernel(&k), Ok(()));
    // buffers hold 12 floats; n below, at and between block multiples, and the full buffer
    for n in [1usize, 3, 4, 5, 8, 11, 12] {
        let out = run(&k, n as i128, 12).expect("the oracle runs it");
        for (i, o) in out.iter().enumerate() {
            let want = if i < n { 2.0 * (i as f32 + 1.0) } else { -1.0 };
            assert_eq!(*o, want, "n = {n}, element {i}");
        }
    }
}

#[test]
fn a_param_grid_is_the_argument_itself() {
    let k = grid_probe(GridDim::Param(0));
    for n in [1, 2, 7] {
        assert_eq!(run(&k, n, 12).unwrap()[0], n as f32, "n = {n}");
    }
}

#[test]
fn grid_dim_read_in_the_kernel_is_the_resolved_value() {
    let k = grid_probe(GridDim::CeilDiv {
        param: 0,
        by: BLOCK,
    });
    for (n, blocks) in [(1, 1.0), (4, 1.0), (5, 2.0), (12, 3.0)] {
        assert_eq!(run(&k, n, 12).unwrap()[0], blocks, "n = {n}");
    }
}

#[test]
fn an_argument_that_gives_no_blocks_is_refused() {
    let k = doubler(GridDim::CeilDiv {
        param: 0,
        by: BLOCK,
    });
    for n in [0i128, -3, i128::from(u32::MAX) * 8] {
        assert!(
            matches!(run(&k, n, 12), Err(KernelTrap::ArgMismatch(_))),
            "n = {n}"
        );
    }
}

#[test]
fn a_malformed_dynamic_grid_fails_the_check() {
    // a parameter local that no scalar parameter has, a zero divisor, a float parameter, and a cluster beside the grid
    let bad_local = doubler(GridDim::CeilDiv {
        param: 9,
        by: BLOCK,
    });
    let zero_by = doubler(GridDim::CeilDiv { param: 0, by: 0 });
    let mut float_param = doubler(GridDim::Param(0));
    float_param.params[2] = Param::scalar("n", F32, 0);
    let clustered = {
        let mut k = doubler(GridDim::Param(0));
        k.launch = k.launch.map(|l| l.with_cluster([1, 1, 1]));
        k
    };
    for (what, k) in [
        ("local", bad_local),
        ("by", zero_by),
        ("float", float_param),
        ("cluster", clustered),
    ] {
        assert_eq!(check_kernel(&k), Err(KernelError::BadLaunch), "{what}");
    }
    // the control: the well-formed kernel passes
    assert_eq!(check_kernel(&doubler(GridDim::Param(0))), Ok(()));
}
