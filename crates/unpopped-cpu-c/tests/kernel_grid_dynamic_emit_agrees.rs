// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of a grid read from scalar arguments agrees with the launch interpreter** (ROADMAP U22): the same kernel
//! compiled once is run at several argument values, each matching `run_launch` bit for bit; `gridDim` read in the kernel is
//! the resolved value in C too.

use unpopped::backend::Backend;
use unpopped::exec::GridDim;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;

#[path = "support/cc.rs"]
mod cc;
#[path = "../../unpopped/tests/support/grid_dynamic.rs"]
mod grid_dynamic;
#[expect(
    dead_code,
    reason = "the shared driver; this file uses the part that runs a launched kernel"
)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[expect(
    dead_code,
    reason = "the shared runner; this file uses the tolerance form"
)]
#[path = "support/launch_run.rs"]
mod launch_run;
use grid_dynamic::{BLOCK, F32, doubler, grid_probe};
use launch_run::agree_launch_within;

fn f32_buf(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn args(n: i128) -> Vec<KernelArg> {
    let input: Vec<f32> = (0..12).map(|i| i as f32 + 1.0).collect();
    vec![f32_buf(&input), f32_buf(&[-1.0; 12]), KernelArg::Int(n)]
}

#[test]
fn the_c_ceil_div_grid_agrees_with_the_oracle_at_each_size() {
    let k = doubler(GridDim::CeilDiv {
        param: 0,
        by: BLOCK,
    });
    assert!(CpuC.lower_kernel(&k).is_ok(), "the kernel lowers");
    for n in [1, 3, 4, 5, 8, 11, 12] {
        agree_launch_within(&format!("doubler n={n}"), &k, args(n), 0);
    }
}

#[test]
fn the_c_grid_dim_is_the_resolved_value() {
    for (grid, sizes) in [
        (
            GridDim::CeilDiv {
                param: 0,
                by: BLOCK,
            },
            vec![1, 5, 12],
        ),
        (GridDim::Param(0), vec![1, 2, 7]),
    ] {
        let k = grid_probe(grid);
        assert!(CpuC.lower_kernel(&k).is_ok(), "the probe lowers");
        for n in sizes {
            agree_launch_within(&format!("grid probe n={n}"), &k, args(n), 0);
        }
    }
}
