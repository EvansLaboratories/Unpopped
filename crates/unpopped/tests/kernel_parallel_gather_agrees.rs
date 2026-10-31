// SPDX-License-Identifier: MIT OR Apache-2.0
//! **E1d-6: a grid-stride launched Gather is bit-exact against the serial kernel**
//! (`docs/e1d-parallel-mapping.md`, "E1d-6 as built").
//!
//! Every output cell reads its own index and source element and is written by one thread, so the
//! launched kernel must equal the serial one on every geometry and every out-of-range policy, on a
//! pre-filled output (a skipped cell keeps its prior bytes), for a full-shape index and a 1-D index
//! broadcast over the other axis. A broken copy (step 1) must disagree.

use unpopped::ir::{OobPolicy, OpDef};
use unpopped::kernel::{KernelFn, Stmt, check_kernel};
use unpopped::kernel_lift::{LiftDecline, lift_gather_opdef, lift_gather_opdef_launched};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;
const GEOMETRIES: [(u32, u32); 5] = [(1, 1), (2, 3), (3, 8), (4, 32), (16, 64)];
const POLICIES: [OobPolicy; 3] = [OobPolicy::Skip, OobPolicy::Clamp, OobPolicy::ZeroFill];

/// Data `[5, 6]` gathered along `axis` into an output `[7, 6]` (axis 0) or `[5, 4]` (axis 1).
#[derive(Clone, Copy)]
struct Case {
    axis: u8,
    out: [i64; 2],
    ids_1d: bool,
}

const CASES: [Case; 3] = [
    Case {
        axis: 0,
        out: [7, 6],
        ids_1d: false,
    },
    Case {
        axis: 1,
        out: [5, 4],
        ids_1d: false,
    },
    Case {
        axis: 0,
        out: [9, 6],
        ids_1d: true,
    },
];
const DATA: [i64; 2] = [5, 6];

fn data() -> TypedBuffer {
    let bytes = (0..30)
        .flat_map(|i| (i as f32 * 1.5 - 7.0).to_le_bytes())
        .collect();
    TypedBuffer::new(F32, DATA.to_vec(), vec![6, 1], bytes)
}

/// In range, the last valid index, one past it, negatives and far values.
fn index(case: Case) -> TypedBuffer {
    let extent = DATA[usize::from(case.axis)] as i32;
    let pool = [
        0,
        extent - 1,
        1,
        extent,
        extent + 7,
        2,
        -1,
        i32::MIN,
        i32::MAX,
        -3,
    ];
    let n = if case.ids_1d {
        case.out[0]
    } else {
        case.out[0] * case.out[1]
    } as usize;
    let bytes = (0..n)
        .flat_map(|i| pool[(i * 3 + i / 4) % pool.len()].to_le_bytes())
        .collect();
    let strides = if case.ids_1d {
        vec![1, 0]
    } else {
        vec![case.out[1], 1]
    };
    TypedBuffer::new(I32, case.out.to_vec(), strides, bytes)
}

fn run(k: &KernelFn, case: Case, launch: bool) -> Vec<u8> {
    let n = (case.out[0] * case.out[1]) as usize * 4;
    let out = TypedBuffer::new(F32, case.out.to_vec(), vec![case.out[1], 1], vec![0xAB; n]);
    let mut args = vec![KernelArg::Buffer(data()), KernelArg::Buffer(index(case))];
    args.push(KernelArg::Buffer(out));
    args.extend(case.out.iter().map(|&e| KernelArg::Int(i128::from(e))));
    args.push(KernelArg::Int(i128::from(DATA[usize::from(case.axis)])));
    if launch {
        run_launch(k, &mut args).expect("the launch runs");
    } else {
        run_kernel(k, &mut args).expect("the kernel runs");
    }
    let KernelArg::Buffer(b) = &args[2] else {
        unreachable!()
    };
    b.bytes.clone()
}

fn op(case: Case, oob: OobPolicy) -> OpDef {
    OpDef::gather("g", &[F32], case.axis, oob, I32)
}

#[test]
fn a_launched_gather_equals_the_serial_one_on_every_geometry_and_policy() {
    for case in CASES {
        for oob in POLICIES {
            let serial = lift_gather_opdef(&op(case, oob), F32, 2).unwrap();
            let want = run(&serial, case, false);
            for (grid, block) in GEOMETRIES {
                let k = lift_gather_opdef_launched(&op(case, oob), F32, 2, block, grid).unwrap();
                assert_eq!(check_kernel(&k), Ok(()), "well formed");
                assert_eq!(
                    run(&k, case, true),
                    want,
                    "{oob:?} axis {} {grid}x{block}",
                    case.axis
                );
            }
        }
    }
}

#[test]
fn a_skipped_cell_keeps_its_prior_bytes_in_the_launched_kernel() {
    let case = CASES[0];
    let k = lift_gather_opdef_launched(&op(case, OobPolicy::Skip), F32, 2, 3, 2).unwrap();
    assert!(
        run(&k, case, true).contains(&0xAB),
        "an out-of-range cell was not written"
    );
}

#[test]
fn a_zero_launch_declines() {
    let o = op(CASES[0], OobPolicy::Skip);
    for (b, g) in [(0, 1), (1, 0)] {
        let r = lift_gather_opdef_launched(&o, F32, 2, b, g);
        assert_eq!(r.err(), Some(LiftDecline::BadLaunch));
    }
}

#[test]
fn a_broken_step_is_seen() {
    let case = CASES[0];
    let o = op(case, OobPolicy::Clamp);
    let want = run(&lift_gather_opdef(&o, F32, 2).unwrap(), case, false);
    let mut k = lift_gather_opdef_launched(&o, F32, 2, 2, 2).unwrap();
    let Some(Stmt::For { step, .. }) = k.body.0.first_mut() else {
        panic!("a loop")
    };
    *step = unpopped::ir::ScalarExpr::ConstInt(1, I32);
    let n = (case.out[0] * case.out[1]) as usize * 4;
    let out = TypedBuffer::new(F32, case.out.to_vec(), vec![case.out[1], 1], vec![0xAB; n]);
    let mut args = vec![KernelArg::Buffer(data()), KernelArg::Buffer(index(case))];
    args.push(KernelArg::Buffer(out));
    args.extend(case.out.iter().map(|&e| KernelArg::Int(i128::from(e))));
    args.push(KernelArg::Int(5));
    let seen = run_launch(&k, &mut args).is_err()
        || !matches!(&args[2], KernelArg::Buffer(b) if b.bytes == want);
    assert!(
        seen,
        "a wrong step must be seen as a data race or a different result"
    );
}
