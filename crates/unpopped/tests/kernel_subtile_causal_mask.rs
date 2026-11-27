// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A causal mask on an accumulator subtile** (survey row 35: ThunderKittens `warp::make_causal` on a diagonal subtile, read from
//! `kernels/mamba2/mamba2.cu`, `docs/ir-union-evidence/conformance-idiom-gaps-verified-2026-11-27.md`). One 32-lane subgroup holds
//! a 16 x 8 tile in the `m16n8` accumulator fragment layout the repo's atoms use (`mma.rs`: with `gid = lane / 4` and
//! `tig = lane % 4`, value `i` is row `gid + 8 * (i / 2)`, column `2 * tig + i % 2`); each lane zeroes the values above the
//! diagonal (`col > row`) with a `Select` and stores them back. The expected tile is Rust's lower triangle of the input. Mutants: a
//! strict triangle (the diagonal zeroed) and a fragment row that forgets the `+ 8` of the upper half; both change values.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const F32: ElementKind = ElementKind::F32;
const ROWS: usize = 16;
const COLS: usize = 8;

#[derive(Clone, Copy, PartialEq)]
enum Mutant {
    None,
    StrictTriangle,
    NoUpperHalfOffset,
}

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}
fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}
fn lane() -> ScalarExpr {
    let spec = CastSpec {
        to: I32,
        round: RoundMode::NearestEven,
        sat: Saturation::Wrap,
    };
    ScalarExpr::Cast(bx(ScalarExpr::Exec(ExecId::ThreadIdx(0))), spec)
}
fn cell(buffer: u32, row: ScalarExpr, col: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![row, col],
    }
}

/// Fragment value `i` of this lane: its (row, column) in the 16 x 8 tile.
fn coords(i: i64, mutant: Mutant) -> (ScalarExpr, ScalarExpr) {
    let gid = ScalarExpr::Div(bx(lane()), bx(ci(4)));
    let tig = ScalarExpr::Sub(
        bx(lane()),
        bx(ScalarExpr::Mul(
            bx(ScalarExpr::Div(bx(lane()), bx(ci(4)))),
            bx(ci(4)),
        )),
    );
    let half = if mutant == Mutant::NoUpperHalfOffset {
        0
    } else {
        8 * (i / 2)
    };
    let row = ScalarExpr::Add(bx(gid), bx(ci(half)));
    let col = ScalarExpr::Add(bx(ScalarExpr::Mul(bx(tig), bx(ci(2)))), bx(ci(i % 2)));
    (row, col)
}

/// Per fragment value `i` (locals `3i`, `3i + 1`, `3i + 2`: the loaded value, the row, the column; a local is declared once):
/// load, `Select` on the triangle, store.
fn body(mutant: Mutant) -> Vec<Stmt> {
    let mut v = Vec::new();
    for i in 0..4_u32 {
        let (val, r, c) = (3 * i, 3 * i + 1, 3 * i + 2);
        let (row, col) = coords(i64::from(i), mutant);
        v.push(Stmt::Let {
            local: r,
            init: row,
        });
        v.push(Stmt::Let {
            local: c,
            init: col,
        });
        v.push(Stmt::Load {
            local: val,
            place: cell(0, ScalarExpr::Local(r), ScalarExpr::Local(c)),
        });
        let op = if mutant == Mutant::StrictTriangle {
            BinaryOp::CmpGt
        } else {
            BinaryOp::CmpGe
        };
        let keep = ScalarExpr::Binary(op, bx(ScalarExpr::Local(r)), bx(ScalarExpr::Local(c)));
        let spec = CastSpec {
            to: F32,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        };
        let zero = ScalarExpr::Cast(bx(ci(0)), spec);
        v.push(Stmt::Store {
            place: cell(1, ScalarExpr::Local(r), ScalarExpr::Local(c)),
            value: ScalarExpr::Select(bx(keep), bx(ScalarExpr::Local(val)), bx(zero)),
        });
    }
    v
}

fn run(mutant: Mutant) -> Vec<f32> {
    let k = KernelFn::new(
        "causal",
        I32,
        vec![
            Param::buffer("inp", F32, BufferRole::Read, Some(2)),
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(2)),
        ],
        (0..4)
            .flat_map(|_| [F32, I32, I32].map(LocalDecl::scalar))
            .collect(),
        Block(body(mutant)),
    )
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(1, 32));
    assert_eq!(check_kernel(&k), Ok(()), "well formed");
    let shape = vec![ROWS as i64, COLS as i64];
    let strides = vec![COLS as i64, 1];
    let input: Vec<u8> = (1..=(ROWS * COLS) as u32)
        .flat_map(|x| (x as f32).to_le_bytes())
        .collect();
    let a = TypedBuffer::new(F32, shape.clone(), strides.clone(), input);
    let b = TypedBuffer::new(F32, shape, strides, vec![0u8; 4 * ROWS * COLS]);
    let mut args = vec![KernelArg::Buffer(a), KernelArg::Buffer(b)];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    out.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// The lower triangle (diagonal kept) of the input `1..=128`, row-major.
fn reference() -> Vec<f32> {
    (0..ROWS * COLS)
        .map(|n| {
            let (r, c) = (n / COLS, n % COLS);
            if c <= r { (n + 1) as f32 } else { 0.0 }
        })
        .collect()
}

#[test]
fn the_fragment_mask_is_the_lower_triangle_of_the_tile() {
    assert_eq!(run(Mutant::None), reference());
}

#[test]
fn the_mutants_change_the_tile_by_value() {
    assert_ne!(run(Mutant::StrictTriangle), reference());
    assert_ne!(run(Mutant::NoUpperHalfOffset), reference());
}
