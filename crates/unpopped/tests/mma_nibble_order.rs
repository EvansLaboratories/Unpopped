// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The nibble order the MMA harness packs 4-bit fragments in is the oracle's, pinned from outside the harness.** The harness
//! packs the fragment buffers of the `I4`, `U4` and `E2M1` atoms (two to a byte) and the oracle unpacks them; a swapped
//! order in both would still agree with itself, so this test fixes the order on a known byte, `0x21`, and reads it through the
//! oracle's own load: element 0 is the LOW nibble (`1`), element 1 the HIGH nibble (`2`). `E2M1` codes 1 and 2 are 0.5 and 1.0
//! (OCP MX), so that format is pinned by value too.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, F4E2M1, F32, I4, I32, U4};

#[path = "support/mma_run.rs"]
#[allow(dead_code, reason = "shared helpers: each test file uses a subset")]
mod support;

/// `e` as an `F32`: a sub-byte integer widens to `I32` first (the IR's rule), `E2M1` casts directly.
fn widen(ty: ElementKind, e: ScalarExpr) -> ScalarExpr {
    let cast = |e, to| {
        ScalarExpr::Cast(
            Box::new(e),
            CastSpec {
                to,
                round: RoundMode::NearestEven,
                sat: Saturation::Wrap,
            },
        )
    };
    if ty == F4E2M1 {
        cast(e, F32)
    } else {
        cast(cast(e, I32), F32)
    }
}

/// `out[tid] = (f32) in[tid]` over two threads: the oracle's own load of the two nibbles of one byte.
fn read_two(ty: ElementKind, packed: &[u8]) -> Vec<f32> {
    let tid = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let at = |b: u32| Place::Logical {
        buffer: b,
        coords: vec![tid.clone()],
    };
    let k = KernelFn::new(
        "nibbles",
        F32,
        vec![
            Param::buffer("in", ty, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(ty)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value: widen(ty, ScalarExpr::Local(0)),
            },
        ]),
    )
    .with_launch(LaunchShape::new([2, 1, 1], [1, 1, 1]));
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_packed_bytes(ty, &[2], packed)),
        KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[0.0, 0.0])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!("out is a buffer")
    };
    out.bytes
        .chunks(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn the_oracle_reads_the_low_nibble_as_element_0() {
    assert_eq!(read_two(I4, &[0x21]), vec![1.0, 2.0], "I4");
    assert_eq!(read_two(U4, &[0x21]), vec![1.0, 2.0], "U4");
    assert_eq!(read_two(F4E2M1, &[0x21]), vec![0.5, 1.0], "E2M1");
}

#[test]
fn the_harness_packs_the_same_order_as_the_oracle_reads() {
    // the harness's packed bytes for elements [1, 2] are exactly 0x21, the byte the oracle reads above as [1, 2]
    for (ty, xs) in [(I4, [1.0, 2.0]), (U4, [1.0, 2.0]), (F4E2M1, [0.5, 1.0])] {
        assert_eq!(support::buffer(ty, &xs).raw_bytes(), &[0x21], "{ty:?}");
    }
}
