// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A launched copy of `F16` and `Bf16` through shared memory agrees with the interpreter** (ROADMAP U14): the 16-bit
//! patterns `unpopped_cpu_c::CpuC::lower_kernel` spells in an `unsigned short` move through `CopyAsync`, a barrier and
//! `Load`/`Store` of a `Shared` array unchanged, a NaN payload included.

use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[allow(dead_code)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[path = "support/launch_run.rs"]
mod launch_run;
use launch_run::agree_launch;

const F32: ElementKind = ElementKind::F32;
const U32: ElementKind = ElementKind::U32;

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn narrow(t: ElementKind, patterns: &[u16]) -> KernelArg {
    let bytes = patterns.iter().flat_map(|p| p.to_le_bytes()).collect();
    KernelArg::Buffer(TypedBuffer::new(
        t,
        vec![patterns.len() as i64],
        vec![1],
        bytes,
    ))
}

#[test]
fn a_launched_copy_through_shared_memory_agrees() {
    let pats: Vec<u16> = vec![
        0x3C00, 0x4000, 0x7E00, 0x8000, 0x0001, 0xFC00, 0x7BFF, 0x1234,
    ];
    for (tag, t) in [
        ("shared_f16", ElementKind::F16),
        ("shared_bf16", ElementKind::Bf16),
    ] {
        let k = mirrored_copy(tag, t);
        // the shared array of the pattern is spelled in the 16-bit carrier, not a float
        let src = CpuC.lower_kernel(&k).expect("lowers").source;
        assert!(
            src.contains("unsigned short"),
            "{tag}:
{src}"
        );
        agree_launch(tag, &k, vec![narrow(t, &pats), narrow(t, &[0; 8])]);
    }
}

/// `2 * v + k`, in `U32`.
fn pair_index(v: ScalarExpr, k: i64) -> ScalarExpr {
    ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(
            Box::new(v),
            Box::new(ScalarExpr::ConstInt(2, U32)),
        )),
        Box::new(ScalarExpr::ConstInt(k, U32)),
    )
}

/// Thread `t` of 4 copies `in[2t..2t+2]` (4 bytes) into shared memory, waits, and after a barrier writes the mirror
/// thread's pair to `out[2t..2t+2]`.
fn mirrored_copy(tag: &str, t: ElementKind) -> KernelFn {
    let params = vec![
        Param::buffer("in", t, BufferRole::Read, Some(1)),
        Param::buffer("out", t, BufferRole::Write, Some(1)),
    ];
    KernelFn::new(
        tag,
        F32,
        params,
        vec![LocalDecl::scalar(t), LocalDecl::scalar(t)],
        Block(mirrored_copy_body()),
    )
    .with_allocs(vec![AllocDecl::new(
        "tile",
        MemorySpace::Shared,
        t,
        Extent::Static(8),
    )])
    .with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]))
}

/// The statements of [`mirrored_copy`]: copy, commit, wait, barrier, then the two reads and writes.
fn mirrored_copy_body() -> Vec<Stmt> {
    let tid = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let mirror = ScalarExpr::Sub(
        Box::new(ScalarExpr::ConstInt(3, U32)),
        Box::new(tid.clone()),
    );
    let mut body = vec![
        Stmt::CopyAsync {
            dst: Place::Alloc {
                alloc: 0,
                index: pair_index(tid.clone(), 0),
            },
            src: at(0, pair_index(tid.clone(), 0)),
            elems: 2,
        },
        Stmt::AsyncCommit,
        Stmt::AsyncWait { pending: 0 },
        Stmt::Barrier {
            scope: Scope::Block,
            kind: BarrierKind::Full,
        },
    ];
    for k in 0..2 {
        body.push(Stmt::Load {
            local: k as u32,
            place: Place::Alloc {
                alloc: 0,
                index: pair_index(mirror.clone(), k),
            },
        });
        body.push(Stmt::Store {
            place: at(1, pair_index(tid.clone(), k)),
            value: ScalarExpr::Local(k as u32),
        });
    }
    body
}
