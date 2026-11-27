// SPDX-License-Identifier: MIT OR Apache-2.0
//! The opaque-tile statements and `MemorySpace::Tile` (ROADMAP T2) are declined by name by `unpopped-cpu-c`, which spells
//! `Shared` and `Private` arrays and the per-lane `Stmt::Mma`: never lowered to something else, never silently dropped.
//! The same kernel without the tile lowers.

use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind::{F32, U32};

fn kernel(allocs: Vec<AllocDecl>, body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        "tile",
        F32,
        vec![Param::buffer("out", F32, BufferRole::Write, Some(1))],
        vec![LocalDecl::scalar(F32)],
        Block(body),
    )
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(32, 32))
    .with_allocs(allocs)
}

/// One of each tile statement over the tile `1` and the `Shared` allocation `0`.
fn statements() -> [(&'static str, Stmt); 4] {
    let u = |v| ScalarExpr::ConstInt(v, U32);
    [
        (
            "TileLoad",
            Stmt::TileLoad {
                d: 1,
                src: Place::Alloc {
                    alloc: 0,
                    index: u(0),
                },
                elements_per_row: u(2),
                origin: [u(0), u(0)],
            },
        ),
        (
            "TileFill",
            Stmt::TileFill {
                d: 1,
                value: ScalarExpr::Const(1.0),
            },
        ),
        (
            "TileMma",
            Stmt::TileMma {
                d: 1,
                a: 1,
                b: 1,
                c: 1,
            },
        ),
        (
            "TileStore",
            Stmt::TileStore {
                a: 1,
                dst: Place::Alloc {
                    alloc: 0,
                    index: u(0),
                },
                elements_per_row: u(2),
                origin: [u(0), u(0)],
            },
        ),
    ]
}

#[test]
fn a_tile_allocation_and_each_tile_statement_decline_by_name() {
    let shared = AllocDecl::new("s", MemorySpace::Shared, F32, Extent::Static(4));
    let plain = kernel(vec![shared.clone()], vec![]);
    assert!(CpuC.lower_kernel(&plain).is_ok(), "the control lowers");
    let allocs = vec![shared, AllocDecl::tile("t", F32, 2, 2)];
    for (name, st) in statements() {
        let k = kernel(allocs.clone(), vec![st]);
        assert_eq!(check_kernel(&k), Ok(()), "{name} is well formed");
        let err = CpuC.lower_kernel(&k).expect_err(name);
        assert!(format!("{err:?}").contains("Tile"), "{name}: {err:?}");
    }
    // the allocation alone is declined too
    let alloc_only = kernel(allocs, vec![]);
    let err = CpuC
        .lower_kernel(&alloc_only)
        .expect_err("a Tile allocation");
    assert!(format!("{err:?}").contains("Tile"), "{err:?}");
}
