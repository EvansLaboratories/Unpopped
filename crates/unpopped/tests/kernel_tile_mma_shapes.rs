// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The WMMA and Apple `simdgroup_matrix` shapes as T2 opaque-tile products** (ROADMAP U7 theme 3, rows "wmma shapes" and
//! "Metal 8x8"; `docs/design-opaque-tile-mma.md`, `docs/mma-atom-coverage.md`). PTX ISA 9.5 section 9.7.16.4.1 leaves the
//! WMMA fragment layout unspecified and MSL 4.1 section 2.4 (page 38) leaves the `simdgroup_matrix` mapping unspecified, so
//! neither is an `MmaAtom`: each shape is a `TileLoad` of A (`m x k`), B (`k x n`) and C (`m x n`), a `TileMma` and a
//! `TileStore`, compared with a plain triple loop in `f64`. Shapes: WMMA `m16n16k16`, `m8n32k16`, `m32n8k16` over f16 and
//! bf16 into f32 (PTX 9.7.16.1 Table 44, p. 510); WMMA f64 `m8n8k4`; Apple 8x8x8 over half, bfloat and float into float
//! (MSL section 6.8). WMMA u8/s8, sub-byte, b1 and tf32 are not tile element types today (`kernel_check/tile.rs`) and are
//! not built here. Layouts are row-major (`elements_per_row` = the row length); a transposed tile is not an option (patent
//! hold, no `transpose`).

use unpopped::exec::{AllocDecl, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, Param, Place, Stmt, check_kernel};
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, Bf16, F16, F32, F64, U32};

#[path = "support/mma_run.rs"]
#[allow(dead_code, reason = "shared helpers: each test file uses a subset")]
mod support;
use support::{buffer, decode, value};

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![u(0)],
    }
}

fn load(d: u32, buf: u32, per_row: u32) -> Stmt {
    Stmt::TileLoad {
        d,
        src: at(buf),
        elements_per_row: u(per_row.into()),
        origin: [u(0), u(0)],
    }
}

/// `D = A.B + C` over `m x k`, `k x n`, `m x n` tiles loaded from the buffers `a`, `b`, `c` and stored to `d`.
fn product(m: u32, n: u32, k: u32, elem: ElementKind, acc: ElementKind) -> KernelFn {
    let body = vec![
        load(0, 0, k),
        load(1, 1, n),
        load(2, 2, n),
        Stmt::TileMma {
            d: 3,
            a: 0,
            b: 1,
            c: 2,
        },
        Stmt::TileStore {
            a: 3,
            dst: at(3),
            elements_per_row: u(n.into()),
            origin: [u(0), u(0)],
        },
    ];
    KernelFn::new(
        "tile_product",
        acc,
        vec![
            Param::buffer("a", elem, BufferRole::Read, Some(1)),
            Param::buffer("b", elem, BufferRole::Read, Some(1)),
            Param::buffer("c", acc, BufferRole::Read, Some(1)),
            Param::buffer("d", acc, BufferRole::Write, Some(1)),
        ],
        vec![],
        Block(body),
    )
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(32, 32))
    .with_allocs(vec![
        AllocDecl::tile("a", elem, m, k),
        AllocDecl::tile("b", elem, k, n),
        AllocDecl::tile("c", acc, m, n),
        AllocDecl::tile("d", acc, m, n),
    ])
}

/// Run one shape and compare every element of D with the triple loop.
fn check(m: u32, n: u32, k: u32, elem: ElementKind, acc: ElementKind) {
    let kernel = product(m, n, k, elem, acc);
    assert_eq!(check_kernel(&kernel), Ok(()), "{m}x{n}x{k} {elem:?}");
    let a: Vec<f64> = (0..m * k).map(|i| value(elem, 1, i)).collect();
    let b: Vec<f64> = (0..k * n).map(|i| value(elem, 2, i)).collect();
    let c: Vec<f64> = (0..m * n).map(|i| value(acc, 3, i)).collect();
    let mut args = vec![
        KernelArg::Buffer(buffer(elem, &a)),
        KernelArg::Buffer(buffer(elem, &b)),
        KernelArg::Buffer(buffer(acc, &c)),
        KernelArg::Buffer(buffer(acc, &vec![0.0; (m * n) as usize])),
    ];
    run_launch(&kernel, &mut args).unwrap_or_else(|e| panic!("{m}x{n}x{k} {elem:?}: {e:?}"));
    let KernelArg::Buffer(out) = &args[3] else {
        unreachable!("d is a buffer")
    };
    let got: Vec<f64> = out
        .bytes
        .chunks(out.bytes.len() / (m * n) as usize)
        .map(|x| decode(acc, x))
        .collect();
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let want: Vec<f64> = (0..m * n)
        .map(|ij| {
            let (i, j) = (ij / n, ij % n);
            (0..k).fold(c[ij], |s, l| s + a[i * k + l] * b[l * n + j])
        })
        .collect();
    assert_eq!(got, want, "{m}x{n}x{k} {elem:?}");
}

#[test]
fn the_wmma_half_shapes_are_tile_products() {
    for elem in [F16, Bf16] {
        for (m, n, k) in [(16, 16, 16), (8, 32, 16), (32, 8, 16)] {
            check(m, n, k, elem, F32);
        }
    }
}

#[test]
fn the_wmma_f64_shape_is_a_tile_product() {
    check(8, 8, 4, F64, F64);
}

#[test]
fn the_apple_8x8_shapes_are_tile_products() {
    for elem in [F16, Bf16, F32] {
        check(8, 8, 8, elem, F32);
    }
}

#[test]
fn a_shape_with_the_wrong_k_is_refused_by_the_checker() {
    // control: the wmma shapes above pass; a B tile whose rows are not the A tile's columns does not.
    let mut k = product(16, 16, 16, F16, F32);
    k.allocs[1] = AllocDecl::tile("b", F16, 8, 16);
    assert!(check_kernel(&k).is_err(), "k disagrees between A and B");
}
