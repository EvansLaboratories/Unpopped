// SPDX-License-Identifier: MIT OR Apache-2.0
//! Conformance kernels for the shapes that have no per-lane fragment layout and so are not `MmaAtom`s (ROADMAP U7 theme 3):
//! the WMMA shapes of PTX 9.7.16.4 (m16n16k16, m8n32k16, m32n8k16 over f16 and bf16; m8n8k4 over f64) and Apple's
//! `simdgroup_matrix` 8x8x8 (half, bfloat, float). Each runs as a T2 tile kernel (`TileLoad`, `TileMma`, `TileStore`) on the
//! oracle against a hand-written product. A, B and C sit inside larger row-major buffers (row stride = columns + 2, origin
//! (1, 1)) so the stride and the origin are part of what is checked, and D is stored into a sentinel-filled buffer whose border
//! must survive. Each shape also runs two value mutants (C not loaded; A read at the wrong origin) that must disagree.
//! Shapes the tile rules do not admit are in `docs/mma-atom-coverage.md`.

use unpopped::exec::{AllocDecl, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, Bf16, F16, F32, F64, U32};

const PAD: i64 = 2;
const SENTINEL: f64 = -777.0;

struct Shape {
    name: &'static str,
    m: u32,
    n: u32,
    k: u32,
    elem: ElementKind,
    acc: ElementKind,
}

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![u(0)],
    }
}

/// Which deliberate defect to build in (`None` is the conforming kernel).
#[derive(Clone, Copy, PartialEq)]
enum Mutant {
    None,
    NoAccumulator,
    WrongOriginA,
}

fn kernel(s: &Shape, mutant: Mutant) -> KernelFn {
    let ld = |cols: u32| u(i64::from(cols) + PAD);
    let load = |d: u32, buffer: u32, cols: u32, origin: i64| Stmt::TileLoad {
        d,
        src: at(buffer),
        elements_per_row: ld(cols),
        origin: [u(origin), u(origin)],
    };
    let c_stmt = if mutant == Mutant::NoAccumulator {
        Stmt::TileFill {
            d: 2,
            value: ScalarExpr::Const(0.0),
        }
    } else {
        load(2, 2, s.n, 1)
    };
    let body = vec![
        load(
            0,
            0,
            s.k,
            if mutant == Mutant::WrongOriginA { 0 } else { 1 },
        ),
        load(1, 1, s.n, 1),
        c_stmt,
        Stmt::TileMma {
            d: 3,
            a: 0,
            b: 1,
            c: 2,
        },
        Stmt::TileStore {
            a: 3,
            dst: at(3),
            elements_per_row: ld(s.n),
            origin: [u(1), u(1)],
        },
    ];
    KernelFn::new(
        s.name,
        s.acc,
        vec![
            Param::buffer("a", s.elem, BufferRole::Read, Some(1)),
            Param::buffer("b", s.elem, BufferRole::Read, Some(1)),
            Param::buffer("c", s.acc, BufferRole::Read, Some(1)),
            Param::buffer("d", s.acc, BufferRole::Write, Some(1)),
        ],
        Vec::<LocalDecl>::new(),
        Block(body),
    )
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(32, 32))
    .with_allocs(vec![
        AllocDecl::tile("a", s.elem, s.m, s.k),
        AllocDecl::tile("b", s.elem, s.k, s.n),
        AllocDecl::tile("c", s.acc, s.m, s.n),
        AllocDecl::tile("d", s.acc, s.m, s.n),
    ])
}

/// A padded `rows x cols` matrix (row stride `cols + PAD`, a border of `PAD` columns and rows) whose interior element
/// `(i, j)` is `f(i, j)`, and whose border is `border`.
fn padded(rows: u32, cols: u32, border: f64, f: impl Fn(i64, i64) -> f64) -> Vec<f64> {
    let (r, c) = (i64::from(rows) + PAD, i64::from(cols) + PAD);
    (0..r)
        .flat_map(|y| {
            let f = &f;
            (0..c).map(move |x| {
                if y >= 1 && x >= 1 && y <= i64::from(rows) && x <= i64::from(cols) {
                    f(y - 1, x - 1)
                } else {
                    border
                }
            })
        })
        .collect()
}

fn fa(i: i64, l: i64) -> f64 {
    ((i * 7 + l * 3) % 5 - 2) as f64
}
fn fb(l: i64, j: i64) -> f64 {
    ((l * 5 + j * 3) % 7 - 3) as f64
}
fn fc(i: i64, j: i64) -> f64 {
    ((i + 2 * j) % 4) as f64
}

fn run(s: &Shape, mutant: Mutant) -> Vec<f64> {
    let k = kernel(s, mutant);
    assert_eq!(check_kernel(&k), Ok(()), "{} is well formed", s.name);
    let buf = |dt, rows: u32, cols: u32, data: Vec<f64>| {
        let n = i64::from((rows + PAD as u32) * (cols + PAD as u32));
        KernelArg::Buffer(TypedBuffer::from_f64_as(dt, &[n], &data))
    };
    let mut args = vec![
        buf(s.elem, s.m, s.k, padded(s.m, s.k, 9.0, fa)),
        buf(s.elem, s.k, s.n, padded(s.k, s.n, 9.0, fb)),
        buf(s.acc, s.m, s.n, padded(s.m, s.n, 9.0, fc)),
        buf(s.acc, s.m, s.n, padded(s.m, s.n, SENTINEL, |_, _| SENTINEL)),
    ];
    run_launch(&k, &mut args).unwrap_or_else(|e| panic!("{}: {e:?}", s.name));
    let KernelArg::Buffer(d) = &args[3] else {
        unreachable!()
    };
    d.to_f64_vec()
}

/// The hand-written product, in the same padded layout as D.
fn want(s: &Shape) -> Vec<f64> {
    padded(s.m, s.n, SENTINEL, |i, j| {
        fc(i, j)
            + (0..i64::from(s.k))
                .map(|l| fa(i, l) * fb(l, j))
                .sum::<f64>()
    })
}

fn conforms(s: &Shape) {
    let w = want(s);
    assert_eq!(run(s, Mutant::None), w, "{}: D = A B + C", s.name);
    assert_ne!(
        run(s, Mutant::NoAccumulator),
        w,
        "{}: a kernel that drops C must disagree",
        s.name
    );
    assert_ne!(
        run(s, Mutant::WrongOriginA),
        w,
        "{}: a kernel that reads A at the wrong origin must disagree",
        s.name
    );
}

const fn shape(name: &'static str, m: u32, n: u32, k: u32, elem: ElementKind) -> Shape {
    let acc = if matches!(elem, F64) { F64 } else { F32 };
    Shape {
        name,
        m,
        n,
        k,
        elem,
        acc,
    }
}

#[test]
fn wmma_f16_shapes_run_as_tiles() {
    for (n, m, nn, k) in [
        ("wmma_m16n16k16_f16", 16, 16, 16),
        ("wmma_m8n32k16_f16", 8, 32, 16),
        ("wmma_m32n8k16_f16", 32, 8, 16),
    ] {
        conforms(&shape(n, m, nn, k, F16));
    }
}

#[test]
fn wmma_bf16_shapes_run_as_tiles() {
    for (n, m, nn, k) in [
        ("wmma_m16n16k16_bf16", 16, 16, 16),
        ("wmma_m8n32k16_bf16", 8, 32, 16),
        ("wmma_m32n8k16_bf16", 32, 8, 16),
    ] {
        conforms(&shape(n, m, nn, k, Bf16));
    }
}

#[test]
fn wmma_f64_m8n8k4_runs_as_a_tile() {
    conforms(&shape("wmma_m8n8k4_f64", 8, 8, 4, F64));
}

#[test]
fn apple_simdgroup_8x8x8_runs_as_a_tile_in_half_bfloat_and_float() {
    conforms(&shape("simdgroup_8x8x8_half", 8, 8, 8, F16));
    conforms(&shape("simdgroup_8x8x8_bfloat", 8, 8, 8, Bf16));
    conforms(&shape("simdgroup_8x8x8_float", 8, 8, 8, F32));
}
