// SPDX-License-Identifier: MIT OR Apache-2.0
//! The opaque-tile statements (ROADMAP T2, `docs/design-opaque-tile-mma.md`): `TileLoad`, `TileMma`, `TileFill`, `TileStore` over
//! `MemorySpace::Tile` allocations. A rectangular GEMM through two loads with an origin and a row stride, a fill and an MMA,
//! stored back, against a hand-written product; the uninitialised tile; and the checker's rules each with an accepted control.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt,
    check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{F32, U32};

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn shared(i: ScalarExpr) -> Place {
    Place::Alloc { alloc: 0, index: i }
}

fn tile(name: &str, rows: u32, cols: u32) -> AllocDecl {
    AllocDecl::tile(name, F32, rows, cols)
}

/// Copy the 32 inputs to a `Shared` array (every lane one element) and fence it, then run `body` over the tiles.
fn kernel(tiles: Vec<AllocDecl>, body: Vec<Stmt>) -> KernelFn {
    let mut stmts = vec![
        Stmt::Load {
            local: 0,
            place: at(0, tid()),
        },
        Stmt::Store {
            place: shared(tid()),
            value: ScalarExpr::Local(0),
        },
        Stmt::Barrier {
            scope: Scope::Block,
            kind: BarrierKind::Full,
        },
    ];
    stmts.extend(body);
    let mut allocs = vec![AllocDecl::new(
        "s",
        MemorySpace::Shared,
        F32,
        Extent::Static(32),
    )];
    allocs.extend(tiles);
    KernelFn::new(
        "tile",
        F32,
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(F32)],
        Block(stmts),
    )
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(32, 32))
    .with_allocs(allocs)
}

fn load(d: u32, origin: [i64; 2]) -> Stmt {
    Stmt::TileLoad {
        d,
        src: shared(u(0)),
        elements_per_row: u(8),
        origin: [u(origin[0]), u(origin[1])],
    }
}

fn store(a: u32) -> Stmt {
    Stmt::TileStore {
        a,
        dst: at(1, u(0)),
        elements_per_row: u(2),
        origin: [u(0), u(0)],
    }
}

fn run(k: &KernelFn) -> Result<Vec<f32>, KernelTrap> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let input: Vec<u8> = (0..32u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let buf =
        |bytes: Vec<u8>, n: i64| KernelArg::Buffer(TypedBuffer::new(F32, vec![n], vec![1], bytes));
    let mut args = vec![buf(input, 32), buf(vec![0u8; 16], 4)];
    run_launch(k, &mut args)?;
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    Ok(out
        .bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn refused(k: &KernelFn) -> &'static str {
    match check_kernel(k) {
        Err(KernelError::Tile { why, .. }) => why,
        other => panic!("expected a Tile refusal, got {other:?}"),
    }
}

#[test]
fn a_rectangular_tile_product_matches_a_hand_written_one() {
    // M is the 4 x 8 matrix M[r][c] = 8 r + c. A (2 x 3) is M at (1, 1); B (3 x 2) is M at (0, 4); D = A B + C with C = 1.
    let k = kernel(
        vec![
            tile("a", 2, 3),
            tile("b", 3, 2),
            tile("c", 2, 2),
            tile("d", 2, 2),
        ],
        vec![
            load(1, [1, 1]),
            load(2, [0, 4]),
            Stmt::TileFill {
                d: 3,
                value: ScalarExpr::Const(1.0),
            },
            Stmt::TileMma {
                d: 4,
                a: 1,
                b: 2,
                c: 3,
            },
            store(4),
        ],
    );
    let m = |r: usize, c: usize| (8 * r + c) as f32;
    let want: Vec<f32> = (0..2)
        .flat_map(|i| {
            (0..2).map(move |j| 1.0 + (0..3).map(|l| m(1 + i, 1 + l) * m(l, 4 + j)).sum::<f32>())
        })
        .collect();
    assert_eq!(run(&k), Ok(want));
}

#[test]
fn a_fill_and_an_accumulating_mma_in_place() {
    // c = 2; c = a b + c with the d of the statement the same tile as c; a = b = 2 x 2 of M at (0, 0)
    let k = kernel(
        vec![tile("a", 2, 2), tile("c", 2, 2)],
        vec![
            load(1, [0, 0]),
            Stmt::TileFill {
                d: 2,
                value: ScalarExpr::Const(2.0),
            },
            Stmt::TileMma {
                d: 2,
                a: 1,
                b: 1,
                c: 2,
            },
            store(2),
        ],
    );
    // A = [[0, 1], [8, 9]]; A A = [[8, 9], [72, 89]]
    assert_eq!(run(&k), Ok(vec![10.0, 11.0, 74.0, 91.0]));
}

#[test]
fn storing_a_tile_nothing_wrote_traps() {
    let k = kernel(vec![tile("a", 2, 2)], vec![store(1)]);
    assert!(matches!(
        run(&k),
        Err(KernelTrap::UninitializedPrivate { .. })
    ));
}

#[test]
fn a_tile_allocation_states_its_shape_and_no_ordinary_access_reaches_it() {
    let ok = kernel(vec![tile("a", 2, 2)], vec![load(1, [0, 0]), store(1)]);
    assert_eq!(check_kernel(&ok), Ok(()), "the control");
    // a Tile with no shape, and a shape on another space
    let mut bare = tile("a", 2, 2);
    bare.tile = None;
    assert!(refused(&kernel(vec![bare], vec![])).contains("without a tile shape"));
    let mut stray = AllocDecl::new("p", MemorySpace::Private, F32, Extent::Static(4));
    stray.tile = Some(unpopped::exec::TileShape::new(2, 2));
    assert!(refused(&kernel(vec![stray], vec![])).contains("not a Tile"));
    // an extent that is not rows * cols
    let mut ragged = tile("a", 2, 2);
    ragged.extent = Extent::Static(5);
    assert!(refused(&kernel(vec![ragged], vec![])).contains("rows * cols"));
    // an element type outside the float set
    let ints = AllocDecl::tile("a", U32, 2, 2);
    assert!(refused(&kernel(vec![ints], vec![])).contains("element type"));
    // an ordinary load of a tile element
    let peek = kernel(
        vec![tile("a", 2, 2)],
        vec![Stmt::Load {
            local: 0,
            place: Place::Alloc {
                alloc: 1,
                index: u(0),
            },
        }],
    );
    assert!(refused(&peek).contains("ordinary access"));
}

#[test]
fn each_tile_statement_rule_refuses_and_its_control_passes() {
    let mma = |d, a, b, c| Stmt::TileMma { d, a, b, c };
    let ok = kernel(
        vec![tile("a", 2, 3), tile("b", 3, 2), tile("c", 2, 2)],
        vec![mma(3, 1, 2, 3)],
    );
    assert_eq!(check_kernel(&ok), Ok(()), "(2x3)(3x2)+(2x2)");
    // inner dimensions that disagree, an accumulator of the wrong shape
    let inner = kernel(
        vec![tile("a", 2, 3), tile("b", 2, 2), tile("c", 2, 2)],
        vec![mma(3, 1, 2, 3)],
    );
    assert!(refused(&inner).contains("shapes"));
    let wrong_c = kernel(
        vec![tile("a", 2, 3), tile("b", 3, 2), tile("c", 2, 3)],
        vec![mma(3, 1, 2, 3)],
    );
    assert!(refused(&wrong_c).contains("shapes"));
    // a statement over an allocation that is not a tile
    let not_tile = kernel(vec![tile("a", 2, 2)], vec![load(0, [0, 0])]);
    assert!(refused(&not_tile).contains("not a Tile"));
    // a fill whose value is not the tile's element type
    let fill = |v| {
        kernel(
            vec![tile("a", 2, 2)],
            vec![Stmt::TileFill { d: 1, value: v }],
        )
    };
    assert_eq!(check_kernel(&fill(ScalarExpr::Const(1.0))), Ok(()));
    assert!(refused(&fill(u(1))).contains("element type"));
    // a place of another element type
    let wrong_place = kernel(
        vec![tile("a", 2, 2)],
        vec![Stmt::TileLoad {
            d: 1,
            src: at(0, u(0)),
            elements_per_row: u(2),
            origin: [u(0), u(0)],
        }],
    );
    assert_eq!(
        check_kernel(&wrong_place),
        Ok(()),
        "a global F32 buffer place is accepted"
    );
    let float_stride = kernel(
        vec![tile("a", 2, 2)],
        vec![Stmt::TileLoad {
            d: 1,
            src: shared(u(0)),
            elements_per_row: ScalarExpr::Const(2.0),
            origin: [u(0), u(0)],
        }],
    );
    assert!(refused(&float_stride).contains("integer"));
}

#[test]
fn a_tile_statement_under_divergent_control_is_refused() {
    let guarded = |cond| {
        kernel(
            vec![tile("a", 2, 2)],
            vec![Stmt::If {
                cond,
                then: Block(vec![load(1, [0, 0])]),
                els: Block(vec![]),
            }],
        )
    };
    assert_eq!(check_kernel(&guarded(u(1))), Ok(()), "uniform control");
    assert_eq!(
        check_kernel(&guarded(tid())),
        Err(KernelError::BarrierDivergent)
    );
}

#[test]
fn a_tile_allocation_round_trips_through_serde() {
    let a = tile("a", 2, 3);
    let back: AllocDecl = serde_json::from_str(&serde_json::to_string(&a).unwrap()).unwrap();
    assert_eq!(back, a);
    let plain = AllocDecl::new("s", MemorySpace::Shared, F32, Extent::Static(4));
    assert!(!serde_json::to_string(&plain).unwrap().contains("tile"));
}
