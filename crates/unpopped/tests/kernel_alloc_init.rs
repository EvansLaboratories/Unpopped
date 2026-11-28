// SPDX-License-Identifier: MIT OR Apache-2.0
//! An initial table on an allocation (`AllocDecl::init`, ROADMAP U40 and U66, `docs/options-u40-constant-tables.md` option A): the checker's
//! rules each with an accepted control beside the refusal, the oracle's meaning (a `Constant` table is read-only and the same in every
//! block, a `Private` table is every thread's own copy, a float is its bit pattern so `-0.0` and `1.5` survive), and the serde default (a
//! kernel without one serializes as before).

use unpopped::exec::{AllocDecl, AllocLayout, ExecId, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, F32, I8, U8, U32};

const N: u32 = 8;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

/// The thread's index in the grid (two blocks of `N`).
fn gid() -> ScalarExpr {
    ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(
            Box::new(ScalarExpr::Exec(ExecId::BlockIdx(0))),
            Box::new(ScalarExpr::ConstInt(i64::from(N), U32)),
        )),
        Box::new(tid()),
    )
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

/// `out[g] = table[idx[g]]`: thread `g` of the grid reads the table at the index the input names.
fn kernel(table: AllocDecl) -> KernelFn {
    let elem = table.elem;
    let body = Block(vec![
        Stmt::Load {
            local: 0,
            place: at(0, gid()),
        },
        Stmt::Load {
            local: 1,
            place: Place::Alloc {
                alloc: 0,
                index: ScalarExpr::Local(0),
            },
        },
        Stmt::Store {
            place: at(1, gid()),
            value: ScalarExpr::Local(1),
        },
    ]);
    KernelFn::new(
        "init",
        elem,
        vec![
            Param::buffer("idx", U32, BufferRole::Read, Some(1)),
            Param::buffer("out", elem, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(U32), LocalDecl::scalar(elem)],
        body,
    )
    .with_launch(LaunchShape::new([N, 1, 1], [2, 1, 1]))
    .with_allocs(vec![table])
}

fn table(space: MemorySpace, elem: ElementKind, values: Vec<i128>) -> AllocDecl {
    AllocDecl::new("t", space, elem, Extent::Static(N)).with_init(values)
}

fn run(k: &KernelFn, idx: &[u32]) -> Result<Vec<u8>, KernelTrap> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let elem = k.allocs[0].elem;
    let ib: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
    // two blocks of N threads: every block sees the same table
    let n = (2 * N) as usize;
    let w = match elem {
        F32 | U32 => 4,
        _ => 1,
    };
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::new(U32, vec![n as i64], vec![1], ib)),
        KernelArg::Buffer(TypedBuffer::new(
            elem,
            vec![n as i64],
            vec![1],
            vec![0; n * w],
        )),
    ];
    run_launch(k, &mut args)?;
    let KernelArg::Buffer(b) = &args[1] else {
        unreachable!()
    };
    Ok(b.bytes.clone())
}

fn refused(k: &KernelFn) -> &'static str {
    match check_kernel(k) {
        Err(KernelError::AllocInit { why, .. }) => why,
        other => panic!("expected an AllocInit refusal, got {other:?}"),
    }
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

const TABLE: [i128; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
const IDX: [u32; 16] = [7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7];

#[test]
fn a_constant_table_is_read_by_every_block_and_a_private_one_by_every_thread() {
    for space in [MemorySpace::Constant, MemorySpace::Private] {
        let k = kernel(table(space, U32, TABLE.to_vec()));
        let got = words(&run(&k, &IDX).expect("runs"));
        let want: Vec<u32> = IDX.iter().map(|&i| (TABLE[i as usize]) as u32).collect();
        assert_eq!(got, want, "{space:?}");
    }
}

#[test]
fn a_float_table_is_its_bit_patterns() {
    // 1.5, -0.0 (the sign survives), 2.5, +inf, and four more
    let bits: Vec<i128> = [1.5f32, -0.0, 2.5, f32::INFINITY, 0.25, -7.0, 1e-3, 3.0]
        .iter()
        .map(|f| i128::from(f.to_bits()))
        .collect();
    let k = kernel(table(MemorySpace::Constant, F32, bits.clone()));
    let got = words(&run(&k, &IDX).expect("runs"));
    let want: Vec<u32> = IDX.iter().map(|&i| bits[i as usize] as u32).collect();
    assert_eq!(got, want);
    assert_eq!(
        got[6],
        (-0.0f32).to_bits(),
        "a negative zero keeps its sign"
    );
}

#[test]
fn the_checker_refuses_each_violation_and_accepts_its_control() {
    let ok = || table(MemorySpace::Constant, U32, TABLE.to_vec());
    assert_eq!(check_kernel(&kernel(ok())), Ok(()));
    // only Constant and Private: a Shared table has no initializer in any target
    let mut shared = ok();
    shared.space = MemorySpace::Shared;
    assert!(refused(&kernel(shared)).contains("Constant or Private"));
    // one value per element
    let short = table(MemorySpace::Constant, U32, TABLE[..7].to_vec());
    assert!(refused(&kernel(short)).contains("one value per element"));
    // each value representable: U8 255 accepted, 256 refused; I8 -128 accepted, -129 refused; a float's bits below 2^32
    let u8s = |v: i128| table(MemorySpace::Constant, U8, vec![v; 8]);
    assert_eq!(check_kernel(&kernel(u8s(255))), Ok(()));
    assert!(refused(&kernel(u8s(256))).contains("not representable"));
    let i8s = |v: i128| table(MemorySpace::Constant, I8, vec![v; 8]);
    assert_eq!(check_kernel(&kernel(i8s(-128))), Ok(()));
    assert!(refused(&kernel(i8s(-129))).contains("not representable"));
    let f32s = |v: i128| table(MemorySpace::Constant, F32, vec![v; 8]);
    assert_eq!(check_kernel(&kernel(f32s(0xffff_ffff))), Ok(()));
    assert!(refused(&kernel(f32s(1 << 32))).contains("not representable"));
    assert!(refused(&kernel(f32s(-1))).contains("not representable"));
    // a layout beside a table is not defined
    let laid = ok().with_layout("(2,4):(4,1)".parse::<AllocLayout>().unwrap());
    let mut priv_laid = laid;
    priv_laid.space = MemorySpace::Private;
    assert!(refused(&kernel(priv_laid)).contains("layout"));
}

#[test]
fn a_store_into_a_constant_table_is_still_refused() {
    let mut k = kernel(table(MemorySpace::Constant, U32, TABLE.to_vec()));
    k.body.0.push(Stmt::Store {
        place: Place::Alloc {
            alloc: 0,
            index: ScalarExpr::ConstInt(0, U32),
        },
        value: ScalarExpr::ConstInt(1, U32),
    });
    assert!(matches!(
        check_kernel(&k),
        Err(KernelError::WriteToConstant { .. })
    ));
}

#[test]
fn a_constant_allocation_without_values_is_still_unreadable_and_serde_omits_none() {
    let bare = AllocDecl::new("t", MemorySpace::Constant, U32, Extent::Static(N));
    let k = kernel(bare);
    assert_eq!(check_kernel(&k), Ok(()));
    assert!(matches!(run(&k, &IDX), Err(KernelTrap::Unsupported(_))));
    let text = serde_json::to_string(&k).unwrap();
    assert!(
        !text.contains("\"init\":"),
        "a kernel without a table keeps its text: {text}"
    );
    let with = kernel(table(MemorySpace::Constant, U32, TABLE.to_vec()));
    let text = serde_json::to_string(&with).unwrap();
    assert!(
        text.contains("\"init\":[10,20,30,40,50,60,70,80]"),
        "{text}"
    );
    assert_eq!(serde_json::from_str::<KernelFn>(&text).unwrap(), with);
}
