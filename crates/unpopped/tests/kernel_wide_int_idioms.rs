// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Wide integers in the idioms that need them** (ROADMAP E2, theme 4): the 64x64 -> 128 widening multiply (`mul.wide`),
//! a 128-bit dot-product accumulator over 64-bit operands, and the 128x128 -> 256 widening multiply. Each is a kernel from
//! ordinary ops (`Cast` up, `Mul`/`Add` at the wide type, `Shr` + `Cast` down), compared with an independent host
//! reference (Rust's `u128`/`i128`, and a 64-bit-limb schoolbook for the 256-bit product), and each has a mutant that does
//! the arithmetic one width too small and must change the VALUE. The oracle runs them; no carry-chain lowering is implied.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, I64, I128, U32, U64, U128, U256};

fn cast(x: ScalarExpr, to: ElementKind) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(x),
        CastSpec {
            to,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
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

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

/// `x >> bits` at `x`'s wide type `t` (the count is a `t` constant built by a `Cast`).
fn shr(x: ScalarExpr, bits: i64, t: ElementKind) -> ScalarExpr {
    ScalarExpr::Binary(
        unpopped::ir::BinaryOp::Shr,
        Box::new(x),
        Box::new(cast(ScalarExpr::ConstInt(bits, U64), t)),
    )
}

fn buffer_arg(t: ElementKind, n: usize, bytes: Vec<u8>) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(t, vec![n as i64], vec![1], bytes))
}

fn launch(k: &mut KernelFn, n: u32) {
    k.launch = Some(LaunchShape::new([n, 1, 1], [1, 1, 1]));
}

// ---- 64 x 64 -> 128 ------------------------------------------------------------------------------------------------

/// Buffers 0 `a`, 1 `b` (U64), 2 `hi`, 3 `lo` (U64). `narrow`: multiply at U64 first (the mutant).
fn mul_wide(narrow: bool) -> KernelFn {
    let p = if narrow {
        cast(mul(l(0), l(1)), U128)
    } else {
        mul(cast(l(0), U128), cast(l(1), U128))
    };
    let mut k = KernelFn::new(
        "mul_wide",
        U32,
        vec![
            Param::buffer("a", U64, BufferRole::Read, Some(1)),
            Param::buffer("b", U64, BufferRole::Read, Some(1)),
            Param::buffer("hi", U64, BufferRole::Write, Some(1)),
            Param::buffer("lo", U64, BufferRole::Write, Some(1)),
        ],
        vec![
            LocalDecl::scalar(U64),
            LocalDecl::scalar(U64),
            LocalDecl::scalar(U128),
        ],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0, tid()),
            },
            Stmt::Load {
                local: 1,
                place: at(1, tid()),
            },
            Stmt::Let { local: 2, init: p },
            Stmt::Store {
                place: at(2, tid()),
                value: cast(shr(l(2), 64, U128), U64),
            },
            Stmt::Store {
                place: at(3, tid()),
                value: cast(l(2), U64),
            },
        ]),
    );
    launch(&mut k, 6);
    k
}

const EDGE: [u64; 6] = [0, 1, u64::MAX, 1 << 63, 0xFFFF_FFFF, 0x1234_5678_9ABC_DEF0];

/// Includes `u64::MAX * u64::MAX`: the narrow (mutant) `Mul` forms the product in an `i128`, which that pair overflows.
/// **Before the oracle hotfix (`wrapping_mul` in `oracle/kernel_eval.rs`) the mutant run panics in a debug build**
/// ("attempt to multiply with overflow"); after it the run is trap-free. This is the regression vector.
fn mul_wide_data() -> ([u64; 6], [u64; 6]) {
    (
        [EDGE[2], EDGE[3], EDGE[2], EDGE[4], EDGE[5], EDGE[1]],
        [EDGE[2], EDGE[3], EDGE[0], EDGE[4], EDGE[5], EDGE[2]],
    )
}

fn words(a: &KernelArg) -> Vec<u64> {
    let KernelArg::Buffer(b) = a else {
        unreachable!("a buffer")
    };
    b.bytes
        .chunks(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn run_mul_wide(narrow: bool) -> (Vec<u64>, Vec<u64>) {
    let k = mul_wide(narrow);
    assert_eq!(check_kernel(&k), Ok(()));
    let (a, b) = mul_wide_data();
    let bytes = |v: &[u64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let mut args = [
        buffer_arg(U64, 6, bytes(&a)),
        buffer_arg(U64, 6, bytes(&b)),
        buffer_arg(U64, 6, vec![0; 48]),
        buffer_arg(U64, 6, vec![0; 48]),
    ];
    run_launch(&k, &mut args).unwrap();
    (words(&args[2]), words(&args[3]))
}

#[test]
fn the_widening_multiply_matches_rusts_u128() {
    let (a, b) = mul_wide_data();
    let want: Vec<u128> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| u128::from(*x) * u128::from(*y))
        .collect();
    let (hi, lo) = run_mul_wide(false);
    let got: Vec<u128> = hi
        .iter()
        .zip(&lo)
        .map(|(h, w)| (u128::from(*h) << 64) | u128::from(*w))
        .collect();
    assert_eq!(got, want);
    assert!(hi.iter().any(|h| *h != 0), "the data reach the high word");
}

#[test]
fn a_product_taken_at_64_bits_loses_the_high_word() {
    let (hi, lo) = run_mul_wide(false);
    let (nhi, nlo) = run_mul_wide(true);
    assert_ne!(nhi, hi, "the mutant is caught by value in the high word");
    assert_eq!(
        nlo, lo,
        "and the low word is the part a 64-bit multiply keeps"
    );
}

// ---- 128-bit dot-product accumulator ---------------------------------------------------------------------------------

const DOT_K: i64 = 4;

/// Buffers 0 `a`, 1 `b` (I64, `DOT_K` per thread), 2 `out` (I128). `narrow`: accumulate at I64 (the mutant).
fn dot128(narrow: bool) -> KernelFn {
    let idx = add(mul(tid(), ScalarExpr::ConstInt(DOT_K, U32)), l(3));
    let term = mul(cast(l(1), I128), cast(l(2), I128));
    let upd = if narrow {
        cast(add(cast(l(0), I64), cast(term, I64)), I128)
    } else {
        add(l(0), term)
    };
    let mut k = KernelFn::new(
        "dot128",
        U32,
        vec![
            Param::buffer("a", I64, BufferRole::Read, Some(1)),
            Param::buffer("b", I64, BufferRole::Read, Some(1)),
            Param::buffer("out", I128, BufferRole::Write, Some(1)),
        ],
        vec![
            LocalDecl::scalar(I128),
            LocalDecl::scalar(I64),
            LocalDecl::scalar(I64),
            LocalDecl::scalar(U32),
        ],
        Block(vec![
            Stmt::Let {
                local: 0,
                init: cast(ScalarExpr::ConstInt(0, I64), I128),
            },
            Stmt::For {
                unroll: Unroll::None,
                local: 3,
                start: ScalarExpr::ConstInt(0, U32),
                end: ScalarExpr::ConstInt(DOT_K, U32),
                step: ScalarExpr::ConstInt(1, U32),
                body: Block(vec![
                    Stmt::Load {
                        local: 1,
                        place: at(0, idx.clone()),
                    },
                    Stmt::Load {
                        local: 2,
                        place: at(1, idx),
                    },
                    Stmt::Assign {
                        local: 0,
                        value: upd,
                    },
                ]),
            },
            Stmt::Store {
                place: at(2, tid()),
                value: l(0),
            },
        ]),
    );
    launch(&mut k, 2);
    k
}

fn dot_data() -> (Vec<i64>, Vec<i64>) {
    (
        vec![i64::MIN, i64::MAX, i64::MIN, 5, 7, -3, i64::MAX, i64::MIN],
        vec![i64::MIN, i64::MAX, 3, -2, i64::MAX, i64::MIN, 1, -1],
    )
}

fn run_dot(narrow: bool) -> Vec<i128> {
    let k = dot128(narrow);
    assert_eq!(check_kernel(&k), Ok(()));
    let (a, b) = dot_data();
    let bytes = |v: &[i64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let mut args = [
        buffer_arg(I64, 8, bytes(&a)),
        buffer_arg(I64, 8, bytes(&b)),
        buffer_arg(I128, 2, vec![0; 32]),
    ];
    run_launch(&k, &mut args).unwrap();
    let KernelArg::Buffer(o) = &args[2] else {
        unreachable!("a buffer")
    };
    o.bytes
        .chunks(16)
        .map(|c| i128::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn the_128_bit_dot_product_matches_rusts_i128() {
    let (a, b) = dot_data();
    let want: Vec<i128> = (0..2)
        .map(|t| {
            (0..DOT_K as usize)
                .map(|k| i128::from(a[t * 4 + k]) * i128::from(b[t * 4 + k]))
                .sum()
        })
        .collect();
    assert_eq!(run_dot(false), want);
    assert!(want.iter().any(|v| i64::try_from(*v).is_err()), "past I64");
}

#[test]
fn a_dot_product_accumulated_at_64_bits_wraps() {
    assert_ne!(run_dot(true), run_dot(false));
}

// ---- 128 x 128 -> 256 -------------------------------------------------------------------------------------------------

/// Buffers 0 `a`, 1 `b` (U128), 2 `out` (U256). `narrow`: multiply at U128, then widen (the mutant).
fn mul256(narrow: bool) -> KernelFn {
    let value = if narrow {
        cast(mul(l(0), l(1)), U256)
    } else {
        mul(cast(l(0), U256), cast(l(1), U256))
    };
    let mut k = KernelFn::new(
        "mul256",
        U32,
        vec![
            Param::buffer("a", U128, BufferRole::Read, Some(1)),
            Param::buffer("b", U128, BufferRole::Read, Some(1)),
            Param::buffer("out", U256, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(U128), LocalDecl::scalar(U128)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0, tid()),
            },
            Stmt::Load {
                local: 1,
                place: at(1, tid()),
            },
            Stmt::Store {
                place: at(2, tid()),
                value,
            },
        ]),
    );
    launch(&mut k, 4);
    k
}

/// The 256-bit product of two `u128`s as eight 32-bit limbs by schoolbook (independent of any wide type).
fn schoolbook(a: u128, b: u128) -> [u8; 32] {
    let limbs = |x: u128| [0, 1, 2, 3].map(|i| ((x >> (32 * i)) & 0xFFFF_FFFF) as u64);
    let (la, lb) = (limbs(a), limbs(b));
    let mut r = [0u64; 9];
    for i in 0..4 {
        let mut carry = 0u64;
        for j in 0..4 {
            let t = r[i + j] + la[i] * lb[j] + carry;
            r[i + j] = t & 0xFFFF_FFFF;
            carry = t >> 32;
        }
        r[i + 4] += carry;
    }
    let mut out = [0u8; 32];
    for (i, w) in r.iter().take(8).enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&(*w as u32).to_le_bytes());
    }
    out
}

fn run_mul256(narrow: bool) -> Vec<[u8; 32]> {
    let k = mul256(narrow);
    assert_eq!(check_kernel(&k), Ok(()));
    let (a, b) = mul256_data();
    let bytes = |v: &[u128]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let mut args = [
        buffer_arg(U128, 4, bytes(&a)),
        buffer_arg(U128, 4, bytes(&b)),
        buffer_arg(U256, 4, vec![0; 128]),
    ];
    run_launch(&k, &mut args).unwrap();
    let KernelArg::Buffer(o) = &args[2] else {
        unreachable!("a buffer")
    };
    o.bytes
        .chunks(32)
        .map(|c| <[u8; 32]>::try_from(c).unwrap())
        .collect()
}

fn mul256_data() -> (Vec<u128>, Vec<u128>) {
    (
        vec![
            u128::MAX,
            1 << 127,
            u128::MAX,
            0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210,
        ],
        vec![
            u128::MAX,
            1 << 127,
            0,
            0xFFFF_0000_FFFF_0000_FFFF_0000_FFFF_0001,
        ],
    )
}

#[test]
fn the_256_bit_widening_product_matches_the_limb_schoolbook() {
    let (a, b) = mul256_data();
    let want: Vec<[u8; 32]> = a.iter().zip(&b).map(|(x, y)| schoolbook(*x, *y)).collect();
    assert_eq!(run_mul256(false), want);
    // the reference itself, by hand: (2^128-1)^2 = 2^256 - 2^129 + 1, high half 2^128 - 2, low half 1
    let mut high = [0xFF; 16];
    high[0] = 0xFE;
    assert_eq!(want[0][16..], high);
    assert_eq!(want[0][..16], 1u128.to_le_bytes());
}

#[test]
fn a_product_taken_at_128_bits_loses_the_high_half() {
    assert_ne!(run_mul256(true), run_mul256(false));
}
