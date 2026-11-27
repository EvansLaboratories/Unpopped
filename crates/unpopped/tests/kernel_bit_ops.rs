// SPDX-License-Identifier: MIT OR Apache-2.0
//! `UnaryOp::Popc` / `Clz` / `Brev`: integer bit operations of the kernel layer, defined at the operand's own width (the
//! `w`-bit two's-complement pattern) and returning the operand's type. Sources: PTX ISA `popc`, `clz`, `brev`; CUDA `__popc`,
//! `__clz`, `__brev` (`__clz(0) == 32`).
//!
//! The expected values come from an independent per-bit loop over the pattern (never `count_ones`, `leading_zeros` or
//! `reverse_bits`, which the oracle uses), at every integer width, including `0` and the all-ones pattern.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{ScalarExpr, UnaryOp};
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped::types::{TypeError, infer_ty};
use unpopped_vocab::ElementKind;

const INTS: [(ElementKind, u32, bool); 8] = [
    (ElementKind::I8, 8, true),
    (ElementKind::U8, 8, false),
    (ElementKind::I16, 16, true),
    (ElementKind::U16, 16, false),
    (ElementKind::I32, 32, true),
    (ElementKind::U32, 32, false),
    (ElementKind::I64, 64, true),
    (ElementKind::U64, 64, false),
];

fn at_tid(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    }
}

/// `out[tid] = op(in[tid])` at element type `t`, one block of `n` threads.
fn kernel(op: UnaryOp, t: ElementKind, n: u32) -> KernelFn {
    let mut k = KernelFn::new(
        "bitop",
        ElementKind::I32,
        vec![
            Param::buffer("in", t, BufferRole::Read, Some(1)),
            Param::buffer("out", t, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(t)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at_tid(0),
            },
            Stmt::Store {
                place: at_tid(1),
                value: ScalarExpr::Unary(op, Box::new(ScalarExpr::Local(0))),
            },
        ]),
    );
    k.launch = Some(LaunchShape::new([n, 1, 1], [1, 1, 1]));
    k
}

/// The width-`w` pattern of `v` as a bit vector, least significant first (independent of the std bit intrinsics).
fn bits(v: i128, w: u32) -> Vec<bool> {
    (0..w).map(|i| (v >> i) & 1 == 1).collect()
}

fn from_bits(b: &[bool], w: u32, signed: bool) -> i128 {
    let mut v: i128 = 0;
    for (i, &x) in b.iter().enumerate() {
        if x {
            v += 1i128 << i;
        }
    }
    if signed && b[(w - 1) as usize] {
        v -= 1i128 << w;
    }
    v
}

fn reference(op: UnaryOp, v: i128, w: u32, signed: bool) -> i128 {
    let b = bits(v, w);
    match op {
        UnaryOp::Popc => b.iter().filter(|x| **x).count() as i128,
        UnaryOp::Clz => b.iter().rev().take_while(|x| !**x).count() as i128,
        UnaryOp::Brev => {
            let r: Vec<bool> = b.iter().rev().copied().collect();
            from_bits(&r, w, signed)
        }
        _ => unreachable!(),
    }
}

fn inputs(w: u32, signed: bool) -> Vec<i128> {
    let all_ones = if signed { -1 } else { (1i128 << w) - 1 };
    let msb = if signed {
        -(1i128 << (w - 1))
    } else {
        1i128 << (w - 1)
    };
    let max = if signed {
        (1i128 << (w - 1)) - 1
    } else {
        (1i128 << w) - 1
    };
    // 0, 1, the all-ones pattern, only the top bit, the type's maximum, and an asymmetric pattern (0b...0110_1001)
    vec![0, 1, all_ones, msb, max, 0x69]
}

fn run(op: UnaryOp, t: ElementKind, w: u32, vals: &[i128]) -> Vec<i128> {
    let bytes = (w / 8) as usize;
    let data: Vec<u8> = vals
        .iter()
        .flat_map(|v| v.to_le_bytes()[..bytes].to_vec())
        .collect();
    let n = vals.len() as i64;
    let k = kernel(op, t, vals.len() as u32);
    assert_eq!(check_kernel(&k), Ok(()), "{op:?} {t:?} is well formed");
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::new(t, vec![n], vec![1], data)),
        KernelArg::Buffer(TypedBuffer::new(
            t,
            vec![n],
            vec![1],
            vec![0u8; bytes * vals.len()],
        )),
    ];
    run_launch(&k, &mut args).unwrap_or_else(|e| panic!("{op:?} {t:?}: {e:?}"));
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    out.bytes
        .chunks(bytes)
        .map(|c| {
            let mut full = [0u8; 16];
            full[..bytes].copy_from_slice(c);
            let raw = u128::from_le_bytes(full);
            let (_, signed) = INTS.iter().find(|i| i.0 == t).map(|i| (i.1, i.2)).unwrap();
            if signed && (raw >> (w - 1)) & 1 == 1 {
                raw as i128 - (1i128 << w)
            } else {
                raw as i128
            }
        })
        .collect()
}

#[test]
fn popc_clz_brev_match_the_per_bit_reference_at_every_width() {
    for (t, w, signed) in INTS {
        let vals = inputs(w, signed);
        for op in [UnaryOp::Popc, UnaryOp::Clz, UnaryOp::Brev] {
            let want: Vec<i128> = vals.iter().map(|v| reference(op, *v, w, signed)).collect();
            assert_eq!(run(op, t, w, &vals), want, "{op:?} at {t:?}");
        }
    }
}

#[test]
fn the_defining_corner_values_are_pinned_by_hand() {
    // not through the reference: the numbers the PTX/CUDA text gives
    assert_eq!(
        run(UnaryOp::Clz, ElementKind::U32, 32, &[0, 1]),
        vec![32, 31]
    );
    assert_eq!(run(UnaryOp::Clz, ElementKind::U8, 8, &[0, 1]), vec![8, 7]);
    assert_eq!(run(UnaryOp::Popc, ElementKind::I8, 8, &[-1]), vec![8]); // 8, not 64: the pattern is the type's width
    assert_eq!(run(UnaryOp::Clz, ElementKind::I32, 32, &[-1]), vec![0]);
    assert_eq!(run(UnaryOp::Brev, ElementKind::U8, 8, &[1]), vec![128]);
    assert_eq!(run(UnaryOp::Brev, ElementKind::I8, 8, &[1]), vec![-128]);
    assert_eq!(
        run(UnaryOp::Brev, ElementKind::U32, 32, &[1]),
        vec![0x8000_0000]
    );
}

#[test]
fn a_bit_op_is_refused_on_a_float_and_in_a_plan_body_with_the_integer_control() {
    // a float operand in a kernel body
    let mut k = kernel(UnaryOp::Popc, ElementKind::F32, 4);
    k.compute = ElementKind::F32;
    assert!(matches!(
        check_kernel(&k),
        Err(KernelError::Type(TypeError::IntegerOnlyOperand { .. }))
    ));
    // the integer control is accepted
    assert_eq!(
        check_kernel(&kernel(UnaryOp::Popc, ElementKind::U32, 4)),
        Ok(())
    );
    // a plan-body typing refuses it even at an integer operand type
    let e = ScalarExpr::Unary(
        UnaryOp::Popc,
        Box::new(ScalarExpr::ConstInt(5, ElementKind::U32)),
    );
    assert!(matches!(
        infer_ty(&e, ElementKind::F32),
        Err(TypeError::IntegerOnlyOperand { .. })
    ));
}
