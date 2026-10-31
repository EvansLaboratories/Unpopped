// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The plan-level oracle takes a shift count modulo the width, as the kernel interpreter does**
//! (`docs/int-div-shift-semantics.md` rule 3; openclownfish#26). `oracle::evaluate` used to answer
//! `0` (or the sign fill) for a count outside `0..width`, so a backend that adopted the ruled guards
//! disagreed with the referee on every out-of-range count. The expectation is the host's native
//! `wrapping_shl`/`wrapping_shr`, which mask the count by the width exactly as the rule says.

use unpopped::ir::{BinaryOp, Expr, OpDef, input};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, try_structure_key};

const COUNTS: [i128; 8] = [0, 1, 31, 32, 33, 63, 64, 255];
const VALUES: [i128; 6] = [1, 3, 0x1234_5678, -1, -8, 0x7fff_ffff];

/// The ruled result, from the host's native integer types.
fn native(t: ElementKind, shl: bool, a: i128, b: i128) -> i128 {
    let n = b as u32;
    match (t, shl) {
        (ElementKind::I32, true) => i128::from((a as i32).wrapping_shl(n)),
        (ElementKind::I32, false) => i128::from((a as i32).wrapping_shr(n)),
        (ElementKind::U32, true) => i128::from((a as u32).wrapping_shl(n)),
        (ElementKind::U32, false) => i128::from((a as u32).wrapping_shr(n)),
        (ElementKind::I64, true) => i128::from((a as i64).wrapping_shl(n)),
        (ElementKind::I64, false) => i128::from((a as i64).wrapping_shr(n)),
        (ElementKind::U64, true) => i128::from((a as u64).wrapping_shl(n)),
        (ElementKind::U64, false) => i128::from((a as u64).wrapping_shr(n)),
        _ => unreachable!("an integer compute dtype"),
    }
}

fn bytes(t: ElementKind, xs: &[i128]) -> Vec<u8> {
    xs.iter()
        .flat_map(|&x| match t {
            ElementKind::I32 | ElementKind::U32 => (x as u32).to_le_bytes().to_vec(),
            _ => (x as u64).to_le_bytes().to_vec(),
        })
        .collect()
}

fn read(t: ElementKind, b: &[u8]) -> Vec<i128> {
    match t {
        ElementKind::I32 => b
            .chunks(4)
            .map(|c| i128::from(i32::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        ElementKind::U32 => b
            .chunks(4)
            .map(|c| i128::from(u32::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        ElementKind::I64 => b
            .chunks(8)
            .map(|c| i128::from(i64::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        _ => b
            .chunks(8)
            .map(|c| i128::from(u64::from_le_bytes(c.try_into().unwrap())))
            .collect(),
    }
}

/// Run `a <op> b` through `oracle::evaluate` for every (value, count) pair.
fn run(t: ElementKind, shl: bool) -> (Vec<i128>, Vec<(i128, i128)>) {
    let pairs: Vec<(i128, i128)> = VALUES
        .iter()
        .flat_map(|&a| COUNTS.iter().map(move |&b| (a, b)))
        .collect();
    let n = pairs.len() as i64;
    let a: Vec<i128> = pairs.iter().map(|p| p.0).collect();
    let b: Vec<i128> = pairs.iter().map(|p| p.1).collect();
    let d = OperandDesc::new(1, &[n], &[1], t, 256);
    let descs = [d, d, d];
    let key =
        try_structure_key(OpCategory::BinaryElementwise, &descs, ArchSku::Sm89).expect("a key");
    let which = if shl { BinaryOp::Shl } else { BinaryOp::Shr };
    let body = input(0).binary(which, input(1));
    let op = OpDef::elementwise("sh", 2, &[t], Expr(body.0));
    let plan = try_build_plan(&op, &key).expect("the shift plans");
    let ins = [
        TypedBuffer::new(t, vec![n], vec![1], bytes(t, &a)),
        TypedBuffer::new(t, vec![n], vec![1], bytes(t, &b)),
    ];
    let out = evaluate(&plan, &descs, &ins, &[]).remove(0);
    (read(t, &out.bytes), pairs)
}

#[test]
fn out_of_range_counts_are_taken_modulo_the_width() {
    let mut out_of_range = 0;
    for t in [
        ElementKind::I32,
        ElementKind::U32,
        ElementKind::I64,
        ElementKind::U64,
    ] {
        for shl in [true, false] {
            let (got, pairs) = run(t, shl);
            for (g, (a, b)) in got.iter().zip(&pairs) {
                let width = if matches!(t, ElementKind::I32 | ElementKind::U32) {
                    32
                } else {
                    64
                };
                out_of_range += i32::from(*b >= width);
                assert_eq!(*g, native(t, shl, *a, *b), "{t:?} shl={shl} {a} by {b}");
            }
        }
    }
    assert!(
        out_of_range > 0,
        "the vectors reach counts at or above the width"
    );
}
