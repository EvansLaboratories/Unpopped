// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP E19: `oracle::evaluate` of an integer reduction at a dtype narrower than `I32`
//! (`I8`, `U8`, `I16`, `U16`). It used to panic ("integer reduction dtype I8 out of scope");
//! openclownfish X2 PR1 (#33) lifts narrow-integer elementwise ops and scans but could not check a
//! narrow reduction against the oracle.
//!
//! The rule is the IR's own: arithmetic at an integer dtype wraps at the dtype's width (the
//! elementwise `Add`/`Mul` rule, `docs/ir-kernel-layer.md`), so a `Sum` is the wrapping fold of the
//! elements, `Prod` the wrapping product, `Max`/`Min` exact. The references below are the host's own
//! `wrapping_add`, `wrapping_mul`, `max`, `min` over the native type, which share no code with the
//! oracle.

use unpopped::ir::{OpDef, ReduceOp, input};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

/// Run `op` over one row of `vals` (little-endian, `width` bytes each) and return the single result.
fn reduce(dtype: ElementKind, op: ReduceOp, vals: &[i128], width: usize) -> i128 {
    let n = vals.len() as i64;
    let def = OpDef::reduction("r", 1, &[dtype], input(0), op);
    let bytes: Vec<u8> = vals
        .iter()
        .flat_map(|v| v.to_le_bytes()[..width].to_vec())
        .collect();
    let inp = TypedBuffer::new(dtype, vec![1, n], vec![n, 1], bytes);
    let d_in = OperandDesc::new(2, &[1, n], &[n, 1], dtype, 256);
    let d_out = OperandDesc::new(1, &[1], &[1], dtype, 256);
    let descs = [d_in, d_out];
    let key = structure_key(OpCategory::Reduction, &descs, ArchSku::Sm89);
    let plan = try_build_plan(&def, &key).expect("a narrow integer reduction plans");
    let out = evaluate(&plan, &descs, &[inp], &[]).remove(0).bytes;
    let mut buf = [0u8; 16];
    buf[..width].copy_from_slice(&out[..width]);
    let raw = i128::from_le_bytes(buf);
    let signed = matches!(dtype, ElementKind::I8 | ElementKind::I16);
    let bits = (width * 8) as u32;
    if signed && raw >> (bits - 1) & 1 == 1 {
        raw - (1i128 << bits)
    } else {
        raw
    }
}

macro_rules! check {
    ($dtype:expr, $t:ty, $vals:expr) => {{
        let vals: Vec<$t> = $vals;
        let wide: Vec<i128> = vals.iter().map(|v| *v as i128).collect();
        let w = std::mem::size_of::<$t>();
        let sum = vals.iter().fold(0 as $t, |a, b| a.wrapping_add(*b));
        let prod = vals.iter().fold(1 as $t, |a, b| a.wrapping_mul(*b));
        let (mx, mn) = (*vals.iter().max().unwrap(), *vals.iter().min().unwrap());
        assert_eq!(reduce($dtype, ReduceOp::Sum, &wide, w), sum as i128, "Sum {:?}", $dtype);
        assert_eq!(reduce($dtype, ReduceOp::Prod, &wide, w), prod as i128, "Prod {:?}", $dtype);
        assert_eq!(reduce($dtype, ReduceOp::Max, &wide, w), mx as i128, "Max {:?}", $dtype);
        assert_eq!(reduce($dtype, ReduceOp::Min, &wide, w), mn as i128, "Min {:?}", $dtype);
    }};
}

#[test]
fn signed_narrow_reductions_wrap_at_the_dtype_width() {
    check!(ElementKind::I8, i8, vec![100, 100, 100, -128, 127, -1, 5]);
    check!(ElementKind::I16, i16, vec![30000, 30000, -32768, 32767, -1, 7]);
}

#[test]
fn unsigned_narrow_reductions_wrap_at_the_dtype_width() {
    check!(ElementKind::U8, u8, vec![200, 100, 255, 0, 17, 3]);
    check!(ElementKind::U16, u16, vec![60000, 60000, 65535, 0, 9, 3]);
}
