// SPDX-License-Identifier: MIT OR Apache-2.0
//! **KISS-OPS-6.13-0010/-0011: a window with no in-bounds tap is declined, not answered with the
//! monoid identity** (CireSnave board 345 "B"). Pins the helper, the oracle's window evaluation and
//! the extent-aware lift, with the KISS negative control `n=1, k=2, d=2, s=1, p=1`, which passes the
//! plan gate's `2*pad <= span` assert and has one window with no real tap.

use unpopped::ir::{OpDef, ReduceOp};
use unpopped::kernel_lift::{LiftDecline, lift_window_opdef, lift_window_opdef_for_extents};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped::window_taps::{check_window_taps, window_out_extent};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, try_structure_key};

const F32: ElementKind = ElementKind::F32;

/// `(k_in, size, stride, dilation, pad)` of the KISS control.
const CONTROL: (i64, u8, u8, u8, u8) = (1, 2, 1, 2, 1);

fn win(size: u8, stride: u8, dilation: u8, pad: u8) -> OpDef {
    OpDef::window_simple(
        "w",
        &[F32],
        ReduceOp::Max,
        1,
        size,
        stride,
        dilation,
        pad,
        pad,
        false,
    )
}

/// Run the oracle on a 1-row input of `k_in` ones; the output extent is the KISS formula's.
fn oracle(k_in: i64, size: u8, stride: u8, dilation: u8, pad: u8) -> Vec<u8> {
    let k_out = window_out_extent(
        k_in,
        size.into(),
        stride.into(),
        dilation.into(),
        pad.into(),
        pad.into(),
    );
    let ins = OperandDesc::new(2, &[1, k_in], &[k_in, 1], F32, 256);
    let outs = OperandDesc::new(2, &[1, k_out], &[k_out, 1], F32, 256);
    let ops = [ins, outs];
    let key = try_structure_key(OpCategory::Pooling, &ops, ArchSku::Sm89).expect("a key");
    let op = win(size, stride, dilation, pad);
    let plan = try_build_plan(&op, &key).expect("plans");
    let bytes: Vec<u8> = (0..k_in).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    let src = TypedBuffer::new(F32, vec![1, k_in], vec![k_in, 1], bytes);
    evaluate(&plan, &ops, std::slice::from_ref(&src), &[])
        .remove(0)
        .bytes
}

#[test]
fn the_kiss_control_is_declined_by_the_helper_and_names_its_window() {
    let (k_in, size, stride, dilation, pad) = CONTROL;
    let span = i64::from(dilation) * (i64::from(size) - 1) + 1;
    assert!(
        2 * i64::from(pad) <= span,
        "the control passes the plan gate's assert"
    );
    let k_out = window_out_extent(
        k_in,
        size.into(),
        stride.into(),
        dilation.into(),
        pad.into(),
        pad.into(),
    );
    assert_eq!(k_out, 1, "KISS extent: floor((1+2-3)/1)+1");
    let err = check_window_taps(
        k_in,
        k_out,
        size.into(),
        stride.into(),
        dilation.into(),
        pad.into(),
    )
    .expect_err("no in-bounds tap");
    assert_eq!((err.output, err.k_in, err.k_out), (0, 1, 1));
    assert!(err.to_string().contains("no in-bounds tap"));
    // The positive control: the same helper accepts a window that has real taps.
    assert_eq!(check_window_taps(5, 5, 3, 1, 1, 1), Ok(()));
}

#[test]
fn the_helper_agrees_with_an_enumeration_and_the_span_assert_is_not_enough() {
    let (mut declined, mut passes_span_but_declined) = (0, 0);
    for k_in in 0..9i64 {
        for size in 1..5u32 {
            for stride in 1..4u32 {
                for dilation in 1..4u32 {
                    for pad in 0..4u32 {
                        let k_out = window_out_extent(k_in, size, stride, dilation, pad, pad);
                        let mut first_bad = None;
                        for o in 0..k_out {
                            let any = (0..i64::from(size)).any(|kk| {
                                let p = o * i64::from(stride) - i64::from(pad)
                                    + kk * i64::from(dilation);
                                p >= 0 && p < k_in
                            });
                            if !any && first_bad.is_none() {
                                first_bad = Some(o);
                            }
                        }
                        let got = check_window_taps(k_in, k_out, size, stride, dilation, pad)
                            .err()
                            .map(|e| e.output);
                        assert_eq!(
                            got, first_bad,
                            "k_in {k_in} size {size} s {stride} d {dilation} p {pad}"
                        );
                        let span = i64::from(dilation) * (i64::from(size) - 1) + 1;
                        if first_bad.is_some() {
                            declined += 1;
                            if 2 * i64::from(pad) <= span {
                                passes_span_but_declined += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(declined > 0, "the sweep reaches declined windows");
    assert!(
        passes_span_but_declined > 0,
        "some geometry passes 2*pad <= span and still has a window with no real tap"
    );
}

#[test]
#[should_panic(expected = "has no in-bounds tap")]
fn the_oracle_declines_the_kiss_control() {
    let (k_in, size, stride, dilation, pad) = CONTROL;
    let _ = oracle(k_in, size, stride, dilation, pad);
}

#[test]
fn the_oracle_still_evaluates_a_window_with_real_taps() {
    // Positive control for the panic test above: the same call shape, a legal window.
    let out = oracle(5, 3, 1, 1, 1);
    assert_eq!(out.len(), 5 * 4);
    assert!(
        out.chunks(4)
            .all(|c| f32::from_le_bytes(c.try_into().unwrap()) == 1.0)
    );
}

#[test]
fn the_lift_for_known_extents_declines_the_control_and_accepts_a_legal_window() {
    let (k_in, size, stride, dilation, pad) = CONTROL;
    let op = win(size, stride, dilation, pad);
    let Err(LiftDecline::WindowWithoutRealTap(e)) =
        lift_window_opdef_for_extents(&op, F32, 2, k_in, 1)
    else {
        panic!("the control must be declined")
    };
    assert_eq!(e.output, 0);
    let legal = win(3, 1, 1, 1);
    assert!(lift_window_opdef_for_extents(&legal, F32, 2, 5, 5).is_ok());
    // The extent-free lift is unchanged: it has no extents to check.
    assert!(lift_window_opdef(&op, F32, 2).is_ok());
}
