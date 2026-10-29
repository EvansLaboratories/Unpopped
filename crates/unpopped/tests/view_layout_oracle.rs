// SPDX-License-Identifier: MIT OR Apache-2.0
//! `view_layout` against the oracle's real addressing.
//!
//! The differential: run a one-input copy op through `oracle::evaluate` with
//! a `View` on its input (the oracle reads through `read_strided` +
//! `input_perm`, a separate code path from `view_to_layout`), and require every
//! output element to be the producer element at `view_to_layout(..).call(i)`.
//! Producer data is the distinct values `1, 2, 3, ...`, so a wrong address reads
//! a wrong value. Then the round trip `layout_to_view(view_to_layout(v)) == v`
//! and the cases `layout_to_view` answers with `View::Layout` or `None`.

use unpopped::ir::{OpDef, View, input};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::build_plan;
use unpopped::view_layout::{ViewLayoutError, layout_to_view, view_to_layout};
use unpopped_vocab::layout_algebra::{IntTuple, Layout, LayoutError};
use unpopped_vocab::{ArchSku, AxisMask, ElementKind, OpCategory, OperandDesc, structure_key};

const F32: ElementKind = ElementKind::F32;

fn row_major(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

/// Run `copy(view(x))` through the oracle: returns the output in row-major
/// iteration order. `strides` are the producer's physical strides.
fn oracle_copy(view: &View, iter_shape: &[i64], strides: &[i64], data: &[f32]) -> Vec<f32> {
    let rank = iter_shape.len();
    let a = OperandDesc::new(rank, iter_shape, strides, F32, 256);
    let o = OperandDesc::new(rank, iter_shape, &row_major(iter_shape), F32, 256);
    let operands = vec![a, o];
    let key = structure_key(OpCategory::UnaryElementwise, &operands, ArchSku::Sm89);
    let op = OpDef::elementwise("copy", 1, &[F32], input(0)).with_views(vec![view.clone()]);
    let plan = build_plan(&op, &key);
    let buf = TypedBuffer::from_f32(&[data.len() as i64], data);
    let out = evaluate(&plan, &operands, &[buf], &[]).remove(0);
    (0..out.len())
        .map(|i| f32::from_bits(out.bits_at(i) as u32))
        .collect()
}

fn iota(n: i64) -> Vec<f32> {
    (1..=n).map(|v| v as f32).collect()
}

/// The differential for one case; returns the layout so callers can reuse it.
fn check(view: &View, iter_shape: &[i64], strides: &[i64], data_len: i64) -> Layout {
    let data = iota(data_len);
    let layout = view_to_layout(view, iter_shape, strides).expect("view_to_layout");
    let got = oracle_copy(view, iter_shape, strides, &data);
    let n: i64 = iter_shape.iter().product();
    assert_eq!(got.len() as i64, n, "{view:?}: output length");
    for i in 0..n {
        let want = data[layout.call(i).unwrap() as usize];
        assert_eq!(
            got[i as usize], want,
            "{view:?} over {iter_shape:?} strides {strides:?}: element {i} (layout {layout})"
        );
    }
    layout
}

fn perms(n: u8) -> Vec<Vec<u8>> {
    fn go(cur: &mut Vec<u8>, left: &mut Vec<u8>, out: &mut Vec<Vec<u8>>) {
        if left.is_empty() {
            out.push(cur.clone());
            return;
        }
        for k in 0..left.len() {
            let x = left.remove(k);
            cur.push(x);
            go(cur, left, out);
            cur.pop();
            left.insert(k, x);
        }
    }
    let mut out = Vec::new();
    go(&mut Vec::new(), &mut (0..n).collect(), &mut out);
    out
}

#[test]
fn every_permute_of_rank_2_and_3_matches_the_oracle() {
    for producer in [vec![3i64, 5], vec![2, 3, 4]] {
        let strides = row_major(&producer);
        let size: i64 = producer.iter().product();
        for perm in perms(producer.len() as u8) {
            let iter: Vec<i64> = perm.iter().map(|&p| producer[p as usize]).collect();
            let view = View::Permute { perm: perm.clone() };
            check(&view, &iter, &strides, size);
        }
    }
}

#[test]
fn identity_and_same_rank_reshape_match_the_oracle() {
    for producer in [vec![7i64], vec![3, 5], vec![2, 3, 4]] {
        let strides = row_major(&producer);
        let size: i64 = producer.iter().product();
        check(&View::Identity, &producer, &strides, size);
        let r = View::Reshape {
            producer_rank: producer.len() as u8,
        };
        check(&r, &producer, &strides, size);
    }
}

#[test]
fn every_broadcast_mask_of_rank_3_matches_the_oracle() {
    let iter = [2i64, 3, 4];
    for mask in 1u8..7 {
        // Producer stride 0 on masked axes, row-major over the rest.
        let kept: Vec<i64> = (0..3)
            .filter(|&d| mask >> d & 1 == 0)
            .map(|d| iter[d as usize])
            .collect();
        let kept_strides = row_major(&kept);
        let mut it = kept_strides.iter();
        let strides: Vec<i64> = (0..3)
            .map(|d| {
                if mask >> d & 1 == 1 {
                    0
                } else {
                    *it.next().unwrap()
                }
            })
            .collect();
        let view = View::Broadcast {
            bcast: AxisMask(mask),
        };
        check(&view, &iter, &strides, kept.iter().product());
    }
}

#[test]
fn recognition_round_trips_through_the_layout() {
    for producer in [vec![3i64, 5], vec![2, 3, 4]] {
        let strides = row_major(&producer);
        for perm in perms(producer.len() as u8) {
            let iter: Vec<i64> = perm.iter().map(|&p| producer[p as usize]).collect();
            let view = View::Permute { perm: perm.clone() };
            let l = view_to_layout(&view, &iter, &strides).unwrap();
            let back = layout_to_view(&l, &strides);
            // The identity permutation reads exactly like Identity.
            let want = if perm.iter().copied().eq(0..perm.len() as u8) {
                View::Identity
            } else {
                view.clone()
            };
            assert_eq!(back, Some(want), "{view:?} over {producer:?} (layout {l})");
        }
    }
    let iter = [2i64, 3, 4];
    for mask in 1u8..7 {
        let strides: Vec<i64> = [12, 4, 1]
            .iter()
            .enumerate()
            .map(|(d, &s)| if mask >> d & 1 == 1 { 0 } else { s })
            .collect();
        let view = View::Broadcast {
            bcast: AxisMask(mask),
        };
        let l = view_to_layout(&view, &iter, &strides).unwrap();
        assert_eq!(layout_to_view(&l, &strides), Some(view), "mask {mask:#b}");
    }
    let l = view_to_layout(&View::Reshape { producer_rank: 2 }, &[3, 5], &[5, 1]).unwrap();
    assert_eq!(layout_to_view(&l, &[5, 1]), Some(View::Identity));
}

#[test]
fn what_no_producer_relative_variant_can_say_is_the_layout_variant_or_none() {
    let p = [12i64, 4, 1];
    let leaf = |s: &[i64], d: &[i64]| {
        Layout::new(
            IntTuple::ints(s.iter().copied()),
            IntTuple::ints(d.iter().copied()),
        )
        .unwrap()
    };
    let lay = |shape: &[i64], stride: &[i64]| View::Layout {
        shape: shape.to_vec(),
        stride: stride.to_vec(),
    };
    // Never a nearest Identity/Permute: the Layout variant says exactly the layout
    // (iteration-axis order, so the reverse of the layout's modes).
    // A stride the producer does not have.
    assert_eq!(
        layout_to_view(&leaf(&[4, 3, 2], &[1, 4, 7]), &p),
        Some(lay(&[2, 3, 4], &[7, 4, 1]))
    );
    // A rank different from the producer's.
    assert_eq!(
        layout_to_view(&leaf(&[4, 3], &[1, 4]), &p),
        Some(lay(&[3, 4], &[4, 1]))
    );
    // A permuted-and-broadcast operand (Permute and Broadcast stay orthogonal).
    assert_eq!(
        layout_to_view(&leaf(&[4, 3, 2], &[4, 1, 0]), &p),
        Some(lay(&[2, 3, 4], &[0, 1, 4]))
    );
    // Ambiguous: two producer axes share a stride, so two permutations fit.
    assert_eq!(
        layout_to_view(&leaf(&[3, 4, 5], &[1, 4, 1]), &[4, 1, 1]),
        Some(lay(&[5, 4, 3], &[1, 4, 1]))
    );
    // A nested layout is not one mode per axis: still None.
    let nested = Layout::new(
        IntTuple::tuple([IntTuple::ints([2, 2]), IntTuple::Int(3)]),
        IntTuple::tuple([IntTuple::ints([1, 2]), IntTuple::Int(4)]),
    )
    .unwrap();
    assert_eq!(layout_to_view(&nested, &[4, 1]), None);
}

#[test]
fn view_to_layout_declines_what_it_cannot_address() {
    let strides = [12, 4, 1];
    let shape = [2i64, 3, 4];
    assert_eq!(
        view_to_layout(
            &View::Permute {
                perm: vec![0, 0, 1]
            },
            &shape,
            &strides
        ),
        Err(ViewLayoutError::NotAPermutation)
    );
    assert_eq!(
        view_to_layout(&View::Permute { perm: vec![1, 0] }, &shape, &strides),
        Err(ViewLayoutError::NotAPermutation)
    );
    assert_eq!(
        view_to_layout(
            &View::Broadcast {
                bcast: AxisMask(0b010)
            },
            &shape,
            &strides
        ),
        Err(ViewLayoutError::BroadcastDisagrees { axis: 1 })
    );
    assert_eq!(
        view_to_layout(&View::Reshape { producer_rank: 2 }, &shape, &strides),
        Err(ViewLayoutError::RankChangingReshape)
    );
    assert_eq!(
        view_to_layout(&View::Identity, &shape, &[1, 2]),
        Err(ViewLayoutError::RankMismatch)
    );
    let lay = |s: &[i64], d: &[i64]| View::Layout {
        shape: s.to_vec(),
        stride: d.to_vec(),
    };
    assert_eq!(
        view_to_layout(&lay(&[2, 3, 5], &[12, 4, 1]), &shape, &strides),
        Err(ViewLayoutError::ExtentMismatch)
    );
    assert_eq!(
        view_to_layout(&lay(&[2, 3], &[4, 1]), &shape, &strides),
        Err(ViewLayoutError::RankMismatch)
    );
    assert_eq!(
        view_to_layout(&View::Identity, &shape, &[12, -4, 1]),
        Err(ViewLayoutError::Layout(LayoutError::NegativeStride(-4)))
    );
}

/// The differential for a `View::Layout`: the operand's own strides are dense and
/// irrelevant (the view's strides are the truth); `data_len` covers the largest
/// offset.
fn check_layout_view(shape: &[i64], stride: &[i64], data_len: i64) {
    let view = View::Layout {
        shape: shape.to_vec(),
        stride: stride.to_vec(),
    };
    // Operand strides are deliberately NOT the view's: a read that consulted them
    // would see the wrong elements.
    let dense = row_major(shape);
    let data = iota(data_len);
    let layout = view_to_layout(&view, shape, &dense).expect("view_to_layout");
    let got = oracle_copy(&view, shape, &dense, &data);
    let n: i64 = shape.iter().product();
    for i in 0..n {
        assert_eq!(
            got[i as usize],
            data[layout.call(i).unwrap() as usize],
            "{view:?}: element {i} (layout {layout})"
        );
    }
}

#[test]
fn a_layout_view_matches_the_oracle_on_strided_and_reshaped_reads() {
    // Every other column of a dense 3x4 producer.
    check_layout_view(&[3, 2], &[4, 2], 12);
    // A dense rank-1 producer of 6 read as 2x3 (a reshape the four variants cannot say).
    check_layout_view(&[2, 3], &[3, 1], 6);
    // A transposed read with a gap.
    check_layout_view(&[3, 2], &[1, 4], 12);
    // A broadcast written as stride 0.
    check_layout_view(&[2, 3], &[0, 1], 3);
    // Rank 1, stride 3.
    check_layout_view(&[4], &[3], 10);
    // Rank 3 mixing all of the above.
    check_layout_view(&[2, 3, 2], &[12, 2, 0], 24);
}

#[test]
fn a_layout_view_round_trips_through_the_layout() {
    let dense = [12i64, 4, 1];
    let view = View::Layout {
        shape: vec![2, 3, 4],
        stride: vec![7, 4, 1],
    };
    let l = view_to_layout(&view, &[2, 3, 4], &dense).unwrap();
    assert_eq!(layout_to_view(&l, &dense), Some(view));
}

// ---- the plan/oracle gates a Layout view must hit

fn plan_with(view: View, in_strides: &[i64], shape: &[i64]) -> Vec<f32> {
    oracle_copy(&view, shape, in_strides, &iota(64))
}

#[test]
#[should_panic(expected = "Layout view AND a broadcast mask")]
fn a_layout_view_on_a_key_broadcast_operand_is_refused() {
    let v = View::Layout {
        shape: vec![2, 3],
        stride: vec![0, 1],
    };
    plan_with(v, &[0, 1], &[2, 3]);
}

#[test]
#[should_panic(expected = "is invalid for iteration rank")]
fn a_layout_view_of_the_wrong_rank_is_refused() {
    let v = View::Layout {
        shape: vec![6],
        stride: vec![1],
    };
    plan_with(v, &[3, 1], &[2, 3]);
}

#[test]
#[should_panic(expected = "is invalid for iteration rank")]
fn a_layout_view_with_a_negative_stride_is_refused() {
    let v = View::Layout {
        shape: vec![2, 3],
        stride: vec![3, -1],
    };
    plan_with(v, &[3, 1], &[2, 3]);
}

#[test]
#[should_panic(expected = "extents differ from the iteration extents")]
fn a_layout_view_whose_extents_differ_from_the_iteration_is_caught_by_the_oracle() {
    let v = View::Layout {
        shape: vec![3, 2],
        stride: vec![2, 1],
    };
    plan_with(v, &[3, 1], &[2, 3]);
}

#[test]
fn a_layout_view_counts_as_address_affecting() {
    use unpopped::plan::view_is_addressing;
    let v = View::Layout {
        shape: vec![2, 3],
        stride: vec![3, 1],
    };
    assert!(view_is_addressing(&v));
    // Controls: the same predicate is false for the non-addressing variants.
    assert!(!view_is_addressing(&View::Identity));
    assert!(!view_is_addressing(&View::Reshape { producer_rank: 2 }));
}
