// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Attention backward at the kernel layer** (ROADMAP U10 A5, `docs/options-u10-attention.md`): from the saved row
//! log-sum-exp the kernels recompute `P`, form `dS = P * (dP - rowsum(dout * o))` and produce `dQ` (one thread per query row)
//! and `dK`, `dV` (one thread per key row). The host reference is written as whole matrices in `f64`, and is itself checked by
//! finite differences of the forward loss, so a wrong formula cannot hide behind a kernel that copies it.

use unpopped::kernel::check_kernel;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[expect(
    dead_code,
    reason = "the shared forward kernels; this file uses their shapes and data"
)]
#[path = "support/attention.rs"]
mod attention;
#[path = "support/attention_backward.rs"]
mod attention_backward;
use attention::{D, DV, F32, LK, LQ, attention_data};
use attention_backward::{
    backward_dkv, backward_dq, backward_inputs, backward_reference, dout_data, loss_f64,
};

fn f32_buf(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn floats(a: &KernelArg) -> Vec<f32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!("a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn close(got: &[f32], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    let scale = want.iter().fold(1.0f64, |a, w| a.max(w.abs()));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (*g as f64 - w).abs() <= 2e-3 * scale,
            "{what}[{i}]: got {g}, want {w}"
        );
    }
}

fn inputs() -> Vec<KernelArg> {
    backward_inputs().iter().map(|v| f32_buf(v)).collect()
}

#[test]
fn the_dq_kernel_matches_the_whole_matrix_reference() {
    let k = backward_dq();
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = inputs();
    args.push(f32_buf(&[0.0; LQ * D]));
    run_launch(&k, &mut args).expect("the oracle runs the dQ kernel");
    close(&floats(&args[6]), &backward_reference().0, "dQ");
}

#[test]
fn the_dkv_kernel_matches_the_whole_matrix_reference() {
    let k = backward_dkv();
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = inputs();
    args.push(f32_buf(&[0.0; LK * D]));
    args.push(f32_buf(&[0.0; LK * DV]));
    run_launch(&k, &mut args).expect("the oracle runs the dK/dV kernel");
    let (_, dk, dv) = backward_reference();
    close(&floats(&args[6]), &dk, "dK");
    close(&floats(&args[7]), &dv, "dV");
}

#[test]
fn the_reference_is_the_gradient_of_the_forward_loss() {
    // Central differences of L = sum(dout * attention(q, k, v)) in f64 against the analytic reference. The step is small
    // enough for the softmax (scale 6) and large enough for f64 round-off.
    let (q, k, v) = attention_data();
    let f = |x: &[f32]| x.iter().map(|&a| a as f64).collect::<Vec<_>>();
    let (q, k, v, dout) = (f(&q), f(&k), f(&v), f(&dout_data()));
    let (dq, dk, dv) = backward_reference();
    let h = 1e-6;
    let check = |which: usize, grad: &[f64]| {
        let base = [&q, &k, &v][which];
        for i in 0..base.len() {
            let mut hi = [q.clone(), k.clone(), v.clone()];
            let mut lo = hi.clone();
            hi[which][i] += h;
            lo[which][i] -= h;
            let fd = (loss_f64(&hi[0], &hi[1], &hi[2], &dout)
                - loss_f64(&lo[0], &lo[1], &lo[2], &dout))
                / (2.0 * h);
            assert!(
                (fd - grad[i]).abs() <= 1e-5 * grad[i].abs().max(1.0),
                "operand {which} element {i}: finite difference {fd}, analytic {}",
                grad[i]
            );
        }
    };
    check(0, &dq);
    check(1, &dk);
    check(2, &dv);
}

#[test]
fn the_gradients_are_not_degenerate() {
    // A reference of zeros, or one where dQ == dK == 0 for want of a softmax that is not one-hot, would let a kernel that
    // writes nothing pass. Every gradient has a nonzero entry, and the row constant delta is not zero for every row.
    let (dq, dk, dv) = backward_reference();
    for (name, g) in [("dQ", &dq), ("dK", &dk), ("dV", &dv)] {
        assert!(g.iter().any(|x| x.abs() > 1e-3), "{name} is all ~0");
    }
    let [_, _, _, o, dout, _] = backward_inputs();
    let nonzero_delta = (0..LQ)
        .filter(|&i| {
            (0..DV)
                .map(|d| dout[i * DV + d] * o[i * DV + d])
                .sum::<f32>()
                .abs()
                > 1e-3
        })
        .count();
    assert!(nonzero_delta >= 2, "delta matters only if it is not zero");
}
