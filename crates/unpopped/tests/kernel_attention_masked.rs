// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Masked attention at the kernel layer** (ROADMAP U10 A3, `docs/options-u10-attention.md`): A2's online softmax with a
//! visibility test per key (`j + lag <= row`: causal for `lag = 0`) and an optional ALiBi bias `slope * (row - j)` subtracted
//! from a visible score. A row that sees no key yields zeros, not NaN (Q4 of the options note). Each variant must match a host
//! loop in the same `f32` order.

use unpopped::kernel::check_kernel;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[expect(dead_code, reason = "the shared kernels; this file uses its own")]
#[path = "support/attention.rs"]
mod attention;
use attention::{DV, F32, LQ, attention_data, masked_attention, masked_reference};

fn f32_buf(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn run(lag: usize, slope: Option<f64>) -> Vec<f32> {
    let k = masked_attention(lag, slope);
    assert_eq!(check_kernel(&k), Ok(()));
    let (q, kk, v) = attention_data();
    let mut args = [
        f32_buf(&q),
        f32_buf(&kk),
        f32_buf(&v),
        f32_buf(&[0.0; LQ * DV]),
    ];
    run_launch(&k, &mut args).expect("the oracle runs it");
    let KernelArg::Buffer(b) = &args[3] else {
        unreachable!("a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4 * w.abs().max(1.0),
            "{what}[{i}]: got {g}, want {w}"
        );
    }
}

#[test]
fn causal_alibi_and_fully_masked_rows_match_the_host_loop() {
    for (lag, slope) in [
        (0, None),
        (0, Some(0.75)),
        (1, None),
        (1, Some(0.75)),
        (2, Some(0.5)),
    ] {
        close(
            &run(lag, slope),
            &masked_reference(lag, slope),
            &format!("lag {lag} slope {slope:?}"),
        );
    }
}

#[test]
fn a_row_that_sees_no_key_is_zeros_and_the_variants_differ() {
    let hidden = run(1, None);
    assert!(
        hidden[..DV].iter().all(|&x| x == 0.0),
        "row 0 sees nothing: {hidden:?}"
    );
    assert!(hidden.iter().all(|x| x.is_finite()));
    assert!(
        hidden[DV..].iter().any(|&x| x != 0.0),
        "later rows see keys"
    );
    let causal = run(0, None);
    assert_ne!(causal, hidden, "the diagonal key matters");
    assert_ne!(causal, run(0, Some(0.75)), "the ALiBi bias matters");
    // Row 0 of the causal kernel sees exactly key 0, so its output is V[0].
    let (_, _, v) = attention_data();
    assert_eq!(&causal[..DV], &v[..DV]);
}
