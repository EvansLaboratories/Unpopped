// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Flash-style attention at the kernel layer** (ROADMAP U10 A2, `docs/options-u10-attention.md`): the online softmax keeps a
//! running max `m`, sum `l` and output accumulator per row and column, rescaling by `exp(m - m')` whenever a key raises the max,
//! and never materializes the scores. Its result must match A1's naive kernel (run by the oracle) and the host loop within a
//! stated bound. The data (see A1) has scores past 89 and rows whose maximum arrives last, so a missing rescale or a wrong
//! factor is visible.

use unpopped::kernel::check_kernel;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/attention.rs"]
mod attention;
use attention::{
    DV, F32, LK, LQ, attention_data, attention_reference, flash_attention, naive_attention,
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
fn the_flash_kernel_matches_the_host_loop_and_the_naive_kernel() {
    let k = flash_attention();
    assert_eq!(check_kernel(&k), Ok(()));
    let (q, kk, v) = attention_data();
    let mut args = [
        f32_buf(&q),
        f32_buf(&kk),
        f32_buf(&v),
        f32_buf(&[0.0; LQ * DV]),
    ];
    run_launch(&k, &mut args).expect("the oracle runs the flash kernel");
    let got = floats(&args[3]);

    let mut naive = [
        f32_buf(&q),
        f32_buf(&kk),
        f32_buf(&v),
        f32_buf(&[0.0; LQ * LK]),
        f32_buf(&[0.0; LQ * DV]),
    ];
    run_launch(&naive_attention(), &mut naive).expect("the oracle runs the naive kernel");
    close(&got, &floats(&naive[4]), "flash vs naive");
    close(&got, &attention_reference().1, "flash vs host");
}

#[test]
fn some_row_raises_its_maximum_after_the_first_key() {
    // Otherwise the rescale factor is always exp(NO_SCORE - s0) then 1, and a wrong factor hides.
    let (s, _) = attention_reference();
    let raised = (0..LQ).filter(|&i| {
        let row = &s[i * LK..(i + 1) * LK];
        (1..LK).any(|j| row[j] > row[..j].iter().cloned().fold(f32::MIN, f32::max))
    });
    assert!(
        raised.count() >= 2,
        "rows whose running max moves more than once"
    );
}

#[test]
fn rows_whose_every_score_is_very_negative_still_normalize() {
    // q = 3, k = -4: every score is SCALE * 4 * -12 = -288, where exp underflows an f32. A running max that starts at 0
    // instead of "no score yet" gives p = 0, l = 0 and 0/0; the true answer is the mean of V's rows (all weights equal).
    use attention::{D, LK};
    let k = flash_attention();
    let q = vec![3.0f32; LQ * D];
    let kk = vec![-4.0f32; LK * D];
    let (_, _, v) = attention_data();
    let mut args = [
        f32_buf(&q),
        f32_buf(&kk),
        f32_buf(&v),
        f32_buf(&[0.0; LQ * DV]),
    ];
    run_launch(&k, &mut args).expect("the oracle runs it");
    let mean: Vec<f32> = (0..DV)
        .map(|d| (0..LK).map(|j| v[j * DV + d]).sum::<f32>() / LK as f32)
        .collect();
    let want: Vec<f32> = (0..LQ).flat_map(|_| mean.clone()).collect();
    close(&floats(&args[3]), &want, "uniform weights");
}
