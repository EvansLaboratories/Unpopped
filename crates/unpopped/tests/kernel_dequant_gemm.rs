// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The dequantizing GEMM at the kernel layer** (ROADMAP U9b S3, `docs/options-u9b-subbyte-dequant.md`): int4 weights widened
//! with a `Cast`, a per-group zero point subtracted and a per-group scale applied, against wider activations. The oracle's
//! result must equal a host loop for both `I4` (sign-extended) and `U4` (zero-extended) weights.

use unpopped::kernel::check_kernel;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

#[path = "support/dequant.rs"]
mod dequant;
use dequant::{C, I32, K, R, dequant_data, dequant_gemm, dequant_reference};

fn i32_buf(v: &[i32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        I32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

#[test]
fn the_dequant_gemm_matches_the_host_loop_for_i4_and_u4_weights() {
    for (sub, signed) in [(ElementKind::I4, true), (ElementKind::U4, false)] {
        let k = dequant_gemm(sub);
        assert_eq!(check_kernel(&k), Ok(()), "{sub:?}");
        let (packed, zero, scale, x) = dequant_data();
        let mut args = [
            KernelArg::Buffer(TypedBuffer::new(sub, vec![(R * K) as i64], vec![1], packed)),
            i32_buf(&zero),
            i32_buf(&scale),
            i32_buf(&x),
            i32_buf(&[0; R * C]),
        ];
        run_launch(&k, &mut args).expect("the oracle runs it");
        let KernelArg::Buffer(out) = &args[4] else {
            unreachable!("a buffer")
        };
        let got: Vec<i32> = out
            .bytes
            .chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, dequant_reference(signed), "{sub:?}");
    }
}

#[test]
fn the_two_weight_types_give_different_results_so_the_test_can_tell_them_apart() {
    assert_ne!(dequant_reference(true), dequant_reference(false));
}
