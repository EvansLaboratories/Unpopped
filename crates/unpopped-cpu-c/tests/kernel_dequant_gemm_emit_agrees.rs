// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of the dequantizing GEMM agrees with the launch interpreter and the host loop** (ROADMAP U9b S3,
//! `docs/options-u9b-subbyte-dequant.md`): `I4` and `U4` weights packed two to a byte, compiled with a host C compiler, run,
//! and the `out` buffer compared byte for byte with `run_launch` and with the host loop.

use unpopped::backend::Backend;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[path = "support/dequant.rs"]
mod dequant;
#[expect(
    dead_code,
    reason = "the shared data builders; this file builds its own"
)]
#[path = "support/kernel_data.rs"]
mod kernel_data;
#[expect(
    dead_code,
    reason = "the shared driver; this file uses the part that runs a launched kernel"
)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[path = "support/launch_run.rs"]
mod launch_run;
use dequant::{C, I32, K, R, dequant_data, dequant_gemm, dequant_reference};
use launch_run::agree_launch;

fn i32_buf(v: &[i32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        I32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn args(sub: ElementKind) -> Vec<KernelArg> {
    let (packed, zero, scale, x) = dequant_data();
    vec![
        KernelArg::Buffer(TypedBuffer::new(sub, vec![(R * K) as i64], vec![1], packed)),
        i32_buf(&zero),
        i32_buf(&scale),
        i32_buf(&x),
        i32_buf(&vec![0; R * C]),
    ]
}

#[test]
fn the_c_dequant_gemm_agrees_with_the_oracle_and_the_host_loop() {
    for (sub, signed) in [(ElementKind::I4, true), (ElementKind::U4, false)] {
        let k = dequant_gemm(sub);
        assert!(CpuC.lower_kernel(&k).is_ok(), "{sub:?} lowers");
        agree_launch(&format!("dequant {sub:?}"), &k, args(sub));
        let mut a = args(sub);
        run_launch(&k, &mut a).expect("the oracle runs it");
        let KernelArg::Buffer(out) = &a[4] else {
            unreachable!("a buffer")
        };
        let got: Vec<i32> = out
            .bytes
            .chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(
            got,
            dequant_reference(signed),
            "{sub:?} against the host loop"
        );
    }
}
