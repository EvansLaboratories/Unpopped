// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Paged-KV decode attention at the kernel layer** (ROADMAP U10 A4, `docs/options-u10-attention.md`): K and V live in a pool of
//! fixed-size pages, an `I32` block table maps (sequence, logical page) to a pool page, and each sequence has its own length
//! (Q5: a table in a buffer, no pointers). The result must match a host loop that gathers each sequence's tokens through the
//! table. Unused table entries name a poison page and the pool is not in logical order, so a wrong page, a wrong slot or a read
//! past a length changes the answer.

use unpopped::kernel::check_kernel;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

#[expect(dead_code, reason = "the shared kernels; this file uses the paged one")]
#[path = "support/attention.rs"]
mod attention;
use attention::{
    DV, F32, I32, LENS, MAXP, NSEQ, PAGE, TABLE, paged_attention, paged_data, paged_reference,
};

fn buf(ty: ElementKind, bytes: Vec<u8>, n: usize) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(ty, vec![n as i64], vec![1], bytes))
}

fn f32_buf(v: &[f32]) -> KernelArg {
    buf(
        F32,
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        v.len(),
    )
}

fn i32_buf(v: &[i32]) -> KernelArg {
    buf(
        I32,
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        v.len(),
    )
}

fn run(table: &[i32], lens: &[i32]) -> Vec<f32> {
    let k = paged_attention();
    assert_eq!(check_kernel(&k), Ok(()));
    let (q, kp, vp) = paged_data();
    let mut args = [
        f32_buf(&q),
        f32_buf(&kp),
        f32_buf(&vp),
        i32_buf(table),
        i32_buf(lens),
        f32_buf(&[0.0; NSEQ * DV]),
    ];
    run_launch(&k, &mut args).expect("the oracle runs it");
    let KernelArg::Buffer(b) = &args[5] else {
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
fn the_paged_kernel_matches_the_host_gather() {
    close(&run(&TABLE, &LENS), &paged_reference(), "paged");
}

#[test]
fn the_data_can_tell_the_paging_mutants_apart() {
    let base = run(&TABLE, &LENS);
    assert!(base.iter().all(|x| x.is_finite()));
    let row = |v: &[f32], s: usize| v[s * DV..(s + 1) * DV].to_vec();
    // One more token changes a sequence's answer (the next token is real data or poison).
    for s in 0..NSEQ {
        let mut lens = LENS;
        lens[s] += 1;
        if lens[s] as usize <= MAXP * PAGE {
            assert_ne!(row(&run(&TABLE, &lens), s), row(&base, s), "length of {s}");
        }
    }
    // Every used table entry matters: pointing it at another page changes that sequence's output.
    for s in 0..NSEQ {
        for p in 0..(LENS[s] as usize).div_ceil(PAGE) {
            let mut t = TABLE;
            let e = &mut t[s * MAXP + p];
            *e = if *e == 0 { 1 } else { *e - 1 };
            assert_ne!(row(&run(&t, &LENS), s), row(&base, s), "table[{s}][{p}]");
        }
    }
    // A zero-length sequence yields zeros.
    let empty = run(&TABLE, &[0, 2, 5]);
    assert!(empty[..DV].iter().all(|&x| x == 0.0));
}
