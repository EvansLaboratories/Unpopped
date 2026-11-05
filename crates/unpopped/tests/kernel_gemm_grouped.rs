// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Grouped GEMM in the oracle** (ROADMAP U9 K5, `docs/options-u9-gemm-family.md`): three GEMMs of different shapes in
//! one launch, one block per problem, described by a problem table of element offsets (the kernel-layer form of a
//! grouped / pointer-array GEMM until pointer values exist). Checked against a host loop; `d` is exactly the packed
//! outputs, so a thread writing outside its problem would run off or overwrite a neighbour.

use unpopped::kernel::check_kernel;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

#[path = "support/grouped.rs"]
mod grouped;
use grouped::{COLS, I32, PROBLEMS, U32, grouped_data, grouped_gemm};

fn buf(dtype: ElementKind, shape: &[i64], bytes: Vec<u8>) -> KernelArg {
    let strides: Vec<i64> = (0..shape.len())
        .map(|d| shape[d + 1..].iter().product())
        .collect();
    KernelArg::Buffer(TypedBuffer::new(dtype, shape.to_vec(), strides, bytes))
}

fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn words(a: &KernelArg) -> Vec<i32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!("a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `(args, expected D)`: the table, `a`, `b`, and a zeroed `d` one element longer than needed on each side is not used;
/// `d` is exactly the packed outputs, so an out-of-problem write is out of bounds or overwrites a neighbour.
fn args() -> (Vec<KernelArg>, Vec<i32>) {
    let (tbl, a, b, dlen, d) = grouped_data();
    let tbl_bytes: Vec<u8> = tbl.iter().flat_map(|x| x.to_le_bytes()).collect();
    (
        vec![
            buf(U32, &[PROBLEMS.len() as i64, COLS as i64], tbl_bytes),
            buf(I32, &[a.len() as i64], i32s(&a)),
            buf(I32, &[b.len() as i64], i32s(&b)),
            buf(I32, &[dlen as i64], i32s(&vec![0; dlen])),
        ],
        d,
    )
}

#[test]
fn every_problem_of_the_group_matches_the_host_loop() {
    let k = grouped_gemm();
    assert_eq!(check_kernel(&k), Ok(()));
    let (mut args, want) = args();
    run_launch(&k, &mut args).unwrap();
    assert_eq!(words(&args[3]), want);
    // the problems differ in shape, and the outputs are not all one value
    assert!(want.iter().any(|v| *v < 0) && want.iter().any(|v| *v > 0));
}
