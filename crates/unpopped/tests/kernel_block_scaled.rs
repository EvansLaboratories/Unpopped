// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A block-scaled GEMM over the 512-byte scale-factor atom** (CUTLASS survey row 22; the kernel is `support/block_scaled.rs`).
//! The scales are packed into the atom by a NESTED loop over `(i, j, c)` with `row = i + 32 j` (independent of the kernel's
//! `r % 32`, `r / 32` index arithmetic), and the reference multiplies from the LOGICAL scale matrices. Three corpora: ordinary,
//! all-negative data, and a NaN scale that must reach exactly one output row. Mutants (swapped row factors, a column-major atom)
//! fail by value. One atom only: 128 rows and 4 scale columns; the tiling of atoms is not covered.

use unpopped::kernel::check_kernel;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/block_scaled.rs"]
mod fixture;
use fixture::{F32, K, M, Mutant, N, SFC, VEC, block_scaled_gemm};

fn buf(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

struct Case {
    a: Vec<f32>,
    b: Vec<f32>,
    /// Logical scales `[128][4]` for A and B.
    sfa: Vec<f32>,
    sfb: Vec<f32>,
}

fn ordinary() -> Case {
    let sf = |salt: usize| -> Vec<f32> {
        (0..M * SFC)
            .map(|i| [0.5, 1.0, 2.0, 4.0][(i * 7 + salt + i / 5) % 4])
            .collect()
    };
    Case {
        a: (0..M * K)
            .map(|i| ((i * 5 + i / 7) % 9) as f32 - 4.0)
            .collect(),
        b: (0..N * K).map(|i| ((i * 3 + 1) % 7) as f32 - 3.0).collect(),
        sfa: sf(1),
        sfb: sf(3),
    }
}

fn negative() -> Case {
    let mut c = ordinary();
    c.a.iter_mut().for_each(|x| *x = -x.abs() - 1.0);
    c.b.iter_mut().for_each(|x| *x = x.abs() + 1.0);
    c
}

fn nan() -> Case {
    let mut c = ordinary();
    c.sfa[77 * SFC + 2] = f32::NAN;
    c
}

/// Pack logical scales `[128][4]` into the atom by its nested form: `((i, j), (_, c))` at `i * 16 + j * 4 + c`, row `i + 32 j`.
fn pack_atom(sf: &[f32]) -> Vec<f32> {
    let mut atom = vec![0.0; 512];
    for i in 0..32 {
        for j in 0..4 {
            for c in 0..SFC {
                atom[i * 16 + j * 4 + c] = sf[(i + 32 * j) * SFC + c];
            }
        }
    }
    atom
}

fn reference(c: &Case) -> Vec<f32> {
    let mut out = vec![0.0; M * N];
    for m in 0..M {
        for n in 0..N {
            let mut acc = 0.0f32;
            for kb in 0..SFC {
                let mut inner = 0.0f32;
                for j in 0..VEC {
                    inner += c.a[m * K + kb * VEC + j] * c.b[n * K + kb * VEC + j];
                }
                acc += c.sfa[m * SFC + kb] * c.sfb[n * SFC + kb] * inner;
            }
            out[m * N + n] = acc;
        }
    }
    out
}

fn run(c: &Case, mutant: Mutant) -> Vec<f32> {
    let k = block_scaled_gemm(mutant);
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = [
        buf(&c.a),
        buf(&c.b),
        buf(&pack_atom(&c.sfa)),
        buf(&pack_atom(&c.sfb)),
        buf(&vec![0.0; M * N]),
    ];
    run_launch(&k, &mut args).expect("the oracle runs it");
    let KernelArg::Buffer(b) = &args[4] else {
        unreachable!("a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}

fn same(got: &[f32], want: &[f32]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()))
}

#[test]
fn the_block_scaled_gemm_matches_the_logical_reference_on_every_corpus() {
    for (name, c) in [
        ("ordinary", ordinary()),
        ("negative", negative()),
        ("nan", nan()),
    ] {
        assert!(same(&run(&c, Mutant::None), &reference(&c)), "{name}");
    }
}

#[test]
fn the_corpus_is_not_blind() {
    // the NaN scale reaches exactly output row 77
    let want = reference(&nan());
    let rows: Vec<usize> = (0..M)
        .filter(|m| want[m * N..(m + 1) * N].iter().any(|x| x.is_nan()))
        .collect();
    assert_eq!(rows, vec![77]);
    // the atom is a real permutation of the logical scales, and the scales move the result
    let c = ordinary();
    assert_ne!(pack_atom(&c.sfa), c.sfa[..512].to_vec());
    let mut flat = c;
    flat.sfa.iter_mut().for_each(|x| *x = 1.0);
    assert!(!same(&reference(&flat), &reference(&ordinary())));
}

#[test]
fn the_atom_layout_mutants_fail_by_value() {
    let c = ordinary();
    let want = reference(&c);
    for m in [Mutant::SwappedRowFactors, Mutant::ColumnMajorAtom] {
        assert!(!same(&run(&c, m), &want), "{m:?}");
    }
}
