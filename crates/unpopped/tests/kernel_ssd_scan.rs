// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A matrix-state scan kernel (CUTLASS survey row 93, Mamba2 SSD / linear-attention scans).** The kernel is the scan
//! inside a Mamba2 SSD block: `H_t = a_t H_{t-1} + B_t (outer) x_t` over a 2 x 2 state with the readout
//! `y_t = C_t^T H_t`, one lane per time step, as one `Stmt::CollectiveTuple { ScanWith }` over the five-component state
//! `(A, H)` (see `support/ssd_scan.rs`). The reference is a plain sequential loop over time written here, not the
//! combine. Three corpora: ordinary values with a negative decay, a zero decay and a decay above one; a corpus whose last
//! row is all negative; a corpus with a NaN input. Four mutants of the module are built and must fail the comparison.

use unpopped::exec::Scope;
use unpopped::kernel::check_module;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch_module;
use unpopped_vocab::ElementKind;

#[path = "support/ssd_scan.rs"]
mod fixture;
use fixture::{Mutant, N, SG, ssd_module};

/// One corpus: decay `a[t]`, `b[t][i]`, `x[t][j]`, `c[t][i]`.
struct Corpus {
    a: [f32; N],
    b: [[f32; 2]; N],
    x: [[f32; 2]; N],
    c: [[f32; 2]; N],
}

/// The outputs, per lane: the carried decay, the 2 x 2 state (row-major) and `y`.
#[derive(Debug)]
struct Out {
    a: Vec<f32>,
    h: Vec<f32>,
    y: Vec<f32>,
}

fn ordinary() -> Corpus {
    Corpus {
        // negative, zero (a reset of the state), above one, and a sign flip
        a: [0.5, -0.75, 1.0, 0.0, 2.0, -1.5, 0.25, 1.0],
        b: [
            [1.0, 2.0],
            [0.5, -1.0],
            [-2.0, 0.25],
            [3.0, 1.0],
            [-1.0, -0.5],
            [0.75, 2.0],
            [1.5, -3.0],
            [2.0, 1.0],
        ],
        x: [
            [1.0, -1.0],
            [2.0, 0.5],
            [-0.5, 1.5],
            [1.0, 1.0],
            [-2.0, 3.0],
            [0.25, -0.25],
            [4.0, 1.0],
            [-1.0, 2.0],
        ],
        c: [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [-1.0, 0.5],
            [0.5, 0.5],
            [1.0, -2.0],
            [3.0, 1.0],
        ],
    }
}

/// The last row (lane 7) is all negative, so a lane that drops its sign shows.
fn negative_last_row() -> Corpus {
    let mut c = ordinary();
    c.a[7] = -0.5;
    c.b[7] = [-1.0, -2.0];
    c.x[7] = [-3.0, -0.5];
    c.c[7] = [-1.0, -1.0];
    c
}

fn with_nan() -> Corpus {
    let mut c = ordinary();
    c.x[6][1] = f32::NAN;
    c
}

/// The reference: the recurrence in time order, restarting at every group of `group` lanes (the subgroup scope).
fn reference(c: &Corpus, group: usize) -> Out {
    let mut out = Out {
        a: vec![],
        h: vec![],
        y: vec![],
    };
    let (mut acc, mut h) = (1.0f32, [[0.0f32; 2]; 2]);
    for t in 0..N {
        if t % group == 0 {
            // the first lane of a group is its own element, not folded into a unit
            acc = c.a[t];
            for i in 0..2 {
                for j in 0..2 {
                    h[i][j] = c.b[t][i] * c.x[t][j];
                }
            }
        } else {
            acc *= c.a[t];
            for i in 0..2 {
                for j in 0..2 {
                    h[i][j] = c.a[t] * h[i][j] + c.b[t][i] * c.x[t][j];
                }
            }
        }
        out.a.push(acc);
        out.h.extend([h[0][0], h[0][1], h[1][0], h[1][1]]);
        for j in 0..2 {
            out.y.push(c.c[t][0] * h[0][j] + c.c[t][1] * h[1][j]);
        }
    }
    out
}

fn buf(shape: Vec<i64>, v: &[f32]) -> KernelArg {
    let strides = if shape.len() == 1 {
        vec![1]
    } else {
        vec![shape[1], 1]
    };
    let bytes = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    KernelArg::Buffer(TypedBuffer::new(ElementKind::F32, shape, strides, bytes))
}

fn zeros(shape: Vec<i64>) -> KernelArg {
    let n: i64 = shape.iter().product();
    buf(shape, &vec![0.0; n as usize])
}

fn f32s(a: &KernelArg) -> Vec<f32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!()
    };
    b.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn run(c: &Corpus, scope: Scope, mutant: Mutant) -> Out {
    let flat = |m: &[[f32; 2]; N]| m.iter().flatten().copied().collect::<Vec<_>>();
    let n = N as i64;
    let mut args = vec![
        buf(vec![n], &c.a),
        buf(vec![n, 2], &flat(&c.b)),
        buf(vec![n, 2], &flat(&c.x)),
        buf(vec![n, 2], &flat(&c.c)),
        zeros(vec![n]),
        zeros(vec![n, 4]),
        zeros(vec![n, 2]),
    ];
    run_launch_module(&ssd_module(scope, mutant), 0, &mut args).expect("runs");
    Out {
        a: f32s(&args[4]),
        h: f32s(&args[5]),
        y: f32s(&args[6]),
    }
}

/// Equal to the last bit, or both NaN.
fn same(got: &[f32], want: &[f32]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()))
}

fn agrees(got: &Out, want: &Out) -> bool {
    same(&got.a, &want.a) && same(&got.h, &want.h) && same(&got.y, &want.y)
}

fn corpora() -> [(&'static str, Corpus); 3] {
    [
        ("ordinary", ordinary()),
        ("negative last row", negative_last_row()),
        ("nan input", with_nan()),
    ]
}

#[test]
fn the_ssd_module_is_checker_valid_at_both_scopes() {
    for scope in [Scope::Block, Scope::Subgroup] {
        assert_eq!(check_module(&ssd_module(scope, Mutant::None), 0), Ok(()));
    }
}

#[test]
fn the_oracle_run_equals_the_sequential_recurrence_at_block_scope() {
    for (name, c) in corpora() {
        let (got, want) = (run(&c, Scope::Block, Mutant::None), reference(&c, N));
        assert!(agrees(&got, &want), "{name}: {got:?} vs {want:?}");
    }
}

#[test]
fn the_oracle_run_equals_the_recurrence_restarted_in_each_subgroup() {
    for (name, c) in corpora() {
        let (got, want) = (run(&c, Scope::Subgroup, Mutant::None), reference(&c, SG));
        assert!(agrees(&got, &want), "{name}: {got:?} vs {want:?}");
    }
}

#[test]
fn the_corpus_is_not_blind_it_has_a_nan_lane_a_zero_decay_and_distinct_scopes() {
    // the NaN input reaches the output of lane 6 and 7 only, so a comparison that ignored NaN would pass wrongly
    let want = reference(&with_nan(), N);
    assert!(
        want.h[6 * 4 + 1].is_nan() && want.h[7 * 4 + 3].is_nan() && !want.h[5 * 4 + 1].is_nan()
    );
    // a zero decay at lane 3 makes the state there only that lane's own element
    let o = ordinary();
    let r = reference(&o, N);
    assert_eq!(r.h[3 * 4], o.b[3][0] * o.x[3][0]);
    // the two scopes disagree, so a scope the checker ignored would show
    assert!(!agrees(&reference(&o, N), &reference(&o, SG)));
}

#[test]
fn every_mutant_of_the_combine_or_the_collective_fails_the_comparison() {
    for mutant in [Mutant::WrongScale, Mutant::DropDecayProduct, Mutant::Reduce] {
        let caught = corpora()
            .iter()
            .any(|(_, c)| !agrees(&run(c, Scope::Block, mutant), &reference(c, N)));
        assert!(
            caught,
            "{mutant:?} agrees with the reference on every corpus"
        );
        // each mutant is still a valid module: it is the comparison that refuses it, not the checker
        assert_eq!(check_module(&ssd_module(Scope::Block, mutant), 0), Ok(()));
    }
}
