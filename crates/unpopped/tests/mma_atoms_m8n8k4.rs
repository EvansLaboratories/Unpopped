// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The Volta `mma.sync.aligned.m8n8k4` f16 atoms (ROADMAP U70): four independent 8x8x4 MMAs per warp, every fragment element
//! where PTX ISA 9.5 section 9.7.16.5.1 says, and no group reads or writes another group's lanes.** Formulas from the text
//! (Figures 49-55, pp. 531-536), written here as `(row, col)` of `%laneid`, independently of `unpopped::mma`. Patent column:
//! applies 519 B; US12314717B2 wrapper read 2026-11-27; not a clearance; no attorney; findings not advice.

use unpopped::mma::{Major, MmaAtom};
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, F16, F32};

#[path = "support/mma_run.rs"]
#[allow(dead_code, reason = "shared helpers: each test file uses a subset")]
mod support;
use support::{buffer, decode, mma_kernel, value};

/// `(row, col)` of value `i` of lane `laneid` for operand `which` (0 = A, 1 = B, 2 = C/D), the PTX text for the given majors.
fn table(a: Major, b: Major, acc: ElementKind, which: usize, laneid: u32, i: u32) -> (u32, u32) {
    let (lo, hi) = (laneid % 4, u32::from(laneid >= 16));
    match which {
        0 => match a {
            Major::Row => (lo + 4 * hi, i),
            _ => (i + 4 * hi, lo),
        },
        1 => match b {
            Major::Row => (lo, i + 4 * hi),
            _ => (i, lo + 4 * hi),
        },
        _ if acc == F16 => (lo + 4 * hi, i),
        _ => (
            (laneid & 1) + (i & 2) + 4 * hi,
            (i & 4) + (laneid & 2) + (i & 1),
        ),
    }
}

/// The computation (group) lane `laneid` takes part in: lanes 0-3 and 16-19 are computation 1, and so on.
fn group(laneid: u32) -> u32 {
    (laneid / 4) % 4
}

const MAJORS: [Major; 2] = [Major::Row, Major::Col];
const ACCS: [ElementKind; 2] = [F16, F32];

#[test]
fn each_group_places_its_fragments_where_the_ptx_text_says() {
    for (a, b, acc) in all() {
        let atom = MmaAtom::nvidia_m8n8k4_f16(a, b, acc);
        assert_eq!((atom.mnk(), atom.lanes(), atom.groups()), ((8, 8, 4), 8, 4));
        assert_eq!(atom.subgroup_lanes(), 32);
        for (which, rows, cols) in [(0, 8, 4), (1, 4, 8), (2, 8, 8)] {
            for g in 0..4 {
                let mut seen = vec![0u32; (rows * cols) as usize];
                for local in 0..8 {
                    let laneid = atom.group_lane(g, local).unwrap();
                    assert_eq!(group(laneid), g, "{} lane map", atom.name());
                    for v in 0..atom.values(which) {
                        let (r, c) = table(a, b, acc, which, laneid, v);
                        let id = r + rows * c;
                        assert_eq!(
                            atom.id(which, local, v),
                            Some(id),
                            "{} op {which} group {g} lane {laneid} v {v}",
                            atom.name()
                        );
                        seen[id as usize] += 1;
                    }
                }
                assert!(
                    seen.iter().all(|&n| n == 1),
                    "{} op {which} group {g}",
                    atom.name()
                );
            }
        }
    }
}

fn all() -> Vec<(Major, Major, ElementKind)> {
    let mut v = Vec::new();
    for a in MAJORS {
        for b in MAJORS {
            for acc in ACCS {
                v.push((a, b, acc));
            }
        }
    }
    v
}

/// Per group: logical A (8x4), B (4x8), C (8x8) column-major, from seed `seed + 16 * g`.
fn logical(g: u32, seed: u32, acc: ElementKind) -> [Vec<f64>; 3] {
    let mk = |ty, s: u32, len: u32| (0..len).map(|i| value(ty, s + 16 * g, i)).collect();
    [
        mk(F16, seed, 32),
        mk(F16, seed + 1, 32),
        mk(acc, seed + 2, 64),
    ]
}

/// `D = C + A.B` per group, then every lane's D fragment through the table.
fn reference(a: Major, b: Major, acc: ElementKind, l: &[[Vec<f64>; 3]; 4]) -> Vec<f64> {
    let d: Vec<Vec<f64>> = l
        .iter()
        .map(|[ma, mb, mc]| {
            let mut d = vec![0.0; 64];
            for col in 0..8 {
                for row in 0..8 {
                    d[row + 8 * col] = (0..4).fold(mc[row + 8 * col], |s, k| {
                        s + ma[row + 8 * k] * mb[k + 4 * col]
                    });
                }
            }
            d
        })
        .collect();
    (0..32)
        .flat_map(|lane| (0..8).map(move |i| (lane, i)))
        .map(|(lane, i)| {
            let (r, c) = table(a, b, acc, 2, lane, i);
            d[group(lane) as usize][(r + 8 * c) as usize]
        })
        .collect()
}

/// Run the atom over `l` (one logical set per group) and return every lane's D fragment.
fn run(atom: &MmaAtom, a: Major, b: Major, acc: ElementKind, l: &[[Vec<f64>; 3]; 4]) -> Vec<f64> {
    let frag = |which: usize, vals: u32, rows: u32| -> Vec<f64> {
        (0..32)
            .flat_map(|lane| (0..vals).map(move |i| (lane, i)))
            .map(|(lane, i)| {
                let (r, c) = table(a, b, acc, which, lane, i);
                l[group(lane) as usize][which][(r + rows * c) as usize]
            })
            .collect()
    };
    let fc = frag(2, 8, 8);
    let mut args = vec![
        KernelArg::Buffer(buffer(F16, &frag(0, 4, 8))),
        KernelArg::Buffer(buffer(F16, &frag(1, 4, 4))),
        KernelArg::Buffer(buffer(acc, &fc)),
        KernelArg::Buffer(buffer(acc, &vec![0.0; fc.len()])),
    ];
    run_launch(&mma_kernel(atom), &mut args).unwrap_or_else(|e| panic!("{}: {e:?}", atom.name()));
    let KernelArg::Buffer(out) = &args[3] else {
        unreachable!("fd is a buffer")
    };
    out.bytes
        .chunks(out.bytes.len() / fc.len())
        .map(|bytes| {
            if acc == F16 {
                f64::from(half::f16::from_le_bytes(bytes.try_into().unwrap()))
            } else {
                decode(acc, bytes)
            }
        })
        .collect()
}

#[test]
fn the_four_groups_compute_independently_in_the_oracle() {
    for (a, b, acc) in all() {
        let atom = MmaAtom::nvidia_m8n8k4_f16(a, b, acc);
        let l = [0, 1, 2, 3].map(|g| logical(g, 1, acc));
        assert_eq!(
            run(&atom, a, b, acc, &l),
            reference(a, b, acc, &l),
            "{}",
            atom.name()
        );
    }
}

#[test]
fn changing_one_groups_inputs_changes_only_that_groups_lanes() {
    let (a, b, acc) = (Major::Row, Major::Col, F32);
    let atom = MmaAtom::nvidia_m8n8k4_f16(a, b, acc);
    let base = [0, 1, 2, 3].map(|g| logical(g, 1, acc));
    let mut changed = base.clone();
    changed[1][0] = changed[1][0].iter().map(|x| x + 1.0).collect();
    let (d0, d1) = (
        run(&atom, a, b, acc, &base),
        run(&atom, a, b, acc, &changed),
    );
    assert_ne!(d0, d1, "control: the perturbation reaches group 1");
    for lane in 0..32usize {
        let same = d0[lane * 8..lane * 8 + 8] == d1[lane * 8..lane * 8 + 8];
        assert_eq!(same, group(lane as u32) != 1, "lane {lane}");
    }
}

#[test]
fn every_other_atom_is_one_group_with_the_identity_lane_map() {
    for atom in [
        MmaAtom::nvidia_m16n8k16_f16_f32(),
        MmaAtom::nvidia_m16n8k32_s8_s32(),
        MmaAtom::nvidia_m16n8k8_tf32_f32(),
    ] {
        assert_eq!((atom.groups(), atom.subgroup_lanes()), (1, atom.lanes()));
        assert!((0..atom.lanes()).all(|l| atom.group_lane(0, l) == Some(l)));
    }
}
