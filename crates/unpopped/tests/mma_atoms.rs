// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The shipped MMA atoms put every fragment element where the PTX ISA says** (ROADMAP U7 S2). Each atom's
//! thread-value layouts are checked against the PTX fragment tables written here as plain `(row, col)` formulas,
//! independently of the strides in `unpopped::mma`; a wrong lane layout is the likeliest defect in an atom
//! definition and nothing else would notice it. Also: the construction-time bijection check refuses a layout that
//! maps two fragment elements to one logical id.

use unpopped::exec::Scope;
use unpopped::mma::{AtomSpec, MmaAtom, MmaAtomError};
use unpopped_vocab::ElementKind;
use unpopped_vocab::layout_algebra::Layout;

/// `(row, col)` of fragment element `i` of `lane` per the PTX tables, for operand 0/1/2 (A/B/C) of each atom.
type Table = fn(usize, u32, u32) -> (u32, u32);

/// PTX ISA "Matrix fragments for mma.m16n8k16 with floating point type": `groupID = lane >> 2`,
/// `threadID_in_group = lane % 4`.
fn m16n8k16(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid + 8 * ((i / 2) % 2), tig * 2 + (i % 2) + 8 * (i / 4)),
        1 => (tig * 2 + (i % 2) + 8 * (i / 2), gid),
        _ => (gid + 8 * (i / 2), tig * 2 + (i % 2)),
    }
}

/// PTX ISA "Matrix fragments for mma.m16n8k32 with integer type .s8".
fn m16n8k32(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid + 8 * ((i / 4) % 2), tig * 4 + (i % 4) + 16 * (i / 8)),
        1 => (tig * 4 + (i % 4) + 16 * (i / 4), gid),
        _ => (gid + 8 * (i / 2), tig * 2 + (i % 2)),
    }
}

fn check(atom: &MmaAtom, table: Table) {
    let (m, n, k) = atom.mnk();
    for (which, rows) in [(0, m), (1, k), (2, m)] {
        let cols = [k, n, n][which];
        let mut seen = vec![0u32; (rows * cols) as usize];
        for lane in 0..atom.lanes() {
            for v in 0..atom.values(which) {
                let (r, c) = table(which, lane, v);
                let want = r + rows * c;
                assert_eq!(
                    atom.id(which, lane, v),
                    Some(want),
                    "{} operand {which} lane {lane} value {v}",
                    atom.name()
                );
                seen[want as usize] += 1;
            }
        }
        // the table itself covers the matrix once: the control that the formulas are a bijection
        assert!(
            seen.iter().all(|&c| c == 1),
            "{} table {which}",
            atom.name()
        );
    }
}

#[test]
fn the_f16_and_bf16_atoms_match_the_ptx_fragment_tables() {
    check(&MmaAtom::nvidia_m16n8k16_f16_f32(), m16n8k16);
    check(&MmaAtom::nvidia_m16n8k16_bf16_f32(), m16n8k16);
}

#[test]
fn the_s8_atom_matches_the_ptx_fragment_tables() {
    check(&MmaAtom::nvidia_m16n8k32_s8_s32(), m16n8k32);
}

#[test]
fn the_atoms_state_their_types_and_shape() {
    let a = MmaAtom::nvidia_m16n8k16_bf16_f32();
    assert_eq!(a.mnk(), (16, 8, 16));
    assert_eq!(
        a.dtypes(),
        (ElementKind::Bf16, ElementKind::Bf16, ElementKind::F32)
    );
    assert_eq!((a.values(0), a.values(1), a.values(2)), (8, 4, 4));
    let s = MmaAtom::nvidia_m16n8k32_s8_s32();
    assert_eq!(
        s.dtypes(),
        (ElementKind::I8, ElementKind::I8, ElementKind::I32)
    );
    assert_eq!((s.values(0), s.values(1), s.values(2)), (16, 8, 4));
    assert_eq!(a.id(0, 32, 0), None);
    assert_eq!(a.id(0, 0, 8), None);
    assert_eq!(a.id(3, 0, 0), None);
}

fn leaves(modes: &[(i64, i64)]) -> Layout {
    let l: Vec<Layout> = modes
        .iter()
        .map(|&(s, d)| Layout::leaf(s, d).unwrap())
        .collect();
    Layout::from_modes(&l).unwrap()
}

/// A 4-lane, 2x2x2 toy atom to test the construction checks on.
fn toy(a: &Layout, b: &Layout, c: &Layout, scope: Scope) -> Result<MmaAtom, MmaAtomError> {
    MmaAtom::new(spec("toy", (2, 2, 2), scope), [a, b, c])
}

fn spec(name: &str, mnk: (u32, u32, u32), scope: Scope) -> AtomSpec<'_> {
    AtomSpec {
        name,
        mnk,
        dtypes: (ElementKind::F32, ElementKind::F32, ElementKind::F32),
        scope,
        lanes: 4,
    }
}

#[test]
fn a_layout_that_is_not_a_bijection_is_refused() {
    let ok = leaves(&[(4, 1), (1, 0)]);
    let good = toy(&ok, &ok, &ok, Scope::Subgroup);
    assert!(good.is_ok(), "control: {good:?}");
    // two lanes land on the same logical element
    let dup = leaves(&[(4, 0)]);
    assert_eq!(
        toy(&dup, &ok, &ok, Scope::Subgroup),
        Err(MmaAtomError::NotABijection("A"))
    );
    assert_eq!(
        toy(&ok, &ok, &dup, Scope::Subgroup),
        Err(MmaAtomError::NotABijection("C"))
    );
    // a layout that reaches past the matrix
    let past = leaves(&[(4, 2)]);
    assert_eq!(
        toy(&ok, &past, &ok, Scope::Subgroup),
        Err(MmaAtomError::NotABijection("B"))
    );
}

#[test]
fn a_wrong_size_a_wrong_scope_and_a_zero_dimension_are_refused() {
    let ok = leaves(&[(4, 1)]);
    let short = leaves(&[(2, 1)]);
    assert!(matches!(
        toy(&short, &ok, &ok, Scope::Subgroup),
        Err(MmaAtomError::BadLayout(_))
    ));
    assert_eq!(
        toy(&ok, &ok, &ok, Scope::Block),
        Err(MmaAtomError::BadScope)
    );
    let zero = MmaAtom::new(spec("z", (0, 2, 2), Scope::Subgroup), [&ok, &ok, &ok]);
    assert_eq!(zero, Err(MmaAtomError::ZeroDimension));
}
