// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`Stmt::MatrixLoad` in the launch interpreter** (ROADMAP U8b, `docs/options-u8-copy-atoms.md` S2): `ldmatrix`'s lane
//! distribution of 1, 2 or 4 8 x 8 tiles of 16-bit elements, plain and transposed, written in the test as `(row, col)`
//! formulas from the PTX ISA's description, independent of the oracle's. The checker's refusals and the race a missing
//! barrier makes each have a program that raises them and a control that does not.

use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace};
use unpopped::kernel::{KernelError, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/matrix_load.rs"]
mod matrix_load;
use matrix_load::{H16, matrix_expected, matrix_input, matrix_load_kernel};

fn buf(v: &[u16]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        H16,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn run(count: u32, trans: bool, barrier: bool) -> Result<Vec<u16>, KernelTrap> {
    let k = matrix_load_kernel(count, trans, barrier);
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = [buf(&matrix_input(count)), buf(&[0; 32 * 8])];
    run_launch(&k, &mut args)?;
    let KernelArg::Buffer(b) = &args[1] else {
        unreachable!("a buffer")
    };
    Ok(b.bytes
        .chunks(2)
        .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

#[test]
fn every_lane_receives_the_element_ptx_says() {
    for count in [1, 2, 4] {
        for trans in [false, true] {
            assert_eq!(
                run(count, trans, true).expect("the oracle runs it"),
                matrix_expected(count, trans),
                "count {count}, trans {trans}"
            );
        }
    }
}

#[test]
fn the_transpose_is_not_the_identity() {
    // otherwise a `trans` that does nothing would pass
    assert_ne!(matrix_expected(4, true), matrix_expected(4, false));
}

#[test]
fn lanes_that_supply_no_row_are_not_read() {
    // count 1 allocates 64 elements; lanes 8..32 name addresses 64.. which are out of bounds and must not be evaluated
    let k = matrix_load_kernel(1, false, true);
    assert_eq!(k.allocs[0].extent, Extent::Static(64));
    assert!(run(1, false, true).is_ok());
}

#[test]
fn reading_a_row_another_lane_has_not_synchronized_is_a_data_race() {
    assert!(matches!(
        run(4, false, false),
        Err(KernelTrap::DataRace | KernelTrap::UninitializedShared { .. })
    ));
}

#[test]
fn the_checker_refuses_what_a_warp_load_cannot_be() {
    let refused = |k| {
        matches!(
            check_kernel(&k),
            Err(KernelError::MatrixLoadUnsupported { .. })
        )
    };
    let mut subgroup16 = matrix_load_kernel(4, false, true);
    subgroup16.launch = Some(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(16, 16));
    let mut bad_dst = matrix_load_kernel(4, false, true);
    bad_dst.allocs[1] = AllocDecl::new("frag", MemorySpace::Private, H16, Extent::Static(7));
    let mut shared_dst = matrix_load_kernel(4, false, true);
    shared_dst.allocs[1] = AllocDecl::new("frag", MemorySpace::Shared, H16, Extent::Static(8));
    let mut wide = matrix_load_kernel(4, false, true);
    wide.allocs[0] = AllocDecl::new(
        "tiles",
        MemorySpace::Shared,
        unpopped_vocab::ElementKind::U32,
        Extent::Static(256),
    );
    for (what, k) in [
        ("subgroup", subgroup16),
        ("dst length", bad_dst),
        ("dst space", shared_dst),
    ] {
        assert!(refused(k), "{what}");
    }
    // the control
    assert_eq!(check_kernel(&matrix_load_kernel(4, true, true)), Ok(()));
    // a wider element is a type error elsewhere (the staging store) or a refusal here; it must not check
    assert!(check_kernel(&wide).is_err());
}
