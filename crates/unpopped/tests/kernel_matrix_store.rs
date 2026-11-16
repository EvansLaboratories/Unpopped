// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`Stmt::MatrixStore` in the launch interpreter** (ROADMAP U8b slice 2, `docs/options-u8-copy-atoms.md` S2):
//! `stmatrix`'s lane distribution for 1, 2 or 4 8 x 8 tiles of 16-bit elements, plain and transposed, written in the test as
//! `(row, col) -> (lane, element)` formulas from the PTX ISA's description, independent of the oracle's. The race a missing
//! barrier makes and the checker's refusals each have a program that raises them and a control that does not.

use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace};
use unpopped::kernel::{KernelError, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/matrix_store.rs"]
mod matrix_store;
use matrix_store::{H16, fragment_input, matrix_store_kernel, tiles_expected};

fn buf(v: &[u16]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        H16,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn run(count: u32, trans: bool, barrier: bool) -> Result<Vec<u16>, KernelTrap> {
    let k = matrix_store_kernel(count, trans, barrier);
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = [buf(&fragment_input()), buf(&vec![0; 64 * count as usize])];
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
fn every_tile_cell_holds_the_element_ptx_says() {
    for count in [1, 2, 4] {
        for trans in [false, true] {
            assert_eq!(
                run(count, trans, true).expect("the oracle runs it"),
                tiles_expected(count, trans),
                "count {count}, trans {trans}"
            );
        }
    }
}

#[test]
fn the_transpose_is_not_the_identity() {
    // otherwise a `trans` that does nothing would pass
    assert_ne!(tiles_expected(4, true), tiles_expected(4, false));
}

#[test]
fn lanes_that_supply_no_row_are_not_evaluated() {
    // count 1 allocates 64 elements; lanes 8..32 name addresses 64.. which are out of bounds and must not be evaluated
    let k = matrix_store_kernel(1, false, true);
    assert_eq!(k.allocs[0].extent, Extent::Static(64));
    assert!(run(1, false, true).is_ok());
}

#[test]
fn reading_a_cell_another_lane_stored_without_a_barrier_is_a_data_race() {
    assert!(matches!(
        run(4, false, false),
        Err(KernelTrap::DataRace | KernelTrap::UninitializedShared { .. })
    ));
}

#[test]
fn the_checker_refuses_what_a_warp_store_cannot_be() {
    let refused = |k| {
        matches!(
            check_kernel(&k),
            Err(KernelError::MatrixStoreUnsupported { .. })
        )
    };
    let mut subgroup16 = matrix_store_kernel(4, false, true);
    subgroup16.launch = Some(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(16, 16));
    let mut bad_src = matrix_store_kernel(4, false, true);
    bad_src.allocs[1] = AllocDecl::new("frag", MemorySpace::Private, H16, Extent::Static(7));
    let mut shared_src = matrix_store_kernel(4, false, true);
    shared_src.allocs[1] = AllocDecl::new("frag", MemorySpace::Shared, H16, Extent::Static(8));
    for (what, k) in [
        ("subgroup", subgroup16),
        ("fragment length", bad_src),
        ("fragment space", shared_src),
    ] {
        assert!(refused(k), "{what}");
    }
    // the control
    assert_eq!(check_kernel(&matrix_store_kernel(4, true, true)), Ok(()));
}
