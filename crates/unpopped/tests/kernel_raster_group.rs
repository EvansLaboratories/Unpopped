// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Grouped block rasterization** (survey row 31). A 5 x 4 grid of tiles with groups of two rows (the last group has one row, so
//! the clamp matters). The oracle's `(tile_m, tile_n)` per tile id is compared with a reference that enumerates the order by
//! nested loops (groups, then columns, then rows of the group) instead of the kernel's divide-and-modulo, and the mapping is
//! checked to be a bijection of the grid. Integers: there is no NaN; each mutant must change the mapping.

use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/raster_group.rs"]
mod fixture;
use fixture::{GROUP_M, Mutant, NUM_M, NUM_N, TILES, U32, raster_group};

fn buf() -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        U32,
        vec![TILES as i64],
        vec![1],
        vec![0; 4 * TILES],
    ))
}

fn words(a: &KernelArg) -> Vec<u32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!("a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn run(mutant: Mutant) -> Vec<(u32, u32)> {
    let k: KernelFn = raster_group(mutant);
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = vec![buf(), buf()];
    run_launch(&k, &mut args).expect("the oracle runs it");
    words(&args[0]).into_iter().zip(words(&args[1])).collect()
}

/// The visiting order by nested loops: groups of `GROUP_M` rows, within a group column by column, within a column row by row.
fn reference() -> Vec<(u32, u32)> {
    let mut order = Vec::new();
    for first in (0..NUM_M).step_by(GROUP_M) {
        for n in 0..NUM_N {
            for m in first..(first + GROUP_M).min(NUM_M) {
                order.push((m as u32, n as u32));
            }
        }
    }
    order
}

#[test]
fn the_mapping_matches_the_nested_loop_order() {
    assert_eq!(run(Mutant::None), reference());
}

#[test]
fn the_mapping_is_a_bijection_of_the_grid_and_differs_from_row_major() {
    let mut seen = run(Mutant::None);
    let row_major: Vec<(u32, u32)> = (0..TILES)
        .map(|t| ((t / NUM_N) as u32, (t % NUM_N) as u32))
        .collect();
    assert_ne!(seen, row_major, "the grouping must reorder the tiles");
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), TILES);
    assert!(
        seen.iter()
            .all(|&(m, n)| (m as usize) < NUM_M && (n as usize) < NUM_N)
    );
}

#[test]
fn the_mutants_fail_by_value() {
    for mutant in [Mutant::NoClamp, Mutant::ModuloN] {
        assert_ne!(run(mutant), reference(), "{mutant:?}");
    }
}
