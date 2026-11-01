// SPDX-License-Identifier: MIT OR Apache-2.0
//! **One rule, one author**: `window_taps::window_out_extent` (the reader-side check) and
//! `shape::windowed_extent` (plan shape inference) state the same output-extent rule, so they are
//! one function underneath and cannot drift. This pins that they agree over a grid of geometries,
//! and that the wide form saturates instead of overflowing on an adversarial extent.

use unpopped::shape::windowed_extent;
use unpopped::window_taps::window_out_extent;

#[test]
fn the_two_entry_points_agree_on_every_small_geometry() {
    let mut cases = 0;
    for k_in in 0..=20i64 {
        for size in 1..=5u8 {
            for stride in 1..=3u8 {
                for dilation in 1..=3u8 {
                    for (lo, hi) in [(0u8, 0u8), (1, 0), (0, 2), (2, 3)] {
                        let want = windowed_extent(k_in, size, stride, dilation, lo, hi);
                        let got = window_out_extent(
                            k_in,
                            u32::from(size),
                            u32::from(stride),
                            u32::from(dilation),
                            u32::from(lo),
                            u32::from(hi),
                        );
                        assert_eq!(got, want, "{k_in} {size} {stride} {dilation} {lo} {hi}");
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 21 * 5 * 3 * 3 * 4);
}

#[test]
fn an_adversarial_extent_saturates_and_a_zero_stride_reads_as_one() {
    assert_eq!(
        window_out_extent(i64::MAX, 1, 1, 1, u32::MAX, u32::MAX),
        i64::MAX
    );
    assert_eq!(
        window_out_extent(10, 3, 0, 1, 0, 0),
        window_out_extent(10, 3, 1, 1, 0, 0)
    );
}
