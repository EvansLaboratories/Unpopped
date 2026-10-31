// SPDX-License-Identifier: MIT OR Apache-2.0
//! The real-tap rule for pooling windows (KISS-OPS-6.13-0010/-0011, CireSnave board 345 "B").
//!
//! An output window `o` of a 1-D window op reads the positions
//! `p = o * stride - pad_lo + kk * dilation` for `kk in 0..size`; a tap is **real** when
//! `0 <= p < k_in`. The rule: every output window has at least one real tap. A producer must not
//! write a shape that breaks it and a reader must decline one. The plan gate sees structure only
//! (no extents), and `2 * pad <= dilation * (size - 1) + 1` does not imply the rule once
//! `dilation > 1` (`n = 1, k = 2, d = 2, s = 1, p = 1` passes it and its one window has no real
//! tap), so the rule is checked here, where the extents are known: by the oracle's window
//! evaluation and by [`crate::kernel_lift::lift_window_opdef_for_extents`].

use std::fmt;

/// The output extent KISS states for a window axis: `floor((n + pad_lo + pad_hi - D) / stride) + 1`
/// with `D = dilation * (size - 1) + 1`, and `0` when no window fits (`n + pad_lo + pad_hi < D`).
/// `size`, `stride` and `dilation` must be at least 1.
#[must_use]
pub fn window_out_extent(
    k_in: i64,
    size: u32,
    stride: u32,
    dilation: u32,
    pad_lo: u32,
    pad_hi: u32,
) -> i64 {
    let span = i64::from(dilation) * (i64::from(size) - 1) + 1;
    let padded = k_in + i64::from(pad_lo) + i64::from(pad_hi);
    if padded < span {
        0
    } else {
        (padded - span) / i64::from(stride.max(1)) + 1
    }
}

/// An output window with no real tap: the first one, and the geometry that produced it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WindowNoRealTap {
    /// The first output position whose window has no tap inside `[0, k_in)`.
    pub output: i64,
    /// The input extent of the window axis.
    pub k_in: i64,
    /// The output extent of the window axis.
    pub k_out: i64,
    /// Window size (taps).
    pub size: u32,
    /// Stride between windows.
    pub stride: u32,
    /// Spacing between taps.
    pub dilation: u32,
    /// Padding before the axis.
    pub pad_lo: u32,
}

impl fmt::Display for WindowNoRealTap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Window output {} has no in-bounds tap (k_in {}, k_out {}, size {}, stride {}, \
             dilation {}, pad_lo {}): KISS-OPS-6.13-0011 requires a reader to decline it",
            self.output, self.k_in, self.k_out, self.size, self.stride, self.dilation, self.pad_lo
        )
    }
}

impl std::error::Error for WindowNoRealTap {}

/// Check the real-tap rule for a window axis of input extent `k_in` and output extent `k_out`.
///
/// # Errors
///
/// [`WindowNoRealTap`] naming the first output window with no tap inside `[0, k_in)`.
pub fn check_window_taps(
    k_in: i64,
    k_out: i64,
    size: u32,
    stride: u32,
    dilation: u32,
    pad_lo: u32,
) -> Result<(), WindowNoRealTap> {
    for output in 0..k_out.max(0) {
        let real = (0..i64::from(size)).any(|kk| {
            let p = output * i64::from(stride) - i64::from(pad_lo) + kk * i64::from(dilation);
            (0..k_in).contains(&p)
        });
        if !real {
            return Err(WindowNoRealTap {
                output,
                k_in,
                k_out,
                size,
                stride,
                dilation,
                pad_lo,
            });
        }
    }
    Ok(())
}
