// SPDX-License-Identifier: MIT OR Apache-2.0
//! Deterministic input buffers for the kernel-emitter differentials. Separate from `kernel_run`
//! so a test that builds its own data does not carry (and warn about) these.

use unpopped::oracle::TypedBuffer;
use unpopped_vocab::ElementKind;

/// `n` distinct values that exercise rounding (not representable in a short decimal).
pub(crate) fn values(t: ElementKind, n: usize, seed: u32) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            let x = f64::from((i as u32 * 7919 + seed * 104_729) % 2003) / 97.0 - 10.0;
            match t {
                ElementKind::F64 => x.to_le_bytes().to_vec(),
                ElementKind::F32 => (x as f32).to_le_bytes().to_vec(),
                ElementKind::I64 | ElementKind::U64 => (x as i64).to_le_bytes().to_vec(),
                _ => (x as i32).to_le_bytes().to_vec(),
            }
        })
        .collect()
}

pub(crate) fn dense(t: ElementKind, shape: &[i64], seed: u32) -> TypedBuffer {
    let n = shape.iter().product::<i64>() as usize;
    let mut strides = vec![1i64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    TypedBuffer::new(t, shape.to_vec(), strides, values(t, n, seed))
}

pub(crate) fn zeroed(t: ElementKind, shape: &[i64]) -> TypedBuffer {
    let mut b = dense(t, shape, 0);
    b.bytes.fill(0);
    b
}
