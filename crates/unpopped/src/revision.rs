// SPDX-License-Identifier: MIT OR Apache-2.0
//! The kernel revision hash (a leaf: `backend` and `contract` both depend on it, neither on the other for it; ROADMAP U27).

/// [`revision_hash`] extended over a binary payload: after the source bytes, a `\0words\0` tag and
/// each word little-endian. `None` is the source-only hash, byte for byte.
pub(crate) fn revision_hash_of(src: &str, words: Option<&[u32]>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |b: u8| {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    src.bytes().for_each(&mut feed);
    if let Some(words) = words {
        b"\0words\0".iter().copied().for_each(&mut feed);
        words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .for_each(&mut feed);
    }
    h
}
