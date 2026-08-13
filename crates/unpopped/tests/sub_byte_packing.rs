//! `i4` / `u4` / `b1` — the packing is KISS's, and it is checked against the
//! **byte layout**, not just against a round-trip.
//!
//! # Why a round-trip is not enough here
//!
//! A buffer packed high-nibble-first round-trips through this crate perfectly.
//! So does one packed MSB-first for `b1`. Both would disagree with every other
//! implementation on the first byte exchanged, and nothing inside this repository
//! would notice — the same shape as the reduce-field divergence, where a decoder
//! that accepted its own encoder's output hid a byte-level disagreement for
//! weeks.
//!
//! KISS-CLASSIFY §6.0-0001 makes the packing convention **normative**, alongside
//! the token and the storage width — packing is part of what the dtype *is*. So
//! these tests assert the actual bytes:
//!
//! * `i4`/`u4` — two per byte, **low nibble = even index, high nibble = odd**;
//!   `i4` sign-extended on read, `u4` zero-extended.
//! * `b1` — eight per byte, **LSB = lowest logical index**.

use unpopped::oracle::TypedBuffer;
use unpopped_vocab::ElementKind;

/// The nibble order is low-first, asserted on the byte itself.
///
/// Elements `[1, 2]` must produce the single byte `0x21` — `1` in the low nibble
/// (even index 0), `2` in the high nibble (odd index 1). The reversed convention
/// would produce `0x12`, and both decode "correctly" under their own reader.
#[test]
fn i4_packs_low_nibble_first() {
    let buf = TypedBuffer::from_sub_byte(ElementKind::I4, &[2], &[1, 2]);
    assert_eq!(
        buf.raw_bytes(),
        &[0x21],
        "low nibble = even index; 0x12 would be the reversed convention"
    );
    assert_eq!(buf.to_i128_vec(), vec![1, 2]);
}

/// `i4` is **sign-extended** on read; `u4` is **zero-extended**. Same bytes,
/// different values — which is the whole reason they are two dtypes.
#[test]
fn i4_sign_extends_where_u4_zero_extends() {
    // -1 is 0b1111; +15 is the same nibble read unsigned.
    let signed = TypedBuffer::from_sub_byte(ElementKind::I4, &[2], &[-1, -8]);
    let unsigned = TypedBuffer::from_sub_byte(ElementKind::U4, &[2], &[15, 8]);
    assert_eq!(
        signed.raw_bytes(),
        unsigned.raw_bytes(),
        "identical storage: 0xF and 0x8 in the two nibbles"
    );
    assert_eq!(signed.to_i128_vec(), vec![-1, -8], "i4 sign-extends");
    assert_eq!(unsigned.to_i128_vec(), vec![15, 8], "u4 zero-extends");
}

/// Every `i4` value survives, and the range is exactly `[-8, 7]`.
#[test]
fn every_i4_and_u4_value_round_trips() {
    let i4: Vec<i8> = (-8..=7).collect();
    let buf = TypedBuffer::from_sub_byte(ElementKind::I4, &[16], &i4);
    assert_eq!(buf.raw_bytes().len(), 8, "16 nibbles pack into 8 bytes");
    assert_eq!(
        buf.to_i128_vec(),
        i4.iter().map(|&v| i128::from(v)).collect::<Vec<_>>()
    );

    let u4: Vec<i8> = (0..=15).collect();
    let buf = TypedBuffer::from_sub_byte(ElementKind::U4, &[16], &u4);
    assert_eq!(
        buf.to_i128_vec(),
        u4.iter().map(|&v| i128::from(v)).collect::<Vec<_>>()
    );
}

/// `b1` packs LSB-first, asserted on the byte.
///
/// Elements `[1,0,0,0,0,0,0,1]` must be `0x81` — bit 0 set for logical index 0
/// and bit 7 for index 7. MSB-first would give the same byte for this
/// *palindromic* input, so the test uses an asymmetric one too.
#[test]
fn b1_packs_lsb_first() {
    let sym = TypedBuffer::from_sub_byte(ElementKind::B1, &[8], &[1, 0, 0, 0, 0, 0, 0, 1]);
    assert_eq!(sym.raw_bytes(), &[0x81]);

    // Asymmetric: index 1 set and nothing else. LSB-first gives 0x02;
    // MSB-first would give 0x40. A palindrome cannot tell those apart, which is
    // why this case exists.
    let asym = TypedBuffer::from_sub_byte(ElementKind::B1, &[8], &[0, 1, 0, 0, 0, 0, 0, 0]);
    assert_eq!(
        asym.raw_bytes(),
        &[0x02],
        "LSB = lowest logical index; 0x40 would be MSB-first"
    );
    assert_eq!(asym.to_i128_vec(), vec![0, 1, 0, 0, 0, 0, 0, 0]);
}

/// A partial trailing byte is allocated and does not corrupt its neighbours.
///
/// Three `i4` elements need two bytes with the high nibble of the second unused;
/// nine `b1` elements need two bytes with seven bits spare. Both are the case a
/// `len / elems_per_byte` allocation gets wrong by one.
#[test]
fn partial_trailing_bytes_are_allocated_and_isolated() {
    let i4 = TypedBuffer::from_sub_byte(ElementKind::I4, &[3], &[7, -8, 3]);
    assert_eq!(i4.raw_bytes().len(), 2, "3 nibbles need 2 bytes");
    assert_eq!(i4.to_i128_vec(), vec![7, -8, 3]);

    let b1 = TypedBuffer::from_sub_byte(ElementKind::B1, &[9], &[1, 0, 1, 0, 1, 0, 1, 0, 1]);
    assert_eq!(b1.raw_bytes().len(), 2, "9 bits need 2 bytes");
    assert_eq!(b1.to_i128_vec(), vec![1, 0, 1, 0, 1, 0, 1, 0, 1]);
    assert_eq!(
        b1.raw_bytes()[1],
        0x01,
        "the ninth bit is bit 0 of the second byte, with the rest clear"
    );
}

/// Writing one sub-byte element must not disturb the ones sharing its byte.
///
/// The read-modify-write case. A store that assigns the whole byte instead of
/// masking would zero its neighbour, and with dense data that damage is easy to
/// mistake for an arithmetic bug.
#[test]
fn writing_one_element_preserves_its_byte_neighbours() {
    let full = TypedBuffer::from_sub_byte(ElementKind::I4, &[4], &[1, 2, 3, 4]);
    assert_eq!(full.to_i128_vec(), vec![1, 2, 3, 4]);
    assert_eq!(full.raw_bytes(), &[0x21, 0x43]);

    let bits = TypedBuffer::from_sub_byte(ElementKind::B1, &[8], &[1, 1, 1, 1, 1, 1, 1, 1]);
    assert_eq!(
        bits.raw_bytes(),
        &[0xFF],
        "eight sets must not clobber each other"
    );
}
