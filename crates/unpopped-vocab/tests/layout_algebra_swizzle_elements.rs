// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Swizzle::in_elements`: a byte-address swizzle read over elements of a power-of-two width.
//! The identity `byte.apply(i * w) == w * elems.apply(i)` is checked for every valid `(bits, base, shift, width)` over a
//! small grid and every offset below 4096; a broken derivation must fail it.

use unpopped_vocab::layout_algebra::{LayoutError, Swizzle};

type Derive = dyn Fn(&Swizzle, u32) -> Result<Swizzle, LayoutError>;

fn check(derive: &Derive) -> Result<usize, String> {
    let mut cases = 0;
    for bits in 0..=3u32 {
        for base in 0..=5u32 {
            for shift in [-5i32, -3, 3, 4, 5] {
                let Ok(byte) = Swizzle::new(bits, base, shift) else {
                    continue;
                };
                for k in 0..=base {
                    let w = 1u32 << k;
                    let e = derive(&byte, w).map_err(|x| format!("{byte:?} w={w}: {x:?}"))?;
                    for i in 0..4096i64 {
                        let (b, el) = (byte.apply(i * i64::from(w)).unwrap(), e.apply(i).unwrap());
                        if b != el * i64::from(w) {
                            return Err(format!(
                                "{byte:?} w={w} i={i}: bytes {b}, elements {el} * {w}"
                            ));
                        }
                    }
                    cases += 1;
                }
            }
        }
    }
    Ok(cases)
}

#[test]
fn the_element_swizzle_is_the_byte_swizzle_over_whole_elements() {
    assert!(check(&|s, w| s.in_elements(w)).unwrap() >= 100);
    // mutants: the base is not reduced; the base is reduced twice
    assert!(check(&|s, _| Ok(*s)).is_err());
    assert!(
        check(&|s, w| Swizzle::new(
            s.bits(),
            s.base().saturating_sub(2 * w.trailing_zeros()),
            s.shift()
        ))
        .is_err()
    );
}

#[test]
fn the_ptx_128_byte_pattern_over_16_bit_elements() {
    // Swizzle<3,4,3> on bytes (PTX ISA 9.5 Figures 159-168, via docs/design-wgmma-descriptors.md 5c) over 2-byte elements.
    let b = Swizzle::new(3, 4, 3).unwrap();
    assert_eq!(b.in_elements(2).unwrap(), Swizzle::new(3, 3, 3).unwrap());
    assert_eq!(b.in_elements(1).unwrap(), b);
    assert_eq!(b.in_elements(16).unwrap(), Swizzle::new(3, 0, 3).unwrap());
    // byte 0x80 (bit 7) moves bit 4: 0x90; element 0x40 over 2-byte elements is byte 0x80
    assert_eq!(b.apply(0x80).unwrap(), 0x90);
    assert_eq!(b.in_elements(2).unwrap().apply(0x40).unwrap(), 0x48);
}

#[test]
fn what_it_declines_each_beside_an_accepted_control() {
    let b = Swizzle::new(3, 4, 3).unwrap();
    assert!(b.in_elements(16).is_ok()); // control: width 2^base is the widest accepted
    assert_eq!(b.in_elements(32), Err(LayoutError::BadSwizzle));
    assert!(b.in_elements(4).is_ok());
    assert_eq!(b.in_elements(3), Err(LayoutError::NotDivisible));
    assert_eq!(b.in_elements(0), Err(LayoutError::NotDivisible));
    assert_eq!(b.in_elements(12), Err(LayoutError::NotDivisible));
}
