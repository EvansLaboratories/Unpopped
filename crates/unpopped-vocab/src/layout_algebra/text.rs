// SPDX-License-Identifier: MIT OR Apache-2.0
//! The canonical text of a [`Layout`], a [`Swizzle`] and a [`ComposedLayout`], and its parser.
//!
//! `shape:stride` with each side an integer or a parenthesised, comma-separated tuple, no spaces
//! (`((2,3),4):((1,2),6)`); a swizzle `swz(B,M,S)` (CuTe's `Swizzle<B,M,S>`: `bits`, `base`, `shift`);
//! a swizzled layout `swz(B,M,S)(shape:stride)`. Written for KISS Contract 6.5 `output_layouts`
//! (KISS board 378). **One way to say one thing:** the parser declines a space, a leading `+`, a
//! leading zero and `-0`, so `parse(print(x)) == x` and a printed text parses to one value only.
//! `S` alone may be negative; every other integer is non-negative, and the layout's own checks
//! ([`Layout::new`]) decide shape and stride leaves.

use super::{ComposedLayout, IntTuple, Layout, LayoutError, MAX_DEPTH, Swizzle};
use std::fmt;
use std::str::FromStr;

impl fmt::Display for Swizzle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "swz({},{},{})", self.bits(), self.base(), self.shift())
    }
}

impl fmt::Display for ComposedLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.swizzle(), self.layout())
    }
}

/// A cursor over the bytes of the text.
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            at: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn eat(&mut self, c: u8) -> bool {
        let hit = self.peek() == Some(c);
        if hit {
            self.at += 1;
        }
        hit
    }

    fn expect(&mut self, c: u8, what: &'static str) -> Result<(), LayoutError> {
        if self.eat(c) {
            Ok(())
        } else {
            Err(LayoutError::Parse(what))
        }
    }

    fn keyword(&mut self, word: &str, what: &'static str) -> Result<(), LayoutError> {
        let end = self.at + word.len();
        if self.bytes.get(self.at..end) == Some(word.as_bytes()) {
            self.at = end;
            Ok(())
        } else {
            Err(LayoutError::Parse(what))
        }
    }

    fn finish(&self) -> Result<(), LayoutError> {
        if self.at == self.bytes.len() {
            Ok(())
        } else {
            Err(LayoutError::Parse("trailing text"))
        }
    }

    /// A canonical integer: digits with no leading zero (`0` alone is fine), and a leading `-` only
    /// when `signed` (and never `-0`).
    fn int(&mut self, signed: bool) -> Result<i64, LayoutError> {
        let negative = signed && self.eat(b'-');
        let start = self.at;
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.at += 1;
        }
        let digits = &self.bytes[start..self.at];
        if digits.is_empty() {
            return Err(LayoutError::Parse("expected an integer"));
        }
        if digits.len() > 1 && digits[0] == b'0' {
            return Err(LayoutError::Parse("leading zero"));
        }
        if negative && digits == b"0" {
            return Err(LayoutError::Parse("-0"));
        }
        let mut v: i64 = 0;
        for &d in digits {
            v = v
                .checked_mul(10)
                .and_then(|x| x.checked_add(i64::from(d - b'0')))
                .ok_or(LayoutError::Overflow)?;
        }
        Ok(if negative { -v } else { v })
    }

    /// An [`IntTuple`] at nesting `level` (1 for the outermost tuple); deeper than [`MAX_DEPTH`] is
    /// declined before recursing, so no input can overflow the stack.
    fn tuple(&mut self, level: usize) -> Result<IntTuple, LayoutError> {
        if !self.eat(b'(') {
            return self.int(false).map(IntTuple::Int);
        }
        if level > MAX_DEPTH {
            return Err(LayoutError::TooDeep);
        }
        if self.peek() == Some(b')') {
            return Err(LayoutError::EmptyTuple);
        }
        let mut items = vec![self.tuple(level + 1)?];
        while self.eat(b',') {
            items.push(self.tuple(level + 1)?);
        }
        self.expect(b')', "expected `,` or `)`")?;
        Ok(IntTuple::Tuple(items))
    }

    fn layout(&mut self) -> Result<Layout, LayoutError> {
        let shape = self.tuple(1)?;
        self.expect(b':', "expected `:` between shape and stride")?;
        let stride = self.tuple(1)?;
        Layout::new(shape, stride)
    }

    fn swizzle(&mut self) -> Result<Swizzle, LayoutError> {
        self.keyword("swz(", "expected `swz(`")?;
        let bits = u32::try_from(self.int(false)?).map_err(|_| LayoutError::BadSwizzle)?;
        self.expect(b',', "expected `,`")?;
        let base = u32::try_from(self.int(false)?).map_err(|_| LayoutError::BadSwizzle)?;
        self.expect(b',', "expected `,`")?;
        let shift = i32::try_from(self.int(true)?).map_err(|_| LayoutError::BadSwizzle)?;
        self.expect(b')', "expected `)` after the swizzle")?;
        Swizzle::new(bits, base, shift)
    }
}

impl FromStr for Layout {
    type Err = LayoutError;

    /// Parse `shape:stride`; see the module docs for the canonical form.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut c = Cursor::new(s);
        let l = c.layout()?;
        c.finish()?;
        Ok(l)
    }
}

impl FromStr for Swizzle {
    type Err = LayoutError;

    /// Parse `swz(B,M,S)`; the parameters must satisfy [`Swizzle::new`].
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut c = Cursor::new(s);
        let z = c.swizzle()?;
        c.finish()?;
        Ok(z)
    }
}

impl FromStr for ComposedLayout {
    type Err = LayoutError;

    /// Parse `swz(B,M,S)(shape:stride)`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut c = Cursor::new(s);
        let z = c.swizzle()?;
        c.expect(b'(', "expected `(` before the layout")?;
        let l = c.layout()?;
        c.expect(b')', "expected `)` after the layout")?;
        c.finish()?;
        Ok(ComposedLayout::new(z, l))
    }
}
