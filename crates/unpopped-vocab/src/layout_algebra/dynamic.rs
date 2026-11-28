// SPDX-License-Identifier: MIT OR Apache-2.0
//! A layout whose shape and stride leaves may be kernel scalar parameters (`docs/design-dynamic-layout.md`, CuTe rows 3 and 66).
//!
//! A [`DynLayout`] is a **family of static layouts indexed by a launch's scalar arguments**: [`DynLayout::bind`] replaces every
//! parameter leaf by its value and returns an ordinary [`Layout`], so the whole algebra and every check stay on static layouts.
//! The rank and the nesting (the profile) are static; only leaves are dynamic. The layout algebra is never applied to an unbound
//! `DynLayout`: it has no `composition`, `zipped_divide` or `slice`.
//!
//! Text: `shape:stride` as for a [`Layout`], a leaf being an integer or `?N` for the integer scalar parameter whose local is `N`
//! (`(?0,?1):(?2,1)`). [`DynLayout::parse_with`] also takes names (`(m,n):(lda,1)`), resolved through a table the caller supplies;
//! the printed form always uses `?N`, so `parse(print(x)) == x`.

use super::{IntTuple, Layout, LayoutError, MAX_DEPTH, Swizzle};
use std::fmt;
use std::str::FromStr;

/// One leaf: a static integer or the integer scalar parameter whose local is the payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Dim {
    /// A value known now.
    Static(i64),
    /// The value of the scalar parameter local `.0`, known at launch.
    Param(u32),
}

/// An integer-or-tuple of [`Dim`]: the dynamic analogue of [`IntTuple`].
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum DynTuple {
    /// A leaf.
    Leaf(Dim),
    /// A non-empty tuple.
    Tuple(Vec<DynTuple>),
}

impl DynTuple {
    fn depth(&self) -> usize {
        match self {
            Self::Leaf(_) => 0,
            Self::Tuple(t) => 1 + t.iter().map(Self::depth).max().unwrap_or(0),
        }
    }

    /// The same nesting as `other` (every tuple the same arity at every level).
    fn congruent(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Leaf(_), Self::Leaf(_)) => true,
            (Self::Tuple(a), Self::Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.congruent(y))
            }
            _ => false,
        }
    }

    fn collect_params(&self, out: &mut Vec<u32>) {
        match self {
            Self::Leaf(Dim::Param(p)) => out.push(*p),
            Self::Leaf(Dim::Static(_)) => {}
            Self::Tuple(t) => t.iter().for_each(|x| x.collect_params(out)),
        }
    }

    fn bind(&self, values: &[(u32, i64)]) -> Result<IntTuple, LayoutError> {
        match self {
            Self::Leaf(Dim::Static(v)) => Ok(IntTuple::Int(*v)),
            Self::Leaf(Dim::Param(p)) => values
                .iter()
                .find(|(l, _)| l == p)
                .map(|(_, v)| IntTuple::Int(*v))
                .ok_or(LayoutError::UnboundParam(*p)),
            Self::Tuple(t) => Ok(IntTuple::Tuple(
                t.iter().map(|x| x.bind(values)).collect::<Result<_, _>>()?,
            )),
        }
    }
}

/// A layout with the same profile for `shape` and `stride`, whose leaves may be parameters.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct DynLayout {
    shape: DynTuple,
    stride: DynTuple,
    swizzle: Option<Swizzle>,
}

impl DynLayout {
    /// A dynamic layout. [`LayoutError::NotCongruent`] unless `shape` and `stride` have the same profile,
    /// [`LayoutError::TooDeep`] past `MAX_DEPTH`, [`LayoutError::NonPositiveShape`] for a static shape leaf below 1 and
    /// [`LayoutError::NegativeStride`] for a negative static stride leaf (a parameter leaf is checked when it is bound).
    ///
    /// # Errors
    ///
    /// As above.
    pub fn new(shape: DynTuple, stride: DynTuple) -> Result<Self, LayoutError> {
        if !shape.congruent(&stride) {
            return Err(LayoutError::NotCongruent);
        }
        if shape.depth() > MAX_DEPTH {
            return Err(LayoutError::TooDeep);
        }
        fn statics(
            t: &DynTuple,
            f: &mut dyn FnMut(i64) -> Result<(), LayoutError>,
        ) -> Result<(), LayoutError> {
            match t {
                DynTuple::Leaf(Dim::Static(v)) => f(*v),
                DynTuple::Leaf(Dim::Param(_)) => Ok(()),
                DynTuple::Tuple(x) => x.iter().try_for_each(|y| statics(y, f)),
            }
        }
        statics(&shape, &mut |v| {
            if v < 1 {
                Err(LayoutError::NonPositiveShape(v))
            } else {
                Ok(())
            }
        })?;
        statics(&stride, &mut |v| {
            if v < 0 {
                Err(LayoutError::NegativeStride(v))
            } else {
                Ok(())
            }
        })?;
        Ok(Self {
            shape,
            stride,
            swizzle: None,
        })
    }

    /// The same layout under the byte swizzle `swizzle` (`docs/design-dynamic-layout.md` section 8): XOR-bit data, as for an
    /// static allocation layout. [`DynLayout::bind`] still returns the static layout; the consumer applies the
    /// swizzle to its offsets and checks, at the launch, that its period divides the extent.
    #[must_use]
    pub fn with_swizzle(mut self, swizzle: Swizzle) -> Self {
        self.swizzle = Some(swizzle);
        self
    }

    /// The byte swizzle, if any.
    #[must_use]
    pub fn swizzle(&self) -> Option<Swizzle> {
        self.swizzle
    }

    /// The shape.
    #[must_use]
    pub fn shape(&self) -> &DynTuple {
        &self.shape
    }

    /// The stride.
    #[must_use]
    pub fn stride(&self) -> &DynTuple {
        &self.stride
    }

    /// The parameter locals the layout names, sorted and without repeats.
    #[must_use]
    pub fn params(&self) -> Vec<u32> {
        let mut out = Vec::new();
        self.shape.collect_params(&mut out);
        self.stride.collect_params(&mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The static layout this one is when every parameter takes the value `values` gives it (`(local, value)` pairs).
    ///
    /// # Errors
    ///
    /// [`LayoutError::UnboundParam`] for a parameter `values` omits, then [`Layout::new`]'s errors for the bound leaves
    /// (a non-positive shape, a negative stride).
    pub fn bind(&self, values: &[(u32, i64)]) -> Result<Layout, LayoutError> {
        Layout::new(self.shape.bind(values)?, self.stride.bind(values)?)
    }

    /// Parse the text form with names: a leaf that is not an integer or `?N` is looked up in `names` (`(name, local)` pairs).
    ///
    /// # Errors
    ///
    /// [`LayoutError::Parse`] for a malformed text or a name `names` lacks, then [`DynLayout::new`]'s errors.
    pub fn parse_with(text: &str, names: &[(&str, u32)]) -> Result<Self, LayoutError> {
        if text.starts_with("swz(") {
            // `swz(B,M,S)(shape:stride)`: the swizzle has no nested parenthesis, so its text ends at the first `)`
            let end = text
                .find(')')
                .ok_or(LayoutError::Parse("expected `)` after the swizzle"))?;
            let swizzle: Swizzle = text[..=end].parse()?;
            let inner = text[end + 1..]
                .strip_prefix('(')
                .and_then(|r| r.strip_suffix(')'))
                .ok_or(LayoutError::Parse("expected `(layout)` after the swizzle"))?;
            let layout = Self::parse_with(inner, names)?;
            if layout.swizzle.is_some() {
                return Err(LayoutError::Parse("a second swizzle"));
            }
            return Ok(layout.with_swizzle(swizzle));
        }
        let mut p = Parser {
            bytes: text.as_bytes(),
            at: 0,
            names,
        };
        let shape = p.tuple(1)?;
        p.expect(b':', "expected `:` between shape and stride")?;
        let stride = p.tuple(1)?;
        if p.at != p.bytes.len() {
            return Err(LayoutError::Parse("trailing text"));
        }
        Self::new(shape, stride)
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
    names: &'a [(&'a str, u32)],
}

impl Parser<'_> {
    fn expect(&mut self, c: u8, what: &'static str) -> Result<(), LayoutError> {
        if self.bytes.get(self.at) == Some(&c) {
            self.at += 1;
            Ok(())
        } else {
            Err(LayoutError::Parse(what))
        }
    }

    fn leaf(&mut self) -> Result<Dim, LayoutError> {
        let start = self.at;
        while self
            .bytes
            .get(self.at)
            .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == b'?')
        {
            self.at += 1;
        }
        let tok = std::str::from_utf8(&self.bytes[start..self.at])
            .map_err(|_| LayoutError::Parse("not text"))?;
        let canonical = |digits: &str| -> Result<u64, LayoutError> {
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                return Err(LayoutError::Parse("expected an integer"));
            }
            if digits.len() > 1 && digits.starts_with('0') {
                return Err(LayoutError::Parse("leading zero"));
            }
            digits.parse().map_err(|_| LayoutError::Overflow)
        };
        if let Some(rest) = tok.strip_prefix('?') {
            let n = canonical(rest)?;
            return u32::try_from(n)
                .map(Dim::Param)
                .map_err(|_| LayoutError::Overflow);
        }
        if tok.bytes().next().is_some_and(|b| b.is_ascii_digit()) {
            let n = canonical(tok)?;
            return i64::try_from(n)
                .map(Dim::Static)
                .map_err(|_| LayoutError::Overflow);
        }
        self.names
            .iter()
            .find(|(n, _)| *n == tok)
            .map(|(_, l)| Dim::Param(*l))
            .ok_or(LayoutError::Parse("an unknown parameter name"))
    }

    fn tuple(&mut self, level: usize) -> Result<DynTuple, LayoutError> {
        if self.bytes.get(self.at) != Some(&b'(') {
            return self.leaf().map(DynTuple::Leaf);
        }
        self.at += 1;
        if level > MAX_DEPTH {
            return Err(LayoutError::TooDeep);
        }
        if self.bytes.get(self.at) == Some(&b')') {
            return Err(LayoutError::EmptyTuple);
        }
        let mut items = vec![self.tuple(level + 1)?];
        while self.bytes.get(self.at) == Some(&b',') {
            self.at += 1;
            items.push(self.tuple(level + 1)?);
        }
        self.expect(b')', "expected `,` or `)`")?;
        Ok(DynTuple::Tuple(items))
    }
}

impl FromStr for DynLayout {
    type Err = LayoutError;

    fn from_str(text: &str) -> Result<Self, LayoutError> {
        Self::parse_with(text, &[])
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Static(v) => write!(f, "{v}"),
            Self::Param(p) => write!(f, "?{p}"),
        }
    }
}

impl fmt::Display for DynTuple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Leaf(d) => write!(f, "{d}"),
            Self::Tuple(t) => {
                f.write_str("(")?;
                for (i, x) in t.iter().enumerate() {
                    if i > 0 {
                        f.write_str(",")?;
                    }
                    write!(f, "{x}")?;
                }
                f.write_str(")")
            }
        }
    }
}

impl fmt::Display for DynLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.swizzle {
            Some(sw) => write!(f, "{sw}({}:{})", self.shape, self.stride),
            None => write!(f, "{}:{}", self.shape, self.stride),
        }
    }
}
