// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Place::Raw` in the interpreter (ROADMAP U9c, `docs/options-u9c-pointer-values.md`): a typed access at a **byte** offset
//! into a buffer's storage. Byte 0 is the view's first element (`base_offset` elements into the storage); shape and strides are
//! not consulted. A misaligned or out-of-range access is a typed trap; the race detector sees every buffer element the bytes
//! touch (conservative: two threads writing different bytes of one wider element are reported).

use super::kernel_eval::Machine;
use super::kernel_mem::AccessKind;
use super::kernel_trap::KernelTrap;
use super::{Val, store_val};
use crate::exec::MemorySpace;
use crate::ir::ScalarExpr;
use crate::kernel::Place;
use crate::oracle::kernel_eval::KernelArg;
use crate::types::{self, Ty};
use unpopped_vocab::ElementKind;

/// Where a checked raw access lands.
pub(super) struct RawAt {
    /// The root argument index of the buffer.
    pub(super) root: usize,
    /// The absolute byte position in the buffer's storage.
    pub(super) at: usize,
    /// The access type.
    pub(super) elem: ElementKind,
    /// Its size in bytes.
    pub(super) size: usize,
    /// How many consecutive `elem`-sized values the access spans (1 except for an async copy's source).
    pub(super) count: usize,
}

impl Machine<'_> {
    /// The evaluated byte offset of a raw place and the size in bytes of its access type, the offset checked for alignment: a
    /// negative or misaligned offset is `MisalignedAccess` against `buffer` (a parameter or allocation index).
    fn raw_offset(
        &self,
        (buffer, byte_offset, elem): (u32, &ScalarExpr, ElementKind),
    ) -> Result<(i128, usize), KernelTrap> {
        let Ty::Scalar(t) = types::infer_ty_in(byte_offset, self.k.compute, &self.local_tys)
            .map_err(|_| KernelTrap::Unsupported("byte offset type"))?
        else {
            return Err(KernelTrap::Unsupported("packed byte offset"));
        };
        let off = self.eval_int(byte_offset, t)?;
        let size = types::raw_access_bytes(elem).ok_or(KernelTrap::Unsupported(
            "a Raw access of a non-byte element",
        ))? as usize;
        if off < 0 || off % size as i128 != 0 {
            return Err(KernelTrap::MisalignedAccess {
                buffer,
                byte_offset: off,
            });
        }
        Ok((off, size))
    }

    /// Evaluate, align- and bounds-check a `Place::Raw`, and log the access for the race detector.
    pub(super) fn raw_at(
        &mut self,
        place: &Place,
        kind: AccessKind,
        new: Option<[u8; 16]>,
    ) -> Result<RawAt, KernelTrap> {
        self.raw_span(place, kind, 1, new)
    }

    /// [`Self::raw_at`] for `count` consecutive values (the source of an async copy): aligned to one value, bounds-checked
    /// and logged over all `count`.
    pub(super) fn raw_span(
        &mut self,
        place: &Place,
        kind: AccessKind,
        count: usize,
        new: Option<[u8; 16]>,
    ) -> Result<RawAt, KernelTrap> {
        let Place::Raw {
            buffer,
            byte_offset,
            elem,
        } = place
        else {
            unreachable!("raw_at is called on a Raw place");
        };
        let (off, size) = self.raw_offset((*buffer, byte_offset, *elem))?;
        let root = self.root(*buffer);
        let KernelArg::Buffer(tb) = &self.args[root] else {
            return Err(KernelTrap::ArgMismatch(
                "a Place names a non-buffer argument",
            ));
        };
        if types::elem_bits(tb.dtype) % 8 != 0 {
            return Err(KernelTrap::Unsupported("a Raw access to a sub-byte buffer"));
        }
        let esz = types::elem_size(tb.dtype);
        let oob = KernelTrap::OutOfBounds { buffer: *buffer };
        let at = usize::try_from(off)
            .ok()
            .and_then(|o| o.checked_add(usize::try_from(tb.base_offset).ok()?.checked_mul(esz)?))
            .ok_or_else(|| oob.clone())?;
        let span = size * count;
        if span == 0 || at.checked_add(span).is_none_or(|end| end > tb.bytes.len()) {
            return Err(oob);
        }
        // a store of the very bytes already there: a benign race when the launch allows it (`Leniency::same_value_writes`)
        let same = new.is_some_and(|nb| tb.bytes[at..at + span] == nb[..span]);
        for e in at / esz..=(at + span - 1) / esz {
            self.note_global_write(root, e, kind, same)?;
        }
        Ok(RawAt {
            root,
            at,
            elem: *elem,
            size,
            count,
        })
    }

    /// Evaluate, align- and bounds-check a `Place::AllocRaw`; log every allocation element the bytes touch. Returns the
    /// allocation, its space, the byte position inside it, the access type and its size.
    fn alloc_raw_at(
        &mut self,
        place: &Place,
        kind: AccessKind,
        new: Option<[u8; 16]>,
    ) -> Result<(u32, MemorySpace, usize, ElementKind, usize), KernelTrap> {
        let Place::AllocRaw {
            alloc,
            byte_offset,
            elem,
        } = place
        else {
            unreachable!("alloc_raw_at is called on an AllocRaw place");
        };
        let space = self.alloc_k.allocs[*alloc as usize].space;
        let (off, size) = self.raw_offset((*alloc, byte_offset, *elem))?;
        let c = self
            .ctx
            .as_mut()
            .ok_or(KernelTrap::Unsupported("an allocation outside run_launch"))?;
        let m = c.mem(*alloc, space).ok_or(KernelTrap::Unsupported(
            "an AllocRaw on a space with no instance",
        ))?;
        let (esz, total) = (
            types::elem_size(m.dtype),
            m.init.len() * types::elem_size(m.dtype),
        );
        let at = usize::try_from(off)
            .ok()
            .filter(|o| o.checked_add(size).is_some_and(|end| end <= total))
            .ok_or(KernelTrap::AllocOutOfBounds { alloc: *alloc })?;
        // a store of the very bytes the span already holds (every element already stored): a benign race when allowed
        let same = new.is_some_and(|nb| {
            (at / esz..=(at + size - 1) / esz).all(|e| m.init[e])
                && m.bytes[at..at + size] == nb[..size]
        });
        for e in at / esz..=(at + size - 1) / esz {
            if c.peer.is_none() && c.asyncq.covers(*alloc, e) {
                return Err(KernelTrap::AsyncRace { alloc: *alloc });
            }
            c.log_shared_write(*alloc, e, space, kind, same)?;
        }
        Ok((*alloc, space, at, *elem, size))
    }

    /// A raw load from an allocation: every element the bytes touch must already be fully stored.
    pub(super) fn alloc_raw_load(&mut self, place: &Place) -> Result<Val, KernelTrap> {
        let (alloc, space, at, elem, size) = self.alloc_raw_at(place, AccessKind::Read, None)?;
        let c = self
            .ctx
            .as_mut()
            .expect("alloc_raw_at found a launch context");
        let m = c.mem(alloc, space).expect("the space holds an instance");
        let esz = types::elem_size(m.dtype);
        if (at / esz..=(at + size - 1) / esz).any(|e| !m.init[e]) {
            return Err(if space == MemorySpace::Private {
                KernelTrap::UninitializedPrivate { alloc }
            } else {
                KernelTrap::UninitializedShared { alloc }
            });
        }
        let mut bits = 0u128;
        for (i, b) in m.bytes[at..at + size].iter().enumerate() {
            bits |= u128::from(*b) << (8 * i);
        }
        Ok(Val::Raw(bits, elem))
    }

    /// A raw store into an allocation: the bytes land little-endian; an element counts as stored only when the store covers all of it.
    pub(super) fn alloc_raw_store(&mut self, place: &Place, v: Val) -> Result<(), KernelTrap> {
        let Place::AllocRaw { elem: stored, .. } = place else {
            unreachable!("alloc_raw_store is called on an AllocRaw place");
        };
        let mut tmp = [0u8; 16];
        store_val(v, *stored, &mut tmp, 0);
        let (alloc, space, at, elem, size) =
            self.alloc_raw_at(place, AccessKind::Write, Some(tmp))?;
        let c = self
            .ctx
            .as_mut()
            .expect("alloc_raw_at found a launch context");
        let m = c
            .mem_mut(alloc, space)
            .expect("the space holds an instance");
        m.bytes[at..at + size].copy_from_slice(&tmp[..size]);
        let esz = types::elem_size(m.dtype);
        for e in at / esz..=(at + size - 1) / esz {
            if at <= e * esz && (e + 1) * esz <= at + size {
                m.init[e] = true;
            }
        }
        Ok(())
    }

    /// `n` consecutive raw values from `src`, read now (the source of an async copy), each as `elem` bits.
    pub(super) fn raw_read_n(&mut self, src: &Place, n: usize) -> Result<Vec<Val>, KernelTrap> {
        let a = self.raw_span(src, AccessKind::Read, n, None)?;
        let KernelArg::Buffer(tb) = &self.args[a.root] else {
            unreachable!("raw_span returned a buffer");
        };
        Ok(tb.bytes[a.at..a.at + a.size * a.count]
            .chunks(a.size)
            .map(|c| {
                let bits = c
                    .iter()
                    .enumerate()
                    .fold(0u128, |acc, (i, b)| acc | u128::from(*b) << (8 * i));
                Val::Raw(bits, a.elem)
            })
            .collect())
    }

    /// The value of a raw load: the stored bytes, little-endian, as `elem` bits.
    pub(super) fn raw_load(&mut self, place: &Place) -> Result<Val, KernelTrap> {
        let a = self.raw_at(place, AccessKind::Read, None)?;
        let KernelArg::Buffer(tb) = &self.args[a.root] else {
            unreachable!("raw_at returned a buffer");
        };
        let mut bits = 0u128;
        for (i, b) in tb.bytes[a.at..a.at + a.size].iter().enumerate() {
            bits |= u128::from(*b) << (8 * i);
        }
        Ok(Val::Raw(bits, a.elem))
    }

    /// A raw store: `v` encoded as `elem` (verbatim when it already is one), written little-endian at the byte position.
    pub(super) fn raw_store(&mut self, place: &Place, v: Val) -> Result<(), KernelTrap> {
        let Place::Raw { elem: stored, .. } = place else {
            unreachable!("raw_store is called on a Raw place");
        };
        let mut tmp = [0u8; 16];
        store_val(v, *stored, &mut tmp, 0);
        let a = self.raw_at(place, AccessKind::Write, Some(tmp))?;
        let KernelArg::Buffer(tb) = &mut self.args[a.root] else {
            unreachable!("raw_at returned a buffer");
        };
        tb.bytes[a.at..a.at + a.size].copy_from_slice(&tmp[..a.size]);
        Ok(())
    }
}
