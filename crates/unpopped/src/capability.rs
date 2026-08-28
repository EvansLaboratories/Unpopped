//! What a target *can do*, as the numbers a schedule chooser needs.
//!
//! # Why this exists
//!
//! Unpopped **chooses nothing** hardware-dependent today. A caller hands
//! [`crate::generate`] a finished [`unpopped_vocab::StructureKey`] — `vec_width` included — and
//! the plan gate honours it. That is the right default (the key names the
//! *request*, and a key that named one particular answer could not be compared
//! against another), but it means a caller who does not already know a good
//! `vec_width` for their device has nowhere to ask.
//!
//! This module is the asking. It carries **no policy** — it does not decide a
//! schedule, it states what the hardware permits, so that a *variant* chooser
//! can enumerate what is legal and a measurement can pick.
//!
//! # Where candidates actually live, corrected by measurement
//!
//! A first design had this feeding a generator of alternative
//! [`unpopped_vocab::StructureKey`]s. **That is not expressible, and the reason is load-bearing
//! rather than incidental:** every field of a key is *derived* from
//! `(op, operands, target)` — `idx` from the largest offset, `work` from
//! `frame_work_class`, and `vec_width` from the operand's own alignment and
//! extent via `classify_vec_width`. **A key is a pure function of the request,
//! so two candidates for one request cannot differ in it.**
//!
//! That is the identity property working as intended — the key names the
//! request, which is exactly what lets N kernels compete inside one cell — but
//! it means variation belongs somewhere else. **That somewhere is
//! [`crate::backend::Backend::lower_variants`]**, which already returns a
//! `Vec<Variant>` carrying a `tag`, its kernels, and a
//! [`crate::backend::VariantFidelity`]. Candidates share a `structure_key` and
//! differ by revision hash, which is precisely the shape a consumer's profiler
//! keys siblings on.
//!
//! So: capabilities inform **which variants are worth emitting**, not which key
//! to build.
//!
//! # Two sources, and only one of them is authoritative
//!
//! **Per-device facts must be queried.** `multiprocessor_count`, L2 size and
//! clocks vary between parts that share a compute capability — an RTX 4070 and a
//! 4090 are both `sm_89` with *identical* per-SM limits and very different SM
//! counts. A caller holding a live device should build a `TargetCapabilities`
//! from its own driver query ([`TargetCapabilities::from_queried`]); through
//! `baracuda-driver` that is `Device::attribute`, which is public and generic
//! over `CUdevice_attribute`, so every field below is reachable.
//!
//! **The static table is for ahead-of-time generation only** — building a kernel
//! for a device that is not in the machine. [`cuda_capabilities`] is transcribed
//! from NVIDIA's *Technical Specifications per Compute Capability* table, and
//! transcription is exactly the step that goes stale: a queried value always
//! wins over a row here, and the row exists so that a cross-compile is possible
//! at all, not because it is a better answer.
//!
//! **An unknown compute capability returns `None`, never a nearby row.** A
//! future architecture that is not in this table is a `None` the caller must
//! handle, because a plausible-looking wrong row is the failure mode that does
//! not announce itself — a tile sized against the wrong shared-memory limit
//! still compiles, still runs, and is merely slower or illegal.

use unpopped_vocab::{TargetId, VecWidth};

/// The resource limits a schedule chooser needs, for one target.
///
/// `#[non_exhaustive]`: this grows as choosers learn to use more of the machine
/// (L2 size, memory bandwidth, tensor-core shapes), and a caller must not be
/// broken by a field it does not read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct TargetCapabilities {
    /// Threads in one block / workgroup.
    pub max_threads_per_block: u32,
    /// Bytes of shared / group memory one **block** may hold. On CUDA this is
    /// the opt-in maximum (`cudaFuncSetAttribute`), not the 48 KiB default.
    pub max_shared_mem_per_block: u32,
    /// Bytes of shared memory per SM — the occupancy denominator, and larger
    /// than the per-block figure on every architecture that has both.
    pub max_shared_mem_per_sm: u32,
    /// 32-bit registers per SM.
    pub regs_per_sm: u32,
    /// Registers one thread may use before spilling.
    pub max_regs_per_thread: u32,
    /// Warp / subgroup width.
    pub warp_size: u32,
    /// Resident warps per SM at full occupancy.
    pub max_warps_per_sm: u32,
    /// Resident blocks per SM.
    pub max_blocks_per_sm: u32,
    /// Widest single vector access, in bytes (CUDA's `float4` is 16).
    pub max_vector_bytes: u32,
    /// SM count — **per device, not per compute capability**. `None` when only
    /// the capability is known, which is why it is not a bare `u32`: a
    /// cross-compile genuinely does not know it, and defaulting would invent a
    /// number that an occupancy calculation would then trust.
    pub multiprocessor_count: Option<u32>,
}

impl TargetCapabilities {
    /// Build from values a caller queried off a live device.
    ///
    /// This is the authoritative path. Every argument corresponds to a driver
    /// attribute rather than to a table row, so it stays correct for parts that
    /// did not exist when this crate was written.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn from_queried(
        max_threads_per_block: u32,
        max_shared_mem_per_block: u32,
        max_shared_mem_per_sm: u32,
        regs_per_sm: u32,
        max_regs_per_thread: u32,
        warp_size: u32,
        max_warps_per_sm: u32,
        max_blocks_per_sm: u32,
        max_vector_bytes: u32,
        multiprocessor_count: Option<u32>,
    ) -> Self {
        Self {
            max_threads_per_block,
            max_shared_mem_per_block,
            max_shared_mem_per_sm,
            regs_per_sm,
            max_regs_per_thread,
            warp_size,
            max_warps_per_sm,
            max_blocks_per_sm,
            max_vector_bytes,
            multiprocessor_count,
        }
    }

    /// The widest [`VecWidth`] legal for `elem_bytes`-sized elements here.
    ///
    /// Legality only — a wider access being *permitted* says nothing about it
    /// being *faster*, which is a measurement rather than a property. Returns
    /// `Scalar` when even `V2` would exceed [`Self::max_vector_bytes`].
    #[must_use]
    pub fn widest_vec_width(&self, elem_bytes: u32) -> VecWidth {
        if elem_bytes == 0 {
            return VecWidth::Scalar;
        }
        let fits = |lanes: u32| elem_bytes.saturating_mul(lanes) <= self.max_vector_bytes;
        if fits(8) {
            VecWidth::V8
        } else if fits(4) {
            VecWidth::V4
        } else if fits(2) {
            VecWidth::V2
        } else {
            VecWidth::Scalar
        }
    }

    /// Blocks resident per SM at `threads_per_block`, ignoring registers and
    /// shared memory.
    ///
    /// **This is a ceiling, not an occupancy figure.** Real occupancy also
    /// depends on registers per thread, which `ptxas` decides and does not
    /// publish — so it is not computable before emission, only measurable after
    /// (`ptxas -v`). A chooser should use this to discard the clearly illegal
    /// and leave the ranking to measurement.
    #[must_use]
    pub fn block_ceiling(&self, threads_per_block: u32) -> u32 {
        if threads_per_block == 0 || self.warp_size == 0 {
            return 0;
        }
        let warps = threads_per_block.div_ceil(self.warp_size);
        if warps == 0 {
            return 0;
        }
        (self.max_warps_per_sm / warps).min(self.max_blocks_per_sm)
    }
}

/// Per-compute-capability limits for CUDA, transcribed from NVIDIA's
/// *Technical Specifications per Compute Capability*.
///
/// **Keyed by capability, not by part.** Every `sm_89` device has these
/// per-SM limits whatever its SM count, which is why `multiprocessor_count` is
/// `None` here and must be queried.
///
/// Returns `None` for a capability this table does not carry. **Deliberately
/// not "the nearest row"**: Blackwell and later are absent because their figures
/// are not transcribed here, and answering with an Ada row would be a wrong
/// answer wearing a right one's clothes.
#[must_use]
pub fn cuda_capabilities(major: u32, minor: u32) -> Option<TargetCapabilities> {
    // smem_block, smem_sm, warps_sm, blocks_sm
    let (smem_block, smem_sm, warps_sm, blocks_sm) = match (major, minor) {
        (7, 0) | (7, 2) => (98_304, 98_304, 64, 32),
        (7, 5) => (65_536, 65_536, 32, 16),
        (8, 0) => (166_912, 167_936, 64, 32),
        (8, 6) => (101_376, 102_400, 48, 16),
        (8, 7) => (166_912, 167_936, 48, 16),
        (8, 9) => (101_376, 102_400, 48, 24),
        (9, 0) => (232_448, 233_472, 64, 32),
        _ => return None,
    };
    Some(TargetCapabilities {
        max_threads_per_block: 1024,
        max_shared_mem_per_block: smem_block,
        max_shared_mem_per_sm: smem_sm,
        regs_per_sm: 65_536,
        max_regs_per_thread: 255,
        warp_size: 32,
        max_warps_per_sm: warps_sm,
        max_blocks_per_sm: blocks_sm,
        max_vector_bytes: 16,
        multiprocessor_count: None,
    })
}

/// Capabilities for a `cuda:sm<NN>` target token, when the table carries it.
///
/// Parses the capability set as NVIDIA's two-or-three digit `sm` form — `sm80`,
/// `sm90`, `sm90a` — and looks it up. Any other namespace returns `None`: a
/// `vulkan:` token's limits come from `VkPhysicalDeviceLimits`, not from here,
/// and inventing them from a CUDA table would be the same wrong-row error one
/// vendor over.
#[must_use]
pub fn capabilities_for(target: TargetId) -> Option<TargetCapabilities> {
    let token = target.as_str();
    let (ns, cap) = token.split_once(':')?;
    if ns != "cuda" {
        return None;
    }
    let digits = cap.strip_prefix("sm")?;
    // `sm90a` and friends: the trailing letter is an architecture-specific
    // feature flag, not a different resource envelope.
    let digits: String = digits.chars().take_while(char::is_ascii_digit).collect();
    let (major, minor) = match digits.len() {
        2 => (digits[0..1].parse().ok()?, digits[1..2].parse().ok()?),
        3 => (digits[0..2].parse().ok()?, digits[2..3].parse().ok()?),
        _ => return None,
    };
    cuda_capabilities(major, minor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use unpopped_vocab::ArchSku;

    /// Rows differ between capabilities.
    ///
    /// The failure this catches is a table that compiles and looks populated
    /// while every arm returns the same tuple — a copy-paste that a
    /// "does sm89 return Some?" test passes happily. Turing's 64 KiB and
    /// Hopper's 227 KiB are the widest spread in the table.
    #[test]
    fn the_table_discriminates_between_capabilities() {
        let t75 = cuda_capabilities(7, 5).expect("sm75 is in the table");
        let t90 = cuda_capabilities(9, 0).expect("sm90 is in the table");
        assert_ne!(
            t75.max_shared_mem_per_block, t90.max_shared_mem_per_block,
            "Turing and Hopper returned the same shared-memory limit"
        );
        assert_ne!(t75.max_warps_per_sm, t90.max_warps_per_sm);
        // Ada and Ampere-consumer share warps/SM but not blocks/SM: a table
        // keyed too coarsely would collapse them.
        let t86 = cuda_capabilities(8, 6).expect("sm86");
        let t89 = cuda_capabilities(8, 9).expect("sm89");
        assert_eq!(t86.max_warps_per_sm, t89.max_warps_per_sm);
        assert_ne!(
            t86.max_blocks_per_sm, t89.max_blocks_per_sm,
            "sm86 and sm89 differ in blocks/SM and the table lost it"
        );
    }

    /// An unknown capability is `None`, never the nearest row.
    ///
    /// This is the assertion the whole table rests on. A wrong row does not
    /// announce itself: a tile sized against another architecture's shared
    /// memory still compiles and still runs.
    #[test]
    fn an_unknown_capability_is_none_rather_than_a_neighbour() {
        assert!(
            cuda_capabilities(10, 0).is_none(),
            "Blackwell is not transcribed"
        );
        assert!(cuda_capabilities(12, 0).is_none());
        assert!(
            cuda_capabilities(6, 1).is_none(),
            "Pascal is not transcribed"
        );
        assert!(cuda_capabilities(8, 5).is_none(), "8.5 does not exist");
    }

    #[test]
    fn a_cuda_target_token_resolves_and_other_namespaces_do_not() {
        let sm89 = capabilities_for(ArchSku::Sm89.into()).expect("sm89 token resolves");
        assert_eq!(sm89, cuda_capabilities(8, 9).unwrap());
        assert_eq!(
            capabilities_for(ArchSku::Sm90.into()),
            cuda_capabilities(9, 0),
            "three-digit sm90 must parse as 9.0, not 90.x"
        );
        // `sm90a`: the trailing feature letter is not a different envelope.
        assert_eq!(
            capabilities_for(ArchSku::Sm90a.into()),
            cuda_capabilities(9, 0)
        );
    }

    /// A `vulkan:` token gets `None`, not a CUDA row.
    ///
    /// Its limits come from `VkPhysicalDeviceLimits`. Answering from a CUDA
    /// table would be the wrong-row error one vendor over, and it is the exact
    /// mistake a `strip_prefix("sm")` that ignored the namespace would make.
    #[test]
    fn a_non_cuda_namespace_is_refused() {
        let vk = TargetId::parse("vulkan:sg32.arith-f16").expect("a valid vulkan token");
        assert!(
            capabilities_for(vk).is_none(),
            "a vulkan target was answered from the CUDA table"
        );
    }

    #[test]
    fn vector_width_is_capped_by_the_targets_widest_access() {
        let c = cuda_capabilities(8, 9).unwrap();
        assert_eq!(c.max_vector_bytes, 16);
        assert_eq!(c.widest_vec_width(2), VecWidth::V8, "f16: 8 x 2 = 16");
        assert_eq!(c.widest_vec_width(4), VecWidth::V4, "f32: 4 x 4 = 16");
        assert_eq!(c.widest_vec_width(8), VecWidth::V2, "f64: 2 x 8 = 16");
        assert_eq!(
            c.widest_vec_width(16),
            VecWidth::Scalar,
            "even V2 overflows"
        );
        assert_eq!(c.widest_vec_width(0), VecWidth::Scalar, "no divide by zero");
    }

    /// The ceiling is bounded by BOTH limits, and by the tighter of the two.
    ///
    /// A version that returned only `max_warps_per_sm / warps` passes at 256
    /// threads and is wrong at 32, where the blocks/SM cap binds instead.
    #[test]
    fn the_block_ceiling_honours_whichever_limit_binds() {
        let c = cuda_capabilities(8, 9).unwrap(); // 48 warps/SM, 24 blocks/SM
        assert_eq!(c.block_ceiling(256), 6, "8 warps each -> 48/8");
        assert_eq!(
            c.block_ceiling(32),
            24,
            "1 warp each would allow 48 blocks, but blocks/SM caps at 24"
        );
        assert_eq!(c.block_ceiling(1024), 1, "32 warps each -> 48/32 = 1");
        assert_eq!(c.block_ceiling(0), 0, "no divide by zero");
    }
}
