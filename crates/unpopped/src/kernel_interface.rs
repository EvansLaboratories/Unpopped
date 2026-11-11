// SPDX-License-Identifier: MIT OR Apache-2.0
//! What a consumer needs to **launch** a kernel a backend lowered with [`crate::Backend::lower_kernel`], as data
//! (`GeneratedKernel::interface`, PM ruling 2026-11-11 on fishstringer#17, with the field lists of Fishstringer, Vulkane and
//! Baracuda's launchers).
//!
//! Two halves. The **core-derived** half comes from the [`KernelFn`] alone, so every emitter agrees on it:
//! [`KernelInterface::of`] reads the launch geometry, the parameter list and the dynamic `Shared` allocations. The
//! **emitter-set** half is what only the emitter knows: backend-assigned binding slots, the layout of a parameter block,
//! the capabilities and compiler arguments the emitted source needs, launch bounds, the entry symbol. It is empty by
//! default and set through the `with_*` builders, so a backend that sets nothing is unchanged. Nothing here is part of
//! the kernel's revision hash: the emitter-set fields follow from the source text the emitter wrote.

use crate::exec::{Extent, LaunchShape, MemorySpace};
use crate::kernel::{KernelFn, Param, ParamKind};
use unpopped_vocab::ElementKind;

/// A `Shared` allocation whose element count is a launch argument (`Extent::Param`: CUDA `extern __shared__`, sized by the
/// launch). The size in bytes is `size_param`'s value times the element size; a consumer that needs more than the target's
/// default dynamic limit raises it before the launch (CUDA: `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DynamicShared {
    /// Index into [`KernelFn::allocs`].
    pub alloc: u32,
    /// The allocation's name.
    pub name: String,
    /// Its element type.
    pub elem: ElementKind,
    /// Index into [`KernelInterface::params`] of the scalar parameter that gives the element count.
    pub size_param: usize,
}

/// A binding slot the emitter assigned to a parameter (a Vulkan set and binding, an HLSL register space and register).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Binding {
    /// Index into [`KernelInterface::params`].
    pub param: usize,
    /// The descriptor set or register space.
    pub space: u32,
    /// The binding or register number.
    pub slot: u32,
}

impl Binding {
    /// Parameter `param` is bound at `slot` of `space`.
    #[must_use]
    pub const fn new(param: usize, space: u32, slot: u32) -> Self {
        Self { param, space, slot }
    }
}

/// One field of the parameter block the emitter packs the scalars, the grid dimensions and each buffer's shape, strides and
/// base offset into.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ParamsBlockField {
    /// The field's name.
    pub name: String,
    /// Its byte offset in the block.
    pub offset: u32,
    /// Its element type.
    pub ty: ElementKind,
}

impl ParamsBlockField {
    /// A field `name` of type `ty` at byte `offset`.
    #[must_use]
    pub fn new(name: impl Into<String>, offset: u32, ty: ElementKind) -> Self {
        Self {
            name: name.into(),
            offset,
            ty,
        }
    }
}

/// The layout of the parameter block: its fields, total size and alignment, and whether it is passed by address
/// (a CUDA `__grid_constant__` parameter) rather than copied by value.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ParamsBlock {
    /// The fields in offset order.
    pub fields: Vec<ParamsBlockField>,
    /// Total size in bytes.
    pub size: u32,
    /// Alignment in bytes.
    pub align: u32,
    /// Passed by address of the argument (`__grid_constant__`) instead of by value.
    pub by_address: bool,
}

impl ParamsBlock {
    /// An empty block of `size` bytes aligned to `align`.
    #[must_use]
    pub fn new(size: u32, align: u32) -> Self {
        Self {
            fields: Vec::new(),
            size,
            align,
            by_address: false,
        }
    }

    /// The block with `field` appended.
    #[must_use]
    pub fn with_field(mut self, field: ParamsBlockField) -> Self {
        self.fields.push(field);
        self
    }

    /// The block passed by address (`__grid_constant__`).
    #[must_use]
    pub fn passed_by_address(mut self) -> Self {
        self.by_address = true;
        self
    }
}

/// What the emitter guarantees about a buffer parameter, beyond its [`crate::kernel::BufferRole`]: the launcher asserts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct BufferGuarantee {
    /// Index into [`KernelInterface::params`].
    pub param: usize,
    /// The byte alignment the buffer's base must have, if the emitted code relies on one.
    pub align_bytes: Option<u32>,
    /// The buffer does not alias another parameter (`__restrict__`, `noalias`).
    pub restrict: bool,
}

impl BufferGuarantee {
    /// Parameter `param` needs `align_bytes` alignment (if `Some`) and is `restrict` (if true).
    #[must_use]
    pub const fn new(param: usize, align_bytes: Option<u32>, restrict: bool) -> Self {
        Self {
            param,
            align_bytes,
            restrict,
        }
    }
}

/// Occupancy bounds the emitted kernel was compiled with (`__launch_bounds__`): they change the legal block size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct LaunchBounds {
    /// The most threads per block a launch may use.
    pub max_threads: u32,
    /// The fewest blocks per multiprocessor the compiler must leave room for, if stated.
    pub min_blocks_per_sm: Option<u32>,
}

impl LaunchBounds {
    /// At most `max_threads` threads per block and, if `Some`, at least `min_blocks_per_sm` blocks per multiprocessor.
    #[must_use]
    pub const fn new(max_threads: u32, min_blocks_per_sm: Option<u32>) -> Self {
        Self {
            max_threads,
            min_blocks_per_sm,
        }
    }
}

/// What a specialization constant stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpecTarget {
    /// The block size on axis `.0` (0, 1 or 2): its value in [`KernelInterface::launch`] is the default.
    BlockDim(u8),
    /// The element count of the dynamic `Shared` allocation at index `.0` of [`KernelInterface::dynamic_shared`].
    DynamicShared(usize),
}

/// A value the emitter left as a specialization constant instead of baking it into the source (a SPIR-V `SpecId`, an HLSL
/// or Slang specialization parameter): the host supplies it when it builds the pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct SpecConstant {
    /// The constant's id.
    pub id: u32,
    /// What it stands for.
    pub target: SpecTarget,
}

impl SpecConstant {
    /// Specialization constant `id` stands for `target`.
    #[must_use]
    pub const fn new(id: u32, target: SpecTarget) -> Self {
        Self { id, target }
    }
}

/// The launch interface of a lowered kernel. See the module docs for the two halves.
#[derive(Clone, Debug, PartialEq, Default)]
#[non_exhaustive]
pub struct KernelInterface {
    /// The kernel's launch geometry (`None`: a kernel with no launch shape).
    pub launch: Option<LaunchShape>,
    /// The parameters in declaration order: buffers with element type, role and rank, scalars with type and local.
    pub params: Vec<Param>,
    /// The `Shared` allocations sized by a launch argument.
    pub dynamic_shared: Vec<DynamicShared>,
    /// The symbol to link or look up, when it differs from `GeneratedKernel::name` (a templated instantiation's mangled name).
    pub entry_symbol: Option<String>,
    /// Backend-assigned binding slots, by parameter.
    pub bindings: Vec<Binding>,
    /// The layout of the parameter block.
    pub params_block: Option<ParamsBlock>,
    /// What the emitted code relies on about a buffer: alignment, no aliasing.
    pub buffer_guarantees: Vec<BufferGuarantee>,
    /// The occupancy bounds the kernel was compiled with.
    pub launch_bounds: Option<LaunchBounds>,
    /// The most registers per thread the kernel was compiled with (`__maxnreg__`).
    pub max_regs: Option<u32>,
    /// The kernel must be launched cooperatively (`cudaLaunchCooperativeKernel`).
    pub cooperative: bool,
    /// Capabilities the emitted source needs (`shaderInt64`, 16-bit arithmetic, float or 64-bit atomics, a subgroup size),
    /// by name, so a consumer can decline before it builds a pipeline.
    pub requires: Vec<String>,
    /// Compiler arguments the emitted source needs (`-denorm-mode-fp32 preserve`).
    pub compiler_args: Vec<String>,
    /// The values left as specialization constants rather than baked in.
    pub spec_constants: Vec<SpecConstant>,
    /// Where the emitted kernel differs from the kernel's semantics in a way the source does not show (an async copy lowered
    /// synchronously, a barrier widened), one sentence each, so a consumer can decline or warn.
    pub fidelity_notes: Vec<String>,
}

impl KernelInterface {
    /// The core-derived half for `kernel`: its launch geometry, its parameters, and the `Shared` allocations a scalar
    /// parameter sizes. The emitter-set half is empty.
    #[must_use]
    pub fn of(kernel: &KernelFn) -> Self {
        let size_param = |local: u32| {
            kernel
                .params
                .iter()
                .position(|p| matches!(p.kind, ParamKind::Scalar { local: l, .. } if l == local))
        };
        let dynamic_shared = kernel
            .allocs
            .iter()
            .enumerate()
            .filter(|(_, a)| a.space == MemorySpace::Shared)
            .filter_map(|(i, a)| match a.extent {
                Extent::Param(local) => Some(DynamicShared {
                    alloc: u32::try_from(i).unwrap_or(u32::MAX),
                    name: a.name.clone(),
                    elem: a.elem,
                    size_param: size_param(local)?,
                }),
                _ => None,
            })
            .collect();
        Self {
            launch: kernel.launch,
            params: kernel.params.clone(),
            dynamic_shared,
            ..Self::default()
        }
    }

    /// The interface with the symbol to link or look up.
    #[must_use]
    pub fn with_entry_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.entry_symbol = Some(symbol.into());
        self
    }

    /// The interface with `binding` appended.
    #[must_use]
    pub fn with_binding(mut self, binding: Binding) -> Self {
        self.bindings.push(binding);
        self
    }

    /// The interface with the parameter block's layout.
    #[must_use]
    pub fn with_params_block(mut self, block: ParamsBlock) -> Self {
        self.params_block = Some(block);
        self
    }

    /// The interface with `guarantee` appended.
    #[must_use]
    pub fn with_buffer_guarantee(mut self, guarantee: BufferGuarantee) -> Self {
        self.buffer_guarantees.push(guarantee);
        self
    }

    /// The interface with its launch bounds.
    #[must_use]
    pub fn with_launch_bounds(mut self, bounds: LaunchBounds) -> Self {
        self.launch_bounds = Some(bounds);
        self
    }

    /// The interface with its register cap.
    #[must_use]
    pub fn with_max_regs(mut self, regs: u32) -> Self {
        self.max_regs = Some(regs);
        self
    }

    /// The interface marked as needing a cooperative launch.
    #[must_use]
    pub fn cooperative(mut self) -> Self {
        self.cooperative = true;
        self
    }

    /// The interface with the capability `name` appended to what the source needs.
    #[must_use]
    pub fn with_requires(mut self, name: impl Into<String>) -> Self {
        self.requires.push(name.into());
        self
    }

    /// The interface with the compiler argument `arg` appended.
    #[must_use]
    pub fn with_compiler_arg(mut self, arg: impl Into<String>) -> Self {
        self.compiler_args.push(arg.into());
        self
    }

    /// The interface with `constant` appended to the specialization constants.
    #[must_use]
    pub fn with_spec_constant(mut self, constant: SpecConstant) -> Self {
        self.spec_constants.push(constant);
        self
    }

    /// The interface with the fidelity note `note` appended.
    #[must_use]
    pub fn with_fidelity_note(mut self, note: impl Into<String>) -> Self {
        self.fidelity_notes.push(note.into());
        self
    }
}
