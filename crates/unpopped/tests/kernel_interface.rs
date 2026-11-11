// SPDX-License-Identifier: MIT OR Apache-2.0
//! `GeneratedKernel::interface` (PM ruling 2026-11-11 on fishstringer#17): what a consumer needs to launch a lowered
//! kernel. `KernelInterface::of` derives the launch, the parameters and the dynamic `Shared` allocations from the `KernelFn`;
//! the emitter-set fields are empty until a `with_*` builder sets them. The expected values are worked by hand.

use unpopped::GeneratedKernel;
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace, SubgroupSize};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, ParamKind};
use unpopped::kernel_interface::{
    Binding, BufferGuarantee, KernelInterface, LaunchBounds, ParamsBlock, ParamsBlockField,
    SpecConstant, SpecTarget,
};
use unpopped_vocab::ElementKind::{F16, F32, I32, U32};

/// `inp` and `out` buffers, scalar `n` (local 0) and scalar `m` (local 1), a `tile` (`F32`, sized by `n`) and a `halves` (`F16`,
/// sized by `m`) in `Shared`, a `scratch` (`F32`, sized by `n`) in `Private`, a `fixed` (`U32`, 16 elements) in `Shared`.
fn kernel() -> KernelFn {
    KernelFn::new(
        "dyn_smem",
        F32,
        vec![
            Param::buffer("inp", F32, BufferRole::Read, Some(1)),
            Param::scalar("n", I32, 0),
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(2)),
            Param::scalar("m", I32, 1),
        ],
        vec![LocalDecl::scalar(I32), LocalDecl::scalar(I32)],
        Block(vec![]),
    )
    .with_launch(LaunchShape::new([64, 2, 1], [8, 1, 1]))
    .with_allocs(vec![
        AllocDecl::new("tile", MemorySpace::Shared, F32, Extent::Param(0)),
        AllocDecl::new("fixed", MemorySpace::Shared, U32, Extent::Static(16)),
        AllocDecl::new("halves", MemorySpace::Shared, F16, Extent::Param(1)),
        AllocDecl::new("scratch", MemorySpace::Private, F32, Extent::Param(0)),
    ])
}

#[test]
fn the_core_derived_half_is_read_from_the_kernel() {
    let k = kernel();
    let i = KernelInterface::of(&k);
    assert_eq!(i.launch, k.launch);
    assert_eq!(i.params, k.params);
    // two dynamic Shared allocations, in allocation order: `tile` (alloc 0) sized by `n` (param 1) and `halves` (alloc 2) sized
    // by `m` (param 3); the static one and the Private one are not dynamic Shared memory
    let got: Vec<_> = i
        .dynamic_shared
        .iter()
        .map(|d| (d.alloc, d.name.as_str(), d.elem, d.size_param))
        .collect();
    assert_eq!(got, [(0, "tile", F32, 1), (2, "halves", F16, 3)]);
    assert!(matches!(
        i.params[i.dynamic_shared[1].size_param].kind,
        ParamKind::Scalar { local: 1, .. }
    ));
}

#[test]
fn a_kernel_with_no_launch_and_no_allocations_has_an_empty_interface() {
    let k = KernelFn::new("plain", F32, vec![], vec![], Block(vec![]));
    let i = KernelInterface::of(&k);
    assert_eq!(i.launch, None);
    assert!(i.params.is_empty() && i.dynamic_shared.is_empty());
    assert_eq!(i, KernelInterface::default());
}

#[test]
fn a_dynamic_extent_with_no_scalar_parameter_is_left_out_rather_than_guessed() {
    let k = KernelFn::new("orphan", F32, vec![], vec![], Block(vec![])).with_allocs(vec![
        AllocDecl::new("tile", MemorySpace::Shared, F32, Extent::Param(5)),
    ]);
    assert!(KernelInterface::of(&k).dynamic_shared.is_empty());
}

#[test]
fn the_emitter_set_half_is_empty_by_default() {
    let i = KernelInterface::of(&kernel());
    assert_eq!(i.entry_symbol, None);
    assert!(i.bindings.is_empty() && i.buffer_guarantees.is_empty());
    assert!(i.params_block.is_none() && i.launch_bounds.is_none() && i.max_regs.is_none());
    assert!(!i.cooperative);
    assert!(i.requires.is_empty() && i.compiler_args.is_empty());
    assert!(i.spec_constants.is_empty() && i.fidelity_notes.is_empty());
}

#[test]
fn every_emitter_set_field_is_set_by_its_builder() {
    let block = ParamsBlock::new(32, 16)
        .with_field(ParamsBlockField::new("n", 0, I32))
        .with_field(ParamsBlockField::new("grid_dim_x", 4, U32))
        .passed_by_address();
    let i = KernelInterface::of(&kernel())
        .with_entry_symbol("_Z8dyn_smemPfi")
        .with_binding(Binding::new(0, 0, 3))
        .with_binding(Binding::new(2, 1, 4))
        .with_params_block(block.clone())
        .with_buffer_guarantee(BufferGuarantee::new(0, Some(16), true))
        .with_launch_bounds(LaunchBounds::new(256, Some(2)))
        .with_max_regs(64)
        .cooperative()
        .with_requires("shaderInt64")
        .with_requires("16bit-arith")
        .with_compiler_arg("-denorm-mode-fp32")
        .with_compiler_arg("preserve")
        .with_spec_constant(SpecConstant::new(0, SpecTarget::BlockDim(0)))
        .with_spec_constant(SpecConstant::new(1, SpecTarget::DynamicShared(0)))
        .with_fidelity_note("an async copy is lowered synchronously");
    assert_eq!(i.entry_symbol.as_deref(), Some("_Z8dyn_smemPfi"));
    assert_eq!(
        i.bindings,
        [Binding::new(0, 0, 3), Binding::new(2, 1, 4)],
        "in the order set"
    );
    assert_eq!(i.params_block, Some(block));
    assert_eq!(
        i.buffer_guarantees,
        [BufferGuarantee::new(0, Some(16), true)]
    );
    assert_eq!(i.launch_bounds, Some(LaunchBounds::new(256, Some(2))));
    assert_eq!(i.max_regs, Some(64));
    assert!(i.cooperative);
    assert_eq!(i.requires, ["shaderInt64", "16bit-arith"]);
    assert_eq!(i.compiler_args, ["-denorm-mode-fp32", "preserve"]);
    assert_eq!(
        i.spec_constants,
        [
            SpecConstant::new(0, SpecTarget::BlockDim(0)),
            SpecConstant::new(1, SpecTarget::DynamicShared(0))
        ]
    );
    assert_eq!(i.fidelity_notes, ["an async copy is lowered synchronously"]);
    // the builders leave the core-derived half as it was
    assert_eq!(i.params, kernel().params);
    assert_eq!(i.dynamic_shared.len(), 2);
}

#[test]
fn a_generated_kernel_carries_its_interface_and_the_revision_does_not_see_it() {
    let bare = GeneratedKernel::new("k".into(), "src".into());
    assert!(bare.interface().is_none());
    let with = bare
        .clone()
        .with_interface(KernelInterface::of(&kernel()).with_requires("shaderInt64"));
    assert_eq!(with.interface().map(|i| i.requires.len()), Some(1));
    assert_eq!(with.revision_hash(), bare.revision_hash());
    // a subgroup requirement rides in the launch, whole
    let k = kernel().with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]));
    let mut shape = k.launch.unwrap();
    shape.subgroup = Some(SubgroupSize::new(32, 32));
    let i = KernelInterface::of(&k.with_launch(shape));
    assert_eq!(
        i.launch.and_then(|l| l.subgroup),
        Some(SubgroupSize::new(32, 32))
    );
}
