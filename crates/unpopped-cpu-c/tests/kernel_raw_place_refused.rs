// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C refuses a `Place::Raw` and a `Place::AllocRaw`, by name** (ROADMAP U9c): the C ABI hands a kernel a buffer's
//! base, shape and strides but not its storage length, so the byte-range trap the oracle raises cannot be emitted. The control is
//! the same kernel with an ordinary place, which lowers.

use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

const U32: ElementKind = ElementKind::U32;

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn load_from(place: Place) -> KernelFn {
    KernelFn::new(
        "raw",
        U32,
        vec![Param::buffer("in", U32, BufferRole::Read, Some(1))],
        vec![LocalDecl::scalar(U32)],
        Block(vec![Stmt::Load { local: 0, place }]),
    )
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]))
    .with_allocs(vec![AllocDecl::new(
        "s",
        MemorySpace::Shared,
        U32,
        Extent::Static(4),
    )])
}

#[test]
fn a_raw_place_is_refused_with_its_reason_and_the_control_lowers() {
    let raw = Place::Raw {
        buffer: 0,
        byte_offset: u(4),
        elem: U32,
    };
    let araw = Place::AllocRaw {
        alloc: 0,
        byte_offset: u(4),
        elem: U32,
    };
    for place in [raw, araw] {
        let err = CpuC
            .lower_kernel(&load_from(place))
            .expect_err("a raw place is refused");
        assert!(
            format!("{err:?}").contains("Place::Raw"),
            "the refusal names the place: {err:?}"
        );
    }
    // as the source of an async copy too
    let copy = KernelFn::new(
        "raw_copy",
        U32,
        vec![Param::buffer("in", U32, BufferRole::Read, Some(1))],
        vec![],
        Block(vec![Stmt::CopyAsync {
            dst: Place::Alloc {
                alloc: 0,
                index: u(0),
            },
            src: Place::Raw {
                buffer: 0,
                byte_offset: u(0),
                elem: U32,
            },
            elems: 4,
            valid: None,
            cache: unpopped::kernel::CacheHint::Auto,
            l2_prefetch: None,
        }]),
    )
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]))
    .with_allocs(vec![AllocDecl::new(
        "s",
        MemorySpace::Shared,
        U32,
        Extent::Static(4),
    )]);
    let err = CpuC.lower_kernel(&copy).expect_err("refused");
    assert!(format!("{err:?}").contains("Place::Raw"), "{err:?}");
    let control = load_from(Place::Logical {
        buffer: 0,
        coords: vec![u(0)],
    });
    assert!(CpuC.lower_kernel(&control).is_ok());
}
