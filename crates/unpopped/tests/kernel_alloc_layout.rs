// SPDX-License-Identifier: MIT OR Apache-2.0
//! A layout and a swizzle on an allocation (ROADMAP T1, `docs/design-swizzled-allocation.md`): `AllocDecl::layout` and
//! `align_bytes`, the checker rules with an accepted control beside each refusal, the oracle's meaning (a layout permutes the
//! storage, so a kernel that only uses `Place::Alloc` computes the same result, and `Place::AllocRaw` sees the physical bytes),
//! and the serde default (a document written before the field loads with `None`).

use unpopped::exec::{AllocDecl, AllocLayout, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt,
    check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, U8, U32};
use unpopped_vocab::layout_algebra::Swizzle;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn layout(text: &str) -> AllocLayout {
    text.parse().unwrap_or_else(|e| panic!("{text}: {e:?}"))
}

/// `out[t] = sh[63 - t]` after every thread stored `in[t]` to `sh[t]`: 64 threads, one `U32` shared array.
fn kernel(alloc: AllocDecl) -> KernelFn {
    let rev = ScalarExpr::Sub(Box::new(ScalarExpr::ConstInt(63, U32)), Box::new(tid()));
    let body = vec![
        Stmt::Load {
            local: 0,
            place: at(0, tid()),
        },
        Stmt::Store {
            place: Place::Alloc {
                alloc: 0,
                index: tid(),
            },
            value: ScalarExpr::Local(0),
        },
        Stmt::Barrier {
            scope: Scope::Block,
            kind: BarrierKind::Full,
        },
        Stmt::Load {
            local: 1,
            place: Place::Alloc {
                alloc: 0,
                index: rev,
            },
        },
        Stmt::Store {
            place: at(1, tid()),
            value: ScalarExpr::Local(1),
        },
    ];
    KernelFn::new(
        "layout",
        ElementKind::I32,
        vec![
            Param::buffer("in", U32, BufferRole::Read, Some(1)),
            Param::buffer("out", U32, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(U32), LocalDecl::scalar(U32)],
        Block(body),
    )
    .with_launch(LaunchShape::new([64, 1, 1], [1, 1, 1]))
    .with_allocs(vec![alloc])
}

fn shared_u32(extent: u32) -> AllocDecl {
    AllocDecl::new("sh", MemorySpace::Shared, U32, Extent::Static(extent))
}

fn run(k: &KernelFn) -> Vec<u32> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let data: Vec<u8> = (0..64u32).flat_map(|i| (i * 7 + 3).to_le_bytes()).collect();
    let buf = |bytes| KernelArg::Buffer(TypedBuffer::new(U32, vec![64], vec![1], bytes));
    let mut args = vec![buf(data), buf(vec![0u8; 256])];
    run_launch(k, &mut args).unwrap_or_else(|e| panic!("{e:?}"));
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    out.bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn refused(k: &KernelFn) -> &'static str {
    match check_kernel(k) {
        Err(KernelError::AllocLayout { why, .. }) => why,
        other => panic!("expected an AllocLayout refusal, got {other:?}"),
    }
}

#[test]
fn a_layout_permutes_storage_and_changes_no_result() {
    let plain = run(&kernel(shared_u32(64)));
    let want: Vec<u32> = (0..64u32).map(|t| (63 - t) * 7 + 3).collect();
    assert_eq!(plain, want, "the unlaid-out run");
    // a transpose, an identity, and a swizzle on top of each (byte swz(2,2,3): period 128 bytes = 32 U32 elements)
    let swz = Swizzle::new(2, 2, 3).unwrap();
    for text in ["(8,8):(1,8)", "(8,8):(8,1)", "64:1", "(4,16):(16,1)"] {
        let l = layout(text);
        let decl = shared_u32(64).with_layout(l.clone());
        assert_eq!(run(&kernel(decl)), want, "layout {text}");
        let decl = shared_u32(64)
            .with_layout(l.with_swizzle(swz))
            .with_align_bytes(128);
        assert_eq!(run(&kernel(decl)), want, "layout {text} under {swz}");
    }
}

#[test]
fn the_layout_is_observable_through_alloc_raw_at_the_physical_byte() {
    // swz(3,4,3) over a 1024-byte U8 array: logical index 0x80 lands at byte 0x90 (bit 7 XORed into bit 4)
    let u8_of = |v: i64| {
        ScalarExpr::Cast(
            Box::new(ScalarExpr::ConstInt(v, U32)),
            CastSpec {
                to: U8,
                round: RoundMode::NearestEven,
                sat: Saturation::Wrap,
            },
        )
    };
    let probe = |decl: AllocDecl, raw_at: i64| {
        let k = KernelFn::new(
            "raw",
            ElementKind::I32,
            vec![Param::buffer("out", U8, BufferRole::Write, Some(1))],
            vec![LocalDecl::scalar(U8)],
            Block(vec![
                Stmt::Store {
                    place: Place::Alloc {
                        alloc: 0,
                        index: ScalarExpr::ConstInt(0x80, U32),
                    },
                    value: u8_of(0xab),
                },
                Stmt::Load {
                    local: 0,
                    place: Place::AllocRaw {
                        alloc: 0,
                        byte_offset: ScalarExpr::ConstInt(raw_at, U32),
                        elem: U8,
                    },
                },
                Stmt::Store {
                    place: at(0, ScalarExpr::ConstInt(0, U32)),
                    value: ScalarExpr::Local(0),
                },
            ]),
        )
        .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]))
        .with_allocs(vec![decl]);
        assert_eq!(check_kernel(&k), Ok(()));
        let mut args = vec![KernelArg::Buffer(TypedBuffer::new(
            U8,
            vec![1],
            vec![1],
            vec![0],
        ))];
        // the raw read of a byte never stored traps (uninitialised): the stored one is the only readable byte
        (run_launch(&k, &mut args), args)
    };
    let shared = || AllocDecl::new("sh", MemorySpace::Shared, U8, Extent::Static(1024));
    let swizzled = shared()
        .with_layout(layout("swz(3,4,3)(1024:1)"))
        .with_align_bytes(1024);
    let (r, args) = probe(swizzled.clone(), 0x90);
    r.expect("byte 0x90 holds the stored value");
    let KernelArg::Buffer(out) = &args[0] else {
        unreachable!()
    };
    assert_eq!(out.bytes, vec![0xab]);
    // the control: byte 0x80, where the unswizzled array would hold it, was never written
    assert!(
        probe(swizzled, 0x80).0.is_err(),
        "0x80 is not where it landed"
    );
    // and without the layout the value is at byte 0x80
    let (r, args) = probe(shared(), 0x80);
    r.expect("identity layout: byte 0x80");
    let KernelArg::Buffer(out) = &args[0] else {
        unreachable!()
    };
    assert_eq!(out.bytes, vec![0xab]);
}

#[test]
fn the_swizzle_is_read_over_elements_for_a_wide_element() {
    // byte swz(2,2,3) over 64 U32 (256 bytes) is the element swizzle (2,0,3): bits [3,5) XORed into bits [0,2), so logical index
    // 8 lands at element 9 (byte 36). Read over bytes without the element conversion it would stay at element 8 (byte 32).
    let swizzled = shared_u32(64)
        .with_layout(layout("swz(2,2,3)(64:1)"))
        .with_align_bytes(128);
    let probe = |raw_at: i64| {
        let k = KernelFn::new(
            "raw",
            ElementKind::I32,
            vec![Param::buffer("out", U32, BufferRole::Write, Some(1))],
            vec![LocalDecl::scalar(U32)],
            Block(vec![
                Stmt::Store {
                    place: Place::Alloc {
                        alloc: 0,
                        index: ScalarExpr::ConstInt(8, U32),
                    },
                    value: ScalarExpr::Cast(
                        Box::new(ScalarExpr::ConstInt(77, U32)),
                        CastSpec {
                            to: U32,
                            round: RoundMode::NearestEven,
                            sat: Saturation::Wrap,
                        },
                    ),
                },
                Stmt::Load {
                    local: 0,
                    place: Place::AllocRaw {
                        alloc: 0,
                        byte_offset: ScalarExpr::ConstInt(raw_at, U32),
                        elem: U32,
                    },
                },
                Stmt::Store {
                    place: at(0, ScalarExpr::ConstInt(0, U32)),
                    value: ScalarExpr::Local(0),
                },
            ]),
        )
        .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]))
        .with_allocs(vec![swizzled.clone()]);
        assert_eq!(check_kernel(&k), Ok(()));
        let mut args = vec![KernelArg::Buffer(TypedBuffer::new(
            U32,
            vec![1],
            vec![1],
            vec![0; 4],
        ))];
        (run_launch(&k, &mut args), args)
    };
    let (r, args) = probe(36);
    r.expect("element 9 (byte 36) holds the stored value");
    let KernelArg::Buffer(out) = &args[0] else {
        unreachable!()
    };
    assert_eq!(out.bytes, 77u32.to_le_bytes());
    assert!(
        probe(32).0.is_err(),
        "element 8 (byte 32) was never written"
    );
}

#[test]
fn each_checker_rule_refuses_and_its_control_passes() {
    let ok = shared_u32(64).with_layout(layout("(8,8):(1,8)"));
    assert_eq!(check_kernel(&kernel(ok.clone())), Ok(()));
    // 1. space: a layout belongs to a Shared allocation
    let private = AllocDecl::new("p", MemorySpace::Private, U32, Extent::Static(64))
        .with_layout(layout("(8,8):(1,8)"));
    assert!(refused(&kernel(private)).contains("Shared"));
    // 2. size: the layout's size equals the extent, and a dynamic extent is refused with a layout
    let small = shared_u32(32).with_layout(layout("(8,8):(1,8)"));
    assert!(refused(&kernel(small)).contains("size"));
    let dynamic = AllocDecl::new("d", MemorySpace::Shared, U32, Extent::Param(0))
        .with_layout(layout("(8,8):(1,8)"));
    let k = KernelFn::new(
        "d",
        ElementKind::I32,
        vec![Param::scalar("n", ElementKind::I32, 0)],
        vec![LocalDecl::scalar(ElementKind::I32)],
        Block(vec![]),
    )
    .with_launch(LaunchShape::new([64, 1, 1], [1, 1, 1]))
    .with_allocs(vec![dynamic]);
    assert!(refused(&k).contains("Static"));
    // the control: the same dynamic allocation without a layout is accepted
    let plain_dynamic = AllocDecl::new("d", MemorySpace::Shared, U32, Extent::Param(0));
    let k = KernelFn::new(
        "d",
        ElementKind::I32,
        vec![Param::scalar("n", ElementKind::I32, 0)],
        vec![LocalDecl::scalar(ElementKind::I32)],
        Block(vec![]),
    )
    .with_launch(LaunchShape::new([64, 1, 1], [1, 1, 1]))
    .with_allocs(vec![plain_dynamic]);
    assert_eq!(check_kernel(&k), Ok(()));
    // 3. range: a layout whose cosize exceeds the extent
    let wide = shared_u32(64).with_layout(layout("(8,8):(1,9)"));
    assert!(refused(&kernel(wide)).contains("cosize"));
    // 4. injective: (2,2):(1,0) aliases; the control (2,2):(1,2) does not
    let alias = AllocDecl::new("a", MemorySpace::Shared, U32, Extent::Static(4))
        .with_layout(layout("(2,2):(1,0)"));
    assert!(refused(&kernel(alias)).contains("injective"));
    let control = AllocDecl::new("a", MemorySpace::Shared, U32, Extent::Static(4))
        .with_layout(layout("(2,2):(1,2)"));
    assert_eq!(check_kernel(&kernel(control)), Ok(()));
    // 5. swizzle: the period must divide the byte extent, the base must be at or above the element width, and the allocation
    // must be aligned to the period
    let swz = |bits, base, shift| Swizzle::new(bits, base, shift).unwrap();
    let aligned = shared_u32(64)
        .with_layout(AllocLayout::new(layout("(8,8):(1,8)").layout).with_swizzle(swz(2, 2, 3)))
        .with_align_bytes(128);
    assert_eq!(
        check_kernel(&kernel(aligned)),
        Ok(()),
        "the swizzled control"
    );
    let unaligned = shared_u32(64)
        .with_layout(AllocLayout::new(layout("64:1").layout).with_swizzle(swz(2, 2, 3)));
    assert!(refused(&kernel(unaligned)).contains("align_bytes"));
    let misaligned = shared_u32(64)
        .with_layout(AllocLayout::new(layout("64:1").layout).with_swizzle(swz(2, 2, 3)))
        .with_align_bytes(64);
    assert!(refused(&kernel(misaligned)).contains("align_bytes"));
    // a base below the element width: a U32 element is 4 bytes, so base 1 would mix bits inside one element
    let low = shared_u32(64)
        .with_layout(AllocLayout::new(layout("64:1").layout).with_swizzle(swz(2, 1, 3)))
        .with_align_bytes(128);
    assert!(refused(&kernel(low)).contains("below the element width"));
    // an extent the period does not divide (48 elements against a 32-element period)
    let ragged = AllocDecl::new("r", MemorySpace::Shared, U32, Extent::Static(48))
        .with_layout(AllocLayout::new(layout("48:1").layout).with_swizzle(swz(2, 2, 3)))
        .with_align_bytes(128);
    assert!(refused(&kernel(ragged)).contains("divide"));
    // align_bytes is a power of two on any allocation
    assert!(refused(&kernel(shared_u32(64).with_align_bytes(12))).contains("power of two"));
    assert_eq!(
        check_kernel(&kernel(shared_u32(64).with_align_bytes(16))),
        Ok(())
    );
}

#[test]
fn the_field_round_trips_and_a_document_without_it_loads_with_none() {
    let decl = shared_u32(64)
        .with_layout(layout("swz(2,2,3)((8,8):(1,8))"))
        .with_align_bytes(128);
    let text = serde_json::to_string(&decl).unwrap();
    assert!(text.contains("swz(2,2,3)((8,8):(1,8))"), "{text}");
    let back: AllocDecl = serde_json::from_str(&text).unwrap();
    assert_eq!(back, decl);
    // a plain layout is its bare text
    let plain = shared_u32(64).with_layout(layout("(8,8):(1,8)"));
    let back: AllocDecl = serde_json::from_str(&serde_json::to_string(&plain).unwrap()).unwrap();
    assert_eq!(back, plain);
    // the document before the field: no `layout`, no `align_bytes`
    let old = serde_json::to_value(shared_u32(64)).unwrap();
    assert!(
        old.get("layout").is_none() && old.get("align_bytes").is_none(),
        "{old}"
    );
    let loaded: AllocDecl = serde_json::from_value(old).unwrap();
    assert_eq!((loaded.layout, loaded.align_bytes), (None, None));
    // a malformed layout text is a load error, not a silent identity
    let bad = serde_json::json!({"name":"s","space":"Shared","elem":"u32","extent":{"Static":4},"layout":"not a layout"});
    assert!(serde_json::from_value::<AllocDecl>(bad).is_err());
}
