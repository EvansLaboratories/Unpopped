// SPDX-License-Identifier: MIT OR Apache-2.0
//! Union batch 1 (KISS 0.24.0, board 501 and 515; plan E1a): the seven new dtype rows are storage and key identity ONLY. Each has
//! its token, width and kind (pinned against KISS's manifest in `unpopped-vocab`), survives the text codec, and is **declined by
//! name** wherever a kernel would compute it: `check_kernel` refuses a kernel that declares one (`DtypeNotComputed`), and a `Cast`
//! into one is still the typed `IllegalCast`. Before this guard the oracle ran a `Load` of an `F128`/`I128`/`U128`/`F8*` buffer
//! without holding a value of the type and panicked on `I256`/`U256` (probed 2026-11-26): the silent default this repository refuses.
//! Each refusal sits beside its accepted control.

use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::types::TypeError;
use unpopped_vocab::ElementKind;

const ROWS: [(ElementKind, &str, u32); 7] = [
    (ElementKind::F128, "f128", 128),
    (ElementKind::I128, "i128", 128),
    (ElementKind::U128, "u128", 128),
    (ElementKind::I256, "i256", 256),
    (ElementKind::U256, "u256", 256),
    (ElementKind::F8E3M4, "f8e3m4", 8),
    (ElementKind::F8UE4M3, "f8ue4m3", 8),
];

fn launch() -> LaunchShape {
    LaunchShape::new([1, 1, 1], [1, 1, 1])
}

fn kernel(
    compute: ElementKind,
    params: Vec<Param>,
    locals: Vec<LocalDecl>,
    body: Vec<Stmt>,
) -> KernelFn {
    KernelFn::new("k", compute, params, locals, Block(body)).with_launch(launch())
}

fn load_from(buffer_elem: ElementKind, local_elem: ElementKind) -> KernelFn {
    kernel(
        ElementKind::F32,
        vec![Param::buffer("x", buffer_elem, BufferRole::Read, Some(1))],
        vec![LocalDecl::scalar(local_elem)],
        vec![Stmt::Load {
            local: 0,
            place: Place::Logical {
                buffer: 0,
                coords: vec![ScalarExpr::ConstInt(0, ElementKind::I64)],
            },
        }],
    )
}

fn refused(dtype: ElementKind) -> Result<(), KernelError> {
    Err(KernelError::DtypeNotComputed { dtype })
}

#[test]
fn a_kernel_that_declares_a_batch_1_dtype_is_refused_by_name_and_the_control_passes() {
    for (dt, name, _) in ROWS {
        // a buffer parameter, a local, a scalar parameter, an allocation, the compute type: each refused
        assert_eq!(
            check_kernel(&load_from(dt, ElementKind::F32)),
            refused(dt),
            "{name} buffer parameter"
        );
        assert_eq!(
            check_kernel(&kernel(
                ElementKind::F32,
                vec![],
                vec![LocalDecl::scalar(dt)],
                vec![]
            )),
            refused(dt),
            "{name} local"
        );
        assert_eq!(
            check_kernel(&kernel(
                ElementKind::F32,
                vec![Param::scalar("s", dt, 0)],
                vec![LocalDecl::scalar(dt)],
                vec![]
            )),
            refused(dt),
            "{name} scalar parameter"
        );
        let with_alloc =
            kernel(ElementKind::F32, vec![], vec![], vec![]).with_allocs(vec![AllocDecl::new(
                "a",
                MemorySpace::Shared,
                dt,
                Extent::Static(4),
            )]);
        assert_eq!(check_kernel(&with_alloc), refused(dt), "{name} allocation");
        assert_eq!(
            check_kernel(&kernel(dt, vec![], vec![], vec![])),
            refused(dt),
            "{name} compute type"
        );
    }
    // the controls: the same shapes over an ordinary dtype pass
    assert_eq!(
        check_kernel(&load_from(ElementKind::F32, ElementKind::F32)),
        Ok(())
    );
    assert_eq!(
        check_kernel(&kernel(
            ElementKind::F32,
            vec![],
            vec![LocalDecl::scalar(ElementKind::U32)],
            vec![]
        )),
        Ok(())
    );
    assert_eq!(
        check_kernel(
            &kernel(ElementKind::F32, vec![], vec![], vec![]).with_allocs(vec![AllocDecl::new(
                "a",
                MemorySpace::Shared,
                ElementKind::F32,
                Extent::Static(4)
            )])
        ),
        Ok(())
    );
}

#[test]
fn a_cast_into_a_batch_1_dtype_is_still_the_typed_illegal_cast() {
    for (dt, name, _) in ROWS {
        let k = kernel(
            ElementKind::F32,
            vec![],
            vec![LocalDecl::scalar(ElementKind::F32)],
            vec![Stmt::Let {
                local: 0,
                init: ScalarExpr::Cast(
                    Box::new(ScalarExpr::Const(1.0)),
                    CastSpec {
                        to: dt,
                        round: RoundMode::NearestEven,
                        sat: Saturation::Wrap,
                    },
                ),
            }],
        );
        assert!(
            matches!(
                check_kernel(&k),
                Err(KernelError::Type(TypeError::IllegalCast { .. }))
            ),
            "{name}: {:?}",
            check_kernel(&k)
        );
    }
}

#[test]
fn each_row_survives_the_text_codec() {
    for (dt, name, _) in ROWS {
        let text = serde_json::to_string(&LocalDecl::scalar(dt)).expect("serialises");
        let back: LocalDecl = serde_json::from_str(&text).expect("parses");
        assert_eq!(back.elem, dt, "{name}: {text}");
    }
}

#[test]
fn the_helper_names_exactly_the_seven_rows() {
    let named: Vec<ElementKind> = ElementKind::ALL
        .into_iter()
        .filter(|t| unpopped::types::is_union_batch1(*t))
        .collect();
    assert_eq!(named, ROWS.iter().map(|r| r.0).collect::<Vec<_>>());
}
