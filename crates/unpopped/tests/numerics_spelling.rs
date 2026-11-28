// SPDX-License-Identifier: MIT OR Apache-2.0
//! The exact-spelling hint on an approximate node (ROADMAP U37): `Numerics::spelling` names the CUDA fast-math intrinsic a node
//! was lifted from. It is a hint, not a meaning: the checker admits it only on an `approx` node of the operation the intrinsic
//! lowers to, the oracle computes the same with it, and a document without it loads with `None`.

use unpopped::exec::LaunchShape;
use unpopped::ir::{ApproxSpelling, Numerics, QualOp, ScalarExpr, UnaryOp};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped_vocab::ElementKind::{F32, U32};

fn kernel(value: ScalarExpr) -> KernelFn {
    KernelFn::new(
        "spelling",
        F32,
        vec![Param::buffer("out", F32, BufferRole::Write, Some(1))],
        vec![LocalDecl::scalar(F32)],
        Block(vec![Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![ScalarExpr::ConstInt(0, U32)],
            },
            value,
        }]),
    )
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

fn node(op: UnaryOp, n: Numerics) -> ScalarExpr {
    ScalarExpr::Qualified(
        Box::new(ScalarExpr::Unary(op, Box::new(ScalarExpr::Const(1.5)))),
        n,
    )
}

fn approx(spelling: Option<ApproxSpelling>) -> Numerics {
    Numerics {
        approx: true,
        spelling,
        ..Numerics::default()
    }
}

#[test]
fn a_spelling_is_admitted_only_on_an_approx_node_of_its_operation() {
    let ok = |op, sp| check_kernel(&kernel(node(op, approx(Some(sp)))));
    assert_eq!(ok(UnaryOp::Exp2, ApproxSpelling::Expf), Ok(()));
    assert_eq!(ok(UnaryOp::Exp2, ApproxSpelling::Exp10f), Ok(()));
    assert_eq!(ok(UnaryOp::Log2, ApproxSpelling::Logf), Ok(()));
    // the wrong operation: a `__logf` on an Exp2 node
    let wrong = ok(UnaryOp::Exp2, ApproxSpelling::Logf);
    assert!(format!("{wrong:?}").contains("spelling hint"), "{wrong:?}");
    // a spelling on a node that is not approx
    let exact = Numerics {
        spelling: Some(ApproxSpelling::Expf),
        ..Numerics::default()
    };
    assert!(
        format!("{:?}", check_kernel(&kernel(node(UnaryOp::Exp2, exact))))
            .contains("spelling hint")
    );
    // the controls: the same nodes without the hint pass
    assert_eq!(
        check_kernel(&kernel(node(UnaryOp::Exp2, approx(None)))),
        Ok(())
    );
}

#[test]
fn qualops_of_every_spelling_take_approx() {
    // every spelling names an operation that may be marked approx, so no spelling is unreachable
    for sp in [
        ApproxSpelling::Expf,
        ApproxSpelling::Exp10f,
        ApproxSpelling::Log2f,
        ApproxSpelling::Logf,
        ApproxSpelling::Log10f,
        ApproxSpelling::Fdividef,
    ] {
        assert!(sp.op().takes_approx(), "{sp:?}");
        assert!(sp.token().starts_with("__"), "{sp:?}");
    }
    assert_eq!(ApproxSpelling::Fdividef.op(), QualOp::Div);
}

#[test]
fn the_field_round_trips_and_a_document_without_it_loads_with_none() {
    let n = approx(Some(ApproxSpelling::Log10f));
    let text = serde_json::to_string(&n).unwrap();
    assert!(text.contains("Log10f"), "{text}");
    assert_eq!(serde_json::from_str::<Numerics>(&text).unwrap(), n);
    let plain = serde_json::to_value(Numerics::default()).unwrap();
    assert!(plain.get("spelling").is_none(), "{plain}");
    let old = serde_json::json!({"round": "NearestEven", "ftz": false, "approx": true});
    assert_eq!(
        serde_json::from_value::<Numerics>(old).unwrap(),
        approx(None)
    );
}
