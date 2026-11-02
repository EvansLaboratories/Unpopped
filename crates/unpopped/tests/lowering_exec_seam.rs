// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Lowering::exec` (0.41.2): the seam a launched `lower_kernel` supplies to spell a kernel-layer
//! execution id. Without it an `Exec` leaf declines in-band, exactly as it did before the seam existed;
//! with it the leaf is spelled by the id the node carries.

use unpopped::backend::{Decline, DeclinedOp, Lowering, Spelling, lower_expr};
use unpopped::exec::ExecId;
use unpopped::ir::{BinaryOp, ScalarExpr, UnaryOp};

fn leaf(_: u8) -> Result<Spelling, unpopped::backend::LowerError> {
    Ok(Spelling::Spelled("x".into()))
}

fn unary(_: UnaryOp, a: String) -> Result<Spelling, unpopped::backend::LowerError> {
    Ok(Spelling::Spelled(a))
}

fn binary(_: BinaryOp, a: String, b: String) -> Result<Spelling, unpopped::backend::LowerError> {
    Ok(Spelling::Spelled(format!("({a} ? {b})")))
}

#[test]
fn an_exec_leaf_declines_without_the_seam() {
    let lo = Lowering::builder(&leaf, &unary, &binary).build();
    let r =
        lower_expr(&ScalarExpr::Exec(ExecId::ThreadIdx(0)), &lo).expect("a decline, not an error");
    match r {
        Spelling::Declined(Decline::UnsupportedOp { op, why }) => {
            assert_eq!(op, DeclinedOp::Kernel);
            assert!(why.contains("ThreadIdx"), "names the id: {why}");
        }
        other => panic!("expected the default decline, got {other:?}"),
    }
}

#[test]
fn an_exec_leaf_is_spelled_by_the_seam_with_the_id_it_carries() {
    let exec = |id: ExecId| Ok(Spelling::Spelled(format!("<{id:?}>")));
    let lo = Lowering::builder(&leaf, &unary, &binary)
        .exec(&exec)
        .build();
    for id in [
        ExecId::ThreadIdx(2),
        ExecId::BlockDim(1),
        ExecId::GridDim(0),
    ] {
        let r = lower_expr(&ScalarExpr::Exec(id), &lo).expect("spelled");
        assert_eq!(r, Spelling::Spelled(format!("<{id:?}>")));
    }
}
