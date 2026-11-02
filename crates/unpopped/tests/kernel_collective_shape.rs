// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Stmt::Collective` (0.42.0; PM ruling 2026-11-02 on `docs/options-u5-collectives-and-richer-reductions.md`,
//! P5): the shape rides the one `Stmt` break. Nothing runs a collective yet: `check_kernel` refuses it with a typed
//! error, so no producer builds a kernel an interpreter would have to guess about, and the later slice that lands
//! the semantics only lifts that refusal. The shape survives serialization, every kind of it.

use unpopped::exec::Scope;
use unpopped::ir::{ReduceOp, ScalarExpr};
use unpopped::kernel::{
    Block, CollectiveKind, KernelError, KernelFn, LocalDecl, Param, ShuffleMode, Stmt, VoteKind,
    check_kernel,
};
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;

fn collective(kind: CollectiveKind) -> Stmt {
    Stmt::Collective {
        kind,
        value: ScalarExpr::ConstInt(1, I32),
        result: 0,
        scope: Scope::Subgroup,
    }
}

fn kernel(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        "k",
        I32,
        Vec::<Param>::new(),
        vec![LocalDecl::scalar(I32)],
        Block(body),
    )
}

fn every_kind() -> Vec<CollectiveKind> {
    let lane = || ScalarExpr::ConstInt(3, I32);
    vec![
        CollectiveKind::Shuffle {
            mode: ShuffleMode::Xor,
            operand: lane(),
        },
        CollectiveKind::Vote(VoteKind::Ballot),
        CollectiveKind::Reduce(ReduceOp::Xor),
        CollectiveKind::Scan {
            op: ReduceOp::Sum,
            exclusive: true,
        },
        CollectiveKind::Broadcast { lane: lane() },
    ]
}

#[test]
fn a_collective_is_refused_with_a_typed_error_until_its_semantics_land() {
    for kind in every_kind() {
        let r = check_kernel(&kernel(vec![collective(kind.clone())]));
        assert!(
            matches!(r, Err(KernelError::CollectiveUnsupported { .. })),
            "{kind:?}: {r:?}"
        );
    }
}

#[test]
fn the_same_kernel_without_the_collective_checks() {
    assert_eq!(check_kernel(&kernel(vec![])), Ok(()));
}

#[test]
fn every_kind_round_trips() {
    for kind in every_kind() {
        let k = kernel(vec![collective(kind)]);
        let text = serde_json::to_string(&k).expect("a kernel serializes");
        let back: KernelFn = serde_json::from_str(&text).expect("and reads back");
        assert_eq!(back, k);
    }
}
