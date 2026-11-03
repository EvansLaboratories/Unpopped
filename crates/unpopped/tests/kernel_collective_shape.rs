// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Stmt::Collective` (0.42.0; PM ruling 2026-11-02 on `docs/options-u5-collectives-and-richer-reductions.md`,
//! P5): the shape rode the one `Stmt` break (0.42.0) and was refused until its semantics landed (0.42.3,
//! `kernel_collective.rs`). What stays here: the shape survives serialization, every kind of it, and a collective needs
//! a launch.

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
fn an_unlaunched_kernel_has_no_scope_to_rendezvous_in() {
    // semantics landed in 0.42.3 (`kernel_collective.rs`); without a launch there are no lanes to exchange between
    for kind in every_kind() {
        let r = check_kernel(&kernel(vec![collective(kind.clone())]));
        assert_eq!(r, Err(KernelError::ExecWithoutLaunch), "{kind:?}");
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
