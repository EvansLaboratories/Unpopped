// SPDX-License-Identifier: MIT OR Apache-2.0
//! **U38d, step 1: caller-stated struct layouts.** `LiftContext::with_struct_layout(name, StructLayout)` carries the caller's own
//! `sizeof` and `offsetof` (a compiled harness under `nvcc` is the control); the walker derives none. A layout that is not
//! self-consistent, or two different layouts for one name, decline the lift by name before the source is read. (The walker
//! applying a layout to a struct read through a pointer is the next step.)

use unpopped::exec::LaunchShape;
use unpopped_convert::{
    FieldLayout, Frontend, KernelLiftError, LiftContext, StructLayout, lift_kernel_in,
};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(src, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);
const SRC: &str = "void k(float* out) { out[0] = 1.0f; }";

fn lift(ctx: &LiftContext) -> Result<(), String> {
    match lift_kernel_in(&C, SRC, "k", ctx) {
        Ok(_) => Ok(()),
        Err(KernelLiftError::NotYetLifted { construct, .. }) => Err(construct),
        Err(other) => panic!("{other:?}"),
    }
}

fn ctx() -> LiftContext {
    LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

// block_q4_0 as nvcc lays it out: half d (2) then uint8_t qs[16] (16): size 18
fn q4_0() -> StructLayout {
    StructLayout::new(
        18,
        vec![
            FieldLayout::new("d", 0, "half", 1),
            FieldLayout::new("qs", 2, "uint8_t", 16),
        ],
    )
}

#[test]
fn a_consistent_layout_is_accepted_and_unused_by_a_kernel_that_does_not_read_the_struct() {
    assert_eq!(
        lift(&ctx().with_struct_layout("block_q4_0", q4_0())),
        Ok(())
    );
    // the same layout twice is one layout
    assert_eq!(
        lift(
            &ctx()
                .with_struct_layout("block_q4_0", q4_0())
                .with_struct_layout("block_q4_0", q4_0())
        ),
        Ok(())
    );
}

#[test]
fn an_inconsistent_layout_declines_by_name() {
    let bad = |layout: StructLayout, needle: &str| {
        let r = lift(&ctx().with_struct_layout("S", layout));
        let msg = r.expect_err("must decline");
        assert!(msg.contains("`S`") && msg.contains(needle), "{msg}");
    };
    // a field past the end (qs[16] at byte 4 ends at 20 > 18)
    bad(
        StructLayout::new(
            18,
            vec![
                FieldLayout::new("d", 0, "half", 1),
                FieldLayout::new("qs", 4, "uint8_t", 16),
            ],
        ),
        "past the struct's size",
    );
    // two fields overlapping (a seeded wrong offset: qs at byte 1 over d's second byte)
    bad(
        StructLayout::new(
            18,
            vec![
                FieldLayout::new("d", 0, "half", 1),
                FieldLayout::new("qs", 1, "uint8_t", 16),
            ],
        ),
        "overlap",
    );
    bad(StructLayout::new(0, vec![]), "size of 0");
    bad(
        StructLayout::new(
            8,
            vec![
                FieldLayout::new("a", 0, "float", 1),
                FieldLayout::new("a", 4, "float", 1),
            ],
        ),
        "twice",
    );
    bad(
        StructLayout::new(8, vec![FieldLayout::new("a", 0, "quaternion", 1)]),
        "quaternion",
    );
    bad(
        StructLayout::new(8, vec![FieldLayout::new("a", 0, "float", 0)]),
        "count of 0",
    );
    bad(
        StructLayout::new(8, vec![FieldLayout::new("", 0, "float", 1)]),
        "no name",
    );
}

#[test]
fn two_different_layouts_for_one_name_decline() {
    let other = StructLayout::new(20, vec![FieldLayout::new("d", 0, "half", 1)]);
    let msg = lift(
        &ctx()
            .with_struct_layout("block_q4_0", q4_0())
            .with_struct_layout("block_q4_0", other),
    )
    .expect_err("must decline");
    assert!(msg.contains("twice with different layouts"), "{msg}");
}
