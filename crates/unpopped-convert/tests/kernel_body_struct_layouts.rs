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

// ---- step 2: a struct read through a pointer into a byte buffer, with the caller's layout ----

use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

const BLOCK: &str = "typedef struct { half d; uint8_t qs[4]; } block_q;\n";

/// Two blocks, 6 bytes each (`half d` at 0, `uint8_t qs[4]` at 2): d0 = 1.5, qs0 = 1..4; d1 = -2, qs1 = 5..8.
fn blocks() -> Vec<u8> {
    vec![0x00, 0x3e, 1, 2, 3, 4, 0x00, 0xc0, 5, 6, 7, 8]
}

fn q6() -> StructLayout {
    StructLayout::new(
        6,
        vec![
            FieldLayout::new("d", 0, "half", 1),
            FieldLayout::new("qs", 2, "uint8_t", 4),
        ],
    )
}

fn try_read(src: &str, layout: StructLayout) -> Option<Vec<f64>> {
    let ctx = ctx().with_struct_layout("block_q", layout);
    let k = lift_kernel_in(&C, src, "k", &ctx).unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[6], &[0.0; 6])),
        KernelArg::Buffer(TypedBuffer::from_u8(&[12], &blocks())),
    ];
    run_launch(&k, &mut args).ok()?;
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    Some(o.to_f64_vec())
}

fn read(src: &str, layout: StructLayout) -> Vec<f64> {
    try_read(src, layout).expect("runs")
}

const READS: &str = "void k(float* out, const void* vx) { const block_q* x = (const block_q*)vx; \
    out[0] = __half2float(x[0].d); out[1] = (float)x[0].qs[3]; out[2] = __half2float(x[1].d); out[3] = (float)x[1].qs[0]; \
    out[4] = (float)(x + 1)->qs[2]; out[5] = __half2float(x->d); }";

#[test]
fn members_read_through_a_pointer_use_the_callers_stride_and_offsets() {
    let src = format!("{BLOCK}{READS}");
    // x[0].d = 1.5, x[0].qs[3] = 4, x[1].d = -2, x[1].qs[0] = 5, (x + 1)->qs[2] = 7, x->d = 1.5
    assert_eq!(read(&src, q6()), vec![1.5, 4.0, -2.0, 5.0, 7.0, 1.5]);
}

#[test]
fn a_layout_that_is_consistent_but_wrong_gives_a_wrong_result() {
    // the caller's numbers decide the bytes read: `qs` stated at byte 4 of an 8-byte struct reads other bytes, and the stride 8
    // puts block 1 at byte 8
    let wrong = StructLayout::new(
        8,
        vec![
            FieldLayout::new("d", 0, "half", 1),
            FieldLayout::new("qs", 4, "uint8_t", 4),
        ],
    );
    // either the run traps (block 1 at byte 8 is past the 12-byte buffer) or it reads other bytes
    assert_ne!(
        try_read(&format!("{BLOCK}{READS}"), wrong),
        Some(vec![1.5, 4.0, -2.0, 5.0, 7.0, 1.5])
    );
}

#[test]
fn a_missing_field_a_missing_layout_or_a_whole_struct_declines_by_name() {
    let decline = |src: &str, layout: Option<StructLayout>, needle: &str| {
        let mut c = ctx();
        if let Some(l) = layout {
            c = c.with_struct_layout("block_q", l);
        }
        match lift_kernel_in(&C, &format!("{BLOCK}{src}"), "k", &c) {
            Err(KernelLiftError::NotYetLifted { construct, .. }) => {
                assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
            }
            other => panic!("{needle}: {other:?}"),
        }
    };
    let no_field = "void k(float* out, const void* vx) { const block_q* x = (const block_q*)vx; out[0] = (float)x[0].nothing; }";
    decline(no_field, Some(q6()), "does not give");
    // no layout supplied: the type is not derived
    decline(READS, None, "struct");
    let array_whole = "void k(float* out, const void* vx) { const block_q* x = (const block_q*)vx; out[0] = (float)x[0].qs; }";
    decline(array_whole, Some(q6()), "index it");
    let whole = "void k(float* out, const void* vx) { const block_q* x = (const block_q*)vx; out[0] = (float)x[0]; }";
    decline(whole, Some(q6()), "whole");
    let scalar_subscript = "void k(float* out, const void* vx) { const block_q* x = (const block_q*)vx; out[0] = (float)x[0].d[1]; }";
    decline(scalar_subscript, Some(q6()), "subscripted");
}

#[test]
fn an_array_field_of_wider_elements_scales_by_the_element_width() {
    // `typedef struct { int16_t a[3]; } blk16;` is 6 bytes; a = 10, -20, 30 (little endian)
    let src = "typedef struct { int16_t a[3]; } blk16;\n\
        void k(float* out, const void* vx) { const blk16* x = (const blk16*)vx; out[0] = (float)x[0].a[2]; out[1] = (float)x[0].a[1]; }";
    let layout = StructLayout::new(6, vec![FieldLayout::new("a", 0, "int16_t", 3)]);
    let ctx = ctx().with_struct_layout("blk16", layout);
    let k = lift_kernel_in(&C, src, "k", &ctx).unwrap();
    let bytes: Vec<u8> = [10i16, -20, 30]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[0.0; 2])),
        KernelArg::Buffer(TypedBuffer::from_u8(&[6], &bytes)),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    assert_eq!(o.to_f64_vec(), vec![30.0, -20.0]);
}

// ---- step 3: vector-typed fields (`half2 dm`) ----

fn vec_layout() -> StructLayout {
    // block_q4_1 as nvcc lays it out: half2 dm (4 bytes) at 0, uint8_t qs[4] at 4 (shortened); size 8
    StructLayout::new(
        8,
        vec![
            FieldLayout::new("dm", 0, "half2", 1),
            FieldLayout::new("qs", 4, "uint8_t", 4),
        ],
    )
}

fn run_vec(src: &str, layout: StructLayout, n: usize) -> Vec<f64> {
    let ctx = ctx().with_struct_layout("block_q", layout);
    let k = lift_kernel_in(&C, src, "k", &ctx).unwrap();
    // two 8-byte blocks: dm0 = (1.5, 2.0), qs0 = 1..4; dm1 = (-2.0, 0.5), qs1 = 5..8
    let bytes: Vec<u8> = vec![
        0x00, 0x3e, 0x00, 0x40, 1, 2, 3, 4, 0x00, 0xc0, 0x00, 0x38, 5, 6, 7, 8,
    ];
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &vec![0.0; n])),
        KernelArg::Buffer(TypedBuffer::from_u8(&[16], &bytes)),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

const BLOCK2: &str = "typedef struct { half2 dm; uint8_t qs[4]; } block_q;\n";

#[test]
fn a_vector_field_is_read_whole_and_by_lane() {
    let src = format!(
        "{BLOCK2}void k(float* out, const void* vx) {{ const block_q* x = (const block_q*)vx; \
         float2 a = __half22float2(x[0].dm); out[0] = a.x; out[1] = a.y; \
         out[2] = __half2float(x[1].dm.x); out[3] = __half2float(x[1].dm.y); out[4] = (float)x[1].qs[3]; }}"
    );
    assert_eq!(
        run_vec(&src, vec_layout(), 5),
        vec![1.5, 2.0, -2.0, 0.5, 8.0]
    );
}

#[test]
fn a_vector_field_used_as_a_scalar_or_a_wrong_shape_declines() {
    let decline = |src: &str, needle: &str| {
        let c = ctx().with_struct_layout("block_q", vec_layout());
        match lift_kernel_in(&C, &format!("{BLOCK2}{src}"), "k", &c) {
            Err(KernelLiftError::NotYetLifted { construct, .. }) => {
                assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
            }
            other => panic!("{needle}: {other:?}"),
        }
    };
    decline(
        "void k(float* out, const void* vx) { const block_q* x = (const block_q*)vx; out[0] = __half2float(x[0].dm); }",
        "vector field",
    );
    decline(
        "void k(float* out, const void* vx) { const block_q* x = (const block_q*)vx; float4 a = x[0].dm; out[0] = a.x; }",
        "read as a",
    );
    decline(
        "void k(float* out, const void* vx) { const block_q* x = (const block_q*)vx; out[0] = __half2float(x[0].dm.w); }",
        "member",
    );
}

#[test]
fn a_vector_field_that_runs_past_the_struct_is_an_inconsistent_layout() {
    let bad = StructLayout::new(6, vec![FieldLayout::new("dm", 4, "half2", 1)]);
    let msg = lift(&ctx().with_struct_layout("S", bad)).expect_err("must decline");
    assert!(msg.contains("past the struct's size"), "{msg}");
}
