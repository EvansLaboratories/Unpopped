// SPDX-License-Identifier: MIT OR Apache-2.0
//! **U19 census slice 5: struct parameters with array fields** (`docs/options-u23b-struct-array-fields.md`, option A, approved
//! by pm-kernels 2026-11-22). `struct Dims { int32_t v[8]; }` passed by value flattens to eight scalar fields (`shape.v[0]` ..
//! `shape.v[7]`) of the one `ParamKind::Struct`; a read at a runtime index is a `Select` ladder, a constant index the element
//! itself. **The choice recorded here: an index outside `0..N` reads `0`** (C leaves it undefined, so a consumer that runs a
//! real kernel which indexes out of range gets whatever its memory held, and this lift gets zero: they differ exactly there,
//! and only there). No IR change; the flattened fields are the C layout of the array byte for byte.

use unpopped::exec::LaunchShape;
use unpopped::kernel::{KernelFn, ParamKind};
use unpopped::kernel_interface::ParamsBlock;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel_launched};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut blanked = src.to_string();
    for q in ["__grid_constant__", "__device__", "__forceinline__"] {
        blanked = blanked.replace(q, &" ".repeat(q.len()));
    }
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(blanked, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(&C, src, "k", LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

fn run(k: &KernelFn, out: usize, extra: Vec<KernelArg>) -> Vec<f64> {
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(
        &[out as i64],
        &vec![0.0; out],
    ))];
    args.extend(extra);
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

fn ints(v: &[i64]) -> KernelArg {
    KernelArg::Struct(v.iter().map(|x| KernelArg::Int(i128::from(*x))).collect())
}

const DIMS: &str = "struct Dims { int v[4]; };\n";

fn not_yet(r: Result<KernelFn, KernelLiftError>, needle: &str) {
    match r {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
        }
        other => panic!("expected NotYetLifted({needle}), got {other:?}"),
    }
}

#[test]
fn the_array_flattens_to_scalar_fields_and_a_runtime_index_selects_the_element() {
    let k = lift(&format!(
        "{DIMS}void k(float* out, Dims shape, int idx) {{ out[0] = (float)shape.v[idx]; }}"
    ))
    .unwrap();
    let ParamKind::Struct { fields } = &k.params[1].kind else {
        panic!("{:?}", k.params[1])
    };
    let names: Vec<_> = fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names,
        ["shape.v[0]", "shape.v[1]", "shape.v[2]", "shape.v[3]"]
    );
    let at = |i: i64| {
        run(
            &k,
            1,
            vec![ints(&[10, 20, 30, 40]), KernelArg::Int(i128::from(i))],
        )[0]
    };
    assert_eq!([at(0), at(1), at(2), at(3)], [10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn an_index_outside_the_array_reads_zero_at_n_and_below_zero() {
    let k = lift(&format!(
        "{DIMS}void k(float* out, Dims shape, int idx) {{ out[0] = (float)shape.v[idx]; }}"
    ))
    .unwrap();
    let at = |i: i64| {
        run(
            &k,
            1,
            vec![ints(&[10, 20, 30, 40]), KernelArg::Int(i128::from(i))],
        )[0]
    };
    assert_eq!(at(4), 0.0, "index N");
    assert_eq!(at(-1), 0.0, "index -1");
    assert_eq!(at(1000), 0.0);
    // an unsigned index never goes below zero, but a huge one is outside too
    let ku = lift(&format!(
        "{DIMS}void k(float* out, Dims shape, unsigned int idx) {{ out[0] = (float)shape.v[idx]; }}"
    ))
    .unwrap();
    let atu = |i: i64| {
        run(
            &ku,
            1,
            vec![ints(&[10, 20, 30, 40]), KernelArg::Int(i128::from(i))],
        )[0]
    };
    assert_eq!(atu(3), 40.0);
    assert_eq!(atu(4_000_000_000), 0.0);
}

#[test]
fn a_constant_index_is_the_element_itself_and_a_constant_out_of_range_is_zero() {
    let k = lift(&format!(
        "{DIMS}void k(float* out, Dims shape) {{ out[0] = (float)shape.v[2]; out[1] = (float)shape.v[9]; out[2] = (float)shape.v[1 + 2]; }}"
    ))
    .unwrap();
    assert_eq!(
        run(&k, 3, vec![ints(&[10, 20, 30, 40])]),
        vec![30.0, 0.0, 40.0]
    );
}

#[test]
fn a_strided_offset_loop_over_the_array_matches_a_rust_reference() {
    // the shape of Baracuda's strided kernels: a runtime `rank` bounds a loop that indexes two arrays by the loop variable
    let k = lift(
        "struct D { int v[4]; };\n\
         void k(float* out, D shape, D stride, int rank) {\n\
           int off = 0;\n\
           for (int d = rank - 1; d >= 0; --d) { off += (shape.v[d] % 3) * stride.v[d]; }\n\
           out[0] = (float)off; }",
    )
    .unwrap();
    let (shape, stride) = ([7i64, 5, 9, 4], [100i64, 10, 1, 1000]);
    for rank in 0..=4i64 {
        let want: i64 = (0..rank as usize).map(|d| (shape[d] % 3) * stride[d]).sum();
        let got = run(
            &k,
            1,
            vec![
                ints(&shape),
                ints(&stride),
                KernelArg::Int(i128::from(rank)),
            ],
        );
        assert_eq!(got, vec![want as f64], "rank {rank}");
    }
}

#[test]
fn a_nested_struct_and_a_float_array_and_a_wide_element_flatten_the_same_way() {
    let k = lift(
        "struct In { int v[2]; };\nstruct F { float w[3]; };\nstruct S { In a; F f; int n; };\n\
         void k(float* out, S s, int i) { out[0] = (float)s.a.v[i]; out[1] = s.f.w[i]; out[2] = (float)s.n; }",
    )
    .unwrap();
    let args = |i: i64| {
        vec![
            KernelArg::Struct(vec![
                KernelArg::Int(5),
                KernelArg::Int(6),
                KernelArg::Float(0.5),
                KernelArg::Float(1.5),
                KernelArg::Float(2.5),
                KernelArg::Int(9),
            ]),
            KernelArg::Int(i128::from(i)),
        ]
    };
    assert_eq!(run(&k, 3, args(1)), vec![6.0, 1.5, 9.0]);
    assert_eq!(run(&k, 3, args(2)), vec![0.0, 2.5, 9.0]); // v has two elements: 2 is outside, w has three
}

#[test]
fn the_flattened_fields_are_the_c_layout_of_the_array() {
    let k =
        lift("struct L { long long v[8]; };\nvoid k(float* out, L l) { out[0] = (float)l.v[0]; }")
            .unwrap();
    let block = ParamsBlock::of_struct(&k.params[1]).unwrap();
    // eight consecutive 8-byte elements: the C layout of `int64_t v[8]`
    assert_eq!(block.size, 64);
    assert_eq!(block.align, 8);
    let offsets: Vec<u32> = block.fields.iter().map(|f| f.offset).collect();
    assert_eq!(offsets, [0, 8, 16, 24, 32, 40, 48, 56]);
}

#[test]
fn sixty_four_elements_flatten_and_sixty_five_decline() {
    let ok = lift("struct B { int v[64]; };\nvoid k(float* out, B b) { out[0] = (float)b.v[63]; }");
    assert!(ok.is_ok(), "{ok:?}");
    not_yet(
        lift("struct B { int v[65]; };\nvoid k(float* out, B b) { out[0] = (float)b.v[0]; }"),
        "at most 64",
    );
}

#[test]
fn an_element_is_never_assigned_and_an_array_of_structs_still_declines() {
    not_yet(
        lift(&format!(
            "{DIMS}void k(float* out, Dims shape) {{ shape.v[0] = 1; out[0] = 1.0f; }}"
        )),
        "read-only",
    );
    not_yet(
        lift(
            "struct C { int a; };\nstruct B { C v[2]; };\nvoid k(float* out, B b) { out[0] = 1.0f; }",
        ),
        "only scalar fields, arrays of scalars and nested structs",
    );
    // the array used whole, not indexed
    assert!(lift(&format!("{DIMS}void k(float* out, Dims shape) {{ const int* p = shape.v; out[0] = (float)p[0]; }}")).is_err());
}

#[test]
fn a_struct_parameter_with_an_array_passes_whole_to_a_helper() {
    // the census wall `the field shape.d, which cannot be copied`: `pc_axis_coord(i, shape, axis)` reads `shape.d[d]` at a
    // runtime index in a helper that takes the struct by value (and by const reference)
    let helper = "int coord(int linear, const Dims s, int axis) { int rem = linear; int c[4] = {0}; \
        for (int d = 3; d >= 0; --d) { int e = s.v[d]; if (e <= 0) { c[d] = 0; continue; } c[d] = rem % e; rem = rem / e; } return c[axis]; }\n";
    let by_ref = helper.replace("const Dims s", "const Dims& s");
    for h in [helper.to_string(), by_ref] {
        let src = format!(
            "{DIMS}{h}void k(float* out, Dims shape) {{ for (int i = 0; i < 6; i++) {{ out[i] = (float)coord(i * 7, shape, 2); }} }}"
        );
        let k = lift(&src).unwrap();
        // shape (1, 2, 3, 4) row-major; `axis` 2 is the third coordinate of the linear index
        let shape = [1i64, 2, 3, 4];
        let want: Vec<f64> = (0..6)
            .map(|i: i64| {
                let mut rem = i * 7;
                let mut c = [0i64; 4];
                for d in (0..4).rev() {
                    c[d] = rem % shape[d];
                    rem /= shape[d];
                }
                c[2] as f64
            })
            .collect();
        assert_eq!(run(&k, 6, vec![ints(&shape)]), want);
        // control: a different shape gives different coordinates (the array is the caller's, not a constant)
        let other = [2i64, 2, 2, 2];
        assert_ne!(run(&k, 6, vec![ints(&other)]), want);
    }
}

#[test]
fn a_helper_that_writes_its_copy_of_the_array_still_declines() {
    // the copy shares the parameter's read-only locals, so a write to an element is refused rather than aliased
    let src = format!(
        "{DIMS}float f(Dims s) {{ s.v[0] = 5; return (float)s.v[0]; }}\nvoid k(float* out, Dims shape) {{ out[0] = f(shape); }}"
    );
    assert!(lift(&src).is_err());
}
