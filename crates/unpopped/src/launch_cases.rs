// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The shared cross-emitter conformance corpus for `Backend::lower_kernel`** (ROADMAP U20, part a).
//!
//! Every emitter that overrides [`Backend::lower_kernel`](crate::backend::Backend::lower_kernel) (Baracuda's CUDA emitter,
//! fishstringer's Slang emitter, openclownfish, `unpopped-cpu-c`) runs the same cases, so "the emitter agrees with the
//! oracle" means the same thing in every repo. A [`LaunchCase`] holds a launched [`KernelFn`], its arguments, and the
//! bytes the launch interpreter ([`run_launch`]) leaves in every buffer: the expected result.
//!
//! The corpus is the first slice: launched elementwise kernels over `F32`, `I32` and `U32`, grid-strided so the launch is
//! smaller than the data. It grows with the emitters (a params block and strided views are the next slice); a case is
//! only ever added, never changed, so a pinned consumer's list does not move under it.
//!
//! # The rounding model the corpus assumes
//!
//! The oracle evaluates an expression at `f64` and rounds **once per statement** (to the local's or buffer's type), not once
//! per operation as C does for `float`. Two consequences, both stated per case in [`LaunchCase::ulps`]:
//!
//! * A statement with **one** `F32` operation (`+`, `-`, `*`, `/`, `sqrt`) is correctly rounded either way: the `f64`
//!   result rounded to `f32` is the IEEE `f32` result (double rounding is innocuous because 53 >= 2 * 24 + 2). Such a case has
//!   `ulps == 0` and an emitter must match it bit for bit. In particular **`F32` `Div` is exact in the oracle**: a device
//!   whose divide is 1 ULP off (an approximate or non-IEEE divide) does not meet the corpus at `ulps == 0`; it must use the
//!   IEEE-rounded divide or decline `F32` `Div`, which is what openclownfish does (openclownfish#54).
//! * A statement with **several** operations differs from per-operation rounding by up to a few ULPs unless the emitter also
//!   evaluates at `f64` (the `with_fp64` spelling). Such a case carries the tolerance, 3 ULPs for a three-operation statement
//!   (the bound `unpopped-cpu-c`'s own differentials use), and an emitter that evaluates at `f64` matches at 0.
//!
//! The tolerance is a statement about the corpus, not a grant: an emitter that needs more than `ulps` declines.
//!
//! # How an emitter uses it
//!
//! ```
//! use unpopped::launch_cases::launch_cases;
//! for case in launch_cases() {
//!     // 1. lower `case.kernel` with your backend and compare the interface to `KernelInterface::of(&case.kernel)`;
//!     // 2. run the emitted code on `case.args` (buffers and scalars, one per parameter, in order);
//!     // 3. compare each buffer in `case.expected` byte for byte with what your run left at that parameter.
//!     assert!(!case.expected.is_empty(), "{}", case.name);
//! }
//! ```

use crate::exec::SubgroupSize;
use crate::ir::{Expr, OpDef, ScalarExpr, UnaryOp, input};
use crate::kernel::KernelFn;
use crate::kernel_lift::lift_elementwise_opdef_launched;
use crate::oracle::TypedBuffer;
use crate::oracle::kernel_eval::KernelArg;
use crate::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;
use unpopped_vocab::ElementKind::{I32, U32};

/// One launched kernel with its arguments and the oracle's result.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LaunchCase {
    /// A stable name, unique in the corpus: `<op>_<dtype>`.
    pub name: String,
    /// The launched kernel to lower.
    pub kernel: KernelFn,
    /// One argument per parameter, in declaration order: the inputs, the zeroed output, then one `I32` extent per axis.
    pub args: Vec<KernelArg>,
    /// `(parameter index, bytes the oracle left there)` for every buffer the launch writes.
    pub expected: Vec<(usize, TypedBuffer)>,
    /// The most `F32` ULPs an emitted result may differ from [`expected`](Self::expected): 0 means bit for bit. See the
    /// module docs for why single-operation cases are 0.
    pub ulps: u64,
}

/// Deterministic, rounding-sensitive input values: `n` distinct patterns for `dtype`, from `seed`.
fn pattern(dtype: ElementKind, n: usize, seed: u32) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            let x = f64::from((i as u32 * 7919 + seed * 104_729) % 2003) / 97.0 - 10.0;
            match dtype {
                ElementKind::F32 => (x as f32).to_le_bytes().to_vec(),
                _ => (x as i32).to_le_bytes().to_vec(),
            }
        })
        .collect()
}

fn dense(dtype: ElementKind, shape: &[i64], seed: u32, zero: bool) -> TypedBuffer {
    let n = shape.iter().product::<i64>() as usize;
    let mut strides = vec![1i64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    let bytes = if zero {
        vec![0; n * 4]
    } else {
        pattern(dtype, n, seed)
    };
    TypedBuffer::new(dtype, shape.to_vec(), strides, bytes)
}

/// One case: `body` over two inputs of `dtype`, launched with `block` threads and `grid` blocks over `shape`.
fn case(
    op: &str,
    dtype: ElementKind,
    body: ScalarExpr,
    shape: &[i64],
    (block, grid): (u32, u32),
    ulps: u64,
) -> LaunchCase {
    let ins = [dense(dtype, shape, 1, false), dense(dtype, shape, 2, false)];
    case_over(op, dtype, body, shape, (block, grid), ulps, ins)
}

/// [`case`] with the two input buffers given, so an input can be a strided, offset, flipped or broadcast view of its storage
/// (the kernel reads through the bound buffer's own shape, strides and base offset). The output is always dense.
fn case_over(
    op: &str,
    dtype: ElementKind,
    body: ScalarExpr,
    shape: &[i64],
    launch: (u32, u32),
    ulps: u64,
    ins: [TypedBuffer; 2],
) -> LaunchCase {
    let (block, grid) = launch;
    let rank = u8::try_from(shape.len()).expect("rank fits u8");
    let opdef = OpDef::elementwise(op, 2, &[dtype], Expr(body));
    let kernel = lift_elementwise_opdef_launched(&opdef, dtype, rank, block, grid)
        .expect("the corpus's kernels lift");
    let [in0, in1] = ins;
    let mut args: Vec<KernelArg> = vec![
        KernelArg::Buffer(in0),
        KernelArg::Buffer(in1),
        KernelArg::Buffer(dense(dtype, shape, 0, true)),
    ];
    args.extend(shape.iter().map(|&e| KernelArg::Int(i128::from(e))));
    let mut run = args.clone();
    run_launch(&kernel, &mut run).expect("the oracle runs the corpus's kernels");
    let out = 2;
    let KernelArg::Buffer(result) = run.swap_remove(out) else {
        unreachable!("parameter {out} is the output buffer")
    };
    LaunchCase {
        name: format!("{op}_{dtype:?}").to_lowercase(),
        kernel,
        args,
        expected: vec![(out, result)],
        ulps,
    }
}

/// A view of `len` storage elements (from `seed`): logical `shape` read at `strides` from element `base`.
fn view(
    dtype: ElementKind,
    shape: &[i64],
    strides: &[i64],
    base: i64,
    len: usize,
    seed: u32,
) -> TypedBuffer {
    let mut b = TypedBuffer::new(
        dtype,
        shape.to_vec(),
        strides.to_vec(),
        pattern(dtype, len, seed),
    );
    b.base_offset = base;
    b
}

fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

fn abs(e: &ScalarExpr) -> ScalarExpr {
    ScalarExpr::Unary(UnaryOp::Abs, bx(e.clone()))
}

/// The corpus, in a fixed order. Each call rebuilds it (the oracle runs once per case; the cases are small).
#[must_use]
pub fn launch_cases() -> Vec<LaunchCase> {
    let (a, b) = (input(0).0, input(1).0);
    vec![
        // `|a| * |b| + |a|`: the oracle rounds once, per-operation C three times, so this case carries 3 ULPs. `Abs` keeps
        // both addends positive: a cancelling sum has an unbounded ULP error (the data, not the emitter, would be measured).
        case(
            "fma_shape",
            ElementKind::F32,
            ScalarExpr::Add(bx(ScalarExpr::Mul(bx(abs(&a)), bx(abs(&b)))), bx(abs(&a))),
            &[5, 7],
            (4, 2),
            3,
        ),
        // ONE F32 operation: correctly rounded in the oracle, so 0 ULPs (a non-IEEE device divide fails here). It must stay one
        // operation: `b / (a + 11)` was tried first and CI showed 1-ULP differences, because the oracle keeps `a + 11` at f64
        // while C rounds it to f32 first. Zero denominators are in the data: inf and the canonical NaN agree on both sides.
        case(
            "div",
            ElementKind::F32,
            ScalarExpr::Div(bx(a.clone()), bx(b.clone())),
            &[6, 5],
            (8, 2),
            0,
        ),
        case(
            "sub",
            ElementKind::I32,
            ScalarExpr::Sub(bx(a.clone()), bx(b.clone())),
            &[3, 11],
            (8, 1),
            0,
        ),
        case(
            "add",
            ElementKind::U32,
            ScalarExpr::Add(bx(a.clone()), bx(b.clone())),
            &[37],
            (16, 3),
            0,
        ),
        // ---- the second slice (appended; nothing above changes): views and launch geometry ----
        //
        // A transposed first input: logical `[4, 6]` read from column-major storage (strides `[1, 4]`).
        case_over(
            "view_transposed",
            I32,
            ScalarExpr::Sub(bx(a.clone()), bx(b.clone())),
            &[4, 6],
            (8, 2),
            0,
            [
                view(I32, &[4, 6], &[1, 4], 0, 24, 1),
                dense(I32, &[4, 6], 2, false),
            ],
        ),
        // A pitched sub-view: rows 8 apart, the first input starting at element 2 and the second at 1 (`base_offset`).
        case_over(
            "view_pitched",
            I32,
            ScalarExpr::Add(bx(a.clone()), bx(b.clone())),
            &[3, 5],
            (4, 3),
            0,
            [
                view(I32, &[3, 5], &[8, 1], 2, 24, 1),
                view(I32, &[3, 5], &[8, 1], 1, 24, 2),
            ],
        ),
        // A flipped last axis (negative stride, from the row's end) against a row broadcast (stride 0).
        case_over(
            "view_flipped_broadcast",
            I32,
            ScalarExpr::Sub(bx(a.clone()), bx(b.clone())),
            &[4, 5],
            (16, 1),
            0,
            [
                view(I32, &[4, 5], &[5, -1], 4, 20, 1),
                view(I32, &[4, 5], &[0, 1], 0, 5, 2),
            ],
        ),
        // Launch geometry: a block that is not a power of two, fewer threads than elements (grid-stride wraps).
        case(
            "launch_odd_block",
            U32,
            ScalarExpr::Add(bx(a.clone()), bx(b.clone())),
            &[37],
            (3, 5),
            0,
        ),
        // More threads than elements: the idle threads must write nothing.
        case(
            "launch_idle_threads",
            U32,
            ScalarExpr::Sub(bx(a.clone()), bx(b.clone())),
            &[5],
            (32, 4),
            0,
        ),
        // One thread does it all.
        case(
            "launch_single_thread",
            I32,
            ScalarExpr::Add(bx(a.clone()), bx(b.clone())),
            &[2, 9],
            (1, 1),
            0,
        ),
        // A subgroup-size range on the launch (no barrier or shuffle, so every size in it gives the one result).
        subgroup_case(
            "launch_subgroup_range",
            ScalarExpr::Add(bx(a), bx(b)),
            (8, 32),
        ),
    ]
}

/// A one-operation `F32` case over `[64]` (block 64, grid 2) whose launch requires a subgroup size in `range`.
fn subgroup_case(op: &str, body: ScalarExpr, range: (u32, u32)) -> LaunchCase {
    let mut c = case(op, ElementKind::F32, body, &[64], (64, 2), 0);
    let launch = c.kernel.launch.as_mut().expect("a launched kernel");
    launch.subgroup = Some(SubgroupSize::new(range.0, range.1));
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_and_every_case_has_an_expected_output() {
        let cases = launch_cases();
        let mut names: Vec<_> = cases.iter().map(|c| c.name.clone()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), cases.len());
        assert!(cases.iter().all(|c| !c.expected.is_empty()));
    }

    #[test]
    fn every_viewed_input_is_read_through_its_view_not_the_dense_storage() {
        // The value mutant of the view cases, one input at a time: read that input's storage densely (strides and base offset
        // ignored) and the output must change, so an emitter that ignores any one view is caught, and a case whose view is
        // really dense (or only one of whose inputs is a view) cannot pass for more than it is.
        let mut viewed = 0;
        for c in launch_cases() {
            for input in 0..2 {
                let mut args = c.args.clone();
                let KernelArg::Buffer(b) = &mut args[input] else {
                    unreachable!()
                };
                let dense_strides = dense(b.dtype, &b.shape, 0, true).strides;
                if b.strides == dense_strides && b.base_offset == 0 {
                    continue;
                }
                viewed += 1;
                b.strides = dense_strides;
                b.base_offset = 0;
                let need = b.shape.iter().product::<i64>() as usize * 4;
                b.bytes.resize(need.max(b.bytes.len()), 0);
                run_launch(&c.kernel, &mut args).expect("the dense reading runs");
                let KernelArg::Buffer(got) = &args[2] else {
                    unreachable!()
                };
                assert_ne!(got.bytes, c.expected[0].1.bytes, "{} input {input}", c.name);
            }
        }
        assert_eq!(viewed, 5, "transposed 1, pitched 2, flipped/broadcast 2");
    }

    #[test]
    fn the_subgroup_case_carries_its_range_and_the_others_do_not() {
        for c in launch_cases() {
            let sg = c.kernel.launch.expect("launched").subgroup;
            assert_eq!(
                sg.is_some(),
                c.name == "launch_subgroup_range_f32",
                "{}",
                c.name
            );
        }
    }

    #[test]
    fn the_expected_output_is_not_the_zeroed_buffer() {
        // A corpus whose oracle left the output untouched would pass any emitter that writes nothing.
        for c in launch_cases() {
            let (_, out) = &c.expected[0];
            assert!(out.bytes.iter().any(|&b| b != 0), "{}", c.name);
        }
    }
}
