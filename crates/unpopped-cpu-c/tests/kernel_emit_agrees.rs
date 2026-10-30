// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The emitted C agrees with the kernel interpreter** (`docs/e1-emit-from-the-kernel-layer.md`,
//! stage E1a): a kernel is lowered by `CpuC::lower_kernel`, compiled with a real host C compiler,
//! run, and its output buffers compared **byte for byte** with `oracle::kernel_eval::run_kernel`
//! on the same arguments. The traps (`OutOfBounds`, `ZeroStep`) must agree too.
//!
//! The serial C is a correctness reference. Nothing here times anything.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;

use unpopped::backend::{Backend, LowerError};
use unpopped::ir::{BinaryOp, Expr, OpDef, ScalarExpr, UnaryOp, View, input};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::kernel_lift::lift_elementwise_opdef;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap, run_kernel};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;

const F32: ElementKind = ElementKind::F32;
const F64: ElementKind = ElementKind::F64;
const I32: ElementKind = ElementKind::I32;

fn ctype(t: ElementKind) -> &'static str {
    match t {
        ElementKind::F32 => "float",
        ElementKind::F64 => "double",
        _ => "int",
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b},");
        s
    })
}

/// `n` distinct values that exercise rounding (not representable in a short decimal).
fn values(t: ElementKind, n: usize, seed: u32) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            let x = f64::from((i as u32 * 7919 + seed * 104_729) % 2003) / 97.0 - 10.0;
            match t {
                ElementKind::F64 => x.to_le_bytes().to_vec(),
                ElementKind::F32 => (x as f32).to_le_bytes().to_vec(),
                _ => (x as i32).to_le_bytes().to_vec(),
            }
        })
        .collect()
}

fn dense(t: ElementKind, shape: &[i64], seed: u32) -> TypedBuffer {
    let n = shape.iter().product::<i64>() as usize;
    let mut strides = vec![1i64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    TypedBuffer::new(t, shape.to_vec(), strides, values(t, n, seed))
}

fn zeroed(t: ElementKind, shape: &[i64]) -> TypedBuffer {
    let mut b = dense(t, shape, 0);
    b.bytes.fill(0);
    b
}

/// The C `main` that binds `args`, calls the kernel and prints the status then each writable
/// buffer's bytes.
fn driver(k: &KernelFn, name: &str, args: &[KernelArg]) -> String {
    let mut s = String::from("#include <stdio.h>\n#include <string.h>\nint main(void) {\n");
    let mut call = Vec::new();
    let mut outs = Vec::new();
    for (i, (p, a)) in k.params.iter().zip(args).enumerate() {
        match a {
            KernelArg::Buffer(b) => {
                let n = b.bytes.len() / (if b.dtype == F64 { 8 } else { 4 });
                let t = ctype(b.dtype);
                let _ = writeln!(s, "  static unsigned char r{i}[] = {{{}}};", hex(&b.bytes));
                let _ = writeln!(
                    s,
                    "  static {t} b{i}[{n}]; memcpy(b{i}, r{i}, sizeof r{i});"
                );
                let lst = |v: &[i64]| {
                    v.iter()
                        .map(|x| format!("{x}LL"))
                        .collect::<Vec<_>>()
                        .join(",")
                };
                let _ = writeln!(
                    s,
                    "  static long long s{i}[] = {{{}}}, t{i}[] = {{{}}};",
                    lst(&b.shape),
                    lst(&b.strides)
                );
                call.push(format!("b{i}, s{i}, t{i}, {}LL", b.base_offset));
                if matches!(&p.kind, unpopped::kernel::ParamKind::Buffer { role, .. } if *role != BufferRole::Read)
                {
                    outs.push((i, n * (if b.dtype == F64 { 8 } else { 4 })));
                }
            }
            KernelArg::Int(v) => call.push(format!("{v}")),
            KernelArg::Float(v) => call.push(format!("{v:?}")),
            other => panic!("an argument kind the driver cannot bind: {other:?}"),
        }
    }
    let _ = writeln!(
        s,
        "  int st = {name}({});\n  printf(\"%d\\n\", st);",
        call.join(", ")
    );
    for (i, len) in outs {
        let _ = writeln!(
            s,
            "  for (int j = 0; j < {len}; ++j) printf(\"%02x\", ((unsigned char*)b{i})[j]);\n  printf(\"\\n\");"
        );
    }
    s.push_str("  return 0;\n}\n");
    s
}

/// What the C program printed: its status and the bytes of each writable buffer.
struct Ran {
    status: i32,
    outs: Vec<Vec<u8>>,
}

fn run_c(tag: &str, k: &KernelFn, args: &[KernelArg]) -> Option<Ran> {
    let Some(compiler) = cc::find_compiler() else {
        eprintln!("SKIP {tag}: no host C compiler");
        return None;
    };
    let g = CpuC
        .lower_kernel(k)
        .unwrap_or_else(|e| panic!("{tag}: lowering refused: {e}"));
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("kernel_emit");
    std::fs::create_dir_all(&dir).unwrap();
    let (c, exe) = (dir.join(format!("{tag}.c")), dir.join(format!("{tag}.exe")));
    std::fs::write(&c, format!("{}\n{}", g.source, driver(k, &g.name, args))).unwrap();
    compiler
        .compile(&c, &exe, true)
        .unwrap_or_else(|e| panic!("{tag}: {e}\n{}", g.source));
    let out = Command::new(&exe).output().unwrap();
    assert!(out.status.success(), "{tag}: the program crashed");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let mut lines = text.lines();
    let status = lines.next().unwrap().trim().parse().unwrap();
    let outs = lines
        .map(|l| {
            (0..l.len() / 2)
                .map(|i| u8::from_str_radix(&l[2 * i..2 * i + 2], 16).unwrap())
                .collect()
        })
        .collect();
    Some(Ran { status, outs })
}

/// Distance in representable `f32` values; two NaNs are equal.
fn ulps(a: f32, b: f32) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::from(a.is_nan() != b.is_nan()) * u64::MAX;
    }
    let key = |x: f32| {
        let i = i64::from(x.to_bits() as i32);
        if i < 0 { i64::from(i32::MIN) - i } else { i }
    };
    key(a).abs_diff(key(b))
}

/// Output buffers agree: byte for byte, or for `F32` within `tol` ulps (the oracle evaluates a
/// compound `f32` expression at `f64` and rounds once, where `float` C rounds every operation;
/// the spec's tolerance mode, section 7.1). `F64` is always byte-exact.
fn same_bytes(tag: &str, got: &[Vec<u8>], want: &[Vec<u8>], dtype: ElementKind, tol: u64) {
    assert_eq!(got.len(), want.len(), "{tag}: buffer count");
    for (g, w) in got.iter().zip(want) {
        if dtype == F32 && tol > 0 {
            let f = |v: &[u8]| {
                v.chunks(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect::<Vec<_>>()
            };
            for (i, (a, b)) in f(g).into_iter().zip(f(w)).enumerate() {
                assert!(
                    ulps(a, b) <= tol,
                    "{tag}: element {i}: C {a:?} vs interpreter {b:?} ({} ulps > {tol})",
                    ulps(a, b)
                );
            }
        } else {
            assert_eq!(g, w, "{tag}: output bytes differ");
        }
    }
}

/// Run `k` both ways on `args` and assert they agree exactly.
fn agree(tag: &str, k: &KernelFn, args: Vec<KernelArg>) {
    agree_within(tag, k, args, 0);
}

/// [`agree`] with an `F32` tolerance of `tol` ulps.
fn agree_within(tag: &str, k: &KernelFn, args: Vec<KernelArg>, tol: u64) {
    let Some(c) = run_c(tag, k, &args) else {
        return;
    };
    let mut interp = args;
    let r = run_kernel(k, &mut interp);
    match r {
        Ok(()) => {
            assert_eq!(
                c.status, 0,
                "{tag}: the interpreter finished, the C reported {}",
                c.status
            );
            let want: Vec<Vec<u8>> =
                k.params
                    .iter()
                    .zip(&interp)
                    .filter_map(|(p, a)| match (&p.kind, a) {
                        (
                            unpopped::kernel::ParamKind::Buffer { role, .. },
                            KernelArg::Buffer(b),
                        ) if *role != BufferRole::Read => Some(b.bytes.clone()),
                        _ => None,
                    })
                    .collect();
            same_bytes(tag, &c.outs, &want, k.compute, tol);
        }
        Err(KernelTrap::OutOfBounds { .. }) => {
            assert_eq!(c.status, 1, "{tag}: interpreter OutOfBounds")
        }
        Err(KernelTrap::ZeroStep) => assert_eq!(c.status, 2, "{tag}: interpreter ZeroStep"),
        Err(e) => panic!("{tag}: unexpected interpreter trap {e:?}"),
    }
}

fn elementwise(name: &str, n_in: u8, dtype: ElementKind, body: ScalarExpr) -> OpDef {
    OpDef::elementwise(name, n_in, &[dtype], Expr(body))
}

fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

fn c(v: f64) -> ScalarExpr {
    ScalarExpr::Const(v)
}

/// The lifted kernel's arguments: inputs, a zeroed output, one `I32` extent per axis.
fn ew_args(dtype: ElementKind, n_in: usize, shape: &[i64]) -> Vec<KernelArg> {
    let mut a: Vec<KernelArg> = (0..n_in)
        .map(|i| KernelArg::Buffer(dense(dtype, shape, i as u32 + 1)))
        .collect();
    a.push(KernelArg::Buffer(zeroed(dtype, shape)));
    a.extend(shape.iter().map(|&e| KernelArg::Int(i128::from(e))));
    a
}

#[test]
fn elementwise_binary_with_constants_agrees_f32_and_f64() {
    // `x * 0.1 + y * 0.1`: three roundings. F64 is byte-exact with the interpreter; F32 is within
    // 3 ulps of it (single rounding at f64 versus one rounding per operation).
    for dtype in [F32, F64] {
        // Constants reach an emitter already rounded to the compute precision (the optimizer's
        // contract), so the F32 literal is the f32 nearest 0.1, not the double.
        let tenth = if dtype == F32 {
            f64::from(0.1_f32)
        } else {
            0.1
        };
        // `Abs` keeps both addends positive: with a cancelling sum a ulp-of-the-result bound
        // would be unbounded and the comparison would measure the data, not the emitter.
        let abs = |e| ScalarExpr::Unary(UnaryOp::Abs, bx(e));
        let body = ScalarExpr::Add(
            bx(ScalarExpr::Mul(bx(abs(input(0).0)), bx(c(tenth)))),
            bx(ScalarExpr::Mul(bx(abs(input(1).0)), bx(c(tenth)))),
        );
        let k = lift_elementwise_opdef(&elementwise("consts", 2, dtype, body), dtype, 2).unwrap();
        agree_within(
            &format!("consts_{dtype:?}"),
            &k,
            ew_args(dtype, 2, &[5, 7]),
            3,
        );
    }
}

#[test]
fn f32_arithmetic_rounds_every_operation_like_float() {
    // The interpreter cannot see this: a C literal left a `double` evaluates the whole
    // expression at double and rounds once, which sits inside the interpreter's tolerance.
    // So the emitted C is also held to Rust's own per-operation `f32` arithmetic, exactly.
    let tenth = 0.1_f32;
    let body = ScalarExpr::Add(
        bx(ScalarExpr::Mul(bx(input(0).0), bx(c(f64::from(tenth))))),
        bx(ScalarExpr::Mul(bx(input(1).0), bx(c(f64::from(tenth))))),
    );
    let k = lift_elementwise_opdef(&elementwise("rne", 2, F32, body), F32, 1).unwrap();
    let args = ew_args(F32, 2, &[64]);
    let Some(ran) = run_c("per_op_rounding", &k, &args) else {
        return;
    };
    let f = |i: usize| match &args[i] {
        KernelArg::Buffer(b) => b
            .bytes
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<_>>(),
        _ => unreachable!(),
    };
    let want: Vec<u8> = f(0)
        .iter()
        .zip(f(1))
        .flat_map(|(x, y)| ((x * tenth) + (y * tenth)).to_le_bytes())
        .collect();
    assert_eq!(
        ran.outs[0], want,
        "the emitted C does not round per operation"
    );
}

#[test]
fn elementwise_unary_and_select_ops_agree() {
    let body = ScalarExpr::Binary(
        BinaryOp::Max,
        bx(ScalarExpr::Unary(
            UnaryOp::Sqrt,
            bx(ScalarExpr::Unary(UnaryOp::Abs, bx(input(0).0))),
        )),
        bx(ScalarExpr::Unary(UnaryOp::Neg, bx(input(1).0))),
    );
    let k = lift_elementwise_opdef(&elementwise("un", 2, F32, body), F32, 1).unwrap();
    agree("unary", &k, ew_args(F32, 2, &[33]));
}

#[test]
fn a_runtime_base_offset_and_a_flipped_stride_agree() {
    // Input 0 sits in junk padding (offset 3) and input 1 is read backwards (stride -1): both are
    // the buffer's own layout, which the emitted address arithmetic must honour.
    let k = lift_elementwise_opdef(
        &elementwise(
            "off",
            2,
            F32,
            ScalarExpr::Sub(
                bx(input(0).0),
                bx(ScalarExpr::Mul(bx(input(1).0), bx(c(2.0)))),
            ),
        ),
        F32,
        1,
    )
    .unwrap();
    let mut args = ew_args(F32, 2, &[5]);
    let padded = |mut b: TypedBuffer, pad: usize| {
        b.bytes = std::iter::repeat_n(0x7f_u8, pad * 4)
            .chain(b.bytes)
            .collect();
        b
    };
    let KernelArg::Buffer(a0) = args[0].clone() else {
        unreachable!()
    };
    let mut a0 = padded(a0, 3);
    a0.base_offset = 3;
    args[0] = KernelArg::Buffer(a0);
    let KernelArg::Buffer(a1) = args[1].clone() else {
        unreachable!()
    };
    let mut a1 = a1;
    a1.strides = vec![-1];
    a1.base_offset = 4;
    args[1] = KernelArg::Buffer(a1);
    agree("offset_and_flip", &k, args);
}

#[test]
fn a_permuted_view_agrees() {
    // Input 1 is read through `Permute`: the producer coordinates swap, so a wrong axis order
    // in the emitted address arithmetic reads another element.
    let mut op = elementwise(
        "perm",
        2,
        F32,
        ScalarExpr::Sub(
            bx(input(0).0),
            bx(ScalarExpr::Mul(bx(input(1).0), bx(c(2.0)))),
        ),
    );
    op.views = vec![View::Identity, View::Permute { perm: vec![1, 0] }];
    let k = lift_elementwise_opdef(&op, F32, 2).unwrap();
    let mut args = ew_args(F32, 2, &[4, 4]);
    args[1] = KernelArg::Buffer(dense(F32, &[4, 4], 9));
    agree("permute", &k, args);
}

/// A hand-written kernel with every control construct: `For` with a negative step, nested loops
/// with `Break` and `Continue`, a `While`, an `If` and an early `Return`.
fn control_flow() -> KernelFn {
    let l = ScalarExpr::Local;
    let ci = |v| ScalarExpr::ConstInt(v, I32);
    let cmp = |op, a, b| ScalarExpr::Binary(op, bx(a), bx(b));
    let acc = |v: f64| Stmt::Assign {
        local: 2,
        value: ScalarExpr::Add(bx(l(2)), bx(c(v))),
    };
    let inner = Stmt::For {
        local: 3,
        start: ci(0),
        end: ci(6),
        step: ci(1),
        body: Block(vec![
            Stmt::If {
                cond: cmp(BinaryOp::CmpEq, l(3), ci(1)),
                then: Block(vec![Stmt::Continue]),
                els: Block(vec![]),
            },
            Stmt::If {
                cond: cmp(BinaryOp::CmpEq, l(3), ci(3)),
                then: Block(vec![Stmt::Break]),
                els: Block(vec![]),
            },
            acc(1.5),
        ]),
    };
    let outer = Stmt::For {
        local: 4,
        start: l(0),
        end: ci(-1),
        step: ci(-1),
        body: Block(vec![inner, acc(10.25)]),
    };
    let tail = [
        Stmt::If {
            cond: cmp(BinaryOp::CmpLt, l(0), ci(3)),
            then: Block(vec![acc(1000.0)]),
            els: Block(vec![acc(2000.0)]),
        },
        Stmt::While {
            cond: cmp(BinaryOp::CmpLt, l(5), ci(3)),
            body: Block(vec![
                Stmt::Assign {
                    local: 5,
                    value: ScalarExpr::Add(bx(l(5)), bx(ci(1))),
                },
                acc(100.0),
            ]),
        },
        Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![ci(0)],
            },
            value: l(2),
        },
        Stmt::If {
            cond: cmp(BinaryOp::CmpEq, l(0), ci(2)),
            then: Block(vec![Stmt::Return]),
            els: Block(vec![]),
        },
        Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![ci(1)],
            },
            value: l(2),
        },
    ];
    let mut body = vec![
        Stmt::Let {
            local: 2,
            init: c(0.0),
        },
        Stmt::Let {
            local: 5,
            init: ci(0),
        },
        outer,
    ];
    body.extend(tail);
    KernelFn::new(
        "control",
        F32,
        vec![
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
            Param::scalar("n", I32, 0),
        ],
        vec![
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
        ],
        Block(body),
    )
}

#[test]
fn control_flow_agrees_over_many_extents() {
    for n in [0, 1, 2, 3, 4, 7] {
        let args = vec![KernelArg::Buffer(zeroed(F32, &[2])), KernelArg::Int(n)];
        agree(&format!("control_{n}"), &control_flow(), args);
    }
}

#[test]
fn an_out_of_bounds_access_and_a_zero_step_trap_the_same_way() {
    let oob = KernelFn::new(
        "oob",
        F32,
        vec![
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
            Param::scalar("i", I32, 0),
        ],
        vec![LocalDecl::scalar(I32)],
        Block(vec![Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![ScalarExpr::Local(0)],
            },
            value: c(1.0),
        }]),
    );
    for i in [-1, 0, 1, 2, 3] {
        agree(
            &format!("oob_{i}"),
            &oob,
            vec![KernelArg::Buffer(zeroed(F32, &[3])), KernelArg::Int(i)],
        );
    }
    let zero = KernelFn::new(
        "zero",
        F32,
        vec![
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
            Param::scalar("s", I32, 0),
        ],
        vec![LocalDecl::scalar(I32), LocalDecl::scalar(I32)],
        Block(vec![Stmt::For {
            local: 1,
            start: ScalarExpr::ConstInt(0, I32),
            end: ScalarExpr::ConstInt(3, I32),
            step: ScalarExpr::Local(0),
            body: Block(vec![]),
        }]),
    );
    agree(
        "zero_step",
        &zero,
        vec![KernelArg::Buffer(zeroed(F32, &[1])), KernelArg::Int(0)],
    );
    agree(
        "one_step",
        &zero,
        vec![KernelArg::Buffer(zeroed(F32, &[1])), KernelArg::Int(1)],
    );
}

#[test]
fn a_kernel_outside_the_emitters_coverage_is_the_typed_kernel_decline() {
    use unpopped::exec::{LaunchShape, Scope};
    let base = control_flow();
    let launched = base
        .clone()
        .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    let barrier = {
        let mut k = launched.clone();
        k.body.0.insert(
            0,
            Stmt::Barrier {
                scope: Scope::Block,
            },
        );
        k
    };
    let f16 = KernelFn::new("h", ElementKind::F16, vec![], vec![], Block(vec![]));
    let ill_formed = KernelFn::new("bad", F32, vec![], vec![], Block(vec![Stmt::Break]));
    for (what, k) in [
        ("launch", launched),
        ("barrier", barrier),
        ("f16", f16),
        ("ill-formed", ill_formed),
    ] {
        let e = CpuC.lower_kernel(&k).expect_err(what);
        assert!(
            matches!(e, LowerError::UnsupportedKernel { .. }),
            "{what}: {e:?}"
        );
    }
}
