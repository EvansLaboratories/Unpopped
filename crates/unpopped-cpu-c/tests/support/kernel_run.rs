// SPDX-License-Identifier: MIT OR Apache-2.0
//! The compile-and-run side of the kernel-emitter differential: bind a kernel's arguments in a C
//! `main`, run the emitted function, and compare its output bytes with `run_kernel`'s.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;

use unpopped::backend::Backend;
use unpopped::kernel::{BufferRole, KernelFn};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap, run_kernel};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

use crate::cc;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const F64: ElementKind = ElementKind::F64;

pub(crate) fn ctype(t: ElementKind) -> &'static str {
    match t {
        ElementKind::F32 => "float",
        ElementKind::F64 => "double",
        _ => "int",
    }
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b},");
        s
    })
}

/// `n` distinct values that exercise rounding (not representable in a short decimal).
pub(crate) fn values(t: ElementKind, n: usize, seed: u32) -> Vec<u8> {
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

pub(crate) fn dense(t: ElementKind, shape: &[i64], seed: u32) -> TypedBuffer {
    let n = shape.iter().product::<i64>() as usize;
    let mut strides = vec![1i64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    TypedBuffer::new(t, shape.to_vec(), strides, values(t, n, seed))
}

pub(crate) fn zeroed(t: ElementKind, shape: &[i64]) -> TypedBuffer {
    let mut b = dense(t, shape, 0);
    b.bytes.fill(0);
    b
}

/// The C declarations that give buffer argument `i` its storage, shape and strides.
pub(crate) fn bind_buffer(i: usize, b: &TypedBuffer) -> String {
    let n = b.bytes.len() / (if b.dtype == F64 { 8 } else { 4 });
    let lst = |v: &[i64]| {
        v.iter()
            .map(|x| format!("{x}LL"))
            .collect::<Vec<_>>()
            .join(",")
    };
    format!(
        "  static unsigned char r{i}[] = {{{}}};
  static {} b{i}[{n}]; memcpy(b{i}, r{i}, sizeof r{i});
  static long long s{i}[] = {{{}}}, t{i}[] = {{{}}};
",
        hex(&b.bytes),
        ctype(b.dtype),
        lst(&b.shape),
        lst(&b.strides)
    )
}

/// The C `main` that binds `args`, calls the kernel and prints the status then each writable
/// buffer's bytes.
pub(crate) fn driver(k: &KernelFn, name: &str, args: &[KernelArg]) -> String {
    let mut s = String::from("#include <stdio.h>\n#include <string.h>\nint main(void) {\n");
    let mut call = Vec::new();
    let mut outs = Vec::new();
    for (i, (p, a)) in k.params.iter().zip(args).enumerate() {
        match a {
            KernelArg::Buffer(b) => {
                let n = b.bytes.len() / (if b.dtype == F64 { 8 } else { 4 });
                s.push_str(&bind_buffer(i, b));
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
pub(crate) struct Ran {
    pub(crate) status: i32,
    pub(crate) outs: Vec<Vec<u8>>,
}

pub(crate) fn run_c(tag: &str, k: &KernelFn, args: &[KernelArg]) -> Option<Ran> {
    run_c_mutated(tag, k, args, |src| src)
}

/// [`run_c`] with `mutate` applied to the generated source before it is compiled: a seeded
/// mutant of the emitter's output, for the tests that prove a construct is observable.
pub(crate) fn run_c_mutated(
    tag: &str,
    k: &KernelFn,
    args: &[KernelArg],
    mutate: impl FnOnce(String) -> String,
) -> Option<Ran> {
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
    let source = mutate(g.source.clone());
    std::fs::write(
        &c,
        format!(
            "{source}
{}",
            driver(k, &g.name, args)
        ),
    )
    .unwrap();
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
pub(crate) fn ulps(a: f32, b: f32) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::from(a.is_nan() != b.is_nan()) * u64::MAX;
    }
    let key = |x: f32| {
        let i = i64::from(x.to_bits() as i32);
        if i < 0 { i64::from(i32::MIN) - i } else { i }
    };
    key(a).abs_diff(key(b))
}

/// `v` with every NaN replaced by one canonical NaN.
///
/// **What this stops distinguishing: the sign and payload of a NaN, and nothing else.** An
/// operation that makes a NaN (`inf - inf`, `0 * inf`) gets the host's choice of sign (x86 makes
/// the negative one) where the interpreter makes the positive one. That is not semantic here: the
/// spec treats NaN as one value and no op in the kernel language reads a NaN's sign or payload.
/// Every other bit pattern, `-0.0` vs `0.0` and the infinities included, is still compared
/// exactly; `nan_is_equal_only_to_nan` (kernel_emit_contraction_window_agrees.rs) keeps it so.
pub(crate) fn canonical_nans(v: &[u8], dtype: ElementKind) -> Vec<u8> {
    let w = if dtype == F64 { 8 } else { 4 };
    v.chunks(w)
        .flat_map(|c| {
            let nan = if w == 8 {
                f64::from_le_bytes(c.try_into().unwrap()).is_nan()
            } else {
                f32::from_le_bytes(c.try_into().unwrap()).is_nan()
            };
            match (nan, w) {
                (false, _) => c.to_vec(),
                (true, 8) => f64::NAN.to_le_bytes().to_vec(),
                (true, _) => f32::NAN.to_le_bytes().to_vec(),
            }
        })
        .collect()
}

/// Output buffers agree: bit-exact except NaN sign/payload ([`canonical_nans`]), or for `F32` within `tol` ulps (the oracle evaluates a
/// compound `f32` expression at `f64` and rounds once, where `float` C rounds every operation;
/// the spec's tolerance mode, section 7.1). `F64` is always byte-exact.
pub(crate) fn same_bytes(
    tag: &str,
    got: &[Vec<u8>],
    want: &[Vec<u8>],
    dtype: ElementKind,
    tol: u64,
) {
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
        } else if dtype == F32 || dtype == F64 {
            assert_eq!(
                canonical_nans(g, dtype),
                canonical_nans(w, dtype),
                "{tag}: output bytes differ"
            );
        } else {
            assert_eq!(g, w, "{tag}: output bytes differ");
        }
    }
}

/// Run `k` both ways on `args` and assert they agree exactly.
pub(crate) fn agree(tag: &str, k: &KernelFn, args: Vec<KernelArg>) {
    agree_within(tag, k, args, 0);
}

/// [`agree`] with an `F32` tolerance of `tol` ulps.
pub(crate) fn agree_within(tag: &str, k: &KernelFn, args: Vec<KernelArg>, tol: u64) {
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
