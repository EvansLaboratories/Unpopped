//! Consumer-shaped end-to-end test: generate → compile → **launch** → compare.
//!
//! Every other test in this crate inspects the emitted *text*. This one runs it.
//! That distinction is the whole point: a regression shipped in
//! `baracuda-kernelgen` alpha.78 in which a kernel synthesized, loaded and
//! launched with no error and simply never wrote its output. Byte-identity
//! goldens could not catch it — the emitted bytes were unchanged — and neither
//! side of the seam had a routine test that launched a kernel and looked at the
//! answer. This is the neutral half of closing that gap.
//!
//! # Why NaN is the sentinel
//!
//! Filling the output with `NAN` before the call is what makes "wrote zeros"
//! distinguishable from "never wrote". Zero-filling cannot tell those apart, and
//! all-zeros-with-no-error was exactly the failure mode. Any surviving NaN in a
//! cell the kernel was supposed to write is a *never-wrote* failure, reported
//! separately from a wrong-value failure. (Borrowed from Baracuda's on-device
//! `alpha78_relu_add_ondevice` harness — the trick is theirs.)
//!
//! # Skipping
//!
//! This needs a host C compiler. Where none is found the test **skips loudly**
//! rather than failing: it is a real-execution test, and a machine without a
//! toolchain cannot run one. It must never silently pass, so the skip prints why.

use std::path::{Path, PathBuf};
use std::process::Command;

use unpopped::cpu_c::CpuC;
use unpopped::ir::{OpDef, input};
use unpopped::oracle::{Fidelity, TypedBuffer, compare, evaluate};
use unpopped::{build_plan, generate};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

/// A host C compiler and how to drive it.
struct CCompiler {
    /// Base command, already carrying whatever environment it needs (the MSVC
    /// tool from `cc`'s registry lookup carries INCLUDE/LIB; a PATH `cc` does not
    /// need any).
    cmd: &'static str,
    /// `cl.exe`-style flags (`-Fe:`) rather than GNU-style (`-o`).
    msvc: bool,
    /// Whether the link line needs `-lm`.
    ///
    /// This is *not* implied by the driver being GNU-style. Visual Studio ships
    /// a `clang` that targets `x86_64-pc-windows-msvc`, where the math functions
    /// live in the UCRT and there is no `m.lib` at all — passing `-lm` there is
    /// a hard `lld-link: could not open 'm.lib'`. So ask the driver what it
    /// targets instead of assuming from its name.
    needs_libm: bool,
}

/// Locate a host C compiler, or `None`.
///
/// MSVC is deliberately handled by *not* trying to configure it here: bare
/// `cl.exe` fails with "no include path set" unless a `vcvars` environment is
/// present, so we only accept it when it is already on `PATH` (i.e. the caller
/// is in a developer shell). Everything else is a plain PATH lookup.
fn find_compiler() -> Option<CCompiler> {
    for (cmd, msvc) in [
        ("cc", false),
        ("gcc", false),
        ("clang", false),
        ("cl", true),
    ] {
        if msvc {
            // `cl` with no arguments prints its banner and exits non-zero; a
            // successful *spawn* is the signal that it is on PATH.
            if Command::new(cmd).output().is_ok() {
                return Some(CCompiler {
                    cmd,
                    msvc,
                    needs_libm: false,
                });
            }
            continue;
        }
        if Command::new(cmd).arg("--version").output().is_err() {
            continue;
        }
        // `-dumpmachine` is understood by gcc, clang and any `cc` symlinked to
        // either. If it fails, fall back to needing libm — the Unix default.
        let triple = Command::new(cmd)
            .arg("-dumpmachine")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        let needs_libm = !triple.contains("windows-msvc");
        return Some(CCompiler {
            cmd,
            msvc,
            needs_libm,
        });
    }
    None
}

impl CCompiler {
    /// Compile `src` to `exe`. Returns the compiler's diagnostics on failure.
    fn compile(&self, src: &Path, exe: &Path) -> Result<(), String> {
        let mut c = Command::new(self.cmd);
        if self.msvc {
            c.arg("-nologo")
                .arg(src)
                .arg(format!("-Fe:{}", exe.display()));
        } else {
            c.arg(src).arg("-o").arg(exe).arg("-O0");
            if self.needs_libm {
                c.arg("-lm");
            }
        }
        let out = c.output().map_err(|e| format!("spawn failed: {e}"))?;
        if out.status.success() {
            return Ok(());
        }
        Err(format!(
            "compile failed\n--- stdout ---\n{}\n--- stderr ---\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}

/// Wrap an emitted kernel in a `main` that NaN-fills the output, calls it, and
/// prints every output element at full `f64` precision.
///
/// The prototype is re-declared rather than included, so the harness also checks
/// that the emitted signature is what the caller believes it is — a mismatch is
/// a compile error here rather than undefined behaviour at run time.
fn harness(kernel_src: &str, name: &str, n_in: usize, inputs: &[Vec<f32>]) -> String {
    let n = inputs[0].len();
    let mut s = String::new();
    s.push_str(kernel_src);
    s.push_str("\n#include <stdio.h>\n\nint main(void) {\n");

    for (i, data) in inputs.iter().enumerate() {
        let lits: Vec<String> = data.iter().map(|v| format!("{v:?}f")).collect();
        s.push_str(&format!(
            "    const float in{i}[{n}] = {{{}}};\n",
            lits.join(", ")
        ));
    }
    // The sentinel. Every cell the kernel is contracted to write must be
    // overwritten; a survivor is a never-wrote bug, not a wrong-value bug.
    s.push_str(&format!("    float out[{n}];\n"));
    s.push_str(&format!(
        "    for (int i = 0; i < {n}; ++i) out[i] = NAN;\n"
    ));

    let args: Vec<String> = (0..n_in).map(|i| format!("in{i}")).collect();
    s.push_str(&format!("    {name}({}, out, {n});\n", args.join(", ")));
    s.push_str(&format!(
        "    for (int i = 0; i < {n}; ++i) printf(\"%.17g\\n\", (double)out[i]);\n"
    ));
    s.push_str("    return 0;\n}\n");
    s
}

/// Generate, compile, run, and return the output elements.
fn run_kernel(
    cc: &CCompiler,
    dir: &Path,
    tag: &str,
    op: &OpDef,
    dtype: ElementKind,
    cat: OpCategory,
    inputs: &[Vec<f32>],
) -> Result<(Vec<f64>, TypedBuffer), String> {
    let n = inputs[0].len() as i64;
    let d = OperandDesc::new(1, &[n], &[1], dtype, 256);
    let mut operands = vec![d; inputs.len()];
    operands.push(d); // output
    let key = structure_key(cat, &operands, ArchSku::Sm89);

    let kernel = generate(op, &key, &CpuC);
    let src = harness(&kernel.source, &kernel.name, inputs.len(), inputs);

    let c_file = dir.join(format!("{tag}.c"));
    let exe = dir.join(format!("{tag}.exe"));
    std::fs::write(&c_file, &src).map_err(|e| format!("write: {e}"))?;
    cc.compile(&c_file, &exe)?;

    let out = Command::new(&exe)
        .output()
        .map_err(|e| format!("run failed: {e}"))?;
    if !out.status.success() {
        return Err(format!("kernel exited {:?}", out.status.code()));
    }
    let actual: Vec<f64> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse::<f64>().unwrap_or(f64::NAN))
        .collect();

    // The oracle leg — an independent f64 evaluator that shares no lowering code
    // with the emitter, so a bug cannot hide in both.
    let plan = build_plan(op, &key);
    let bufs: Vec<TypedBuffer> = inputs
        .iter()
        .map(|d| TypedBuffer::from_f32(&[n], d))
        .collect();
    let expected = evaluate(&plan, &operands, &bufs, &[]);
    Ok((actual, expected.into_iter().next().unwrap()))
}

/// Assert the kernel wrote every cell, then that it wrote the right values.
///
/// The two failures are reported separately on purpose: "never wrote" and "wrote
/// the wrong number" have completely different causes, and collapsing them is
/// what made the alpha.78 regression a multi-hour bisect instead of a one-line
/// diagnosis.
fn assert_wrote_and_correct(tag: &str, actual: &[f64], expected: &TypedBuffer) {
    let want = expected.to_f64_vec();
    assert_eq!(
        actual.len(),
        want.len(),
        "{tag}: kernel printed {} values, expected {}",
        actual.len(),
        want.len()
    );

    let unwritten: Vec<usize> = actual
        .iter()
        .enumerate()
        .filter(|(i, v)| v.is_nan() && !want[*i].is_nan())
        .map(|(i, _)| i)
        .collect();
    assert!(
        unwritten.is_empty(),
        "{tag}: NEVER-WROTE — the NaN sentinel survived at {} of {} indices {:?}. \
         The kernel launched without error and left these cells untouched; this is \
         the alpha.78 class, NOT a numeric error.",
        unwritten.len(),
        actual.len(),
        &unwritten[..unwritten.len().min(8)]
    );

    let got = TypedBuffer::from_f32(
        &[actual.len() as i64],
        &actual.iter().map(|v| *v as f32).collect::<Vec<_>>(),
    );
    // BitExact: `add`/`mul` at f32 are exact-wrapping arithmetic, and the C
    // emitter and the f64 oracle must agree to the bit. A tolerance here would
    // hide precisely the class of emitter bug this test exists to catch.
    if let Err(e) = compare(expected, &got, Fidelity::BitExact) {
        panic!("{tag}: WRONG-VALUE vs the CPU oracle — {e}");
    }
}

#[test]
fn cpu_c_kernels_launch_and_match_the_oracle() {
    let Some(cc) = find_compiler() else {
        eprintln!(
            "SKIP cpu_c_kernels_launch_and_match_the_oracle: no host C compiler \
             (tried cc, gcc, clang, cl). This test compiles and RUNS generated \
             kernels; without a toolchain it cannot execute. On Windows, run from \
             a Visual Studio developer shell so `cl` is on PATH."
        );
        return;
    };

    let dir: PathBuf = std::env::temp_dir().join("unpopped-e2e");
    std::fs::create_dir_all(&dir).expect("scratch dir");

    let a: Vec<f32> = vec![3.0, -1.5, 0.0, 2.25, -7.0, 10.0, 0.5];
    let b: Vec<f32> = vec![0.0, 1.5, 4.0, -2.25, 5.0, -10.0, 0.25];

    // add: the simplest thing that can fail to write.
    let add = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
    let (actual, expected) = run_kernel(
        &cc,
        &dir,
        "add_f32",
        &add,
        ElementKind::F32,
        OpCategory::BinaryElementwise,
        &[a.clone(), b.clone()],
    )
    .expect("add_f32 end-to-end");
    assert_wrote_and_correct("add_f32", &actual, &expected);

    // mul: a second shape, to catch a harness that only works for one.
    let mul = OpDef::elementwise("mul", 2, &[ElementKind::F32], input(0) * input(1));
    let (actual, expected) = run_kernel(
        &cc,
        &dir,
        "mul_f32",
        &mul,
        ElementKind::F32,
        OpCategory::BinaryElementwise,
        &[a.clone(), b.clone()],
    )
    .expect("mul_f32 end-to-end");
    assert_wrote_and_correct("mul_f32", &actual, &expected);
}
