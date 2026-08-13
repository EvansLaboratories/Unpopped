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

/// `f32::NAN` shortened so the input tables below line up column-wise.
const NAN_F: f32 = f32::NAN;

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
    ///
    /// `optimize` selects `-O2` / `/O2` instead of the unoptimized default. It
    /// matters for exactly one caller and it is not a performance knob: the
    /// NaN-propagation test is checking whether an **optimizer** contracts the
    /// emitted ternary into a NaN-suppressing max instruction. Run at `-O0` that
    /// test passes trivially and proves nothing, because there is no optimizer to
    /// do the contracting. The value-correctness tests stay unoptimized so a
    /// failure there is unambiguously the emitter's rather than the compiler's.
    fn compile(&self, src: &Path, exe: &Path, optimize: bool) -> Result<(), String> {
        let mut c = Command::new(self.cmd);
        if self.msvc {
            c.arg("-nologo")
                .arg(if optimize { "-O2" } else { "-Od" })
                .arg(src)
                .arg(format!("-Fe:{}", exe.display()));
        } else {
            c.arg(src)
                .arg("-o")
                .arg(exe)
                .arg(if optimize { "-O2" } else { "-O0" });
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

/// A C float literal for `v`.
///
/// Rust's `{:?}` renders these as `NaN` and `inf`, which are not C. The kernel
/// source already pulls in `<math.h>` (the harness's own sentinel loop uses
/// `NAN`), so the macros are available.
fn c_float_lit(v: f32) -> String {
    if v.is_nan() {
        // The payload is not preserved through the printf/parse hop, and it does
        // not need to be — the tests that use NaN inputs assert NaN-ness, not a
        // specific bit pattern. See `max_propagates_nan_through_a_real_compiler`.
        if v.is_sign_negative() {
            "-(float)NAN".to_string()
        } else {
            "(float)NAN".to_string()
        }
    } else if v.is_infinite() {
        if v.is_sign_negative() {
            "-(float)INFINITY".to_string()
        } else {
            "(float)INFINITY".to_string()
        }
    } else {
        format!("{v:?}f")
    }
}

/// Parse one printed output element.
///
/// Deliberately **not** `parse().unwrap_or(NAN)`. Falling back to NaN on a parse
/// failure silently converts "the kernel printed something unreadable" into "the
/// kernel produced NaN", which is a real vacuity hazard: it would let a NaN
/// assertion pass on garbage output, and it makes an unreadable line masquerade
/// as the NEVER-WROTE sentinel and get diagnosed as the wrong bug entirely.
///
/// NaN spelling is platform-dependent — glibc prints `nan` / `-nan`, MSVC prints
/// `-nan(ind)` — so accept those explicitly and panic on anything else.
fn parse_out_elem(line: &str) -> f64 {
    let t = line.trim();
    let lower = t.to_ascii_lowercase();
    if lower.starts_with("nan") || lower.starts_with("-nan") || lower.starts_with("+nan") {
        return f64::NAN;
    }
    t.parse::<f64>()
        .unwrap_or_else(|e| panic!("kernel printed an unparseable output element {t:?}: {e}"))
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
        let lits: Vec<String> = data.iter().map(|v| c_float_lit(*v)).collect();
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
    optimize: bool,
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
    cc.compile(&c_file, &exe, optimize)?;

    let out = Command::new(&exe)
        .output()
        .map_err(|e| format!("run failed: {e}"))?;
    if !out.status.success() {
        return Err(format!("kernel exited {:?}", out.status.code()));
    }
    let actual: Vec<f64> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(parse_out_elem)
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
        false,
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
        false,
    )
    .expect("mul_f32 end-to-end");
    assert_wrote_and_correct("mul_f32", &actual, &expected);
}

/// `Max` propagates NaN **after the C compiler has had its way with the source**.
///
/// `docs/conformance.md` N2 makes NaN propagation through `Max`/`Min` normative:
/// if either operand is NaN the result is NaN. That is *not* C `fmax`, not IEEE
/// `maxNum`, and not GLSL `max` — every C-family target offers a built-in with
/// the opposite behavior, so the emitter deliberately refuses them and spells the
/// ternary out longhand:
///
/// ```c
/// (a != a ? a : (b != b ? b : (a >= b ? a : b)))
/// ```
///
/// # Why a text golden cannot cover this
///
/// The rule is implemented by *writing* that ternary and then assuming no
/// compiler in the chain contracts it back into a max instruction. Every other
/// test in this crate compares emitted source, which is exactly the layer at
/// which the ternary is still present — so the assumption they all rest on is the
/// one thing they structurally cannot check. Only compiling and running with NaN
/// input can. (The same assumption is unverified on CUDA, where the chain is
/// nvcc → ptxas → driver JIT and PTX `max.f32` is NaN-suppressing without the
/// `.NaN` modifier. Settling that needs the on-device equivalent of this test.)
///
/// # Why the NaN sentinel is not used here
///
/// The other tests pre-fill the output with NaN so a survivor means NEVER-WROTE.
/// That is unusable when NaN is the *expected answer*. Instead the input mixes
/// both cases: lanes 0–1 have a NaN operand and must produce NaN, lanes 2–3 have
/// none and must produce an exact finite value. The finite lanes are the positive
/// control — they prove the kernel ran and computed, so an all-NaN output from a
/// kernel that never wrote cannot pass.
///
/// # Why the SINGLE-NaN lanes are the ones that matter
///
/// Verified by mutation: replacing the emitter's ternary with `fmaxf` produces
/// `[1.0, 1.0, 3.0, 3.0, -1.0, 5.0, NaN]`. Lane 6 — where *both* operands are
/// NaN — still yields NaN, because `fmax(NaN, NaN)` is NaN. A test built only on
/// the both-NaN case would pass the exact mutation this exists to catch. Lanes
/// 0 and 1, with one NaN operand each, are what discriminate; lane 6 is kept only
/// because it is free and covers the fold.
#[test]
fn max_propagates_nan_through_a_real_compiler() {
    let Some(cc) = find_compiler() else {
        eprintln!(
            "SKIP max_propagates_nan_through_a_real_compiler: no host C compiler \
             (tried cc, gcc, clang, cl). This test compiles and RUNS a kernel to \
             check that NaN propagation survives the optimizer; without a \
             toolchain it cannot execute."
        );
        return;
    };

    let dir: PathBuf = std::env::temp_dir().join("unpopped-e2e");
    std::fs::create_dir_all(&dir).expect("scratch dir");

    // Seven lanes, not four: the planner vectorizes a 4-element contiguous cell
    // to `Vectorized { width: 4 }`, which the CpuC v1 emitter declines. Seven
    // keeps it on the scalar path this backend serves.
    //
    // lane:            0    1    2    3     4    5     6
    // NaN operand:     a    b    -    -     -    -    both
    let a: Vec<f32> = vec![NAN_F, 1.0, 2.0, 3.0, -1.0, 5.0, NAN_F];
    let b: Vec<f32> = vec![1.0, NAN_F, 3.0, 2.0, -2.0, 4.0, NAN_F];
    // Lanes 2/3 are ordered oppositely and 4 is negative, so a lowering that
    // returns one fixed operand — or that confuses max with min — fails the
    // controls rather than sliding through.
    const NAN_LANES: &[usize] = &[0, 1, 6];
    const FINITE: &[(usize, f64)] = &[(2, 3.0), (3, 3.0), (4, -1.0), (5, 5.0)];

    let max = OpDef::elementwise("maxop", 2, &[ElementKind::F32], input(0).max(input(1)));
    let (actual, expected) = run_kernel(
        &cc,
        &dir,
        "max_nan_f32",
        &max,
        ElementKind::F32,
        OpCategory::BinaryElementwise,
        &[a, b],
        // OPTIMIZED — the whole point. See `CCompiler::compile`.
        true,
    )
    .expect("max_nan_f32 end-to-end");

    let want = expected.to_f64_vec();
    assert_eq!(actual.len(), 7, "expected 7 output lanes, got {actual:?}");

    // Harness precondition: the ORACLE must itself say the NaN lanes are NaN. If
    // this trips, the oracle stopped implementing N2 and the rest of the test
    // would be checking the emitter against a reference that no longer encodes
    // the rule.
    for &lane in NAN_LANES {
        assert!(
            want[lane].is_nan(),
            "oracle no longer propagates NaN through Max at lane {lane} — it \
             produced {want:?}. Conformance rule N2 is what this test protects, \
             and the oracle is the reference for it."
        );
    }

    // The claim.
    for &lane in NAN_LANES {
        assert!(
            actual[lane].is_nan(),
            "NaN was NOT propagated at lane {lane}: the compiled kernel gave {} \
             where N2 requires NaN. The emitter spells a NaN-checking ternary, so \
             a finite result here means something in the C toolchain contracted it \
             into a NaN-suppressing max. Every source-level golden in this crate is \
             blind to that, which is why this test compiles and runs. \
             Full output: {actual:?}",
            actual[lane]
        );
    }

    // The positive control: without these, a kernel that wrote NaN everywhere —
    // or never wrote at all, leaving the caller's buffer untouched — would
    // satisfy every assertion above.
    for &(lane, expect) in FINITE {
        assert_eq!(
            actual[lane], expect,
            "finite lane {lane} is the positive control and must be exactly \
             {expect}, got {}. If this fails, the kernel is not computing max at \
             all and the NaN lanes prove nothing. Full output: {actual:?}",
            actual[lane]
        );
    }
}

/// **16-bit integer wrapping agrees between the emitted C and the oracle.**
///
/// `S16`/`U16` were added to the vocabulary to close a KISS §6.1 conformance gap,
/// and then wired through the *lowering* path — `scalar_ctype` spells `short`,
/// `is_int_dtype` admits them, and the oracle models their arithmetic. That last
/// part is where a dtype gets silently wrong: C promotes `short` to `int`, does
/// the arithmetic at 32 bits, and truncates on the store, so the observable
/// result of an overflowing `add` is a *wrapped* 16-bit value. Wire the ctype
/// without teaching the oracle that, and the emitter wraps while the oracle
/// computes in `f64` — the two disagree only on overflow, which no
/// small-value test would ever reach.
///
/// So the inputs are chosen to overflow. `20000 + 20000 = 40000` is not
/// representable in `i16`; the correct answer is `-25536`. If the oracle were
/// still on the float path it would say `40000` and this test would fail — which
/// is exactly the check that makes adding the dtype meaningful rather than
/// merely declared.
#[test]
fn s16_arithmetic_wraps_identically_in_the_emitter_and_the_oracle() {
    let Some(cc) = find_compiler() else {
        eprintln!(
            "SKIP s16_arithmetic_wraps_identically_in_the_emitter_and_the_oracle: no host              C compiler. This compiles and RUNS an s16 kernel to check 16-bit wrapping."
        );
        return;
    };
    let dir: PathBuf = std::env::temp_dir().join("unpopped-e2e");
    std::fs::create_dir_all(&dir).expect("scratch dir");

    //                       overflow  underflow   plain   identity
    let a: Vec<i16> = vec![20000, -20000, 7, -3, 0, 32767, 1];
    let b: Vec<i16> = vec![20000, -20000, 5, -4, 0, 1, -1];
    let n = a.len() as i64;

    let add = OpDef::elementwise("addi16", 2, &[ElementKind::I16], input(0) + input(1));
    let d = OperandDesc::new(1, &[n], &[1], ElementKind::I16, 256);
    let operands = vec![d; 3];
    let key = structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89);

    let kernel = generate(&add, &key, &CpuC);
    assert!(
        kernel.source.contains("short"),
        "the s16 kernel must be spelled in `short`, got:
{}",
        kernel.source
    );

    // A `short`-typed harness: the f32 one would defeat the point by widening.
    let lit = |v: &[i16]| {
        v.iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    };
    let src = format!(
        "{}
#include <stdio.h>

int main(void) {{
             const short in0[{n}] = {{{}}};
             const short in1[{n}] = {{{}}};
             short out[{n}];
             for (int i = 0; i < {n}; ++i) out[i] = -333;
             {}(in0, in1, out, {n});
             for (int i = 0; i < {n}; ++i) printf(\"%d\\n\", (int)out[i]);
             return 0;
}}
",
        kernel.source,
        lit(&a),
        lit(&b),
        kernel.name
    );

    let c_file = dir.join("s16_add.c");
    let exe = dir.join("s16_add.exe");
    std::fs::write(&c_file, &src).expect("write");
    cc.compile(&c_file, &exe, false)
        .expect("compile s16 kernel");
    let out = std::process::Command::new(&exe).output().expect("run");
    assert!(out.status.success(), "s16 kernel exited {:?}", out.status);
    let actual: Vec<i32> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.trim()
                .parse::<i32>()
                .expect("s16 kernel printed a non-integer")
        })
        .collect();

    // The oracle leg — an independent evaluator that shares no lowering code.
    let plan = build_plan(&add, &key);
    let bufs = vec![
        TypedBuffer::from_i16(&[n], &a),
        TypedBuffer::from_i16(&[n], &b),
    ];
    let expected = evaluate(&plan, &operands, &bufs, &[]);
    let want = expected.into_iter().next().unwrap().to_f64_vec();

    assert_eq!(actual.len(), a.len(), "printed {actual:?}");
    for (i, (&got, &wanted)) in actual.iter().zip(want.iter()).enumerate() {
        assert!(
            (got as f64 - wanted).abs() < 0.5,
            "lane {i}: emitted C gave {got}, oracle expected {wanted}.              a={} b={}. A mismatch on the OVERFLOWING lanes means the oracle is              not modelling 16-bit wrapping (it would say 40000 where C says              -25536); a mismatch elsewhere means the s16 lowering is wrong.",
            a[i],
            b[i]
        );
        assert_ne!(got, -333, "lane {i} was never written");
    }

    // Pin the wrap explicitly, so the test states the property rather than only
    // asserting agreement — two components could agree and both be wrong.
    assert_eq!(
        actual[0], -25536,
        "20000 + 20000 must wrap to -25536 at i16"
    );
    assert_eq!(
        actual[1], 25536,
        "-20000 + -20000 must wrap to 25536 at i16"
    );
    assert_eq!(actual[5], -32768, "32767 + 1 must wrap to i16::MIN");
}

/// `u32` computes as genuinely **unsigned**, verified against a real C compiler
/// on the two cases where signed and unsigned actually diverge.
///
/// # Why this dtype needed its own test
///
/// `u8` and `u16` have unsigned storage but *signed* arithmetic: C's integer
/// promotions lift any type of rank below `int` to signed `int`, so they compute
/// at 32-bit signed width and the store truncates back. `unsigned int` has the
/// same rank as `int` and does **not** promote — its arithmetic is unsigned
/// modulo 2³². Modelling it the way `u8` is modelled would read
/// `3_000_000_000u32` as negative.
///
/// The divergence is invisible for `+`, `-`, `*` and the bitwise ops (identical
/// bit patterns under two's complement) and shows up in exactly two places, both
/// exercised here:
///
/// 1. the **value** produced above `i32::MAX`, and
/// 2. `>>`, which C performs as a *logical* shift on an unsigned operand and an
///    *arithmetic* one on a signed operand.
///
/// Lane 1 is the load-bearing case: `3_000_000_000 >> 1` is `1_500_000_000`
/// unsigned, but `-1_294_967_296 >> 1` = `-647_483_648` signed — a wrong answer
/// that a signed model produces silently and confidently.
#[test]
fn u32_arithmetic_is_unsigned_in_the_emitter_and_the_oracle() {
    let Some(cc) = find_compiler() else {
        eprintln!(
            "SKIP u32_arithmetic_is_unsigned_in_the_emitter_and_the_oracle: no host \
             C compiler. This compiles and RUNS a u32 kernel to check unsigned wrap + shift."
        );
        return;
    };
    let dir: PathBuf = std::env::temp_dir().join("unpopped-e2e");
    std::fs::create_dir_all(&dir).expect("scratch dir");

    //                 above i32::MAX   wraps 2^32   plain   max      identity
    let a: Vec<u32> = vec![3_000_000_000, 4_000_000_000, 7, u32::MAX, 1, 2_147_483_648];
    let b: Vec<u32> = vec![1, 1_000_000_000, 5, 1, 0, 1];
    let n = a.len() as i64;

    let d = OperandDesc::new(1, &[n], &[1], ElementKind::U32, 4);
    let operands = vec![d; 3];
    let key = structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89);

    let lit = |v: &[u32]| {
        v.iter()
            .map(|x| format!("{x}u"))
            .collect::<Vec<_>>()
            .join(", ")
    };

    // Two ops: Add exercises unsigned WRAP; Shr exercises the logical-vs-
    // arithmetic shift, which is where a signed model gives a wrong answer
    // rather than merely a differently-spelled one.
    for (tag, op, shifts) in [
        (
            "add",
            OpDef::elementwise("addu32", 2, &[ElementKind::U32], input(0) + input(1)),
            false,
        ),
        (
            "shr",
            OpDef::elementwise(
                "shru32",
                2,
                &[ElementKind::U32],
                input(0).binary(unpopped::ir::BinaryOp::Shr, input(1)),
            ),
            true,
        ),
    ] {
        // Shift amounts must be in range for every lane.
        let b_used: Vec<u32> = if shifts {
            vec![1, 3, 2, 31, 0, 1]
        } else {
            b.clone()
        };

        let kernel = generate(&op, &key, &CpuC);
        assert!(
            kernel.source.contains("unsigned int"),
            "{tag}: the u32 kernel must be spelled `unsigned int`, got:\n{}",
            kernel.source
        );

        let src = format!(
            "{}
#include <stdio.h>

int main(void) {{
    const unsigned int in0[{n}] = {{{}}};
    const unsigned int in1[{n}] = {{{}}};
    unsigned int out[{n}];
    for (int i = 0; i < {n}; ++i) out[i] = 123u;
    {}(in0, in1, out, {n});
    for (int i = 0; i < {n}; ++i) printf(\"%u\n\", out[i]);
    return 0;
}}
",
            kernel.source,
            lit(&a),
            lit(&b_used),
            kernel.name
        );

        let c_file = dir.join(format!("u32_{tag}.c"));
        let exe = dir.join(format!("u32_{tag}.exe"));
        std::fs::write(&c_file, &src).expect("write");
        cc.compile(&c_file, &exe, true)
            .unwrap_or_else(|e| panic!("{tag}: compile u32 kernel: {e}"));
        let out = std::process::Command::new(&exe).output().expect("run");
        assert!(
            out.status.success(),
            "{tag}: kernel exited {:?}",
            out.status
        );
        let actual: Vec<u32> = String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<u32>().expect("non-integer"))
            .collect();

        // The oracle leg — an independent evaluator sharing no lowering code.
        let plan = build_plan(&op, &key);
        let bufs = vec![
            TypedBuffer::from_u32(&[n], &a),
            TypedBuffer::from_u32(&[n], &b_used),
        ];
        let want = evaluate(&plan, &operands, &bufs, &[])
            .into_iter()
            .next()
            .unwrap()
            .to_f64_vec();

        assert_eq!(actual.len(), a.len(), "{tag}: printed {actual:?}");
        for (i, (&got, &wanted)) in actual.iter().zip(want.iter()).enumerate() {
            assert!(
                (f64::from(got) - wanted).abs() < 0.5,
                "{tag} lane {i}: emitted C gave {got}, oracle expected {wanted}. \
                 a={} b={}. A negative expectation here means the oracle is modelling \
                 u32 as SIGNED — the exact bug this dtype's admission had to rule out.",
                a[i],
                b_used[i]
            );
            assert_ne!(got, 123, "{tag} lane {i} was never written");
        }
    }
}
