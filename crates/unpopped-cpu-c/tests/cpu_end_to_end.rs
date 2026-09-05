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
//! # Finding a compiler, and why the skip was dangerous
//!
//! This needs a host C compiler. On Windows that normally means Visual Studio,
//! which the installer deliberately does NOT put on `PATH` — `vcvars64.bat` is
//! the supported way in. `find_compiler` therefore falls back to locating the
//! install with `vswhere.exe` and driving `cl` through that script.
//!
//! Until it did, this entire file was silently vacuous on such a machine: no
//! compiler found, every test printed `SKIP` and reported `ok`, and the suite
//! was green while compiling nothing — on a box with a working toolchain in
//! Program Files. **A loud skip is still a pass**, and four real-execution tests
//! passing without executing anything is exactly the failure this file exists to
//! prevent in generated kernels.
//!
//! Where there genuinely is no toolchain the tests still skip loudly rather than
//! failing, and the skip prints why.

use std::path::{Path, PathBuf};
use std::process::Command;

use unpopped::ir::{BinaryOp, OpDef, UnaryOp, input};
use unpopped::oracle::{Fidelity, TypedBuffer, compare, evaluate};
use unpopped::{build_plan, generate};
use unpopped_cpu_c::CpuC;
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
    /// A `vcvars64.bat` to `call` before the compiler, for the normal Windows
    /// case where Visual Studio is installed but deliberately not on `PATH`.
    vcvars: Option<PathBuf>,
}

/// Locate a host C compiler, or `None`.
///
/// A plain `PATH` lookup first, then — on Windows — a **Visual Studio install
/// that is not on `PATH`**, which is the normal state of a Windows box rather
/// than an unusual one: the installer does not modify the global environment,
/// and `vcvars64.bat` is the supported way in.
///
/// Skipping that fallback is what made this whole file silently vacuous here.
/// `find_compiler` returned `None`, every e2e test printed `SKIP` and passed,
/// and the suite reported green while compiling nothing — on a machine with a
/// perfectly good toolchain sitting in Program Files. A test that cannot run
/// where it is written is not a test there, and one that says `ok` while doing
/// so is worse than one that fails.
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
                    vcvars: None,
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
            vcvars: None,
        });
    }
    let found = find_vs_install().map(|vcvars| CCompiler {
        cmd: "cl",
        msvc: true,
        needs_libm: false,
        vcvars: Some(vcvars),
    });

    // UNDER CI, A MISSING COMPILER IS A FAILURE RATHER THAN A SKIP.
    //
    // The file header records the decision to skip loudly where there is
    // genuinely no toolchain, and that stays right for a developer's machine —
    // nobody should be blocked from running the rest of the suite by a missing
    // `cc`. But it leaves a hole this file's own words already name: **a loud
    // skip is still a pass**, the `eprintln!` is CAPTURED by the harness on a
    // passing test, and the run reports `ok` either way. So in a CI log a skip
    // and a real execution are indistinguishable, and the strongest gate this
    // workspace has could stop running with nothing to show for it.
    //
    // A runner always has a toolchain. `CI` is set by GitHub Actions and by
    // every other runner worth naming, so where it is set a `None` here is not
    // "no toolchain available" — it is "the toolchain we were promised is
    // missing", which is a fact worth failing on.
    if found.is_none() && std::env::var_os("CI").is_some() {
        panic!(
            "no host C compiler under CI (tried cc, gcc, clang, cl, and the              vswhere/vcvars fallback). Every real-execution test in this file              would SKIP and report `ok`, so the suite would go green while              compiling nothing. Under CI that is a failure, not a skip."
        );
    }
    found
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
        // A `vcvars` install has to be entered through `cmd`, because the batch
        // file's whole job is to mutate the environment of the process that
        // calls it. Spawning `cl` directly after running it in a *different*
        // process would inherit nothing.
        if let Some(vcvars) = &self.vcvars {
            let opt = if optimize { "/O2" } else { "/Od" };
            // Driven through a generated .bat rather than `cmd /C "<line>"`.
            // Rust escapes arguments for `CreateProcess`, `cmd` then applies its
            // own quoting rules to what arrives, and a command line carrying
            // three quoted Windows paths does not survive both. The failure mode
            // is silent: `cmd` exits non-zero having printed nothing at all, so
            // the compile looks like it failed rather than like it never ran.
            // A batch file has exactly one quoting layer.
            let bat = src.with_extension("build.bat");
            let script = format!(
                "@echo off\r\ncall \"{}\" >nul 2>&1\r\ncl /nologo {opt} \"{}\" /Fe:\"{}\"\r\n",
                vcvars.display(),
                src.display(),
                exe.display()
            );
            std::fs::write(&bat, script).map_err(|e| format!("write build script: {e}"))?;
            let out = Command::new(&bat)
                .current_dir(src.parent().unwrap_or(Path::new(".")))
                .output()
                .map_err(|e| format!("spawn build script failed: {e}"))?;
            if out.status.success() {
                return Ok(());
            }
            return Err(format!(
                "compile failed (via vcvars)\n--- stdout ---\n{}\n--- stderr ---\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ));
        }
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

/// The `vcvars64.bat` of the newest Visual Studio install carrying the C/C++
/// tools, or `None`.
///
/// `vswhere.exe` is Microsoft's supported discovery mechanism and ships at a
/// fixed location, which is what makes this a lookup rather than a guess at
/// version-numbered directory names.
#[cfg(windows)]
fn find_vs_install() -> Option<PathBuf> {
    let pf86 = std::env::var("ProgramFiles(x86)")
        .unwrap_or_else(|_| r"C:\Program Files (x86)".to_string());
    let vswhere = PathBuf::from(pf86)
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");
    if !vswhere.exists() {
        return None;
    }
    let out = Command::new(&vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let root = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if root.is_empty() {
        return None;
    }
    let bat = PathBuf::from(root)
        .join("VC")
        .join("Auxiliary")
        .join("Build")
        .join("vcvars64.bat");
    bat.exists().then_some(bat)
}

#[cfg(not(windows))]
fn find_vs_install() -> Option<PathBuf> {
    None
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
    // The kernel's trailing `n` argument is passed the ELEMENT count, which is
    // only right when the emitter's count unit is elements
    // (`Backend::effective_count_width == 1`). CpuC v1 takes the trait default
    // of 1, so this holds today — but it is an assumption this harness makes and
    // never states, and a vectorized emitter whose `n` counts vec4 groups would
    // be driven 4x wrong here with no test going red.
    //
    // `the_harness_element_count_matches_the_emitter_count_unit` pins it, so the
    // day CpuC gains a vectorized path this fails loudly instead of silently
    // mis-driving the kernel. Distinguishing "wrote zeros" from "never wrote"
    // (the NaN sentinel below) does not help if `n` itself is wrong: the kernel
    // would correctly write the range it was told to, and the survivors would
    // read as a never-wrote bug in the emitter rather than a wrong-`n` bug in
    // the caller.
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
    for (int i = 0; i < {n}; ++i) printf(\"%u\\n\", out[i]);
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

/// **FP8 round-trips through a real C compiler, against an independently-written
/// codec.**
///
/// This is the strongest differential in the suite, and it is worth naming why.
/// The emitted kernel carries a software FP8 codec written in C
/// (`cfamily::fp8_helpers`); the oracle carries one written in Rust
/// (`oracle::fp8_e4m3fn_to_f64` and friends). Neither was derived from the other
/// — both were written from KISS-CLASSIFY §6.1-0010/-0011 and the OCP OFP8
/// definitions. So agreement here is two independent readings of a format
/// producing the same bytes, which is what a differential is supposed to mean and
/// what a shared decode table would quietly destroy.
///
/// # Why FP8 got a lowering before f16/bf16, which have been "ready" longer
///
/// The f16/bf16 arms spell `__half2float` — a CUDA name emitted from the module
/// that calls itself neutral, tripwired in
/// `unpopped/tests/neutral_spelling.rs`. Fixing
/// that means replacing a vendor intrinsic with an emitted software codec, which
/// **rewrites every existing f16 golden including Baracuda's physical CUDA
/// corpus**, so it is gated on a coordinated regen.
///
/// FP8 needs the same mechanism and has **no existing goldens to break**. So it
/// goes first, and the seam it proves — `narrow_load_fn`/`narrow_store_fn`, one
/// name over two strategies — is what the f16 arms move onto at the regen, with
/// callers unchanged.
/// KISS-OPS-6.16-0009, proved by executing the kernel rather than by reading it.
///
/// A `max_prop` decomposes to comparison-and-`select` — **no arithmetic** — so
/// its result is the moved operand, **bits exact, payload and sign included**.
///
/// # Why this corpus discriminates and a NaN check would not
///
/// E5M2 has SIX NaN encodings: `0x7D`/`0x7E`/`0x7F` and their negatives. The
/// store codec is `if (x != x) return 0x7F`, so the pre-fix lowering
/// (`store(load(x))`) mapped **all six onto `0x7F`**.
///
/// **A test asserting "the result is NaN" passes on both lowerings.** Only
/// asserting the exact byte separates them — which is the difference between a
/// predicate the right answer satisfies and one the wrong answer fails.
#[test]
fn a_moved_nan_keeps_its_exact_encoding_through_a_real_compiler() {
    let Some(cc) = find_compiler() else {
        eprintln!(
            "SKIP a_moved_nan_keeps_its_exact_encoding_through_a_real_compiler:              no host C compiler."
        );
        return;
    };
    let dir: PathBuf = std::env::temp_dir().join("unpopped-e2e");
    std::fs::create_dir_all(&dir).expect("scratch dir");

    let dt = ElementKind::Fp8E5M2;
    // Every E5M2 NaN encoding: exponent all-ones with a non-zero mantissa.
    let nans: Vec<u8> = vec![0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF];
    // The other operand is +0, which is FINITE — so `max` takes the `a != a`
    // branch and the NaN is the MOVED operand rather than a comparison winner.
    let zeros: Vec<u8> = vec![0x00; nans.len()];
    let n = nans.len() as i64;

    let op = OpDef::elementwise("mxnan", 2, &[dt], input(0).binary(BinaryOp::Max, input(1)));
    let d = OperandDesc::new(1, &[n], &[1], dt, 1);
    let operands = vec![d; 3];
    let key = structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89);
    let kernel = generate(&op, &key, &CpuC);

    let lit = |v: &[u8]| {
        v.iter()
            .map(|x| format!("{x}u"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let src = format!(
        "{}
#include <stdio.h>

int main(void) {{
    const unsigned char in0[{n}] = {{{}}};
    const unsigned char in1[{n}] = {{{}}};
    unsigned char out[{n}];
    for (int i = 0; i < {n}; ++i) out[i] = 0xAAu;
    {}(in0, in1, out, {n});
    for (int i = 0; i < {n}; ++i) printf(\"%u\\n\", (unsigned)out[i]);
    return 0;
}}
",
        kernel.source,
        lit(&nans),
        lit(&zeros),
        kernel.name
    );

    let c_file = dir.join("fp8_move_nan.c");
    let exe = dir.join("fp8_move_nan.exe");
    std::fs::write(&c_file, &src).expect("write");
    cc.compile(&c_file, &exe, false)
        .unwrap_or_else(|e| panic!("compile the moved-NaN kernel: {e}"));
    let out = std::process::Command::new(&exe).output().expect("run");
    assert!(out.status.success(), "exited {:?}", out.status);
    let actual: Vec<u8> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse::<u8>().expect("non-integer"))
        .collect();

    assert_eq!(actual.len(), nans.len(), "printed {} lines", actual.len());
    assert_eq!(
        actual, nans,
        "a moved NaN did not survive with its exact bits. Pre-fix, EVERY one of          these collapsed to 0x7F, because the store codec is          `if (x != x) return 0x7F` and re-encoding was applied to a value that          was SELECTED, never computed (KISS-OPS-6.16-0009).
  in : {nans:02X?}
           out: {actual:02X?}"
    );
}

#[test]
fn fp8_kernels_round_trip_against_the_oracles_independent_codec() {
    let Some(cc) = find_compiler() else {
        eprintln!(
            "SKIP fp8_kernels_round_trip_against_the_oracles_independent_codec: \
             no host C compiler."
        );
        return;
    };
    let dir: PathBuf = std::env::temp_dir().join("unpopped-e2e");
    std::fs::create_dir_all(&dir).expect("scratch dir");

    for (tag, dt) in [
        ("e4m3fn", ElementKind::Fp8E4M3FN),
        ("e5m2", ElementKind::Fp8E5M2),
    ] {
        // Every finite non-NaN pattern of each sign. The whole domain is 256
        // values, so "works for all inputs" is enumerable rather than sampled.
        let pats: Vec<u8> = (0u8..=255).collect();
        let n = pats.len() as i64;

        let op = OpDef::elementwise("addfp8", 2, &[dt], input(0) + input(1));
        let d = OperandDesc::new(1, &[n], &[1], dt, 1);
        let operands = vec![d; 3];
        let key = structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89);
        let kernel = generate(&op, &key, &CpuC);
        assert!(
            kernel.source.contains("unsigned char"),
            "{tag}: FP8 must be stored as a byte:\n{}",
            kernel.source
        );
        assert!(
            !kernel.source.contains("__half") && !kernel.source.contains("__nv_"),
            "{tag}: the FP8 codec must be portable C, with no vendor intrinsic:\n{}",
            kernel.source
        );

        // Second operand is +0 (0x00), so `a + 0` must reproduce `a` exactly for
        // every finite pattern — an identity that isolates the CODEC from the
        // arithmetic. A codec that rounds wrongly fails here even though the add
        // is trivially right.
        let zeros: Vec<u8> = vec![0u8; pats.len()];
        let lit = |v: &[u8]| {
            v.iter()
                .map(|x| format!("{x}u"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let src = format!(
            "{}
#include <stdio.h>

int main(void) {{
    const unsigned char in0[{n}] = {{{}}};
    const unsigned char in1[{n}] = {{{}}};
    unsigned char out[{n}];
    for (int i = 0; i < {n}; ++i) out[i] = 0xAAu;
    {}(in0, in1, out, {n});
    for (int i = 0; i < {n}; ++i) printf(\"%u\\n\", (unsigned)out[i]);
    return 0;
}}
",
            kernel.source,
            lit(&pats),
            lit(&zeros),
            kernel.name
        );

        let c_file = dir.join(format!("fp8_{tag}.c"));
        let exe = dir.join(format!("fp8_{tag}.exe"));
        std::fs::write(&c_file, &src).expect("write");
        cc.compile(&c_file, &exe, false)
            .unwrap_or_else(|e| panic!("{tag}: compile FP8 kernel: {e}"));
        let out = std::process::Command::new(&exe).output().expect("run");
        assert!(out.status.success(), "{tag}: exited {:?}", out.status);
        let actual: Vec<u8> = String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<u8>().expect("non-integer"))
            .collect();
        assert_eq!(
            actual.len(),
            pats.len(),
            "{tag}: printed {} lines",
            actual.len()
        );

        // The oracle leg, through its own codec.
        let plan = build_plan(&op, &key);
        let bufs = vec![
            TypedBuffer::from_fp8_bits(dt, &[n], &pats),
            TypedBuffer::from_fp8_bits(dt, &[n], &zeros),
        ];
        let want = evaluate(&plan, &operands, &bufs, &[])
            .into_iter()
            .next()
            .unwrap()
            .to_f64_vec();
        let got_vals = TypedBuffer::from_fp8_bits(dt, &[n], &actual).to_f64_vec();

        for (i, (&g, &w)) in got_vals.iter().zip(want.iter()).enumerate() {
            if w.is_nan() {
                assert!(
                    g.is_nan(),
                    "{tag} pattern {:#04x}: oracle says NaN, kernel produced {g}",
                    pats[i]
                );
                continue;
            }
            assert_eq!(
                g, w,
                "{tag} pattern {:#04x}: kernel produced {g}, oracle expected {w}. \
                 The two FP8 codecs — emitted C and oracle Rust — disagree, which is \
                 exactly what this test exists to detect.",
                pats[i]
            );
        }
    }
}

/// **Sub-byte kernels round-trip through a real C compiler, packing intact.**
///
/// The emitted kernel packs and unpacks per KISS-CLASSIFY §6.1 in C; the oracle
/// does the same in Rust; neither was derived from the other. This is where a
/// packing disagreement would show as a wrong *value* rather than a wrong byte
/// count — the failure a round-trip inside one implementation cannot see.
///
/// # The store is a read-modify-write, and that is why `n` is odd here
///
/// Two `i4` elements share a byte, so writing one must preserve its neighbour.
/// An odd element count also forces a partial trailing byte, which is the
/// off-by-one an `n / 2` allocation gets wrong. `b1` uses a count that is not a
/// multiple of 8 for the same reason.
///
/// This works because `cpu_c`'s loop is **serial**. In a threaded backend two
/// lanes would read-modify-write the same byte and race — noted on
/// `cfamily::sub_byte_helpers`, because the code that looks copyable is the code
/// that gets copied.
#[test]
fn sub_byte_kernels_preserve_packing_through_a_real_compiler() {
    let Some(cc) = find_compiler() else {
        eprintln!("SKIP sub_byte_kernels_preserve_packing_through_a_real_compiler: no C compiler.");
        return;
    };
    let dir: PathBuf = std::env::temp_dir().join("unpopped-e2e");
    std::fs::create_dir_all(&dir).expect("scratch dir");

    // (tag, dtype, a, b) — `a + b` stays in range for every lane, so any
    // mismatch is packing rather than overflow.
    let cases: Vec<(&str, ElementKind, Vec<i8>, Vec<i8>)> = vec![
        // 7 elements: odd, so the last byte is half-used.
        (
            "i4",
            ElementKind::I4,
            vec![-8, -1, 0, 1, 7, 3, -4],
            vec![0, 1, 0, 2, 0, -3, 4],
        ),
        (
            "u4",
            ElementKind::U4,
            vec![0, 1, 15, 7, 8, 2, 9],
            vec![0, 2, 0, 8, 7, 13, 6],
        ),
        // 9 elements: not a multiple of 8.
        (
            "b1",
            ElementKind::B1,
            vec![1, 0, 1, 1, 0, 0, 1, 0, 1],
            vec![0, 0, 1, 0, 1, 0, 1, 1, 0],
        ),
    ];

    for (tag, dt, a, b) in cases {
        let n = a.len() as i64;
        // `b1` is a 1-bit operand and `+` would overflow it. `BitXor` is both
        // admissible and the semantically right choice: KISS-CLASSIFY §6.1
        // describes b1 as the binary-GEMM operand with "xor+popcount
        // accumulation", so xor IS its arithmetic.
        let op = if dt == ElementKind::B1 {
            OpDef::elementwise(
                "sbop",
                2,
                &[dt],
                input(0).binary(unpopped::ir::BinaryOp::BitXor, input(1)),
            )
        } else {
            OpDef::elementwise("sbop", 2, &[dt], input(0) + input(1))
        };
        let d = OperandDesc::new(1, &[n], &[1], dt, 1);
        let operands = vec![d; 3];
        let key = structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89);
        let kernel = generate(&op, &key, &CpuC);

        let bufs = vec![
            TypedBuffer::from_sub_byte(dt, &[n], &a),
            TypedBuffer::from_sub_byte(dt, &[n], &b),
        ];
        let lit = |v: &[u8]| {
            v.iter()
                .map(|x| format!("{x}u"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let (pa, pb) = (bufs[0].raw_bytes(), bufs[1].raw_bytes());
        let out_bytes = pa.len();
        let src = format!(
            "{}
#include <stdio.h>

int main(void) {{
    const unsigned char in0[{}] = {{{}}};
    const unsigned char in1[{}] = {{{}}};
    unsigned char out[{out_bytes}];
    for (int i = 0; i < {out_bytes}; ++i) out[i] = 0x5Au;
    {}(in0, in1, out, {n});
    for (int i = 0; i < {out_bytes}; ++i) printf(\"%u\\n\", (unsigned)out[i]);
    return 0;
}}
",
            kernel.source,
            pa.len(),
            lit(pa),
            pb.len(),
            lit(pb),
            kernel.name
        );

        let c_file = dir.join(format!("sb_{tag}.c"));
        let exe = dir.join(format!("sb_{tag}.exe"));
        std::fs::write(&c_file, &src).expect("write");
        cc.compile(&c_file, &exe, false)
            .unwrap_or_else(|e| panic!("{tag}: compile: {e}"));
        let out = std::process::Command::new(&exe).output().expect("run");
        assert!(out.status.success(), "{tag}: exited {:?}", out.status);
        let got_bytes: Vec<u8> = String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<u8>().expect("non-integer"))
            .collect();

        let plan = build_plan(&op, &key);
        let want = evaluate(&plan, &operands, &bufs, &[])
            .into_iter()
            .next()
            .unwrap();

        // Compare the DECODED values — a byte comparison would also pass if both
        // sides packed the same way but decoded wrongly.
        let got = TypedBuffer::from_sub_byte(dt, &[n], &vec![0i8; a.len()]);
        let _ = got; // shape only; the real decode is below.
        let decoded: Vec<i128> = {
            let mut tb = TypedBuffer::from_sub_byte(dt, &[n], &vec![0i8; a.len()]);
            let _ = &mut tb;
            // Rebuild a buffer over the kernel's actual bytes.
            TypedBuffer::from_packed_bytes(dt, &[n], &got_bytes).to_i128_vec()
        };
        assert_eq!(
            decoded,
            want.to_i128_vec(),
            "{tag}: kernel and oracle disagree. Equal-length byte arrays with \
             different values means the PACKING diverged, not the arithmetic."
        );
        assert_ne!(
            got_bytes,
            vec![0x5Au8; out_bytes],
            "{tag}: every byte still holds the fill — the kernel never ran"
        );
    }
}

/// Complex arithmetic agrees between the emitted struct helpers and the oracle.
///
/// # Why this test carries more weight than the other dtype differentials
///
/// For every other dtype the emitter and the oracle compute the *same operation*
/// on different representations. For complex they compute a **different
/// expression**: `(a + bi)(c + di) = (ac - bd) + (ad + bc)i` is four
/// multiplies, a subtract and an add, spelled out by hand on both sides from the
/// same algebra and shared by nothing. The C helper is text this crate emits; the
/// oracle's is a Rust closure. A transcription slip in either — a `+` where a `-`
/// belongs, `ad + bc` written `ac + bd` — is invisible to any round-trip and
/// shows up only here.
///
/// The component-wise mistake `(ac, bd)` is the one to beat: it compiles, it
/// preserves shape, it round-trips, and it is wrong. Every input below has a
/// non-zero imaginary part on both operands, so component-wise multiplication
/// disagrees on every single element rather than on a lucky one.
///
/// # And the negative space
///
/// This also proves the emitted C is *acceptable to a real compiler*. MSVC
/// rejects C99 `_Complex` outright (`error C2440`), so a kernel that reached for
/// the obvious spelling would fail here rather than in a downstream consumer's
/// build. That the compiler under test is usually MSVC on this platform is the
/// point, not an accident: it is the strictest of the three about this feature.
#[test]
fn complex_kernels_compute_the_mixing_product_a_real_compiler_accepts() {
    let Some(cc) = find_compiler() else {
        eprintln!("SKIP complex_kernels_...: no C compiler.");
        return;
    };
    let dir: PathBuf = std::env::temp_dir().join("unpopped-e2e");
    std::fs::create_dir_all(&dir).expect("scratch dir");

    // Both operands carry a non-zero imaginary part everywhere, so `(ac, bd)`
    // differs from the true product on every element. The last pair is a pure
    // rotation: (0+1i)(0+1i) = -1, which component-wise renders as (0, 1) —
    // a sign error a magnitude-only check would sail past.
    let a: Vec<(f64, f64)> = vec![(3.0, 4.0), (-1.5, 2.25), (0.5, -0.75), (0.0, 1.0)];
    let b: Vec<(f64, f64)> = vec![(1.0, -2.0), (2.0, 0.5), (-4.0, 1.5), (0.0, 1.0)];
    let n = a.len() as i64;

    for (tag, dt, ctype, bufs) in [
        (
            "c64",
            ElementKind::Complex64,
            "unpopped_c64",
            vec![
                TypedBuffer::from_complex64(
                    &[n],
                    &a.iter()
                        .map(|&(r, i)| (r as f32, i as f32))
                        .collect::<Vec<_>>(),
                ),
                TypedBuffer::from_complex64(
                    &[n],
                    &b.iter()
                        .map(|&(r, i)| (r as f32, i as f32))
                        .collect::<Vec<_>>(),
                ),
            ],
        ),
        (
            "c128",
            ElementKind::Complex128,
            "unpopped_c128",
            vec![
                TypedBuffer::from_complex128(&[n], &a),
                TypedBuffer::from_complex128(&[n], &b),
            ],
        ),
    ] {
        for body in [Body::Mul, Body::Div] {
            // The round-trip `(a * b) / b == a` looks like the elegant test and is
            // the WRONG one: a fully component-wise mul paired with a component-wise
            // div satisfies it exactly. Any self-consistent pair of inverses does.
            // So each operation is checked against its own independent reference
            // instead, and `body` says which is under test.
            let op = match body {
                Body::Mul => OpDef::elementwise("cmul", 2, &[dt], input(0) * input(1)),
                Body::Div => OpDef::elementwise("cdiv", 2, &[dt], input(0) / input(1)),
            };
            let width = if dt == ElementKind::Complex64 { 8 } else { 16 };
            let d = OperandDesc::new(1, &[n], &[1], dt, width);
            let operands = vec![d; 3];
            let key = structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89);
            let kernel = generate(&op, &key, &CpuC);

            let lit = |v: &[(f64, f64)]| {
                v.iter()
                    .map(|&(r, i)| format!("{{ {r:?}, {i:?} }}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            // The fill is a value the correct answer never produces, so "the kernel
            // never ran" cannot masquerade as a pass.
            let src = format!(
                "{}
#include <stdio.h>

int main(void) {{
    const {ctype} in0[{n}] = {{{}}};
    const {ctype} in1[{n}] = {{{}}};
    {ctype} out[{n}];
    for (int i = 0; i < {n}; ++i) {{ out[i].re = -999.0; out[i].im = -999.0; }}
    {}(in0, in1, out, {n});
    for (int i = 0; i < {n}; ++i) printf(FMT, out[i].re, out[i].im);
    return 0;
}}
",
                kernel.source,
                lit(&a),
                lit(&b),
                kernel.name
            );
            // Built from a char literal rather than written inline: a `\n` inside a
            // Rust string that becomes C source has bitten this file repeatedly, and
            // the failure mode is an unterminated C string literal 40 lines away.
            let fmt = format!("\"%.17g %.17g{}n\"", '\\');
            let src = src.replace("FMT", &fmt);

            let bt = body.tag();
            let c_file = dir.join(format!("cx_{tag}_{bt}.c"));
            let exe = dir.join(format!("cx_{tag}_{bt}.exe"));
            std::fs::write(&c_file, &src).expect("write");
            cc.compile(&c_file, &exe, false)
            .unwrap_or_else(|e| panic!("{tag}/{bt}: compile failed — if this is `_Complex`, the emitter regressed to a spelling MSVC rejects:\n{e}"));
            let out = std::process::Command::new(&exe).output().expect("run");
            assert!(out.status.success(), "{tag}/{bt}: exited {:?}", out.status);
            let got: Vec<(f64, f64)> = String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| {
                    let mut p = l.split_whitespace();
                    let r: f64 = p.next().unwrap().parse().expect("re");
                    let i: f64 = p.next().unwrap().parse().expect("im");
                    (r, i)
                })
                .collect();
            assert_eq!(
                got.len(),
                a.len(),
                "{tag}/{bt}: kernel printed {} rows",
                got.len()
            );

            let plan = build_plan(&op, &key);
            let want = evaluate(&plan, &operands, &bufs, &[])
                .into_iter()
                .next()
                .unwrap()
                .to_complex_vec();

            // c64 computes at `float`, so exact equality is the wrong bar; c128 is
            // f64 on both sides and the tolerance costs nothing there.
            let tol = if dt == ElementKind::Complex64 {
                1e-5
            } else {
                1e-12
            };
            for (k, (&(gr, gi), &(wr, wi))) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (gr - wr).abs() <= tol && (gi - wi).abs() <= tol,
                    "{tag}/{bt}[{k}]: kernel ({gr}, {gi}) vs oracle ({wr}, {wi}).\n\
                 A real part that equals a*c and an imaginary that equals b*d \
                 means the emitted arithmetic is component-wise, not complex."
                );
                assert_ne!((gr, gi), (-999.0, -999.0), "{tag}/{bt}[{k}]: fill survived");
            }

            // A THIRD formulation, written here and shared with neither side.
            //
            // The oracle's `Div` runs Smith's algorithm and so does the emitted C —
            // deliberately, so the comparison above is not a tolerance negotiation
            // between differently-conditioned formulas. But that also means a
            // transcription error in Smith's recurrence would have to be caught by
            // something that is not Smith's recurrence. The textbook quotient is
            // that something: numerically worse in general, exact enough on these
            // well-scaled inputs, and wrong in entirely different ways.
            let independent: Vec<(f64, f64)> = a
                .iter()
                .zip(b.iter())
                .map(|(&(ar, ai), &(br, bi))| match body {
                    Body::Mul => (ar * br - ai * bi, ar * bi + ai * br),
                    Body::Div => {
                        let den = br * br + bi * bi;
                        ((ar * br + ai * bi) / den, (ai * br - ar * bi) / den)
                    }
                })
                .collect();
            for (k, (&(ir, ii), &(wr, wi))) in independent.iter().zip(want.iter()).enumerate() {
                assert!(
                    (ir - wr).abs() <= tol && (ii - wi).abs() <= tol,
                    "{tag}/{bt}[{k}]: oracle ({wr}, {wi}) disagrees with the textbook \
                 formula ({ir}, {ii}) — one of the two rules is mistranscribed"
                );
            }

            // Positive control on the test itself: the component-wise answer must
            // actually DIFFER, or the comparisons above prove nothing.
            let cw: Vec<(f64, f64)> = a
                .iter()
                .zip(b.iter())
                .map(|(&(ar, ai), &(br, bi))| match body {
                    Body::Mul => (ar * br, ai * bi),
                    Body::Div => (ar / br, ai / bi),
                })
                .collect();
            let distinguishing = cw
                .iter()
                .zip(want.iter())
                .filter(|(c, w)| (c.0 - w.0).abs() > tol || (c.1 - w.1).abs() > tol)
                .count();
            // Not every lane can distinguish: component-wise division of an operand
            // by ITSELF is `(1, 1)` and the true quotient is `1 + 0i`, which agree in
            // the real part — and for `(0+1i)/(0+1i)` they agree entirely. Requiring
            // a majority keeps the control honest without pretending otherwise.
            assert!(
                distinguishing * 2 > cw.len(),
                "{tag}/{bt}: only {distinguishing}/{} lanes distinguish component-wise \
             from complex — this input set barely tests the rule",
                cw.len()
            );
        }
    }
}

/// Which operation a pass of the complex differential is exercising.
///
/// Two separate bodies rather than the round-trip `(a * b) / b == a`, which
/// reads as the elegant test and is the wrong one: a component-wise multiply
/// paired with a component-wise divide satisfies it exactly, as does any
/// self-consistent pair of inverses. An identity between two suspects cannot
/// convict either.
#[derive(Clone, Copy, PartialEq)]
enum Body {
    Mul,
    Div,
}

impl Body {
    fn tag(self) -> &'static str {
        match self {
            Body::Mul => "mul",
            Body::Div => "div",
        }
    }
}

/// **The harness passes an ELEMENT count as the kernel's `n`; pin that this is
/// what the emitter actually means by `n`.**
///
/// `harness()` emits `kernel(in0, .., out, n)` with `n = inputs[0].len()`. That
/// is only correct when the emitter's count unit is elements —
/// [`Backend::effective_count_width`] `== 1`. CpuC v1 does not override the
/// trait default of `1`, so it holds; nothing stated it.
///
/// # Why this is worth a test rather than a comment
///
/// The failure it guards is silent and misattributing. A vectorized emitter
/// whose `n` counts vec4 groups would be handed 4x the count it expects. The
/// NaN sentinel — this file's main safety net — does **not** catch it: the
/// kernel would faithfully write the range it was told to write, and the
/// surviving NaNs (or the overrun) would read as a *never-wrote* bug in the
/// emitter rather than a *wrong-`n`* bug in the caller. The sentinel
/// distinguishes "wrote zeros" from "never wrote"; it cannot distinguish either
/// from "was asked for the wrong range".
///
/// So this is deliberately a test of an assumption that is TRUE today. It exists
/// to fail on the day CpuC gains a vectorized path, at which point `harness()`
/// must derive `n` from the count unit rather than from the input length.
#[test]
fn the_harness_element_count_matches_the_emitter_count_unit() {
    use unpopped::backend::Backend;

    let n: i64 = 8;
    // The dtypes and categories this file actually drives through `harness()`.
    let cases: [(ElementKind, OpCategory, &str); 3] = [
        (ElementKind::F32, OpCategory::BinaryElementwise, "add"),
        (ElementKind::F64, OpCategory::BinaryElementwise, "addf64"),
        (ElementKind::I16, OpCategory::BinaryElementwise, "addi16"),
    ];

    for (dtype, cat, name) in cases {
        let op = OpDef::elementwise(name, 2, &[dtype], input(0) + input(1));
        let d = OperandDesc::new(1, &[n], &[1], dtype, 256);
        let operands = vec![d; 3];
        let key = structure_key(cat, &operands, ArchSku::Sm89);
        let plan = build_plan(&op, &key);

        assert_eq!(
            CpuC.effective_count_width(&plan),
            1,
            "{name}/{dtype:?}: `harness()` passes an element count as the kernel's \
             `n`. A count width other than 1 means `n` counts groups, and every \
             end-to-end case in this file would be driving the kernel over the \
             wrong range — see the comment in `harness()`."
        );
    }
}

/// The differential whose ABSENCE let the sign-edit defect live in two places at
/// once — and the reason it could not have caught it before 2026-09-05.
///
/// # Why this test is the finding, not the fix
///
/// Until today the emitter lowered fp8 `neg`/`abs`/`copysign` as
/// `store(op(load(x)))`, and the oracle encoded a NaN result as a bare `0x7f`
/// with the sign computed on the next line and never applied. **Both collapsed a
/// NaN's sign, in the same direction.** So a comparison between them would have
/// been GREEN on the broken pair, and the defect was found by an outside question
/// rather than by anything in this workspace.
///
/// ⚠️ **An oracle that is wrong in the same direction as the thing it checks is
/// not a check.** That is what makes agreement between two implementations weak
/// evidence when they share a mistake — and both of these did, because
/// promote-compute-demote is the obvious way to write either one.
///
/// # What it asserts
///
/// The whole 256-byte domain per dtype, so "correct for all inputs" is
/// **enumerated rather than sampled** — and the NaN encodings, which are the only
/// patterns that discriminate, are 6 of 256 for e5m2 and 2 of 256 for e4m3fn. A
/// random sample would miss them most of the time.
#[test]
fn a_sign_edit_agrees_with_the_oracle_on_every_fp8_byte() {
    let Some(cc) = find_compiler() else {
        eprintln!(
            "SKIP a_sign_edit_agrees_with_the_oracle_on_every_fp8_byte: \
             no host C compiler."
        );
        return;
    };
    let dir: PathBuf = std::env::temp_dir().join("unpopped-e2e");
    std::fs::create_dir_all(&dir).expect("scratch dir");

    let pats: Vec<u8> = (0u8..=255).collect();
    let n = pats.len() as i64;
    // For `copysign` the second operand walks the domain BACKWARDS, so every
    // (magnitude, sign) pairing is exercised rather than sign always coming from
    // the same byte. `0x00` throughout would make every result positive and the
    // test would pass on an implementation that ignored operand 1 entirely.
    let rev: Vec<u8> = pats.iter().rev().copied().collect();
    let lit = |v: &[u8]| {
        v.iter()
            .map(|x| format!("{x}u"))
            .collect::<Vec<_>>()
            .join(", ")
    };

    // Every case is measured before anything is asserted. A per-case assert stops
    // at the first failure and hides the rest — it reported the sign-edit
    // divergence and concealed whether the pure-move class shares it.
    let mut known: Vec<String> = Vec::new();
    let mut unexpected: Vec<String> = Vec::new();

    for (tag, dt) in [
        ("e4m3fn", ElementKind::Fp8E4M3FN),
        ("e5m2", ElementKind::Fp8E5M2),
    ] {
        let cases: Vec<(&str, OpDef, usize)> = vec![
            (
                "neg",
                OpDef::elementwise("sneg", 1, &[dt], input(0).unary(UnaryOp::Neg)),
                1,
            ),
            (
                "abs",
                OpDef::elementwise("sabs", 1, &[dt], input(0).unary(UnaryOp::Abs)),
                1,
            ),
            (
                "copysign",
                OpDef::elementwise(
                    "scs",
                    2,
                    &[dt],
                    input(0).binary(BinaryOp::Copysign, input(1)),
                ),
                2,
            ),
            // Not a sign edit -- a pure MOVE, fixed in August. Included because
            // it is the same bit-preserving class and the same oracle path, so
            // if the oracle collapses a payload it collapses it here too.
            (
                "max",
                OpDef::elementwise("smx", 2, &[dt], input(0).binary(BinaryOp::Max, input(1))),
                2,
            ),
        ];

        for (name, op, n_in) in cases {
            let d = OperandDesc::new(1, &[n], &[1], dt, 1);
            let operands = vec![d; n_in + 1];
            let key = structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89);
            let kernel = generate(&op, &key, &CpuC);

            // The emitted leg must not reach a codec at all — asserted here as
            // well as in `narrow_float_moves_do_not_round`, because a lowering
            // that promoted AND happened to agree with a broken oracle would
            // otherwise pass this test for the wrong reason.
            assert!(
                !kernel.source.contains(&format!("unpopped_{tag}_store")),
                "{tag} {name}: re-encoded a value that was never computed:\n{}",
                kernel.source
            );

            let (decls, call) = if n_in == 1 {
                (
                    format!("    const unsigned char in0[{n}] = {{{}}};", lit(&pats)),
                    format!("{}(in0, out, {n});", kernel.name),
                )
            } else {
                (
                    format!(
                        "    const unsigned char in0[{n}] = {{{}}};\n    const unsigned char in1[{n}] = {{{}}};",
                        lit(&pats),
                        lit(&rev)
                    ),
                    format!("{}(in0, in1, out, {n});", kernel.name),
                )
            };
            let src = format!(
                "{}
#include <stdio.h>

int main(void) {{
{decls}
    unsigned char out[{n}];
    for (int i = 0; i < {n}; ++i) out[i] = 0xAAu;
    {call}
    for (int i = 0; i < {n}; ++i) printf(\"%u\\n\", (unsigned)out[i]);
    return 0;
}}
",
                kernel.source
            );

            let c_file = dir.join(format!("fp8_sign_{tag}_{name}.c"));
            let exe = dir.join(format!("fp8_sign_{tag}_{name}.exe"));
            std::fs::write(&c_file, &src).expect("write");
            cc.compile(&c_file, &exe, false)
                .unwrap_or_else(|e| panic!("{tag} {name}: compile: {e}"));
            let out = std::process::Command::new(&exe).output().expect("run");
            assert!(
                out.status.success(),
                "{tag} {name}: exited {:?}",
                out.status
            );
            let actual: Vec<u8> = String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.trim().parse::<u8>().expect("non-integer"))
                .collect();
            assert_eq!(actual.len(), pats.len(), "{tag} {name}: short output");

            // The oracle leg, through its own independently-written codec.
            let plan = build_plan(&op, &key);
            let mut bufs = vec![TypedBuffer::from_fp8_bits(dt, &[n], &pats)];
            if n_in == 2 {
                bufs.push(TypedBuffer::from_fp8_bits(dt, &[n], &rev));
            }
            let want: Vec<u8> = evaluate(&plan, &operands, &bufs, &[])
                .into_iter()
                .next()
                .expect("one output")
                .raw_bytes()
                .to_vec();
            assert_eq!(want.len(), pats.len(), "{tag} {name}: oracle short");

            for (idx, (a, w)) in actual.iter().zip(want.iter()).enumerate() {
                if a == w {
                    continue;
                }
                let (i0, i1) = (pats[idx], rev[idx]);
                // Classified HERE, where the operand bytes are in hand, rather
                // than by matching the message text later: for a 2-input op the
                // NaN can arrive through EITHER operand, and a carve-out keyed
                // on `in0` alone silently mis-sorts every `max`/`copysign` case.
                let nan = |b: u8| dt == ElementKind::Fp8E5M2 && (b & 0x7C) == 0x7C && (b & 3) != 0;
                let known_gap = nan(i0) || (n_in == 2 && nan(i1));
                let line = if n_in == 2 {
                    format!(
                        "  {tag} {name}: in0 0x{i0:02X}, in1 0x{i1:02X} -> emitter 0x{a:02X}, oracle 0x{w:02X}"
                    )
                } else {
                    format!("  {tag} {name}: in 0x{i0:02X} -> emitter 0x{a:02X}, oracle 0x{w:02X}")
                };
                if known_gap {
                    known.push(line);
                } else {
                    unexpected.push(line);
                }
            }
        }
    }

    // ⚠️ ONE KNOWN GAP, BOUNDED AND POLICED HERE RATHER THAN EXCLUDED.
    //
    // The oracle evaluates through `f64`, so it cannot carry an e5m2 NaN's
    // PAYLOAD: the format has six NaN encodings (0x7D/0x7E/0x7F and negatives)
    // and a round trip through `f64` collapses them to one. e4m3fn is unaffected
    // because 0x7F/0xFF are all it has, so its canonical form IS its only form.
    //
    // Anything else — any e4m3fn divergence, or an e5m2 divergence where neither
    // operand was a NaN — is a real defect and fails here.
    assert!(
        unexpected.is_empty(),
        "emitter and oracle disagree OUTSIDE the known e5m2 NaN-payload gap. Both \
         legs are independent implementations of the same bit-preserving op, so a \
         disagreement here is a defect in one of them:\n{}",
        unexpected.join("\n")
    );

    // And the gap must still EXIST. When the oracle learns to carry the payload,
    // this fires and says to delete the carve-out rather than leaving a
    // permanently-green exclusion nobody revisits.
    assert!(
        !known.is_empty(),
        "the e5m2 NaN-payload divergence is GONE, so the oracle now carries the \
         payload. Delete this carve-out and assert that there are no differences \
         at all — an exclusion that no longer excludes anything is a dead end."
    );
}
