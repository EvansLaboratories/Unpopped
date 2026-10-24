# Unpopped roadmap

What this lane is doing, in what order, and **what each item waits on**. Read at
`origin/main` `c6ca534` (0.22.0) on 2026-10-23; every row names the ref it was
read at, so re-read it before acting on it. History is in `CHANGELOG.md`;
knowingly-undone work and its reasoning is `docs/deferred.md`. This file is the
queue, not the archive: delete a row when it lands.

## 1. In flight

| # | Item | State | Waits on |
|---|---|---|---|
| 2 | **Run `emitted_slang_compiles_with_slangc` on the self-hosted runner** (CireSnave 5.1g/5.1h: an environment-bound test is run there, never ignored). It stays `#[ignore]` meanwhile: a documented, counted last resort. Owner: this lane. Detector: the runner job, once it exists (no hosted job can run it) | `slangc` exists on the PM's PC (`C:/VulkanSDK/1.4.350.0/Bin`) | PM: runner registered for Unpopped, which is a **PUBLIC** repo, so the **ephemeral** runner (mandatory for public repos; persistent is private-only) under 5.1h (push-to-main, schedule, dispatch; same-repo PRs only behind the environment gate; fork PRs never run on the runner, so fork code is never executed there) + `slangc` on the `gh-runner` PATH. Then this lane adds the job and drops the attribute |
| 3 | **Harness family**: external test/profile harnesses wrapping Unpopped, now in its own public repo `ThinkersJournal/unpopped-harness` (crate `unpopped-harness` 0.0.3 at `843e285`; lanes cannot push there, the PM pushes by SHA; protection requires verify / cargo deny / SPDX). Targets: cpu-c (reference), CUDA, Vulkan (Slang). Done and measured there: exhaustive 2^32 f32 relu on all three; CUDA `Max(x,0)` == Relu; Slang relu is exact on AMD 610M and RTX 4070 only with `-denorm-mode-fp32 preserve` (Unpopped#84; PM ruling 2026-10-21: a kernel whose measured result flushes declines Exact). Design: `docs/harness-family.md`. Owner: this lane | Next tracks (binary ops, f64, chains on Vulkan, timing) are this lane's call. GPU and Vulkan tests need a self-hosted runner (PM list) |

## 2. KISS items this lane consumes

| Item | State | Where |
|---|---|---|
| Derivation vectors, KISS `026feba` (0.4.3) | **63 of 63 match**; `KNOWN_GAP` is empty and asserted exactly | `crates/unpopped-vocab/tests/kiss_derivation_vectors.rs` |
| M/N/K and batch roles (§6.6-0016, KISS#543) | **closed** at 0.19.0 behind `CellFacts::contraction_roles`; `ContractionRoles` since 0.20.0 | #64, #66 |
| Weight role (§6.6-0019) | **closed** in 0.17.3; `WeightRoleNotDeclared` and `ContractionRolesNotDeclared` are this crate's decline names, not KISS's | `CellFacts::weight_operand` |
| §6.6-0021 | the infallible `structure_key` still keys a scale dtype at operand 0; `try_structure_key` declines. Deprecating the infallible form means migrating ~200 callers, which the PM ruled is **not this lane** (bulk mechanical work, CireSnave's cost rule) | `CHANGELOG.md` |
| An FMA SIMT cell can never serve an `st` key (§6.17-0002) | **known gap**, told to the KISS lane; a caller wanting bit-exact F32 gets a decline until KISS adds a "third value" row | |

KISS `main` moves faster than this lane re-vendors. Re-vendor when KISS says a
blob changed, not on a schedule, and record the commit in `kiss/README.md`.

## 3. Next

1. **`Approx` qualifier** (`docs/idiom-lifting-design.md` §4.3): the full fix for
   `__expf` / `__logf`, which #57 only made refuse. Needs KISS-coordinated
   numerics axes.
2. **Re-vendor on KISS change**, with the derivation vectors re-run
   (`cargo test -p unpopped-vocab --test kiss_derivation_vectors -- --nocapture`).
3. **CodeRipper SUP-004** (portfolio check, advisory; baseline on `21e6279a`): `crates/unpopped/tests/the_declaration_form_is_a_seam.rs:337` reads `../../../docs/normative-seams.md`, outside its crate, so the test cannot run from the packaged `.crate`. Fix: make the test not need a file outside the crate (vendor the text it checks into the crate, or `include_str!` from inside it).
4. **CodeRipper lint-suppression** (baseline: 37 `#[allow(..)]` without a reason: 32 in `crates/unpopped` (`optimize.rs` 10, `oracle.rs` 6, `plan.rs` 5, `ir.rs` 3, `shape.rs` 2, tests 5, `capability.rs` 1), 4 in `unpopped-cpu-c` tests, 1 in `unpopped-vocab` tests): give each a `reason = "..."`. Do not delete an allow to silence the rule.
5. **CodeRipper licence, holder** (baseline: `LICENSE-MIT` names a person as copyright holder; the same text is in the four `crates/*/LICENSE-MIT` copies): PM instruction 2026-10-24 is holder "Eric Evans", EvansLaboratories. Exact wording to be confirmed by the PM before editing.
6. **CodeRipper licence, SPDX** (baseline: 104 of 105 source files carry no `SPDX-License-Identifier` in their first 15 lines, first hit `crates/unpopped-conformance/src/lib.rs:1`): add the identifier matching `license` in Cargo.toml, enumerating files from `git ls-files`.
7. **CodeRipper rules that did not run** (baseline: `reachability`, `unused-return-values`, `unused-parameters` failed on a virtual workspace root): fixed upstream by coderipper#127; this PR repins to it. Re-read the next run for new findings and add one row per rule.

## 4. Held or blocked

| Item | Waits on |
|---|---|
| **Leg B**: on-device `__expf` round-trip (`docs/idiom-lifting-design.md` §5). baracuda's pin at `baracuda-cuda-parse/tests/convert.rs:84` fails when they take 0.15.0 (PM relayed it) | a rig: no P40 yet |
| Joint GPU milestone plan, `docs/joint-gpu-milestone-plan.md`. Ownership: Unpopped owns capability rows and per-sm plan decisions, baracuda owns CUDA spelling per sm. U1 (#40) and U2 (#42) are done; M4/M5 are blocked on the P40 | PM, before more U-items |
| Moving `ArchSku` into `baracuda-cuda-vocab` | PM says go |
| Groupshared / cooperative scope call (`docs/groupshared-sizing.md`) | CireSnave or PM |
| CpuC declines `Select` for every narrow float | parity gap, not a bug; no owner |

## 5. Standing constraints

- Never run, launch or spawn anything on Opus; subagents carry an explicit
  `model: sonnet|haiku`.
- Every pushed change moves the version, docs and CI included; the PM allocates
  the number at gate. Pre-1.0, a breaking change moves the second number.
- kiss-ref enums are `#[non_exhaustive]` from 0.4.0: a new `match` over one
  needs a wildcard arm.
- `cargo public-api` and `cargo semver-checks` are not installed here; an API
  diff in a PR is "read, not a tool run".
