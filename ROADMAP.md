# Unpopped roadmap

What this lane is doing, in what order, and **what each item waits on**. Read at
`origin/main` `2a78c233` (0.17.0) on 2026-10-14; every row names the ref it was
read at, so re-read it before acting on it. History is in `CHANGELOG.md`;
knowingly-undone work and its reasoning is `docs/deferred.md`. This file is the
queue, not the archive: delete a row when it lands.

Items marked **(PM)** were assigned or ruled by the portfolio PM on 2026-10-14
and are relayed here, not independently verified.

## 1. In flight

| # | Item | State | Waits on |
|---|---|---|---|
| 1 | **Publish `0.17.0`** (sk5): `STRUCTURE_KEY_VERSION` 4 → 5, derived seven-class `cell_class`, keyed `dequant_form`, extent-1 mask, and four keyer defects | **merged** (#58, `2a78c233`) | PM publishes; baracuda takes it from crates.io, never from git or a path |
| 2 | **Hygiene PR** (PM, from CodeRipper triage at coderipper `6d00540f`): `.github/CODEOWNERS` (`* @ciresnave`); pin every `uses:` in `verify.yml` to a full SHA with the version in a trailing comment; top-level `permissions: contents: read`; DOC-009 glossary (15 doc pages, threshold 10) | not started; docs and CI only, no behaviour change | nothing |

### What #58 closed, and what it did not

- **Measured** against KISS `ff12265` derivation vectors (blob `697aa543`, 36):
  PASS 35, FAIL 0, DECLINE-DIFFERS 0, SKIP 1. 10 carry `spec_golden`; the other
  21 positives are reference-deriver output, so agreement there is two derivers
  agreeing.
- **Not covered:** a declared contraction. The weight role became a declared
  input in `0.17.3`; the M/N/K roles are still positional (issue #31), and the
  batch role is undefined upstream (KISS#543), which is the one skipped vector.
- **Baracuda** regenerates its 3 sk4 test sites and 2 goldens once `0.17.0` is
  on crates.io, and fuel is blocked on the same (PM). The `[NOTICE]` went to
  the baracuda lane on merge, with a request for its harness table
  (`unpopped-vocab` + `serde_json`): that is the independent measurement, since
  21 of the 36 vectors are KISS reference output.

## 2. KISS items this lane consumes

| Item | State | Where |
|---|---|---|
| `structure_key_vectors.json` blob `e5eec85c`, `structure_key_derivation_vectors.json` blob `697aa543`, `dtype_manifest.json` blob `24ff38f0` | vendored from KISS `ff12265`; the KISS lane measured the first two byte-identical at `64e0cd8` (0.2.40) | `crates/unpopped-vocab/kiss/` |
| R1.4 innermost axis of extent 0 (§6.3-0011, §6.5-0012) | **closed**: `[8,0]` derives `da`/`v1`, measured; KISS reads it the same way | `tests/mixed_rank_frames_and_the_innermost_axis.rs`, derivation vector `innermost_axis_extent_zero_is_axis_rank_minus_1` |
| R1.6 target-match vector (§6.8-0002) | **closed**: `target_match_vectors` has been vendored and asserted since the `bc16715` copy | `kiss_byte_match.rs::target_match_vectors_are_byte_exact` |
| KISS#543 batch role | **ruled** (KISS 0.4.2, §6.6-0023) and implemented at 0.19.0 behind `CellFacts::contraction_roles`; 63 of 63 vectors match, `KNOWN_GAP` empty | `tests/kiss_derivation_vectors.rs` |
| §6.6-0019 weight role (`<wdt>`) | **closed** in `0.17.3`: `CellFacts::weight_operand`; `WeightRoleNotDeclared` is this crate's decline name, not KISS's | issue #31 stays open for M/N/K |
| §6.6-0021 | the infallible `structure_key` still keys a scale dtype at operand 0; `try_structure_key` declines. Deprecating the infallible form means migrating ~200 callers, which the PM ruled is **not this lane** (bulk mechanical work, CireSnave's cost rule) | `CHANGELOG.md` |

KISS `main` moves faster than this lane re-vendors. Re-vendor when KISS says a
blob changed, not on a schedule, and record the commit in `kiss/README.md`.

## 3. Next (none started; the PM assigns)

1. **Explicit M/N/K axis roles** (§6.6-0016). The weight role is now declared
   (`CellFacts::weight_operand`, ruled option A, `0.17.3`); M/N/K are still
   inferred from `lhs`/`rhs`/`out` shapes, which §6.6-0016 forbids, and rank-3
   needs the batch role that KISS#543 has not ruled (option B). Not started.
2. **`Approx` qualifier** (`docs/idiom-lifting-design.md` §4.3): the full fix for
   `__expf` / `__logf`, which #57 only made refuse. Needs KISS-coordinated
   numerics axes.
3. **Re-vendor on KISS change**, with the derivation vectors re-run
   (`cargo test -p unpopped-vocab --test kiss_derivation_vectors -- --nocapture`).

## 4. Held or blocked

| Item | Waits on |
|---|---|
| **Leg B**: on-device `__expf` round-trip (`docs/idiom-lifting-design.md` §5). baracuda's pin at `baracuda-cuda-parse/tests/convert.rs:84` fails when they take 0.15.0 (PM relayed it) | a rig: no P40 yet |
| Joint GPU milestone plan, `docs/joint-gpu-milestone-plan.md`. Ownership: Unpopped owns capability rows and per-sm plan decisions, baracuda owns CUDA spelling per sm. U1 (#40) and U2 (#42) are done; M4/M5 are blocked on the P40 | PM, before more U-items |
| Moving `ArchSku` into `baracuda-cuda-vocab` | PM says go |
| Groupshared / cooperative scope call (`docs/groupshared-sizing.md`) | CireSnave or PM |
| Unit-axis stride-0 mask question | **ruled** upstream (KISS-CLASSIFY-6.6-0008, KISS#520) and implemented in #58; delete this row at merge |
| CpuC declines `Select` for every narrow float | parity gap, not a bug; no owner |
| Open issues #4 (published-vs-workspace divergence, the class), #6 (`supports_dtype` has three states) | design decisions, not scheduled |

## 5. Standing constraints

- Never run, launch or spawn anything on Opus; subagents carry an explicit
  `model: sonnet|haiku`.
- Every pushed change moves the version, docs and CI included; the PM allocates
  the number at gate. Pre-1.0, a breaking change moves the second number.
- kiss-ref enums are `#[non_exhaustive]` from 0.4.0: a new `match` over one
  needs a wildcard arm.
- `cargo public-api` and `cargo semver-checks` are not installed here; an API
  diff in a PR is "read, not a tool run".
