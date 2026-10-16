# Unpopped roadmap

What this lane is doing, in what order, and **what each item waits on**. Read at
`origin/main` `2172526` (0.20.1) on 2026-10-16; every row names the ref it was
read at, so re-read it before acting on it. History is in `CHANGELOG.md`;
knowingly-undone work and its reasoning is `docs/deferred.md`. This file is the
queue, not the archive: delete a row when it lands.

## 1. In flight

| # | Item | State | Waits on |
|---|---|---|---|
| 1 | **Publish `0.20.1`** (and `0.19.1`): typed `PlanError::UnservableContraction` (#67); re-vendored derivation vectors (#65) | merged; `0.20.0` is published (PM read back) | PM publishes with the next code change |
| 2 | **baracuda's second-deriver harness** against the 0.20.x keyer: the PM's acceptance test (15 `ctr` FAIL -> 0) | not this lane's to run | baracuda lane; on a pass, comment on and close **#31** (its M/N/K and batch roles are implemented) |

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

## 3. Next (none unblocked; the PM assigns)

1. **`Approx` qualifier** (`docs/idiom-lifting-design.md` §4.3): the full fix for
   `__expf` / `__logf`, which #57 only made refuse. Needs KISS-coordinated
   numerics axes.
2. **Re-vendor on KISS change**, with the derivation vectors re-run
   (`cargo test -p unpopped-vocab --test kiss_derivation_vectors -- --nocapture`).

## 4. Held or blocked

| Item | Waits on |
|---|---|
| **Leg B**: on-device `__expf` round-trip (`docs/idiom-lifting-design.md` §5). baracuda's pin at `baracuda-cuda-parse/tests/convert.rs:84` fails when they take 0.15.0 (PM relayed it) | a rig: no P40 yet |
| Joint GPU milestone plan, `docs/joint-gpu-milestone-plan.md`. Ownership: Unpopped owns capability rows and per-sm plan decisions, baracuda owns CUDA spelling per sm. U1 (#40) and U2 (#42) are done; M4/M5 are blocked on the P40 | PM, before more U-items |
| Moving `ArchSku` into `baracuda-cuda-vocab` | PM says go |
| Groupshared / cooperative scope call (`docs/groupshared-sizing.md`) | CireSnave or PM |
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
