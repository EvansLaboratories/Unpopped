<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# The local pre-PR gate: full, fast, locked, daily (2026-11-26)

Approved by the PM through pm-kernels (A with the added cpu-c test-file run, C, and B measured). The forge's required checks are **unchanged**: this is the gate a lane runs before it opens a PR.

**Measured** (E1b full gate on the loaded laptop, 2026-11-26 15:36 to 16:19 local): workspace tests 6m47s (321 binaries, 2502 tests); the cpu-c leg 34m19s (79 binaries, 301 tests); fmt, clippy, U38d and two `cargo doc` runs 2m10s; 43m16s in all. The cpu-c leg is runtime, not compile time: all 79 cpu-c test binaries build warm in 0.75 s; it is one C compile and run per differential test (`kernel_launch_emit_agrees`: 12 tests, 74 s).

## `tools/gate.sh [--worktree DIR] [--mode auto|full|fast] [--verdict-only]`

* **Full gate:** workspace tests (cpu-c excluded), the cpu-c leg **run**, fmt per crate, clippy, U38d, `cargo doc` (`-D warnings`) with and without `--features fixtures`.
* **Fast gate:** the same, but the cpu-c leg is `cargo test -p unpopped-cpu-c --no-run` (a compile check of every cpu-c test: it still catches an exhaustive-match break such as a new `ElementKind` row) plus a **run of every file under `crates/unpopped-cpu-c/tests/` that the diff adds or edits**.
* **The rule, in the script:** a fast gate is allowed only if `git diff --name-only origin/main...HEAD` touches none of `FORBIDDEN` in `tools/gate.sh`: `crates/unpopped-cpu-c/` (`src/`, `Cargo.toml`, `build.rs`; in its `tests/` only an ADDED file is allowed, and run), `crates/unpopped/src/{kernel*,oracle/**,ir.rs,types.rs,cast_ref.rs,cfamily/**,contract.rs}`, `crates/unpopped-vocab/src/{element,structure_key}.rs`. The script prints the path check and the verdict (`VERDICT: fast gate` or `VERDICT: full gate`), and `--mode fast` on a forbidden path is **refused with exit 3**. `--mode auto` (the default) picks full on any forbidden path.
* **What a fast gate can miss:** an oracle-versus-serial-C disagreement in the existing differentials (`*_emit_agrees`, `launch_cases_agree`), which only a change to the oracle, the checker, the IR types or the emitter can cause; a PR that only adds files cannot change an existing differential. Breaking, union-row, IR, oracle, emitter and cpu-c-source PRs are **always full** by this rule.
* **The PR body says "fast gate" or "full gate" and quotes the printed verdict line.** A new kernel's vacuity mutant is run by hand and stated in the PR body (the script cannot know it).

## `tools/gate-locked.sh [gate.sh arguments]`

Runs one gate at a time, **first come first served**. The lock is an atomic `mkdir C:/cb/gate.lock` (Git Bash has no `flock`) with an `owner` file (pid, worktree, start epoch); it is released on exit, interrupt and termination, and is **stale**, and taken over, when its owner pid is dead or it is older than 3 h (`GATE_STALE_SECS`). Reads and edits stay outside the lock.

**The queue.** Before it tries the lock a caller takes a **ticket**, an atomic `mkdir C:/cb/gate-queue/<13-digit epoch ms>-<pid>-<worktree>` (`GATE_QUEUE_DIR`), and only the **lowest live ticket** may try the lock, so the order is the order of arrival (until 0.63.x the waiters polled `mkdir` every 30 s and the lock went to whoever polled first; a waiter could starve for hours behind lanes that happened to poll better). A ticket is **dead** when its pid is dead (the same `kill -0` the stale-lock rule uses) or when its heartbeat (touched every poll, `GATE_WAIT_SECS`, default 30 s) is older than `GATE_TICKET_TTL` (3 h): the next poller removes it and logs `removed a dead ticket` / `removed a stale ticket`, so a crashed waiter cannot block the line. A caller drops its ticket when it takes the lock and on exit, interrupt and termination. **An unusable queue directory means no queue, never blocked:** the caller then races for the lock as before. `tools/gate-locked.sh --status` prints the holder (pid, worktree, seconds running) and the numbered queue, and takes nothing.

**Mixed protocols.** The lock itself is still `mkdir C:/cb/gate.lock`, so a lane whose `tools/gate-locked.sh` predates the queue (every worktree has its own copy until it merges `main`) still takes part: it races for the lock as it always did and can overtake a ticketed waiter. The starvation fix therefore reaches a lane only when that lane has merged `main`. `GATE_LOCK_DIR`, `GATE_QUEUE_DIR` and `GATE_CMD` exist for the tests.

## `tools/gate-daily.sh`

A full gate on a detached worktree at `origin/main`, under the lock, for a quiet laptop. **Red means a hotfix PR first, before any new leaf merges.** Scheduling it is the primary lane's.

## Tests

`bash tools/test-gate.sh`: the path rule (every forbidden pattern forces full; docs, new tests and cpu-c test files are fast; `--mode fast` on a forbidden path exits 3; `--mode full` is always allowed), a **mutant** (the oracle pattern deleted from a copy of the script: the same assertions must fail), and the lock and its queue (two simultaneous gates do not overlap and the second waits; a dead-pid lock and an over-age lock are taken over; a fresh live lock is not; the OLDER of two waiters runs first although the newer polls far more often, with a **mutant** that lets any waiter take the lock, which the same assertions catch; three waiters run in order of arrival; a dead-pid ticket and a no-heartbeat ticket are removed and logged; a live older ticket holds the line; a killed waiter leaves no ticket; a dead holder with a waiter in line; an unusable queue directory; `--status`).
