#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# The local pre-PR gate (full or fast). usage: tools/gate.sh [--worktree DIR] [--mode auto|full|fast] [--verdict-only]
#
#   full gate: ws test leg, cpu-c test leg (RUNS every differential), fmt, clippy, U38d, cargo doc x2.
#   fast gate: the same, but the cpu-c leg is `cargo test -p unpopped-cpu-c --no-run` (a compile check of every cpu-c test, which
#              still catches an exhaustive-match break) plus a RUN of every cpu-c test file the diff ADDS (status A); a diff that MODIFIES, deletes or renames an existing one forces FULL.
#
# A fast gate is allowed ONLY when `git diff --name-only $GATE_BASE...HEAD` touches none of FORBIDDEN below. The rule lives here, is
# printed with its verdict, and refuses `--mode fast` on a forbidden path (exit 3). `--mode auto` (default) picks full on any
# forbidden path. Breaking, union-row, IR, oracle, emitter and cpu-c-source PRs are always FULL by this rule. The forge's required
# checks are unchanged: this is the local pre-PR gate. A daily FULL gate on origin/main (tools/gate-daily.sh) catches drift the fast
# gates skip.
set -o pipefail
WT="${PWD}"; MODE=auto; VERDICT_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --worktree) WT="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --verdict-only) VERDICT_ONLY=1; shift;;
    *) echo "gate.sh: unknown argument $1" >&2; exit 2;;
  esac
done
BASE="${GATE_BASE:-origin/main}"

# Paths a fast gate may not touch (extended regexes over repo-relative paths). In the cpu-c TESTS directory only a file the diff ADDS
# is allowed (and it is RUN); modifying, deleting or renaming an existing one forces FULL (handled below); the other cpu-c paths are
# forbidden here.
FORBIDDEN=(
  '^crates/unpopped-cpu-c/(src/|Cargo\.toml|build\.rs)'
  '^crates/unpopped/src/kernel'
  '^crates/unpopped/src/oracle/'
  '^crates/unpopped/src/ir\.rs$'
  '^crates/unpopped/src/types\.rs$'
  '^crates/unpopped/src/cast_ref\.rs$'
  '^crates/unpopped/src/cfamily/'
  '^crates/unpopped/src/contract\.rs$'
  '^crates/unpopped-vocab/src/element\.rs$'
  '^crates/unpopped-vocab/src/structure_key\.rs$'
)

cd "$WT" || { echo "gate.sh: no worktree $WT" >&2; exit 2; }
# `--no-renames`: a rename shows as D + A, so renaming an existing cpu-c test counts as deleting one (full gate).
STATUS_LINES=$(git diff --name-status --no-renames "$BASE"...HEAD 2>/dev/null)
CHANGED=$(printf '%s
' "$STATUS_LINES" | cut -f2-)
if [ -z "$CHANGED" ] && ! git rev-parse --verify -q "$BASE" >/dev/null; then
  echo "gate.sh: base $BASE not found; refusing fast, running FULL" >&2; STATUS_LINES=$'M	<unknown base>'; CHANGED="<unknown base>"
fi
HITS=""; ADDED_CPUC_TESTS=""
while IFS=$'	' read -r st f; do
  [ -z "$f" ] && continue
  if [ "$f" = "<unknown base>" ]; then HITS="$HITS$f
"; continue; fi
  # an EXISTING cpu-c test file modified, deleted or renamed can weaken a differential without the full run noticing: full gate
  if [[ "$f" =~ ^crates/unpopped-cpu-c/tests/[^/]+\.rs$ ]]; then
    if [ "$st" = "A" ]; then ADDED_CPUC_TESTS="$ADDED_CPUC_TESTS $f"; else HITS="$HITS$f   (existing cpu-c test file, status $st: full gate)
"; fi
    continue
  fi
  for re in "${FORBIDDEN[@]}"; do
    if [[ "$f" =~ $re ]]; then HITS="$HITS$f   (matches $re)
"; break; fi
  done
done <<< "$STATUS_LINES"
NCH=$(printf '%s\n' "$CHANGED" | grep -c .)
echo "== path check: $NCH changed path(s) against $BASE...HEAD, ${#FORBIDDEN[@]} forbidden pattern(s)"
if [ -n "$HITS" ]; then printf 'forbidden for a fast gate:\n%s' "$HITS"; else echo "no forbidden path touched"; fi
if [ "$MODE" = fast ] && [ -n "$HITS" ]; then echo "VERDICT: REFUSED (a fast gate was asked for, a forbidden path is touched): run the full gate" ; exit 3; fi
if [ "$MODE" = full ] || { [ "$MODE" = auto ] && [ -n "$HITS" ]; }; then VERDICT=full; else VERDICT=fast; fi
echo "VERDICT: $VERDICT gate"
[ "$VERDICT" = fast ] && echo "cpu-c test files ADDED by this diff, to be RUN:${ADDED_CPUC_TESTS:- none}"
[ "$VERDICT_ONLY" = 1 ] && exit 0

CPUC_TESTS=$(printf '%s\n' "$CHANGED" | grep -E '^crates/unpopped-cpu-c/tests/[^/]+\.rs$' | sed -E 's#.*/([^/]+)\.rs$#\1#')

echo "== ws"; cargo test --workspace --exclude unpopped-cpu-c --no-fail-fast -j 2 2>&1 | tee ws.log | grep -E "^test result|FAILED|failed|error(\[|:)" | awk '/^test result/{b++; split($0,a," "); p+=a[4]; f+=a[6]} /FAILED|error/{print} END{print "bins",b,"passed",p,"failed",f}'
echo "== cpu-c ($VERDICT)"
if [ "$VERDICT" = full ]; then
  cargo test -p unpopped-cpu-c --no-fail-fast -j 2 2>&1 | tee cpuc.log | grep -E "^test result|FAILED|error(\[|:)" | awk '/^test result/{b++; split($0,a," "); p+=a[4]; f+=a[6]} /FAILED|error/{print} END{print "bins",b,"passed",p,"failed",f}'
else
  cargo test -p unpopped-cpu-c --no-run -j 2 2>&1 | tee cpuc.log | grep -E "^error" ; echo "cpu-c compile check (no-run): exit ${PIPESTATUS[0]}"
  for t in $CPUC_TESTS; do
    echo "run cpu-c test file $t"; cargo test -p unpopped-cpu-c --test "$t" -j 2 2>&1 | tee -a cpuc.log | grep -E "^test result|FAILED|error(\[|:)"
  done
fi
echo "== fmt"; for c in crates/*; do (cd "$c" && cargo fmt --check >/dev/null 2>&1 && echo "fmt ok $c" || echo "fmt DIRTY $c"); done
echo "== clippy"; cargo clippy --workspace --all-targets -j 2 2>&1 | grep -cE "^(warning|error)"
echo "== U38d"; grep -rn "with_struct_layout\|layout_name\|struct_layouts" crates | wc -l
echo "== doc"; RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps -j 2 >/dev/null 2>&1; echo "doc-exit $?"
RUSTDOCFLAGS="-D warnings" cargo doc -p unpopped --features fixtures --no-deps -j 2 >/dev/null 2>&1; echo "doc-fixtures-exit $?"
echo "GATE-VERDICT: $VERDICT gate"
echo DONE
