#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Tests of tools/gate.sh (the fast-gate path rule) and tools/gate-locked.sh (the lock). usage: bash tools/test-gate.sh
# Each assertion has its control; the rule is also MUTATED (one pattern removed) and the same assertions must then FAIL.
HERE="$(cd "$(dirname "$0")" && pwd)"
FAILS=0; T=$(mktemp -d)
ok()   { echo "ok   $1"; }
bad()  { echo "FAIL $1"; FAILS=$((FAILS+1)); }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }

# --- a scratch repo with an origin/main ref ---------------------------------------------------------------------------------------
mkrepo() {
  rm -rf "$T/r"; mkdir -p "$T/r"; cd "$T/r" || exit 2
  git init -q . && git config user.email t@t && git config user.name t && git config core.autocrlf false && git config core.safecrlf false
  mkdir -p crates/unpopped/src/oracle crates/unpopped/src/cfamily crates/unpopped/tests docs crates/unpopped-cpu-c/src crates/unpopped-cpu-c/tests crates/unpopped-vocab/src
  for f in crates/unpopped/src/oracle/x.rs crates/unpopped/src/kernel.rs crates/unpopped/src/kernel_check.rs crates/unpopped/src/ir.rs crates/unpopped/src/types.rs \
           crates/unpopped/src/cast_ref.rs crates/unpopped/src/cfamily/c.rs crates/unpopped/src/contract.rs crates/unpopped-vocab/src/element.rs \
           crates/unpopped-vocab/src/structure_key.rs crates/unpopped-cpu-c/src/lib.rs crates/unpopped-cpu-c/tests/t.rs crates/unpopped/tests/a.rs docs/a.md; do echo base > "$f"; done
  git add -A 2>/dev/null && git commit -qm base && git update-ref refs/remotes/origin/main HEAD
  git checkout -q -b work
}
# touch FILE then ask for a verdict with MODE; prints the verdict text and the exit code
verdict() { # script file mode
  cd "$T/r" || exit 2; git checkout -q -- . ; git reset -q --hard origin/main
  echo changed >> "$2"; git commit -qam c 2>/dev/null
  out=$(bash "$1" --worktree "$T/r" --mode "$3" --verdict-only 2>&1); rc=$?
  echo "$out"; echo "rc=$rc"
}

verdict_add() { # script newfile mode: ADD a new file (status A)
  cd "$T/r" || exit 2; git checkout -q -- . ; git reset -q --hard origin/main
  mkdir -p "$(dirname "$2")"; echo new > "$2"; git add -A 2>/dev/null; git commit -qm c 2>/dev/null
  out=$(bash "$1" --worktree "$T/r" --mode "$3" --verdict-only 2>&1); rc=$?
  echo "$out"; echo "rc=$rc"
}
verdict_del() { # script file mode: DELETE an existing file (status D)
  cd "$T/r" || exit 2; git checkout -q -- . ; git reset -q --hard origin/main
  git rm -q "$2" 2>/dev/null; git commit -qm c 2>/dev/null
  out=$(bash "$1" --worktree "$T/r" --mode "$3" --verdict-only 2>&1); rc=$?
  echo "$out"; echo "rc=$rc"
}

run_rule_tests() { # script ; echoes number of failures
  local s="$1" before=$FAILS
  mkrepo
  v=$(verdict "$s" docs/a.md auto);                       check "docs only is a fast gate"        "[[ \"$v\" == *'VERDICT: fast gate'* && \"$v\" == *rc=0* ]]"
  v=$(verdict "$s" crates/unpopped/tests/a.rs auto);      check "a new unpopped test file is fast" "[[ \"$v\" == *'VERDICT: fast gate'* ]]"
  v=$(verdict_add "$s" crates/unpopped-cpu-c/tests/new_agrees.rs auto); check "ADDING a cpu-c test file is fast and the file is printed to be RUN" "[[ \"$v\" == *'VERDICT: fast gate'* && \"$v\" == *new_agrees.rs* ]]"
  v=$(verdict_add "$s" crates/unpopped-cpu-c/tests/new_agrees.rs fast); check "--mode fast on an ADDED cpu-c test file is allowed" "[[ \"$v\" == *'VERDICT: fast gate'* && \"$v\" == *rc=0* ]]"
  v=$(verdict "$s" crates/unpopped-cpu-c/tests/t.rs auto); check "MODIFYING an existing cpu-c test file forces FULL" "[[ \"$v\" == *'VERDICT: full gate'* && \"$v\" == *'existing cpu-c test file, status M'* ]]"
  v=$(verdict "$s" crates/unpopped-cpu-c/tests/t.rs fast); check "--mode fast on a MODIFIED existing cpu-c test file is REFUSED (exit 3)" "[[ \"$v\" == *REFUSED* && \"$v\" == *rc=3* ]]"
  v=$(verdict_del "$s" crates/unpopped-cpu-c/tests/t.rs fast); check "DELETING an existing cpu-c test file is REFUSED for fast (exit 3)" "[[ \"$v\" == *REFUSED* && \"$v\" == *rc=3* ]]"
  for f in crates/unpopped/src/oracle/x.rs crates/unpopped/src/kernel.rs crates/unpopped/src/kernel_check.rs crates/unpopped/src/ir.rs crates/unpopped/src/types.rs \
           crates/unpopped/src/cast_ref.rs crates/unpopped/src/cfamily/c.rs crates/unpopped/src/contract.rs crates/unpopped-vocab/src/element.rs \
           crates/unpopped-vocab/src/structure_key.rs crates/unpopped-cpu-c/src/lib.rs; do
    v=$(verdict "$s" "$f" auto);  check "auto forces FULL for $f" "[[ \"$v\" == *'VERDICT: full gate'* && \"$v\" == *'forbidden for a fast gate'* ]]"
    v=$(verdict "$s" "$f" fast);  check "--mode fast is REFUSED (exit 3) for $f" "[[ \"$v\" == *REFUSED* && \"$v\" == *rc=3* ]]"
  done
  v=$(verdict "$s" docs/a.md full); check "--mode full is always allowed" "[[ \"$v\" == *'VERDICT: full gate'* && \"$v\" == *rc=0* ]]"
  echo $((FAILS - before)) > "$T/n"
}

echo "== the path rule (red first: the assertions below fail against a mutated rule)"
run_rule_tests "$HERE/gate.sh"
echo "== MUTATION: remove the oracle pattern from a copy of the script; the SAME assertions must now fail"
cp "$HERE/gate.sh" "$T/gate-mut.sh"; sed -i "/src\/oracle/d" "$T/gate-mut.sh"
if cmp -s "$HERE/gate.sh" "$T/gate-mut.sh"; then bad "the mutation applied"; else ok "the mutation applied"; fi
FAILS_BEFORE=$FAILS; run_rule_tests "$T/gate-mut.sh" | grep -c "^FAIL" > "$T/mutfails"
MUT=$(cat "$T/mutfails"); FAILS=$FAILS_BEFORE   # the mutant's failures are the point, not ours
check "the mutated rule is CAUGHT (at least one assertion failed)" "[ $MUT -ge 1 ]"

echo "== MUTATION 2: the status rule off (an existing cpu-c test treated like an added one); the same assertions must fail"
# shellcheck disable=SC2016  # the pattern is a literal sed regex: $st must NOT expand
cp "$HERE/gate.sh" "$T/gate-mut2.sh"; sed -i 's/if \[ "\$st" = "A" \]; then ADDED_CPUC_TESTS/if true; then ADDED_CPUC_TESTS/' "$T/gate-mut2.sh"
if cmp -s "$HERE/gate.sh" "$T/gate-mut2.sh"; then bad "mutation 2 applied"; else ok "mutation 2 applied"; fi
FAILS_BEFORE=$FAILS; run_rule_tests "$T/gate-mut2.sh" | grep -c "^FAIL" > "$T/mutfails2"
MUT2=$(cat "$T/mutfails2"); FAILS=$FAILS_BEFORE
check "the status-rule mutant is CAUGHT (at least one assertion failed)" "[ $MUT2 -ge 1 ]"
echo "== the lock"
L="$T/lock"; export GATE_LOCK_DIR="$L" GATE_WAIT_SECS=1
export GATE_CMD="bash -c 'echo \"[\$(date +%s)] start \$\$\"; sleep 4; echo \"[\$(date +%s)] end \$\$\"'"
( bash "$HERE/gate-locked.sh" > "$T/g1.out" 2>&1 ) & P1=$!
sleep 1
( bash "$HERE/gate-locked.sh" > "$T/g2.out" 2>&1 ) & P2=$!
wait $P1 $P2
s1=$(grep -o '\[[0-9]*\] start' "$T/g1.out" | tr -dc 0-9); e1=$(grep -o '\[[0-9]*\] end' "$T/g1.out" | tr -dc 0-9)
s2=$(grep -o '\[[0-9]*\] start' "$T/g2.out" | tr -dc 0-9); e2=$(grep -o '\[[0-9]*\] end' "$T/g2.out" | tr -dc 0-9)
check "both gates ran" "[ -n \"$s1\" ] && [ -n \"$e1\" ] && [ -n \"$s2\" ] && [ -n \"$e2\" ]"
check "the second gate waited (printed waiting)" "grep -q 'waiting for the gate lock' '$T/g2.out'"
check "the second gate started only after the first ended (no overlap)" "[ \"$s2\" -ge \"$e1\" ]"
check "the lock is released afterwards" "[ ! -d '$L' ]"
echo "== stale: a dead pid"
mkdir -p "$L"; printf '999999\nsomewhere\n%s\n' "$(date +%s)" > "$L/owner"
bash "$HERE/gate-locked.sh" > "$T/g3.out" 2>&1
check "a lock with a dead owner is taken over" "grep -q 'stale lock (owner pid 999999 is dead)' '$T/g3.out' && grep -q 'start' '$T/g3.out'"
echo "== stale: too old (live pid)"
mkdir -p "$L"; printf '%s\nsomewhere\n%s\n' "$$" "$(( $(date +%s) - 100 ))" > "$L/owner"
GATE_STALE_SECS=10 bash "$HERE/gate-locked.sh" > "$T/g4.out" 2>&1
check "a lock older than the limit is taken over" "grep -q 'older than 10s' '$T/g4.out' && grep -q 'start' '$T/g4.out'"
echo "== control: a fresh live lock is NOT taken over (waits, then we kill the waiter)"
mkdir -p "$L"; printf '%s\nsomewhere\n%s\n' "$$" "$(date +%s)" > "$L/owner"
( GATE_WAIT_SECS=1 bash "$HERE/gate-locked.sh" > "$T/g5.out" 2>&1 ) & P5=$!; sleep 3; kill $P5 2>/dev/null; wait $P5 2>/dev/null
check "a live fresh lock makes the waiter wait" "grep -q 'waiting for the gate lock' '$T/g5.out' && ! grep -q 'start' '$T/g5.out'"
rm -rf "$L" "$T"
echo "failures: $FAILS"; [ "$FAILS" = 0 ]
