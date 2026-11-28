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
cp "$HERE/gate.sh" "$T/gate-mut2.sh"; sed -i 's/if \[ ".st" = "A" \]; then ADDED_CPUC_TESTS/if true; then ADDED_CPUC_TESTS/' "$T/gate-mut2.sh"
if cmp -s "$HERE/gate.sh" "$T/gate-mut2.sh"; then bad "mutation 2 applied"; else ok "mutation 2 applied"; fi
FAILS_BEFORE=$FAILS; run_rule_tests "$T/gate-mut2.sh" | grep -c "^FAIL" > "$T/mutfails2"
MUT2=$(cat "$T/mutfails2"); FAILS=$FAILS_BEFORE
check "the status-rule mutant is CAUGHT (at least one assertion failed)" "[ $MUT2 -ge 1 ]"
echo "== the lock"
L="$T/lock"; export GATE_LOCK_DIR="$L" GATE_WAIT_SECS=1 GATE_QUEUE_DIR="$T/queue"
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
# --- the queue: first come first served (a ticket per waiter, the lowest LIVE ticket takes the lock) -----------------------------
# a gate whose command prints "start <tag>" / "end <tag>" with a timestamp, for ordering assertions
tagged() { export GATE_CMD="bash -c 'echo \"[\$(date +%s%N)] start $1\"; sleep $2; echo \"[\$(date +%s%N)] end $1\"'"; }
startof() { grep -o '\[[0-9]*\] start' "$1" | tr -dc 0-9; }
endof()   { grep -o '\[[0-9]*\] end' "$1" | tr -dc 0-9; }
run_queue_order() { # script: a holder, then an OLDER slow-polling waiter and a NEWER fast-polling one: FIFO says the older goes first
  local S="$1"
  rm -rf "$L" "$T/queue"
  tagged H 5; ( GATE_WAIT_SECS=1 bash "$S" > "$T/h.out" 2>&1 ) & HP=$!
  sleep 1.5
  tagged A 1; ( GATE_WAIT_SECS=6 bash "$S" > "$T/a.out" 2>&1 ) & AP=$!   # older ticket, polls every 6 s
  sleep 1
  tagged B 1; ( GATE_WAIT_SECS=1 bash "$S" > "$T/b.out" 2>&1 ) & BP=$!   # newer ticket, polls every second
  wait "$HP" "$AP" "$BP"
}
echo "== queue: two waiters (the older takes the lock first, although the newer polls far more often)"
run_queue_order "$HERE/gate-locked.sh"
sa=$(startof "$T/a.out"); sb=$(startof "$T/b.out"); ea=$(endof "$T/a.out")
check "both waiters ran" "[ -n \"$sa\" ] && [ -n \"$sb\" ]"
check "the older waiter started first" "[ \"$sa\" -lt \"$sb\" ]"
check "the newer waiter started after the older one ended" "[ \"$sb\" -ge \"$ea\" ]"
check "the newer waiter was told its position" "grep -q 'waiting in the queue, position 2 of 2' '$T/b.out'"
echo "== MUTATION 3: any waiter may take the lock (the head-of-line test removed); the same assertions must fail"
# shellcheck disable=SC2016  # the pattern is a literal source line: its $ must not expand
cp "$HERE/gate-locked.sh" "$T/gl-mut.sh"; sed -i 's/if \[ "\$first" != "\$TICKET" \]; then/if false; then/' "$T/gl-mut.sh"
if cmp -s "$HERE/gate-locked.sh" "$T/gl-mut.sh"; then bad "mutation 3 applied"; else ok "mutation 3 applied"; fi
cp "$T/gl-mut.sh" "$HERE/gl-mut.tmp.sh"
run_queue_order "$HERE/gl-mut.tmp.sh"; rm -f "$HERE/gl-mut.tmp.sh"
sa=$(startof "$T/a.out"); sb=$(startof "$T/b.out")
check "the take-any-ticket mutant is CAUGHT (the newer waiter overtook the older one)" "[ -z \"$sa\" ] || [ -z \"$sb\" ] || [ \"$sa\" -gt \"$sb\" ]"
echo "== queue: three waiters run in order of arrival"
rm -rf "$L" "$T/queue"
tagged H 4; ( GATE_WAIT_SECS=1 bash "$HERE/gate-locked.sh" > "$T/h.out" 2>&1 ) & HP=$!
sleep 1
for w in 1 2 3; do tagged W$w 1; ( GATE_WAIT_SECS=$((4 - w)) bash "$HERE/gate-locked.sh" > "$T/w$w.out" 2>&1 ) & eval "WP$w=$!"; sleep 0.7; done
wait "$HP" "$WP1" "$WP2" "$WP3"
s1=$(startof "$T/w1.out"); s2=$(startof "$T/w2.out"); s3=$(startof "$T/w3.out")
check "three waiters ran in the order they arrived" "[ -n \"$s1\" ] && [ \"$s1\" -lt \"$s2\" ] && [ \"$s2\" -lt \"$s3\" ]"
check "the queue is empty afterwards" "[ -z \"\$(ls -A '$T/queue' 2>/dev/null)\" ] && [ ! -d '$L' ]"
echo "== queue: a dead waiter's ticket does not block the line"
rm -rf "$L" "$T/queue"; mkdir -p "$T/queue/0000000000001-999999-ghost"; echo 0 > "$T/queue/0000000000001-999999-ghost/hb"
tagged G 1; GATE_WAIT_SECS=1 bash "$HERE/gate-locked.sh" > "$T/g6.out" 2>&1
check "the dead ticket was skipped and removed (and logged)" "grep -q 'removed a dead ticket 0000000000001-999999-ghost' '$T/g6.out' && grep -q 'start G' '$T/g6.out' && [ ! -d '$T/queue/0000000000001-999999-ghost' ]"
echo "== queue: a ticket with no heartbeat for longer than the TTL is removed even when its pid is alive"
rm -rf "$L" "$T/queue"; mkdir -p "$T/queue/0000000000002-$$-hung"; echo 1 > "$T/queue/0000000000002-$$-hung/hb"
tagged S 1; GATE_TICKET_TTL=5 GATE_WAIT_SECS=1 bash "$HERE/gate-locked.sh" > "$T/g7.out" 2>&1
check "the stale-heartbeat ticket was removed" "grep -q 'removed a stale ticket 0000000000002-$$-hung' '$T/g7.out' && grep -q 'start S' '$T/g7.out'"
echo "== queue: a LIVE older ticket holds the line (the control: the newer waiter does not jump it)"
rm -rf "$L" "$T/queue"; mkdir -p "$T/queue/0000000000003-$$-ahead"; date +%s > "$T/queue/0000000000003-$$-ahead/hb"
tagged C 1; ( GATE_WAIT_SECS=1 bash "$HERE/gate-locked.sh" > "$T/g8.out" 2>&1 ) & P8=$!; sleep 3; kill $P8 2>/dev/null; wait $P8 2>/dev/null
check "the waiter behind a live ticket waits and never starts" "grep -q 'position 2 of 2' '$T/g8.out' && ! grep -q 'start C' '$T/g8.out'"
check "a killed waiter leaves no ticket" "[ \"\$(ls '$T/queue')\" = '0000000000003-$$-ahead' ]"
echo "== queue: the holder crashes (a dead owner) with a waiter in line"
rm -rf "$L" "$T/queue"; mkdir -p "$L"; printf '999999
somewhere
%s
' "$(date +%s)" > "$L/owner"
tagged K 1; GATE_WAIT_SECS=1 bash "$HERE/gate-locked.sh" > "$T/g9.out" 2>&1
check "the dead holder's lock is taken over by the waiter at the head" "grep -q 'stale lock (owner pid 999999 is dead)' '$T/g9.out' && grep -q 'start K' '$T/g9.out'"
echo "== queue: an unusable queue directory means no queue, not blocked"
rm -rf "$L" "$T/queue"; : > "$T/notadir"
tagged U 1; GATE_QUEUE_DIR="$T/notadir/q" GATE_WAIT_SECS=1 bash "$HERE/gate-locked.sh" > "$T/g10.out" 2>&1
check "with a queue directory that cannot be made the gate still runs" "grep -q 'start U' '$T/g10.out'"
echo "== --status shows the holder and the queue"
rm -rf "$L" "$T/queue"; mkdir -p "$L" "$T/queue/0000000000004-$$-first" "$T/queue/0000000000005-$$-second"
printf '%s
somewhere
%s
' "$$" "$(date +%s)" > "$L/owner"; for t in first second; do d=$(ls -d "$T"/queue/*-$t); date +%s > "$d/hb"; done
bash "$HERE/gate-locked.sh" --status > "$T/st.out" 2>&1
check "--status names the holder, the queue length and the order" "grep -q '^holder: pid $$,' '$T/st.out' && grep -q '^queue: 2 waiting' '$T/st.out' && grep -n '1. 0000000000004' '$T/st.out' >/dev/null && grep -q '2. 0000000000005' '$T/st.out'"
check "--status took nothing" "[ -d '$L' ] && [ \"\$(sed -n 1p '$L/owner')\" = '$$' ]"
rm -rf "$L" "$T"
echo "failures: $FAILS"; [ "$FAILS" = 0 ]
