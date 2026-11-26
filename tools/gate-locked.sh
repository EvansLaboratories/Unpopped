#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Run the local gate under a machine-wide lock so two gates never run at once. usage: tools/gate-locked.sh [gate.sh arguments]
#
# The lock is an atomic `mkdir` (no flock in Git Bash) at $GATE_LOCK_DIR (default C:/cb/gate.lock) with an `owner` file (pid, worktree,
# start epoch). A waiter polls every 30 s (GATE_WAIT_SECS). A lock is STALE, and taken over, when its owner pid is dead or it is
# older than 3 h (GATE_STALE_SECS). The lock is released on exit, interrupt and termination. Reads and edits stay outside the lock.
LOCK="${GATE_LOCK_DIR:-C:/cb/gate.lock}"
WAIT="${GATE_WAIT_SECS:-30}"
STALE="${GATE_STALE_SECS:-10800}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CMD="${GATE_CMD:-bash \"$HERE/gate.sh\" \"\$@\"}"
mkdir -p "$(dirname "$LOCK")"
release() { if [ "$(sed -n 1p "$LOCK/owner" 2>/dev/null)" = "$$" ]; then rm -rf "$LOCK"; fi; }
trap release EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
while ! mkdir "$LOCK" 2>/dev/null; do
  opid=$(sed -n 1p "$LOCK/owner" 2>/dev/null); ostart=$(sed -n 3p "$LOCK/owner" 2>/dev/null); now=$(date +%s)
  if [ -z "$opid" ]; then
    # the owner file is written a moment after mkdir: give a creator a few seconds, then call it stale
    sleep 3; [ -z "$(sed -n 1p "$LOCK/owner" 2>/dev/null)" ] && { echo "gate-locked: stale lock (no owner), taking over"; rm -rf "$LOCK"; continue; }
    continue
  fi
  if ! kill -0 "$opid" 2>/dev/null; then echo "gate-locked: stale lock (owner pid $opid is dead), taking over"; rm -rf "$LOCK"; continue; fi
  if [ -n "$ostart" ] && [ $((now - ostart)) -gt "$STALE" ]; then echo "gate-locked: stale lock (owner $opid older than ${STALE}s), taking over"; rm -rf "$LOCK"; continue; fi
  echo "gate-locked: waiting for the gate lock held by pid $opid ($(sed -n 2p "$LOCK/owner" 2>/dev/null)), checking again in ${WAIT}s"
  sleep "$WAIT"
done
printf '%s\n%s\n%s\n' "$$" "$PWD" "$(date +%s)" > "$LOCK/owner"
echo "gate-locked: lock taken by pid $$ at $(date +%H:%M:%S)"
bash -c "$CMD" _ "$@"
rc=$?
echo "gate-locked: released at $(date +%H:%M:%S), exit $rc"
exit $rc
