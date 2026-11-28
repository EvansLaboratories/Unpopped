#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Run the local gate under a machine-wide lock so two gates never run at once, FIRST COME FIRST SERVED.
# usage: tools/gate-locked.sh [gate.sh arguments]      run the gate when it is this caller's turn
#        tools/gate-locked.sh --status                 print the holder and the queue, take nothing
#
# The lock is an atomic `mkdir` (no flock in Git Bash) at $GATE_LOCK_DIR (default C:/cb/gate.lock) with an `owner` file (pid, worktree,
# start epoch). A lock is STALE, and taken over, when its owner pid is dead or it is older than 3 h (GATE_STALE_SECS).
#
# The queue: every caller first takes a TICKET, an atomic `mkdir $GATE_QUEUE_DIR/<13-digit epoch ms>-<pid>-<worktree>` (default
# C:/cb/gate-queue), and only the lowest LIVE ticket may try the lock, so the order is the order of arrival (the old 30 s poll gave the
# lock to whoever polled first, and a waiter could starve for hours). A ticket is DEAD when its pid is dead (the same `kill -0` the
# stale-lock rule uses: every pid here is an MSYS pid of one of these scripts) or when its heartbeat (touched every poll) is older than
# GATE_TICKET_TTL (default 3 h); the next poller removes a dead ticket and says so, so a crashed waiter cannot block the line. A caller
# drops its ticket when it takes the lock, and on exit, interrupt and termination.
LOCK="${GATE_LOCK_DIR:-C:/cb/gate.lock}"
QUEUE="${GATE_QUEUE_DIR:-C:/cb/gate-queue}"
WAIT="${GATE_WAIT_SECS:-30}"
STALE="${GATE_STALE_SECS:-10800}"
TTL="${GATE_TICKET_TTL:-10800}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CMD="${GATE_CMD:-bash \"$HERE/gate.sh\" \"\$@\"}"

# the live tickets, oldest first (one name per line); a dead ticket is removed with a log line
live_tickets() {
  local t pid hb now
  now=$(date +%s)
  for t in $(ls -1 "$QUEUE" 2>/dev/null | sort); do
    pid=$(echo "$t" | cut -d- -f2)
    hb=$(cat "$QUEUE/$t/hb" 2>/dev/null || echo 0)
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "gate-locked: removed a dead ticket $t (pid $pid is dead)" >&2; rm -rf "${QUEUE:?}/$t"
    elif [ $((now - hb)) -gt "$TTL" ]; then
      echo "gate-locked: removed a stale ticket $t (no heartbeat for more than ${TTL}s)" >&2; rm -rf "${QUEUE:?}/$t"
    else
      echo "$t"
    fi
  done
}

if [ "$1" = "--status" ]; then
  if [ -d "$LOCK" ]; then
    echo "holder: pid $(sed -n 1p "$LOCK/owner" 2>/dev/null), $(sed -n 2p "$LOCK/owner" 2>/dev/null), running $(( $(date +%s) - $(sed -n 3p "$LOCK/owner" 2>/dev/null || echo 0) ))s"
  else
    echo "holder: none"
  fi
  mkdir -p "$QUEUE"
  q=$(live_tickets); n=0
  echo "queue: $(echo "$q" | grep -c .) waiting"
  for t in $q; do n=$((n + 1)); echo "  $n. $t"; done
  exit 0
fi

mkdir -p "$(dirname "$LOCK")" "$QUEUE"
# the worktree named on the command line (`--worktree X`) or the current directory, as a short file-name-safe label
WT="$PWD"; prev=""
for a in "$@"; do [ "$prev" = "--worktree" ] && WT="$a"; prev="$a"; done
LABEL=$(basename "$WT" | tr -c 'A-Za-z0-9._\n' '_' | cut -c1-40)
TICKET="$(date +%s%3N)-$$-$LABEL"

release() {
  rm -rf "${QUEUE:?}/${TICKET:?}"
  if [ "$(sed -n 1p "$LOCK/owner" 2>/dev/null)" = "$$" ]; then rm -rf "${LOCK:?}"; fi
}
trap release EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# an unusable queue directory (missing, unwritable, or the ticket cannot be made) means NO QUEUE, never "blocked": the caller then
# races for the lock as the old protocol did
HAVE_TICKET=0
if mkdir "$QUEUE/$TICKET" 2>/dev/null; then HAVE_TICKET=1; date +%s > "$QUEUE/$TICKET/hb"; fi

while :; do
  first="$TICKET"
  if [ "$HAVE_TICKET" = 1 ] && [ -d "$QUEUE/$TICKET" ]; then
    date +%s > "$QUEUE/$TICKET/hb"
    live=$(live_tickets)
    # an unreadable listing is "no queue" too
    first=$(echo "$live" | head -n1); [ -n "$first" ] || first="$TICKET"
  fi
  if [ "$first" != "$TICKET" ]; then
    pos=$(echo "$live" | grep -n -x -F "$TICKET" | cut -d: -f1)
    echo "gate-locked: waiting in the queue, position ${pos:-?} of $(echo "$live" | grep -c .), ahead: $first, checking again in ${WAIT}s"
    sleep "$WAIT"; continue
  fi
  # our turn at the head of the line: take the lock when it is free (or stale)
  if mkdir "$LOCK" 2>/dev/null; then break; fi
  opid=$(sed -n 1p "$LOCK/owner" 2>/dev/null); ostart=$(sed -n 3p "$LOCK/owner" 2>/dev/null); now=$(date +%s)
  if [ -z "$opid" ]; then
    # the owner file is written a moment after mkdir: give a creator a few seconds, then call it stale
    sleep 3; [ -z "$(sed -n 1p "$LOCK/owner" 2>/dev/null)" ] && { echo "gate-locked: stale lock (no owner), taking over"; rm -rf "${LOCK:?}"; }
    continue
  fi
  if ! kill -0 "$opid" 2>/dev/null; then echo "gate-locked: stale lock (owner pid $opid is dead), taking over"; rm -rf "${LOCK:?}"; continue; fi
  if [ -n "$ostart" ] && [ $((now - ostart)) -gt "$STALE" ]; then echo "gate-locked: stale lock (owner $opid older than ${STALE}s), taking over"; rm -rf "${LOCK:?}"; continue; fi
  echo "gate-locked: waiting for the gate lock held by pid $opid ($(sed -n 2p "$LOCK/owner" 2>/dev/null)), next in the queue, checking again in ${WAIT}s"
  sleep "$WAIT"
done
printf '%s\n%s\n%s\n' "$$" "$PWD" "$(date +%s)" > "$LOCK/owner"
rm -rf "${QUEUE:?}/${TICKET:?}"
echo "gate-locked: lock taken by pid $$ at $(date +%H:%M:%S)"
bash -c "$CMD" _ "$@"
rc=$?
echo "gate-locked: released at $(date +%H:%M:%S), exit $rc"
exit $rc
