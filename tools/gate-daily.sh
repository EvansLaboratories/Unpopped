#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# The daily FULL gate on origin/main, for a quiet laptop (schedule it; see tools/README of the lane). Red means a hotfix PR first,
# before any new leaf merges. usage: tools/gate-daily.sh [scratch-worktree-dir]
set -o pipefail
D="${1:-C:/Projects/Unpopped-wt-daily}"
git fetch origin main >/dev/null 2>&1
if [ ! -d "$D" ]; then git worktree add --detach "$D" origin/main >/dev/null 2>&1; else git -C "$D" checkout -q --detach origin/main; fi
cd "$D" || exit 2
bash "$(dirname "$0")/gate-locked.sh" --worktree "$D" --mode full 2>&1 | tee "$D/gate-daily.out"
