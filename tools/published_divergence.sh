#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Published-vs-workspace divergence gate (issue #4, the CLASS).
#
# For every publishable crate in crates/, if the version in its Cargo.toml is ALREADY on the
# registry, the `src/` served by the registry must equal the workspace's. A crate whose version is
# not published yet is reported as unreleased and passes: a version that has not shipped cannot have
# diverged from itself. What this catches is the other case: a commit made to version V after V was
# published, which no in-tree build can see (workspace crates resolve by path) and no consumer can
# see (they resolve the registry only).
#
# Honest about what it is NOT: it compares `src/` (the issue's measurement), not tests, examples or
# Cargo.toml, and a registry or network failure is a FAILURE, never a pass.
#
#   tools/published_divergence.sh            run the gate over crates/*
#   tools/published_divergence.sh --selftest control only: shows the comparison can fail
#
# The decision rests on the sparse index, not on the HTTP status of the download: static.crates.io
# answers 403 for a file that does not exist, which is indistinguishable by status alone from a
# real access problem.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
ua="unpopped-divergence-gate (github.com/EvansLaboratories/Unpopped)"

# diff the two src trees; stdout is the number of differing paths. CRLF-safe: a naive `diff -rq`
# reports every file when one side is a CRLF checkout and the other the LF tarball.
count_diff() {
  local out
  out=$(diff -rq --strip-trailing-cr "$1" "$2" || true)
  if [ -z "$out" ]; then echo 0; else printf '%s\n' "$out" | wc -l | tr -d ' '; fi
}

index_path() {
  local c=$1 n=${#1}
  if [ "$n" -le 2 ]; then echo "$n/$c"
  elif [ "$n" -eq 3 ]; then echo "3/${c:0:1}/$c"
  else echo "${c:0:2}/${c:2:2}/$c"; fi
}

# 0 = published, 1 = not published; anything else aborts (set -e).
published() {
  local c=$1 v=$2 f="$work/index-$1" code
  code=$(curl -sS -L -A "$ua" -o "$f" -w '%{http_code}' "https://index.crates.io/$(index_path "$c")")
  case "$code" in
    200) grep -q "\"vers\":\"$v\"" "$f" ;;
    404) return 1 ;;
    *) echo "FAIL: index lookup for $c answered HTTP $code" >&2; exit 2 ;;
  esac
}

fetch() { # crate version -> extracted dir on stdout
  local c=$1 v=$2 code
  code=$(curl -sS -L -A "$ua" -o "$work/$c.crate" -w '%{http_code}' \
    "https://static.crates.io/crates/$c/$c-$v.crate")
  [ "$code" = 200 ] || { echo "FAIL: $c-$v is in the index but download answered HTTP $code" >&2; exit 2; }
  tar xzf "$work/$c.crate" -C "$work"
  echo "$work/$c-$v"
}

control() { # one extracted crate: injecting a line must move the count 0 -> 1
  local d=$1 copy="$work/control-src"
  rm -rf "$copy"; cp -r "$d/src" "$copy"
  local before after f
  before=$(count_diff "$d/src" "$copy")
  f=$(find "$copy" -name '*.rs' | sort | head -1)
  printf '\n// injected by the divergence gate control\n' >> "$f"
  after=$(count_diff "$d/src" "$copy")
  if [ "$before" != 0 ] || [ "$after" != 1 ]; then
    echo "FAIL: control did not discriminate (before=$before after=$after, want 0 then 1)" >&2
    exit 2
  fi
  echo "control: identical copy differs in $before file(s); one injected line differs in $after"
}

if [ "${1:-}" = "--selftest" ]; then
  d="$work/self/c-0.0.0"; mkdir -p "$d/src"; printf 'fn a() {}\n' > "$d/src/lib.rs"
  control "$d"; exit 0
fi

# The comparison must be able to fail before any answer from it is believed, even when every
# crate is unreleased and nothing real gets compared.
self_dir="$work/self/c-0.0.0"; mkdir -p "$self_dir/src"; printf 'fn a() {}
' > "$self_dir/src/lib.rs"
control "$self_dir"

checked=0; unreleased=0; bad=0; controlled=0
for manifest in "$root"/crates/*/Cargo.toml; do
  dir=$(dirname "$manifest")
  grep -q '^publish *= *false' "$manifest" && { echo "skip   $(basename "$dir") (publish = false)"; continue; }
  name=$(grep -m1 '^name *=' "$manifest" | sed 's/.*"\(.*\)".*/\1/')
  ver=$(grep -m1 '^version *=' "$manifest" | sed 's/.*"\(.*\)".*/\1/')
  if [ -z "$name" ] || [ -z "$ver" ]; then echo "FAIL: cannot read name/version from $manifest" >&2; exit 2; fi
  if published "$name" "$ver"; then
    d=$(fetch "$name" "$ver")
    n=$(count_diff "$d/src" "$dir/src")
    checked=$((checked + 1))
    if [ "$controlled" = 0 ]; then control "$d"; controlled=1; fi
    if [ "$n" = 0 ]; then echo "same   $name $ver"
    else echo "DIFFER $name $ver: $n path(s) in src/ differ from the registry"; diff -rq --strip-trailing-cr "$d/src" "$dir/src" || true; bad=$((bad + 1)); fi
  else
    unreleased=$((unreleased + 1)); echo "unreleased $name $ver (not on the registry yet)"
  fi
done

echo "population: $checked compared, $unreleased unreleased, $bad differing"
if [ "$bad" != 0 ]; then
  echo "A published version's source changed without a version bump. Bump the version." >&2
  exit 1
fi
