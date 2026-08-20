#!/usr/bin/env python3
"""Find identifiers a commit range DELETED from code but left named in prose.

A rename that compiles is not a rename that is done. The compiler resolves every
*call*; nothing resolves a name in prose. `cargo build`, `cargo clippy` and
`cargo doc` were all clean on `96c46eb` with **31** references to six
just-renamed gate functions sitting in doc comments across five files, including
two crates that document which gate they defer to.

    python tools/renamed_names_left_in_prose.py <base-ref> <head-ref>
    python tools/renamed_names_left_in_prose.py HEAD~1 HEAD

Scans `//` comments in `.rs` and every line of `.md`. Exit 1 when a candidate is
undispositioned, or when a disposition no longer matches anything.

A HIT IS A CANDIDATE, NOT A DEFECT
----------------------------------

Fuel's taxonomy, and the reason this tool cannot decide for you:

    STALE       prose means the current thing, names the old one
                    -> rename it
    HISTORICAL  prose is ABOUT the rename, or about prior behaviour
                    -> renaming DESTROYS the record
    PINNED      the document describes the code AS OF some revision
                    -> renaming makes it FALSE

**HISTORICAL is the expected case in any repo that documents its own renames**,
and it is not a rare tail. Fuel ran the manual version of this check across a
296-file `Lazy`-prefix drop: twelve prose mentions of names that no longer exist,
**twelve of twelve correct to leave** — including `docs/method-rules.md`
narrating the very rename defect that motivated this tool. A naive exit 1 there
is 12-for-12 wrong-in-effect, and the first person who meets it either sweeps
twelve historical records or stops trusting the check. Both are worse than not
running it.

So candidates are dispositioned once, in `tools/renamed_names_dispositioned.tsv`,
and the tool goes quiet about them. `stale` is not a category there: a stale
mention gets fixed, never recorded.

The ledger is itself ratcheted. A disposition whose name no longer appears
anywhere ALSO fails — otherwise the file accumulates entries that suppress
nothing, and a suppression that has stopped suppressing is the failure this
repository spent a day cataloguing.

WHY THIS SHAPE AND NOT A LINT
-----------------------------

The obvious version — "every backticked name in a comment must resolve" — is not
viable here and the measurement says so: 282 bare names in this workspace's
comments have no in-tree referent, and the large majority are legitimate. They
name things in Fuel (`dispatch_record`, `KernelRef`), in Baracuda
(`emit_scalar`), in C and MSVC (`uint32_t`, `_FCbuild`, `_Fcomplex`), in the KISS
specs (`target_capability`, `OpAttrs`), and in deliberate historical narration. A
gate demanding a 282-entry allowlist would be abandoned within a week, and
rightly. The naive version is the one you reach by reasoning, and it dies on
contact with the count.

The rustdoc gate this repo added the same day (`[workspace.lints.rustdoc]`)
closes the *linked* half: `[`Foo`]` fails when `Foo` dies. It is structurally
blind to the unlinked half — `` `Foo` `` has no referent to check — and that is
the larger half, because most prose names things without linking them.

So this keys on the one signal that is both cheap and specific: **the identifier
existed at `base` and does not exist at `head`.** That is exactly the population
a rename or deletion creates, it needs no allowlist, and it produced zero false
positives on the commit that motivated it.

VALIDATION
----------

    96c46eb~1..96c46eb   7 definitions gone, 4 still named, 31 mentions   -> exit 1
    96c46eb~1..c57e8b8   7 definitions gone, 0 still named                -> exit 0

NOTHING RUNS THIS AUTOMATICALLY
-------------------------------

There is no CI in this repository. It is a check to run when you rename or
delete something, not a gate that runs itself, and saying so is the point: a
guard described as automatic when it is manual is the failure this repo spent a
day cataloguing.
"""

import os
import re
import subprocess
import sys

# Deliberately conservative: only item kinds whose disappearance is unambiguous.
# Fields and variants are matched too loosely by any regex worth writing, and a
# false positive here costs more than a miss — this check has to stay believable
# enough to run.
DEF_PATTERNS = [
    r"\bfn\s+([a-z_][a-z0-9_]*)",
    r"\b(?:struct|enum|trait|union|type)\s+([A-Za-z_][A-Za-z0-9_]*)",
    r"\bconst\s+([A-Z_][A-Z0-9_]*)",
    r"\bstatic\s+([A-Z_][A-Z0-9_]*)",
    r"\bmacro_rules!\s+([a-z_][a-z0-9_]*)",
]

# Advisory only. Sorts the pile so a human triages the likely-HISTORICAL ones
# last; it never suppresses, because "used to" also appears in prose that then
# goes on to describe the current thing by its old name.
HISTORICAL_HINTS = re.compile(
    r"\b(used to|use[dn]? to be|was |were |formerly|previously|renamed|"
    r"before |until |no longer|replaced|old name|it used)\b",
    re.I,
)

LEDGER = "tools/renamed_names_dispositioned.tsv"
CATEGORIES = {"historical", "pinned"}


def git(*args):
    """Bytes, decoded permissively: the tree carries em-dashes and box drawing."""
    return subprocess.run(["git", *args], capture_output=True).stdout.decode("utf-8", "replace")


def files_at(ref, suffix):
    return [p for p in git("ls-tree", "-r", "--name-only", ref).split("\n") if p.endswith(suffix)]


def strip_line_comments(src):
    return "\n".join(l.split("//")[0] for l in src.split("\n"))


def defined_at(ref):
    names = set()
    for path in files_at(ref, ".rs"):
        code = strip_line_comments(git("show", f"{ref}:{path}"))
        for pat in DEF_PATTERNS:
            names.update(re.findall(pat, code))
    return names


def prose_mentions(ref, names):
    """Where each name still appears in prose at `ref`: `//` in Rust, all of Markdown."""
    hits = {n: [] for n in names}
    if not names:
        return hits
    sources = [(p, True) for p in files_at(ref, ".rs")] + [(p, False) for p in files_at(ref, ".md")]
    for path, rust in sources:
        for lineno, line in enumerate(git("show", f"{ref}:{path}").split("\n"), 1):
            text = line.strip()
            if rust and not text.startswith("//"):
                continue
            for name in names:
                if re.search(r"\b" + re.escape(name) + r"\b", text):
                    hits[name].append((f"{path}:{lineno}", bool(HISTORICAL_HINTS.search(text))))
    return hits


def read_ledger():
    """name -> (category, note). Absent file is fine: nothing dispositioned yet."""
    out = {}
    if not os.path.exists(LEDGER):
        return out
    with open(LEDGER, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                print(f"{LEDGER}: malformed row (want name<TAB>category<TAB>note): {line!r}")
                continue
            name, category, note = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if category not in CATEGORIES:
                print(f"{LEDGER}: unknown category {category!r} for {name} (want one of {sorted(CATEGORIES)})")
                continue
            out[name] = (category, note)
    return out


def main(base, head):
    gone = sorted(defined_at(base) - defined_at(head))
    print(f"identifiers whose definition disappeared in {base}..{head}: {len(gone)}")

    hits = prose_mentions(head, gone)
    ledger = read_ledger()
    named = [n for n in gone if hits[n]]

    undispositioned = [n for n in named if n not in ledger]
    for name in undispositioned:
        where = hits[name]
        hinted = sum(1 for _, h in where if h)
        hint = f"  [{hinted}/{len(where)} read as HISTORICAL]" if hinted else ""
        print(f"  {name:<42} {len(where)} prose mention(s){hint}")
        for w, h in where[:5]:
            print(f"      {w}{'   <- historical?' if h else ''}")
        if len(where) > 5:
            print(f"      ... and {len(where) - 5} more")

    quiet = [n for n in named if n in ledger]
    if quiet:
        print(f"\ndispositioned, not shown: {', '.join(f'{n} ({ledger[n][0]})' for n in quiet)}")

    dead = [n for n in ledger if not hits.get(n)]
    for name in dead:
        print(f"\n{LEDGER}: {name} is dispositioned but named nowhere — remove the row.")

    if undispositioned or dead:
        if undispositioned:
            print(f"\nEXIT 1: {len(undispositioned)} name(s) to disposition.")
            print("Each is STALE (rename it), HISTORICAL (leave it, record it),")
            print(f"or PINNED (leave it, record it). Record the last two in {LEDGER}.")
        if dead:
            print(f"\nEXIT 1: {len(dead)} disposition(s) suppressing nothing.")
        return 1

    print("OK: every deleted identifier is either absent from prose or dispositioned.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
