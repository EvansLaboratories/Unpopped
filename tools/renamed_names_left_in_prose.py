#!/usr/bin/env python3
"""Find identifiers a commit range DELETED from code but left named in comments.

A rename that compiles is not a rename that is done. The compiler resolves every
*call*; nothing resolves a name in prose. `cargo build`, `cargo clippy` and
`cargo doc` were all clean on `96c46eb` with **31** references to six
just-renamed gate functions sitting in doc comments across five files, including
two crates that document which gate they defer to.

    python tools/renamed_names_left_in_prose.py <base-ref> <head-ref>
    python tools/renamed_names_left_in_prose.py HEAD~1 HEAD

Exit 1 if any deleted identifier is still named in a comment.

WHY THIS SHAPE AND NOT A LINT
-----------------------------

The obvious version — "every backticked name in a comment must resolve" — is not
viable here and the measurement says so: 282 bare names in this workspace's
comments have no in-tree referent, and the large majority are legitimate. They
name things in Fuel (`dispatch_record`, `KernelRef`), in Baracuda (`emit_scalar`),
in C and MSVC (`uint32_t`, `_FCbuild`, `_Fcomplex`), in the KISS specs
(`target_capability`, `OpAttrs`), and in deliberate historical narration ("it
used to also assert ..."). A gate demanding a 282-entry allowlist would be
abandoned within a week, and rightly.

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

    96c46eb~1..96c46eb   7 definitions gone, 4 still named, 31 mentions   <- the defect
    96c46eb~1..c57e8b8   7 definitions gone, 0 still named                <- after the sweep

NOTHING RUNS THIS AUTOMATICALLY
-------------------------------

There is no CI in this repository. It is a check to run when you rename or
delete something, not a gate that runs itself, and saying so is the point: a
guard described as automatic when it is manual is the failure this repo spent a
day cataloguing.
"""

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


def git(*args):
    """Bytes, decoded permissively: the tree carries em-dashes and box drawing."""
    out = subprocess.run(["git", *args], capture_output=True)
    return out.stdout.decode("utf-8", "replace")


def rust_files(ref):
    return [p for p in git("ls-tree", "-r", "--name-only", ref).split("\n") if p.endswith(".rs")]


def strip_line_comments(src):
    return "\n".join(l.split("//")[0] for l in src.split("\n"))


def defined_at(ref):
    names = set()
    for path in rust_files(ref):
        code = strip_line_comments(git("show", f"{ref}:{path}"))
        for pat in DEF_PATTERNS:
            names.update(re.findall(pat, code))
    return names


def comment_mentions(ref, names):
    """Where each name still appears inside a `//` comment at `ref`."""
    hits = {n: [] for n in names}
    if not names:
        return hits
    for path in rust_files(ref):
        for lineno, line in enumerate(git("show", f"{ref}:{path}").split("\n"), 1):
            stripped = line.strip()
            if not stripped.startswith("//"):
                continue
            for name in names:
                if re.search(r"\b" + re.escape(name) + r"\b", stripped):
                    hits[name].append(f"{path}:{lineno}")
    return hits


def main(base, head):
    gone = sorted(defined_at(base) - defined_at(head))
    print(f"identifiers whose definition disappeared in {base}..{head}: {len(gone)}")

    hits = comment_mentions(head, gone)
    total = 0
    for name in gone:
        where = hits[name]
        if not where:
            continue
        total += len(where)
        print(f"  {name:<42} still named in {len(where)} comment(s)")
        for w in where[:5]:
            print(f"      {w}")
        if len(where) > 5:
            print(f"      ... and {len(where) - 5} more")

    if total:
        print(f"\nFAIL: {total} stale comment mention(s).")
        print("A rename that compiles is not a rename that is done.")
        return 1
    print("OK: no deleted identifier is still named in a comment.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
