#!/usr/bin/env python3
"""Find identifiers a commit range DELETED from code but left named in prose.

A rename that compiles is not a rename that is done. The compiler resolves every
*call*; nothing resolves a name in prose. `cargo build`, `cargo clippy` and
`cargo doc` were all clean on `96c46eb` with **31** references to six
just-renamed gate functions sitting in doc comments across five files, including
two crates that document which gate they defer to.

    python tools/renamed_names_left_in_prose.py <base-ref> <head-ref>
    python tools/renamed_names_left_in_prose.py HEAD~1 HEAD

Exit 1 when a mention is undispositioned, or when a disposition matches nothing.

WHAT IT SCANS, stated because coverage decided by whatever file type the author
happened to think of is a population claim nobody made:

    *.rs    `//` comment lines only
    *.md    every line

Nothing else. Not `*.toml`, not `*.tsv`, not `*.json`, not commit messages —
each of which can name an identifier. The Markdown gap was found by a peer, not
by me, which is the argument for naming the set rather than leaving it implicit.

A NAME MUST BE SPELLED AS CODE
------------------------------

The name has to appear inside backticks, optionally behind a path
(`` `plan::validate_row_reduce` ``). The cost of dropping that requirement was
measured, not guessed: across this repo's full history the bare-word form
reports `header` 31 times and `main` 14 times, because an item by each name once
existed and both words occur in ordinary English. **Prose that refers to an
identifier spells it as one; prose that uses a word uses a word.**

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
and it is not a rare tail. Fuel ran the manual form across a 296-file
`Lazy`-prefix drop: twelve prose mentions of names that no longer exist, **twelve
of twelve correct to leave** — ten of them the write-up of a sweep corrupting a
verification control inside a fenced code block. Had the tool ever swept those,
it would have erased the record of why it must not sweep them.

DISPOSITIONS ARE KEYED ON CONTENT, NOT ON LOCATION
--------------------------------------------------

A row in `tools/renamed_names_dispositioned.tsv` is `(name, hash-of-the-line)`.
Not `(name, file)`. Fuel's argument, and this repo supplied the proof:

`kiss_byte_match.rs` named `arch_from_code` in the **present tense** as a live
precondition — STALE — and a rewording turned it into a record of what the
precondition used to point at — HISTORICAL. Same name, same file, changed only
by someone editing the sentence. **The transition is real and it runs both
ways**, so a row keyed on `(name, file)` would permanently silence every future
mention of that name in that file, including one a later edit reintroduces in
the present tense. That is a 282-entry allowlist arriving one row at a time —
the design rejected below, entering through the door marked "disposition".

Content-keying gives the property actually wanted: **a disposition expires when
its subject changes, and only when its subject changes.** An edited line loses
its row and comes back for fresh judgement; an untouched historical line stays
quiet forever.

It is also what makes `stale` deliberately absent from the categories: a stale
mention gets FIXED, never recorded — and **a fixed line is a changed line, so it
cannot silently inherit a row that no longer describes it.**

The hash covers the line's text only, not the file or line number: a sentence
that moves is the same sentence, and expiring rows on unrelated edits elsewhere
would be churn without judgement.

The ledger is ratcheted both ways. A row matching no current line ALSO fails —
a suppression that has stopped suppressing reads like coverage while providing
none.

WHY THIS SHAPE AND NOT A LINT
-----------------------------

"Every backticked name in a comment must resolve" is not viable here and the
measurement says so: 282 bare names in this workspace's comments have no in-tree
referent, and the large majority are legitimate. They name things in Fuel
(`dispatch_record`, `KernelRef`), in Baracuda (`emit_scalar`), in C and MSVC
(`uint32_t`, `_FCbuild`, `_Fcomplex`), in the KISS specs (`target_capability`,
`OpAttrs`), and in deliberate historical narration. A gate demanding a 282-entry
allowlist would be abandoned within a week, and rightly. The naive version is
the one you reach by reasoning, and it dies on contact with the count.

The rustdoc gate this repo added the same day (`[workspace.lints.rustdoc]`)
closes the *linked* half: `[`Foo`]` fails when `Foo` dies. It is structurally
blind to the unlinked half — `` `Foo` `` has no referent to check — and that is
the larger half, because most prose names things without linking them.

So this keys on the one signal that is both cheap and specific: **the identifier
existed at `base` and does not exist at `head`.**

IT READS COMMITTED CONTENT, NOT THE WORKING TREE
------------------------------------------------

Both refs are read through `git show`, so **an uncommitted edit is invisible to
it**. Run it after committing the rename, not while making it — a clean result
on a dirty tree is a result about a tree you are not looking at. Found by trying
to mutation-test it against a working-tree edit and getting silence.

ARMED, NOT AUTOMATIC — AND SLOW
-------------------------------

There is no CI in this repository. This runs when a person runs it.

It shells out to `git show` for every `.rs` and `.md` file at both refs, so a
296-file rename takes minutes. **Those two facts combine badly and the
combination is the point:** a check that takes minutes and runs only when
remembered is the one skipped exactly when a rename is large — which is when it
matters. Budget for it in the rename, not after.

VALIDATION
----------

    96c46eb~1..96c46eb            4 undispositioned, 31 mentions   -> exit 1
    96c46eb~1..HEAD               clean                            -> exit 0
    unpopped-vocab-v0.2.0..HEAD   3 dispositioned, shown as such    -> exit 0
    a ledger row for a line that no longer exists                  -> exit 1
"""

import hashlib
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


def line_hash(text):
    """Identity of a mention: its own words, nothing else. See the module doc."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()[:12]


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
    """[(name, hash, where, looks_historical)] for every code-spelled mention."""
    if not names:
        return []
    pats = {n: re.compile(r"`[A-Za-z0-9_:]*\b" + re.escape(n) + r"\b[^`]*`") for n in names}
    found = []
    sources = [(p, True) for p in files_at(ref, ".rs")] + [(p, False) for p in files_at(ref, ".md")]
    for path, rust in sources:
        for lineno, line in enumerate(git("show", f"{ref}:{path}").split("\n"), 1):
            text = line.strip()
            if rust and not text.startswith("//"):
                continue
            for name in names:
                if pats[name].search(text):
                    found.append(
                        (name, line_hash(text), f"{path}:{lineno}", bool(HISTORICAL_HINTS.search(text)))
                    )
    return found


def read_ledger():
    """(name, hash) -> (category, note). Absent file is fine: nothing dispositioned."""
    out = {}
    if not os.path.exists(LEDGER):
        return out
    with open(LEDGER, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                print(f"{LEDGER}: malformed row (want name<TAB>hash<TAB>category<TAB>note): {line!r}")
                continue
            name, h, category, note = (p.strip() for p in parts[:4])
            if category not in CATEGORIES:
                print(f"{LEDGER}: unknown category {category!r} for {name} — want one of {sorted(CATEGORIES)}")
                continue
            out[(name, h)] = (category, note)
    return out


def main(base, head):
    gone = sorted(defined_at(base) - defined_at(head))
    print(f"identifiers whose definition disappeared in {base}..{head}: {len(gone)}")

    mentions = prose_mentions(head, gone)
    ledger = read_ledger()

    undispositioned = [m for m in mentions if (m[0], m[1]) not in ledger]
    quiet = [m for m in mentions if (m[0], m[1]) in ledger]

    for name, h, where, hinted in undispositioned:
        print(f"  {name}  at {where}{'   <- reads as HISTORICAL' if hinted else ''}")
        print(f"      to disposition:  {name}\t{h}\t<historical|pinned>\t<why>")

    if quiet:
        names = sorted({f"{n} ({ledger[(n, h)][0]})" for n, h, _, _ in quiet})
        print(f"\ndispositioned, not shown: {', '.join(names)}")

    live = {(n, h) for n, h, _, _ in prose_mentions(head, sorted({k[0] for k in ledger}))}
    dead = [k for k in ledger if k not in live]
    for name, h in dead:
        print(f"\n{LEDGER}: {name} {h} matches no current line — the sentence changed. Remove the row.")

    if undispositioned or dead:
        if undispositioned:
            print(f"\nEXIT 1: {len(undispositioned)} mention(s) to disposition.")
            print("Each is STALE (rename it), HISTORICAL (leave it, record it),")
            print(f"or PINNED (leave it, record it). Paste the last two into {LEDGER}.")
        if dead:
            print(f"\nEXIT 1: {len(dead)} row(s) matching nothing.")
        return 1

    print("OK: every deleted identifier is either absent from prose or dispositioned.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
