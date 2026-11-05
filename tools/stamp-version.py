#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Stamp a version onto a change: the one step that touches the files every parallel PR would otherwise collide on.

    python tools/stamp-version.py <version> [--root DIR] [--dry-run]

A PR branch adds only its own files plus a CHANGELOG fragment (`changelog.d/<topic>.md`, format in
`changelog.d/README.md`) and never edits `CHANGELOG.md`, a `Cargo.toml` version, `Cargo.lock` or the cpu-c README
version. When the PM allocates the number, anyone runs this script, which

 1. sets the version in the root `Cargo.toml` path-dependency lines, each crate's `[package]` version, the four
    `unpopped*` entries of `Cargo.lock` and the `=X.Y.Z` pins of `crates/unpopped-cpu-c/README.md`;
 2. turns the fragments into one `## `<version>` - <kind>` entry above the newest CHANGELOG entry;
 3. deletes the fragments it used.

A restamp (the number changed, or the base moved) is the same command with the new number on a branch whose fragments are
still present; if they were already consumed, `git checkout <base> -- CHANGELOG.md` and re-add the fragment from history.
Standard library only. Line endings of each file are kept.
"""
import argparse
import os
import re
import sys

CRATES = ["unpopped", "unpopped-conformance", "unpopped-cpu-c", "unpopped-vocab"]
EM = "—"


def read(path):
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8")
    return raw.replace("\r\n", "\n"), "\r\n" in raw


def write(path, text, crlf):
    with open(path, "wb") as f:
        f.write((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))


def die(msg):
    print(f"stamp-version: {msg}", file=sys.stderr)
    sys.exit(2)


def fragments(directory):
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, n)
        for n in sorted(os.listdir(directory))
        if n.endswith(".md") and n.lower() != "readme.md"
    ]


def sub_exact(text, pattern, repl, what):
    new, n = re.subn(pattern, repl, text)
    if n == 0:
        die(f"{what}: no match for the old version")
    return new


def version_changes(root, old, new):
    q = re.escape(old)
    out = []
    p = os.path.join(root, "Cargo.toml")
    t, crlf = read(p)
    t = sub_exact(
        t,
        r'(?m)^(unpopped[\w-]* = \{ path = "[^"]+", version = ")' + q + '"',
        lambda m: m.group(1) + new + '"',
        "Cargo.toml",
    )
    out.append((p, t, crlf))
    for c in CRATES:
        p = os.path.join(root, "crates", c, "Cargo.toml")
        t, crlf = read(p)
        t = sub_exact(
            t, r'(?m)^version = "' + q + '"', 'version = "' + new + '"', f"crates/{c}/Cargo.toml"
        )
        out.append((p, t, crlf))
    p = os.path.join(root, "Cargo.lock")
    t, crlf = read(p)
    for c in CRATES:
        t = sub_exact(
            t,
            r'(name = "' + re.escape(c) + r'"\nversion = ")' + q + '"',
            lambda m: m.group(1) + new + '"',
            f"Cargo.lock ({c})",
        )
    out.append((p, t, crlf))
    p = os.path.join(root, "crates", "unpopped-cpu-c", "README.md")
    t, crlf = read(p)
    t = sub_exact(t, r"=" + q + r"\b", "=" + new, "crates/unpopped-cpu-c/README.md")
    out.append((p, t, crlf))
    return out


def parse_fragment(path):
    text, _ = read(path)
    head, sep, body = text.partition("\n---\n")
    if not sep or not head.startswith("kind:"):
        die(f"{path}: expected a first line `kind: ...`, a line `---`, then the entry")
    return head[len("kind:"):].strip(), body.strip("\n")


def changelog_change(root, version, frags):
    if not frags:
        die("no changelog.d/*.md fragment: every pushed change needs a CHANGELOG entry")
    kinds, bodies = [], []
    for f in frags:
        k, b = parse_fragment(f)
        kinds.append(k)
        bodies.append(b)
    kind = next((k for k in kinds if "breaking" in k.lower()), kinds[0])
    entry = f"## `{version}` {EM} {kind}\n\n" + "\n\n".join(bodies) + "\n\n---\n\n"
    p = os.path.join(root, "CHANGELOG.md")
    t, crlf = read(p)
    m = re.search(r"(?m)^## `\d+\.\d+\.\d+`", t)
    if not m:
        die("CHANGELOG.md has no version heading to put the entry above")
    return (p, t[: m.start()] + entry + t[m.start():], crlf)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version")
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not re.fullmatch(r"\d+\.\d+\.\d+", a.version):
        die(f"`{a.version}` is not MAJOR.MINOR.PATCH")
    root = os.path.abspath(a.root)
    cargo, _ = read(os.path.join(root, "crates", "unpopped", "Cargo.toml"))
    m = re.search(r'(?m)^version = "(\d+\.\d+\.\d+)"', cargo)
    if not m:
        die("crates/unpopped/Cargo.toml has no version line")
    old = m.group(1)
    if old == a.version:
        die(f"the crates already say {old}")
    frags = fragments(os.path.join(root, "changelog.d"))
    changes = version_changes(root, old, a.version)
    changes.append(changelog_change(root, a.version, frags))
    for path, _, _ in changes:
        print(("would change " if a.dry_run else "changed ") + os.path.relpath(path, root).replace("\\", "/"))
    if a.dry_run:
        print(f"{old} -> {a.version}; {len(frags)} fragment(s)")
        return
    for path, text, crlf in changes:
        write(path, text, crlf)
    for f in frags:
        os.remove(f)
    print(f"stamped {old} -> {a.version}; removed {len(frags)} fragment(s)")


if __name__ == "__main__":
    main()
