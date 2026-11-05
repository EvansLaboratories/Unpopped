# changelog.d: CHANGELOG fragments

A PR branch never edits `CHANGELOG.md`, a `Cargo.toml` version, `Cargo.lock` or the version pins in
`crates/unpopped-cpu-c/README.md`: those are what parallel PRs collide on. Write one fragment per change instead,
`changelog.d/<topic>.md`:

    kind: **BREAKING**: `Stmt::Foo`, a one-line description
    ---
    The entry body in the CHANGELOG's own style (no heading): what changed, why, how it was verified.

- The first line `kind: ...` is the text after the version in the heading (`## `0.44.2` — <kind>`). When several fragments are
  stamped together, one whose kind contains "breaking" wins.
- A line `---`, then the body.
- `python tools/stamp-version.py <version>` (add `--dry-run` to preview) sets the version in the root `Cargo.toml`
  path-dependency lines, the four crates, the four `unpopped*` entries of `Cargo.lock` and the cpu-c README pins, writes one
  heading above the newest entry from all fragments, and deletes them. The PM allocates the number; the PR may stamp it
  itself (one command) or leave the fragment for the gate.
- Fragments are the only per-PR CHANGELOG state, so two PRs never touch the same line. If a base moved and the number
  changed, re-run the command with the new number on the branch that still has its fragment.
