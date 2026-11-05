kind: CHANGELOG fragments and `tools/stamp-version.py` (the version-stamp pilot; tools and docs only)
---
`changelog.d/<topic>.md` fragments and `python tools/stamp-version.py <version>`, ported from coderipper's pilot to
Unpopped's layout: a PR adds a fragment and never edits `CHANGELOG.md`, a `Cargo.toml` version, `Cargo.lock` or the cpu-c
README pins; the one command sets the version in the root path-dependency lines, the four crates, the four `unpopped*`
lockfile entries and the README pins, writes one `## `<version>` — <kind>` heading from the fragments above the newest entry,
and deletes them. Self-tested on a throwaway fragment (`--dry-run`, then a real run, `cargo check --workspace --locked`
green, CRLF kept, a repeat of the same version refused). This PR carries only its fragment and no number: the gate stamps it.
