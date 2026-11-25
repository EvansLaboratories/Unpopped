kind: Chore: every `allow` carries a reason (GOV-010: 52 `allow(dead_code)`, 2 `clippy::type_complexity`)
---
No behaviour change. Each of the 54 `#[allow(..)]` attributes that CodeRipper's GOV-010 flagged now carries a `reason = "..."` saying why the allow is there (test support modules whose items are used by some test binaries and not others, and a return type the type-complexity lint cannot see through). `coderipper check GOV-010` reported 54 findings on `main` and none after this change (see the PR body for the run). Tests, clippy and fmt unchanged.
