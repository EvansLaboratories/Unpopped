kind: CI: move the portfolio workflow pin to CodeRipper 0.47.5
---
The `portfolio.yml` call in `.github/workflows/verify.yml` moves from the 0.43.49 SHA to `c72a6e03479d914de8ea8a73609ba73e93a0de92` (0.47.5). `migrate-rule-ids.py --dry-run` reported nothing to migrate: this repo names no old rule ids. No code change.
