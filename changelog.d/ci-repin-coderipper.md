kind: ci, the CodeRipper portfolio workflow repinned to v0.48.14 (af3d6876)
---
The `portfolio` job in `.github/workflows/verify.yml` moves from coderipper `467fd259` (installs CodeRipper 0.48.1) to `af3d6876` (tag `v0.48.14`, installs CodeRipper 0.48.12), the newest published tool. No rule is silenced and no code changes; if the newer tool reports a finding the portfolio check was blind to, it is answered on the PR, never configured away.
