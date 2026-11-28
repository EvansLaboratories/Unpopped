kind: docs, the oracle limit on forward progress between blocks, ROADMAP U76
---
`oracle/kernel_launch.rs` now documents that blocks run sequentially in index order, so a spin on a later block's write is `KernelTrap::CrossBlockWait` (acquire-class spin) or `StepLimit` (a `Relaxed` spin) where hardware completes, and what a kernel that needs inter-block progress can use today. ROADMAP U76 schedules the resident-block interleaving scheduler. No code change.
