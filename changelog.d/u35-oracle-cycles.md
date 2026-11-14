kind: the module cycles inside `unpopped::oracle` are dissolved, no behaviour or path change (ROADMAP U35)
---
CodeRipper `MOD-002` reported two Info cycles inside `oracle`: `kernel_call` and `kernel_resume`, and `kernel_eval`, `kernel_mem` and `kernel_sync`. The edges ran back through four things, now moved to where they point: `KernelTrap` is the leaf module `kernel_trap` (still `oracle::kernel_eval::KernelTrap` by re-export), `Stamp` lives in `kernel_sync`, the `impl Machine` block of `kernel_mem` is the new module `kernel_mem_ops` (so `kernel_mem` holds the race rules and the memory types and no longer names the machine), and `extern_reference` moved into `kernel_resume`. Text moved verbatim; items that crossed a module boundary became `pub(super)`.

Verified: `coderipper check MOD-002 --profile extended` reports no cycle (it reported both on main); `MOD-005` lists no new file; the workspace tests are unchanged.
