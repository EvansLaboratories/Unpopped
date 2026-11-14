kind: Breaking: the `convert` feature moves into a new crate, `unpopped-convert` (ROADMAP U31)
---
`unpopped`'s optional `convert` feature (the tree-sitter walkers `Frontend`, `lift`, `lift_elementwise`, `lift_reduction`, `lift_scan` and the `*_with` forms that take a `ResourceMap`) is now the crate `unpopped-convert`, which carries Unpopped's version number. `unpopped` no longer depends on `tree-sitter` and has no `convert` feature or `unpopped::convert` module, so the stable core no longer names its parser technology (CodeRipper `ARC-003`, `ARC-004`).

Migrating: replace `unpopped = { version = "0.46", features = ["convert"] }` with an additional `unpopped-convert = "0.46"` dependency, and `unpopped::convert::` with `unpopped_convert::`. The two crates.io dependents using the feature are `baracuda-cuda-parse` and `fishstringer-slang-parse`; both were notified. Added to the core for the split: `Lifted::inferred` (`Lifted` is `#[non_exhaustive]`), and `lift::unary_fn`, `lift::binary_fn` and `ResourceMap::{output_name, input_names, param_names}` became public.

Verified: the 37 tests of the moved files pass in the new crate (12 unit, 4 + 12 + 9 integration), the workspace builds, `cargo doc` and `cargo clippy -D warnings` are clean, and `coderipper check ARC-003` and `ARC-004` pass on the branch where main reports the two findings each.
