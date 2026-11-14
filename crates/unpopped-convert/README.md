# unpopped-convert

The shared tree-sitter walkers that lift C-family kernel source into [Unpopped](https://github.com/EvansLaboratories/Unpopped)'s neutral IR.

A [`Frontend`] value carries what a source language contributes (kernel marker, residue list, buffer naming, a `parse` function); `lift_elementwise`, `lift_reduction`, `lift_scan` and `lift` do the one CST walk for every language. Whatever does not map stays a typed `LiftError` residue.

No grammar is bundled here. CUDA's parser is `baracuda-cuda-parse`; Slang's is `vulkane-slang-parse`. Both depend on this crate for `Frontend` and the walkers.

This code lived in `unpopped` behind the `convert` feature until 0.46.0. It moved so the stable core no longer names `tree_sitter`. To migrate: depend on `unpopped-convert` (same version number as `unpopped`) instead of `unpopped` with `features = ["convert"]`, and change `unpopped::convert::` to `unpopped_convert::`.

Licensed under MIT OR Apache-2.0.
