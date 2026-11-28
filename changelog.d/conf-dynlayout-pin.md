kind: test-only: the CuTe 3/66 dynamic-mode pin flipped to DynLayout (after #536)
---
`kernel_c_recipes_tiled`: the "dynamic mode declined by name" test became `dyn_layout_binds_runtime_modes_to_a_static_layout`: `(m,n):(lda,1)` parses as a `DynLayout`, binds to `(5,6):(8,1)` and to a different layout at another launch, an unbound parameter is refused, and `Layout` itself stays static. `cargo test -p unpopped --test kernel_c_recipes_tiled` 9 passed; clippy clean.
