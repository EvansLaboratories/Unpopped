kind: fixed
---
`Layout::{inner_partition, outer_partition, local_tile, local_partition}` return `LayoutError::NotDivisible` when a tiler's size does not divide its mode, instead of rounding the number of tiles up (the last tile reached past the layout: 4 into 6 reached offset 49 of 36). No new variant (`NotDivisible` exists); a ragged edge is a mask over the identity tensor. No caller outside `unpopped-vocab` existed (grep at origin/main).
