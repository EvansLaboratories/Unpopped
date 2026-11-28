kind: tests, residue tiling refusal by name and composition_within bounds
---
The residue-tiling finding is split: `zipped_divide` still accepts a non-dividing tile (pinned), while `local_tile`, `inner_partition`, `outer_partition` and `local_partition` refuse it with `NotDivisible`, asserted by an exhaustive e1..24 x t1..8 sweep. `composition_within` is pinned: it accepts exactly the 972 in-extent cases and refuses the other 6084 with `ExceedsExtent`. Verified by two seeded mutants (threshold off by one, divisibility check removed), both caught by value.
