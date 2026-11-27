kind: `CollectiveKind::MatchAny` (PTX `match.any.sync`)
---
Added `CollectiveKind::MatchAny`: each lane gets a `U32` mask of the subgroup lanes whose value has the same bits as its own (a float compares by bit pattern). Subgroup only, at most 32 lanes (a Block is refused by name, a larger scope is a typed oracle trap). Source: PTX ISA `match.any.sync`; CUDA `__match_any_sync`. `CollectiveKind` is `#[non_exhaustive]`, so additive. Pinned by `match_any_gives_each_lane_the_mask_of_lanes_with_the_same_bits` and the cpu-c differential `match_any_agrees_on_integers_and_on_float_bit_patterns`.
