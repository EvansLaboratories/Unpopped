kind: test, float Max/Min/Sum collectives over 32 lanes with NaN, signed zero and infinities
---
`kernel_collective_float_nan.rs`: F32 `Reduce` Max, Min and Sum over a 32-lane subgroup against hand-written vectors (NaN propagates, a zero tie keeps the first lane's sign, +inf plus -inf is NaN, Sum is the lane-order left fold); a NaN-ignoring Max/Min, a tie to the last lane and a reversed or pairwise Sum fail the vectors by value. Rule read from the oracle code; the design note matches it. Tests only.
