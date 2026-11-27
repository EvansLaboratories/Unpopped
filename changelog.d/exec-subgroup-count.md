kind: added
---
`ExecId::SubgroupCount`: the number of subgroups in the block (threads divided by the subgroup size, rounded up; WGSL `num_subgroups`, MSL `simdgroups_per_threadgroup`). `ExecId` is `non_exhaustive`, so additive. The oracle reads it; cpu-c declines it by name. Pure id leaf, no statement.
