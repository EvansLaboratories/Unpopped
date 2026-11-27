kind: added
---
`kernel_interface::check_launch_limits(kernel, &TargetCapabilities)` and the `non_exhaustive` `LaunchLimitError`: a kernel's block thread count against `max_threads_per_block`, and a required subgroup size range against `warp_size`. Additive; the sibling of `check_shared_size`. Per-axis, grid, cluster and occupancy limits are not modelled.
