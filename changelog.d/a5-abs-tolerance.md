kind: attention backward C differential: absolute tolerance (tests only)
---
Corrects the `0.47.5` entry: `kernel_attention_backward_emit_agrees.rs` compares the C against `run_launch` within 1e-4 of the largest gradient, not 3 ulps. The first CI run showed why: `dQ` element 12 is 2e-4 from terms in the hundreds and differed by 4 ulps (C rounds per operation, the oracle per statement), and a cancelling sum has no meaningful ulp bound. A test tag containing `/` also broke the temporary file path and is fixed. No kernel or oracle change.
