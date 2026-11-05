kind: GEMM-family kernel K7: implicit-GEMM convolution forward (ROADMAP U9; tests only)
---
`conv_implicit_gemm` (`tests/support/conv.rs`, copied into cpu-c): NHWC forward convolution as a GEMM over `(r, s, c)` with the
im2col index math inside the loop and zero padding by skipping taps outside the image, no staged matrix. The shapes make an
index mix-up loud: a non-square image (4x5), a non-square filter (3x2), padding on height only, three input channels, two
output channels, two batches (one block each). Oracle, the C compiled with a host compiler, and a direct host convolution
agree. Twelve seeded mutants (pad dropped or added, filter or input coordinates swapped, batch ignored, bounds swapped
or off by one, channel loop short, output coordinates swapped, accumulator not accumulated) are all caught. dgrad, wgrad
and grouped convolution follow only if a consumer asks.
