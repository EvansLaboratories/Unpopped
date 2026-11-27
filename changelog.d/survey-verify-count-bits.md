kind: test, wave count-bits idioms (survey row 46) and survey batch 3
---
`kernel_wave_count_bits.rs`: WaveActiveCountBits as a subgroup Reduce(Sum) and WavePrefixCountBits as an exclusive Scan(Sum) over a predicate, checked by hand-written arrays and against a Ballot mask; an inclusive-scan and a Max mutant change a lane by value. Also survey batch 3 (6 rows) in the verification doc. Tests and docs only.
