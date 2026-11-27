kind: tests: device-scope semaphore probe (survey row 9)
---
Pins what the oracle does with an `AtomicCas(Acquire, Device)` spin across blocks: a producer in the earlier block completes (`out[0]=42`); a consumer in the earlier block is `CrossBlockWait`; an unreleased spin is `CrossBlockWait` while a block remains and `StepLimit` when the spinner is the last block. Weakening Acquire to Relaxed turns the completing case into `DataRace` and the waiting case into `StepLimit`. Tests only, no source change; verified by `cargo test -p unpopped --test kernel_device_semaphore`.
