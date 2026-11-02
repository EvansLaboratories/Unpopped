# unpopped-cpu-c

The portable-C99 **reference emitter** for [Unpopped](https://github.com/EvansLaboratories/Unpopped): it lowers a
neutral `KernelPlan` to C source that compiles on any C compiler, with no vendor intrinsic, no conditional compilation
and no device.

It is the normative reference emitter a new target is read against, and the one every differential test runs through.

## What's in it

- A `Backend` that turns a plan into C99 source, and a `Compiler` that builds and loads it.
- The software codecs the narrow dtypes need: FP8 E4M3/E5M2, sub-byte `i4`/`u4`/`b1`, and a struct complex ABI, so
  `f8e4m3fn` and `c128` lower without naming anyone's toolchain.
- Integer division and shifts that are total (the ruled guards of `docs/int-div-shift-semantics.md`).

## Use

Depend on the exact version of `unpopped` it was released with (pre-1.0; pin exact versions):

```toml
unpopped = "=0.40.21"
unpopped-cpu-c = "=0.40.21"
```

See the `unpopped` crate documentation for the plan and kernel IR this crate consumes.

Licence: see the repository.
