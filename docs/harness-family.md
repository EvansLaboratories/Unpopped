# Harness family: an Unpopped test-and-profile proposal

**Status 2026-10-22:** built and moved to `ThinkersJournal/unpopped-harness` (cpu-c, CUDA and Vulkan targets). "What exists" below is the 2026-10-18 snapshot, kept as written.

From: Unpopped lane. For: Portfolio PM. Date: 2026-10-18. Requested by CireSnave (Eric).

## What CireSnave asked for (his words, 2026-10-18)

> "this should evolve into a set of test harnesses that wrap Unpopped. One would test and
> compare kernels across multiple backends and provide fingerprinting and profiling of the
> execution as much as possible at each step throughout the process. Another couple of
> harnesses would fingerprint and profile Unpopped parsers and emitters across a set of
> kernels. Another would use those to compare the kernels output from Unpopped emitters to
> find hotspots in one emitter as compared to others to find the most likely dark corners to
> dig into to optimize the functionality of that emitter. None of these harnesses are set in
> stone at this point.. I wrote them out to illustrate the potential breadth of types of
> harnesses I am suggesting building. Once the harnesses exist, Unpopped or some other
> project or person could call them to test its own work on parsers, IR optimization rules,
> emitters, and specific kernels."

He also said the harnesses should be external to Unpopped and use the rest of the ecosystem
(Baracuda/CUDA, Vulkane/Vulkan, AOCL, oneMKL, Fuel's Slang kernels, others) and feed results
back into Unpopped.

## What exists (measured 2026-10-18)

`C:\Projects\unpopped-evolve-scratch`: scratch crate, not a git repo, depends on published
unpopped 0.20.13 crates only. It compares a reference kernel to a candidate over all 2^32 f32
inputs through cpu-c (`cl /O2`), with contract tiers Exact / Ulp(n), a two-sentinel
never-wrote check, and 8 planted-bug tests (all pass; they found one defect in the harness
itself). Only unary f32 so far. Only cpu-c can execute on this box: slang compiles to SPIR-V
but nothing dispatches it; the CUDA emitter lives in Baracuda.

## Proposed shape

A separate repo (name TBD) of independent harnesses over a thin common core:

- **core**: contract (tier, domain, reference hash), a `Target` trait (run an emitted kernel on
  inputs, return outputs and timings), fingerprints (kernel structure key, emitted-text hash,
  emitter/compiler/target versions), result records that name all of those.
- **targets**: one adapter per ecosystem project, each calling that project's PUBLISHED API
  (crates.io versions only). cpu-c first (done). Candidates: Vulkane (slang/SPIR-V), Baracuda
  (CUDA), AOCL, oneMKL. Each needs a survey of the owner's dispatch API before wiring.
- **harness A, cross-backend kernel compare**: same kernel, many targets; correctness against
  the contract, fingerprints and profile at each stage (parse, optimize, plan, emit, compile,
  run).
- **harness B, parser/emitter profile**: fingerprint and time Unpopped's own stages across a
  corpus of kernels.
- **harness C, emitter hotspot map**: uses A and B; normalizes per-target timings against each
  target's own ceiling and points at the ops where one emitter is the outlier.
- **callers**: Unpopped CI, other lanes, or a person, invoking a harness to check parser,
  IR-rule, emitter or single-kernel changes. Outputs: optimizer-rule candidates (PRs),
  suspected emitter defects (issues), per-target artifacts.

Offline, spare-time, never in the JIT path. The wider evolution ideas (exact rewrites via
e-graphs, bounded-error approximation search, schedule enumeration) become clients of these
harnesses rather than separate tools.

## Decisions for the PM

1. Repo: new repo, name, owner lane, CI and branch protection from day one (the standing rule).
2. Which lane surveys each target's dispatch API (Baracuda, Vulkane, AOCL, oneMKL, Fuel).
   I can do the read-only survey for any of them.
3. Does Fuel's Slang kernel set, or the backend-agnostic kernel library, already hold the
   dispatch plumbing and a kernel corpus we should reuse?
4. Roadmap row, or a private experiment until harness A shows signal on a second target?

## Known limits and open facts

- Exhaustive 2^32 testing only works for unary f32 bodies; other shapes need stratified
  sampling plus edge cases, and a sample can never accept.
- Raw timings are not comparable across hardware; harness C must normalize.
- Unverified: whether the cpu-c `Max` behaving like relu (NaN-propagating, -0.0-preserving)
  contradicts the IR docs, which say max scrubs NaN. Observed on the harness, not yet chased.
