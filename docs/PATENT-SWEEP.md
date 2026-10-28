# Patent sweep: Unpopped 0.25.1 (covers the 0.26.x line) @ 6a2a17d

Date: 2026-10-25; Run by: unpopped lane, Claude Sonnet 5.5; Method: AI-assisted web search and read of Google Patents pages; NOT legal advice; NOT a freedom-to-operate opinion.

Scope: the `unpopped`, `unpopped-vocab`, `unpopped-cpu-c` and `unpopped-slang` crates at `origin/main` 6a2a17d (0.25.1). 0.26.0 (integer index expressions, PR #130) adds an IR extension of the same kind as the typed-expression and Fma slices already covered below; it introduces no new family of capability. Re-sweep at the next new family (the kernel layer, U2) and at least yearly.

## Ideas swept
1. A vendor-neutral kernel IR (a scalar-expression DAG) that a schedule key lowers, through a per-target `Backend` trait, to source text for several languages (C99, Slang, CUDA).
2. A target that cannot spell an op returns a typed decline (`DeclinedOp`) rather than a guess; unsupported constructs are refused, never approximated.
3. An inward-only e-graph optimizer over one op's value expression, with equality saturation and lowest-cost extraction.
4. A bit-preservation contract on every rewrite (zero signs and quiet-NaN payloads preserved; sign-gated zero identities; no folding across NaN operands).
5. Typed expressions with no implicit promotion: `Cast` with explicit rounding mode and saturation, `Bitcast` between equal-width types, integer index expressions that wrap like the target language.
6. Numerics qualifiers (`Qualified`: rounding mode, flush-to-zero, approximate with a bounded ulp error) attached to an operation as data.
7. Software codecs for low-precision formats (fp8, bf16, f16) emitted as shared C-family text, with the CPU oracle as the bit reference.
8. A CPU oracle that evaluates the IR exactly, and differential tests that compile the emitted text with a real compiler and compare it with the oracle.
9. Measured exactness records: an exhaustive run on one device, stamped with driver, compiler and compiler arguments, read back as an accuracy claim in a contract.
10. A committed, byte-stable dispatch table keyed by a `StructureKey` (op category, shapes, dtype, device class) returning a benchmark winner and margin, with an emit/parse round trip.
11. A source-to-IR lifter that recognizes a grid-stride elementwise CUDA kernel, lifts its body to the IR, and refuses anything it cannot express.
12. Idiom lifting: parsers strip target-specific optimizations back to intent so that each emitter re-lowers for its own target.
13. A self-delimiting contract frame (magic, kind, version, length, CRC-32) that readers hard-reject on any mismatch.
14. Per-device decline of ops and dtypes, driven by capability data, with a conformance matrix across emitters.

## Queries (source, query, date; all 2026-10-25)
Google Patents pages were reached through a web search tool restricted to `patents.google.com` and `patents.justia.com` on most queries; two were unrestricted. The tool returns result lists, not ranked patent-database output, so coverage is partial (see Limitations).
- Unrestricted: "patent vendor-neutral kernel intermediate representation lowering to GPU backends emitter per target declines unsupported op" (0 patents).
- Unrestricted: "patent e-graph equality saturation compiler optimizer bit-preserving rewrite floating point expression extraction lowest cost" (0 patents).
- Unrestricted: "patent lifting GPU kernel source CUDA to intermediate representation re-emit multiple backends refuse unsupported construct" (0 patents).
- Unrestricted: "patent measured numerical exactness record per device driver compiler flags kernel dispatch routing table autotuning winner" (US7191322, US8539453, tangential).
- patents.google.com: "intermediate representation for GPU kernels lowering to multiple target languages code generation patent" (US20150347107A1, US7966610B2, US20070106983A1, US20080066046A1).
- patents.google.com: "equality saturation e-graph program optimization extraction cost function patent" (0 patents).
- patents.google.com: "compiler lowering optimization decision table lookup shape dtype device key select kernel variant dispatch patent claims" (US8296743B2, US7669191B1).
- patents.google.com: "converting source code of one GPU programming language to intermediate representation and generating another language, recognizing idiom, refusing unsupported" (US7921413B2, EP1620793B1, US8533691, US8869103B2).
- patents.google.com: "compiler expression optimizer bit-exact floating point preserving signed zero NaN payload rewrite rules constant folding patent" (hardware FP patents only; US5836014A surfaced and was not examined).
- patents.google.com / justia: "e-graph equivalence classes rewrite tensor program optimizer superoptimization patent" (US20240135076A1, US12693838).
- patents.google.com / justia: "explicit rounding mode saturation cast between floating point and integer types intermediate representation compiler patent" (US20240103872, US20180173527A1, US20150039661A1, US6535898).
- patents.google.com / justia: "verifying generated GPU kernel against CPU reference oracle exhaustive input comparison patent" (US8595701B2 plus papers).
- patents.google.com / justia: "autotuning kernel benchmark winner stored table keyed by operation shape dtype hardware stamp runtime dispatch lookup patent" (0 patents).
- patents.google.com / justia: "low precision floating point 8-bit format software conversion encode decode codec machine learning compiler patent" (EP4318225A1, EP4318224A1, US20240045677A1, US20240045654).

## Hits
"Claims read" says what was actually seen.

| # | patent | title | assignee | status | claim(s) read | overlap | our difference | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | US9740464B2 and continuation US10372431B2 | Unified intermediate representation | Apple Inc. | both granted; Google lists 9740464B2 active, expiry 2034-09-30 | all independent claims of both, in full (`tools/patent-claims.py`): 9740464B2 claims 1, 7, 13; 10372431B2 claims 1, 8, 14 | a device- and language-independent IR compiled to executables for target processors (ideas 1, 11, 12) | 9740464B2 requires an IR model with functions for parallel graphics and non-graphics operations, and a binary of kernels and shaders. 10372431B2 claim 1 requires a library comprising "both graphics operations and non-graphics operations", and two binaries each comprising "both one or more kernels and one or more shaders" for two different target processors. Unpopped's IR is compute-only, has no graphics or shader operations, and emits source text through emitters | clear on all independent claims of both (the graphics-and-shader elements are absent) |
| 2 | US7966610B2 | Application of optimization techniques to intermediate representations for code generation | MathWorks | granted, Google lists active, expiry 2029-10-31 | all independent claims in full: 1, 34, 38, 70, 71 | optimizing an IR before code generation, preserving its algorithmic meaning; dependent claims name common-subexpression elimination and constant replacement (ideas 3, 4) | every independent claim is framed "In a graphical modeling environment" and starts from an IR "derived from ... an executable system model" (claims 70-71 also take the portion to optimize from a user). Unpopped has no graphical modeling environment and no system model; its optimizer takes a scalar expression and no user selection | clear on every independent claim (the modeling-environment element is absent). Whether a preamble limits a claim is a legal question; the sweep does not rely on that |
| 3 | US8296743B2 | Compiler and runtime for heterogeneous multiprocessor systems | Intel | granted, Google lists active, expiry 2031-07-20 | all independent claims in full: 1, 9, 15 | choosing between hardware-specific variants at dispatch (idea 10) | every independent claim requires bundling variants into a generic function that corresponds to a parallel programming keyword, generating meta-wrappers from per-variant annotations, a compiled user-program binary with an embedded run-time call, and dynamic selection and dispatch during execution. Unpopped has no parallel keyword, no meta-wrappers and no runtime dispatch engine; its table is a committed build-time artifact the consumer reads | clear on all independent claims |
| 4 | US20240135076A1 | Super-optimization explorer using e-graph rewriting for high-level synthesis | Intel | pending application (published 2024-04-25) | all three independent claims in full: 1, 9, 17 | builds an e-graph from an IR, extracts a representation from it (idea 3) | every independent claim requires "interleaving transformations to explore equivalent hardware designs", extracting "a selected hardware design" and generating code "optimized for high-level synthesis"; claim 17 adds performing high-level synthesis to a hardware description. Unpopped rewrites a scalar expression for a software kernel and designs no hardware | clear on the published claims. A pending application's claims can change; look again if it is granted |
| 5 | US12693838B2 | Program analysis, design space exploration and verification for HLS via e-graph rewriting | Intel | granted (status not stated by the tool) | all independent claims in full: 1, 11, 17 | e-graph construction and extraction from an IR (idea 3) | every independent claim requires "exploring a design space of equivalent hardware designs" and formal verification of the generated high-level-synthesis program. Unpopped designs no hardware | clear on all independent claims |
| 6 | EP4318224A1; US20240103872A1; US20180173527A1 (read). EP4318225A1, US20240045677A1, US20240045654 (not read) | FP16 to FP8 conversion instruction; truncating float-to-integer conversion with saturation; floating-point instruction format with embedded rounding rule | Intel; Intel; Optimum Semiconductor | applications and one EP application | claim 1 and the other independents of the three read (EP4318224A1 claim set is one block, read in full) | FP8 conversion, saturating conversion, a rounding rule carried with the operation (ideas 5, 6, 7) | all three claim processor hardware: "decoder circuitry", "execution circuitry", an instruction with an opcode or a rounding-rule field. Unpopped's `Cast` and `Qualified` are IR nodes and its codecs are emitted source text; no instruction, decoder or execution circuit is claimed or built | clear on the three read. The other three FP8 hits are the same family of instruction claims and are unread |
| 7 | US7921413B2 / EP1620793B1 | Architecture for generating intermediate representations for program code conversion | IBM (originally Transitive) | filed 2003; the 20-year term has very likely ended (not verified) | US7921413B2 independent claims 1, 14, 15 in full; EP1620793B1 not read | an IR for translating code between architectures (ideas 11, 12) | requires subject-architecture-specific nodes converted into target-architecture-specific nodes by invoking a conversion component, from decoded subject machine code. Unpopped lifts one CUDA kernel idiom from source text into an architecture-neutral IR | clear on US7921413B2; EP1620793B1 unread |
| 8 | US5836014A | Method of constructing a constant-folding mechanism in a multilanguage optimizing compiler | Digital Equipment (now HPE) | filed 1992; term ended | all independent claims in full: 1, 5, 12, 16, 19 | constant folding on an IR (idea 4) | claims a method of producing a constant-folding routine from a universal intermediate language shared by several front ends. Unpopped's optimizer is a rewrite set over a DAG, not a constant-folding-routine generator, and the patent's term has ended | clear (and expired) |
| 9 | US8458680B2; US8146066B2 | Dynamically choosing a processing element for a compute kernel; caching compute kernels for an application on a parallel-processing computer | Google LLC | granted, Google lists active; priority 2006-06-20, estimated expiry 2030-08-08 and 2030-07-05 | all independent claims of both in full (8458680B2: 1, 9, 17; 8146066B2: 1, 8, 15) | choosing a processor type per request and reusing kernels (ideas 1, 10) | charted element by element below; the `Backend` crates lack three or four elements of each independent claim, an adopter's runtime may not | clear for the `unpopped*` crates; **needs-review for any adopter that adds a runtime** (see the chart) |

Hits 1-5 and 7-9 and part of 6 were read in full for every independent claim, with Google Patents text fetched whole by `C:\Projects	ools\patent-claims.py` (2026-10-25). Hit 6 reads three of its six patents; the unread ones are named in the row.

## Needs review (to the PM)
No hit overlaps on an independent claim of the `unpopped*` crates. One item is `needs-review` for the PM: US8458680B2 and US8146066B2 (hit 9) against any adopter that has a runtime choosing a processor type and picking precompiled or cached kernels (Fuel, Baracuda); see the claim chart. Not assessed, and therefore **not cleared**: any adopter's runtime against US8458680B2 and US8146066B2 (hit 9; Fuel and Baracuda chart their own); EP1620793B1, EP4318225A1, US20240045677A1 and US20240045654 (claims unread), and anything a CPC-class search would find that this search tool did not.

## Claim charts added on a PM request (2026-10-25, board 296 E)

Both patents share a 2006-06-20 priority date. Within each, the three independent claims (method, system, medium or program product) recite the same operative elements, so one chart serves each. **Absent elements are listed first.** "Present" means the element exists in the `unpopped*` crates as they are; "downstream" means it exists, if at all, in an adopter (Fuel, Baracuda), which this sweep did not read.

### US 8,458,680 B2, independent claims 1, 9, 17 (choose a processing element, use a precompiled kernel)
Absent from the `unpopped*` crates:
- **"at a runtime system executing at a parallel-processing computer that includes multiple types of processing elements using shared memory"**: no crate is a runtime system; none runs on a machine of several processor types. Nothing here executes a kernel except test harnesses.
- **"at runtime ... choosing a respective type of processing element"** (CPU, GPU or co-processor): no crate makes this choice at run time. A `StructureKey` carries a device class as a table key; the consumer reads the key and picks.
- **"identifying, from a library, a precompiled, processor-specific compute kernel"**: no crate ships a library of precompiled processor-specific kernels. `unpopped-cpu-c` compiles generated C text on demand in its tests; the committed kernel artifacts ship in Baracuda's `baracuda-kernels-sys`, which is downstream.
- **"preparing one or more compute kernels ... that include the precompiled ... kernel"**: follows from the previous element.

Present: operation requests (a `JitRequest` is a request for a region's kernel), and the notion of an intrinsic or primitive operation.

Verdict for the `unpopped*` crates: **clear**, three or four required elements are absent. Verdict for an adopter: **needs-review, not assessed here.** Fuel and Baracuda were charted separately by the PM; an adopter that receives an operation request, selects CPU or GPU at run time, and then picks a precompiled kernel from a shipped library has the shape of this claim, with the "shared memory" and "hand-coded or table-routed" distinctions untested. Unpopped's dispatch-table artifact is the data such a lookup would read, so the PM should hear of any adopter finding.

### US 8,146,066 B2, independent claims 1, 8, 15 (cache compute kernels at runtime)
Absent from the `unpopped*` crates:
- **a runtime system receiving operation requests that "correspond to a code reuse section of the application"** (the dependent claims call it a keyword-delimited section): no crate has such a section or keyword.
- **"dynamically choosing a respective type of processing element for the ... intermediate representation entries"**: as above, no crate chooses a processor type at run time.
- **"identifying, among a plurality of compute kernels previously compiled by the runtime system for the application"**: no crate keeps a set of kernels it previously compiled. `jit.rs` line 244 says only that a pre-built-variant cache "can slot in", which is a note, not code.
- **"scheduling execution of the identified ... kernels at the respective type of processing element"**: no crate schedules or runs a kernel.

Present: "generating one or more intermediate representation entries for the one or more operation requests" in a loose sense: the JIT turns a requested region into op IR (`region_to_op`), and the IR is language- and processor-independent (dependent claim 3).

Verdict for the `unpopped*` crates: **clear**, four required elements are absent. **Design hazard to record:** adding a runtime kernel cache keyed by an IR-derived identifier, together with a run-time choice of processor type and execution scheduling, would supply every remaining element of claim 1. The `jit.rs` note about a pre-built-variant cache should not be built in this crate without a fresh read of this patent (expiry estimated 2030-07-05).

## Addendum: layout algebra (U6), 2026-10-28, `unpopped-vocab::layout_algebra`

Idea 15: a hierarchical `shape:stride` layout value with composition, complement, divide, product, coalesce and an XOR swizzle, as pure data. The idea is published prior art in the CuTe documentation and papers (e.g. arXiv 2603.02298, "CuTe Layout Representation and Algebra"), so novelty is not claimed; this addendum asks only whether a patent reads on it. Same tool and limits as above (web search restricted to `patents.google.com` and `patents.justia.com`; result lists, no claims fetched).

Queries (all 2026-10-28):
- "patent tensor layout algebra composition complement hierarchical shape stride mapping coordinates to memory offsets compiler GPU": no patent on the algebra itself; unrelated tensor patents only (US20210064987A1, US20170200094A1, US20230004365A1, KR20210021258A, US20240176663).
- "patent XOR swizzle shared memory address bank conflict avoidance bit permutation tensor layout": NVIDIA tensor-memory-accelerator family, US20230289304 / US12141082B2 / US20240176663 (hardware unit that swizzles on load into shared memory; patterns held as lookup tables).

| patent | what the search summary says | why flagged | claims |
|---|---|---|---|
| US20230289304A1 | NVIDIA: hardware unit swizzles data as it loads a tensor into shared memory, fixed patterns per layout, to avoid bank conflicts | swizzle for bank-conflict avoidance (`Swizzle`) | unread |
| US12141082B2 | continuation of the above; swizzle encoded as 8x8 lookup tables | same | unread |
| US20240176663A1 | NVIDIA: tensor map cache storage, repeats the bank-swizzling description | same | unread |
| US20210064987A1 | NVIDIA: reinterprets a tensor with overlapping strides so logical elements share memory (stride-0 / overlapping-stride layouts) | `Layout` allows stride 0 and overlapping strides | unread |

Not read: **no claim of any hit above was read.** Verdict: **not cleared, not assessed.** The TMA family describes a hardware unit; `Swizzle` here is an arithmetic function in a vocabulary crate, which is a difference of kind, but that is a reading of titles and summaries, not of claims. Needs the PM's claim-reading pass (`tools/patent-claims.py`) before any "clear".

## Limitations
- AI-assisted search by one model in one session; the search tool returns general web results, mostly papers, and found no patent for several ideas (2, 4, 9, 13, 14). **A null from this tool is not evidence that no patent exists**: the same tool did find patents on other queries, but it is not a patent database and cannot be queried by claim language, class or assignee.
- No query used CPC classes (G06F 8/41, G06F 8/51, G06F 9/45516), USPTO full text, Espacenet or WIPO PATENTSCOPE. These are the next step for anyone with access.
- Claims for the rows marked read were read in full from Google Patents' machine-converted text; the other FP8 and EP hits in rows 6-7 were not read. Status and expiry dates are Google Patents' own assumptions.
- No CPC-class pass: no USPTO or Espacenet access in this session, and the claims tool reads one named patent at a time.
- Ideas 8, 9 and 13 are mostly engineering practice; they were searched with one query each.
- Not legal advice. A freedom-to-operate review by a patent attorney is a separate, later purchase.
