# The Unpopped catalog

**Status: design, not yet built.** Nothing here is implemented except the pieces
explicitly marked *(exists today)*, which are load-bearing enough that the design
is mostly assembly rather than invention.

---

## 1. What it is, in one paragraph

Unpopped is a kernel generator. The catalog is **a set of op logic** — neutral IR
— that Unpopped can generate into any registered emitter's language, on demand,
for any requested cell. It stores no generated kernels. A consumer asks for
*(op identity, cell, target)* and gets a kernel produced fresh, together with its
contract and link row.

The name "server mode" is how this started, and it is the wrong emphasis. Serving
over a socket is a deployment choice at the end of the design, not its centre.
The centre is: **what does Unpopped know, and what can it therefore produce.**

## 2. Why this is a *catalog* and not a *compiler*

KISS draws a line that decides the trust model, and Unpopped sits on one side of
it:

- **Provision by identity** — a *catalog* operation. The consumer names an op the
  provider already knows. The contract is provider-authored, so guarantees are
  known before anything is built, and a consumer **may trust** it.
- **Region synthesis** — a *compiler* operation. The consumer hands over a region
  it composed; the contract is generated from that region, so the consumer
  **MUST verify** it.

The distinction is the trust model, not the payload — both return a kernel and a
contract. Eric's direction is squarely catalog, which is why this design is
unblocked today and depends on none of the sk4 schema event.

**Consequence worth stating plainly, because the code currently reads the other
way:** `jit::synthesize` takes a `JitRequest` carrying a `region: PatternNode`.
That is region synthesis — the compiler shape. The catalog's request carries an
**op identity**, not a region. Both can coexist; they are different operations
with different trust rules, and conflating them would quietly move verification
obligations onto consumers who were told they could trust.

## 3. What is stored: logic, never shape *(exists today)*

A catalog entry's payload is an `OpDef` — the neutral IR. The entire stored form
of `add` is:

```json
"body":   { "Add": [ {"Input": 0}, {"Input": 1} ] },
"access": "Elementwise"
```

That is *what to compute at one coordinate* and *what kind of iteration*. No
loop, no bounds, no indexing, no vector width, no launch shape.

Everything about sizes and shapes is a property of the **cell**, not the op, and
is re-derived at generation. Measured, from one identical stored IR:

| cell | schedule chosen |
|---|---|
| n=7, align 256 | `Scalar` |
| n=1024, align 256 | `Vectorized { width: 4 }` |
| n=1000, align 4 | `Scalar` |

n=1024 and n=1000 differ only in **alignment**, and that alone flips the
strategy. None of this could live in the IR even in principle.

The storage form already exists: `op_to_text` / `op_from_text` round-trip a full
`OpDef` through serde JSON — the whole IR, not just the elementwise subset.

### The one field that is not logic

`OpDef.dtypes` is the op's *accepted* dtype set. The dtype a kernel is actually
generated at comes from the cell, so this is **admissibility metadata**, not
logic. It is the one place the op is asked a question the cell should answer, and
both bugs found while designing this lived there (§9).

## 4. Intake: how logic enters the catalog *(exists today)*

Two doors, and the second is the interesting one.

1. **Authored** — an `OpDef` built with the Rust EDSL, or loaded from op text.
2. **Lifted from source** — a consumer hands Unpopped a kernel in a source
   language; `lift_elementwise` recovers the per-element body into neutral IR.

The lifter's posture is already right and should not be softened:

- **It refuses rather than mis-lifts.** A non-`[i]` index, an unknown call,
  shared memory, an atomic, a library call — each is a typed `LiftError`, never a
  silent approximation. *"Convert what you can, leave the rest in the source
  language"*: portability scales with the lift fraction, honestly.
- **It strips exactly the generated boilerplate.** The grid-stride loop and the
  indexing do not survive the lift, because the generator re-adds them. That is
  the same split as §3, arrived at from the other direction.
- **Round-trip is validated against the oracle**, semantically. It is
  deliberately *not* textually lossless, and textual losslessness would be the
  wrong goal — the whole point is to discard what generation supplies.

**Identity is caller-declared.** `lift_elementwise(src, name, dtypes)` — the
lifter recovers a body but cannot know what the op should be called or which
dtypes it should serve. Those are the caller's assertions, and as of `5196b4a`
the dtype assertion is **validated at intake** against both admissibility gates
rather than trusted. In a catalog that is the difference between a bad request
and a stored broken entry.

## 5. Emitters: compiled in, selected by name

A registry maps a target name to a `Backend`. The host binary or crate depends on
the emitter crates it wants and registers each; selection happens per request.

```rust
let mut reg = EmitterRegistry::new();
reg.register("cuda",  Box::new(UnpoppedCuda));   // unpopped-cuda
reg.register("slang", Box::new(Slang));
reg.register("c",     Box::new(CpuC));
```

No dynamic loading. The `Backend` trait is mid-flight — `lower -> Result` landed
today — and freezing a C ABI for a plugin boundary while the trait is still
moving would buy runtime flexibility nobody has asked for at the cost of a
versioning surface that outlives the decision.

This is also where the emitter carve lands: each reference emitter is its own
crate under the umbrella, and the registry is the seam they plug into.

## 6. The resolve operation

```
resolve(op_id, structure_key, target) -> Entry
```

with

```
Entry { kernel, contract, link, provenance }
```

Everything on the right already exists: `generate` → `contract` → `link_entry`
produce exactly this, and `JitResponse` proves the bundle is the right one for a
consumer to adopt a kernel.

**It must not unwind.** A resolve serves a request from outside, so every refusal
is typed: `try_build_plan` for an inadmissible op/cell pair, `Backend::lower`'s
`LowerError` for a target that cannot emit it. Both landed today, and the second
closed a live bug where an ordinary `u8` add panicked across the JIT boundary.

## 7. Validity: what an entry was baked against

Every generated kernel already carries `Provenance { structure_key, backend,
provider, generator }`, stamped by the core rather than by the backend, so a
backend cannot stamp the wrong cell or forget.

For anything that **caches** a resolve, the validity key is
`(structure_key, revision_hash)` per KISS-Synth §6.7-0004 — never the
`structure_key` alone. The key names *what was asked for*; it does not name *what
was produced*. Change a lowering rule and the same request yields different code
under an unchanged token.

### The gap this design opens, and must close

`Provenance.generator` is the **generator's** version. That is sufficient today
because everything Unpopped bakes ships inside Unpopped. **A catalog breaks that
assumption**: the op logic is now caller-supplied — authored or lifted — so it
varies independently of the generator's version. Edit an op's IR, and
`(op_id, structure_key)` resolves to different code with every existing validity
field unchanged.

That is precisely the rule Fuel paid for four times over: *a held artifact's
validity key MUST name every thing it was baked against, or that thing MUST be
unable to change for the artifact's lifetime — enforced by construction, not by
convention.* A catalog entry is baked against its op logic, and nothing currently
names it.

So an entry's identity must include a **digest of the stored `OpDef`**, not just
the op's name. `op_to_text` already gives a canonical serialization to hash, so
this is cheap — but it has to be designed in, because the failure is silent and
`op_id` looks like sufficient identity right up until someone edits an op.

This also answers a question §10 would otherwise have to: an op *name* is a
label, an op *digest* is an identity, and the catalog needs both.

## 8. Phasing

**Phase 1 — the library.** `unpopped-catalog`: op registry, emitter registry,
`resolve`, and an emit-to-directory command for build-time use. In-process, no
protocol, no daemon. This is where all the design risk actually is.

**Phase 2 — the server.** A thin binary over the same library. Deliberately
sequenced *after* the wire formats are co-pinned (the KISC header and recipe
`Semantics` are still PROVISIONAL). Freezing a protocol over a provisional format
would create a second place where the format is defined — exactly the two-homes
drift that produced today's two bugs.

Non-goals for both phases: not a package registry, not an artifact store, not a
build system, and not a replacement for a consumer's own JIT path.

## 9. What designing this found

Recorded because it is the argument for the design rather than a footnote to it.
Both bugs were in `OpDef.dtypes` — the one field that is metadata rather than
logic — and both were *catalog-shaped*: harmless in a one-shot call, damaging
when an op is stored once and generated on demand.

- **`939b34d`** — `recipe.rs` decided integer-vs-float from `dtypes.first()`, a
  proxy that answers for the whole list. Latent only because no op today mixes
  float and int.
- **`5196b4a`** — `lift_elementwise` never checked that its caller's declared
  dtypes could be honoured by the body it lifted. A float-only `expf` could be
  stored as accepting `I32`.

The second is the one that justifies validating at intake: an op is lifted once
and generated many times, so the error surfaces to whoever *requests* the cell
rather than whoever *stored* it.

## 10. Open

- **Discovery.** Nothing above lets a consumer ask *what is in here* — which ops
  exist, and which cells each can serve. Listing the ops is easy; "which cells"
  is not, because the answer is "any cell the op is admissible at", which is a
  predicate rather than a set. A catalog that cannot be enumerated is awkward to
  build tooling against, and pretending the cell space is enumerable would be
  worse.
- **Op identity across sources.** Two consumers lift semantically identical
  kernels and name them differently. The §7 digest makes this *detectable* —
  identical IR hashes identically — but detecting is not the same as deciding.
  Silent unification would be surprising; a name collision on differing logic is
  a genuine error and should probably be refused.
- **Whether the catalog is a KISS-Consume producer.** Intake refusals already map
  to `ConsumeRefusal`, and §4's dtype refusal exposed that the four categories
  cannot distinguish *inexpressible construct* from *expressible construct, wrong
  declared dtype*. Only the second is caller-actionable. Raised as a vocabulary
  observation, not resolved here.
- **Cross-target consistency.** Nothing requires two emitters to agree, beyond
  each satisfying `docs/conformance.md`. Whether the catalog should *check* that
  — generating the same cell for two targets and comparing against the oracle —
  is a real option and a real cost.
