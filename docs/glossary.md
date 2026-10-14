# Glossary

The project's own terms, each defined once. Where a term is owned by another
project (KISS, Baracuda, Fuel), the entry says so and points at where it is
defined; this file does not restate their rules. For why something is deferred,
see [`deferred.md`](deferred.md); for what changed, `CHANGELOG.md`.

**Accumulator + math-precision (`acc` / `mp`).** The optional trailing field of a
non-contraction structure key, naming the accumulator dtype and whether operand
mantissas are rounded before compute (`st` bit-stable, `rm` reduced-mantissa).
It is omitted when both sit at their defaults, so a canonical token has exactly
one spelling. A contraction cell carries the same facts inside its contraction
group instead.

**Backend / emitter.** A crate implementing `unpopped::Backend`: it turns a
neutral kernel plan into source for one target. Emitters are `unpopped-cpu-c`
(portable C99), `unpopped-slang`, and Baracuda's `baracuda-cuda-emit`. The core
crate holds none, so no emitter is the default by accident of location.

**Baracuda.** The CUDA project that consumes Unpopped and owns the CUDA emitter.
Unpopped depends on no Baracuda crate; the dependency points one way.

**Broadcast mask.** The per-operand bitmask in a structure key: bit `i` is set
iff frame axis `i` has extent greater than 1 and the operand's stride along it
is 0. An axis the operand lacks counts as stride 0, and an own axis of extent 1
counts as stride 0 whatever stride is recorded (since sk5).

**Byte-match.** Comparing a token this crate produces against a vendored KISS
vector as bytes, never field by field, so a different spelling of the same
fields cannot pass. The codec leg is `tests/kiss_byte_match.rs`.

**Capability row.** A sourced record of what one target architecture permits,
as the numbers a schedule chooser needs (`unpopped::capability`). It states what
the hardware allows and carries no policy.

**Cell.** One specialization point: an operation over operands of a given
layout class on a given target. A structure key names a cell.

**Cell class.** One of seven structural classes a cell derives from its operands
and three declared facts: `ctr` (contraction), `ndx` (indexed access), `rdc`
(reduction), `ew1`/`ew2`/`ew3` (pointwise with one, two or three inputs), `oth`.
Derived, never supplied, and it does not name the op: a softmax and a unary
activation over same-shape input and output are both `ew1`. Replaced the 24
op-family codes at sk5.

**CpuC.** The portable-C99 reference emitter, `unpopped-cpu-c`.

**Decline (typed).** A refusal that names its reason instead of guessing:
a derivation that cannot produce a key returns a `DeriveDecline`, and a token
the reader cannot honour a `TokenDecline`. Each has a `wire_name()` in the KISS
decline vocabulary.

**Dequant form.** How a quantized operand's stored bits become values: `linear`
(`dl`) or `codebook` (`dc`). Keyed from sk5 as the optional last component of
the operand's sub-key. Never inferred from the quantization family.

**Derivation vector.** A KISS test case that gives operand descriptors and
declared facts and states the key (or decline) a conformant deriver must
produce. Distinct from the codec vectors, which test the token's spelling.
Only those carrying `spec_golden` are anchored to hand-derived bytes in the
spec; the rest are KISS's reference output.

**Dispatch table.** A mapping from structure-key token to the implementor
(a generated kernel, a vendor library) that serves that cell.

**Fuel.** A sibling tensor runtime that builds operand descriptors from its own
tensors and calls the single canonical key function rather than reimplementing
it.

**Iteration frame.** The shared index space of a cell: its rank is the widest
operand's rank, each axis takes the largest extent any operand has there, and
operands are right-aligned into it. The broadcast mask and the work class are
read over the frame; layout, vector width and divisibility are read from an
operand's own axes.

**KISS.** The standard this project implements: KISS-Classify (the structure
key), KISS-Ops (the op vocabulary) and KISS-Contract (contract framing, KISC).
Owned by the KISS project; clause numbers such as §6.5-0015 refer to it.

**Lift.** Recognizing an existing source-language kernel and expressing it in
the neutral IR (`unpopped::lift`). A lift refuses
anything it cannot express faithfully rather than mis-lifting it.

**Operand role.** Whether an operand is read, written, or both. Operands with
roles are put in canonical order (inputs and in-place operands, then outputs)
before keying.

**Oracle.** The independent CPU interpreter of a kernel plan, sharing no
lowering code with any emitter, so a spelling bug cannot hide in both.

**Reduce field.** Field 8 of a token: `-` (not a reduction), `rall`, `rlast`
(the lone trailing axis) or `x<hh>` (an explicit bitmask). Non-`-` iff the class
is `rdc`, and each reduced set has exactly one accepted spelling.

**Reserved dtype.** A spelling in the shared dtype vocabulary that has no
computation semantics yet. It is recognized on parse, so it can be told apart
from an unknown token, and it declines wherever it appears in a dtype *key*
position.

**Scale dtype.** A dtype (`f8e8m0`, `f8e6m2`) that only ever describes a block
scale. It is valid on a sibling operand and a decline at operand 0.

**Schema version (`sk<N>`).** The structure-key token's version field. A change
that is not additive is a schema event: it moves the number, and an old token is
`UnsupportedSchemaVersion` rather than silently reinterpreted.

**Sibling operand.** How a block's shared scale reaches the key: as its own
entry in the operand list, not as a field on the operand it scales.

**Structure key.** The canonical identity of a layout class: the join token
shared by runtime dispatch, contract predicates and telemetry. A pure function
of the request, so two candidates for one request cannot differ in it. Its
string form is the token.

**Target / `TargetId`.** An interned `<namespace>:<capability-set>` string such
as `cuda:sm89`. The namespace set is open; the capability vocabulary after the
colon belongs to the namespace's maintainer.

**Token.** The string wire form of a structure key, e.g.
`sk5|ew2|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;…|-`. The only normative
form; the in-memory struct is a permitted representation.

**Vendored artifact.** A KISS file copied verbatim into `kiss/` so it can be
compared against its source with `diff`. Never edited; its blob hash and source
commit are recorded in `kiss/README.md`.

**Work class.** `warp`, `block` or `grid` by total frame element count (at most
32, at most 1024, otherwise).
