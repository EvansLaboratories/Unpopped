//! Inward algebraic optimizer — an e-graph over the op IR (Kernel-Seam §5.1).
//!
//! §5.1 charges the synthesizer with building the **best** kernel for the
//! Fuel-chosen region, and explicitly permits an **e-graph / equality-saturation**
//! optimizer pointed *only inward* at that region. This is it: intern the op body
//! ([`ScalarExpr`]) into an e-graph, saturate a set of algebraic rewrites that
//! merge equivalent forms into one e-class, then **extract the lowest-cost form**.
//!
//! It is pointed strictly inward — it rewrites the *value* expression of one op,
//! never scanning a graph or choosing regions (that's Fuel's, §5.1). [`optimize`]
//! is a pure `ScalarExpr -> ScalarExpr` simplification used by JIT synthesis for
//! codegen; the recipe (`pattern:`/`decompose:`) stays the original region so
//! Fuel's matcher still recognizes the subgraph.
//!
//! # Scope (first cut)
//!
//! Total, precision-safe rewrites only: the const-`0`/`1` identities, constant
//! folding of the *algebraic* ops (transcendentals are left unfolded to avoid
//! host-f64 vs device-f32 divergence), and the `neg(neg x) -> x` involution.
//! Equality-saturation extraction picks the cheapest equivalent. The rewrite set
//! is the growth surface (factoring, FMA, perspective-diverse identities); the
//! e-graph machinery underneath does not change as rules are added.
//!
//! # Bit-preservation contract
//!
//! Every rewrite preserves the device result **bits** for all inputs, with one
//! documented carve-out: an eliminated arithmetic op no longer *quietens* a
//! signaling-NaN input, so sNaN payloads are out of contract (the platform
//! compilers make the same call). Zero **signs** and quiet-NaN payloads are IN
//! contract — which is why the zero identities are sign-gated by bits
//! (`x + (-0) -> x` is exact for every `x`, `x + (+0)` is NOT: `(-0)+(+0) = +0`;
//! dually `x - (+0) -> x` is exact and `x - (-0)` is not) and why const folds
//! skip NaN operands (a folded host NaN would drop the device's payload
//! propagation and the authored sign).

use crate::ir::{BinaryOp, ScalarExpr, UnaryOp};
use std::collections::HashMap;
use unpopped_vocab::ElementKind;

type Id = usize;

/// An e-node: an op shape whose children are e-class ids. `Const` stores the
/// `f64` bit pattern so the node is `Hash`/`Eq` (NaN-safe by bits).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum ENode {
    Input(u8),
    Const(u64),
    Param(u8),
    /// Opaque per-row reduced scalar ([`ScalarExpr::Reduced`]) — a leaf, never
    /// folded (a row scalar must not be CSE'd/constant-folded across rows).
    Reduced(u8),
    /// Opaque output-coordinate leaf ([`ScalarExpr::Coord`], increment 0d) —
    /// hash/eq by axis, ZERO rewrite/fold rules (its value varies per output
    /// coordinate, so no host fold is even well-typed). `Coord(d) == Coord(d)`
    /// hash-consing into one e-class is fine (same value at every coordinate);
    /// no rule may equate `Coord(i)` with anything else. Pinned by
    /// `coord_is_an_opaque_leaf_with_no_rules`.
    Coord(u8),
    Add(Id, Id),
    Sub(Id, Id),
    Mul(Id, Id),
    Div(Id, Id),
    Binary(BinaryOp, Id, Id),
    Unary(UnaryOp, Id),
    /// Bitwise ternary select `(cond, a, b)` ([`ScalarExpr::Select`]) — ZERO
    /// rewrite/fold rules, deliberately (the 0b cmp posture: when-in-doubt-
    /// add-no-rule). In particular: no const-cond fold, no `select(c, x, x) →
    /// x` (both shown bit-sound but deferred), NO mask-multiply equation in
    /// either direction (`x*cond ⇎ select(cond, x, 0)` — that rewrite IS the
    /// triu signed-zero/NaN bug; triu must be *authored* as select), no
    /// distribution through ops, no cond simplification, and never a
    /// NaN-cond fold (NaN cond is TRUE). Pinned by
    /// `select_is_never_folded_or_rewritten`.
    Select(Id, Id, Id),
}

/// An e-graph: union-find over e-classes + per-class e-node sets + a hashcons.
#[derive(Default)]
struct EGraph {
    parent: Vec<Id>,
    class_nodes: HashMap<Id, Vec<ENode>>,
    memo: HashMap<ENode, Id>,
}

impl EGraph {
    fn find(&mut self, mut x: Id) -> Id {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

    /// Read-only find (no compression) — for extraction's shared borrows.
    fn find_imm(&self, mut x: Id) -> Id {
        while self.parent[x] != x {
            x = self.parent[x];
        }
        x
    }

    /// Canonicalize an e-node's child ids through the union-find.
    fn canon(&mut self, n: &ENode) -> ENode {
        match *n {
            ENode::Add(a, b) => ENode::Add(self.find(a), self.find(b)),
            ENode::Sub(a, b) => ENode::Sub(self.find(a), self.find(b)),
            ENode::Mul(a, b) => ENode::Mul(self.find(a), self.find(b)),
            ENode::Div(a, b) => ENode::Div(self.find(a), self.find(b)),
            ENode::Binary(op, a, b) => ENode::Binary(op, self.find(a), self.find(b)),
            ENode::Unary(op, a) => ENode::Unary(op, self.find(a)),
            ENode::Select(c, a, b) => ENode::Select(self.find(c), self.find(a), self.find(b)),
            ref leaf => leaf.clone(),
        }
    }

    /// Intern an e-node (hashcons), returning its e-class id.
    fn add(&mut self, n: ENode) -> Id {
        let c = self.canon(&n);
        if let Some(&id) = self.memo.get(&c) {
            return self.find(id);
        }
        let id = self.parent.len();
        self.parent.push(id);
        self.class_nodes.entry(id).or_default().push(c.clone());
        self.memo.insert(c, id);
        id
    }

    /// Merge two e-classes; returns whether they were distinct.
    fn union(&mut self, a: Id, b: Id) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        self.parent[rb] = ra;
        if let Some(rb_nodes) = self.class_nodes.remove(&rb) {
            self.class_nodes.entry(ra).or_default().extend(rb_nodes);
        }
        true
    }

    /// The constant value of an e-class, if it contains a `Const` e-node.
    fn class_const(&self, id: Id) -> Option<f64> {
        let rc = self.find_imm(id);
        self.class_nodes.get(&rc)?.iter().find_map(|n| match n {
            ENode::Const(bits) => Some(f64::from_bits(*bits)),
            _ => None,
        })
    }

    /// Re-canonicalize the class/hashcons index after a batch of unions (child
    /// ids through find, dedup). No congruence merging is needed: a single
    /// interned expression shares each subterm's class, so a simplification
    /// propagates to parents through the shared class id at extraction time.
    fn rebuild_index(&mut self) {
        let old: Vec<(Id, Vec<ENode>)> = self.class_nodes.drain().collect();
        self.memo.clear();
        let mut fresh: HashMap<Id, Vec<ENode>> = HashMap::new();
        for (c, nodes) in old {
            let rc = self.find(c);
            for n in nodes {
                let cn = self.canon(&n);
                self.memo.insert(cn.clone(), rc);
                let v = fresh.entry(rc).or_default();
                if !v.contains(&cn) {
                    v.push(cn);
                }
            }
        }
        self.class_nodes = fresh;
    }
}

fn add_expr(eg: &mut EGraph, e: &ScalarExpr, compute: Compute) -> Id {
    match e {
        ScalarExpr::Input(i) => eg.add(ENode::Input(*i)),
        // Rounded AT INGEST, so every `Const` in the e-graph is a value the
        // device can actually hold. A body carrying `Const(0.1)` at an f32
        // kernel computes with `(float)0.1` on device; folding against the f64
        // `0.1` would diverge before any chaining is involved. Emission is
        // unaffected — `const_lit` spells the rounded value, which converts to
        // the same `float`.
        ScalarExpr::Const(v) => eg.add(ENode::Const(compute.round(*v).to_bits())),
        ScalarExpr::Param(i) => eg.add(ENode::Param(*i)),
        ScalarExpr::Reduced(i) => eg.add(ENode::Reduced(*i)),
        ScalarExpr::Coord(d) => eg.add(ENode::Coord(*d)),
        ScalarExpr::Add(a, b) => {
            let (a, b) = (add_expr(eg, a, compute), add_expr(eg, b, compute));
            eg.add(ENode::Add(a, b))
        }
        ScalarExpr::Sub(a, b) => {
            let (a, b) = (add_expr(eg, a, compute), add_expr(eg, b, compute));
            eg.add(ENode::Sub(a, b))
        }
        ScalarExpr::Mul(a, b) => {
            let (a, b) = (add_expr(eg, a, compute), add_expr(eg, b, compute));
            eg.add(ENode::Mul(a, b))
        }
        ScalarExpr::Div(a, b) => {
            let (a, b) = (add_expr(eg, a, compute), add_expr(eg, b, compute));
            eg.add(ENode::Div(a, b))
        }
        ScalarExpr::Binary(op, a, b) => {
            let (a, b) = (add_expr(eg, a, compute), add_expr(eg, b, compute));
            eg.add(ENode::Binary(*op, a, b))
        }
        ScalarExpr::Unary(op, x) => {
            let x = add_expr(eg, x, compute);
            eg.add(ENode::Unary(*op, x))
        }
        ScalarExpr::Select(c, a, b) => {
            let (c, a, b) = (
                add_expr(eg, c, compute),
                add_expr(eg, a, compute),
                add_expr(eg, b, compute),
            );
            eg.add(ENode::Select(c, a, b))
        }
    }
}

/// The float precision the **device** evaluates this kernel's body at.
///
/// # Why constant folding needs this
///
/// Folds happen in host `f64`. For a *single* operation on `f32`-representable
/// operands that is provably harmless: `f64` carries 53 bits, ≥ 2·24+2, so
/// computing in `f64` and rounding once to `f32` yields the correctly-rounded
/// `f32` result for `+`, `-`, `*`, `/` and `sqrt` (the classic
/// innocuous-double-rounding bound).
///
/// **Chains break it.** The optimizer folds an inner constant expression to an
/// `f64` constant and then folds the outer op on *that*, so an intermediate the
/// device would have rounded to `f32` stays at `f64` precision. Measured, on
/// `sqr(sqr(c))` at `f32`:
///
/// ```text
/// c = -2.7932066917419434
///   fold at f64, round once  : 60.87126159667969   bits 0x42737c2c
///   device, f32 step by step : 60.87126541137695   bits 0x42737c2d
/// ```
///
/// One ULP apart — which violates this module's own bit-preservation contract
/// ("every rewrite preserves the device result **bits** for all inputs"). The
/// contract was right and the implementation did not honour it. Rounding every
/// fold result to the compute precision restores it: each fold now lands on the
/// bits the device would have produced at that step.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Compute {
    /// Single precision — `f32`, and every narrow float, which the emitters
    /// promote to `float` and compute at `f32` (see `cfamily`'s narrow-float
    /// seam).
    F32,
    /// Double precision.
    F64,
}

impl Compute {
    /// The precision a kernel at `dt` computes its body at.
    #[must_use]
    pub fn of(dt: ElementKind) -> Self {
        match dt {
            ElementKind::F64 => Self::F64,
            // f32 and the narrow floats: the narrow ones are promoted to
            // `float` by the load and computed at f32, so their fold precision
            // is f32 too — NOT their storage precision.
            ElementKind::F32
            | ElementKind::F32Strict
            | ElementKind::F16
            | ElementKind::Bf16
            | ElementKind::Fp8E4M3FN
            | ElementKind::Fp8E5M2
            | ElementKind::Fp8E4M3FNUZ
            | ElementKind::Fp8E5M2FNUZ
            | ElementKind::F8E8M0
            | ElementKind::F8E6M2 => Self::F32,
            // Complex components are f32/f64 pairs; the wider component width
            // decides. Integers and Bool never reach a float fold — a `Const`
            // at an integer dtype is refused by the plan gate
            // (`assert_no_int_div_or_const`), so no rounding applies and F64
            // (the identity) is the honest answer rather than a guess.
            ElementKind::Complex64 => Self::F32,
            _ => Self::F64,
        }
    }

    /// Round a fold result to this precision — the bits the device would hold
    /// after that step.
    #[must_use]
    fn round(self, v: f64) -> f64 {
        match self {
            // `as f32` is round-to-nearest-even, which is what the device does.
            #[allow(clippy::cast_possible_truncation)]
            Self::F32 => f64::from(v as f32),
            Self::F64 => v,
        }
    }
}

/// Fold a unary op on a constant — algebraic ops only; transcendentals return
/// `None` (host-f64 vs device-f32 would diverge).
fn eval_unary(op: UnaryOp, v: f64) -> Option<f64> {
    // Never fold a NaN operand: the emitted `NAN` literal is the positive
    // canonical quiet NaN, which would drop an authored sign/payload the
    // runtime op preserves. Left symbolic, the device computes it faithfully.
    if v.is_nan() {
        return None;
    }
    Some(match op {
        UnaryOp::Neg => -v,
        UnaryOp::Abs => v.abs(),
        UnaryOp::Sqr => v * v,
        UnaryOp::Sqrt => v.sqrt(),
        // Rsqrt is NOT folded, and the reason is backend-neutral rather than
        // CUDA-specific. The neutral speller now emits portable `(1.0f/sqrtf(x))`
        // — so "the device computes rsqrtf" is no longer true of the default —
        // but a backend MAY legitimately spell the approximate hardware
        // intrinsic through its own `unary` seam (CUDA's `rsqrtf` is ~2 ulp;
        // HLSL/Slang `rsqrt` likewise). This pass runs on the IR, BEFORE a
        // backend is chosen, so it cannot know which spelling will be used.
        // Folding to an exact host `1/sqrt(v)` would therefore change the
        // emitted bits for some backends and not others.
        //
        // Read the old reason carefully before relaxing this: it said the
        // device emits `rsqrtf`, which stopped being true when the non-portable
        // default was fixed. The DECISION survived that change; the
        // justification did not.
        UnaryOp::Recip => 1.0 / v,
        UnaryOp::Relu => {
            if v < 0.0 {
                0.0
            } else {
                v
            }
        }
        UnaryOp::Floor => v.floor(),
        UnaryOp::Ceil => v.ceil(),
        UnaryOp::Round => v.round_ties_even(),
        // Trunc is exact on FINITE values only by policy: NaN is already guarded
        // above, and ±Inf stays symbolic too (house lesson: fold nothing
        // non-finite — the emitted INFINITY literal round-trip is not worth the
        // risk surface). Every other increment-0a fn is a device approximation
        // and is deliberately NOT folded (the Rsqrt-fold lesson).
        UnaryOp::Trunc if v.is_finite() => v.trunc(),
        UnaryOp::Sign => {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        UnaryOp::Step => {
            if v > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        // Sin/Cos/Rsqrt, the activations, and the whole increment-0a fn set
        // (Erfc…Lgamma): device-approximate — never folded.
        _ => return None,
    })
}

/// The exact reciprocal of `c` when `c` is a finite **normal power of two**
/// whose reciprocal is also finite and normal — the precondition under which
/// `x / c == x * (1/c)` bit-exactly (the reciprocal is exact, so the true
/// product equals the true quotient and both round identically). `None`
/// otherwise (non-pow2, zero, subnormal, or a reciprocal that would leave the
/// normal range).
fn exact_pow2_recip(c: f64) -> Option<f64> {
    const MANTISSA_MASK: u64 = (1u64 << 52) - 1;
    const EXP_MASK: u64 = 0x7ff;
    let is_normal_pow2 = |v: f64| {
        let bits = v.to_bits();
        let exp = (bits >> 52) & EXP_MASK;
        bits & MANTISSA_MASK == 0 && exp != 0 && exp != EXP_MASK
    };
    if !is_normal_pow2(c) {
        return None;
    }
    let r = 1.0 / c;
    is_normal_pow2(r).then_some(r)
}

/// Fold a non-infix binary op on two constants — `Max`/`Min` and integer-clean
/// `Rem`; `Pow` is skipped (host-f64 vs device-f32), `Rem` by zero is skipped.
/// The increment-0a binaries are ALL skipped: `Atan2` is approximate, and the
/// exact bit-level ops (`Copysign`/`Nextafter`/`FmaxIeee`/`FminIeee`/`RemTrunc`)
/// stay unfolded under the when-in-doubt-add-no-rule policy (`Nextafter` in
/// particular is dtype-lattice-dependent, so a host-f64 fold would be wrong).
/// The increment-0b `Cmp*` predicates are ALSO all skipped — even const-const:
/// a fold would need the full NaN gate (any cmp with NaN is false except
/// `CmpNe`), the host f64 compare of two demote-destined constants can disagree
/// with the device's dtype-width compare (two f64 constants that are distinct
/// on the host collapse to one f32/f16 value on the device, flipping
/// `==`/`!=`/`<`), and when-in-doubt-add-no-rule holds. Pinned by
/// `cmp_predicates_are_never_folded_or_rewritten`.
fn eval_binary(op: BinaryOp, x: f64, y: f64, compute: Compute) -> Option<f64> {
    // Max/Min only fold when neither operand is NaN — the kernel propagates NaN
    // (NaN-select), so folding a NaN operand away (host f64::max suppresses it)
    // would disagree with the device.
    Some(match op {
        BinaryOp::Max if !x.is_nan() && !y.is_nan() => x.max(y),
        BinaryOp::Min if !x.is_nan() && !y.is_nan() => x.min(y),
        // floored remainder (torch.remainder), matching the kernel — not `x % y`.
        // Gated finite-in AND finite-out: NaN operands must stay symbolic (the
        // fold would canonicalize the payload/sign the device would propagate),
        // and an overflowing (x/y).floor()*y can produce ±inf, whose `-INFINITY`
        // literal the headerless-nvrtc discipline forbids (see `ScalarExpr` docs).
        BinaryOp::Rem if x.is_finite() && y.is_finite() && y != 0.0 => {
            // Rounded at EVERY step, not merely on the result. Rem is the only
            // composite fold formula here — four operations — and the device
            // rounds after each of them. Rounding only the final value leaves
            // f64 intermediates inside the formula, which is the same bug
            // `Compute` exists to fix, one level down. Measured at f32:
            //
            //   x = -58.526611328125, y = 1.5263065099716187
            //     f64 throughout, round once : 0.9993425607681274
            //     device, rounding each step : 0.9993438720703125
            let q = compute.round(x / y);
            let r = compute.round(x - compute.round(compute.round(q.floor()) * y));
            if !r.is_finite() {
                return None;
            }
            r
        }
        _ => return None,
    })
}

/// One rewrite pass: recognize equivalent forms and `union` them in. Returns
/// whether anything merged.
fn rules(eg: &mut EGraph, compute: Compute) -> bool {
    let snapshot: Vec<ENode> = eg.class_nodes.values().flatten().cloned().collect();
    let mut changed = false;
    for node in snapshot {
        let nid = eg.add(node.clone());
        match node {
            ENode::Add(a, b) => {
                // x + (-0) -> x: bit-exact for EVERY x ((+0)+(-0) = +0,
                // (-0)+(-0) = -0). x + (+0) is NOT an identity: (-0)+(+0) = +0
                // under round-to-nearest, but the passthrough would keep -0.
                // Gate by BITS — `== Some(0.0)` would match -0.0 too.
                let neg_zero = Some((-0.0f64).to_bits());
                if eg.class_const(b).map(f64::to_bits) == neg_zero {
                    changed |= eg.union(nid, a);
                }
                if eg.class_const(a).map(f64::to_bits) == neg_zero {
                    changed |= eg.union(nid, b);
                }
                if let (Some(x), Some(y)) = (eg.class_const(a), eg.class_const(b)) {
                    if !x.is_nan() && !y.is_nan() {
                        let c = eg.add(ENode::Const(compute.round(x + y).to_bits()));
                        changed |= eg.union(nid, c);
                    }
                }
            }
            ENode::Sub(a, b) => {
                // x - (+0) -> x: bit-exact for EVERY x ((-0)-(+0) = -0,
                // (+0)-(+0) = +0). x - (-0) is NOT: (-0)-(-0) = +0, but the
                // passthrough would keep -0. Gate by bits.
                if eg.class_const(b).map(f64::to_bits) == Some(0.0f64.to_bits()) {
                    changed |= eg.union(nid, a);
                }
                if let (Some(x), Some(y)) = (eg.class_const(a), eg.class_const(b)) {
                    if !x.is_nan() && !y.is_nan() {
                        let c = eg.add(ENode::Const(compute.round(x - y).to_bits()));
                        changed |= eg.union(nid, c);
                    }
                }
            }
            ENode::Mul(a, b) => {
                if eg.class_const(b) == Some(1.0) {
                    changed |= eg.union(nid, a);
                }
                if eg.class_const(a) == Some(1.0) {
                    changed |= eg.union(nid, b);
                }
                // NOTE: `x * 0 -> 0` is deliberately ABSENT — it is not
                // value-preserving (NaN*0 = NaN, Inf*0 = NaN, and -x*0 = -0);
                // folding it would silently change the bits a kernel computes.
                // Two-const products fold below (0*0 included, exactly).
                if let (Some(x), Some(y)) = (eg.class_const(a), eg.class_const(b)) {
                    if !x.is_nan() && !y.is_nan() {
                        let c = eg.add(ENode::Const(compute.round(x * y).to_bits()));
                        changed |= eg.union(nid, c);
                    }
                }
            }
            ENode::Div(a, b) => {
                if eg.class_const(b) == Some(1.0) {
                    changed |= eg.union(nid, a);
                }
                // x / 2^k  ->  x * 2^-k: bit-exact (an exact power-of-two
                // reciprocal makes the true product equal the true quotient, so
                // both round identically — incl. NaN/Inf/±0 propagation), and
                // device FDIV is ~4x an FMUL (weights 8 vs 2 drive extraction).
                if let Some(c) = eg.class_const(b) {
                    // Gated on the reciprocal being EXACT at the compute
                    // precision, not rounded to it. This introduces a new
                    // constant rather than folding an existing one, and
                    // rounding could turn an exact reciprocal into something
                    // else entirely: `2^-200` is exact in f64 and flushes to
                    // zero in f32, which would rewrite `x / 2^200` into
                    // `x * 0`. Skipping the rewrite where it is not exact costs
                    // one division and cannot be wrong.
                    if let Some(r) = exact_pow2_recip(c) {
                        if compute.round(r) == r {
                            let rc = eg.add(ENode::Const(r.to_bits()));
                            let m = eg.add(ENode::Mul(a, rc));
                            changed |= eg.union(nid, m);
                        }
                    }
                }
                if let (Some(x), Some(y)) = (eg.class_const(a), eg.class_const(b)) {
                    if y != 0.0 && !x.is_nan() && !y.is_nan() {
                        let c = eg.add(ENode::Const(compute.round(x / y).to_bits()));
                        changed |= eg.union(nid, c);
                    }
                }
            }
            ENode::Unary(UnaryOp::Neg, x) => {
                // neg(neg(y)) -> y
                let xc = eg.find(x);
                let inner = eg.class_nodes.get(&xc).and_then(|ns| {
                    ns.iter().find_map(|n| match n {
                        ENode::Unary(UnaryOp::Neg, y) => Some(*y),
                        _ => None,
                    })
                });
                if let Some(y) = inner {
                    changed |= eg.union(nid, y);
                }
                if let Some(v) = eg.class_const(x) {
                    if !v.is_nan() {
                        let c = eg.add(ENode::Const(compute.round(-v).to_bits()));
                        changed |= eg.union(nid, c);
                    }
                }
            }
            ENode::Unary(op @ (UnaryOp::Abs | UnaryOp::Relu), x) => {
                // abs(abs(y)) -> abs(y) ; relu(relu(y)) -> relu(y): idempotent.
                // abs(neg(y)) -> abs(y): |-y| == |y| bit-exactly (abs clears the
                // sign bit either way; NaN payload untouched). relu(neg) is NOT
                // an identity — do not generalize.
                let xc = eg.find(x);
                let inner = eg.class_nodes.get(&xc).and_then(|ns| {
                    ns.iter().find_map(|n| match n {
                        ENode::Unary(i, y) if *i == op => Some((op, *y)),
                        ENode::Unary(UnaryOp::Neg, y) if op == UnaryOp::Abs => {
                            Some((UnaryOp::Abs, *y))
                        }
                        _ => None,
                    })
                });
                if let Some((outer, y)) = inner {
                    let collapsed = eg.add(ENode::Unary(outer, y));
                    changed |= eg.union(nid, collapsed);
                }
                if let Some(v) = eg.class_const(x) {
                    if let Some(r) = eval_unary(op, v) {
                        let c = eg.add(ENode::Const(compute.round(r).to_bits()));
                        changed |= eg.union(nid, c);
                    }
                }
            }
            ENode::Binary(op, a, b) => {
                // max(x, x) = x ; min(x, x) = x. STRICTLY Max/Min — never the
                // Cmp* predicates: CmpEq(x, x) is NOT 1.0 (it is FALSE for NaN
                // x) and CmpNe(x, x) is NOT 0.0 (TRUE for NaN x); the 0a review
                // proved a widened rule arm here can pass the suite, so the
                // non-rewrite is pinned per op in
                // `cmp_predicates_are_never_folded_or_rewritten`.
                if matches!(op, BinaryOp::Max | BinaryOp::Min) && eg.find(a) == eg.find(b) {
                    changed |= eg.union(nid, a);
                }
                if let (Some(x), Some(y)) = (eg.class_const(a), eg.class_const(b)) {
                    if let Some(r) = eval_binary(op, x, y, compute) {
                        let c = eg.add(ENode::Const(compute.round(r).to_bits()));
                        changed |= eg.union(nid, c);
                    }
                }
            }
            ENode::Unary(op, x) => {
                if let Some(v) = eg.class_const(x) {
                    if let Some(r) = eval_unary(op, v) {
                        let c = eg.add(ENode::Const(compute.round(r).to_bits()));
                        changed |= eg.union(nid, c);
                    }
                }
            }
            _ => {}
        }
    }
    changed
}

fn saturate(eg: &mut EGraph, max_iters: usize, compute: Compute) {
    for _ in 0..max_iters {
        let changed = rules(eg, compute);
        eg.rebuild_index();
        if !changed {
            break;
        }
    }
}

/// Relative op cost for extraction — division and transcendentals dominate.
fn weight(n: &ENode) -> u64 {
    match n {
        // Coord sits in the leaf tier: on the strided schedule the unraveled
        // c{d} already exists for the offset math, so reading it costs a cast.
        ENode::Input(_)
        | ENode::Param(_)
        | ENode::Const(_)
        | ENode::Reduced(_)
        | ENode::Coord(_) => 1,
        ENode::Add(..) | ENode::Sub(..) | ENode::Mul(..) => 2,
        ENode::Div(..) => 8,
        ENode::Binary(op, ..) => match op {
            // Copysign/Nextafter are bit-manipulation ops; FmaxIeee/FminIeee are
            // hardware min/max — all cheap, same tier as the Max/Min selects.
            // The Cmp* predicates (increment 0b) are compare-selects too: one
            // setp + one sel, the same tier as Max/Min. The increment-0c
            // bitwise/shift ops are single ALU instructions and the logical
            // ops a setp pair + sel — all the same compare-select tier.
            BinaryOp::Max
            | BinaryOp::Min
            | BinaryOp::Copysign
            | BinaryOp::Nextafter
            | BinaryOp::FmaxIeee
            | BinaryOp::FminIeee
            | BinaryOp::CmpEq
            | BinaryOp::CmpNe
            | BinaryOp::CmpLt
            | BinaryOp::CmpLe
            | BinaryOp::CmpGt
            | BinaryOp::CmpGe
            | BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor
            | BinaryOp::Shl
            | BinaryOp::Shr
            | BinaryOp::LogicalAnd
            | BinaryOp::LogicalOr
            | BinaryOp::LogicalXor => 2,
            BinaryOp::Rem | BinaryOp::RemTrunc => 8, // division-class
            BinaryOp::Pow | BinaryOp::Atan2 => 16,   // transcendental
        },
        // Select is compare-select class (one setp+selp) — the Max/Min/Cmp* tier.
        ENode::Select(..) => 2,
        ENode::Unary(op, _) => match op {
            UnaryOp::Neg | UnaryOp::Abs | UnaryOp::Relu => 1,
            UnaryOp::Sqr
            | UnaryOp::Floor
            | UnaryOp::Ceil
            | UnaryOp::Round
            | UnaryOp::Sign
            | UnaryOp::Step
            | UnaryOp::Trunc => 2,
            UnaryOp::Sqrt | UnaryOp::Rsqrt | UnaryOp::Recip => 8,
            UnaryOp::Exp
            | UnaryOp::Log
            | UnaryOp::Tanh
            | UnaryOp::Sigmoid
            | UnaryOp::Erf
            | UnaryOp::Gelu
            | UnaryOp::Silu
            | UnaryOp::Sin
            | UnaryOp::Cos
            | UnaryOp::Erfc
            | UnaryOp::Exp2
            | UnaryOp::Expm1
            | UnaryOp::Log2
            | UnaryOp::Log10
            | UnaryOp::Log1p
            | UnaryOp::Sinh
            | UnaryOp::Cosh
            | UnaryOp::Tan
            | UnaryOp::Asin
            | UnaryOp::Acos
            | UnaryOp::Atan
            | UnaryOp::Asinh
            | UnaryOp::Acosh
            | UnaryOp::Atanh
            | UnaryOp::Cbrt
            | UnaryOp::Lgamma => 16,
        },
    }
}

fn children(n: &ENode) -> Vec<Id> {
    match *n {
        // The first 3-child arm — before the 2-child or-pattern so a future
        // reorder can't shadow it.
        ENode::Select(c, a, b) => vec![c, a, b],
        ENode::Add(a, b)
        | ENode::Sub(a, b)
        | ENode::Mul(a, b)
        | ENode::Div(a, b)
        | ENode::Binary(_, a, b) => vec![a, b],
        ENode::Unary(_, a) => vec![a],
        _ => vec![],
    }
}

/// Total cost of an e-node given the best costs of its children, or `None` if a
/// child has no cost yet.
fn enode_cost(eg: &EGraph, n: &ENode, best: &HashMap<Id, (u64, ENode)>) -> Option<u64> {
    let mut sum = weight(n);
    for k in children(n) {
        sum = sum.saturating_add(best.get(&eg.find_imm(k))?.0);
    }
    Some(sum)
}

/// Extract the lowest-cost equivalent expression for `root` (equality-saturation
/// extraction: relax per-class min costs to a fixpoint, then reconstruct).
fn extract(eg: &EGraph, root: Id) -> ScalarExpr {
    let mut best: HashMap<Id, (u64, ENode)> = HashMap::new();
    loop {
        let mut changed = false;
        for (&c, nodes) in &eg.class_nodes {
            let rc = eg.find_imm(c);
            for n in nodes {
                if let Some(k) = enode_cost(eg, n, &best) {
                    if best.get(&rc).is_none_or(|(bk, _)| k < *bk) {
                        best.insert(rc, (k, n.clone()));
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    build(eg, eg.find_imm(root), &best)
}

fn build(eg: &EGraph, c: Id, best: &HashMap<Id, (u64, ENode)>) -> ScalarExpr {
    let n = best[&eg.find_imm(c)].1.clone();
    match n {
        ENode::Input(i) => ScalarExpr::Input(i),
        ENode::Const(bits) => ScalarExpr::Const(f64::from_bits(bits)),
        ENode::Param(i) => ScalarExpr::Param(i),
        ENode::Reduced(i) => ScalarExpr::Reduced(i),
        ENode::Coord(d) => ScalarExpr::Coord(d),
        ENode::Add(a, b) => ScalarExpr::Add(bx(build(eg, a, best)), bx(build(eg, b, best))),
        ENode::Sub(a, b) => ScalarExpr::Sub(bx(build(eg, a, best)), bx(build(eg, b, best))),
        ENode::Mul(a, b) => ScalarExpr::Mul(bx(build(eg, a, best)), bx(build(eg, b, best))),
        ENode::Div(a, b) => ScalarExpr::Div(bx(build(eg, a, best)), bx(build(eg, b, best))),
        ENode::Binary(op, a, b) => {
            ScalarExpr::Binary(op, bx(build(eg, a, best)), bx(build(eg, b, best)))
        }
        ENode::Unary(op, x) => ScalarExpr::Unary(op, bx(build(eg, x, best))),
        ENode::Select(c, a, b) => ScalarExpr::Select(
            bx(build(eg, c, best)),
            bx(build(eg, a, best)),
            bx(build(eg, b, best)),
        ),
    }
}

fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

/// Total cost of a `ScalarExpr` under the same per-node [`weight`] model the
/// e-graph extraction uses (recursive sum of `weight` over the tree), so it is
/// directly comparable to a `KCand.cost`. Used by [`optimize_top_k`] to keep the
/// returned list cost-ascending.
fn expr_cost(e: &ScalarExpr) -> u64 {
    let (node, kids): (ENode, Vec<&ScalarExpr>) = match e {
        ScalarExpr::Input(i) => (ENode::Input(*i), vec![]),
        ScalarExpr::Const(v) => (ENode::Const(v.to_bits()), vec![]),
        ScalarExpr::Param(i) => (ENode::Param(*i), vec![]),
        ScalarExpr::Reduced(i) => (ENode::Reduced(*i), vec![]),
        ScalarExpr::Coord(d) => (ENode::Coord(*d), vec![]),
        ScalarExpr::Add(a, b) => (ENode::Add(0, 0), vec![a, b]),
        ScalarExpr::Sub(a, b) => (ENode::Sub(0, 0), vec![a, b]),
        ScalarExpr::Mul(a, b) => (ENode::Mul(0, 0), vec![a, b]),
        ScalarExpr::Div(a, b) => (ENode::Div(0, 0), vec![a, b]),
        ScalarExpr::Binary(op, a, b) => (ENode::Binary(*op, 0, 0), vec![a, b]),
        ScalarExpr::Unary(op, a) => (ENode::Unary(*op, 0), vec![a]),
        ScalarExpr::Select(c, a, b) => (ENode::Select(0, 0, 0), vec![c, a, b]),
    };
    kids.into_iter()
        .fold(weight(&node), |acc, k| acc.saturating_add(expr_cost(k)))
}

/// Algebraically simplify an op body to the lowest-cost equivalent form via
/// equality saturation. Semantics-preserving within the precision-safe rule set
/// (see the module scope note). Pure:
/// `optimize(optimize(e, d), d) == optimize(e, d)`.
///
/// `dtype` is the kernel's compute dtype, and it is **required rather than
/// defaulted**. Constant folding must land on the bits the device would produce,
/// and that depends on the precision the device computes at — see [`Compute`]
/// for the measured 1-ULP divergence this closes. A defaulted `f64` would
/// silently reintroduce it for every `f32` kernel, which is exactly the bug,
/// so the caller states the dtype.
#[must_use]
pub fn optimize(e: &ScalarExpr, dtype: ElementKind) -> ScalarExpr {
    let compute = Compute::of(dtype);
    let mut eg = EGraph::default();
    let root = add_expr(&mut eg, e, compute);
    saturate(&mut eg, 32, compute);
    extract(&eg, root)
}

// ===========================================================================
// k-best extraction (item 08 — telemetry variant surface)
// ===========================================================================
//
// `optimize_top_k` extracts the *k* lowest-cost equivalent forms of a body
// instead of the single best. It rides the SAME saturated e-graph, cost model
// (`weight`/`children`), and rewrite set as `optimize` — no new IR node, no new
// rule. The invariant that makes it safe to drop into JIT synthesis: `form[0]`
// is bit-identical to `optimize(e)` (so the k==1 case *is* the shipped
// optimizer, and every JIT-selected default form is unchanged).

/// One reconstructed candidate form for an e-class, with its extraction cost.
#[derive(Clone)]
struct KCand {
    /// Total extraction cost (`weight` summed over the reconstructed tree).
    cost: u64,
    /// The reconstructed expression for this candidate.
    expr: ScalarExpr,
}

/// Deterministic total order over reconstructed forms — the tie-break that keeps
/// `optimize_top_k` output stable when two candidates share a cost. Orders by
/// node shape, then children, then leaf payload (`Const` by bits so it is
/// NaN-stable), so it never depends on `HashMap` iteration order.
fn expr_cmp(a: &ScalarExpr, b: &ScalarExpr) -> std::cmp::Ordering {
    fn rank(e: &ScalarExpr) -> u8 {
        match e {
            ScalarExpr::Input(_) => 0,
            ScalarExpr::Const(_) => 1,
            ScalarExpr::Param(_) => 2,
            ScalarExpr::Reduced(_) => 3,
            ScalarExpr::Coord(_) => 4,
            ScalarExpr::Add(..) => 5,
            ScalarExpr::Sub(..) => 6,
            ScalarExpr::Mul(..) => 7,
            ScalarExpr::Div(..) => 8,
            ScalarExpr::Binary(..) => 9,
            ScalarExpr::Unary(..) => 10,
            ScalarExpr::Select(..) => 11,
        }
    }
    match (a, b) {
        (ScalarExpr::Input(x), ScalarExpr::Input(y))
        | (ScalarExpr::Param(x), ScalarExpr::Param(y))
        | (ScalarExpr::Reduced(x), ScalarExpr::Reduced(y))
        | (ScalarExpr::Coord(x), ScalarExpr::Coord(y)) => x.cmp(y),
        (ScalarExpr::Const(x), ScalarExpr::Const(y)) => x.to_bits().cmp(&y.to_bits()),
        (ScalarExpr::Add(a1, a2), ScalarExpr::Add(b1, b2))
        | (ScalarExpr::Sub(a1, a2), ScalarExpr::Sub(b1, b2))
        | (ScalarExpr::Mul(a1, a2), ScalarExpr::Mul(b1, b2))
        | (ScalarExpr::Div(a1, a2), ScalarExpr::Div(b1, b2)) => {
            expr_cmp(a1, b1).then_with(|| expr_cmp(a2, b2))
        }
        (ScalarExpr::Binary(o1, a1, a2), ScalarExpr::Binary(o2, b1, b2)) => format!("{o1:?}")
            .cmp(&format!("{o2:?}"))
            .then_with(|| expr_cmp(a1, b1))
            .then_with(|| expr_cmp(a2, b2)),
        (ScalarExpr::Unary(o1, x), ScalarExpr::Unary(o2, y)) => format!("{o1:?}")
            .cmp(&format!("{o2:?}"))
            .then_with(|| expr_cmp(x, y)),
        (ScalarExpr::Select(c1, a1, b1), ScalarExpr::Select(c2, a2, b2)) => expr_cmp(c1, c2)
            .then_with(|| expr_cmp(a1, a2))
            .then_with(|| expr_cmp(b1, b2)),
        _ => rank(a).cmp(&rank(b)),
    }
}

/// Reconstruct the [`ScalarExpr`] for an e-node given the already-built child
/// forms (in [`children`] order) — the k-best analogue of [`build`], but taking
/// explicit child forms rather than recursing through a single-best table.
fn reconstruct(n: &ENode, ch: Vec<ScalarExpr>) -> ScalarExpr {
    let mut it = ch.into_iter();
    let mut pop = || it.next().expect("reconstruct: child arity mismatch");
    match *n {
        ENode::Input(i) => ScalarExpr::Input(i),
        ENode::Const(bits) => ScalarExpr::Const(f64::from_bits(bits)),
        ENode::Param(i) => ScalarExpr::Param(i),
        ENode::Reduced(i) => ScalarExpr::Reduced(i),
        ENode::Coord(d) => ScalarExpr::Coord(d),
        ENode::Add(..) => ScalarExpr::Add(bx(pop()), bx(pop())),
        ENode::Sub(..) => ScalarExpr::Sub(bx(pop()), bx(pop())),
        ENode::Mul(..) => ScalarExpr::Mul(bx(pop()), bx(pop())),
        ENode::Div(..) => ScalarExpr::Div(bx(pop()), bx(pop())),
        ENode::Binary(op, ..) => ScalarExpr::Binary(op, bx(pop()), bx(pop())),
        ENode::Unary(op, ..) => ScalarExpr::Unary(op, bx(pop())),
        ENode::Select(..) => ScalarExpr::Select(bx(pop()), bx(pop()), bx(pop())),
    }
}

/// All index combinations across the child candidate lists (the bounded Lawler
/// product). Each child list has ≤ `k` entries and there are ≤ 3 children, so the
/// product is ≤ `k³` — bounded, never combinatorial blow-up.
fn cartesian(lists: &[Vec<KCand>]) -> Vec<Vec<KCand>> {
    let mut acc: Vec<Vec<KCand>> = vec![Vec::new()];
    for list in lists {
        let mut next = Vec::with_capacity(acc.len() * list.len());
        for prefix in &acc {
            for item in list {
                let mut row = prefix.clone();
                row.push(item.clone());
                next.push(row);
            }
        }
        acc = next;
    }
    acc
}

/// `true` if two candidate lists carry the same forms in the same order — the
/// fixpoint-convergence test (compare by cost + structure, not `HashMap` order).
fn kcands_equal(a: &[KCand], b: &[KCand]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.cost == y.cost && x.expr == y.expr)
}

/// Bottom-up bounded k-best table over the saturated e-graph: for every e-class,
/// the ≤ `k` lowest-cost, structurally-distinct reconstructed forms, cheapest
/// first. Relaxed to a fixpoint (a class's list only ever gains a cheaper /
/// newly-reachable form), with an **explicit iteration cap as the cycle guard**:
/// an e-graph can be cyclic (e.g. `neg(neg x)` unions the outer class with `x`'s,
/// making a class reference itself), and the cap plus the ≤ `k` truncation keeps
/// a self-referential class from spinning — deeper trees cost strictly more, so
/// they never displace the bounded cheapest set. `debug_assert` pins that real
/// bodies converge *before* the cap (the termination proof the tests exercise).
fn kbest_table(eg: &EGraph, k: usize) -> HashMap<Id, Vec<KCand>> {
    let mut classes: Vec<Id> = eg.class_nodes.keys().copied().collect();
    classes.sort_unstable();
    let mut table: HashMap<Id, Vec<KCand>> = HashMap::new();
    let max_iters = classes.len().saturating_mul(k + 2).saturating_add(8);
    let mut converged = false;
    for _ in 0..max_iters {
        let mut changed = false;
        for &c in &classes {
            let Some(nodes) = eg.class_nodes.get(&c) else {
                continue;
            };
            let mut merged: Vec<KCand> = table.get(&c).cloned().unwrap_or_default();
            let before = merged.clone();
            for n in nodes {
                let kids: Vec<Id> = children(n).iter().map(|&id| eg.find_imm(id)).collect();
                // Every child must already have at least one form this round; a
                // not-yet-populated child (or a cyclic self-reference on the
                // first pass) simply defers this e-node to a later iteration.
                let mut lists: Vec<Vec<KCand>> = Vec::with_capacity(kids.len());
                let mut ready = true;
                for &kid in &kids {
                    match table.get(&kid) {
                        Some(l) if !l.is_empty() => lists.push(l.clone()),
                        _ => {
                            ready = false;
                            break;
                        }
                    }
                }
                if !ready {
                    continue;
                }
                for combo in cartesian(&lists) {
                    let mut cost = weight(n);
                    let mut child_exprs = Vec::with_capacity(combo.len());
                    for cand in &combo {
                        cost = cost.saturating_add(cand.cost);
                        child_exprs.push(cand.expr.clone());
                    }
                    merged.push(KCand {
                        cost,
                        expr: reconstruct(n, child_exprs),
                    });
                }
            }
            // Cheapest first, deterministic tie-break, structurally distinct,
            // capped at k.
            merged.sort_by(|a, b| a.cost.cmp(&b.cost).then_with(|| expr_cmp(&a.expr, &b.expr)));
            merged.dedup_by(|a, b| a.expr == b.expr);
            merged.truncate(k);
            if !kcands_equal(&before, &merged) {
                table.insert(c, merged);
                changed = true;
            }
        }
        if !changed {
            converged = true;
            break;
        }
    }
    debug_assert!(
        converged,
        "kbest_table did not converge within {max_iters} iterations (cycle guard tripped)"
    );
    table
}

/// Extract up to `k` distinct lowest-cost equivalent forms of `e`, cheapest
/// first — the k-best generalization of [`optimize`] that item 08 uses to ship a
/// ranked *variant* set (each form is a bit-preserving equivalent under the same
/// precision-safe rule set; Fuel's telemetry then picks among them per cell).
///
/// # Invariant
///
/// `optimize_top_k(e, 1) == [optimize(e)]` and, for every `k ≥ 1`, `form[0]` is
/// **bit-identical** to [`optimize`]`(e)` — guaranteed *by construction* here
/// (`form[0]` is literally `optimize(e)`), never inferred from a second
/// reconstruction path. This is the load-bearing pin: a k-best that silently
/// changed the k==1 winner would alter every JIT-selected default form.
///
/// `form[1..]` are the next-cheapest structurally-distinct equivalents drawn from
/// the `kbest_table`, deduped against `form[0]` and never cheaper than it, so
/// the whole list is in non-decreasing cost order — this holds even on a
/// pathological cyclic body where `optimize` returns a non-minimal head (see the
/// body comment). [`ScalarExpr::Reduced`] is an opaque leaf with no rule, so a per-row reduced
/// scalar is never cross-folded across forms — the same guarantee `optimize`
/// carries. Termination is guaranteed by the k-best cycle guard.
#[must_use]
pub fn optimize_top_k(e: &ScalarExpr, k: usize, dtype: ElementKind) -> Vec<ScalarExpr> {
    if k == 0 {
        return Vec::new();
    }
    // form[0] IS optimize(e, dtype): the invariant is a construction, not a hope.
    let head = optimize(e, dtype);
    if k == 1 {
        return vec![head];
    }
    // The SAME compute precision as the head, necessarily: a k-best that
    // ingested or folded at a different precision could return a `form[1..]`
    // that is not bit-equivalent to `form[0]`, which is the one invariant this
    // function exists to hold.
    let compute = Compute::of(dtype);
    let mut eg = EGraph::default();
    let root = add_expr(&mut eg, e, compute);
    saturate(&mut eg, 32, compute);
    let root = eg.find_imm(root);
    let table = kbest_table(&eg, k);

    // Keep the list cost-ascending BY CONSTRUCTION even if `optimize` returned a
    // non-minimal head on a pathological cyclic body (a deeply-nested involution
    // chain like neg⁴(x), whose saturation/extract fixpoint is HashMap-order
    // sensitive — no real op body reaches it): never offer an alternative cheaper
    // than form[0]. For a real (acyclic) body the head IS the minimum, so every
    // candidate is `>= head_cost` and NOTHING is dropped — byte-identical output.
    let head_cost = expr_cost(&head);
    let mut forms = vec![head];
    if let Some(cands) = table.get(&root) {
        for cand in cands {
            if forms.len() >= k {
                break;
            }
            if expr_cost(&cand.expr) >= head_cost && forms.iter().all(|f| *f != cand.expr) {
                forms.push(cand.expr.clone());
            }
        }
    }
    forms
}

#[cfg(test)]
mod tests {
    use unpopped_vocab::ElementKind;
    use unpopped_vocab::ElementKind::F32;

    /// **A CHAINED fold lands on the device's bits, not the host's.**
    ///
    /// This is the bug that made `Compute` necessary, pinned with the constant
    /// that exposed it. A single fold in `f64` is provably safe for
    /// `f32` operands (53 bits ≥ 2·24+2, the innocuous-double-rounding bound),
    /// so nothing shorter than a chain can catch this — which is why it survived.
    ///
    /// `sqr(sqr(c))` at `f32`, with `c = -2.7932066917419434`:
    ///
    /// ```text
    /// fold at f64 throughout, round once : 60.87126159667969   bits 0x42737c2c
    /// device, rounding after each step   : 60.87126541137695   bits 0x42737c2d
    /// ```
    ///
    /// One ULP. The module's bit-preservation contract says every rewrite
    /// preserves the device result **bits**; before this, folding violated its
    /// own contract for any `f32` kernel whose body chained constant arithmetic.
    #[test]
    fn a_chained_fold_rounds_at_each_step_like_the_device() {
        let c = -2.793_206_691_741_943_4_f64;
        let sqr = |e| ScalarExpr::Unary(UnaryOp::Sqr, Box::new(e));
        let body = sqr(sqr(ScalarExpr::Const(c)));

        // What the device computes, written out step by step in f32.
        #[allow(clippy::cast_possible_truncation)]
        let inner = (c as f32) * (c as f32);
        let device = f64::from(inner * inner);

        let ScalarExpr::Const(got) = optimize(&body, F32) else {
            panic!("a fully-constant body must fold to a Const");
        };
        assert_eq!(
            got.to_bits(),
            device.to_bits(),
            "folded {got:?} (bits {:#018x}) but the device computes {device:?}              (bits {:#018x}) — the fold kept an f64 intermediate the device rounds",
            got.to_bits(),
            device.to_bits()
        );

        // The negative control: folding the SAME body at f64 must give the other
        // answer. Without it, this test passes for a `Compute` that ignores its
        // argument and rounds everything to f32 unconditionally.
        let ScalarExpr::Const(at_f64) = optimize(&body, ElementKind::F64) else {
            panic!("must fold");
        };
        assert_ne!(
            at_f64.to_bits(),
            got.to_bits(),
            "f32 and f64 folding must differ on this body, or the dtype argument              is not reaching the fold"
        );
        assert_eq!(at_f64, (c * c) * (c * c), "f64 folding is unrounded");
    }

    /// **A chained ARITHMETIC fold rounds at each step too.**
    ///
    /// `Add`/`Sub`/`Mul`/`Div` fold in their own match arms, separate from the
    /// `Unary`/`Binary` ones, and they were missed by the first pass of this
    /// fix — I rounded three fold sites and there are seven. Mutation testing
    /// found it: killing the rounding at one site left the suite green, which
    /// said the site was untested, and chasing that turned up four more folds
    /// with no rounding at all. These are the *most common* folds.
    ///
    /// `(a + b) + c` at `f32`, measured:
    ///
    /// ```text
    /// a = -5.045434474945068, b = 0.190538227558136, c = 2.0781235694885254
    ///   fold at f64 throughout  : -2.7767727375030518
    ///   device, stepwise in f32 : -2.7767724990844727
    /// ```
    #[test]
    fn a_chained_arithmetic_fold_rounds_at_each_step() {
        let (a, b, c) = (
            -5.045_434_474_945_068_f64,
            0.190_538_227_558_136_f64,
            2.078_123_569_488_525_4_f64,
        );
        let k = ScalarExpr::Const;
        let body = ScalarExpr::Add(
            Box::new(ScalarExpr::Add(Box::new(k(a)), Box::new(k(b)))),
            Box::new(k(c)),
        );

        #[allow(clippy::cast_possible_truncation)]
        let device = f64::from((a as f32 + b as f32) + c as f32);
        let ScalarExpr::Const(got) = optimize(&body, F32) else {
            panic!("a fully-constant body must fold");
        };
        assert_eq!(
            got.to_bits(),
            device.to_bits(),
            "folded {got:?}, device computes {device:?}"
        );
    }

    /// **Constants are rounded AT INGEST**, before any folding happens.
    ///
    /// A body carrying `Const(1.1)` at an `f32` kernel computes with
    /// `(float)1.1` on device. Folding against the `f64` `1.1` diverges with no
    /// chaining involved at all — one operation is enough, because the operands
    /// themselves were never values the device could hold.
    ///
    /// ```text
    /// 1.1 + 2.2 at f32:
    ///   f64 operands, round result : 3.299999952316284
    ///   f32 operands (the device)  : 3.3000001907348633
    /// ```
    ///
    /// Emission is unaffected: `const_lit` spells the rounded value, which
    /// converts to the same `float`.
    #[test]
    fn constants_are_rounded_at_ingest_not_only_at_fold() {
        let (a, b) = (1.1_f64, 2.2_f64);
        let body = ScalarExpr::Add(
            Box::new(ScalarExpr::Const(a)),
            Box::new(ScalarExpr::Const(b)),
        );
        #[allow(clippy::cast_possible_truncation)]
        let device = f64::from(a as f32 + b as f32);
        let ScalarExpr::Const(got) = optimize(&body, F32) else {
            panic!("must fold");
        };
        assert_eq!(
            got.to_bits(),
            device.to_bits(),
            "ingest rounding is missing"
        );

        // A single un-chained Const must round too, even with nothing to fold.
        let ScalarExpr::Const(lone) = optimize(&ScalarExpr::Const(0.1), F32) else {
            panic!("a Const stays a Const");
        };
        #[allow(clippy::cast_possible_truncation)]
        let want = f64::from(0.1_f64 as f32);
        assert_eq!(lone.to_bits(), want.to_bits());
    }

    /// **Every fold path lands on the device's bits** — one case per arm.
    ///
    /// # Why a table and not one test per op
    ///
    /// Mutation testing drove this. Neutralising the rounding at each
    /// `compute.round` site individually showed **six of ten survived**: the
    /// code was correct everywhere, but only four paths had a test that could
    /// tell. Six near-identical one-off tests would fix the count and rot
    /// independently; a table keeps the fold set and its coverage in one place,
    /// so a rule added to `rules` without a row here is visibly missing one.
    ///
    /// Each row carries constants **searched for divergence** — values where
    /// folding in `f64` and rounding once genuinely differs from rounding after
    /// every step. A row whose two answers agreed would pass without testing
    /// anything, so the control below asserts each case is discriminating.
    #[test]
    fn every_fold_path_rounds_like_the_device() {
        let k = |v: f64| Box::new(ScalarExpr::Const(v));
        let add = |a, b| ScalarExpr::Add(a, b);
        let sub = |a, b| ScalarExpr::Sub(a, b);
        let mul = |a, b| ScalarExpr::Mul(a, b);
        let div = |a, b| ScalarExpr::Div(a, b);
        let neg = |a| ScalarExpr::Unary(UnaryOp::Neg, a);
        let sqr = |a| ScalarExpr::Unary(UnaryOp::Sqr, a);

        #[allow(clippy::cast_possible_truncation)]
        let g = |v: f64| v as f32;

        // (name, body, the device's stepwise f32 answer)
        let cases: Vec<(&str, ScalarExpr, f32)> = vec![
            {
                let (a, b, c) = (
                    -5.045_434_474_945_068,
                    0.190_538_227_558_136,
                    2.078_123_569_488_525_4,
                );
                (
                    "Add",
                    add(Box::new(add(k(a), k(b))), k(c)),
                    (g(a) + g(b)) + g(c),
                )
            },
            {
                // Chosen with f32-REPRESENTABLE inputs. My first attempt searched
                // with raw f64 operands and did not discriminate: ingest rounds
                // both sides to f32 before anything folds, so what has to differ
                // is the INTERMEDIATE rounding, not the inputs.
                let (a, b) = (-1.337_258_219_718_933, 2.801_646_947_860_717_8);
                (
                    "Sub",
                    sub(Box::new(sub(k(a), k(b))), k(b)),
                    (g(a) - g(b)) - g(b),
                )
            },
            {
                let (a, b) = (3.098_762_955_441_808, 1.093_194_995_175_810_6);
                (
                    "Mul",
                    mul(Box::new(mul(k(a), k(b))), k(b)),
                    (g(a) * g(b)) * g(b),
                )
            },
            {
                let (a, b) = (1.075_022_503_143_812_8, 1.536_825_565_028_883);
                (
                    "Div",
                    div(Box::new(div(k(a), k(b))), k(b)),
                    (g(a) / g(b)) / g(b),
                )
            },
            {
                // Neg's own rounding is UNOBSERVABLE and this row does not try
                // to observe it: negation is exact in every precision, so
                // `round(-v) == -round(v)` always. A search over 3M f32 pairs
                // for a `-(a*b)` divergence found none, and could not — the
                // only rounding this row can detect is the inner `Mul`'s.
                // Kept because the composition is worth covering; not counted
                // as proof of the Neg site.
                let (a, b) = (3.479_583_740_234_375, 3.014_609_813_690_185_5);
                (
                    "Neg∘Mul",
                    neg(Box::new(mul(Box::new(mul(k(a), k(b))), k(b)))),
                    -((g(a) * g(b)) * g(b)),
                )
            },
            {
                let c = -2.793_206_691_741_943_4;
                (
                    "Sqr",
                    sqr(Box::new(sqr(k(c)))),
                    (g(c) * g(c)) * (g(c) * g(c)),
                )
            },
        ];

        for (name, body, device) in cases {
            let ScalarExpr::Const(got) = optimize(&body, F32) else {
                panic!("{name}: a fully-constant body must fold to a Const");
            };
            assert_eq!(
                got.to_bits(),
                f64::from(device).to_bits(),
                "{name}: folded {got:?}, device computes {device:?}"
            );

            // Control: this case must actually DISCRIMINATE. Folding the same
            // body at f64 has to give a different answer, or the row proves
            // nothing about rounding and is dead weight that reads as coverage.
            let ScalarExpr::Const(at_f64) = optimize(&body, ElementKind::F64) else {
                panic!("{name}: must fold at f64 too");
            };
            assert_ne!(
                f64::from(got as f32).to_bits(),
                f64::from(at_f64 as f32).to_bits(),
                "{name}: f32 and f64 folding agree on these constants — the row                  cannot detect a missing round and needs different values"
            );
        }
    }

    /// **The `x / 2^k -> x * 2^-k` rewrite is skipped when the reciprocal is not
    /// exact at the compute precision.**
    ///
    /// The rewrite is bit-exact *only* because an exact power-of-two reciprocal
    /// makes the true product equal the true quotient. At `f32` that stops being
    /// true when the reciprocal underflows: `2^-200` is a fine `f64` and is
    /// **0.0** as an `f32`, so the rewrite would turn `x / 2^200` into `x * 0`.
    /// Gated on exactness rather than rounded — skipping costs one division and
    /// cannot be wrong.
    #[test]
    fn the_pow2_reciprocal_rewrite_is_skipped_when_inexact_at_f32() {
        let big = 2.0_f64.powi(200);
        let body = ScalarExpr::Div(
            Box::new(ScalarExpr::Input(0)),
            Box::new(ScalarExpr::Const(big)),
        );
        // At f32 the divisor itself is +inf and the reciprocal is 0, so the
        // rewrite must not fire — the body keeps its Div.
        let got = optimize(&body, F32);
        assert!(
            matches!(got, ScalarExpr::Div(..)),
            "at f32 the reciprocal underflows to 0; the rewrite must be skipped,              got {got:?}"
        );

        // Control: at f64 the reciprocal IS exact, so the rewrite fires. Without
        // this the test passes for a rewrite that never fires at all.
        let at_f64 = optimize(&body, ElementKind::F64);
        assert!(
            matches!(at_f64, ScalarExpr::Mul(..)),
            "at f64 2^-200 is exact and the rewrite must fire, got {at_f64:?}"
        );
    }

    /// The `Binary` fold arm rounds too — `Rem` is the arm's only rounding op.
    ///
    /// `Max`/`Min` return one of their operands unchanged, so their rounding is
    /// unobservable. `Rem` is `x - (x/y).floor()*y`, which genuinely rounds.
    #[test]
    fn the_binary_fold_arm_rounds_via_rem() {
        let (x, y) = (-58.526_611_328_125, 1.526_306_509_971_618_7);
        let body = ScalarExpr::Binary(
            BinaryOp::Rem,
            Box::new(ScalarExpr::Const(x)),
            Box::new(ScalarExpr::Const(y)),
        );
        let ScalarExpr::Const(got) = optimize(&body, F32) else {
            panic!("a constant Rem must fold");
        };
        let ScalarExpr::Const(at_f64) = optimize(&body, ElementKind::F64) else {
            panic!("must fold");
        };
        assert_ne!(
            got.to_bits(),
            at_f64.to_bits(),
            "f32 and f64 Rem folding must differ here, or the case is not              discriminating"
        );
        #[allow(clippy::cast_possible_truncation)]
        let rounded = f64::from(got as f32);
        assert_eq!(
            got.to_bits(),
            rounded.to_bits(),
            "the f32 fold result must be an f32-representable value"
        );
    }

    /// **Some rounding sites are unobservable by construction; others are real
    /// coverage debt. This test records which, and the count is dated.**
    ///
    /// Mutation testing neutralises each `compute.round` call individually.
    /// Measured **2026-08-14: 7 of 14 sites caught.** That number is a
    /// measurement, not a target — re-run the harness rather than quoting it,
    /// because it moved three times while this fix was being written (3 sites,
    /// then 10, then 14 as each round of mutation testing found folds the
    /// previous round had missed).
    ///
    /// The survivors split into two kinds, and only one is debt:
    ///
    /// * **Unobservable** — the `Neg` fold and the involution arm (`Abs`/`Neg`).
    ///   **No test could catch these**:
    ///   negation and absolute value are exact in every floating-point
    ///   precision, so `round(-v) == -round(v)` and `round(|v|) == |round(v)|`
    ///   identically. A
    ///   3-million-pair search for a `-(a*b)` divergence at `f32` found none.
    ///   The rounding stays anyway — it costs nothing, and the rule "every fold
    ///   result is rounded to the compute precision" then holds with no
    ///   exception a future reader has to re-derive.
    /// * **Debt** — several of `Rem`'s internal steps, and a unary fold arm.
    ///   These *are* observable; the cases here simply do not discriminate them
    ///   yet. Closing them means searching for constants that isolate one
    ///   internal rounding at a time, the way the `Sub` row had to be re-searched
    ///   once ingest rounding made the first attempt non-discriminating.
    ///
    /// Recorded rather than left implicit, because a survivor list with no
    /// reasons reads as "7 bugs" to the next reader and as "all fine" to the one
    /// after that.
    #[test]
    fn neg_and_abs_are_exact_so_their_rounding_cannot_be_observed() {
        for v in [
            0.1_f64,
            -1.1,
            core::f64::consts::PI,
            1e-30,
            -7.777_777_777_777_777,
        ] {
            #[allow(clippy::cast_possible_truncation)]
            let r = f64::from(v as f32);
            #[allow(clippy::cast_possible_truncation)]
            let neg_then_round = f64::from((-v) as f32);
            assert_eq!(neg_then_round.to_bits(), (-r).to_bits(), "neg is exact");
            #[allow(clippy::cast_possible_truncation)]
            let abs_then_round = f64::from(v.abs() as f32);
            assert_eq!(abs_then_round.to_bits(), r.abs().to_bits(), "abs is exact");
        }
    }

    /// Every narrow float folds at `f32`, because that is what the emitters
    /// compute at — not at its own storage precision.
    ///
    /// `cfamily`'s narrow-float seam promotes an `f16`/`bf16`/FP8 load to
    /// `float` and computes there. Folding at the storage precision would round
    /// harder than the device does and produce a constant the kernel never
    /// would.
    #[test]
    fn narrow_floats_fold_at_the_precision_they_compute_at() {
        for dt in [
            ElementKind::F16,
            ElementKind::Bf16,
            ElementKind::Fp8E4M3FN,
            ElementKind::Fp8E5M2,
        ] {
            assert_eq!(
                Compute::of(dt),
                Compute::F32,
                "{dt:?} is promoted to float and computed at f32"
            );
        }
        assert_eq!(Compute::of(ElementKind::F64), Compute::F64);
        assert_eq!(Compute::of(ElementKind::F32Strict), Compute::F32);
    }
    use super::*;
    use crate::ir::{input, konst, reduced};

    fn opt(e: crate::ir::Expr) -> ScalarExpr {
        optimize(&e.0, F32)
    }

    fn neg(e: ScalarExpr) -> ScalarExpr {
        ScalarExpr::Unary(UnaryOp::Neg, Box::new(e))
    }

    #[test]
    fn mul_one_is_identity() {
        assert_eq!(opt(input(0) * konst(1.0)), ScalarExpr::Input(0));
    }

    #[test]
    fn zero_identities_are_sign_gated() {
        // The bit-exact identities: x + (-0) and x - (+0) pass through…
        assert_eq!(opt(input(0) + konst(-0.0)), ScalarExpr::Input(0));
        assert_eq!(opt(input(2) - konst(0.0)), ScalarExpr::Input(2));
        // …but the sign-flipping forms must NOT: (-0)+(+0) = +0 and
        // (-0)-(-0) = +0, so eliminating the op would leak a -0 through.
        assert!(matches!(opt(input(0) + konst(0.0)), ScalarExpr::Add(_, _)));
        assert!(matches!(opt(input(2) - konst(-0.0)), ScalarExpr::Sub(_, _)));
    }

    #[test]
    fn nan_constants_are_never_folded() {
        // Folding a host NaN would emit the positive canonical `NAN` literal,
        // dropping the sign/payload the runtime device op preserves.
        let e = optimize(&neg(ScalarExpr::Const(f64::NAN)), F32);
        assert!(
            matches!(e, ScalarExpr::Unary(UnaryOp::Neg, ref x) if matches!(**x, ScalarExpr::Const(v) if v.is_nan())),
            "neg(NaN) stays symbolic, got {e:?}"
        );
        assert!(matches!(
            opt(konst(f64::NAN) + konst(1.0)),
            ScalarExpr::Add(_, _)
        ));
    }

    #[test]
    fn mul_zero_is_not_folded_for_nonconst_operand() {
        // x * 0 must NOT collapse to 0: for x = NaN or ±Inf the kernel computes
        // NaN (and for finite negative x, -0), so the fold would change bits.
        // The rewrite set is precision-safe by contract.
        let e = opt(input(0) * konst(0.0));
        assert!(
            matches!(e, ScalarExpr::Mul(_, _)),
            "x*0 stays symbolic, got {e:?}"
        );
        // Two-const products still fold exactly (0*0 included).
        assert_eq!(opt(konst(0.0) * konst(5.0)), ScalarExpr::Const(0.0));
    }

    #[test]
    fn div_by_pow2_becomes_mul_by_exact_reciprocal() {
        // x / 4 -> x * 0.25 (bit-exact; FDIV ~4x an FMUL, so extraction prefers it).
        assert_eq!(
            opt(input(0) / konst(4.0)),
            ScalarExpr::Mul(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Const(0.25))
            )
        );
        // Negative power of two too: x / -2 -> x * -0.5.
        assert_eq!(
            opt(input(0) / konst(-2.0)),
            ScalarExpr::Mul(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Const(-0.5))
            )
        );
        // Non-power-of-two divisor stays a division (1/3 is inexact).
        assert!(matches!(opt(input(0) / konst(3.0)), ScalarExpr::Div(_, _)));
        // Zero / subnormal-reciprocal divisors stay divisions.
        assert!(matches!(opt(input(0) / konst(0.0)), ScalarExpr::Div(_, _)));
    }

    /// The `x / 2^k -> x * 2^-k` rule is sound **only because constants promote to
    /// double**, and this measures exactly how much it depends on that.
    ///
    /// [`exact_pow2_recip`] asks "is this a normal power of two, and is its
    /// reciprocal normal" **in f64**, and never sees the kernel's dtype.
    ///
    /// # This test's premise HAS NOW CHANGED, and the test survives it
    ///
    /// It used to say the f64 predicate was correct "because `const_lit` emits a
    /// bare double literal and C's usual arithmetic conversions promote the whole
    /// expression". That is no longer the basis. Constants are now rounded to the
    /// compute precision at INGEST (see [`Compute`]), and the `x / 2^k` rewrite
    /// is separately gated on the reciprocal being exact at that precision. So
    /// soundness comes from those two together, not from double promotion.
    ///
    /// Swept every binade to confirm the replacement holds: with ingest rounding
    /// making `c` compute-representable and the gate making `r` compute-
    /// representable, **zero** admitted constants diverge — pinned below in
    /// `the_pow2_rewrite_is_sound_under_ingest_rounding`.
    ///
    /// The test is kept because the underlying measurement is still true and
    /// still instructive: the raw f64 predicate accepts 2045 constants of which
    /// 44 are wrong under f32 arithmetic. It documents why a dtype-blind
    /// predicate cannot be trusted alone — which is exactly why the gate exists.
    ///
    /// Make constants dtype-correct (`2.0f` rather than `2.0`) and the arithmetic
    /// moves to f32, where the f64 predicate is the WRONG question. Of the 2045
    /// constants it accepts, **44 give different bits under f32 arithmetic**. The
    /// cleanest witness is `c = 2^-149`: a perfectly normal f64 whose reciprocal
    /// `2^149` overflows to infinity in f32, so `0.0 / c` is `0.0` while
    /// `0.0 * (1/c)` is `NaN`.
    ///
    /// This test also pins the FIX, so it is not merely an alarm: asking the same
    /// normality question in f32 admits 253 constants and **zero** of them differ.
    /// So the repair is to make the predicate dtype-aware, not to weaken the rule
    /// — which requires threading the kernel dtype into the optimizer, since
    /// [`optimize`] currently takes only a [`ScalarExpr`].
    ///
    /// **The two changes were welded and have now both landed**, which is why the
    /// premise above moved. This test still fails if the raw predicate is ever
    /// mistaken for a dtype-aware one.
    /// **The `x / 2^k -> x * 2^-k` rewrite is sound at `f32` under the new
    /// basis**, swept rather than argued.
    ///
    /// Soundness now rests on two things acting together, neither sufficient
    /// alone:
    ///
    /// 1. **Ingest rounding** makes the divisor `c` a value the device can hold.
    ///    Without it, a body carrying `2^149` would reach the rule as a normal
    ///    `f64` while the device sees `inf` — and `x / inf` is `±0` where
    ///    `x * 2^-149` is not.
    /// 2. **The exactness gate** rejects a reciprocal the compute precision
    ///    cannot represent, which is the `2^-149` -> `0` direction.
    ///
    /// This walks every `f64` binade, applies both, and checks the rewrite is
    /// bit-identical over a value sweep including the specials that expose
    /// inf/NaN divergence. Zero admitted constants diverge.
    #[test]
    fn the_pow2_rewrite_is_sound_under_ingest_rounding() {
        let compute = Compute::F32;
        let xs: [f64; 8] = [
            1.0,
            -1.0,
            3.0,
            1e-30,
            1e30,
            0.0,
            -0.0,
            f64::from(f32::from_bits(1)),
        ];
        let (mut admitted, mut divergent) = (0u32, 0u32);
        for k in -1074i32..=1023 {
            // Ingest rounding: the divisor is whatever the DEVICE would hold.
            let c = compute.round((2.0f64).powi(k));
            let Some(r) = exact_pow2_recip(c) else {
                continue;
            };
            if compute.round(r) != r {
                continue; // the gate rejects it
            }
            admitted += 1;
            for &x in &xs {
                let x = compute.round(x);
                let by_div = compute.round(x / c);
                let by_mul = compute.round(x * compute.round(r));
                if by_div.to_bits() != by_mul.to_bits() {
                    divergent += 1;
                }
            }
        }
        assert!(
            admitted > 0,
            "harness precondition: the gate must admit SOMETHING, or this test              proves only that rejecting everything is safe"
        );
        assert_eq!(
            divergent, 0,
            "{divergent} admitted constants diverge — the rewrite is not              bit-exact under the ingest-rounding + exactness-gate basis"
        );
    }

    #[test]
    fn pow2_rule_soundness_is_coupled_to_double_promotion() {
        fn normal_pow2_f32(v: f32) -> bool {
            let b = v.to_bits();
            let e = (b >> 23) & 0xff;
            b & ((1u32 << 23) - 1) == 0 && e != 0 && e != 0xff
        }

        // Every f32 binade, both signs, plus the specials that expose inf/NaN
        // divergence (0 * inf is NaN; 0 / subnormal is 0).
        let mut xs: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::NAN,
            f32::INFINITY,
            -f32::INFINITY,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::MAX,
            3.0,
            -7.25,
        ];
        for e in -149i32..=127 {
            xs.push(f32::from_bits((((e + 150) as u32) << 23) | 0x0040_0000));
        }

        let (mut accepted, mut wrong_in_f32, mut f32_guarded, mut wrong_when_guarded) =
            (0u32, 0u32, 0u32, 0u32);
        for k in -1074i32..=1023 {
            let c = (2.0f64).powi(k);
            let Some(r) = exact_pow2_recip(c) else {
                continue;
            };
            accepted += 1;
            let (cf, rf) = (c as f32, r as f32);
            let guarded = normal_pow2_f32(cf) && normal_pow2_f32(rf);
            if guarded {
                f32_guarded += 1;
            }
            let differs = xs.iter().any(|&x| {
                let (a, b) = (x / cf, x * rf);
                !((a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits())
            });
            if differs {
                wrong_in_f32 += 1;
                if guarded {
                    wrong_when_guarded += 1;
                }
            }
        }

        assert_eq!(accepted, 2045, "the f64 predicate's acceptance set moved");
        assert_eq!(
            wrong_in_f32, 44,
            "the f32 hazard set moved — re-derive before trusting the rule at f32"
        );
        assert_eq!(f32_guarded, 253, "the f32-normal acceptance set moved");
        assert_eq!(
            wrong_when_guarded, 0,
            "asking normality in the KERNEL's dtype is the fix; if this is ever \
             non-zero the repair is wrong and the rule needs a stronger precondition"
        );
    }

    #[test]
    fn abs_and_relu_idempotents_collapse() {
        let abs = |e: ScalarExpr| ScalarExpr::Unary(UnaryOp::Abs, Box::new(e));
        let relu = |e: ScalarExpr| ScalarExpr::Unary(UnaryOp::Relu, Box::new(e));
        let x = ScalarExpr::Input(0);
        assert_eq!(optimize(&abs(abs(x.clone())), F32), abs(x.clone()));
        assert_eq!(optimize(&relu(relu(x.clone())), F32), relu(x.clone()));
        // |-y| == |y| bit-exactly (sign-bit op); relu(neg) is NOT rewritten.
        assert_eq!(optimize(&abs(neg(x.clone())), F32), abs(x.clone()));
        let rn = optimize(&relu(neg(x.clone())), F32);
        assert!(
            matches!(&rn, ScalarExpr::Unary(UnaryOp::Relu, inner)
                if matches!(**inner, ScalarExpr::Unary(UnaryOp::Neg, _))),
            "relu(neg(x)) must stay, got {rn:?}"
        );
    }

    #[test]
    fn constants_fold() {
        assert_eq!(opt(konst(2.0) * konst(3.0)), ScalarExpr::Const(6.0));
        assert_eq!(opt(konst(2.0) + konst(5.0)), ScalarExpr::Const(7.0));
    }

    #[test]
    fn neg_neg_cancels() {
        assert_eq!(
            optimize(&neg(neg(ScalarExpr::Input(0))), F32),
            ScalarExpr::Input(0)
        );
    }

    #[test]
    fn redundant_chain_simplifies_under_an_op() {
        // relu(x*1 + (-0)) -> relu(x): the (sign-correct) identities propagate
        // under the Relu via the shared e-class, and extraction picks the
        // cheapest form.
        let body = (input(0) * konst(1.0) + konst(-0.0)).relu();
        assert_eq!(
            optimize(&body.0, F32),
            ScalarExpr::Unary(UnaryOp::Relu, Box::new(ScalarExpr::Input(0)))
        );
    }

    #[test]
    fn transcendentals_are_not_const_folded() {
        // exp(1.0) is left symbolic (host-f64 vs device-f32), not folded to a const.
        let e = opt(konst(1.0).exp());
        assert_eq!(
            e,
            ScalarExpr::Unary(UnaryOp::Exp, Box::new(ScalarExpr::Const(1.0)))
        );
    }

    #[test]
    fn irreducible_body_is_unchanged() {
        let e = input(0) + input(1) * input(2);
        assert_eq!(opt(e.clone()), e.0);
    }

    #[test]
    fn binary_fn_folds_and_simplifies() {
        // const fold max/min; max(x,x) -> x.
        assert_eq!(opt(konst(2.0).max(konst(5.0))), ScalarExpr::Const(5.0));
        assert_eq!(opt(konst(2.0).min(konst(5.0))), ScalarExpr::Const(2.0));
        let max_xx = ScalarExpr::Binary(
            BinaryOp::Max,
            Box::new(ScalarExpr::Input(0)),
            Box::new(ScalarExpr::Input(0)),
        );
        assert_eq!(optimize(&max_xx, F32), ScalarExpr::Input(0));
        // Pow is not const-folded (host/device divergence) — stays symbolic.
        let pow = opt(konst(2.0).pow(konst(3.0)));
        assert!(matches!(pow, ScalarExpr::Binary(BinaryOp::Pow, _, _)));
        // Rem is FLOORED (torch.remainder): -3 rem 2 = 1 (sign-of-divisor), not -1.
        let rem = optimize(
            &ScalarExpr::Binary(
                BinaryOp::Rem,
                Box::new(ScalarExpr::Const(-3.0)),
                Box::new(ScalarExpr::Const(2.0)),
            ),
            F32,
        );
        assert_eq!(rem, ScalarExpr::Const(1.0));
    }

    #[test]
    fn idempotent() {
        let body = (input(0) * konst(1.0) + konst(0.0)).relu().0;
        let once = optimize(&body, F32);
        assert_eq!(optimize(&once, F32), once);
    }

    #[test]
    fn trunc_const_fold_is_finite_gated() {
        use crate::ir::UnaryOp;
        let trunc = |v: f64| ScalarExpr::Unary(UnaryOp::Trunc, Box::new(ScalarExpr::Const(v)));
        // Exact on finite values: fold (round toward zero, both signs).
        assert_eq!(optimize(&trunc(-3.7), F32), ScalarExpr::Const(-3.0));
        assert_eq!(optimize(&trunc(2.9), F32), ScalarExpr::Const(2.0));
        // Non-finite stays symbolic (house lesson: no non-finite const folds).
        assert!(matches!(
            optimize(&trunc(f64::INFINITY), F32),
            ScalarExpr::Unary(UnaryOp::Trunc, _)
        ));
        assert!(matches!(
            optimize(&trunc(f64::NEG_INFINITY), F32),
            ScalarExpr::Unary(UnaryOp::Trunc, _)
        ));
        assert!(matches!(
            optimize(&trunc(f64::NAN), F32),
            ScalarExpr::Unary(UnaryOp::Trunc, _)
        ));
    }

    #[test]
    fn increment_0a_fns_are_never_const_folded() {
        use crate::ir::{BinaryOp, UnaryOp};
        // Device-approximate fns stay symbolic (the Rsqrt-fold lesson) — ALL
        // 17 approximate increment-0a unaries, so a host-fold added for any
        // one of them fails here (mutation-caught gap: sampling 5 let an
        // Exp2 host-fold pass the suite).
        for op in [
            UnaryOp::Erfc,
            UnaryOp::Exp2,
            UnaryOp::Expm1,
            UnaryOp::Log2,
            UnaryOp::Log10,
            UnaryOp::Log1p,
            UnaryOp::Sinh,
            UnaryOp::Cosh,
            UnaryOp::Tan,
            UnaryOp::Asin,
            UnaryOp::Acos,
            UnaryOp::Atan,
            UnaryOp::Asinh,
            UnaryOp::Acosh,
            UnaryOp::Atanh,
            UnaryOp::Cbrt,
            UnaryOp::Lgamma,
        ] {
            let e = optimize(
                &ScalarExpr::Unary(op, Box::new(ScalarExpr::Const(1.5))),
                F32,
            );
            assert!(
                matches!(e, ScalarExpr::Unary(o, _) if o == op),
                "{op:?}(const) must stay symbolic, got {e:?}"
            );
        }
        // …and so do ALL the new binaries — including the exact bit-level ones
        // (when-in-doubt-add-no-rule; Nextafter is dtype-lattice-dependent).
        for op in [
            BinaryOp::Atan2,
            BinaryOp::Copysign,
            BinaryOp::Nextafter,
            BinaryOp::FmaxIeee,
            BinaryOp::FminIeee,
            BinaryOp::RemTrunc,
        ] {
            let e = optimize(
                &ScalarExpr::Binary(
                    op,
                    Box::new(ScalarExpr::Const(-3.0)),
                    Box::new(ScalarExpr::Const(2.0)),
                ),
                F32,
            );
            assert!(
                matches!(e, ScalarExpr::Binary(o, _, _) if o == op),
                "{op:?}(const, const) must stay symbolic, got {e:?}"
            );
        }
        // No new identity rewrites either: the x,x forms of every new binary
        // stay as authored (unlike the pinned max(x,x)/min(x,x) -> x for the
        // NaN-propagating Max/Min). Pinning all three closes the mutation
        // where extending the Max|Min rule arm to FminIeee passed the suite.
        for op in [BinaryOp::FmaxIeee, BinaryOp::FminIeee, BinaryOp::RemTrunc] {
            let same = ScalarExpr::Binary(
                op,
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(0)),
            );
            assert_eq!(
                optimize(&same, F32),
                same,
                "{op:?}(x, x) must stay as authored"
            );
        }
    }

    #[test]
    fn cmp_predicates_are_never_folded_or_rewritten() {
        use crate::ir::BinaryOp;
        const CMPS: [BinaryOp; 6] = [
            BinaryOp::CmpEq,
            BinaryOp::CmpNe,
            BinaryOp::CmpLt,
            BinaryOp::CmpLe,
            BinaryOp::CmpGt,
            BinaryOp::CmpGe,
        ];
        // No const folds, even const-const (NaN gates + host-vs-device compare
        // width + when-in-doubt-add-no-rule): every pair stays symbolic —
        // including the tempting finite pair, the NaN pair, and the ±0 pair
        // (where -0 == +0 is TRUE, a fold bug magnet).
        for op in CMPS {
            for (x, y) in [(2.0, 3.0), (f64::NAN, 1.0), (-0.0, 0.0)] {
                let e = ScalarExpr::Binary(
                    op,
                    Box::new(ScalarExpr::Const(x)),
                    Box::new(ScalarExpr::Const(y)),
                );
                assert!(
                    matches!(optimize(&e, F32), ScalarExpr::Binary(o, _, _) if o == op),
                    "{op:?}({x}, {y}) must stay symbolic"
                );
            }
        }
        // And NO identity rewrites: CmpEq(x, x) must NOT fold to 1.0 — it is
        // FALSE for NaN x (and CmpNe(x, x) is TRUE for NaN x); the reflexive
        // form of every predicate stays exactly as authored. This pins the
        // mutation the 0a review caught for Min|FminIeee: extending the
        // max(x,x)->x rule arm to any Cmp* must fail here.
        for op in CMPS {
            let same = ScalarExpr::Binary(
                op,
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(0)),
            );
            assert_eq!(
                optimize(&same, F32),
                same,
                "{op:?}(x, x) must stay as authored"
            );
        }
    }

    #[test]
    fn int_bitwise_logical_ops_are_never_folded_or_rewritten() {
        use crate::ir::BinaryOp;
        // The e-graph's consts are host f64 — a host-f64 "fold" of an int op
        // is WRONG twice over (f64 cannot represent all i64; wrapping
        // two's-complement differs from float arithmetic), and the e-graph
        // has no dtype context to even know the operand width. ZERO rules for
        // ALL EIGHT increment-0c ops, pinned exhaustively (the 0a lesson: pin
        // all, not a sample — a sampled pin let an Exp2 host-fold pass).
        const INT_OPS: [BinaryOp; 8] = [
            BinaryOp::BitAnd,
            BinaryOp::BitOr,
            BinaryOp::BitXor,
            BinaryOp::Shl,
            BinaryOp::Shr,
            BinaryOp::LogicalAnd,
            BinaryOp::LogicalOr,
            BinaryOp::LogicalXor,
        ];
        // No const folds, even const-const — including the tempting all-int
        // pair, a zero operand (x & 0, x << 0 are classic fold bait), and a
        // negative shift amount (device-inherited behavior, unknowable here).
        for op in INT_OPS {
            for (x, y) in [(6.0, 3.0), (5.0, 0.0), (1.0, -1.0)] {
                let e = ScalarExpr::Binary(
                    op,
                    Box::new(ScalarExpr::Const(x)),
                    Box::new(ScalarExpr::Const(y)),
                );
                assert!(
                    matches!(optimize(&e, F32), ScalarExpr::Binary(o, _, _) if o == op),
                    "{op:?}({x}, {y}) must stay symbolic"
                );
            }
        }
        // And NO identity rewrites: the (x, x) forms stay exactly as authored
        // (BitAnd(x,x)=x / BitOr(x,x)=x / LogicalXor(x,x)=0 are all true on
        // device but the rule set stays empty — when-in-doubt-add-no-rule;
        // this pins the 0a-review mutation class where widening the
        // max(x,x)->x rule arm passed the suite).
        for op in INT_OPS {
            let same = ScalarExpr::Binary(
                op,
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(0)),
            );
            assert_eq!(
                optimize(&same, F32),
                same,
                "{op:?}(x, x) must stay as authored"
            );
        }
        // Cost entries exist (compare-select tier) so extraction still ranks
        // bodies containing them — weight, not rules, is the only e-graph
        // surface the 0c ops touch.
        for op in INT_OPS {
            let n = ENode::Binary(op, 0, 1);
            assert_eq!(weight(&n), 2, "{op:?} must sit in the compare-select tier");
        }
    }

    #[test]
    fn coord_is_an_opaque_leaf_with_no_rules() {
        use crate::ir::{BinaryOp, coord};
        // Representative Coord bodies round-trip optimize(, F32) UNCHANGED — the
        // triu-mask predicate multiply and the alibi relative-position body.
        let triu = (input(0) * coord(1).binary(BinaryOp::CmpGe, coord(0) + konst(0.0))).0;
        assert_eq!(optimize(&triu, F32), triu, "triu-mask body must round-trip");
        let alibi = ((coord(1) - coord(0)) * crate::ir::param(0)).0;
        assert_eq!(optimize(&alibi, F32), alibi, "alibi body must round-trip");
        // A bare Coord leaf is already minimal.
        assert_eq!(optimize(&coord(1).0, F32), ScalarExpr::Coord(1));
        // No rule equates Coord(i) with anything else: Coord(0) - Coord(0)
        // stays symbolic (no x-x rule exists, and none may be added for
        // Coord), and the reflexive compare stays as authored (the same
        // NaN-honesty pin the Cmp* set carries — a widened rule arm matching
        // Coord must fail here).
        let sub_same = ScalarExpr::Sub(
            Box::new(ScalarExpr::Coord(0)),
            Box::new(ScalarExpr::Coord(0)),
        );
        assert_eq!(optimize(&sub_same, F32), sub_same);
        for op in [
            BinaryOp::CmpEq,
            BinaryOp::CmpGe,
            BinaryOp::Max,
            BinaryOp::Min,
        ] {
            let e = ScalarExpr::Binary(
                op,
                Box::new(ScalarExpr::Coord(0)),
                Box::new(ScalarExpr::Coord(1)),
            );
            assert_eq!(optimize(&e, F32), e, "{op:?}(c0, c1) must stay as authored");
        }
        // Coord(0) and Coord(1) never merge (distinct axes = distinct values):
        // c0 + c1 keeps two distinct leaves.
        let e = optimize(&(coord(0) + coord(1)).0, F32);
        assert!(
            matches!(&e, ScalarExpr::Add(a, b)
                if matches!(**a, ScalarExpr::Coord(0)) && matches!(**b, ScalarExpr::Coord(1))),
            "Coord(0)/Coord(1) must stay distinct, got {e:?}"
        );
        // The VALUE-GENERIC bit-exact identities still apply to a Coord
        // operand (they are proofs about every value, not about Coord):
        // c1 * 1.0 -> c1. This is hash-cons/extraction, not a Coord rule.
        assert_eq!(
            optimize(&(coord(1) * konst(1.0)).0, F32),
            ScalarExpr::Coord(1)
        );
    }

    #[test]
    fn select_is_never_folded_or_rewritten() {
        use crate::ir::{BinaryOp, coord, input, konst};
        let sel = |c: ScalarExpr, a: ScalarExpr, b: ScalarExpr| {
            ScalarExpr::Select(Box::new(c), Box::new(a), Box::new(b))
        };
        // ZERO select rules in v1 (the 0b cmp posture: when-in-doubt-add-no-
        // rule, pinned per shape). (1) No const-cond fold — not even the
        // "obviously" decidable ones (1.0 / 0.0 / -0.0 / NaN conds all stay;
        // a NaN cond is TRUE, a fold bug magnet):
        for c in [1.0, 0.0, -0.0, f64::NAN] {
            let e = sel(
                ScalarExpr::Const(c),
                ScalarExpr::Input(0),
                ScalarExpr::Input(1),
            );
            let o = optimize(&e, F32);
            assert!(
                matches!(&o, ScalarExpr::Select(cc, _, _)
                    if matches!(**cc, ScalarExpr::Const(v) if v == c || (v.is_nan() && c.is_nan()))),
                "select(Const({c}), a, b) must stay as authored, got {o:?}"
            );
        }
        // (2) No select(c, x, x) -> x (bit-sound under the sNaN carve-out but
        // deferred — recorded as a deferred candidate, not a rule):
        let same = sel(
            ScalarExpr::Input(0),
            ScalarExpr::Input(1),
            ScalarExpr::Input(1),
        );
        assert_eq!(
            optimize(&same, F32),
            same,
            "select(c, x, x) must stay as authored"
        );
        // (3) NO mask-multiply equation in EITHER direction — that rewrite IS
        // the triu signed-zero/NaN bug. x * cond stays a Mul…
        let cond = input(0).binary(BinaryOp::CmpGt, konst(0.0));
        let mask_mul = (input(1) * cond.clone()).0;
        assert!(
            matches!(optimize(&mask_mul, F32), ScalarExpr::Mul(_, _)),
            "x * cond must NEVER become a select"
        );
        // …and select(cond, x, 0) stays a Select (never a multiply).
        let sel_zero = cond.select(input(1), konst(0.0)).0;
        assert_eq!(
            optimize(&sel_zero, F32),
            sel_zero,
            "select(cond, x, 0) must NEVER become a mask-multiply"
        );
        // (4) No distribution through ops: f(select(c, a, b)) stays as authored.
        let distributed = ScalarExpr::Unary(
            UnaryOp::Relu,
            Box::new(sel(
                ScalarExpr::Input(0),
                ScalarExpr::Input(1),
                ScalarExpr::Input(2),
            )),
        );
        assert_eq!(optimize(&distributed, F32), distributed);
        // (5) No cond simplification: a reflexive-compare cond stays symbolic
        // (CmpEq(x, x) is FALSE for NaN x — same honesty pin as the cmp set),
        // and a Coord cond round-trips (the triu authoring).
        let refl = sel(
            ScalarExpr::Binary(
                BinaryOp::CmpEq,
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(0)),
            ),
            ScalarExpr::Input(1),
            ScalarExpr::Input(2),
        );
        assert_eq!(optimize(&refl, F32), refl);
        let triu = coord(1)
            .binary(BinaryOp::CmpGe, coord(0) + konst(0.0))
            .select(input(0), konst(0.0))
            .0;
        assert_eq!(
            optimize(&triu, F32),
            triu,
            "the triu select body must round-trip"
        );
        // (6) The VALUE-GENERIC bit-exact identities still apply INSIDE a
        // select operand (proofs about every value, not select rules):
        // select(c, x*1, b) -> select(c, x, b) via extraction, order intact.
        let inner = sel(
            ScalarExpr::Input(0),
            (input(1) * konst(1.0)).0,
            ScalarExpr::Input(2),
        );
        assert_eq!(
            optimize(&inner, F32),
            sel(
                ScalarExpr::Input(0),
                ScalarExpr::Input(1),
                ScalarExpr::Input(2)
            )
        );
        // (7) Weight pin: select sits in the compare-select tier (one selp).
        assert_eq!(weight(&ENode::Select(0, 1, 2)), 2);
    }

    #[test]
    fn rem_const_fold_is_finite_gated() {
        use crate::ir::BinaryOp;
        let rem = |a: f64, b: f64| {
            ScalarExpr::Binary(
                BinaryOp::Rem,
                Box::new(ScalarExpr::Const(a)),
                Box::new(ScalarExpr::Const(b)),
            )
        };
        // The legitimate fold still fires…
        assert_eq!(optimize(&rem(7.0, 2.0), F32), ScalarExpr::Const(1.0));
        // …but NaN operands stay symbolic (a fold would canonicalize the
        // payload/sign the device op propagates)…
        for e in [rem(f64::NAN, 2.0), rem(5.0, f64::NAN)] {
            assert!(
                matches!(optimize(&e, F32), ScalarExpr::Binary(BinaryOp::Rem, _, _)),
                "NaN-operand Rem must stay symbolic"
            );
        }
        // …as do infinite operands and finite operands whose floored-mod
        // composite overflows to ±inf (a -INFINITY literal is forbidden under
        // the headerless-nvrtc discipline).
        for e in [
            rem(f64::INFINITY, 2.0),
            rem(5.0, f64::NEG_INFINITY),
            rem(1e308, 1e-308),
        ] {
            assert!(
                matches!(optimize(&e, F32), ScalarExpr::Binary(BinaryOp::Rem, _, _)),
                "non-finite-in or non-finite-out Rem must stay symbolic"
            );
        }
    }

    // -------------------------------------------------------------------
    // optimize_top_k (item 08 k-best) tests
    // -------------------------------------------------------------------

    /// Extraction cost of a reconstructed form, mirroring `weight`+`children`
    /// VERBATIM (a dummy-child e-node feeds the exact shipped `weight`), so the
    /// tests assert cost-ascending order against the same cost model extraction
    /// uses — no parallel cost table to drift.
    fn cost_of(e: &ScalarExpr) -> u64 {
        let (node, kids): (ENode, Vec<&ScalarExpr>) = match e {
            ScalarExpr::Input(i) => (ENode::Input(*i), vec![]),
            ScalarExpr::Const(v) => (ENode::Const(v.to_bits()), vec![]),
            ScalarExpr::Param(i) => (ENode::Param(*i), vec![]),
            ScalarExpr::Reduced(i) => (ENode::Reduced(*i), vec![]),
            ScalarExpr::Coord(d) => (ENode::Coord(*d), vec![]),
            ScalarExpr::Add(a, b) => (ENode::Add(0, 0), vec![a, b]),
            ScalarExpr::Sub(a, b) => (ENode::Sub(0, 0), vec![a, b]),
            ScalarExpr::Mul(a, b) => (ENode::Mul(0, 0), vec![a, b]),
            ScalarExpr::Div(a, b) => (ENode::Div(0, 0), vec![a, b]),
            ScalarExpr::Binary(op, a, b) => (ENode::Binary(*op, 0, 0), vec![a, b]),
            ScalarExpr::Unary(op, a) => (ENode::Unary(*op, 0), vec![a]),
            ScalarExpr::Select(c, a, b) => (ENode::Select(0, 0, 0), vec![c, a, b]),
        };
        kids.into_iter()
            .fold(weight(&node), |acc, k| acc.saturating_add(cost_of(k)))
    }

    /// **Optimizing never costs more than not optimizing.**
    ///
    /// The question this answers, asked by CireSnave: read a kernel in, optimize
    /// it, write it back out — does it come back better, the same, or worse? At
    /// the IR layer it must never be worse, and this is the check rather than
    /// the argument.
    ///
    /// The argument, for what it is worth, is that `add_expr` puts the input's
    /// own nodes in the e-graph and `extract` takes the minimum, so the input is
    /// always a candidate. **The reason that is not sufficient** is
    /// `enode_cost`, which returns `Option` — a node it cannot price is SKIPPED,
    /// and if the input's own form is unpriceable, extraction must pick
    /// something else, which is free to be worse. So the property is measured
    /// over a spread wide enough to reach the odd corners: unpriceable nodes,
    /// saturation limits, and bodies the rewrite set does not touch at all.
    #[test]
    fn optimizing_never_increases_cost() {
        let abs = |e: ScalarExpr| ScalarExpr::Unary(UnaryOp::Abs, Box::new(e));
        let sel = |c: ScalarExpr, a: ScalarExpr, b: ScalarExpr| {
            ScalarExpr::Select(Box::new(c), Box::new(a), Box::new(b))
        };
        let bodies = [
            // already minimal — must not be made worse
            ScalarExpr::Input(0),
            (input(0) + input(1)).0,
            // rewritable
            (input(0) * konst(1.0)).0,
            (input(0) / konst(2.0)).0,
            (input(0) + konst(0.0)).0,
            neg(neg(ScalarExpr::Input(0))),
            abs(abs(ScalarExpr::Input(0))),
            // constant-foldable
            (konst(2.0) * konst(3.0)).0,
            konst(2.0).max(konst(5.0)).0,
            // untouched by any rule
            sel(input(0).0, input(1).0, input(2).0),
            ScalarExpr::Unary(UnaryOp::Erf, Box::new(ScalarExpr::Input(0))),
            (reduced(0) + konst(1e-5)).0,
            ScalarExpr::Coord(1),
            ScalarExpr::Param(0),
            // deep enough to press the saturation limit
            (0..12).fold(ScalarExpr::Input(0), |acc, _| {
                ScalarExpr::Add(Box::new(acc), Box::new(konst(1.0).0))
            }),
        ];

        let mut improved = 0;
        for body in bodies {
            let before = cost_of(&body);
            let after = cost_of(&optimize(&body, F32));
            assert!(
                after <= before,
                "optimize made this MORE expensive ({before} -> {after}): {body:?}"
            );
            if after < before {
                improved += 1;
            }
        }

        // Vacuity control. If nothing in the corpus improves, `after <= before`
        // is satisfied by an optimizer that returns its input unchanged, and the
        // assertion above would hold for a no-op.
        assert!(
            improved > 0,
            "no body improved — the corpus cannot tell a working optimizer from              an identity function"
        );
    }

    /// The #1 pin: `form[0] == optimize(e, F32)` bit-identical, and `top_k(_, 1)` IS
    /// the shipped optimizer — across a spread of body shapes. A k-best that
    /// silently changed the k==1 winner would alter every JIT-selected form.
    #[test]
    fn top_k_form0_is_optimize_across_bodies() {
        let abs = |e: ScalarExpr| ScalarExpr::Unary(UnaryOp::Abs, Box::new(e));
        let bodies = [
            (input(0) / konst(2.0)).0,
            (input(0) * konst(1.0) + konst(-0.0)).0,
            (input(0) + input(1) * input(2)).0,
            neg(neg(ScalarExpr::Input(0))),
            abs(abs(ScalarExpr::Input(0))),
            konst(2.0).max(konst(5.0)).0,
            (reduced(0) + konst(1e-5)).0,
        ];
        for body in bodies {
            let opt = optimize(&body, F32);
            assert_eq!(
                optimize_top_k(&body, 1, F32),
                vec![opt.clone()],
                "k==1 must equal [optimize(e, F32)]"
            );
            for k in [2usize, 3, 5] {
                let forms = optimize_top_k(&body, k, F32);
                assert_eq!(forms[0], opt, "form[0] must be bit-identical to optimize");
            }
        }
    }

    /// The no-new-rule K=2 fixture the brief names: the shipped `x / 2^k ->
    /// x * 2^-k` union puts `Mul(x, 0.5)` and `Div(x, 2)` in one class, so the
    /// two cheapest forms are exactly `[Mul(Input,0.5), Div(Input,2)]`, form[0]
    /// the cheaper Mul (== optimize).
    #[test]
    fn top_k_div_pow2_yields_mul_then_div() {
        let body = (input(0) / konst(2.0)).0;
        let forms = optimize_top_k(&body, 2, F32);
        assert_eq!(
            forms,
            vec![
                ScalarExpr::Mul(
                    Box::new(ScalarExpr::Input(0)),
                    Box::new(ScalarExpr::Const(0.5))
                ),
                ScalarExpr::Div(
                    Box::new(ScalarExpr::Input(0)),
                    Box::new(ScalarExpr::Const(2.0))
                ),
            ]
        );
        assert_eq!(forms[0], optimize(&body, F32));
    }

    /// Forms are structurally distinct and cost-ascending (non-decreasing), and
    /// asking for more forms than exist just returns the ones that do.
    #[test]
    fn top_k_forms_distinct_and_cost_ascending() {
        let body = (input(0) / konst(2.0)).0;
        let forms = optimize_top_k(&body, 5, F32);
        // Only two equivalents exist for this cell.
        assert_eq!(forms.len(), 2);
        for w in forms.windows(2) {
            assert_ne!(w[0], w[1], "forms must be structurally distinct");
            assert!(
                cost_of(&w[0]) <= cost_of(&w[1]),
                "forms must be cost-ascending"
            );
        }
        // An irreducible body has exactly one form even at large k.
        let irr = (input(0) + input(1) * input(2)).0;
        assert_eq!(optimize_top_k(&irr, 8, F32), vec![irr]);
    }

    /// The cycle guard terminates on self-referential classes: `neg(neg x)`
    /// unions the outer class with `x`'s (a class that references itself through
    /// the `Neg` e-node). top-k must return (not spin), with form[0] == x, and a
    /// bounded next form.
    #[test]
    fn top_k_cycle_guard_terminates() {
        let body = neg(neg(ScalarExpr::Input(0)));
        let forms = optimize_top_k(&body, 3, F32);
        assert_eq!(forms[0], ScalarExpr::Input(0), "form[0] == optimize == x");
        // A cyclic class does not blow the list up past k, and stays distinct.
        assert!(forms.len() <= 3);
        for w in forms.windows(2) {
            assert_ne!(w[0], w[1]);
            assert!(cost_of(&w[0]) <= cost_of(&w[1]));
        }
        // A DEEPER nested body must also terminate (the cap never trips). Note
        // `optimize` itself is not bit-deterministic on the artificial neg⁴ chain
        // (its cyclic-extract fixpoint is HashMap-iteration-order sensitive — a
        // pre-existing property no real op body reaches), so we assert only what
        // k-best owns here: it returns a bounded, distinct, cost-ascending list
        // rather than spinning on the self-referential class.
        let deep = neg(neg(neg(neg(ScalarExpr::Input(0)))));
        let deep_forms = optimize_top_k(&deep, 4, F32);
        assert!(
            !deep_forms.is_empty() && deep_forms.len() <= 4,
            "bounded, non-empty"
        );
        for w in deep_forms.windows(2) {
            assert_ne!(w[0], w[1]);
            assert!(cost_of(&w[0]) <= cost_of(&w[1]));
        }
    }

    /// `Reduced` is an opaque leaf with no rule, so it is never cross-folded
    /// across forms — every form that carries `reduced(0)` keeps it as an
    /// untouched leaf, exactly as `optimize` does.
    #[test]
    fn top_k_reduced_never_cross_folded() {
        // reduced(0)/2 -> the pow2 rule still applies (value-generic), but the
        // reduced leaf itself is never folded into a constant or another leaf.
        let body = (reduced(0) / konst(2.0)).0;
        let forms = optimize_top_k(&body, 4, F32);
        assert_eq!(
            forms[0],
            ScalarExpr::Mul(
                Box::new(ScalarExpr::Reduced(0)),
                Box::new(ScalarExpr::Const(0.5))
            )
        );
        for f in &forms {
            let mentions_reduced = format!("{f:?}").contains("Reduced(0)");
            assert!(
                mentions_reduced,
                "each form retains the reduced leaf: {f:?}"
            );
        }
        // A bare reduced leaf is already minimal — single form.
        assert_eq!(
            optimize_top_k(&ScalarExpr::Reduced(1), 3, F32),
            vec![ScalarExpr::Reduced(1)]
        );
    }

    /// Determinism: the same body yields byte-identical top-k across repeated
    /// calls (no `HashMap`-iteration-order leakage into the result).
    #[test]
    fn top_k_is_deterministic() {
        let body = (input(0) / konst(2.0) + input(1) * konst(1.0)).0;
        let first = optimize_top_k(&body, 4, F32);
        for _ in 0..8 {
            assert_eq!(optimize_top_k(&body, 4, F32), first);
        }
        assert_eq!(first[0], optimize(&body, F32));
    }

    /// k == 0 is the empty set (the invariant is vacuous, not a panic).
    #[test]
    fn top_k_zero_is_empty() {
        assert!(optimize_top_k(&ScalarExpr::Input(0), 0, F32).is_empty());
    }
}
