//! The KISS `structure_key` **derivation** leg — KISS-CLASSIFY-6.6-0011.
//!
//! `kiss_byte_match.rs` proves the token *codec* agrees with KISS. This proves
//! the other half: that the key this crate *derives* from operand descriptors
//! and declared facts is the key KISS's reference deriver derives. A codec can
//! round-trip every token and still derive the wrong cell, which is precisely
//! what the sk4 keyer did on a reduced innermost axis.
//!
//! Every vector is KISS's, vendored verbatim (`kiss/structure_key_derivation_vectors.json`).
//! Each is driven through [`derive_structure_key_token`] with the facts the
//! vector declares, and compared as bytes — or, for a decline vector, by the
//! decline's wire name. Nothing is asserted field by field.
//!
//! # What is and is not claimed
//!
//! * Only vectors carrying `spec_golden` reproduce an Appendix A.1 token whose
//!   bytes were hand-derived in the spec text. The rest are **reference-deriver
//!   output**, so agreement on them is two derivers agreeing, not a spec check.
//! * The 27 `ctr`/role-hint vectors KISS 0.4.3 added are pinned by name in
//!   `KNOWN_GAP`, which is asserted **exactly**: it is the work still owed, and
//!   the test fails both when one of them starts passing and when any other
//!   vector fails.

use serde_json::Value;
use unpopped_vocab::{
    ArchSku, AxisRef, CellFacts, ContractionAxes, DequantForm, ElementKind, MpCode, OperandDesc,
    OperandRole, STRUCTURE_KEY_VERSION, derive_structure_key_token, dtype_token,
};

const VECTORS: &str = include_str!("../kiss/structure_key_derivation_vectors.json");

/// Every §6.1 dtype, so a vector's spelling resolves through `dtype_token`
/// rather than through a second hand-written table that could drift from it.
const ALL_DTYPES: [ElementKind; 24] = [
    ElementKind::F16,
    ElementKind::Bf16,
    ElementKind::F32,
    ElementKind::F64,
    ElementKind::I8,
    ElementKind::I16,
    ElementKind::I32,
    ElementKind::I64,
    ElementKind::U8,
    ElementKind::U16,
    ElementKind::U32,
    ElementKind::U64,
    ElementKind::Bool,
    ElementKind::Fp8E4M3FN,
    ElementKind::Fp8E5M2,
    ElementKind::Fp8E4M3FNUZ,
    ElementKind::Fp8E5M2FNUZ,
    ElementKind::F8E8M0,
    ElementKind::F8E6M2,
    ElementKind::I4,
    ElementKind::U4,
    ElementKind::B1,
    ElementKind::Complex64,
    ElementKind::Complex128,
];

fn dtype(name: &str) -> ElementKind {
    *ALL_DTYPES
        .iter()
        .find(|d| dtype_token(**d) == name)
        .unwrap_or_else(|| panic!("vector names a dtype this test cannot spell: {name}"))
}

/// The ctr/role vectors KISS 0.4.3 added that this keyer does not yet match
/// (KISS `f88d389`, the ctr derivation stage and the batch role, §6.6-0023).
/// **This list is the work, not an allowance**: the follow-up PR empties it.
const KNOWN_GAP: [&str; 0] = [];

/// One vector's verdict, in the PM's three-way vocabulary.
#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    Pass,
    Fail(String),
    DeclineDiffers(String),
}

impl Verdict {
    /// The word the table prints.
    fn label(&self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Fail(_) => "FAIL",
            Self::DeclineDiffers(_) => "DECLINE≠",
        }
    }

    /// Which tally the verdict counts toward: pass, fail, decline-differs.
    fn column(&self) -> usize {
        match self {
            Self::Pass => 0,
            Self::Fail(_) => 1,
            Self::DeclineDiffers(_) => 2,
            // Column 3 is kept so the printed tally keeps its shape.
        }
    }
}

/// The operand descriptors a vector lists, in the call order it lists them.
fn operands_of(name: &str, v: &Value) -> Vec<OperandDesc> {
    let ints = |a: &Value| -> Vec<i64> {
        a.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect()
    };
    let mut ops = Vec::new();
    for o in v["operands"].as_array().expect("operands") {
        let (extents, strides) = (ints(&o["extents"]), ints(&o["strides"]));
        let mut d = OperandDesc::new(
            extents.len(),
            &extents,
            &strides,
            dtype(o["dtype"].as_str().unwrap()),
            u32::try_from(o["alignment"].as_u64().unwrap()).unwrap(),
        )
        .with_role(match o["use"].as_str().unwrap() {
            "in" => OperandRole::Input,
            "out" => OperandRole::Output,
            "inout" => OperandRole::InOut,
            other => panic!("{name}: unknown operand use {other}"),
        });
        match o["dequant_form"].as_str() {
            None => {}
            Some("dl") => d = d.with_dequant_form(DequantForm::Linear),
            Some("dc") => d = d.with_dequant_form(DequantForm::Codebook),
            Some(other) => panic!("{name}: unknown dequant_form {other}"),
        }
        ops.push(d);
    }
    ops
}

/// The caller-declared facts a vector states (§6.6-0012).
fn facts_of(name: &str, v: &Value) -> CellFacts {
    let mut facts = CellFacts::new();
    facts.contraction_declared = v["contraction_declared"].as_bool().unwrap();
    facts.index_declared = !v["index_operand"].is_null();
    for a in v["reduced_axes"].as_array().unwrap() {
        facts
            .reduced_axes
            .set(u8::try_from(a.as_u64().unwrap()).unwrap());
    }
    facts.accumulator = v["accumulator"].as_str().map(dtype);
    // The vector's `contraction` object (KISS 0.4.3): every declared role, M/N/K and
    // batch included, so the role-driven derivation is what is measured.
    if let Some(c) = v["contraction"].as_object() {
        let us = |x: &Value| usize::try_from(x.as_u64().unwrap()).unwrap();
        let axis = |r: &Value| AxisRef::new(us(&r["operand"]), us(&r["axis"]));
        let batch: Vec<usize> = c["batch_axes"].as_array().unwrap().iter().map(us).collect();
        facts.weight_operand = c["weight"].as_u64().map(|x| usize::try_from(x).unwrap());
        facts.scale_operand = c["scale"].as_u64().map(|x| usize::try_from(x).unwrap());
        facts.contraction_axes = Some(
            ContractionAxes::new(axis(&c["m"]), axis(&c["n"]), axis(&c["k"]))
                .with_batch_axes(&batch),
        );
    }
    facts.math_fidelity = match v["math_fidelity"].as_str().unwrap() {
        "st" => MpCode::St,
        "rm" => MpCode::Rm,
        other => panic!("{name}: unknown math_fidelity {other}"),
    };
    facts
}

fn run(v: &Value) -> Verdict {
    let name = v["name"].as_str().expect("name");
    assert_eq!(
        v["target"].as_str().expect("target"),
        "cuda:sm89",
        "{name}: every vector is on cuda:sm89, which ArchSku::Sm89 spells"
    );

    let got = derive_structure_key_token(&operands_of(name, v), &facts_of(name, v), ArchSku::Sm89);
    let want = v["value"].as_str().unwrap();
    match (v["expect"].as_str().unwrap(), got) {
        ("token", Ok(t)) if t == want => Verdict::Pass,
        ("token", Ok(t)) => Verdict::Fail(format!(
            "want {want}
      got  {t}"
        )),
        ("token", Err(e)) => {
            Verdict::DeclineDiffers(format!("want {want}; declined {}", e.wire_name()))
        }
        ("decline", Err(e)) if e.wire_name() == want => Verdict::Pass,
        ("decline", Err(e)) => {
            Verdict::DeclineDiffers(format!("want {want}; declined {}", e.wire_name()))
        }
        ("decline", Ok(t)) => Verdict::DeclineDiffers(format!("want decline {want}; keyed {t}")),
        (other, _) => panic!("{name}: unknown expect {other}"),
    }
}

#[test]
fn the_artifact_is_the_one_this_leg_claims() {
    let v: Value = serde_json::from_str(VECTORS).unwrap();
    assert_eq!(v["schema"], "kiss-structure-key-derivation-vectors-v1");
    assert_eq!(
        v["structure_key_schema_version"].as_u64(),
        Some(u64::from(STRUCTURE_KEY_VERSION)),
        "artifact is from a different schema version than this build implements"
    );
    assert_eq!(
        v["vectors"].as_array().unwrap().len(),
        63,
        "vector count changed"
    );
}

/// Prints the table the PM asked for (`--nocapture`) and asserts that nothing
/// outside `KNOWN_GAP` disagrees.
#[test]
fn every_derivation_vector_matches() {
    let v: Value = serde_json::from_str(VECTORS).unwrap();
    let vectors = v["vectors"].as_array().unwrap();

    let mut bad = Vec::new();
    let mut counts = [0usize; 3];
    for vec in vectors {
        let name = vec["name"].as_str().unwrap();
        let golden = if vec["spec_golden"].is_null() {
            ""
        } else {
            " [spec_golden]"
        };
        let verdict = run(vec);
        println!("{:<9} {name}{golden}", verdict.label());
        match &verdict {
            Verdict::Pass => {}
            Verdict::Fail(why) | Verdict::DeclineDiffers(why) => {
                println!("      {why}");
                bad.push(name.to_string());
            }
        }
        counts[verdict.column()] += 1;
    }
    println!(
        "PASS {} | FAIL {} | DECLINE-DIFFERS {} (of {})",
        counts[0],
        counts[1],
        counts[2],
        vectors.len()
    );
    // Exact, both directions: a vector listed here that now passes, or one not
    // listed that now fails, is a red test. The list only ever shrinks.
    let known: Vec<String> = KNOWN_GAP.iter().map(|s| (*s).to_string()).collect();
    let unexpected: Vec<_> = bad.iter().filter(|b| !known.contains(b)).collect();
    let fixed: Vec<_> = known.iter().filter(|k| !bad.contains(k)).collect();
    assert!(
        unexpected.is_empty(),
        "derivation disagrees with KISS beyond the known gap: {unexpected:?}"
    );
    assert!(
        fixed.is_empty(),
        "these vectors now match KISS; remove them from KNOWN_GAP: {fixed:?}"
    );
}

/// Positive control: the harness can fail. A harness that cannot report a
/// disagreement is the vacuous green the byte-match leg exists to prevent.
#[test]
fn the_harness_can_fail() {
    let v: Value = serde_json::from_str(VECTORS).unwrap();
    let mut poisoned = v["vectors"][0].clone();
    // Same cell, but declare a reduction: the derivation must change.
    poisoned["reduced_axes"] = serde_json::json!([1]);
    assert_ne!(run(&poisoned), Verdict::Pass);
}
