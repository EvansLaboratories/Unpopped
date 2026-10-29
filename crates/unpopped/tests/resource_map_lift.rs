// SPDX-License-Identifier: MIT OR Apache-2.0
//! U3 row 3: a declared `ResourceMap` is what the lifts resolve names through
//! (`docs/u3-row3-resource-map-and-frontend.md`).
//!
//! The discriminating cases are the ones name inference cannot do: buffers named off the
//! frontend's convention, operand order that differs from source order, a `Params` field,
//! and a name the map does not declare (which must be loud, not a silent fallback).

use unpopped::artifact::{BindingRole, LayoutRule, Packing, ParamField, ParamsLayout};
use unpopped::convert::{
    Frontend, SLANG, lift, lift_elementwise, lift_elementwise_with, lift_reduction_with,
    lift_scan_with, lift_with, parse_slang,
};
use unpopped::ir::ScalarExpr;
use unpopped::resource_map::{DeclaredBuffer, ResourceMap, ResourceMapError};
use unpopped::{LiftError, LiftProvenance};
use unpopped_vocab::ElementKind as K;

const F32: &[K] = &[K::F32];

fn rd(name: &str, index: u32) -> DeclaredBuffer {
    DeclaredBuffer::new(name, 0, index, BindingRole::Read, K::F32, Packing::Unpacked)
}

fn wr(name: &str, index: u32) -> DeclaredBuffer {
    DeclaredBuffer::new(name, 0, index, BindingRole::Write, K::F32, Packing::Unpacked)
}

fn map(buffers: Vec<DeclaredBuffer>) -> ResourceMap {
    ResourceMap::new(buffers, None)
}

/// Off the Slang convention (`output` / `input{K}`) on every name.
const OFF_CONVENTION: &str = "[numthreads(64,1,1)]\n\
    void k(uint3 tid : SV_DispatchThreadID) { uint i = tid.x; dst[i] = a[i] - b[i]; }";

const ON_CONVENTION: &str = "[numthreads(64,1,1)]\n\
    void k(uint3 tid : SV_DispatchThreadID) { uint i = tid.x; output[i] = input0[i] - input1[i]; }";

fn sub(l: u8, r: u8) -> ScalarExpr {
    ScalarExpr::Sub(
        Box::new(ScalarExpr::Input(l)),
        Box::new(ScalarExpr::Input(r)),
    )
}

#[test]
fn names_off_the_convention_lift_only_with_a_map() {
    let m = map(vec![wr("dst", 0), rd("a", 1), rd("b", 2)]);
    let lifted = lift_elementwise_with(&SLANG, OFF_CONVENTION, "k", F32, Some(&m))
        .expect("declared names lift");
    assert_eq!(lifted.op.body, sub(0, 1));
    // The control: the same source without a map is what 0-of-147 looked like.
    assert!(lift_elementwise(&SLANG, OFF_CONVENTION, "k", F32).is_err());
}

#[test]
fn operand_order_is_declaration_order_not_source_order() {
    let forward = map(vec![wr("dst", 0), rd("a", 1), rd("b", 2)]);
    let reversed = map(vec![wr("dst", 0), rd("b", 1), rd("a", 2)]);
    let f = lift_elementwise_with(&SLANG, OFF_CONVENTION, "k", F32, Some(&forward)).unwrap();
    let r = lift_elementwise_with(&SLANG, OFF_CONVENTION, "k", F32, Some(&reversed)).unwrap();
    assert_eq!(f.op.body, sub(0, 1));
    assert_eq!(r.op.body, sub(1, 0), "a is now the second Read buffer");
}

#[test]
fn provenance_separates_declared_from_inferred() {
    let inferred = lift_elementwise(&SLANG, ON_CONVENTION, "k", F32).unwrap();
    assert_eq!(inferred.provenance, LiftProvenance::Inferred);
    assert!(inferred.resources.is_none());

    // The same kernel declared with the convention's own names: same op, other provenance.
    let m = map(vec![wr("output", 0), rd("input0", 1), rd("input1", 2)]);
    let declared = lift_elementwise_with(&SLANG, ON_CONVENTION, "k", F32, Some(&m)).unwrap();
    assert_eq!(declared.op.body, inferred.op.body);
    assert_eq!(declared.provenance, LiftProvenance::Declared);
    assert_eq!(declared.resources.as_ref(), Some(&m));

    // `None` through the `_with` form is the plain form.
    let none = lift_elementwise_with(&SLANG, ON_CONVENTION, "k", F32, None).unwrap();
    assert_eq!(none.op.body, inferred.op.body);
    assert_eq!(none.provenance, LiftProvenance::Inferred);
}

#[test]
fn an_undeclared_name_is_loud_not_a_fallback_to_inference() {
    // `input0` / `input1` are valid by convention, but this map does not declare them.
    let m = map(vec![wr("output", 0), rd("x", 1), rd("y", 2)]);
    match lift_elementwise_with(&SLANG, ON_CONVENTION, "k", F32, Some(&m)) {
        Err(LiftError::UndeclaredResource(n)) => assert_eq!(n, "input0"),
        other => panic!("expected UndeclaredResource, got {other:?}"),
    }
    // The control: without the map the same source lifts, so the refusal is the map's.
    assert!(lift_elementwise(&SLANG, ON_CONVENTION, "k", F32).is_ok());
}

#[test]
fn a_params_field_resolves_to_its_declared_position() {
    let src = "[numthreads(64,1,1)]\n\
        void k(uint3 tid : SV_DispatchThreadID) { uint i = tid.x; dst[i] = a[i] * scale + bias; }";
    let params = ParamsLayout::new(
        LayoutRule::Scalar,
        vec![
            ParamField::new("scale", 0, K::F32),
            ParamField::new("bias", 4, K::F32),
        ],
    );
    let m = ResourceMap::new(vec![wr("dst", 0), rd("a", 1)], Some(params));
    let lifted = lift_elementwise_with(&SLANG, src, "k", F32, Some(&m)).unwrap();
    assert_eq!(
        lifted.op.body,
        ScalarExpr::Add(
            Box::new(ScalarExpr::Mul(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Param(0)),
            )),
            Box::new(ScalarExpr::Param(1)),
        )
    );
    // Without the Params field in the map, `scale` is undeclared, not guessed.
    let no_params = map(vec![wr("dst", 0), rd("a", 1)]);
    assert!(matches!(
        lift_elementwise_with(&SLANG, src, "k", F32, Some(&no_params)),
        Err(LiftError::UndeclaredResource(n)) if n == "scale"
    ));
}

#[test]
fn reduction_scan_and_the_dispatcher_take_the_map() {
    let sum = "[numthreads(256,1,1)]\n\
        void rsum(uint3 tid : SV_DispatchThreadID){ float acc = 0.0f; for (uint i=0;i<256;i++) acc += src[i]; dst[0] = acc; }";
    let csum = "[numthreads(256,1,1)]\n\
        void csum(uint3 tid : SV_DispatchThreadID){ float acc = 0.0f; for (uint i=0;i<256;i++){ acc += src[i]; dst[i] = acc; } }";
    let m = map(vec![wr("dst", 0), rd("src", 1)]);
    for (name, l) in [
        ("reduction", lift_reduction_with(&SLANG, sum, "x", F32, Some(&m))),
        ("scan", lift_scan_with(&SLANG, csum, "x", F32, Some(&m))),
        ("dispatch/reduction", lift_with(&SLANG, sum, "x", F32, Some(&m))),
        ("dispatch/scan", lift_with(&SLANG, csum, "x", F32, Some(&m))),
    ] {
        let l = l.unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(l.provenance, LiftProvenance::Declared, "{name}");
        // Declared and inferred agree on the op: the same kernel renamed to the convention.
        let src = if name.ends_with("scan") { csum } else { sum };
        let renamed = src.replace("dst", "output").replace("src", "input0");
        let inferred = lift(&SLANG, &renamed, "x", F32).unwrap();
        assert_eq!(l.op, inferred.op, "{name}");
        assert_eq!(inferred.provenance, LiftProvenance::Inferred, "{name}");
    }
    // Controls: the off-convention reduction does not lift by inference.
    assert!(lift(&SLANG, sum, "x", F32).is_err());
}

#[test]
fn a_bad_map_is_refused_before_any_lift() {
    let cases: Vec<(&str, ResourceMap)> = vec![
        ("no output", map(vec![rd("a", 0)])),
        ("two outputs", map(vec![wr("x", 0), wr("y", 1)])),
        ("duplicate name", map(vec![wr("x", 0), rd("x", 1)])),
        ("empty name", map(vec![wr("", 0), rd("a", 1)])),
        ("duplicate slot", map(vec![wr("x", 0), rd("a", 0)])),
    ];
    for (what, m) in cases {
        assert!(m.check().is_err(), "{what} must fail check");
        assert!(
            matches!(
                lift_elementwise_with(&SLANG, OFF_CONVENTION, "k", F32, Some(&m)),
                Err(LiftError::BadResourceMap(_))
            ),
            "{what} must be BadResourceMap"
        );
    }
    // The control for the table: a sound map passes the same check.
    assert_eq!(map(vec![wr("x", 0), rd("a", 1)]).check(), Ok(()));
    assert_eq!(
        map(vec![wr("x", 0), rd("a", 1), rd("b", 2)]).check(),
        Ok(())
    );
    assert_eq!(
        map(vec![rd("a", 0)]).check(),
        Err(ResourceMapError::OutputCount(0))
    );
}

#[test]
fn a_misaligned_params_field_fails_the_manifest_rules() {
    let params = ParamsLayout::new(LayoutRule::Std140, vec![ParamField::new("p", 2, K::F32)]);
    let m = ResourceMap::new(vec![wr("x", 0)], Some(params));
    assert!(matches!(m.check(), Err(ResourceMapError::Manifest(_))));
}

#[test]
fn a_frontend_built_by_new_lifts_like_a_literal() {
    // `Frontend` is `#[non_exhaustive]`: this is the only way to build one from here.
    const MINE: Frontend = Frontend::new(
        "toy",
        "numthreads",
        &["groupshared"],
        "dst",
        "src",
        parse_slang,
    );
    let src = "[numthreads(64,1,1)]\n\
        void k(uint3 tid : SV_DispatchThreadID) { uint i = tid.x; dst[i] = src0[i] - src1[i]; }";
    let l = lift_elementwise(&MINE, src, "k", F32).expect("the descriptor is read");
    assert_eq!(l.op.body, sub(0, 1));
    assert!(lift_elementwise(&MINE, ON_CONVENTION, "k", F32).is_err());
}
