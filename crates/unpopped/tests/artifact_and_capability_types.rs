// SPDX-License-Identifier: MIT OR Apache-2.0
//! U3 build rows 1 and 2: the artifact/manifest types and the capability set, standalone.
//! Each refusal has its admitting control beside it.

use unpopped::artifact::{
    Binding, BindingKind, BindingRole, EntryManifest, KernelArtifact, KernelManifest, LayoutRule,
    ManifestError, Packing, ParamField, ParamsLayout, ToolchainRecord,
};
use unpopped::backend::DtypeSupport;
use unpopped::capset::{
    CapKey, CapSource, CapValue, Capability, CapabilityError, CapabilitySet, DenormMode,
};
use unpopped_vocab::ElementKind as K;

fn tc() -> ToolchainRecord {
    ToolchainRecord::new("1.5", "vulkan1.3", None, vec![])
}

fn bind(set: u32, index: u32) -> Binding {
    Binding::new(
        set,
        index,
        BindingKind::Storage,
        BindingRole::Read,
        K::F32,
        Packing::Unpacked,
    )
}

fn entry(name: &str) -> EntryManifest {
    let mut e = EntryManifest::new(name, [64, 1, 1]);
    e.bindings = vec![bind(0, 0), bind(0, 1)];
    e.params = Some(ParamsLayout::new(
        LayoutRule::Std430,
        vec![
            ParamField::new("n", 0, K::I32),
            ParamField::new("scale", 4, K::F32),
        ],
    ));
    e
}

fn with_entry(e: EntryManifest) -> KernelManifest {
    KernelManifest::new(tc(), vec![e])
}

#[test]
fn a_well_formed_manifest_checks() {
    assert_eq!(with_entry(entry("main")).check(), Ok(()));
}

#[test]
fn an_artifact_reads_back_only_as_what_it_is() {
    let s = KernelArtifact::Source("x".into());
    let w = KernelArtifact::SpirV(vec![0x0723_0203, 1]);
    assert_eq!(s.as_source(), Some("x"));
    assert_eq!(s.as_spirv(), None);
    assert_eq!(w.as_spirv(), Some(&[0x0723_0203, 1][..]));
    assert_eq!(w.as_source(), None);
}

#[test]
fn a_manifest_without_entries_is_refused() {
    assert_eq!(
        KernelManifest::new(tc(), vec![]).check(),
        Err(ManifestError::NoEntries)
    );
}

#[test]
fn entry_names_must_be_unique() {
    let m = KernelManifest::new(tc(), vec![entry("a"), entry("b")]);
    assert_eq!(m.check(), Ok(()));
    let m = KernelManifest::new(tc(), vec![entry("a"), entry("a")]);
    assert_eq!(m.check(), Err(ManifestError::DuplicateEntry("a".into())));
}

#[test]
fn a_zero_workgroup_dimension_is_refused() {
    for axis in 0..3 {
        let mut e = entry("k");
        e.workgroup_size[axis] = 0;
        assert_eq!(
            with_entry(e).check(),
            Err(ManifestError::ZeroWorkgroup("k".into())),
            "axis {axis}"
        );
    }
}

#[test]
fn bindings_must_not_collide_on_set_and_index() {
    let mut e = entry("k");
    e.bindings = vec![bind(0, 0), bind(1, 0), bind(0, 1)];
    assert_eq!(
        with_entry(e).check(),
        Ok(()),
        "same index, different set, is fine"
    );
    let mut e = entry("k");
    e.bindings = vec![bind(0, 1), bind(0, 1)];
    assert_eq!(
        with_entry(e).check(),
        Err(ManifestError::DuplicateBinding {
            entry: "k".into(),
            set: 0,
            index: 1
        })
    );
}

#[test]
fn packing_needs_at_least_two_elements_per_word() {
    let packed = |n| {
        let mut e = entry("k");
        e.bindings[0].packing = Packing::Packed { elems_per_word: n };
        with_entry(e).check()
    };
    assert_eq!(packed(2), Ok(()));
    assert_eq!(
        packed(1),
        Err(ManifestError::DegeneratePacking { entry: "k".into() })
    );
    assert_eq!(
        packed(0),
        Err(ManifestError::DegeneratePacking { entry: "k".into() })
    );
}

#[test]
fn params_are_aligned_sized_and_disjoint() {
    let with = |fields: Vec<ParamField>| {
        let mut e = entry("k");
        e.params = Some(ParamsLayout::new(LayoutRule::Scalar, fields));
        with_entry(e).check()
    };
    assert_eq!(
        with(vec![
            ParamField::new("a", 0, K::F64),
            ParamField::new("b", 8, K::I32)
        ]),
        Ok(())
    );
    assert_eq!(
        with(vec![ParamField::new("a", 2, K::F32)]),
        Err(ManifestError::MisalignedParam {
            entry: "k".into(),
            field: "a".into()
        })
    );
    assert_eq!(
        with(vec![
            ParamField::new("a", 0, K::F64),
            ParamField::new("b", 4, K::I32)
        ]),
        Err(ManifestError::OverlappingParams {
            entry: "k".into(),
            a: "a".into(),
            b: "b".into()
        })
    );
    // Touching, not overlapping, and out of declaration order.
    assert_eq!(
        with(vec![
            ParamField::new("b", 4, K::I32),
            ParamField::new("a", 0, K::I32)
        ]),
        Ok(())
    );
    assert_eq!(
        with(vec![ParamField::new("q", 0, K::Fp8E5M2)]),
        Err(ManifestError::UnsizedParamType {
            entry: "k".into(),
            field: "q".into()
        })
    );
}

#[test]
fn an_entry_with_no_params_block_checks() {
    let mut e = entry("k");
    e.params = None;
    assert_eq!(with_entry(e).check(), Ok(()));
}

fn cap(key: CapKey, value: CapValue, source: CapSource) -> Capability {
    Capability::new(key, value, source)
}

#[test]
fn an_empty_set_knows_nothing_and_denies_nothing() {
    let s = CapabilitySet::new();
    assert_eq!(s.value(&CapKey::SubgroupSize), CapValue::Unknown);
    assert_eq!(s.denorm(K::F32), DenormMode::Unknown);
    assert_eq!(s.dtype_support(K::F16), DtypeSupport::NotExpressible);
    assert_eq!(DenormMode::default(), DenormMode::Unknown);
}

#[test]
fn recorded_facts_read_back_and_do_not_leak_across_keys() {
    let mut s = CapabilitySet::new();
    s.insert(cap(
        CapKey::Dtype(K::F16),
        CapValue::Bool(true),
        CapSource::Reported,
    ))
    .unwrap();
    s.insert(cap(
        CapKey::Dtype(K::F64),
        CapValue::Bool(false),
        CapSource::Measured,
    ))
    .unwrap();
    s.insert(cap(
        CapKey::Denorm(K::F32),
        CapValue::Denorm(DenormMode::Flush),
        CapSource::Measured,
    ))
    .unwrap();
    assert_eq!(s.dtype_support(K::F16), DtypeSupport::Supported);
    assert_eq!(s.dtype_support(K::F64), DtypeSupport::Unsupported);
    assert_eq!(s.dtype_support(K::Bf16), DtypeSupport::NotExpressible);
    assert_eq!(s.denorm(K::F32), DenormMode::Flush);
    assert_eq!(
        s.denorm(K::F16),
        DenormMode::Unknown,
        "another dtype's denorm is not this one's"
    );
    assert_eq!(
        s.get(&CapKey::Dtype(K::F64)).unwrap().source,
        CapSource::Measured
    );
}

#[test]
fn a_key_is_recorded_once() {
    let mut s = CapabilitySet::new();
    let k = CapKey::Dtype(K::F16);
    s.insert(cap(k.clone(), CapValue::Bool(true), CapSource::Reported))
        .unwrap();
    assert_eq!(
        s.insert(cap(k.clone(), CapValue::Bool(false), CapSource::Measured)),
        Err(CapabilityError::DuplicateKey(k.clone()))
    );
    assert_eq!(
        s.value(&k),
        CapValue::Bool(true),
        "the refused insert changed nothing"
    );
    assert_eq!(s.entries.len(), 1);
}

#[test]
fn a_value_of_the_wrong_shape_for_a_dtype_is_silence_not_denial() {
    let mut s = CapabilitySet::new();
    s.insert(cap(
        CapKey::Dtype(K::F16),
        CapValue::Unknown,
        CapSource::Declared,
    ))
    .unwrap();
    assert_eq!(s.dtype_support(K::F16), DtypeSupport::NotExpressible);
}

#[test]
fn ranges_must_be_consistent() {
    let r = |min, max, fixed| {
        CapabilitySet::new().insert(cap(
            CapKey::SubgroupSize,
            CapValue::Range { min, max, fixed },
            CapSource::Reported,
        ))
    };
    assert_eq!(r(32, 64, false), Ok(()));
    assert_eq!(r(32, 32, true), Ok(()));
    assert_eq!(
        r(64, 32, false),
        Err(CapabilityError::BadRange {
            min: 64,
            max: 32,
            fixed: false
        })
    );
    assert_eq!(
        r(32, 64, true),
        Err(CapabilityError::BadRange {
            min: 32,
            max: 64,
            fixed: true
        })
    );
}
