// SPDX-License-Identifier: MIT OR Apache-2.0
//! A binary target's words ride a `GeneratedKernel` (`GeneratedKernel::with_words`, 0.37.1).
//!
//! Vulkane's SPIR-V emitter is the first customer: a `[u32]` stream cannot ride the `String`.
//! What must hold: `new` is unchanged (no words), the words survive `generate` and its
//! provenance stamp, the revision hash and the link-registry row cover the words, and a kernel
//! without words hashes exactly as before.

use unpopped::backend::{Backend, DtypeSupport, GeneratedKernel, LowerError};
use unpopped::ir::{OpDef, input};
use unpopped::link::link_entry;
use unpopped::plan::KernelPlan;
use unpopped::try_generate;
use unpopped_vocab::{
    ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, TargetId, structure_key,
};

/// A backend whose artifact is binary: the same source text compiles to words that depend on the
/// "compiler" it was given.
struct Binary(Vec<u32>);

impl Backend for Binary {
    fn name(&self) -> &str {
        "binary"
    }
    fn provider(&self) -> &str {
        "test"
    }
    fn supports_dtype(&self, _: ElementKind, _: TargetId) -> DtypeSupport {
        DtypeSupport::Supported
    }
    fn lower(&self, _: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        Ok(GeneratedKernel::new("k".into(), "OpFunction".into()).with_words(self.0.clone()))
    }
}

fn cell() -> (OpDef, StructureKey) {
    let d = OperandDesc::new(1, &[7], &[1], ElementKind::F32, 4);
    let key = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89);
    (
        OpDef::elementwise("f", 2, &[ElementKind::F32], input(0) + input(1)),
        key,
    )
}

#[test]
fn new_carries_no_words_and_with_words_adds_them() {
    let text = GeneratedKernel::new("k".into(), "src".into());
    assert_eq!(text.words(), None);
    let bin = text.clone().with_words(vec![0x0723_0203, 7]);
    assert_eq!(bin.words(), Some(&[0x0723_0203, 7][..]));
    assert_eq!((bin.name.as_str(), bin.source.as_str()), ("k", "src"));
}

#[test]
fn the_words_survive_generate_and_its_stamp() {
    let (op, key) = cell();
    let k = try_generate(&op, &key, &Binary(vec![1, 2, 3])).expect("lowers");
    assert_eq!(k.words(), Some(&[1, 2, 3][..]));
    assert!(k.provenance().is_some(), "the stamp is still applied");
}

#[test]
fn the_revision_hash_and_link_row_cover_the_words() {
    let (op, key) = cell();
    let a = try_generate(&op, &key, &Binary(vec![1, 2, 3])).expect("lowers");
    let b = try_generate(&op, &key, &Binary(vec![1, 2, 4])).expect("lowers");
    assert_eq!(a.source, b.source, "same source text");
    assert_ne!(
        a.revision_hash(),
        b.revision_hash(),
        "different words re-mint"
    );
    assert_ne!(
        link_entry(&op, &key, &a).revision_hash,
        link_entry(&op, &key, &b).revision_hash
    );
    let text = GeneratedKernel::new("k".into(), a.source.clone());
    assert_ne!(
        text.revision_hash(),
        a.revision_hash(),
        "words are in the identity"
    );
    // A text kernel hashes as it always did: FNV-1a of its source.
    let fnv = |s: &str| {
        s.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    };
    assert_eq!(text.revision_hash(), fnv(&a.source));
    assert_eq!(link_entry(&op, &key, &text).revision_hash, fnv(&a.source));
}
