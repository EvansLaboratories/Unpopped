// SPDX-License-Identifier: MIT OR Apache-2.0
//! The in-tree C backend overrides only `lower`, so a caller's `CallerCaps` must not change what it emits
//! (`docs/options-caller-caps.md`): the same kernel through `try_generate`, `try_generate_with(LEGACY)` and
//! `try_generate_with(with_interface())`, compared field by field. Control: the source of two different ops differs,
//! so the comparison can tell kernels apart.

use unpopped::backend::CallerCaps;
use unpopped::ir::{OpDef, input};
use unpopped::{try_generate, try_generate_with};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

fn gen_with(op: &OpDef, caps: Option<CallerCaps>) -> unpopped::GeneratedKernel {
    let d = OperandDesc::new(1, &[7], &[1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89);
    match caps {
        None => try_generate(op, &key, &CpuC),
        Some(c) => try_generate_with(op, &key, &CpuC, &c),
    }
    .expect("cpu_c lowers an f32 elementwise op")
}

#[test]
fn caps_do_not_change_what_cpu_c_emits() {
    let add = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
    let mul = OpDef::elementwise("mul", 2, &[ElementKind::F32], input(0) * input(1));
    let plain = gen_with(&add, None);
    for caps in [CallerCaps::LEGACY, CallerCaps::LEGACY.with_interface()] {
        let k = gen_with(&add, Some(caps));
        assert_eq!(plain.name, k.name);
        assert_eq!(plain.source, k.source);
        assert_eq!(plain.interface(), k.interface());
        assert_eq!(plain.provenance(), k.provenance());
        assert_eq!(plain.revision_hash(), k.revision_hash());
    }
    assert_ne!(
        plain.source,
        gen_with(&mul, None).source,
        "control: different ops emit different source"
    );
}
