// SPDX-License-Identifier: MIT OR Apache-2.0
//! CpuC does not compute the OCP MX element dtypes (KISS-CLASSIFY §6.1-0014): they
//! are storage and key identity until a codec lands. Storage support and compute
//! support are different facts (see `storage_is_not_compute.rs`).

use unpopped::backend::Backend;
use unpopped::cfamily::scalar_ctype;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::{ArchSku, ElementKind, TargetId};

#[test]
fn cpu_c_declines_to_compute_an_mx_element_and_spells_no_c_type_for_it() {
    let target = TargetId::from(ArchSku::Sm89);
    for dt in [
        ElementKind::F4E2M1,
        ElementKind::F6E2M3,
        ElementKind::F6E3M2,
    ] {
        assert!(!CpuC.supports_dtype(dt, target).is_supported(), "{dt:?}");
        assert!(scalar_ctype(dt).is_none(), "{dt:?}");
    }
    // Control: a computed dtype is supported and spelled.
    assert!(CpuC.supports_dtype(ElementKind::F32, target).is_supported());
    assert!(scalar_ctype(ElementKind::F32).is_some());
}
