// SPDX-License-Identifier: MIT OR Apache-2.0
//! `unpopped-cpu-c`'s `lower_kernel` states its launch interface (PM ruling 2026-11-11 on fishstringer#17): the kernel's
//! launch geometry and parameters, and no emitter-set field (the C function takes each buffer as a pointer, its shape, its
//! strides and its base offset, which is a calling convention, not a binding).

use unpopped::backend::Backend;
use unpopped::kernel_interface::KernelInterface;
use unpopped_cpu_c::CpuC;

// The fixture shared with the oracle's tests; this file uses one kernel of it.
#[allow(dead_code)]
#[path = "support/device_peer.rs"]
mod fixture;

#[test]
fn a_lowered_kernel_carries_the_interface_derived_from_it() {
    let k = fixture::rank_and_count();
    let g = CpuC.lower_kernel(&k).expect("lowers");
    let i = g.interface().expect("cpu-c states an interface");
    assert_eq!(*i, KernelInterface::of(&k));
    // the three devices of the fixture's launch are in the interface
    assert_eq!(i.launch.map(|l| l.devices), Some(3));
    assert_eq!(i.params.len(), k.params.len());
    assert!(i.bindings.is_empty() && i.requires.is_empty() && i.params_block.is_none());
}
