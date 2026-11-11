// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of a multicast store and reduction agrees with the launch interpreter** (ROADMAP U12 slice 3, the
//! cpu-c reference): a `Place::Multicast` `Store` or `AtomicRmw` is spelled as a loop over the devices of the launch,
//! each iteration a lookup in the table of that device's buffers. The semantics are
//! `crates/unpopped/tests/kernel_multicast.rs`; this file compiles the C with a host compiler, runs it, and compares the
//! written bytes of every device with the interpreter's. The seeded mutants of the emitted C each have to change them.

use unpopped::exec::Scope;
use unpopped::kernel::{AtomicOp, KernelFn};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch_devices;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
// The shared compile-and-run drivers; this file uses a subset of them.
#[allow(dead_code)]
#[path = "support/devices_run.rs"]
mod devices_run;
#[allow(dead_code)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[allow(dead_code)]
#[path = "support/launch_run.rs"]
mod launch_run;
// The fixtures shared with the oracle's test; each file uses part of them.
#[allow(dead_code)]
#[path = "support/device_peer.rs"]
mod fixture;
#[allow(dead_code)]
#[path = "support/multicast.rs"]
mod mc;
use devices_run::{run_c_devices, written_devices};
use fixture::{D, I32};
use mc::{all_reduce, broadcast};

const F32: ElementKind = ElementKind::F32;
/// Elements of every buffer: one per thread of a device (2 blocks of 2).
const N: usize = 4;

fn buffer(dtype: ElementKind, bytes: Vec<u8>) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(dtype, vec![N as i64], vec![1], bytes))
}

/// Per device: `inp[i] = 10 * (device + 1) + i`, `acc` filled with `init`, `res` zeros (all `I32`).
fn int_devices(init: i32) -> Vec<Vec<KernelArg>> {
    let le = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    (0..D as i32)
        .map(|d| {
            let inp: Vec<i32> = (0..N as i32).map(|i| 10 * (d + 1) + i).collect();
            vec![
                buffer(I32, le(&inp)),
                buffer(I32, le(&[init; N])),
                buffer(I32, le(&[0; N])),
            ]
        })
        .collect()
}

/// Per device: `inp[i] = 0.5 * (device + 1) + 0.25 * i`, `acc` and `res` zeros (all `F32`, every value exact).
fn float_devices() -> Vec<Vec<KernelArg>> {
    let le = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    (0..D as i32)
        .map(|d| {
            let inp: Vec<f32> = (0..N as i32)
                .map(|i| 0.5 * (d + 1) as f32 + 0.25 * i as f32)
                .collect();
            vec![
                buffer(F32, le(&inp)),
                buffer(F32, le(&[0.0; N])),
                buffer(F32, le(&[0.0; N])),
            ]
        })
        .collect()
}

/// Every kernel of the fixture with its devices' arguments, by name.
fn kernels() -> Vec<(&'static str, KernelFn, Vec<Vec<KernelArg>>)> {
    let sys = Some(Scope::System);
    vec![
        (
            "mc_red_add",
            all_reduce(I32, AtomicOp::Add, sys),
            int_devices(0),
        ),
        (
            "mc_red_min",
            all_reduce(I32, AtomicOp::Min, sys),
            int_devices(i32::MAX),
        ),
        (
            "mc_red_max",
            all_reduce(I32, AtomicOp::Max, sys),
            int_devices(i32::MIN),
        ),
        (
            "mc_red_f32",
            all_reduce(F32, AtomicOp::Add, sys),
            float_devices(),
        ),
        ("mc_st", broadcast(I32, sys), int_devices(-1)),
        ("mc_st_f32", broadcast(F32, sys), float_devices()),
    ]
}

#[test]
fn a_multicast_store_and_reduction_agree_with_the_launch_interpreter() {
    for (tag, k, args) in kernels() {
        devices_run::agree_devices(tag, &k, args);
    }
}

/// A seeded mutant of the emitted C: its name, the kernel and its arguments, and the text to replace and what replaces it.
struct Mutant {
    tag: &'static str,
    kernel: KernelFn,
    args: Vec<Vec<KernelArg>>,
    from: &'static str,
    to: &'static str,
}

fn mutants() -> Vec<Mutant> {
    let sys = Some(Scope::System);
    let red = |tag, from, to| Mutant {
        tag,
        kernel: all_reduce(I32, AtomicOp::Add, sys),
        args: int_devices(0),
        from,
        to,
    };
    let st = |tag, from, to| Mutant {
        tag,
        kernel: broadcast(I32, sys),
        args: int_devices(-1),
        from,
        to,
    };
    vec![
        // the reduction reaches the first device's copy only
        red("mc_mut_red_one", "md1 < 3; ++md1", "md1 < 1; ++md1"),
        // every device combines into the running device's own copy
        red(
            "mc_mut_red_self",
            "btab1[md1][o1] = (int)",
            "btab1[dev][o1] = (int)",
        ),
        // the store reaches the first device's copy only
        st("mc_mut_st_one", "md1 < 3; ++md1", "md1 < 1; ++md1"),
        // the store writes the running device's copy, not each in turn
        st(
            "mc_mut_st_self",
            "btab1[md1][o1] = mv1;",
            "btab1[dev][o1] = mv1;",
        ),
    ]
}

#[test]
fn the_corpus_catches_seeded_mutants_of_the_emitted_multicast_code() {
    for Mutant {
        tag,
        kernel,
        args,
        from,
        to,
    } in mutants()
    {
        let mut want = args.clone();
        run_launch_devices(&kernel, &mut want).expect("the oracle runs it");
        let (want, _) = written_devices(&kernel, &want);
        let g = unpopped::backend::Backend::lower_kernel(&unpopped_cpu_c::CpuC, &kernel)
            .expect("lowers");
        assert_eq!(
            g.source.matches(from).count(),
            1,
            "{tag}: the anchor `{from}` is in the emitted C once:\n{}",
            g.source
        );
        let Some(ran) = run_c_devices(tag, &kernel, &args, |s| s.replacen(from, to, 1)) else {
            return;
        };
        assert_ne!(ran.outs, want, "{tag}: the mutant went unnoticed");
    }
}
