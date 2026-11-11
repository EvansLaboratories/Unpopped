// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of a multi-device launch agrees with the launch interpreter** (ROADMAP U12 slice 2, the cpu-c
//! reference): the C function takes every device's buffers, runs the blocks of every device in one pass per phase
//! (devices in order, each device's blocks in ascending order, as `run_launch_devices` does), and spells a
//! `Place::Peer` as a lookup in a table of the devices' buffers. The semantics are
//! `crates/unpopped/tests/kernel_device_peer.rs`; this file compiles the C with a host compiler, runs it, and compares the
//! written bytes of every device with the interpreter's. The seeded mutants of the emitted C each have to change them.

use unpopped::kernel::{KernelFn, MemoryOrder};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch_devices;

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
// The fixture shared with the oracle's test; each file uses part of it.
#[allow(dead_code)]
#[path = "support/device_peer.rs"]
mod fixture;
use devices_run::{agree_devices, run_c_devices, written_devices};
use fixture::{D, I32, c};
use unpopped::exec::Scope;

fn buffer(vals: &[i32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        I32,
        vec![vals.len() as i64],
        vec![1],
        vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
    ))
}

/// The arguments of `n` devices: `inp[i]` of device `d` is `base + 10 * (d + 1) + i`, the other buffers are zeros.
fn devices(n: i32, base: i32, inp_len: usize, others: &[usize]) -> Vec<Vec<KernelArg>> {
    (0..n)
        .map(|d| {
            let inp: Vec<i32> = (0..inp_len as i32)
                .map(|i| base + 10 * (d + 1) + i)
                .collect();
            let mut args = vec![buffer(&inp)];
            args.extend(others.iter().map(|len| buffer(&vec![0; *len])));
            args
        })
        .collect()
}

/// Every kernel of the fixture with its devices' arguments, by name.
fn kernels() -> Vec<(&'static str, KernelFn, Vec<Vec<KernelArg>>)> {
    let sys = Scope::System;
    vec![
        (
            "dev_rank_count",
            fixture::rank_and_count(),
            devices(D as i32, 0, 1, &[1]),
        ),
        (
            "dev_all_gather",
            fixture::all_gather(Some(sys)),
            devices(D as i32, 0, 4, &[12, 4]),
        ),
        (
            "dev_chain",
            fixture::chain((MemoryOrder::Release, sys), (MemoryOrder::Acquire, sys)),
            devices(D as i32, 0, 1, &[1, 1]),
        ),
        (
            "dev_device_barrier",
            fixture::device_barrier(),
            devices(2, 0, 1, &[2, 2]),
        ),
        (
            "dev_read_own",
            fixture::read_device(c(0)),
            devices(2, 90, 1, &[1]),
        ),
        (
            "dev_read_peer",
            fixture::read_device(c(1)),
            devices(2, 90, 1, &[1]),
        ),
    ]
}

#[test]
fn a_multi_device_launch_agrees_with_the_launch_interpreter() {
    for (tag, k, args) in kernels() {
        agree_devices(tag, &k, args);
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
    let chain = || {
        let sys = Scope::System;
        fixture::chain((MemoryOrder::Release, sys), (MemoryOrder::Acquire, sys))
    };
    let args = || devices(D as i32, 0, 1, &[1, 1]);
    let m = |tag, from, to| Mutant {
        tag,
        kernel: chain(),
        args: args(),
        from,
        to,
    };
    vec![
        // the partial is sent to the running device, not the next one
        m(
            "dev_mut_peer",
            "btab1[pd4][o4] = l1[t];",
            "btab1[dev][o4] = l1[t];",
        ),
        // every device believes it is device 0
        m(
            "dev_mut_rank",
            "(((unsigned int)dev))",
            "(((unsigned int)0))",
        ),
        // every device reads the buffers of device 0
        m("dev_mut_alias", "btab0[dev]", "btab0[0]"),
        // the last device never runs
        m("dev_mut_loop", "dev < 3; ++dev", "dev < 2; ++dev"),
    ]
}

#[test]
fn the_corpus_catches_seeded_mutants_of_the_emitted_device_code() {
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
