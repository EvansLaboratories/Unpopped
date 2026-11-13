// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of a multicast store and reduction agrees with the launch interpreter** (ROADMAP U12 slice 3, the
//! cpu-c reference): a `Place::Multicast` `Store` or `AtomicRmw` is spelled as a loop over the devices of the launch,
//! each iteration a lookup in the table of that device's buffers. The semantics are
//! `crates/unpopped/tests/kernel_multicast.rs`; this file compiles the C with a host compiler, runs it, and compares the
//! written bytes of every device with the interpreter's. The seeded mutants of the emitted C each have to change them.

use unpopped::exec::Scope;
use unpopped::kernel::{AtomicOp, KernelFn, MemoryOrder};
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
use mc::{all_reduce, broadcast, ld_reduce_then_st, reduce_inputs, st_then_ld_reduce};

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

/// The 16-bit pattern of `x`, which must be exactly representable (a normal number with at most 10 or 7 fraction bits
/// and an exponent in the narrow format's range).
fn narrow_bits(x: f32, dt: ElementKind) -> u16 {
    let bits = x.to_bits();
    if dt == ElementKind::Bf16 {
        assert_eq!(bits & 0xFFFF, 0, "{x} is not a bfloat16");
        return (bits >> 16) as u16;
    }
    assert_eq!(bits & 0x1FFF, 0, "{x} is not a binary16");
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    assert!(
        x == 0.0 || (1..31).contains(&exp),
        "{x} is outside the normal binary16 range"
    );
    if x == 0.0 {
        return sign as u16;
    }
    (sign | ((exp as u32) << 10) | ((bits & 0x7F_FFFF) >> 13)) as u16
}

/// Per device, a 16-bit float's `inp`, `acc` and `res`: values chosen so that device 0 holds `2048` (or `256` for
/// bfloat16) and the others `1`, where adding the devices in turn at 16 bits would round away what one `f32` sum keeps.
fn narrow_devices(dt: ElementKind) -> Vec<Vec<KernelArg>> {
    let le = |v: &[f32]| {
        v.iter()
            .flat_map(|x| narrow_bits(*x, dt).to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let big = if dt == ElementKind::F16 {
        2048.0
    } else {
        256.0
    };
    (0..D as i32)
        .map(|d| {
            let inp = [
                if d == 0 { big } else { 1.0 },
                0.5 * (d + 1) as f32,
                -1.5 * d as f32,
                3.0 - d as f32,
            ];
            vec![
                buffer(dt, le(&inp)),
                buffer(dt, le(&[0.0; N])),
                buffer(dt, le(&[0.0; N])),
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
        (
            "mc_st_f16",
            broadcast(ElementKind::F16, sys),
            narrow_devices(ElementKind::F16),
        ),
        (
            "mc_st_bf16",
            broadcast(ElementKind::Bf16, sys),
            narrow_devices(ElementKind::Bf16),
        ),
    ]
}

/// The `ld_reduce` kernels (ROADMAP U12 slice 4) with their devices' arguments, by name. A 16-bit element is decoded to
/// `float`, combined there and encoded once (ROADMAP U14), as the oracle does.
fn reduce_kernels() -> Vec<(&'static str, KernelFn, Vec<Vec<KernelArg>>)> {
    let acq = MemoryOrder::Acquire;
    vec![
        (
            "mc_ld_add",
            reduce_inputs(I32, (AtomicOp::Add, acq)),
            int_devices(0),
        ),
        (
            "mc_ld_min",
            reduce_inputs(I32, (AtomicOp::Min, MemoryOrder::Relaxed)),
            int_devices(0),
        ),
        (
            "mc_ld_max",
            reduce_inputs(I32, (AtomicOp::Max, acq)),
            int_devices(0),
        ),
        (
            "mc_ld_f32",
            reduce_inputs(F32, (AtomicOp::Add, acq)),
            float_devices(),
        ),
        (
            "mc_st_ld",
            st_then_ld_reduce(I32, (AtomicOp::Add, acq), Some(Scope::System)),
            int_devices(0),
        ),
        (
            "mc_ld_st",
            ld_reduce_then_st(I32, (AtomicOp::Add, acq)),
            int_devices(0),
        ),
    ]
    .into_iter()
    .chain(narrow_reduce_kernels())
    .collect()
}

/// `ld_reduce` of `F16` and `Bf16` for `Add`, `Min` and `Max`, and the store-then-reduce form.
fn narrow_reduce_kernels() -> Vec<(&'static str, KernelFn, Vec<Vec<KernelArg>>)> {
    let mut out = Vec::new();
    for (name, dt) in [("f16", ElementKind::F16), ("bf16", ElementKind::Bf16)] {
        for (op_name, op) in [
            ("add", AtomicOp::Add),
            ("min", AtomicOp::Min),
            ("max", AtomicOp::Max),
        ] {
            let tag: &'static str = Box::leak(format!("mc_ld_{op_name}_{name}").into_boxed_str());
            out.push((
                tag,
                reduce_inputs(dt, (op, MemoryOrder::Acquire)),
                narrow_devices(dt),
            ));
        }
        let tag: &'static str = Box::leak(format!("mc_st_ld_{name}").into_boxed_str());
        out.push((
            tag,
            st_then_ld_reduce(
                dt,
                (AtomicOp::Add, MemoryOrder::Acquire),
                Some(Scope::System),
            ),
            narrow_devices(dt),
        ));
    }
    out
}

#[test]
fn a_multicast_store_and_reduction_agree_with_the_launch_interpreter() {
    for (tag, k, args) in kernels().into_iter().chain(reduce_kernels()) {
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
    let mut all = store_and_red_mutants();
    all.extend(ld_reduce_mutants());
    all.extend(narrow_ld_reduce_mutants());
    all
}

/// Mutants of a multicast `Store` and `AtomicRmw` (slice 3).
fn store_and_red_mutants() -> Vec<Mutant> {
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

/// Mutants of a `MulticastLoadReduce` (slice 4): the devices' inputs reduced by `Add` (60 + 3 * i) and by `Min` (device 0
/// holds the minimum, so a mutant that reads only device 0 would go unnoticed by `Min`: each mutant's data has to tell it
/// apart).
/// The mutants of the 16-bit `ld_reduce` spelling (ROADMAP U14).
fn narrow_ld_reduce_mutants() -> Vec<Mutant> {
    let (add, min) = (
        (AtomicOp::Add, MemoryOrder::Acquire),
        (AtomicOp::Min, MemoryOrder::Relaxed),
    );
    vec![
        // a 16-bit minimum becomes a maximum
        Mutant {
            tag: "mc_mut_ld_min_f16",
            kernel: reduce_inputs(ElementKind::F16, (min.0, min.1)),
            args: narrow_devices(ElementKind::F16),
            from: "fminf(r0v, x0)",
            to: "fmaxf(r0v, x0)",
        },
        // a 16-bit sum rounds to the element after every device instead of once
        Mutant {
            tag: "mc_mut_ld_round_each_f16",
            kernel: reduce_inputs(ElementKind::F16, (add.0, add.1)),
            args: narrow_devices(ElementKind::F16),
            from: "r0v + x0",
            to: "unpopped_f16_load(unp_f16_enc(r0v + x0))",
        },
    ]
}

fn ld_reduce_mutants() -> Vec<Mutant> {
    let ld = |tag, op, order, from, to| Mutant {
        tag,
        kernel: reduce_inputs(I32, (op, order)),
        args: int_devices(0),
        from,
        to,
    };
    let (add, min) = (
        (AtomicOp::Add, MemoryOrder::Acquire),
        (AtomicOp::Min, MemoryOrder::Relaxed),
    );
    vec![
        // the reduction reads the first device's copy only
        ld(
            "mc_mut_ld_one",
            add.0,
            add.1,
            "md0 < 3; ++md0",
            "md0 < 1; ++md0",
        ),
        // the first device's value is combined with the zero it starts from
        ld(
            "mc_mut_ld_seed",
            min.0,
            min.1,
            "md0 == 0 ? x0",
            "md0 < 0 ? x0",
        ),
        // a minimum becomes a maximum
        ld(
            "mc_mut_ld_op",
            min.0,
            min.1,
            "r0v < x0 ? r0v : x0",
            "r0v > x0 ? r0v : x0",
        ),
        // every device reads the running device's own copy
        ld(
            "mc_mut_ld_self",
            add.0,
            add.1,
            "x0 = btab0[md0][o0];",
            "x0 = btab0[dev][o0];",
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
