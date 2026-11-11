// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U12 slice 4 (`docs/options-u12-peer-multicast.md`): `Stmt::MulticastLoadReduce`, `multimem.ld_reduce`: a local
//! takes the reduction by `Add`, `Min` or `Max` over every device's copy of one element. Integers are exact; a float
//! `Add` is one rounded add per device (its order is unspecified in a band); `F16` and `Bf16` accumulate in `f32` and
//! round once. Each copy is read as an atomic access, and an `Acquire` takes in a `System` release. The expected values
//! are worked by hand; the fixtures are `tests/support/multicast.rs`.

use half::{bf16, f16};
use unpopped::exec::{LaunchShape, Scope};
use unpopped::kernel::{AtomicOp, KernelError, KernelFn, MemoryOrder, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch_devices;
use unpopped_vocab::ElementKind;

#[allow(dead_code)]
#[path = "support/device_peer.rs"]
mod fixture;
use fixture::{D, I32, c};
#[allow(dead_code)]
#[path = "support/multicast.rs"]
mod mc;
use mc::{
    acquire_handoff, ld_reduce, ld_reduce_then_st, one, red_then_ld_reduce, reduce_inputs,
    st_then_ld_reduce,
};

const F32: ElementKind = ElementKind::F32;
const N: usize = 4;
const ACQ: MemoryOrder = MemoryOrder::Acquire;

fn buffer(dtype: ElementKind, bytes: Vec<u8>) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(dtype, vec![N as i64], vec![1], bytes))
}

fn le32<T: Copy>(vals: &[T], f: impl Fn(T) -> [u8; 4]) -> Vec<u8> {
    vals.iter().flat_map(|v| f(*v)).collect()
}

fn ints(a: &KernelArg) -> Vec<i32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!()
    };
    b.bytes
        .chunks(4)
        .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}

fn bits16(a: &KernelArg) -> Vec<u16> {
    let KernelArg::Buffer(b) = a else {
        unreachable!()
    };
    b.bytes
        .chunks(2)
        .map(|w| u16::from_le_bytes(w.try_into().unwrap()))
        .collect()
}

/// Per device: `inp[i] = 10 * (device + 1) + i`, `acc` and `res` zeros (all `I32`).
fn int_devices() -> Vec<Vec<KernelArg>> {
    (0..D as i32)
        .map(|d| {
            let inp: Vec<i32> = (0..N as i32).map(|i| 10 * (d + 1) + i).collect();
            let zero = le32(&[0i32; N], i32::to_le_bytes);
            vec![
                buffer(I32, le32(&inp, i32::to_le_bytes)),
                buffer(I32, zero.clone()),
                buffer(I32, zero),
            ]
        })
        .collect()
}

/// Per device: `inp[i] = 0.5 * (device + 1) + 0.25 * i` (all `F32`, every value exact), `acc` and `res` zeros.
fn float_devices() -> Vec<Vec<KernelArg>> {
    (0..D as i32)
        .map(|d| {
            let inp: Vec<f32> = (0..N as i32)
                .map(|i| 0.5 * (d + 1) as f32 + 0.25 * i as f32)
                .collect();
            let zero = le32(&[0f32; N], f32::to_le_bytes);
            vec![
                buffer(F32, le32(&inp, f32::to_le_bytes)),
                buffer(F32, zero.clone()),
                buffer(F32, zero),
            ]
        })
        .collect()
}

/// Per device, 16-bit: every element of `inp` is `first`, `second`, `third` on devices 0, 1, 2; `acc` and `res` zeros.
fn half_devices(elem: ElementKind, vals: [f32; 3]) -> Vec<Vec<KernelArg>> {
    let enc = |x: f32| -> [u8; 2] {
        match elem {
            ElementKind::F16 => f16::from_f32(x).to_bits().to_le_bytes(),
            _ => bf16::from_f32(x).to_bits().to_le_bytes(),
        }
    };
    (0..D as usize)
        .map(|d| {
            let inp: Vec<u8> = (0..N).flat_map(|_| enc(vals[d])).collect();
            vec![
                buffer(elem, inp),
                buffer(elem, vec![0; 2 * N]),
                buffer(elem, vec![0; 2 * N]),
            ]
        })
        .collect()
}

/// Run `k` on `devices`, the checker first; every device's arguments afterwards, or the trap.
fn run(k: &KernelFn, mut devices: Vec<Vec<KernelArg>>) -> Result<Vec<Vec<KernelArg>>, KernelTrap> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    run_launch_devices(k, &mut devices)?;
    Ok(devices)
}

#[test]
fn an_ld_reduce_reduces_the_devices_values_into_every_device() {
    // device d holds 10 * (d + 1) + i: the sum over three devices is 60 + 3 * i
    for (op, want) in [
        (AtomicOp::Add, [60, 63, 66, 69]),
        (AtomicOp::Min, [10, 11, 12, 13]),
        (AtomicOp::Max, [30, 31, 32, 33]),
    ] {
        for order in [MemoryOrder::Relaxed, ACQ] {
            let out = run(&reduce_inputs(I32, (op, order)), int_devices()).unwrap();
            for d in &out {
                assert_eq!(ints(&d[2]), want, "{op:?} {order:?}");
            }
        }
    }
}

#[test]
fn an_ld_reduce_add_on_f32_lands_inside_the_float_band() {
    // 0.5 * (1 + 2 + 3) + 0.75 * i, every partial sum exact in f32
    let out = run(&reduce_inputs(F32, (AtomicOp::Add, ACQ)), float_devices()).unwrap();
    for d in &out {
        let KernelArg::Buffer(b) = &d[2] else {
            unreachable!()
        };
        let got: Vec<f32> = b
            .bytes
            .chunks(4)
            .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        for (g, w) in got.iter().zip([3.0f32, 3.75, 4.5, 5.25]) {
            assert!((g - w).abs() <= 1e-6 * w.abs(), "{g} vs {w}");
        }
    }
}

#[test]
fn a_16_bit_sum_accumulates_in_f32_and_rounds_once() {
    // 2048 + 1 + 1: two f16 adds in turn give 2048 twice (2049 is a tie and rounds to even), one f32 sum 2050 rounds
    // to the f16 2050; bf16: 256 + 1 + 1, the same way, gives 258
    for (elem, vals, want, naive) in [
        (
            ElementKind::F16,
            [2048.0, 1.0, 1.0],
            f16::from_f32(2050.0).to_bits(),
            f16::from_f32(2048.0).to_bits(),
        ),
        (
            ElementKind::Bf16,
            [256.0, 1.0, 1.0],
            bf16::from_f32(258.0).to_bits(),
            bf16::from_f32(256.0).to_bits(),
        ),
    ] {
        assert_ne!(want, naive);
        let out = run(
            &reduce_inputs(elem, (AtomicOp::Add, ACQ)),
            half_devices(elem, vals),
        )
        .unwrap();
        for d in &out {
            assert_eq!(bits16(&d[2]), [want; N], "{elem:?}");
        }
    }
}

#[test]
fn a_16_bit_min_and_max_select_across_devices() {
    for elem in [ElementKind::F16, ElementKind::Bf16] {
        let enc = |x: f32| match elem {
            ElementKind::F16 => f16::from_f32(x).to_bits(),
            _ => bf16::from_f32(x).to_bits(),
        };
        for (op, want) in [(AtomicOp::Min, 1.0), (AtomicOp::Max, 3.0)] {
            let out = run(
                &reduce_inputs(elem, (op, ACQ)),
                half_devices(elem, [3.0, 1.0, 2.0]),
            )
            .unwrap();
            assert_eq!(bits16(&out[0][2]), [enc(want); N], "{elem:?} {op:?}");
        }
    }
}

#[test]
fn the_second_all_reduce_form_is_an_ld_reduce_then_a_multicast_store() {
    // device 0 reduces the three inputs and writes the sum into every device's `acc`
    let out = run(&ld_reduce_then_st(I32, (AtomicOp::Add, ACQ)), int_devices()).unwrap();
    for d in &out {
        assert_eq!(ints(&d[1]), [60, 63, 66, 69]);
        assert_eq!(ints(&d[2]), [60, 63, 66, 69]);
    }
}

#[test]
fn an_ld_reduce_after_a_multicast_store_needs_a_system_barrier() {
    // device 0's `inp` is [10, 11, 12, 13] and lands in all three copies: the sum is three times that
    let out = run(
        &st_then_ld_reduce(I32, (AtomicOp::Add, ACQ), Some(Scope::System)),
        int_devices(),
    )
    .unwrap();
    assert_eq!(ints(&out[1][2]), [30, 33, 36, 39]);
    for sync in [None, Some(Scope::Device), Some(Scope::Cluster)] {
        assert_eq!(
            run(
                &st_then_ld_reduce(I32, (AtomicOp::Add, ACQ), sync),
                int_devices()
            )
            .unwrap_err(),
            KernelTrap::DataRace,
            "{sync:?}"
        );
    }
}

#[test]
fn an_acquire_takes_in_a_system_release_and_a_relaxed_read_does_not() {
    let out = run(&acquire_handoff(ACQ), int_devices()).unwrap();
    for d in &out[1..] {
        assert_eq!(ints(&d[2])[0], 7);
    }
    assert_eq!(
        run(&acquire_handoff(MemoryOrder::Relaxed), int_devices()).unwrap_err(),
        KernelTrap::DataRace
    );
}

#[test]
fn an_ld_reduce_of_an_element_a_red_updates_is_not_a_race() {
    // atomics do not race with each other; the value read depends on the schedule, so only the run is checked
    assert!(run(&red_then_ld_reduce(I32), int_devices()).is_ok());
}

fn refused(k: &KernelFn) -> bool {
    matches!(
        check_kernel(k),
        Err(KernelError::MulticastUnsupported { .. })
    )
}

#[test]
fn the_checker_admits_only_the_ld_reduce_forms_multimem_has() {
    let form = |elem, op, order| one(elem, ld_reduce(0, 0, (op, order)));
    assert_eq!(
        check_kernel(&form(I32, AtomicOp::Add, ACQ)),
        Ok(()),
        "the baseline"
    );
    // no release or ordered read-modify-write on a read
    for order in [
        MemoryOrder::Release,
        MemoryOrder::AcqRel,
        MemoryOrder::SeqCst,
    ] {
        assert!(refused(&form(I32, AtomicOp::Add, order)), "{order:?}");
    }
    for op in [
        AtomicOp::And,
        AtomicOp::Or,
        AtomicOp::Xor,
        AtomicOp::Exchange,
    ] {
        assert!(refused(&form(I32, op, ACQ)), "{op:?}");
    }
    // f32 only adds; 8-bit and 64-bit elements have no ld_reduce here
    assert!(refused(&form(F32, AtomicOp::Min, ACQ)));
    assert!(refused(&form(F32, AtomicOp::Max, ACQ)));
    for elem in [ElementKind::I8, ElementKind::I64, ElementKind::U8] {
        assert!(refused(&form(elem, AtomicOp::Add, ACQ)), "{elem:?}");
    }
}

#[test]
fn the_checker_refuses_an_ld_reduce_off_a_multicast_place_or_in_one_device() {
    let on_logical = one(
        I32,
        Stmt::MulticastLoadReduce {
            local: 0,
            place: fixture::at(0, c(0)),
            op: AtomicOp::Add,
            order: ACQ,
        },
    );
    assert!(refused(&on_logical));
    let one_device = one(I32, ld_reduce(0, 0, (AtomicOp::Add, ACQ)))
        .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    assert!(matches!(
        check_kernel(&one_device),
        Err(KernelError::PeerUnsupported { .. })
    ));
}

#[test]
fn the_checker_refuses_an_ld_reduce_of_a_write_only_buffer() {
    assert_eq!(
        check_kernel(&mc::ld_reduce_of_a_write_only_buffer()),
        Err(KernelError::ReadFromWriteOnly { buffer: 0 })
    );
}

#[test]
fn an_ld_reduce_result_is_divergent_so_a_barrier_under_it_is_refused() {
    assert_eq!(
        check_kernel(&mc::barrier_on_a_reduced_value()),
        Err(KernelError::BarrierDivergent)
    );
}
