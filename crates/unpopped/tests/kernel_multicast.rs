// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U12 slice 3 (`docs/options-u12-peer-multicast.md`): `Place::Multicast` names one element on every device at
//! once. A `Store` writes all copies (`multimem.st`), an `AtomicRmw` at `System` scope with `Release` order combines
//! into all copies (`multimem.red`); a `Load` is refused (`ld_reduce` is slice 4). The expected arrays are worked by
//! hand; the fixtures are `tests/support/multicast.rs`, on `tests/support/device_peer.rs`.

use unpopped::exec::{LaunchShape, Scope};
use unpopped::kernel::{AtomicOp, KernelError, KernelFn, MemoryOrder, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch_devices;
use unpopped_vocab::ElementKind;

#[allow(dead_code)]
#[path = "support/device_peer.rs"]
mod fixture;
use fixture::{D, I32, c, load, store};
#[allow(dead_code)]
#[path = "support/multicast.rs"]
mod mc;
use mc::{all_reduce, broadcast, multicast, one};

const F32: ElementKind = ElementKind::F32;
/// Elements of every buffer: one per thread of a device (2 blocks of 2).
const N: usize = 4;

fn bytes_i32(vals: &[i32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_f32(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn buffer(dtype: ElementKind, bytes: Vec<u8>) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(dtype, vec![N as i64], vec![1], bytes))
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

fn floats(a: &KernelArg) -> Vec<f32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!()
    };
    b.bytes
        .chunks(4)
        .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}

/// Per device: `inp[i] = 10 * (device + 1) + i`, `acc` filled with `init`, `res` zeros (all `I32`).
fn int_devices(init: i32) -> Vec<Vec<KernelArg>> {
    (0..D as i32)
        .map(|d| {
            let inp: Vec<i32> = (0..N as i32).map(|i| 10 * (d + 1) + i).collect();
            vec![
                buffer(I32, bytes_i32(&inp)),
                buffer(I32, bytes_i32(&[init; N])),
                buffer(I32, bytes_i32(&[0; N])),
            ]
        })
        .collect()
}

/// Per device: `inp[i] = 0.5 * (device + 1) + 0.25 * i`, `acc` and `res` zeros (all `F32`, every value exact).
fn float_devices() -> Vec<Vec<KernelArg>> {
    (0..D as i32)
        .map(|d| {
            let inp: Vec<f32> = (0..N as i32)
                .map(|i| 0.5 * (d + 1) as f32 + 0.25 * i as f32)
                .collect();
            vec![
                buffer(F32, bytes_f32(&inp)),
                buffer(F32, bytes_f32(&[0.0; N])),
                buffer(F32, bytes_f32(&[0.0; N])),
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

/// `acc` (argument 1) and `res` (argument 2) of every device, as integers.
fn acc_res(out: &[Vec<KernelArg>]) -> Vec<(Vec<i32>, Vec<i32>)> {
    out.iter().map(|d| (ints(&d[1]), ints(&d[2]))).collect()
}

#[test]
fn a_red_add_all_reduces_integers_exactly_into_every_device() {
    // every device's `acc[i]` receives one contribution per device: 10 + 20 + 30 + 3 * i
    let out = run(
        &all_reduce(I32, AtomicOp::Add, Some(Scope::System)),
        int_devices(0),
    )
    .unwrap();
    for (acc, res) in acc_res(&out) {
        assert_eq!(acc, [60, 63, 66, 69]);
        assert_eq!(res, [60, 63, 66, 69]);
    }
}

#[test]
fn a_red_min_and_max_select_across_devices() {
    let sys = Some(Scope::System);
    let min = run(&all_reduce(I32, AtomicOp::Min, sys), int_devices(i32::MAX)).unwrap();
    let max = run(&all_reduce(I32, AtomicOp::Max, sys), int_devices(i32::MIN)).unwrap();
    for ((min_acc, _), (max_acc, _)) in acc_res(&min).into_iter().zip(acc_res(&max)) {
        assert_eq!(min_acc, [10, 11, 12, 13]);
        assert_eq!(max_acc, [30, 31, 32, 33]);
    }
}

#[test]
fn a_red_add_on_f32_lands_inside_the_float_band() {
    // 0.5 * (1 + 2 + 3) + 0.75 * i, every partial sum exact in f32; a float add has unspecified order, so the check is a
    // band, not the bits
    let out = run(
        &all_reduce(F32, AtomicOp::Add, Some(Scope::System)),
        float_devices(),
    )
    .unwrap();
    let want = [3.0f32, 3.75, 4.5, 5.25];
    for d in &out {
        for got in [floats(&d[1]), floats(&d[2])] {
            for (g, w) in got.iter().zip(want) {
                assert!((g - w).abs() <= 1e-6 * w.abs(), "{g} vs {w}");
            }
        }
    }
}

#[test]
fn a_multicast_store_writes_every_devices_copy() {
    // device 0's input, [10, 11, 12, 13], reaches all three copies
    let out = run(&broadcast(I32, Some(Scope::System)), int_devices(-1)).unwrap();
    for (acc, res) in acc_res(&out) {
        assert_eq!(acc, [10, 11, 12, 13]);
        assert_eq!(res, [10, 11, 12, 13]);
    }
    let out = run(&broadcast(F32, Some(Scope::System)), float_devices()).unwrap();
    for d in &out {
        assert_eq!(floats(&d[1]), [0.5, 0.75, 1.0, 1.25]);
    }
}

#[test]
fn a_read_of_a_multicast_result_races_without_a_system_barrier() {
    // a barrier of one device (`Device`, or a `Cluster` of that device) does not order another device's red or store
    for sync in [None, Some(Scope::Device), Some(Scope::Cluster)] {
        assert_eq!(
            run(&all_reduce(I32, AtomicOp::Add, sync), int_devices(0)).unwrap_err(),
            KernelTrap::DataRace,
            "red {sync:?}"
        );
        assert_eq!(
            run(&broadcast(I32, sync), int_devices(0)).unwrap_err(),
            KernelTrap::DataRace,
            "st {sync:?}"
        );
    }
}

#[test]
fn a_multicast_coordinate_out_of_range_is_a_trap() {
    let k = one(I32, red_at(99));
    assert_eq!(
        run(&k, int_devices(0)).unwrap_err(),
        KernelTrap::OutOfBounds { buffer: 1 }
    );
}

fn red_at(i: i64) -> Stmt {
    Stmt::AtomicRmw {
        op: AtomicOp::Add,
        place: multicast(1, c(i)),
        value: c(1),
        old: None,
        order: MemoryOrder::Release,
        scope: Scope::System,
    }
}

fn refused(k: &KernelFn) -> bool {
    matches!(
        check_kernel(k),
        Err(KernelError::MulticastUnsupported { .. })
    )
}

#[test]
fn the_checker_refuses_what_multimem_cannot_do() {
    // ld_reduce is slice 4; there is no compare-and-swap on a multicast address
    assert!(refused(&one(I32, load(0, multicast(1, c(0))))));
    assert!(refused(&one(
        I32,
        Stmt::AtomicCas {
            place: multicast(1, c(0)),
            expected: c(0),
            desired: c(1),
            old: None,
            order: MemoryOrder::Release,
            scope: Scope::System,
        }
    )));
    // a masked store would write memory the checker cannot see is guarded; use an If
    assert!(refused(&one(
        I32,
        Stmt::MaskedStore {
            place: multicast(1, c(0)),
            value: c(1),
            mask: c(1),
        }
    )));
}

/// Whether the checker refuses `red_at(0)` after `change` edits its `AtomicRmw`.
fn red_refused_with(change: impl Fn(&mut Stmt)) -> bool {
    let mut st = red_at(0);
    change(&mut st);
    refused(&one(I32, st))
}

#[test]
fn the_checker_admits_a_red_and_refuses_an_old_value_or_another_order() {
    assert_eq!(check_kernel(&one(I32, red_at(0))), Ok(()));
    assert!(
        red_refused_with(|s| {
            if let Stmt::AtomicRmw { old, .. } = s {
                *old = Some(0);
            }
        }),
        "an old value"
    );
    for order in [
        MemoryOrder::Relaxed,
        MemoryOrder::Acquire,
        MemoryOrder::AcqRel,
        MemoryOrder::SeqCst,
    ] {
        assert!(
            red_refused_with(|s| {
                if let Stmt::AtomicRmw { order: o, .. } = s {
                    *o = order;
                }
            }),
            "{order:?}"
        );
    }
}

#[test]
fn the_checker_refuses_a_red_below_system_scope_or_with_a_bitwise_op() {
    for scope in [Scope::Block, Scope::Device] {
        assert!(
            red_refused_with(|s| {
                if let Stmt::AtomicRmw { scope: sc, .. } = s {
                    *sc = scope;
                }
            }),
            "{scope:?}"
        );
    }
    for op in [
        AtomicOp::And,
        AtomicOp::Or,
        AtomicOp::Xor,
        AtomicOp::Exchange,
    ] {
        assert!(
            red_refused_with(|s| {
                if let Stmt::AtomicRmw { op: o, .. } = s {
                    *o = op;
                }
            }),
            "{op:?}"
        );
    }
}

#[test]
fn the_checker_admits_only_the_elements_multimem_has() {
    // a 16-bit float Add is a real multimem.red the oracle does not model yet: refused, named
    for elem in [ElementKind::F16, ElementKind::Bf16, ElementKind::I8, F32] {
        let st = Stmt::AtomicRmw {
            op: if elem == F32 {
                AtomicOp::Max
            } else {
                AtomicOp::Add
            },
            place: multicast(1, c(0)),
            value: c(0),
            old: None,
            order: MemoryOrder::Release,
            scope: Scope::System,
        };
        assert!(refused(&one(elem, st)), "red on {elem:?}");
    }
    // a Store takes I32, U32, F32, F16 and Bf16 and nothing narrower
    assert!(refused(&one(
        ElementKind::I8,
        store(multicast(1, c(0)), c(0))
    )));
}

#[test]
fn a_multicast_place_needs_a_launch_of_several_devices() {
    let k = one(I32, store(multicast(1, c(0)), c(0)))
        .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    assert!(matches!(
        check_kernel(&k),
        Err(KernelError::PeerUnsupported { .. })
    ));
}
