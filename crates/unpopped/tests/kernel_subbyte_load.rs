// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Sub-byte (`I4`, `U4`) elements in the kernel layer** (ROADMAP U9b S1, `docs/options-u9b-subbyte-dequant.md`, ruled by the PM
//! 2026-11-05): a kernel `Load`s an element from a `Read` buffer (packed two per byte, low nibble first), `Cast`s it to a wider integer
//! (`I4` sign-extends, `U4` zero-extends) and can narrow an integer back with a wrapping `Cast`. Everything else is a typed refusal:
//! a `Store` of a sub-byte element (a read-modify-write of a byte two threads share), a saturating or float-endpoint `Cast`, and
//! arithmetic on a sub-byte value (widen it first).

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped::types::TypeError;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const F32: ElementKind = ElementKind::F32;
const I4: ElementKind = ElementKind::I4;
const U4: ElementKind = ElementKind::U4;

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    }
}

fn cast(x: ScalarExpr, to: ElementKind, sat: Saturation) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(x),
        CastSpec {
            to,
            round: RoundMode::NearestEven,
            sat,
        },
    )
}

fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn eight_threads(k: KernelFn) -> KernelFn {
    k.with_launch(LaunchShape::new([8, 1, 1], [1, 1, 1]))
}

/// `out[t] = (i32) w[t]` for a `sub` buffer `w`.
fn widen(sub: ElementKind, sat: Saturation, to: ElementKind) -> KernelFn {
    eight_threads(KernelFn::new(
        "widen",
        I32,
        vec![
            Param::buffer("w", sub, BufferRole::Read, Some(1)),
            Param::buffer("out", to, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(sub)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value: cast(local(0), to, sat),
            },
        ]),
    ))
}

/// `out[t] = (i32)(sub) x[t]`: narrow an `I32` to `sub` (wrapping) into a local, then widen it back.
fn round_trip(sub: ElementKind) -> KernelFn {
    eight_threads(KernelFn::new(
        "narrow",
        I32,
        vec![
            Param::buffer("x", I32, BufferRole::Read, Some(1)),
            Param::buffer("out", I32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(I32), LocalDecl::scalar(sub)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Let {
                local: 1,
                init: cast(local(0), sub, Saturation::Wrap),
            },
            Stmt::Store {
                place: at(1),
                value: cast(local(1), I32, Saturation::Wrap),
            },
        ]),
    ))
}

/// Eight nibbles packed two per byte, low nibble first.
fn packed(nibbles: [u8; 8]) -> Vec<u8> {
    nibbles.chunks(2).map(|p| p[0] | p[1] << 4).collect()
}

fn words(a: &KernelArg) -> Vec<i32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!("a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn i32_buf(v: &[i32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        I32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

const NIBBLES: [u8; 8] = [0x8, 0xF, 0x0, 0x1, 0x7, 0x3, 0xC, 0x5];

#[test]
fn a_sub_byte_element_widens_with_sign_or_zero_extension() {
    for (sub, want) in [
        (I4, [-8, -1, 0, 1, 7, 3, -4, 5]),
        (U4, [8, 15, 0, 1, 7, 3, 12, 5]),
    ] {
        let k = widen(sub, Saturation::Wrap, I32);
        assert_eq!(check_kernel(&k), Ok(()), "{sub:?}");
        let bytes = packed(NIBBLES);
        assert_eq!(bytes, [0xF8, 0x10, 0x37, 0x5C], "low nibble first");
        let mut args = [
            KernelArg::Buffer(TypedBuffer::new(sub, vec![8], vec![1], bytes)),
            i32_buf(&[0; 8]),
        ];
        run_launch(&k, &mut args).expect("the oracle runs it");
        assert_eq!(words(&args[1]), want, "{sub:?}");
    }
}

#[test]
fn narrowing_to_a_sub_byte_type_wraps_modulo_16() {
    let x = [9, -9, 16, 7, 8, -8, 100, -1];
    for (sub, want) in [
        (I4, [-7, 7, 0, 7, -8, -8, 4, -1]),
        (U4, [9, 7, 0, 7, 8, 8, 4, 15]),
    ] {
        let k = round_trip(sub);
        assert_eq!(check_kernel(&k), Ok(()), "{sub:?}");
        let mut args = [i32_buf(&x), i32_buf(&[0; 8])];
        run_launch(&k, &mut args).expect("the oracle runs it");
        assert_eq!(words(&args[1]), want, "{sub:?}");
    }
}

/// `out[t] = (i32)(sub) x[t]` in one expression, with no local to mask the value in between.
fn nested(sub: ElementKind) -> KernelFn {
    eight_threads(KernelFn::new(
        "nested",
        I32,
        vec![
            Param::buffer("x", I32, BufferRole::Read, Some(1)),
            Param::buffer("out", I32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(I32)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value: cast(cast(local(0), sub, Saturation::Wrap), I32, Saturation::Wrap),
            },
        ]),
    ))
}

#[test]
fn a_nested_narrowing_cast_wraps_in_the_expression_itself() {
    let x = [9, -9, 16, 7, 8, -8, 100, -1];
    for (sub, want) in [
        (I4, [-7, 7, 0, 7, -8, -8, 4, -1]),
        (U4, [9, 7, 0, 7, 8, 8, 4, 15]),
    ] {
        let k = nested(sub);
        assert_eq!(check_kernel(&k), Ok(()), "{sub:?}");
        let mut args = [i32_buf(&x), i32_buf(&[0; 8])];
        run_launch(&k, &mut args).expect("the oracle runs it");
        assert_eq!(words(&args[1]), want, "{sub:?}");
    }
}

fn illegal_cast_text(k: &KernelFn) -> String {
    match check_kernel(k) {
        Err(KernelError::Type(TypeError::IllegalCast { why, .. })) => why.to_string(),
        other => panic!("expected an IllegalCast, got {other:?}"),
    }
}

#[test]
fn a_saturating_or_float_sub_byte_cast_is_a_typed_refusal() {
    let clamp = illegal_cast_text(&widen(I4, Saturation::Clamp, I32));
    assert!(clamp.contains("Saturation::Wrap"), "{clamp}");
    let float = illegal_cast_text(&widen(I4, Saturation::Wrap, F32));
    assert!(float.contains("widen to i32 first"), "{float}");
}

#[test]
fn a_store_of_a_sub_byte_element_is_refused_buffers_are_read_only_in_kernels() {
    let k = eight_threads(KernelFn::new(
        "store",
        I32,
        vec![
            Param::buffer("w", I4, BufferRole::Read, Some(1)),
            Param::buffer("out", I4, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(I4)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value: local(0),
            },
        ]),
    ));
    match check_kernel(&k) {
        Err(KernelError::TypeMismatch { what }) => {
            assert!(what.contains("read-only in kernels"), "{what}");
        }
        other => panic!("expected the read-only refusal, got {other:?}"),
    }
}

#[test]
fn arithmetic_on_a_sub_byte_value_is_not_defined_widen_it_first() {
    let k = eight_threads(KernelFn::new(
        "add",
        I32,
        vec![
            Param::buffer("w", I4, BufferRole::Read, Some(1)),
            Param::buffer("out", I32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(I4)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value: cast(
                    ScalarExpr::Add(Box::new(local(0)), Box::new(local(0))),
                    I32,
                    Saturation::Wrap,
                ),
            },
        ]),
    ));
    match check_kernel(&k) {
        Err(KernelError::TypeMismatch { what }) => {
            assert!(what.contains("widen it with a Cast first"), "{what}");
        }
        other => panic!("expected the arithmetic refusal, got {other:?}"),
    }
}
