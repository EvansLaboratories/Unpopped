// COPY of `crates/unpopped/tests/support/padding_modes.rs`: a test may not `#[path]` into another crate's directory (sweep SUP-004).
// Change both files together.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The kernels of `kernel_padding_modes.rs`: one 1-D window read of `IN_LEN` elements at positions `-PAD..IN_LEN + PAD`,
//! written to `out[0..TAPS]`, in each padding mode the grid's C4b row names. Zero, NaN and constant fill are a
//! [`Stmt::MaskedLoad`]; replicate (clamp) and reflect are an index remap in front of an ordinary load.

use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const I32: ElementKind = ElementKind::I32;
pub(crate) const IN_LEN: i64 = 5;
pub(crate) const PAD: i64 = 2;
pub(crate) const TAPS: i64 = IN_LEN + 2 * PAD;

/// What to do for a position outside `0..IN_LEN`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Mode {
    /// The value `0.0`.
    Zero,
    /// A NaN.
    Nan,
    /// A constant.
    Constant(f64),
    /// The nearest element (clamp, replicate).
    Replicate,
    /// Mirror about the edge element, which is not repeated (reflect101: `-1 -> 1`, `n -> n - 2`).
    Reflect,
}

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn bin(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}

fn sub(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Sub(Box::new(a), Box::new(b))
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

/// Local 0 is the tap `t`, local 1 the source position `t - PAD`, local 2 the value.
pub(crate) fn window(mode: Mode) -> KernelFn {
    let t = ScalarExpr::Local(0);
    let pos = ScalarExpr::Local(1);
    // an integer that is non-zero exactly when `0 <= pos < IN_LEN`: max(0, min(pos + 1, IN_LEN - pos)). (A comparison of
    // integers is a float 0/1, which an integer `Select` cannot take as its condition, so these remaps use Min and Max.)
    let in_range = bin(
        BinaryOp::Max,
        ci(0),
        bin(
            BinaryOp::Min,
            ScalarExpr::Add(Box::new(pos.clone()), Box::new(ci(1))),
            sub(ci(IN_LEN), pos.clone()),
        ),
    );
    let load = match mode {
        Mode::Zero | Mode::Nan | Mode::Constant(_) => {
            let fill = match mode {
                Mode::Zero => 0.0,
                Mode::Nan => f64::NAN,
                Mode::Constant(c) => c,
                _ => unreachable!(),
            };
            Stmt::MaskedLoad {
                local: 2,
                place: at(0, pos),
                mask: in_range,
                fill: ScalarExpr::Const(fill),
            }
        }
        Mode::Replicate => {
            // min(max(pos, 0), IN_LEN - 1)
            let clamped = bin(
                BinaryOp::Min,
                bin(BinaryOp::Max, pos, ci(0)),
                ci(IN_LEN - 1),
            );
            Stmt::Load {
                local: 2,
                place: at(0, clamped),
            }
        }
        Mode::Reflect => {
            // with n1 = IN_LEN - 1 and |x| = max(x, -x): q = |pos|, mapped = n1 - |n1 - q| (q <= n1: q; q > n1: 2 n1 - q)
            let abs = |x: ScalarExpr| bin(BinaryOp::Max, x.clone(), sub(ci(0), x));
            let n1 = IN_LEN - 1;
            let q = abs(pos);
            let mapped = sub(ci(n1), abs(sub(ci(n1), q)));
            Stmt::Load {
                local: 2,
                place: at(0, mapped),
            }
        }
    };
    KernelFn::new(
        "pad",
        F32,
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
        ],
        vec![
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(F32),
        ],
        Block(vec![Stmt::For {
            local: 0,
            start: ci(0),
            end: ci(TAPS),
            step: ci(1),
            body: Block(vec![
                Stmt::Let {
                    local: 1,
                    init: sub(t.clone(), ci(PAD)),
                },
                load,
                Stmt::Store {
                    place: at(1, t),
                    value: ScalarExpr::Local(2),
                },
            ]),
        }]),
    )
}
