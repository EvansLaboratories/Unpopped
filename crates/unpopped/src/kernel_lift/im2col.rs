// SPDX-License-Identifier: MIT OR Apache-2.0
//! The `Access::Im2Col` lift.
//!
//! The oracle copies one element per output cell `(n, c*kh*kw + ki*kw + kj, oh*oW + ow)` from the
//! input at `(n, c, oh*sh - ph + ki*dh, ow*sw - pw + kj*dw)`, and writes the typed zero when that
//! coordinate falls outside the input. No arithmetic touches the value, so the kernel is a bounds
//! test around one load and one store.
//!
//! The scalar layer has no comparison. "Inside `[0, extent)`" is therefore tested by clamping:
//! `max(0, min(i, extent - 1)) - i` is zero exactly when `i` is inside, so a non-zero condition
//! means outside and takes the `then` arm, which leaves the zero in place.

use unpopped_vocab::ElementKind;

use super::{LiftDecline, check_plain};
use crate::ir::{Access, BinaryOp, OpDef, ScalarExpr};
use crate::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};

/// Lift a 2-D `Access::Im2Col` computed at `dtype` (`F32` or `F64`).
///
/// Parameters: `src` (`Read`, rank 4 NCHW), `out` (`Write`, rank 3 `[N, C*kh*kw, oH*oW]`), then the
/// `I32` extents `N, C, H_in, W_in, oH, oW`. The window geometry is part of the op and is baked
/// into the kernel as constants.
///
/// # Errors
///
/// A [`LiftDecline`].
pub fn lift_im2col_opdef(op: &OpDef, dtype: ElementKind) -> Result<KernelFn, LiftDecline> {
    let Access::Im2Col {
        kernel,
        stride,
        pad,
        dilation,
    } = op.access
    else {
        return Err(LiftDecline::NotIm2Col);
    };
    if !matches!(dtype, ElementKind::F32 | ElementKind::F64) {
        return Err(LiftDecline::UnsupportedForm("a non-float im2col"));
    }
    check_plain(op, dtype, 4)?;
    let g = Geometry {
        kernel,
        stride,
        pad,
        dilation,
    };
    let body = g.loops(g.cell());
    let (params, locals) = signature(dtype);
    Ok(KernelFn::new(op.name.clone(), dtype, params, locals, body))
}

/// Local ids: extents `N, C, H, W, oH, oW` are `0..6`; the loop counters `n, c, ki, kj, oh, ow`
/// are `6..12`; then the source row and column, the value to store, and the loaded element (a `Load` declares
/// its local, so the zero default lives in `VAL` and the load lands in `LOADED`).
const N: u32 = 0;
const C: u32 = 1;
const H: u32 = 2;
const W: u32 = 3;
const OH: u32 = 4;
const OW: u32 = 5;
const CTR: u32 = 6;
const IH: u32 = 12;
const IW: u32 = 13;
const VAL: u32 = 14;
const LOADED: u32 = 15;

/// The window geometry as the op declares it.
struct Geometry {
    kernel: (u8, u8),
    stride: (u8, u8),
    pad: (u8, u8),
    dilation: (u8, u8),
}

pub(super) fn int(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, ElementKind::I32)
}

pub(super) fn local(l: u32) -> ScalarExpr {
    ScalarExpr::Local(l)
}

pub(super) fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

pub(super) fn sub(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Sub(Box::new(a), Box::new(b))
}

pub(super) fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}

fn bin(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}

/// Non-zero exactly when `index` is outside `[0, local(extent))`.
pub(super) fn outside(index: u32, extent: u32) -> ScalarExpr {
    let last = sub(local(extent), int(1));
    let clamped = bin(
        BinaryOp::Max,
        int(0),
        bin(BinaryOp::Min, local(index), last),
    );
    sub(clamped, local(index))
}

impl Geometry {
    /// The counter of loop `i` of `n, c, ki, kj, oh, ow`.
    fn ctr(i: u32) -> ScalarExpr {
        local(CTR + i)
    }

    /// `for ctr(i) in 0..end { inner }`.
    fn for_to(i: u32, end: ScalarExpr, inner: Block) -> Block {
        Block(vec![Stmt::For {
            local: CTR + i,
            start: int(0),
            end,
            step: int(1),
            body: inner,
        }])
    }

    /// Wrap `cell` in the six loops, `n` outermost and `ow` innermost.
    fn loops(&self, cell: Block) -> Block {
        let ends = [
            local(N),
            local(C),
            int(i64::from(self.kernel.0)),
            int(i64::from(self.kernel.1)),
            local(OH),
            local(OW),
        ];
        let mut body = cell;
        for (i, end) in ends.into_iter().enumerate().rev() {
            body = Self::for_to(u32::try_from(i).expect("six loops"), end, body);
        }
        body
    }

    /// `counter * stride - pad + tap * dilation` for one spatial axis.
    fn source_index(out_ctr: u32, tap_ctr: u32, stride: u8, pad: u8, dilation: u8) -> ScalarExpr {
        add(
            sub(
                mul(Self::ctr(out_ctr), int(i64::from(stride))),
                int(i64::from(pad)),
            ),
            mul(Self::ctr(tap_ctr), int(i64::from(dilation))),
        )
    }

    /// One output cell: compute the source coordinate, load it when inside, store the result.
    fn cell(&self) -> Block {
        let load = Stmt::Load {
            local: LOADED,
            place: Place::Logical {
                buffer: 0,
                coords: vec![Self::ctr(0), Self::ctr(1), local(IH), local(IW)],
            },
        };
        let inner = Stmt::If {
            cond: outside(IW, W),
            then: Block(vec![]),
            els: Block(vec![
                load,
                Stmt::Assign {
                    local: VAL,
                    value: local(LOADED),
                },
            ]),
        };
        let guarded = Stmt::If {
            cond: outside(IH, H),
            then: Block(vec![]),
            els: Block(vec![inner]),
        };
        let (kh, kw) = (i64::from(self.kernel.0), i64::from(self.kernel.1));
        let row = add(
            mul(Self::ctr(1), int(kh * kw)),
            add(mul(Self::ctr(2), int(kw)), Self::ctr(3)),
        );
        let col = add(mul(Self::ctr(4), local(OW)), Self::ctr(5));
        Block(vec![
            Stmt::Let {
                local: IH,
                init: Self::source_index(4, 2, self.stride.0, self.pad.0, self.dilation.0),
            },
            Stmt::Let {
                local: IW,
                init: Self::source_index(5, 3, self.stride.1, self.pad.1, self.dilation.1),
            },
            Stmt::Let {
                local: VAL,
                init: ScalarExpr::Const(0.0),
            },
            guarded,
            Stmt::Store {
                place: Place::Logical {
                    buffer: 1,
                    coords: vec![Self::ctr(0), row, col],
                },
                value: local(VAL),
            },
        ])
    }
}

fn signature(dtype: ElementKind) -> (Vec<Param>, Vec<LocalDecl>) {
    let mut params = vec![
        Param::buffer("src", dtype, BufferRole::Read, Some(4)),
        Param::buffer("out", dtype, BufferRole::Write, Some(3)),
    ];
    for (name, l) in ["n", "c", "h", "w", "oh", "ow"].into_iter().zip(0u32..) {
        params.push(Param::scalar(name, ElementKind::I32, l));
    }
    let mut locals = vec![LocalDecl::scalar(ElementKind::I32); 14];
    locals.push(LocalDecl::scalar(dtype)); // VAL
    locals.push(LocalDecl::scalar(dtype)); // LOADED
    (params, locals)
}
