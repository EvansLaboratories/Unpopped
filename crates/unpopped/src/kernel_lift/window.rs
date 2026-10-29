// SPDX-License-Identifier: MIT OR Apache-2.0
//! The `Access::Window` lift (1-D pooling over the innermost axis).
//!
//! Per output cell the oracle folds the valid taps `p = o*stride - pad_lo + kk*dilation` (those
//! inside `[0, k_in)`) in `f64`, in ascending `kk`, then finalises: `Sum` as is, `Mean` divided by
//! `size` (`count_include_pad`) or by the valid-tap count (zero when none were valid), `Max`/`Min`
//! with the NaN-keeping rule and the monoid identity when no tap was valid. One rounding happens at
//! the store. The kernel does exactly that. The identity post-expression is byte-exact; a fused
//! post lifts in tolerance mode, `Reduced(0)` bound to the finalised value rounded once to the
//! compute dtype (`docs/ir-kernel-layer.md` 7.1), as a reduction's does (see
//! [`super::lift_reduction_opdef`]).

use unpopped_vocab::ElementKind;

use super::im2col::{add, int, local, mul, outside, sub};
use super::post;
use super::{
    LiftDecline, check_accumulated_dtype, check_plain, fold_step, is_int_kernel_dtype, leaves_ok,
    scan_identity, substitute_inputs, widen,
};
use crate::ir::{Access, OpDef, ReduceOp, ScalarExpr};
use crate::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};

/// Lift a 1-D `Access::Window` over `rank` axes (the pooled axis is the innermost), computed at
/// `dtype` (`F32` or `F64`; `I32` or `I64` for `Sum`/`Max`/`Min` with the identity post, where the
/// accumulator is the dtype itself and the oracle's exact wrapping fold is matched).
///
/// Parameters: the inputs (`Read`, rank `rank`), the output (`Write`, rank `rank`), then one `I32`
/// extent per input axis and finally the output extent `k_out` of the pooled axis. The window
/// geometry is part of the op and is baked in as constants.
///
/// # Errors
///
/// A [`LiftDecline`].
pub fn lift_window_opdef(
    op: &OpDef,
    dtype: ElementKind,
    rank: u8,
) -> Result<KernelFn, LiftDecline> {
    let g = check_window(op, dtype, rank)?;
    let w = Win::new(op.n_inputs, dtype, rank);
    let mut body = Block(vec![
        Stmt::Let {
            local: w.acc,
            init: w.zero(),
        },
        Stmt::Let {
            local: w.have,
            init: int(0),
        },
        Stmt::Let {
            local: w.cnt,
            init: int(0),
        },
    ]);
    body.0.push(w.taps(&g));
    body.0.extend(w.finish(&g));
    let value = w.stored_value(&g, &mut body);
    body.0.push(Stmt::Store {
        place: Place::Logical {
            buffer: w.n,
            coords: w.out_coords(),
        },
        value,
    });
    for d in (0..w.r).rev() {
        body = w.for_cell(d, body);
    }
    let (params, mut locals) = w.signature();
    if g.post.is_some() {
        locals.push(LocalDecl::scalar(dtype)); // red
    }
    Ok(KernelFn::new(op.name.clone(), dtype, params, locals, body))
}

/// The window as the op declares it.
struct Geometry {
    op: ReduceOp,
    size: u8,
    stride: u8,
    dilation: u8,
    pad_lo: u8,
    count_include_pad: bool,
    pre: ScalarExpr,
    /// The fused post-expression, or `None` for the identity (`Reduced(0)`).
    post: Option<ScalarExpr>,
}

fn check_window(op: &OpDef, dtype: ElementKind, rank: u8) -> Result<Geometry, LiftDecline> {
    let Access::Window {
        op: wop,
        axis,
        size,
        stride,
        dilation,
        pad_lo,
        count_include_pad,
        pre,
        post,
        ..
    } = &op.access
    else {
        return Err(LiftDecline::NotWindow);
    };
    check_accumulated_dtype(dtype, *wop, post, "a window")?;
    let post = if matches!(post, ScalarExpr::Reduced(0)) {
        None
    } else {
        post::check_post(post, 1)?;
        if post::reads_input(post) {
            return Err(LiftDecline::UnsupportedLeaf("an Input in a window post"));
        }
        Some(post.clone())
    };
    if matches!(wop, ReduceOp::Prod) || usize::from(*axis) + 1 != usize::from(rank) {
        return Err(LiftDecline::UnsupportedForm(
            "a window that is not a pool over the last axis",
        ));
    }
    check_plain(op, dtype, rank)?;
    leaves_ok(pre)?;
    Ok(Geometry {
        op: *wop,
        size: *size,
        stride: *stride,
        dilation: *dilation,
        pad_lo: *pad_lo,
        count_include_pad: *count_include_pad,
        pre: pre.clone(),
        post,
    })
}

/// Local ids. Input extents `0..r`, then the output extent `k_out`; the cell counters (the leading
/// axes, then the output position) and the tap counter; then the source position, the loaded
/// inputs, the accumulator, the `have` flag, the valid-tap count, the current value and the result.
struct Win {
    dtype: ElementKind,
    r: u32,
    n: u32,
    k_out: u32,
    tap: u32,
    pos: u32,
    acc: u32,
    have: u32,
    cnt: u32,
    val: u32,
    res: u32,
    /// The result rounded to the compute dtype (declared only when a post is lifted).
    red: u32,
}

impl Win {
    fn new(n_inputs: u8, dtype: ElementKind, rank: u8) -> Self {
        let (r, n) = (u32::from(rank), u32::from(n_inputs));
        let counters = r + 1;
        let pos = counters + r + 1;
        let acc = pos + 1 + n;
        Self {
            dtype,
            r,
            n,
            k_out: r,
            tap: counters + r,
            pos,
            acc,
            have: acc + 1,
            cnt: acc + 2,
            val: acc + 3,
            res: acc + 4,
            red: acc + 5,
        }
    }

    /// The counter of cell axis `d` (the leading axes, then the output position at `r - 1`).
    fn cell(&self, d: u32) -> u32 {
        self.r + 1 + d
    }

    fn loaded(&self, k: u32) -> u32 {
        self.pos + 1 + k
    }

    fn is_int(&self) -> bool {
        is_int_kernel_dtype(self.dtype)
    }

    /// The accumulator's, current value's and result's type.
    fn acc_ty(&self) -> ElementKind {
        if self.is_int() {
            self.dtype
        } else {
            ElementKind::F64
        }
    }

    /// The accumulator's starting value (a `Max`/`Min` is guarded by `have`).
    fn zero(&self) -> ScalarExpr {
        if self.is_int() {
            ScalarExpr::ConstInt(0, self.dtype)
        } else {
            self.up(ScalarExpr::Const(0.0))
        }
    }

    fn up(&self, e: ScalarExpr) -> ScalarExpr {
        if self.dtype == ElementKind::F64 || self.is_int() {
            e
        } else {
            ScalarExpr::Cast(Box::new(e), widen(ElementKind::F64))
        }
    }

    /// `for cell(d) in 0..extent { inner }`; the pooled axis runs to `k_out`.
    fn for_cell(&self, d: u32, inner: Block) -> Block {
        let end = if d + 1 == self.r { self.k_out } else { d };
        Block(vec![Stmt::For {
            local: self.cell(d),
            start: int(0),
            end: local(end),
            step: int(1),
            body: inner,
        }])
    }

    fn out_coords(&self) -> Vec<ScalarExpr> {
        (0..self.r).map(|d| local(self.cell(d))).collect()
    }

    /// The loop over taps: each valid tap is loaded, mapped by `pre` and folded.
    fn taps(&self, g: &Geometry) -> Stmt {
        let o = local(self.cell(self.r - 1));
        let position = add(
            sub(mul(o, int(i64::from(g.stride))), int(i64::from(g.pad_lo))),
            mul(local(self.tap), int(i64::from(g.dilation))),
        );
        let mut valid: Vec<Stmt> = (0..self.n)
            .map(|k| Stmt::Load {
                local: self.loaded(k),
                place: Place::Logical {
                    buffer: k,
                    coords: (0..self.r - 1)
                        .map(|d| local(self.cell(d)))
                        .chain([local(self.pos)])
                        .collect(),
                },
            })
            .collect();
        valid.push(Stmt::Let {
            local: self.val,
            init: self.up(substitute_inputs(&g.pre, &|k| self.loaded(k))),
        });
        valid.extend(fold_step(
            g.op,
            self.acc,
            self.have,
            self.val,
            self.is_int(),
        ));
        valid.push(Stmt::Assign {
            local: self.cnt,
            value: add(local(self.cnt), int(1)),
        });
        Stmt::For {
            local: self.tap,
            start: int(0),
            end: int(i64::from(g.size)),
            step: int(1),
            body: Block(vec![
                Stmt::Let {
                    local: self.pos,
                    init: position,
                },
                Stmt::If {
                    cond: outside(self.pos, self.r - 1),
                    then: Block(vec![]),
                    els: Block(valid),
                },
            ]),
        }
    }

    /// Set `res` from the accumulator: `Sum` as is, `Mean` divided, `Max`/`Min` the identity when
    /// no tap was valid.
    fn finish(&self, g: &Geometry) -> Vec<Stmt> {
        let acc = || local(self.acc);
        let set = |value| Stmt::Assign {
            local: self.res,
            value,
        };
        let to_f64 = |e| ScalarExpr::Cast(Box::new(e), widen(ElementKind::F64));
        let div = |d| ScalarExpr::Div(Box::new(acc()), Box::new(d));
        match g.op {
            ReduceOp::Mean if g.count_include_pad => vec![Stmt::Let {
                local: self.res,
                init: div(to_f64(int(i64::from(g.size)))),
            }],
            ReduceOp::Mean => vec![
                Stmt::Let {
                    local: self.res,
                    init: self.up(ScalarExpr::Const(0.0)),
                },
                Stmt::If {
                    cond: local(self.cnt),
                    then: Block(vec![set(div(to_f64(local(self.cnt))))]),
                    els: Block(vec![]),
                },
            ],
            ReduceOp::Max | ReduceOp::Min => vec![
                Stmt::Let {
                    local: self.res,
                    init: self.up(scan_identity(g.op, self.dtype)),
                },
                Stmt::If {
                    cond: local(self.have),
                    then: Block(vec![set(acc())]),
                    els: Block(vec![]),
                },
            ],
            _ => vec![Stmt::Let {
                local: self.res,
                init: acc(),
            }],
        }
    }

    fn stored(&self) -> ScalarExpr {
        let res = local(self.res);
        if self.dtype == ElementKind::F64 || self.is_int() {
            res
        } else {
            ScalarExpr::Cast(Box::new(res), widen(self.dtype))
        }
    }

    /// The value to store. With a fused post, `red` (the result rounded once to the compute dtype)
    /// is declared at the end of `body` and the post reads it as `Reduced(0)`.
    fn stored_value(&self, g: &Geometry, body: &mut Block) -> ScalarExpr {
        let Some(post) = &g.post else {
            return self.stored();
        };
        body.0.push(Stmt::Let {
            local: self.red,
            init: self.stored(),
        });
        post::bind(post, &|leaf| {
            matches!(leaf, ScalarExpr::Reduced(_)).then(|| ScalarExpr::Local(self.red))
        })
    }

    fn signature(&self) -> (Vec<Param>, Vec<LocalDecl>) {
        let rank = u8::try_from(self.r).expect("rank is at most 8");
        let mut params: Vec<Param> = (0..self.n)
            .map(|k| Param::buffer(format!("in{k}"), self.dtype, BufferRole::Read, Some(rank)))
            .collect();
        params.push(Param::buffer(
            "out",
            self.dtype,
            BufferRole::Write,
            Some(rank),
        ));
        params.extend((0..self.r).map(|d| Param::scalar(format!("n{d}"), ElementKind::I32, d)));
        params.push(Param::scalar("k_out", ElementKind::I32, self.k_out));
        // Extents (r + 1), cell counters (r), tap counter and source position: all `I32`.
        let mut locals = vec![LocalDecl::scalar(ElementKind::I32); self.pos as usize + 1];
        locals.extend((0..self.n).map(|_| LocalDecl::scalar(self.dtype)));
        locals.push(LocalDecl::scalar(self.acc_ty())); // acc
        locals.push(LocalDecl::scalar(ElementKind::I32)); // have
        locals.push(LocalDecl::scalar(ElementKind::I32)); // cnt
        locals.push(LocalDecl::scalar(self.acc_ty())); // val
        locals.push(LocalDecl::scalar(self.acc_ty())); // res
        (params, locals)
    }
}
