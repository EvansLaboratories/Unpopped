// SPDX-License-Identifier: MIT OR Apache-2.0
//! Expressions: operands, operators, C's usual arithmetic conversions, casts and literals (ROADMAP U19).

use super::*;

impl<'s, 't> Walker<'s, 't> {
    pub(super) fn constant(&self, v: i64) -> (ScalarExpr, E) {
        if int_range(self.compute).is_some() {
            (ScalarExpr::ConstInt(v, self.compute), self.compute)
        } else {
            (ScalarExpr::Const(v as f64), self.compute)
        }
    }

    pub(super) fn ty_of(&self, e: &ScalarExpr, at: Node<'_>) -> R<E> {
        let tys: Vec<Ty> = self.locals.iter().map(|l| l.ty()).collect();
        match infer_ty_in(e, self.compute, &tys) {
            Ok(Ty::Scalar(t)) => Ok(t),
            Ok(_) => self.not_yet(at, "a packed value"),
            Err(why) => self.outside(at, format!("`{}`: {why:?}", text(at, self.src))),
        }
    }

    pub(super) fn expr(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let (e, _) = self.expr_raw(n)?;
        let t = self.ty_of(&e, n)?;
        Ok((e, t))
    }

    pub(super) fn expr_raw(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        match n.kind() {
            "parenthesized_expression" => match n.named_child(0) {
                Some(i) => self.expr_raw(i),
                None => self.not_yet(n, "an empty parenthesis"),
            },
            "identifier" => self.identifier(n),
            "number_literal" => self.number(n),
            // `true`/`false` promote to `int` 1/0, as in C
            "true" => Ok((ScalarExpr::ConstInt(1, E::I32), E::I32)),
            "false" => Ok((ScalarExpr::ConstInt(0, E::I32), E::I32)),
            "binary_expression" => self.binary_node(n),
            "unary_expression" => self.unary_node(n),
            "conditional_expression" => self.conditional(n),
            "cast_expression" => self.cast_node(n),
            "subscript_expression" | "pointer_expression" => self.read_element(n),
            "field_expression" => self.builtin(n),
            "call_expression" => self.call_expr(n),
            "sizeof_expression" => self.sizeof_expr(n),
            "qualified_identifier" => self.qualified_value(n),
            other => self.not_yet(n, format!("a `{other}` expression")),
        }
    }

    pub(super) fn identifier(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let name = text(n, self.src);
        match self.lookup(name) {
            Some(Sym::Scalar(l, t, _)) => Ok((ScalarExpr::Local(l), t)),
            Some(Sym::Struct(i)) => self.struct_as_value(&i, n),
            Some(Sym::Buffer(..) | Sym::Alloc(..) | Sym::Pointer(..) | Sym::ParamArray(..)) => {
                self.not_yet(n, format!("the array or pointer `{name}` used as a value"))
            }
            Some(Sym::Vector(..) | Sym::VecAlloc(..)) => self.not_yet(
                n,
                format!("the vector `{name}` used as a scalar (read a member, or assign it whole)"),
            ),
            None => match self.global_const(name, n)? {
                Some(v) => Ok(v),
                None if self.limit_constant(name).is_some() => {
                    Ok(self.limit_constant(name).expect("checked above"))
                }
                None => self.not_yet(
                    n,
                    self.unexpanded_or(
                        name,
                        format!(
                            "the name `{name}`, which is not a parameter, local or file-scope constant"
                        ),
                    ),
                ),
            },
        }
    }

    pub(super) fn number(&self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        match parse_literal(text(n, self.src)) {
            Some(Lit::Int(v, t)) => Ok((ScalarExpr::ConstInt(v, t), t)),
            Some(Lit::Float(v, t)) if t == self.compute => Ok((ScalarExpr::Const(v), t)),
            Some(Lit::Float(_, t)) => self.not_yet(
                n,
                format!(
                    "a {t:?} literal in a kernel whose compute type is {:?}",
                    self.compute
                ),
            ),
            None => self.not_yet(n, format!("the literal `{}`", text(n, self.src))),
        }
    }

    pub(super) fn binary_node(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let (Some(l), Some(r), Some(op)) = (
            n.child_by_field_name("left"),
            n.child_by_field_name("right"),
            n.child_by_field_name("operator"),
        ) else {
            return self.not_yet(n, "a binary expression of an unexpected shape");
        };
        let op = text(op, self.src);
        // `p == nullptr` / `p != nullptr` (also `NULL`, either side): the null check of `truth_expr`
        if matches!(op, "==" | "!=") {
            let is_null = |x: Node<'_>| matches!(text(x, self.src), "nullptr" | "NULL");
            let ptr = if is_null(r) {
                Some(l)
            } else if is_null(l) {
                Some(r)
            } else {
                None
            };
            if let Some(p) = ptr.filter(|p| self.is_pointer_name(*p)) {
                // present unless the caller declared it null; the result is a C `int`, whatever the compute type
                let present = !self.pointer_is_null(text(p, self.src));
                let v = i64::from(present == (op == "!="));
                return Ok((ScalarExpr::ConstInt(v, E::I32), E::I32));
            }
        }
        let logical = matches!(op, "&&" | "||");
        let lv = if logical {
            self.truth_expr(l)?
        } else {
            self.expr(l)?
        };
        // A constant left operand decides the right one before it runs: `1 && r` and `0 || r` always evaluate `r` (so a read in
        // it is unconditional and may be hoisted); `0 && r` and `1 || r` never do (so it is not evaluated at all).
        let known = if logical { const_truth(&lv.0) } else { None };
        if let Some(t) = known {
            let short = (op == "&&") != t;
            if short {
                return Ok(self.constant(i64::from(t)));
            }
            let rv = self.truth_expr(r)?;
            return Ok(self.logical(op, lv.0, rv.0));
        }
        let before = self.hoisted.len();
        let rv = if logical {
            self.truth_expr(r)?
        } else {
            self.expr(r)?
        };
        if logical {
            if self.hoisted.len() != before {
                return self.not_yet(r, "a memory read in the right operand of `&&`/`||` (it is conditionally evaluated)");
            }
            return Ok(self.logical(op, lv.0, rv.0));
        }
        self.binary(op, lv, rv, n)
    }

    pub(super) fn logical(&self, op: &str, l: ScalarExpr, r: ScalarExpr) -> (ScalarExpr, E) {
        let (one, _) = self.constant(1);
        let (zero, _) = self.constant(0);
        let truth = ScalarExpr::Select(Box::new(r), Box::new(one.clone()), Box::new(zero.clone()));
        let e = if op == "&&" {
            ScalarExpr::Select(Box::new(l), Box::new(truth), Box::new(zero))
        } else {
            ScalarExpr::Select(Box::new(l), Box::new(one), Box::new(truth))
        };
        (e, self.compute)
    }

    pub(super) fn binary(
        &self,
        op: &str,
        l: (ScalarExpr, E),
        r: (ScalarExpr, E),
        at: Node<'_>,
    ) -> R<(ScalarExpr, E)> {
        if matches!(op, "<<" | ">>") {
            return self.shift(op, l, r, at);
        }
        let common = self.common_type(l.1, r.1, at)?;
        let a = Box::new(self.convert(l.0, l.1, common, at)?);
        let b = Box::new(self.convert(r.0, r.1, common, at)?);
        let e = match op {
            "+" => ScalarExpr::Add(a, b),
            "-" => ScalarExpr::Sub(a, b),
            "*" => ScalarExpr::Mul(a, b),
            "/" => ScalarExpr::Div(a, b),
            "%" => return self.remainder(a, b, common, at),
            "&" => ScalarExpr::Binary(BinaryOp::BitAnd, a, b),
            "|" => ScalarExpr::Binary(BinaryOp::BitOr, a, b),
            "^" => ScalarExpr::Binary(BinaryOp::BitXor, a, b),
            "<" => ScalarExpr::Binary(BinaryOp::CmpLt, a, b),
            "<=" => ScalarExpr::Binary(BinaryOp::CmpLe, a, b),
            ">" => ScalarExpr::Binary(BinaryOp::CmpGt, a, b),
            ">=" => ScalarExpr::Binary(BinaryOp::CmpGe, a, b),
            "==" => ScalarExpr::Binary(BinaryOp::CmpEq, a, b),
            "!=" => ScalarExpr::Binary(BinaryOp::CmpNe, a, b),
            other => return self.not_yet(at, format!("the operator `{other}`")),
        };
        let t = self.ty_of(&e, at)?;
        Ok((e, t))
    }

    /// C `a % b` on integers: truncating, sign of the dividend, and `0` for a zero divisor (and for `MIN % -1`), as the
    /// total `Div` composes it (`docs/int-div-shift-semantics.md`, the PM ruling 2026-10-31): `Select(b == 0, 0, a - (a / b) * b)`.
    /// A zero divisor is undefined in C, so any total result is a refinement. Both operands are pure (a memory read is
    /// already a local), so naming each twice repeats no effect.
    pub(super) fn remainder(
        &self,
        a: Box<ScalarExpr>,
        b: Box<ScalarExpr>,
        ty: E,
        at: Node<'_>,
    ) -> R<(ScalarExpr, E)> {
        if int_range(ty).is_none() {
            return self.outside(at, "`%` on a floating-point operand (not valid C)");
        }
        let quotient = ScalarExpr::Div(a.clone(), b.clone());
        let product = ScalarExpr::Mul(Box::new(quotient), b.clone());
        let rem = ScalarExpr::Sub(a, Box::new(product));
        let is_zero = ScalarExpr::Binary(BinaryOp::CmpEq, b, Box::new(ScalarExpr::ConstInt(0, ty)));
        let e = ScalarExpr::Select(
            Box::new(is_zero),
            Box::new(ScalarExpr::ConstInt(0, ty)),
            Box::new(rem),
        );
        let t = self.ty_of(&e, at)?;
        Ok((e, t))
    }

    /// A shift's result has the type of its **left** operand alone (the right is not converted to it).
    pub(super) fn shift(
        &self,
        op: &str,
        l: (ScalarExpr, E),
        r: (ScalarExpr, E),
        at: Node<'_>,
    ) -> R<(ScalarExpr, E)> {
        let lt = promote(l.1);
        let a = self.convert(l.0, l.1, lt, at)?;
        let b = self.convert(r.0, r.1, lt, at)?;
        let bop = if op == "<<" {
            BinaryOp::Shl
        } else {
            BinaryOp::Shr
        };
        let e = ScalarExpr::Binary(bop, Box::new(a), Box::new(b));
        let t = self.ty_of(&e, at)?;
        Ok((e, t))
    }

    pub(super) fn common_type(&self, a: E, b: E, at: Node<'_>) -> R<E> {
        let (fa, fb) = (int_range(a).is_none(), int_range(b).is_none());
        if fa || fb {
            return match (fa, fb) {
                (true, true) if a == b => Ok(a),
                (true, true) if matches!((a, b), (E::F64, _) | (_, E::F64)) => Ok(E::F64),
                (true, false) => Ok(a),
                (false, true) => Ok(b),
                _ => self.not_yet(at, format!("an operation on {a:?} and {b:?}")),
            };
        }
        let (a, b) = (promote(a), promote(b));
        if a == b {
            return Ok(a);
        }
        let (ra, rb) = (int_rank(a), int_rank(b));
        Ok(if ra == rb {
            // equal width, different signedness: the unsigned one
            if is_unsigned(a) { a } else { b }
        } else if ra > rb {
            a
        } else {
            b
        })
    }

    /// `e` of type `from` as type `to`, by a `Cast` when they differ (C's implicit conversion made explicit).
    pub(super) fn convert(&self, e: ScalarExpr, from: E, to: E, at: Node<'_>) -> R<ScalarExpr> {
        if from == to {
            return Ok(e);
        }
        let float_to_int = int_range(from).is_none() && int_range(to).is_some();
        let spec = CastSpec {
            to,
            round: if float_to_int {
                RoundMode::TowardZero
            } else {
                RoundMode::NearestEven
            },
            sat: if float_to_int {
                Saturation::Clamp
            } else {
                Saturation::Wrap
            },
        };
        let out = ScalarExpr::Cast(Box::new(e), spec);
        self.ty_of(&out, at)?;
        Ok(out)
    }

    pub(super) fn unary_node(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let (Some(arg), Some(op)) = (
            n.child_by_field_name("argument"),
            n.child_by_field_name("operator"),
        ) else {
            return self.not_yet(n, "a unary expression of an unexpected shape");
        };
        let negation = text(op, self.src) == "!";
        let (v, t) = if negation {
            self.truth_expr(arg)?
        } else {
            self.expr(arg)?
        };
        match text(op, self.src) {
            "+" => Ok((v, t)),
            "-" if int_range(t).is_some() => {
                let wide = promote(t);
                let v = self.convert(v, t, wide, arg)?;
                Ok((
                    ScalarExpr::Sub(Box::new(ScalarExpr::ConstInt(0, wide)), Box::new(v)),
                    wide,
                ))
            }
            "-" => Ok((ScalarExpr::Unary(UnaryOp::Neg, Box::new(v)), t)),
            "!" => {
                let (one, _) = self.constant(1);
                let (zero, _) = self.constant(0);
                Ok((
                    ScalarExpr::Select(Box::new(v), Box::new(zero), Box::new(one)),
                    self.compute,
                ))
            }
            other => self.not_yet(n, format!("the unary operator `{other}`")),
        }
    }

    pub(super) fn conditional(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let (Some(c), Some(a), Some(b)) = (
            n.child_by_field_name("condition"),
            n.child_by_field_name("consequence"),
            n.child_by_field_name("alternative"),
        ) else {
            return self.not_yet(n, "a conditional of an unexpected shape");
        };
        let cond = self.truth_expr(c)?.0;
        let before = self.hoisted.len();
        let (av, at) = self.expr(a)?;
        let mid = self.hoisted.len();
        let (bv, bt) = self.expr(b)?;
        let common = self.common_type(at, bt, n)?;
        let av = self.convert(av, at, common, n)?;
        let bv = self.convert(bv, bt, common, n)?;
        if self.hoisted.len() == before {
            return Ok((
                ScalarExpr::Select(Box::new(cond), Box::new(av), Box::new(bv)),
                common,
            ));
        }
        // An arm reads memory, and only the arm taken may: `If (c) { loads_a; r = a } else { loads_b; r = b }`, the value in `r`.
        let mut b_loads = self.hoisted.split_off(mid);
        let mut a_loads = self.hoisted.split_off(before);
        let r = self.new_local(common);
        a_loads.push(Stmt::Assign {
            local: r,
            value: av,
        });
        b_loads.push(Stmt::Assign {
            local: r,
            value: bv,
        });
        let zero = self.zero_of(common, n)?;
        self.hoisted.push(Stmt::Let {
            local: r,
            init: zero,
        });
        self.hoisted.push(Stmt::If {
            cond,
            then: Block(a_loads),
            els: Block(b_loads),
        });
        Ok((ScalarExpr::Local(r), common))
    }

    pub(super) fn cast_node(&mut self, n: Node<'_>) -> R<(ScalarExpr, E)> {
        let (Some(ty), Some(v)) = (
            n.child_by_field_name("type"),
            n.child_by_field_name("value"),
        ) else {
            return self.not_yet(n, "a cast of an unexpected shape");
        };
        let to = self.type_of(ty.child_by_field_name("type"), ty)?;
        let (e, from) = self.expr(v)?;
        Ok((self.convert(e, from, to, n)?, to))
    }
}

/// The truth of a constant expression, if it is one.
fn const_truth(e: &ScalarExpr) -> Option<bool> {
    match e {
        ScalarExpr::ConstInt(v, _) => Some(*v != 0),
        ScalarExpr::Const(v) => Some(*v != 0.0),
        _ => None,
    }
}
