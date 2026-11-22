// SPDX-License-Identifier: MIT OR Apache-2.0
//! `asm volatile` statements (ROADMAP U19 slice 4c-3, `docs/options-u19-slice5-decode-census.md`): a **closed table of PTX
//! templates**, each mapped to the IR form that means the same. Anything not in the table declines with the template text, so
//! a fast approximation is never lifted as the accurate operation and an unknown instruction is never guessed.
//!
//! Table: `ex2`/`lg2`/`rcp` `.approx[.ftz].f32` to `Qualified(Exp2|Log2|Recip, Numerics { approx, ftz })`;
//! `shfl.sync.bfly.b32 ... 0x1f, 0xffffffff` to a `Subgroup` XOR shuffle; `cp.async.commit_group` to `AsyncCommit`;
//! `cp.async.wait_group N` to `AsyncWait`; `griddepcontrol.wait` and `.launch_dependents` to the grid-dependency statements.

use super::{R, Sym, Walker, text};
use tree_sitter::Node;
use unpopped::exec::Scope;
use unpopped::ir::{Numerics, RoundMode, ScalarExpr, UnaryOp};
use unpopped::kernel::{CollectiveKind, ShuffleMode, Stmt};
use unpopped_vocab::ElementKind as E;

/// One operand of an `asm` statement: its constraint (`"=f"`, `"f"`, `"r"`, `"n"`) and its C expression node.
pub(super) struct Operand<'n> {
    pub(super) constraint: String,
    pub(super) value: Node<'n>,
}

impl<'s, 't> Walker<'s, 't> {
    /// An `asm` statement; its effect goes to `out`.
    pub(super) fn asm_statement(&mut self, e: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        let template = self.asm_template(e);
        let outs = self.asm_operands(e, "output_operands");
        let ins = self.asm_operands(e, "input_operands");
        let words: Vec<&str> = template
            .split([' ', ','])
            .filter(|w| !w.is_empty())
            .collect();
        let head = words.first().copied().unwrap_or("");
        match head {
            "cp.async.commit_group" if outs.is_empty() && ins.is_empty() => {
                out.push(Stmt::AsyncCommit);
                Ok(())
            }
            h if h.starts_with("cp.async.c") && h != "cp.async.commit_group"
                || words.contains(&"@p") =>
            {
                self.asm_copy_async(e, &words, &ins, out)
            }
            "cp.async.wait_group" => self.asm_wait_group(e, &words, &outs, &ins, out),
            "griddepcontrol.wait" if outs.is_empty() && ins.is_empty() => {
                out.push(Stmt::GridDepWait);
                Ok(())
            }
            "griddepcontrol.launch_dependents" if outs.is_empty() && ins.is_empty() => {
                out.push(Stmt::GridDepLaunchDependents);
                Ok(())
            }
            h if h.starts_with("shfl.sync.bfly.b32") => {
                self.asm_shuffle(e, &words, &outs, &ins, out)
            }
            h if approx_op(h).is_some() => self.asm_approx(e, h, &words, &outs, &ins, out),
            _ => self.not_yet(e, format!("the asm template `{template}`")),
        }
    }

    /// The template with escapes and separators flattened: `"cp.async.commit_group;\n"` becomes `cp.async.commit_group`.
    fn asm_template(&self, e: Node<'_>) -> String {
        let raw = e
            .child_by_field_name("assembly_code")
            .map_or("", |c| text(c, self.src));
        raw.trim_matches('"')
            .replace("\\n", " ")
            .replace("\\t", " ")
            .replace(';', " ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn asm_operands<'n>(&self, e: Node<'n>, field: &str) -> Vec<Operand<'n>> {
        let Some(list) = e.child_by_field_name(field) else {
            return Vec::new();
        };
        let mut c = list.walk();
        list.named_children(&mut c)
            .filter_map(|o| {
                let constraint = o.child_by_field_name("constraint")?;
                let value = o.child_by_field_name("value")?;
                Some(Operand {
                    constraint: text(constraint, self.src).trim_matches('"').to_string(),
                    value,
                })
            })
            .collect()
    }

    /// `"=f"(y)`: the local `y` the result lands in (a plain `F32` local of this function).
    fn asm_out_local(&self, o: &Operand<'_>) -> R<u32> {
        if o.constraint != "=f" {
            return self.not_yet(
                o.value,
                format!("the asm output constraint `{}`", o.constraint),
            );
        }
        match (o.value.kind() == "identifier")
            .then(|| self.lookup(text(o.value, self.src)))
            .flatten()
        {
            Some(Sym::Scalar(l, E::F32, super::Fixed::No)) => Ok(l),
            _ => self.not_yet(o.value, "an asm output that is not a plain `float` local"),
        }
    }

    fn asm_approx(
        &mut self,
        e: Node<'_>,
        head: &str,
        words: &[&str],
        outs: &[Operand<'_>],
        ins: &[Operand<'_>],
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        let Some(op) = approx_op(head) else {
            return self.not_yet(e, format!("the asm template `{head}`"));
        };
        let ftz = head.contains(".ftz");
        let shape_ok = words.len() == 3 && head.ends_with(".f32") && head.contains(".approx");
        let (Some(o), Some(i), true) = (
            outs.first(),
            ins.first(),
            shape_ok && outs.len() == 1 && ins.len() == 1,
        ) else {
            return self.not_yet(e, format!("the asm template `{}`", words.join(" ")));
        };
        if i.constraint != "f" {
            return self.not_yet(
                i.value,
                format!("the asm input constraint `{}`", i.constraint),
            );
        }
        let local = self.asm_out_local(o)?;
        let (x, xt) = self.expr(i.value)?;
        let x = self.convert(x, xt, E::F32, i.value)?;
        out.append(&mut self.hoisted);
        let n = Numerics {
            round: RoundMode::NearestEven,
            ftz,
            approx: true,
        };
        let value = ScalarExpr::Qualified(Box::new(ScalarExpr::Unary(op, Box::new(x))), n);
        out.push(Stmt::Assign { local, value });
        Ok(())
    }

    fn asm_wait_group(
        &mut self,
        e: Node<'_>,
        words: &[&str],
        outs: &[Operand<'_>],
        ins: &[Operand<'_>],
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        let n = match ins {
            [i] if outs.is_empty() && words.len() == 2 && i.constraint == "n" => {
                text(i.value, self.src).trim().parse::<u32>().ok()
            }
            _ => None,
        };
        let Some(pending) = n else {
            return self.not_yet(
                e,
                "a `cp.async.wait_group` whose count is not one integer literal `\"n\"(N)`",
            );
        };
        out.push(Stmt::AsyncWait { pending });
        Ok(())
    }

    /// `shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff` with `"=f"(y) : "f"(x), "r"(lane_mask)`: a XOR shuffle over the
    /// subgroup. Any other width or member mask declines (a partial mask is not a full-participation collective).
    fn asm_shuffle(
        &mut self,
        e: Node<'_>,
        words: &[&str],
        outs: &[Operand<'_>],
        ins: &[Operand<'_>],
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        let full =
            words.len() == 6 && words[4] == "0x1f" && words[5].eq_ignore_ascii_case("0xffffffff");
        let (true, [o], [x, m]) = (full, outs, ins) else {
            return self.not_yet(e, format!("the asm template `{}`", words.join(" ")));
        };
        if x.constraint != "f" || m.constraint != "r" {
            return self.not_yet(
                e,
                "a shuffle with operand constraints other than `\"f\"(value), \"r\"(lane mask)`",
            );
        }
        let local = self.asm_out_local(o)?;
        let (v, vt) = self.expr(x.value)?;
        let v = self.convert(v, vt, E::F32, x.value)?;
        let (mask, mt) = self.expr(m.value)?;
        let mask = self.convert(mask, mt, E::U32, m.value)?;
        out.append(&mut self.hoisted);
        // a `Collective` declares its result, and `y` is already declared: shuffle into a fresh local, then assign it
        let fresh = self.new_local(E::F32);
        out.push(Stmt::Collective {
            kind: CollectiveKind::Shuffle {
                mode: ShuffleMode::Xor,
                operand: mask,
            },
            value: v,
            result: fresh,
            scope: Scope::Subgroup,
        });
        out.push(Stmt::Assign {
            local,
            value: ScalarExpr::Local(fresh),
        });
        Ok(())
    }
}

/// The IR operation of an `ex2`/`lg2`/`rcp` PTX mnemonic.
fn approx_op(head: &str) -> Option<UnaryOp> {
    let stem = head.split('.').next()?;
    match stem {
        "ex2" => Some(UnaryOp::Exp2),
        "lg2" => Some(UnaryOp::Log2),
        "rcp" => Some(UnaryOp::Recip),
        _ => None,
    }
}
