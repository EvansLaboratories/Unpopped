// SPDX-License-Identifier: MIT OR Apache-2.0
//! Statements: blocks, assignments, `if`/`while`/counted `for`, `#pragma unroll` (ROADMAP U19).

use super::*;

impl<'s, 't> Walker<'s, 't> {
    pub(super) fn block_of(&mut self, n: Node<'_>) -> R<Block> {
        self.scopes.push(HashMap::new());
        let mut out = Vec::new();
        let r = if n.kind() == "compound_statement" {
            let mut c = n.walk();
            let kids: Vec<_> = n.named_children(&mut c).collect();
            kids.into_iter().try_for_each(|k| self.stmt(k, &mut out))
        } else {
            self.stmt(n, &mut out)
        };
        self.scopes.pop();
        r.map(|()| Block(out))
    }

    pub(super) fn stmt(&mut self, n: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        if n.kind() == "preproc_call" {
            return self.pragma(n);
        }
        // a hint binds the statement right after it: any other statement drops it, a `for` takes it
        let unroll = std::mem::take(&mut self.pending_unroll);
        match n.kind() {
            "compound_statement" => {
                let b = self.block_of(n)?;
                // a nested `{ }` is a scope, not a statement: splice its statements (the IR scopes `Let` by block, so
                // keep them in a one-armed `If` would change nothing, and splicing is exact).
                out.extend(b.0);
                Ok(())
            }
            "declaration" => self.declaration(n, out),
            "alias_declaration" | "type_definition" => self.alias_statement(n),
            "static_assert_declaration" => self.static_assert(n),
            "expression_statement" => self.expression_statement(n, out),
            "if_statement" => self.if_statement(n, out),
            "while_statement" => self.while_statement(n, out),
            "for_statement" => self.for_statement(n, out, unroll),
            "break_statement" => push(out, Stmt::Break),
            "continue_statement" => push(out, Stmt::Continue),
            "return_statement" => {
                if n.named_child_count() > 0 {
                    return self.not_yet(n, "a `return` with a value");
                }
                push(out, Stmt::Return)
            }
            "comment" | ";" => Ok(()),
            other => self.not_yet(n, format!("a `{other}` statement")),
        }
    }

    pub(super) fn flush(&mut self, out: &mut Vec<Stmt>) {
        out.append(&mut self.hoisted);
    }

    pub(super) fn expression_statement(&mut self, n: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        let Some(e) = n.named_child(0) else {
            return Ok(());
        };
        match e.kind() {
            "assignment_expression" => self.assignment(e, out),
            "update_expression" => self.update(e, out),
            "call_expression" => self.call_statement(e, out),
            "gnu_asm_expression" => self.asm_statement(e, out),
            // `cg::thread_block_tile<32> w = cg::tiled_partition<32>(b);` parses as a comparison chain
            "binary_expression" => match self.tile_declaration(n) {
                Some(r) => r,
                None => self.not_yet(e, "a `binary_expression` expression statement"),
            },
            // `(void)x;` silences an unused-variable warning and does nothing: only a plain name or literal, never a call or an assignment
            "cast_expression" if self.is_void_discard(e) => Ok(()),
            other => self.not_yet(e, format!("a `{other}` expression statement")),
        }
    }

    /// `(void)name;` / `(void)3;`: a cast to `void` of something with no effect.
    fn is_void_discard(&self, e: Node<'_>) -> bool {
        let void = e
            .child_by_field_name("type")
            .is_some_and(|t| text(t, self.src).trim() == "void");
        void && e
            .child_by_field_name("value")
            .is_some_and(|v| matches!(v.kind(), "identifier" | "number_literal"))
    }

    pub(super) fn assignment(&mut self, n: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        let (Some(lhs), Some(rhs), Some(op)) = (
            n.child_by_field_name("left"),
            n.child_by_field_name("right"),
            n.child_by_field_name("operator"),
        ) else {
            return self.not_yet(n, "an assignment of an unexpected shape");
        };
        let op = text(op, self.src).trim_end_matches('=').to_string();
        match lhs.kind() {
            "identifier" => self.assign_local(lhs, &op, rhs, out),
            "subscript_expression" | "pointer_expression" => {
                if self.vector_store(lhs, &op, rhs, out)? {
                    return Ok(());
                }
                self.assign_element(lhs, &op, rhs, out)
            }
            "field_expression" => self.assign_member(lhs, &op, rhs, out),
            other => self.not_yet(lhs, format!("an assignment to a `{other}`")),
        }
    }

    pub(super) fn assign_local(
        &mut self,
        lhs: Node<'_>,
        op: &str,
        rhs: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        let name = text(lhs, self.src);
        if let Some(Sym::Vector(ls, elem, lanes)) = self.lookup(name) {
            return self.assign_vector(ls, (elem, lanes), op, rhs, out);
        }
        if let Some(Sym::Struct(dst)) = self.lookup(name) {
            return self.assign_struct(&dst, op, rhs, out);
        }
        if let Some(Sym::Pointer(p)) = self.lookup(name) {
            return self.move_pointer(lhs, &p, op, rhs, out);
        }
        let Some(Sym::Scalar(local, ty, fixed)) = self.lookup(name) else {
            return self.not_yet(
                lhs,
                format!("an assignment to `{name}`, which is not a scalar local"),
            );
        };
        match fixed {
            Fixed::Param => {
                return self.not_yet(lhs, format!("an assignment to the parameter `{name}`"));
            }
            Fixed::Counter => {
                return self.not_yet(lhs, format!("an assignment to the loop counter `{name}`"));
            }
            Fixed::No => {}
        }
        let (r, rt) = self.expr(rhs)?;
        let value = self.compound(ScalarExpr::Local(local), ty, op, (r, rt), lhs)?;
        self.flush(out);
        out.push(Stmt::Assign { local, value });
        Ok(())
    }

    pub(super) fn assign_element(
        &mut self,
        lhs: Node<'_>,
        op: &str,
        rhs: Node<'_>,
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        let (place, elem) = self.element_place(lhs)?;
        let (r, rt) = self.expr(rhs)?;
        let value = if op.is_empty() {
            self.convert(r, rt, elem, rhs)?
        } else {
            let local = self.new_local(elem);
            self.hoisted.push(Stmt::Load {
                local,
                place: place.clone(),
            });
            self.compound(ScalarExpr::Local(local), elem, op, (r, rt), lhs)?
        };
        self.flush(out);
        out.push(Stmt::Store { place, value });
        Ok(())
    }

    /// `old op= rhs` as the value stored back, converted to the target's type; plain `=` converts `rhs` alone.
    pub(super) fn compound(
        &self,
        old: ScalarExpr,
        ty: E,
        op: &str,
        rhs: (ScalarExpr, E),
        at: Node<'_>,
    ) -> R<ScalarExpr> {
        if op.is_empty() {
            return self.convert(rhs.0, rhs.1, ty, at);
        }
        let (v, vt) = self.binary(op, (old, ty), rhs, at)?;
        self.convert(v, vt, ty, at)
    }

    pub(super) fn update(&mut self, n: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        let (Some(arg), Some(op)) = (
            n.child_by_field_name("argument"),
            n.child_by_field_name("operator"),
        ) else {
            return self.not_yet(n, "an increment of an unexpected shape");
        };
        let one = Self::lit_one_for(self.lookup(text(arg, self.src)));
        let Some(one) = one else {
            return self.not_yet(
                arg,
                "an increment of something that is not an integer local",
            );
        };
        let sign = if text(op, self.src) == "++" { "+" } else { "-" };
        self.assign_local_value(arg, sign, one, out)
    }

    pub(super) fn lit_one_for(sym: Option<Sym>) -> Option<(ScalarExpr, E)> {
        match sym {
            Some(Sym::Scalar(_, t, _)) if int_range(t).is_some() => {
                Some((ScalarExpr::ConstInt(1, t), t))
            }
            _ => None,
        }
    }

    pub(super) fn assign_local_value(
        &mut self,
        lhs: Node<'_>,
        op: &str,
        v: (ScalarExpr, E),
        out: &mut Vec<Stmt>,
    ) -> R<()> {
        let name = text(lhs, self.src);
        let Some(Sym::Scalar(local, ty, fixed)) = self.lookup(name) else {
            return self.not_yet(lhs, format!("an update of `{name}`"));
        };
        if fixed != Fixed::No {
            return self.not_yet(
                lhs,
                format!("an update of the parameter or loop counter `{name}`"),
            );
        }
        let value = self.compound(ScalarExpr::Local(local), ty, op, v, lhs)?;
        out.push(Stmt::Assign { local, value });
        Ok(())
    }

    pub(super) fn condition(&mut self, n: Node<'_>) -> R<ScalarExpr> {
        let inner = if n.kind() == "condition_clause" {
            n.child_by_field_name("value")
        } else {
            Some(n)
        };
        let Some(inner) = inner else {
            return self.not_yet(n, "a condition with a declaration");
        };
        Ok(self.truth_expr(inner)?.0)
    }

    pub(super) fn if_statement(&mut self, n: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        let (Some(c), Some(then)) = (
            n.child_by_field_name("condition"),
            n.child_by_field_name("consequence"),
        ) else {
            return self.not_yet(n, "an `if` of an unexpected shape");
        };
        let cond = self.condition(c)?;
        self.flush(out);
        let then = self.block_of(then)?;
        let els = match n.child_by_field_name("alternative") {
            Some(a) => match a.named_child(0) {
                Some(s) => self.block_of(s)?,
                None => Block::default(),
            },
            None => Block::default(),
        };
        out.push(Stmt::If { cond, then, els });
        Ok(())
    }

    pub(super) fn while_statement(&mut self, n: Node<'_>, out: &mut Vec<Stmt>) -> R<()> {
        let (Some(c), Some(body)) = (
            n.child_by_field_name("condition"),
            n.child_by_field_name("body"),
        ) else {
            return self.not_yet(n, "a `while` of an unexpected shape");
        };
        let cond = self.condition(c)?;
        // taken now: the body's own statements flush `hoisted`, and must not flush the condition's loads
        let mut lead = std::mem::take(&mut self.hoisted);
        let mut body = self.block_of(body)?;
        if lead.is_empty() {
            out.push(Stmt::While { cond, body });
            return Ok(());
        }
        // The condition reads memory: it must be re-read every iteration, so `while (1) { loads; if (!c) break; body }`.
        let (zero, _) = self.constant(0);
        lead.push(Stmt::If {
            cond: ScalarExpr::Select(
                Box::new(cond),
                Box::new(zero.clone()),
                Box::new(self.constant(1).0),
            ),
            then: Block(vec![Stmt::Break]),
            els: Block::default(),
        });
        lead.append(&mut body.0);
        out.push(Stmt::While {
            cond: ScalarExpr::ConstInt(1, E::I32),
            body: Block(lead),
        });
        Ok(())
    }

    /// `#pragma unroll` and `#pragma unroll N`: a request to the emitter, never a meaning (`Stmt::For::unroll`). Any other
    /// directive is a typed decline.
    pub(super) fn pragma(&mut self, n: Node<'_>) -> R<()> {
        let directive = n
            .child_by_field_name("directive")
            .map(|d| text(d, self.src));
        let arg = n
            .child_by_field_name("argument")
            .map_or("", |a| text(a, self.src).trim());
        let mut words = arg.split_whitespace();
        if directive != Some("#pragma") || words.next() != Some("unroll") {
            return self.not_yet(n, format!("the directive `{}`", text(n, self.src).trim()));
        }
        self.pending_unroll = match (words.next(), words.next()) {
            (None, _) => Unroll::Full,
            (Some(c), None) => match c.parse::<u32>() {
                Ok(0) | Err(_) => {
                    return self
                        .not_yet(n, "a `#pragma unroll` count that is not a positive literal");
                }
                Ok(1) => Unroll::None,
                Ok(k) => Unroll::Count(k),
            },
            _ => return self.not_yet(n, "a `#pragma unroll` with trailing text"),
        };
        Ok(())
    }

    pub(super) fn for_statement(
        &mut self,
        n: Node<'_>,
        out: &mut Vec<Stmt>,
        unroll: Unroll,
    ) -> R<()> {
        let Some(shape) = self.counted_for(n)? else {
            if self.constant_for(n, out)? {
                return Ok(());
            }
            return self.not_yet(
                n,
                "a `for` that is not a counted loop `for (T i = a; i < b; i += s)`",
            );
        };
        let (name, ty, start, end, step, body) = shape;
        // the start is evaluated once, before the loop: its loads go before the `For`
        self.flush(out);
        let local = self.new_local(ty);
        self.scopes.push(HashMap::new());
        self.scopes
            .last_mut()
            .expect("a scope is open")
            .insert(name, Sym::Scalar(local, ty, Fixed::Counter));
        let body = self.block_of(body);
        self.scopes.pop();
        out.push(Stmt::For {
            local,
            start,
            end,
            step,
            unroll,
            body: body?,
        });
        Ok(())
    }

    /// The pieces of `for (T i = a; i < b; i += s) body`, or `None` for any other shape.
    #[allow(clippy::type_complexity)]
    pub(super) fn counted_for<'n>(
        &mut self,
        n: Node<'n>,
    ) -> R<Option<(String, E, ScalarExpr, ScalarExpr, ScalarExpr, Node<'n>)>> {
        let (Some(init), Some(cond), Some(upd), Some(body)) = (
            n.child_by_field_name("initializer"),
            n.child_by_field_name("condition"),
            n.child_by_field_name("update"),
            n.child_by_field_name("body"),
        ) else {
            return Ok(None);
        };
        let Some((name, ty, a)) = self.for_init(init)? else {
            return Ok(None);
        };
        let (Some(b), Some(op)) = (
            cond.child_by_field_name("right"),
            cond.child_by_field_name("operator"),
        ) else {
            return Ok(None);
        };
        let lhs_is_counter = cond
            .child_by_field_name("left")
            .is_some_and(|l| text(l, self.src) == name);
        let op = text(op, self.src);
        // `i < b` and `i <= b` count up; `i > b` and `i >= b` count down (a signed counter only: an unsigned `i >= 0` never ends in C)
        let down = matches!(op, ">" | ">=");
        // `i != b`: the direction is the update's (`i++`/`i += s` up, `i--`/`i -= s` down). The IR stops when the counter is no
        // longer below (above) `b`; C stops only on reaching it, so the two agree wherever the C loop ends at all.
        let ne = op == "!=";
        if cond.kind() != "binary_expression"
            || !(matches!(op, "<" | "<=" | "!=") || down)
            || !lhs_is_counter
        {
            return Ok(None);
        }
        if down && int_range(ty).is_some_and(|(lo, _)| lo == 0) {
            return Ok(None);
        }
        let assigned = self.loop_assigned(n);
        let mut down = down;
        let mut step = self.for_step(upd, &name, ty, down, &assigned)?;
        if ne && step.is_none() && int_range(ty).is_none_or(|(lo, _)| lo != 0) {
            down = true;
            step = self.for_step(upd, &name, ty, down, &assigned)?;
        }
        let Some(step) = step else {
            return Ok(None);
        };
        self.require_invariant(b, &assigned)?;
        let (be, bt) = self.expr(b)?;
        let mut be = self.convert(be, bt, ty, b)?;
        // a `!=` loop whose start and bound are both known now is left to the constant-loop path, which runs it as C does and
        // declines one that never ends; the IR's stop-when-past refinement is only for bounds known at run time
        if ne && self.launch_fold(&a).is_some() && self.launch_fold(&be).is_some() {
            return Ok(None);
        }
        if op == ">=" {
            // the IR's limit is exclusive: `i >= b` stops above `b - 1`
            be = ScalarExpr::Sub(Box::new(be), Box::new(ScalarExpr::ConstInt(1, ty)));
        } else if op == "<=" {
            // `i <= b` stops at `b + 1`
            be = ScalarExpr::Add(Box::new(be), Box::new(ScalarExpr::ConstInt(1, ty)));
        }
        Ok(Some((name, ty, a, be, step, body)))
    }

    pub(super) fn for_init(&mut self, init: Node<'_>) -> R<Option<(String, E, ScalarExpr)>> {
        if init.kind() != "declaration" {
            return Ok(None);
        }
        let ty = self.type_of(init.child_by_field_name("type"), init)?;
        let d = init.child_by_field_name("declarator");
        let (Some(d), true) = (d, int_range(ty).is_some()) else {
            return Ok(None);
        };
        let (Some(id), Some(v)) = (
            d.child_by_field_name("declarator"),
            d.child_by_field_name("value"),
        ) else {
            return Ok(None);
        };
        let (a, at) = self.expr(v)?;
        let a = self.convert(a, at, ty, v)?;
        Ok(Some((text(id, self.src).to_string(), ty, a)))
    }

    pub(super) fn for_step(
        &mut self,
        upd: Node<'_>,
        counter: &str,
        ty: E,
        down: bool,
        assigned: &HashSet<String>,
    ) -> R<Option<ScalarExpr>> {
        // an update that moves the counter the way the condition tests: `++` / `+=` up, `--` / `-=` down
        let (inc, assign) = if down { ("--", "-=") } else { ("++", "+=") };
        let sign = if down { -1 } else { 1 };
        match upd.kind() {
            "update_expression" => {
                let ok = upd
                    .child_by_field_name("argument")
                    .is_some_and(|a| text(a, self.src) == counter)
                    && upd
                        .child_by_field_name("operator")
                        .is_some_and(|o| text(o, self.src) == inc);
                Ok(ok.then_some(ScalarExpr::ConstInt(sign, ty)))
            }
            "assignment_expression" => {
                let l = upd.child_by_field_name("left");
                let op = upd.child_by_field_name("operator");
                let r = upd.child_by_field_name("right");
                let (Some(l), Some(op), Some(r)) = (l, op, r) else {
                    return Ok(None);
                };
                if text(l, self.src) != counter || text(op, self.src) != assign {
                    return Ok(None);
                }
                self.require_invariant(r, assigned)?;
                let (s, st) = self.expr(r)?;
                let s = self.convert(s, st, ty, r)?;
                Ok(Some(if down {
                    ScalarExpr::Sub(Box::new(ScalarExpr::ConstInt(0, ty)), Box::new(s))
                } else {
                    s
                }))
            }
            _ => Ok(None),
        }
    }

    /// The names that can change while the loop `n` runs: those assigned inside it (its body and update), plus every name passed
    /// to a reference or pointer parameter of a source-defined function it calls (the callee can assign through it). A call
    /// that cannot be matched to its definition counts every name the function assigns anywhere. A bound naming a variable
    /// assigned only *outside* the loop (an enclosing loop's counter, a value set before it) is the same on every iteration.
    fn loop_assigned(&self, n: Node<'_>) -> HashSet<String> {
        let mut out = HashSet::new();
        collect_assigned(n, self.src, &mut out);
        if !self.add_escaping_args(n, &mut out) {
            return self.assigned.clone();
        }
        out
    }

    /// Adds the names a defined function called inside `n` can assign through its arguments; `false` if a call has no usable
    /// definition (the caller then falls back to the function-wide set).
    fn add_escaping_args(&self, n: Node<'_>, out: &mut HashSet<String>) -> bool {
        if n.kind() == "call_expression"
            && let Some(f) = n.child_by_field_name("function")
            && let Some(&func) = self.functions.get(text(f, self.src))
        {
            let params = super::inline::callee_params(func);
            let args = self.call_args(n);
            let Some(params) = params else { return false };
            let mut c = params.walk();
            let params: Vec<_> = params.named_children(&mut c).collect();
            if params.len() != args.len() {
                return false;
            }
            for (p, a) in params.into_iter().zip(args) {
                let t = text(p, self.src);
                if p.kind() != "parameter_declaration" {
                    // a shape not understood (a variadic, a default): anything in the argument may alias
                    collect_identifiers(a, self.src, out);
                } else if t.contains('&') {
                    // a reference: the callee assigns the argument's variable itself
                    lvalue_roots(a, self.src, out);
                } else if t.contains('*') {
                    // a pointer: a scalar local is reachable only as `&local`; an offset (`p + n`) only names elements
                    address_taken(a, self.src, out);
                }
                // a by-value parameter is the callee's own copy
            }
        }
        let mut c = n.walk();
        n.named_children(&mut c)
            .all(|k| self.add_escaping_args(k, out))
    }

    /// A `For` bound is read once by the IR and every iteration by C: they agree only if nothing it names is assigned while the
    /// loop runs (`assigned`, from `loop_assigned`) and nothing is loaded from memory.
    pub(super) fn require_invariant(&self, n: Node<'_>, assigned: &HashSet<String>) -> R<()> {
        if n.kind() == "subscript_expression" || n.kind() == "call_expression" {
            return self.not_yet(n, "a loop bound that reads memory or calls a function");
        }
        if n.kind() == "identifier" && assigned.contains(text(n, self.src)) {
            return self.not_yet(
                n,
                "a loop bound or step that names a variable the function assigns",
            );
        }
        let mut c = n.walk();
        n.named_children(&mut c)
            .try_for_each(|k| self.require_invariant(k, assigned))
    }

    // ---- expressions ----
}

fn collect_identifiers(n: Node<'_>, src: &str, out: &mut HashSet<String>) {
    if n.kind() == "identifier" {
        out.insert(text(n, src).to_string());
    }
    let mut c = n.walk();
    for k in n.named_children(&mut c) {
        collect_identifiers(k, src, out);
    }
}

/// The variables an lvalue argument names: `x`, `x.f`, `x[i]` and `(x)` give `x`; any other shape gives every identifier in it.
fn lvalue_roots(n: Node<'_>, src: &str, out: &mut HashSet<String>) {
    match n.kind() {
        "identifier" => {
            out.insert(text(n, src).to_string());
        }
        "parenthesized_expression" => {
            if let Some(k) = n.named_child(0) {
                lvalue_roots(k, src, out);
            }
        }
        "subscript_expression" | "field_expression" => {
            if let Some(k) = n.child_by_field_name("argument") {
                lvalue_roots(k, src, out);
            }
        }
        _ => collect_identifiers(n, src, out),
    }
}

/// The variables whose address an argument takes (`&x`, `&x.f`, `&x[i]`): the only scalars a pointer parameter can reach.
fn address_taken(n: Node<'_>, src: &str, out: &mut HashSet<String>) {
    if n.kind() == "pointer_expression"
        && n.child_by_field_name("operator")
            .is_some_and(|o| text(o, src) == "&")
        && let Some(k) = n.child_by_field_name("argument")
    {
        lvalue_roots(k, src, out);
    }
    let mut c = n.walk();
    for k in n.named_children(&mut c) {
        address_taken(k, src, out);
    }
}
