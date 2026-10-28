// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hand-checked worked examples for `unpopped_vocab::layout_algebra`, and the
//! malformed-input declines. The expected values were derived by hand from the
//! definitions (CuTe `01_layout.md` / `02_layout_algebra.md`), not copied from
//! this crate's output.

use unpopped_vocab::layout_algebra::{IntTuple, Layout, LayoutError};

fn l(s: &str) -> Layout {
    // "2,(3,4):1,(2,6)"-style parser, flat enough for the examples below.
    fn parse(it: &mut std::iter::Peekable<std::str::Chars<'_>>) -> IntTuple {
        if it.peek() == Some(&'(') {
            it.next();
            let mut v = vec![];
            loop {
                v.push(parse(it));
                match it.next() {
                    Some(',') => continue,
                    Some(')') => break,
                    c => panic!("bad char {c:?}"),
                }
            }
            IntTuple::Tuple(v)
        } else {
            let mut n = String::new();
            while it.peek().is_some_and(|c| c.is_ascii_digit()) {
                n.push(it.next().unwrap());
            }
            IntTuple::Int(n.parse().unwrap())
        }
    }
    let (a, b) = s.split_once(':').unwrap();
    Layout::new(
        parse(&mut a.chars().peekable()),
        parse(&mut b.chars().peekable()),
    )
    .unwrap()
}

#[test]
fn worked_examples() {
    // CuTe 01_layout.md: (2,(3,4)):(1,(2,6)) has size 24, rank 2, depth 2, cosize 24.
    let a = l("(2,(3,4)):(1,(2,6))");
    assert_eq!((a.size(), a.rank(), a.depth(), a.cosize()), (24, 2, 2, 24));
    assert_eq!(a.to_string(), "(2,(3,4)):(1,(2,6))");
    // Natural coordinate (1,(2,3)) -> 1 + 2*2 + 3*6 = 23; linear index 23 agrees.
    let crd = IntTuple::tuple([IntTuple::Int(1), IntTuple::ints([2, 3])]);
    assert_eq!(a.crd2idx(&crd), Ok(23));
    assert_eq!(a.call(23), Ok(23));
    // An integer against a nested mode is a linear index into it: 11 -> (2,3) -> 22.
    assert_eq!(
        a.crd2idx(&IntTuple::tuple([IntTuple::Int(1), IntTuple::Int(11)])),
        Ok(23)
    );
    assert_eq!(a.shape().idx2crd(23), Ok(crd));
    // Out of range and wrong profile are errors, not wrong answers.
    assert_eq!(
        a.crd2idx(&IntTuple::ints([2, 0])),
        Err(LayoutError::OutOfRange)
    );
    assert_eq!(
        a.crd2idx(&IntTuple::ints([0, 0, 0])),
        Err(LayoutError::ProfileMismatch)
    );
    assert_eq!(a.call(-1), Err(LayoutError::OutOfRange));
}

#[test]
fn worked_coalesce_and_composition() {
    // coalesce: (2,(1,6)):(1,(6,2)) -> 12:1
    assert_eq!(l("(2,(1,6)):(1,(6,2))").coalesce().to_string(), "12:1");
    assert_eq!(l("(2,1):(3,9)").coalesce().to_string(), "2:3");
    assert_eq!(l("(1,1):(5,7)").coalesce().to_string(), "1:0");
    // Two stride-0 modes merge; a stride-0 next to a nonzero one does not.
    assert_eq!(l("(2,3):(0,0)").coalesce().to_string(), "6:0");
    assert_eq!(l("(2,3):(0,1)").coalesce().to_string(), "(2,3):(0,1)");

    // composition: (6:1) o (3:2) = 3:2 ; (4,6):(1,4) o (2,3):(3,... skipped modes).
    assert_eq!(l("6:1").composition(&l("3:2")).unwrap().to_string(), "3:2");
    assert_eq!(
        l("(4,6):(1,4)").composition(&l("6:2")).unwrap().to_string(),
        "(2,3):(2,4)"
    );
    // by-mode with a hierarchical right operand keeps its profile:
    assert_eq!(
        l("20:2")
            .composition(&l("(5,4):(4,1)"))
            .unwrap()
            .to_string(),
        "(5,4):(8,2)"
    );
    // Divisibility failure is a typed decline.
    assert_eq!(
        l("(4,2):(1,4)").composition(&l("2:3")),
        Err(LayoutError::NotDivisible)
    );
    assert_eq!(
        l("(4,2):(1,4)").composition(&l("2:0")).unwrap().to_string(),
        "2:0"
    );
}

#[test]
fn worked_complement_divide_product() {
    // complement: 4:2 in 16 -> (2,2):(1,8) ; 2:1 in 6 -> 3:2
    assert_eq!(l("4:2").complement(16).unwrap().to_string(), "(2,2):(1,8)");
    assert_eq!(l("2:1").complement(6).unwrap().to_string(), "3:2");
    assert_eq!(l("4:1").complement(4).unwrap().to_string(), "1:0");
    assert_eq!(l("2:0").complement(8), Err(LayoutError::ZeroStrideMode));
    assert_eq!(
        l("(2,2):(1,1)").complement(8),
        Err(LayoutError::NotComplementable)
    );
    assert_eq!(
        l("4:2").complement(0),
        Err(LayoutError::NonPositiveCotarget(0))
    );

    // divide: 8:1 / 2:1 = (2,4):(1,2) ; 8:1 / 2:4 = (2,4):(4,1)
    assert_eq!(
        l("8:1").logical_divide(&l("2:1")).unwrap().to_string(),
        "(2,4):(1,2)"
    );
    assert_eq!(
        l("8:1").logical_divide(&l("2:4")).unwrap().to_string(),
        "(2,4):(4,1)"
    );
    // product: 2:1 x 3:1 = (2,3):(1,2)
    assert_eq!(
        l("2:1").logical_product(&l("3:1")).unwrap().to_string(),
        "(2,3):(1,2)"
    );
}

#[test]
fn validation_rejects_malformed_layouts() {
    let i = IntTuple::Int;
    assert_eq!(
        Layout::new(i(2), IntTuple::ints([1, 2])),
        Err(LayoutError::NotCongruent)
    );
    assert_eq!(
        Layout::new(IntTuple::Tuple(vec![]), IntTuple::Tuple(vec![])),
        Err(LayoutError::EmptyTuple)
    );
    assert_eq!(
        Layout::new(i(0), i(1)),
        Err(LayoutError::NonPositiveShape(0))
    );
    assert_eq!(
        Layout::new(i(2), i(-1)),
        Err(LayoutError::NegativeStride(-1))
    );
    assert_eq!(Layout::new(i(i64::MAX), i(2)), Err(LayoutError::Overflow));
    let mut deep = i(2);
    let mut deep_s = i(1);
    for _ in 0..20 {
        deep = IntTuple::Tuple(vec![deep]);
        deep_s = IntTuple::Tuple(vec![deep_s]);
    }
    assert_eq!(Layout::new(deep, deep_s), Err(LayoutError::TooDeep));
    // col_major agrees with the by-hand layout of the worked example.
    let cm =
        Layout::col_major(IntTuple::tuple([IntTuple::Int(2), IntTuple::ints([3, 4])])).unwrap();
    assert_eq!(cm.to_string(), "(2,(3,4)):(1,(2,6))");
}
