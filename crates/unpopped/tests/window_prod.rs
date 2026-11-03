// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A `Prod` window** (ROADMAP U2 D2; `docs/options-u2-lift-declines.md`): an `Access::Window` whose combine is `Prod`
//! is the product of the in-bounds taps of each window, from the multiplicative identity `1`. It is an **Unpopped
//! extension, not KISS** (KISS has `avg_pool` and `max_pool` only; ThinkersJournal/KISS#767). KISS-OPS-6.13-0011 makes a
//! window with no in-bounds tap inexpressible, so the empty product never arises (`window_without_real_tap_declines.rs`
//! holds the decline). Floats multiply in `f64` and round once at the store; integers wrap at their width.
//!
//! The oracle is held to a product written here in Rust over the same window arithmetic; the lifted kernel is held to
//! the oracle byte for byte. The corpus has a zero (giving `-0` or `0`), a NaN, an infinity beside a zero (NaN), mixed
//! signs, and geometries that pad, stride and dilate, so edge windows hold fewer taps than `size` (a padding tap that
//! contributed `0` instead of `1` gives zero there).

use unpopped::ir::{OpDef, ReduceOp};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::lift_window_opdef;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, try_structure_key};

#[derive(Clone, Copy)]
struct Case {
    rows: i64,
    k_in: i64,
    size: i64,
    stride: i64,
    dilation: i64,
    pad: i64,
}

impl Case {
    fn k_out(&self) -> i64 {
        let span = self.dilation * (self.size - 1) + 1;
        (self.k_in + 2 * self.pad - span) / self.stride + 1
    }

    fn op(&self, dtype: ElementKind) -> OpDef {
        OpDef::window_simple(
            "win",
            &[dtype],
            ReduceOp::Prod,
            1,
            self.size as u8,
            self.stride as u8,
            self.dilation as u8,
            self.pad as u8,
            self.pad as u8,
            false,
        )
    }

    /// The in-bounds input positions of output `o`.
    fn taps(&self, o: i64) -> Vec<i64> {
        (0..self.size)
            .map(|t| o * self.stride - self.pad + t * self.dilation)
            .filter(|p| (0..self.k_in).contains(p))
            .collect()
    }
}

const BASE: Case = Case {
    rows: 2,
    k_in: 9,
    size: 3,
    stride: 2,
    dilation: 1,
    pad: 1,
};
const DILATED: Case = Case {
    rows: 2,
    k_in: 12,
    size: 3,
    stride: 3,
    dilation: 2,
    pad: 2,
};

const FLOATS: [f64; 11] = [
    2.0,
    -3.0,
    0.5,
    4.0,
    0.0,
    f64::INFINITY,
    -1.5,
    f64::NAN,
    1.25,
    -0.5,
    8.0,
];
const INTS: [i64; 11] = [7, -3, 5, 65_536, 2, 65_536, -9, 3, 1_000_003, -2, 11];

fn float_buffer(dtype: ElementKind, c: Case) -> (TypedBuffer, Vec<f64>) {
    let vals: Vec<f64> = (0..(c.rows * c.k_in) as usize)
        .map(|i| FLOATS[i % FLOATS.len()])
        .collect();
    let bytes = vals
        .iter()
        .flat_map(|&v| match dtype {
            ElementKind::F32 => (v as f32).to_le_bytes().to_vec(),
            _ => v.to_le_bytes().to_vec(),
        })
        .collect();
    let b = TypedBuffer::new(dtype, vec![c.rows, c.k_in], vec![c.k_in, 1], bytes);
    // the reference reads what the buffer holds (an F32 buffer rounds its inputs)
    let read: Vec<f64> = match dtype {
        ElementKind::F32 => vals.iter().map(|&v| f64::from(v as f32)).collect(),
        _ => vals,
    };
    (b, read)
}

fn int_buffer(c: Case) -> (TypedBuffer, Vec<i64>) {
    let vals: Vec<i64> = (0..(c.rows * c.k_in) as usize)
        .map(|i| INTS[i % INTS.len()])
        .collect();
    let bytes = vals
        .iter()
        .flat_map(|&v| (v as i32).to_le_bytes())
        .collect();
    (
        TypedBuffer::new(
            ElementKind::I32,
            vec![c.rows, c.k_in],
            vec![c.k_in, 1],
            bytes,
        ),
        vals,
    )
}

fn oracle(c: Case, dtype: ElementKind, src: &TypedBuffer) -> Vec<u8> {
    let k_out = c.k_out();
    let ops = [
        OperandDesc::new(2, &[c.rows, c.k_in], &[c.k_in, 1], dtype, 256),
        OperandDesc::new(2, &[c.rows, k_out], &[k_out, 1], dtype, 256),
    ];
    let key = try_structure_key(OpCategory::Pooling, &ops, ArchSku::Sm89).expect("a key");
    let op = c.op(dtype);
    let plan = try_build_plan(&op, &key).expect("a Prod window plans");
    evaluate(&plan, &ops, std::slice::from_ref(src), &[])
        .remove(0)
        .bytes
}

fn lifted(c: Case, dtype: ElementKind, src: &TypedBuffer, len: usize) -> Vec<u8> {
    let (rows, k_in, k_out) = (c.rows, c.k_in, c.k_out());
    let kernel = lift_window_opdef(&c.op(dtype), dtype, 2).expect("a Prod window lifts");
    assert_eq!(check_kernel(&kernel), Ok(()));
    let out = TypedBuffer::new(dtype, vec![rows, k_out], vec![k_out, 1], vec![0xAB; len]);
    let mut args = vec![KernelArg::Buffer(src.clone()), KernelArg::Buffer(out)];
    for e in [rows, k_in, k_out] {
        args.push(KernelArg::Int(i128::from(e)));
    }
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(got) = &args[1] else {
        unreachable!()
    };
    got.bytes[..len].to_vec()
}

/// The reference: per row and output, the product of the in-bounds taps from `1`, left to right in `f64`.
fn float_reference(c: Case, dtype: ElementKind, vals: &[f64]) -> Vec<u8> {
    let mut out = Vec::new();
    for r in 0..c.rows {
        for o in 0..c.k_out() {
            let p = c
                .taps(o)
                .iter()
                .fold(1.0f64, |acc, &t| acc * vals[(r * c.k_in + t) as usize]);
            match dtype {
                ElementKind::F32 => out.extend((p as f32).to_le_bytes()),
                _ => out.extend(p.to_le_bytes()),
            }
        }
    }
    out
}

fn same_bits(a: &[u8], b: &[u8], width: usize) -> bool {
    // a NaN compares equal to a NaN of any payload; everything else by bytes
    let nan = |c: &[u8]| match width {
        4 => f32::from_le_bytes(c.try_into().unwrap()).is_nan(),
        _ => f64::from_le_bytes(c.try_into().unwrap()).is_nan(),
    };
    a.len() == b.len()
        && a.chunks(width)
            .zip(b.chunks(width))
            .all(|(x, y)| x == y || (nan(x) && nan(y)))
}

#[test]
fn the_oracle_matches_a_product_written_in_rust() {
    for c in [BASE, DILATED] {
        for (dtype, width) in [(ElementKind::F64, 8), (ElementKind::F32, 4)] {
            let (src, vals) = float_buffer(dtype, c);
            let got = oracle(c, dtype, &src);
            let want = float_reference(c, dtype, &vals);
            assert!(same_bits(&got, &want, width), "{dtype:?} {}", c.k_in);
        }
    }
}

#[test]
fn an_integer_product_wraps_at_its_width() {
    let c = BASE;
    let (src, vals) = int_buffer(c);
    let got = oracle(c, ElementKind::I32, &src);
    let mut want = Vec::new();
    for r in 0..c.rows {
        for o in 0..c.k_out() {
            let p = c.taps(o).iter().fold(1i64, |acc, &t| {
                i64::from((acc.wrapping_mul(vals[(r * c.k_in + t) as usize])) as i32)
            });
            want.extend((p as i32).to_le_bytes());
        }
    }
    assert_eq!(got, want);
    // control: 65536 * 65536 overflows an i32, so the corpus really exercises the wrap
    assert_eq!((65_536i32).wrapping_mul(65_536), 0);
}

#[test]
fn an_edge_window_multiplies_only_the_taps_it_has() {
    // pad 1, size 3, stride 2: output 0 has taps at -1, 0, 1, so two in bounds. Row 0 of the F64 corpus starts
    // 2, -3: the first output is 2 * -3 = -6, not 0 (a padding tap that contributed 0) and not 2 * -3 * 1 again.
    let (src, _) = float_buffer(ElementKind::F64, BASE);
    let got = oracle(BASE, ElementKind::F64, &src);
    assert_eq!(f64::from_le_bytes(got[..8].try_into().unwrap()), -6.0);
}

#[test]
fn the_lifted_kernel_agrees_with_the_oracle_byte_for_byte() {
    for c in [BASE, DILATED] {
        for dtype in [ElementKind::F64, ElementKind::F32] {
            let (src, _) = float_buffer(dtype, c);
            let want = oracle(c, dtype, &src);
            assert_eq!(lifted(c, dtype, &src, want.len()), want, "{dtype:?}");
        }
        let (src, _) = int_buffer(c);
        let want = oracle(c, ElementKind::I32, &src);
        assert_eq!(lifted(c, ElementKind::I32, &src, want.len()), want, "I32");
    }
}
