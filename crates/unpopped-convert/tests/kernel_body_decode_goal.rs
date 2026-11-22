// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The U19 goal test** (ROADMAP U19 slice 5): lift the whole of FlashInfer's `BatchDecodeWithPagedKVCacheKernel` (Baracuda's
//! flattened single configuration, `tests/fixtures/README.md`), **run it on the oracle**, and compare `o` and `lse` against an
//! independent host reference: softmax(sm_scale * q . K^T) V in `f64`, gathered by hand out of the same page table, over the
//! f16-rounded inputs, so the only error measured is the kernel's.
//!
//! **What this result is, and is not: a LENIENT run, never "oracle-clean".** The strict `run_launch` refuses this kernel three ways,
//! each a real property of FlashInfer's code that hardware shrugs off, and each allowed here by an opt-in
//! (`run_launch_lenient`, `Leniency`):
//!
//! 1. it **reads shared slots no thread filled**: a predicated `cp.async` that does not fill leaves the tokens past a request's
//!    length unwritten, `compute_qk` reads them, and the logit is masked to `-INF` afterwards (`UninitializedShared`). The opt-in
//!    is *proven benign here*, not assumed: `decode_output_does_not_depend_on_the_never_filled_shared_slots` runs the same launches
//!    with those slots zero and with two different non-zero patterns and compares `o` and `lse` bitwise.
//! 2. every `tx` thread of a row **stores the same value** to `smem_md` in `sync_state` (`DataRace` in shared), and
//! 3. to `lse[...]` under `if (tz == 0)` (`DataRace` in global memory). Both are same-bits stores; `same_value_writes` allows
//!    those only, and `tests/kernel_lenient.rs` pins that a different value, `+0.0` against `-0.0`, or any read still races.
//!
//! The lse of an empty request is upstream's `-INF` sentinel merged over the 8 slices (about -4.99e4), not an SDPA value, so it is
//! asserted to be that sentinel rather than compared. The error bound is against an `f64` reference, on the cases below only
//! (several request counts, heads, page sizes, shuffled physical pages, `kv_len` at page edges, a long request that runs the
//! `kv_offset_smem` refresh, an empty request, `lse` null); a control with a 1% wrong scale must be seen.
//! Found by running it, each now its own regression test: `a_local_alias_of_a_null_buffer_is_null_too`
//! (`kernel_body_context.rs`) and `a_c_remainder_in_a_float_kernel_runs_on_the_oracle` (`kernel_body_decode.rs`).
use unpopped::exec::LaunchShape;
use unpopped::kernel::ParamKind;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::{Leniency, run_launch_lenient};
use unpopped_convert::{Frontend, LiftContext, lift_kernel_in};
use unpopped_vocab::ElementKind as E;

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut b = src.to_string();
    for q in [
        "__host__",
        "__device__",
        "__forceinline__",
        "__grid_constant__",
        "__global__",
        "__shared__",
    ] {
        b = b.replace(q, &" ".repeat(q.len()));
    }
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(b, None)
}

struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
}

fn i32s(v: &[i32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::from_i32(&[v.len() as i64], v))
}
fn halves(v: &[f64]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::from_f64_as(E::F16, &[v.len() as i64], v))
}
fn dummy_i32() -> KernelArg {
    i32s(&[0])
}

fn fastdiv(d: u32) -> (u32, u32, u32) {
    if d <= 1 {
        return (d, 0, 0);
    }
    let l = 32 - (d - 1).leading_zeros();
    let m = (((1u64 << 32) * ((1u64 << l) - u64::from(d))) / u64::from(d) + 1) as u32;
    (d, m, l)
}

fn run_case(b: usize, h: usize, page: usize, lens: &[usize], seed: u32) {
    run_case_scaled(b, h, page, lens, seed, 1.0, true, Leniency::all(), false);
}

fn run_case_scaled(
    b: usize,
    h: usize,
    page: usize,
    lens: &[usize],
    seed: u32,
    ref_scale: f64,
    expect_ok: bool,
    len: Leniency,
    lse_null: bool,
) -> Vec<u8> {
    const D: usize = 128;
    let src = include_str!("fixtures/flashinfer_paged_decode_f16_d128_g1_flat.cu");
    let c = Frontend::new("c", "void", &[], "out", "in", parse);
    let ctx = LiftContext::new(
        LaunchShape::new([16, 1, 8], [b as u32, h as u32, 1]).with_subgroup(32, 32),
    )
    .with_dynamic_shared_bytes(36864)
    .with_define("__CUDACC_VER_MAJOR__", 13)
    .with_define("__CUDA_ARCH__", 800)
    .with_null_buffer("q_rope_offset")
    .with_null_buffer("maybe_alibi_slopes")
    .with_null_buffer("block_valid_mask_ptr")
    .with_null_buffer("rope_pos_offset");
    let ctx = if lse_null {
        ctx.with_null_buffer("lse_ptr")
    } else {
        ctx
    };
    let k = lift_kernel_in(
        &c,
        &src,
        "BatchDecodeWithPagedKVCacheKernel_f16_d128_g1_none",
        &ctx,
    )
    .expect("lifts");
    let mut rng = Rng(seed);
    let mut indptr = vec![0i32; b + 1];
    let mut last = vec![0i32; b];
    for i in 0..b {
        let np = lens[i].div_ceil(page);
        indptr[i + 1] = indptr[i] + np as i32;
        last[i] = if np > 0 {
            (lens[i] - (np - 1) * page) as i32
        } else {
            0
        };
    }
    let total = indptr[b] as usize;
    let pool = total + 3;
    let mut perm: Vec<i32> = (0..pool as i32).collect();
    for i in (1..pool).rev() {
        let j = (rng.next() as usize) % (i + 1);
        perm.swap(i, j);
    }
    let indices: Vec<i32> = perm[..total].to_vec();
    let kvn = pool * h * page * D;
    let val = |r: &mut Rng, div: f64| f64::from((r.next() % 2001) as i32 - 1000) / div;
    let kk: Vec<f64> = (0..kvn).map(|_| val(&mut rng, 1000.0)).collect();
    let vv: Vec<f64> = (0..kvn).map(|_| val(&mut rng, 1000.0)).collect();
    let qq: Vec<f64> = (0..b * h * D).map(|_| val(&mut rng, 500.0)).collect();
    let (kb, vb, qb) = (halves(&kk), halves(&vv), halves(&qq));
    // the kernel reads the f16-rounded values, so the reference does too
    let rd = |a: &KernelArg| match a {
        KernelArg::Buffer(t) => t.to_f64_vec(),
        _ => unreachable!(),
    };
    let (kr, vr, qr) = (rd(&kb), rd(&vb), rd(&qb));
    let sm_scale = (1.0f32 / 128f32.sqrt()) as f64;
    let ref_sm = sm_scale * ref_scale;
    let mut want_o = vec![0.0f64; b * h * D];
    let mut want_l = vec![0.0f64; b * h];
    for r in 0..b {
        for hh in 0..h {
            let n = lens[r];
            let mut s = Vec::new();
            for t in 0..n {
                let (pg, e) = (t / page, t % page);
                let phys = indices[indptr[r] as usize + pg] as usize;
                let base = ((phys * h + hh) * page + e) * D;
                let dot: f64 = (0..D)
                    .map(|d| qr[(r * h + hh) * D + d] * kr[base + d])
                    .sum();
                s.push(dot * ref_sm);
            }
            if n == 0 {
                want_l[r * h + hh] = f64::NAN; // an empty request: lse is upstream's sentinel near -INF, not an SDPA value
                continue;
            }
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let den: f64 = s.iter().map(|x| (x - m).exp()).sum();
            want_l[r * h + hh] = (m + den.ln()) / std::f64::consts::LN_2;
            for d in 0..D {
                let mut acc = 0.0;
                for t in 0..n {
                    let (pg, e) = (t / page, t % page);
                    let phys = indices[indptr[r] as usize + pg] as usize;
                    acc += (s[t] - m).exp() / den * vr[((phys * h + hh) * page + e) * D + d];
                }
                want_o[(r * h + hh) * D + d] = acc;
            }
        }
    }
    let (pd, pm, pl) = fastdiv(page as u32);
    let ri: Vec<i32> = (0..b as i32).collect();
    let oi: Vec<i32> = (0..=b as i32).collect();
    let by_name = |name: &str| -> KernelArg {
        match name {
            "params.padded_batch_size" | "params.batch_size" => KernelArg::Int(b as i128),
            "params.num_qo_heads" | "params.num_heads" => KernelArg::Int(h as i128),
            "params.q_stride_n" => KernelArg::Int((h * D) as i128),
            "params.q_stride_h" | "params.head_dim" | "params.stride_n" => {
                KernelArg::Int(D as i128)
            }
            "params.window_left" => KernelArg::Int(-1),
            "params.logits_soft_cap" => KernelArg::Float(0.0),
            "params.sm_scale" => KernelArg::Float(sm_scale),
            "params.rope_rcp_scale" => KernelArg::Float(1.0),
            "params.rope_rcp_theta" => KernelArg::Float(1e-4),
            "params.partition_kv" => KernelArg::Int(0),
            "params.page_size.d_" => KernelArg::Int(i128::from(pd)),
            "params.page_size.m_" => KernelArg::Int(i128::from(pm)),
            "params.page_size.l_" => KernelArg::Int(i128::from(pl)),
            "params.stride_page" => KernelArg::Int((h * page * D) as i128),
            "params.stride_h" => KernelArg::Int((page * D) as i128),
            other => panic!("unmapped struct field {other}"),
        }
    };
    let ParamKind::Struct { fields } = &k.params[16].kind else {
        panic!("param 16 is not the struct")
    };
    let sargs = KernelArg::Struct(fields.iter().map(|f| by_name(&f.name)).collect());
    let mut args = vec![
        qb,
        dummy_i32(),
        halves(&vec![0.0; b * h * D]),
        KernelArg::Buffer(TypedBuffer::from_f32(&[(b * h) as i64], &vec![0.0; b * h])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0])),
        i32s(&ri),
        i32s(&vec![0; b]),
        i32s(&oi),
        i32s(&[i32::MAX]),
        KernelArg::Buffer(TypedBuffer::from_u8(&[1], &[0])),
        kb,
        vb,
        i32s(&indices),
        i32s(&indptr),
        i32s(&last),
        dummy_i32(),
        sargs,
    ];
    let t0 = std::time::Instant::now();
    run_launch_lenient(&k, &mut args, len).expect("the lifted kernel runs on the oracle");
    println!(
        "RUN b={b} h={h} page={page} lens={lens:?} took {:?}",
        t0.elapsed()
    );
    let (got_o, got_l) = (rd(&args[2]), rd(&args[3]));
    let (mut eo, mut el) = (0.0f64, 0.0f64);
    for i in 0..want_o.len() {
        eo = eo.max((got_o[i] - want_o[i]).abs());
    }
    for i in 0..want_l.len() {
        if want_l[i].is_nan() {
            assert!(
                got_l[i] < -4.0e4,
                "an empty request's lse is the -INF sentinel: {}",
                got_l[i]
            );
        } else if !lse_null {
            el = el.max((got_l[i] - want_l[i]).abs());
        }
    }
    println!("LSE got={:?} want={:?}", &got_l, &want_l);
    println!(
        "O got[0..8]={:?}
  want[0..8]={:?}",
        &got_o[..8],
        &want_o[..8]
    );
    println!("MAXERR o={eo:e} lse={el:e}");
    if expect_ok {
        assert!(eo < 1e-3, "o error {eo}");
        if lse_null {
            assert!(
                got_l.iter().all(|x| *x == 0.0),
                "a null lse must stay untouched"
            );
        } else {
            assert!(el < 1e-5, "lse error {el}");
        }
    } else {
        assert!(el > 1e-3, "the control must be seen: o {eo} lse {el}");
    }
    let bytes = |a: &KernelArg| match a {
        KernelArg::Buffer(t) => t.bytes.clone(),
        _ => unreachable!(),
    };
    let mut out = bytes(&args[2]);
    out.extend(bytes(&args[3]));
    out
}

#[test]
fn decode_vs_reference_small() {
    run_case(1, 1, 16, &[5], 7);
}

#[test]
fn decode_control_a_one_percent_wrong_scale_is_seen() {
    run_case_scaled(1, 1, 16, &[40], 11, 1.01, false, Leniency::all(), false);
}

#[test]
fn decode_multi_request_multi_head_shuffled_pages() {
    run_case(2, 2, 16, &[37, 16], 21);
}

#[test]
fn decode_several_tiles_and_a_partial_last_page() {
    run_case(1, 1, 16, &[130], 31);
}

#[test]
fn decode_page_size_one_and_a_short_request() {
    run_case(2, 1, 1, &[9, 3], 41);
}

#[test]
fn decode_long_enough_to_refresh_the_offset_table() {
    // 19 iterations of 32 tokens: the kv_offset_smem refresh at (iter + 2) % 16 == 0 runs
    run_case(1, 1, 16, &[600], 51);
}

#[test]
fn decode_output_does_not_depend_on_the_never_filled_shared_slots() {
    // zero_shared is only benign if the OUTPUT ignores the slots no thread filled: the same launch with those slots zero and with
    // two different non-zero patterns must give the same bits, for o and for lse
    for (b, h, page, lens, seed) in [
        (1, 1, 16, vec![5usize], 7u32),
        (2, 2, 16, vec![37, 16], 21),
        (2, 1, 1, vec![9, 3], 41),
    ] {
        let go = |len: Leniency| run_case_scaled(b, h, page, &lens, seed, 1.0, true, len, false);
        let zero = go(Leniency::all());
        assert_eq!(
            zero,
            go(Leniency::all().with_shared_pattern(3)),
            "zeros vs pattern 3 for {lens:?}"
        );
        assert_eq!(
            zero,
            go(Leniency::all().with_shared_pattern(41)),
            "zeros vs pattern 41 for {lens:?}"
        );
    }
}

#[test]
fn decode_page_boundaries_and_an_empty_request() {
    // exactly one full page, one past it, exactly two pages; and a zero-length request beside a normal one
    run_case(3, 1, 16, &[16, 17, 32], 61);
    run_case(2, 1, 16, &[0, 21], 71);
}

#[test]
fn decode_with_lse_null_writes_o_and_leaves_lse_alone() {
    run_case_scaled(2, 2, 16, &[37, 16], 21, 1.0, true, Leniency::all(), true);
}
