//! Scratch census (not committed): run `lift_kernel_in` over a corpus, read-only and parse-only, and tabulate the outcome
//! per kernel. Usage: census <cuda|slang> <label> <dir> [path-substring-filter]
use regex::Regex;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use unpopped::exec::LaunchShape;
use unpopped_convert::{Frontend, KernelLiftError, LiftContext, lift_kernel_in};

/// baracuda-cuda-parse@ce2911956 pins `unpopped-convert ~0.47`, so its `CUDA` const is a 0.47 `Frontend` and cannot be passed to this
/// worktree's walker. The same descriptor from the crate's own exports: its `parse_cuda` and its (private) residue list, copied.
const CUDA_RESIDUE: &[&str] = &["__shared__", "atomicAdd", "atomicCAS", "__syncthreads", "cublas", "cudnn", "printf", "asm", "cp.async", "__shfl"];
fn cuda_frontend() -> Frontend {
    Frontend::new("cuda", "__global__", CUDA_RESIDUE, "out", "in", baracuda_cuda_parse::parse_cuda)
}

include!("constructs.rs");
include!("inst.rs");

fn walk(dir: &Path, exts: &[&str], out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, exts, out);
        } else if p.extension().and_then(|x| x.to_str()).is_some_and(|x| exts.contains(&x)) {
            out.push(p);
        }
    }
}

/// `__global__` kernel names in `src` (text-level; a prototype counts, and is classed separately).
fn cuda_kernels(src: &str) -> Vec<(String, bool, bool, usize)> {
    let lb = Regex::new(r"__launch_bounds__\s*\([^)]*\)|__maxnreg__\s*\([^)]*\)|__cluster_dims__\s*\([^)]*\)").unwrap();
    let cleaned = lb.replace_all(src, " ");
    let re = Regex::new(r"(?s)__global__\s+([^;{}()]*?)\b([A-Za-z_]\w*)\s*\(").unwrap();
    let mut out = Vec::new();
    for c in re.captures_iter(&cleaned) {
        let m = c.get(0).unwrap();
        let name = c[2].to_string();
        let mut from = m.start().saturating_sub(200);
        while !cleaned.is_char_boundary(from) {
            from += 1;
        }
        let before = &cleaned[from..m.start()];
        let is_template = before.contains("template");
        // a prototype: the first of `;` or `{` after the match is `;`
        let rest = &cleaned[m.end()..];
        let proto = match (rest.find(';'), rest.find('{')) {
            (Some(s), Some(b)) => s < b,
            (Some(_), None) => true,
            _ => false,
        };
        out.push((name, is_template, proto, m.start()));
    }
    out
}

fn slang_kernels(src: &str) -> Vec<(String, bool, bool, usize)> {
    let re = Regex::new(r"(?s)\[\s*numthreads[^\]]*\]\s*(?:\[[^\]]*\]\s*)*void\s+([A-Za-z_]\w*)\s*\(").unwrap();
    re.captures_iter(src).map(|c| (c[1].to_string(), false, false, c.get(0).unwrap().start())).collect()
}

fn normalize(s: &str) -> String {
    let q = Regex::new(r"`[^`]*`").unwrap();
    let n = Regex::new(r"\b\d+\b").unwrap();
    let t = n.replace_all(&q.replace_all(s, "`X`"), "N").to_string();
    t.chars().take(110).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (lang, label, dir) = (a[1].as_str(), a[2].as_str(), PathBuf::from(&a[3]));
    let instantiate_mode = a.iter().any(|x| x == "--instantiate");
    let filter = a.iter().skip(4).find(|x| !x.starts_with("--")).cloned().unwrap_or_default();
    let (fe, exts, extract): (Frontend, &[&str], fn(&str) -> Vec<(String, bool, bool, usize)>) = match lang {
        "cuda" => (cuda_frontend(), &["cu", "cuh"], cuda_kernels),
        "slang" => (fishstringer_slang_parse::SLANG, &["slang"], slang_kernels),
        _ => panic!("cuda or slang"),
    };
    let mut files = Vec::new();
    walk(&dir, exts, &mut files);
    files.sort();
    files.retain(|f| f.to_string_lossy().replace('\\', "/").contains(&filter));
    let mut classes: BTreeMap<String, (usize, String)> = BTreeMap::new();
    let mut by_group: BTreeMap<String, [usize; 3]> = BTreeMap::new(); // files, kernels, lifted
    let (mut kernels, mut lifted, mut templates, mut template_lifted, mut protos) = (0, 0, 0, 0, 0);
    let mut files_with = 0;
    let mut lifted_list = Vec::new();
    let mut detail: BTreeMap<String, (usize, String)> = BTreeMap::new();
    let mut syntax_examples: Vec<String> = Vec::new();
    let mut present: BTreeMap<&str, usize> = BTreeMap::new();
    let constructs: &[(&str, &str)] = if lang == "slang" { SLANG_CONSTRUCTS } else { CUDA_CONSTRUCTS };
    let compiled: Vec<(&str, Regex)> = constructs.iter().map(|(l, r)| (*l, Regex::new(r).unwrap())).collect();
    for f in &files {
        let Ok(bytes) = std::fs::read(f) else { continue };
        let src = String::from_utf8_lossy(&bytes).to_string();
        let rel = f.strip_prefix(&dir).unwrap_or(f).to_string_lossy().replace('\\', "/");
        let group = rel.split('/').take(if rel.contains("vendor") { 4 } else { 2 }).collect::<Vec<_>>().join("/");
        let ks = extract(&src);
        by_group.entry(group.clone()).or_default()[0] += 1;
        if !ks.is_empty() {
            files_with += 1;
        }
        for (l, re) in &compiled {
            if !ks.is_empty() && re.is_match(&src) {
                *present.entry(l).or_default() += ks.len();
            }
        }
        for (name, is_template, proto, at) in ks {
            kernels += 1;
            by_group.entry(group.clone()).or_default()[1] += 1;
            if is_template {
                templates += 1;
            }
            if proto {
                protos += 1;
            }
            let ctx = LiftContext::new(LaunchShape::new([128, 1, 1], [1, 1, 1]).with_subgroup(32, 32))
                .with_dynamic_shared_bytes(49152)
                .with_define("__CUDACC_VER_MAJOR__", 13)
                .with_define("__CUDA_ARCH__", 800);
            let inst_src;
            let use_src: &str = if instantiate_mode && is_template {
                let at_orig = src.find(&format!("{name}(")).or_else(|| src.find(&name)).unwrap_or(at);
                let k_at = src[..at_orig].rfind("__global__").unwrap_or(at_orig);
                inst_src = instantiate(&src, k_at);
                &inst_src
            } else {
                &src
            };
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lift_kernel_in(&fe, use_src, &name, &ctx)));
            let (class, line) = match r {
                Err(_) => ("PANIC in the walker".to_string(), 0),
                Ok(Ok(_)) => {
                    lifted += 1;
                    if is_template {
                        template_lifted += 1;
                    }
                    by_group.entry(group.clone()).or_default()[2] += 1;
                    lifted_list.push(format!("{rel}::{name}"));
                    continue;
                }
                Ok(Err(KernelLiftError::NotYetLifted { construct, line })) => {
                    let raw: String = Regex::new(r"\d+").unwrap().replace_all(&construct, "N").chars().take(120).collect();
                    detail.entry(format!("NotYetLifted: {raw}")).or_insert((0, format!("{rel}:{line} `{name}`"))).0 += 1;
                    (format!("NotYetLifted: {}", normalize(&construct)), line)
                }
                Ok(Err(KernelLiftError::OutsideIr { construct, line, .. })) => (format!("OutsideIr: {}", normalize(&construct)), line),
                Ok(Err(KernelLiftError::NoSuchFunction { .. })) => ("NoSuchFunction (prototype / macro-generated / out of the CST)".to_string(), 0),
                Ok(Err(KernelLiftError::NotAKernel { .. })) => ("NotAKernel".to_string(), 0),
                Ok(Err(KernelLiftError::SyntaxError { line })) => {
                    if syntax_examples.len() < 8 {
                        let text = src.lines().nth(line.saturating_sub(1)).unwrap_or("").trim().chars().take(110).collect::<String>();
                        syntax_examples.push(format!("{rel}:{line} `{name}`: {text}"));
                    }
                    ("SyntaxError (grammar)".to_string(), line)
                }
                Ok(Err(KernelLiftError::Invalid(e))) => (format!("Invalid (walker bug): {}", normalize(&format!("{e:?}"))), 0),
                Ok(Err(e)) => (format!("other: {}", normalize(&format!("{e:?}"))), 0),
            };
            let tag = if is_template { format!("{class}  [template]") } else { class };
            let e = classes.entry(tag).or_insert((0, format!("{rel}:{line} `{name}`")));
            e.0 += 1;
        }
    }
    println!("CENSUS label={label} lang={lang} dir={} filter={filter:?} heuristic-instantiation={instantiate_mode}", dir.display());
    println!("files scanned: {}  files with a kernel: {files_with}", files.len());
    println!("kernels found: {kernels} (of which template: {templates}, prototype-only: {protos})");
    println!("LIFTED: {lifted} of {kernels} (template kernels lifted: {template_lifted})");
    println!("-- by group (files, kernels, lifted)");
    for (g, v) in &by_group {
        println!("  {g}: {} files, {} kernels, {} lifted", v[0], v[1], v[2]);
    }
    let mut v: Vec<_> = classes.iter().collect();
    v.sort_by(|x, y| y.1.0.cmp(&x.1.0));
    println!("-- decline classes: {} distinct; first decline per kernel, count | class | first example", v.len());
    for (c, (n, ex)) in v.iter().take(25) {
        println!("  {n:5} | {c} | {ex}");
    }
    let mut d: Vec<_> = detail.iter().collect();
    d.sort_by(|x, y| y.1.0.cmp(&x.1.0));
    println!("-- NotYetLifted by exact construct, top 30 (first decline per kernel)");
    for (c, (n, ex)) in d.iter().take(40) {
        println!("  {n:5} | {c} | {ex}");
    }
    if !syntax_examples.is_empty() {
        println!("-- grammar errors, examples (the walker's first ERROR node inside the kernel)");
        for e in &syntax_examples {
            println!("  {e}");
        }
    }
    println!("-- constructs PRESENT (kernels in files containing it; static text match, not a decline; a kernel can need several)");
    let mut pv: Vec<_> = present.iter().collect();
    pv.sort_by(|x, y| y.1.cmp(x.1));
    for (l, n) in pv {
        println!("  {n:5} | {l}");
    }
    if lifted_list.len() <= 40 {
        println!("-- lifted:");
        for l in &lifted_list {
            println!("  {l}");
        }
    }
}
