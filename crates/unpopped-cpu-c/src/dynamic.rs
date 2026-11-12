// SPDX-License-Identifier: MIT OR Apache-2.0
//! Dynamic-extent allocations (ROADMAP U16): an `Extent::Param` allocation, such as CUDA's `extern __shared__`, whose
//! element count is a scalar launch argument.
//!
//! C has no portable way to size an automatic array at run time (MSVC has no VLAs), so a kernel with such an allocation
//! is emitted as two functions. The entry point reads the scalar, refuses a count that is not positive (status 2),
//! `calloc`s one zeroed copy of the allocation for every block of the grid (status 3 when that fails), calls the body
//! (`<name>_impl`, which takes the arrays as extra pointer parameters `sh<i>_dyn`) and frees them whatever the body
//! returns, so every `return 1` of a bounds check in the body stays a plain return.

use unpopped::backend::LowerError;
use unpopped::exec::{AllocDecl, Extent, MemorySpace};
use unpopped::kernel::{KernelFn, ParamKind};

use crate::kernel::{ctype, refuse};

/// The index of the scalar parameter bound to local `local`, if there is one.
fn param_of_local(k: &KernelFn, local: u32) -> Option<usize> {
    k.params
        .iter()
        .position(|p| matches!(p.kind, ParamKind::Scalar { local: l, .. } if l == local))
}

/// The element count of `a` as a C expression (`long long`): a literal, or the scalar parameter.
pub(crate) fn extent_expr(k: &KernelFn, a: &AllocDecl) -> Result<String, LowerError> {
    match a.extent {
        Extent::Static(n) => Ok(n.to_string()),
        Extent::Param(local) => param_of_local(k, local)
            .map(|i| format!("((long long)p{i})"))
            .ok_or_else(|| {
                refuse(format!(
                    "allocation `{}` is sized by a local that is no scalar parameter",
                    a.name
                ))
            }),
        _ => Err(refuse("an extent this emitter has not been taught")),
    }
}

/// The elements one block holds of `a`: its extent, times the block's threads for a `Private` array.
pub(crate) fn len_expr(k: &KernelFn, a: &AllocDecl, threads: u32) -> Result<String, LowerError> {
    let n = extent_expr(k, a)?;
    Ok(match (a.space, a.extent) {
        (MemorySpace::Private, Extent::Static(n)) => {
            (u64::from(n) * u64::from(threads)).to_string()
        }
        (MemorySpace::Private, _) => format!("({n} * {threads})"),
        _ => n,
    })
}

/// The allocations whose count is a launch argument, with their indexes.
pub(crate) fn dynamic_allocs(k: &KernelFn) -> Vec<(usize, &AllocDecl)> {
    k.allocs
        .iter()
        .enumerate()
        .filter(|(_, a)| matches!(a.extent, Extent::Param(_)))
        .collect()
}

/// The names the entry point passes on to its body: what `signature` declared, in order.
fn forwarded(k: &KernelFn) -> Vec<String> {
    let mut out = Vec::new();
    let devices = k.launch.map_or(1, |l| l.devices);
    for d in 0..devices {
        let sfx = if devices > 1 {
            format!("_d{d}")
        } else {
            String::new()
        };
        for (i, p) in k.params.iter().enumerate() {
            match p.kind {
                ParamKind::Buffer { .. } => out.extend([
                    format!("b{i}{sfx}"),
                    format!("b{i}{sfx}_shape"),
                    format!("b{i}{sfx}_stride"),
                    format!("b{i}{sfx}_base"),
                ]),
                _ => out.push(format!("p{i}")),
            }
        }
    }
    out
}

/// `signature` with the dynamic arrays appended as pointer parameters, as a `static` function.
pub(crate) fn impl_signature(k: &KernelFn, signature: &str) -> Result<String, LowerError> {
    let mut extra = Vec::new();
    for (i, a) in dynamic_allocs(k) {
        extra.push(format!("{}* sh{i}_dyn", ctype(a.elem, "an allocation")?));
    }
    let open = signature
        .strip_suffix(')')
        .ok_or_else(|| refuse("a signature this emitter did not write"))?;
    Ok(format!("static {open},\n    {})", extra.join(",\n    ")))
}

/// The entry point around the body `<name>_impl`: sizes, allocates, calls, frees.
pub(crate) fn entry(k: &KernelFn, signature: &str, name: &str) -> Result<String, LowerError> {
    let Some(info) = crate::launch::LaunchInfo::of(k) else {
        return Err(refuse("a dynamic extent without a launch shape"));
    };
    if info.devices > 1 {
        // `signature` repeats a scalar parameter per device, so the entry point has no single count to read
        return Err(refuse(
            "a dynamic-extent allocation in a launch of several devices (ROADMAP U16)",
        ));
    }
    let blocks = u64::from(info.grid.iter().product::<u32>()) * u64::from(info.devices);
    let threads = info.threads();
    let dynamic = dynamic_allocs(k);
    let mut text = format!("{signature} {{\n");
    for (i, a) in &dynamic {
        let (t, len) = (ctype(a.elem, "an allocation")?, len_expr(k, a, threads)?);
        text.push_str(&format!(
            "    long long dn{i} = (long long)({len}); if (dn{i} <= 0) return 2;\n"
        ));
        text.push_str(&format!(
            "    {t}* sh{i}_dyn = ({t}*)calloc((size_t)(dn{i} * {blocks}LL), sizeof({t}));\n"
        ));
        let freed: String = dynamic
            .iter()
            .take_while(|(j, _)| j < i)
            .map(|(j, _)| format!("free(sh{j}_dyn); "))
            .collect();
        text.push_str(&format!("    if (!sh{i}_dyn) {{ {freed}return 3; }}\n"));
    }
    let mut args = forwarded(k);
    args.extend(dynamic.iter().map(|(i, _)| format!("sh{i}_dyn")));
    text.push_str(&format!("    int st = {name}_impl({});\n", args.join(", ")));
    for (i, _) in &dynamic {
        text.push_str(&format!("    free(sh{i}_dyn);\n"));
    }
    text.push_str("    return st;\n}\n");
    Ok(text)
}
