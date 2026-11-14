// SPDX-License-Identifier: MIT OR Apache-2.0
//! What every module of this emitter shares (ROADMAP U28): the refusal, the C type of a scalar, buffer pointer and
//! signature spelling, the launch geometry, and the [`Emitter`] state itself. A leaf: it names no sibling but `typed`,
//! so `kernel`, `launch`, `dynamic`, `collective` and `mma` depend on it and never on each other for these.

use unpopped::backend::{LowerError, Lowering};
use unpopped::kernel::{BufferRole, KernelFn, ParamKind};
use unpopped_vocab::ElementKind;

use crate::typed;

pub(crate) fn refuse(detail: impl Into<String>) -> LowerError {
    LowerError::UnsupportedKernel {
        detail: format!("cpu_c: {}", detail.into()),
    }
}

/// The C type of a scalar the emitter handles, or the refusal.
pub(crate) fn ctype(t: ElementKind, what: &str) -> Result<&'static str, LowerError> {
    // a sub-byte value lives in a local as its widened nibble (`I4` sign-extended); only a buffer packs two to a byte
    match t {
        ElementKind::I4 => return Ok("signed char"),
        ElementKind::U4 => return Ok("unsigned char"),
        // a narrow float is its 16-bit pattern in C (`narrow.rs`)
        ElementKind::F16 | ElementKind::Bf16 => return Ok("unsigned short"),
        _ => {}
    }
    typed::c_type(t).ok_or_else(|| {
        refuse(format!(
            "{what} has dtype {t:?}, which this emitter does not spell"
        ))
    })
}

/// The C element type a buffer pointer points to, `const` unless the kernel writes the buffer.
pub(crate) fn buffer_ptr_base(elem: ElementKind, role: BufferRole) -> Result<String, LowerError> {
    let t = if unpopped::types::is_subbyte_int(elem) {
        "unsigned char" // packed two elements to a byte, low nibble first
    } else {
        ctype(elem, "a buffer")?
    };
    let q = if matches!(role, BufferRole::Write | BufferRole::ReadWrite) {
        ""
    } else {
        "const "
    };
    Ok(format!("{q}{t}"))
}

/// The C type of a buffer pointer: `const T*` unless the kernel writes the buffer.
pub(crate) fn buffer_ptr(elem: ElementKind, role: BufferRole) -> Result<String, LowerError> {
    Ok(format!("{}*", buffer_ptr_base(elem, role)?))
}

/// The out-pointer parameter of a callee's `i`-th value: `res`, then `res1`, `res2`, ...
pub(crate) fn res_name(i: usize) -> String {
    if i == 0 {
        "res".to_string()
    } else {
        format!("res{i}")
    }
}

/// The signature, with one line per C argument.
pub(crate) fn signature(
    k: &KernelFn,
    name: &str,
    (ids, results): (bool, &[&'static str]),
) -> Result<String, LowerError> {
    let mut args = Vec::new();
    // a launch of several devices takes every device's buffers, device after device, named `b<i>_d<device>`
    let devices = k.launch.map_or(1, |l| l.devices);
    for d in 0..devices {
        let sfx = if devices > 1 {
            format!("_d{d}")
        } else {
            String::new()
        };
        for (i, p) in k.params.iter().enumerate() {
            match &p.kind {
                ParamKind::Buffer { elem, role, .. } => {
                    args.push(format!("{}* b{i}{sfx}", buffer_ptr_base(*elem, *role)?));
                    args.push(format!("const long long* b{i}{sfx}_shape"));
                    args.push(format!("const long long* b{i}{sfx}_stride"));
                    args.push(format!("long long b{i}{sfx}_base"));
                }
                ParamKind::Scalar { ty, .. } => {
                    args.push(format!("{} p{i}", ctype(*ty, "a scalar")?));
                }
                _ => return Err(refuse("a parameter kind this emitter has not been taught")),
            }
        }
    }
    if ids {
        for n in ["ti0", "ti1", "ti2", "bi0", "bi1", "bi2"] {
            args.push(format!("unsigned int {n}"));
        }
    }
    for (i, t) in results.iter().enumerate() {
        args.push(format!("{t}* {}", res_name(i)));
    }
    Ok(format!("int {name}(\n    {})", args.join(",\n    ")))
}

/// `x` as a C identifier fragment.
pub(crate) fn ident(x: &str) -> String {
    x.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

pub(crate) struct Emitter<'a> {
    pub(crate) k: &'a KernelFn,
    pub(crate) lo: &'a Lowering<'a>,
    pub(crate) out: String,
    pub(crate) depth: usize,
    /// Fresh-name counter for the offset and loop temporaries.
    pub(crate) fresh: u32,
    /// The launch, when the kernel has one: locals become per-thread arrays (`launch.rs`).
    pub(crate) launch: Option<LaunchInfo>,
    /// Thread-level C loops open inside the current thread run (a `Break` needs one).
    pub(crate) thread_loops: u32,
    /// The module a `Call` resolves in (`module.rs`); `None` for a lone kernel.
    pub(crate) module: Option<&'a ModuleCtx<'a>>,
    /// In a callee: the result locals, declared up front and stored through `res`, `res1`, ... on every exit.
    pub(crate) results: Vec<u32>,
}

/// The functions of the module being lowered and the C name of each.
pub(crate) struct ModuleCtx<'a> {
    pub(crate) fns: &'a [KernelFn],
    /// The module's extern declarations: an `ExternCall` is spelled as a call of its reference.
    pub(crate) externs: &'a [unpopped::kernel::ExternDecl],
    pub(crate) names: Vec<String>,
    /// The entry's launch: when present, a callee takes the thread's ids as six extra C arguments.
    pub(crate) launch: Option<LaunchInfo>,
}

/// The launch geometry an [`Emitter`] spells ids from.
#[derive(Clone, Copy)]
pub(crate) struct LaunchInfo {
    pub(crate) block: [u32; 3],
    pub(crate) grid: [u32; 3],
    /// The subgroup size a `Subgroup` collective groups by: the launch's required minimum (the oracle's default),
    /// or `1` when the launch states none.
    pub(crate) subgroup: u32,
    /// Devices the launch spans (`1`: one device). Every device runs the whole grid on its own buffers.
    pub(crate) devices: u32,
}

impl LaunchInfo {
    pub(crate) fn of(k: &KernelFn) -> Option<Self> {
        k.launch.as_ref().map(|l| Self {
            block: l.block,
            grid: l.grid,
            subgroup: l.subgroup.map_or(1, |s| s.min.max(1)),
            devices: l.devices,
        })
    }

    pub(crate) fn threads(self) -> u32 {
        self.block.iter().product()
    }
}
