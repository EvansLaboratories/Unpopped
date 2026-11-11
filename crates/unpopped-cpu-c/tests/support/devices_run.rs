// SPDX-License-Identifier: MIT OR Apache-2.0
//! The multi-device side of the kernel-emitter differential (ROADMAP U12 slice 2): the serial C of a launch that spans
//! several devices, bound to one set of buffers per device, against `run_launch_devices`.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;

use unpopped::backend::Backend;
use unpopped::kernel::{BufferRole, KernelFn, ParamKind};
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch_devices;
use unpopped_cpu_c::CpuC;

use crate::cc;
use crate::kernel_run::{Ran, bind_buffer, same_bytes, width};

/// The C `main` that binds every device's buffers (`b<device * params + i>`), calls the kernel and prints the status
/// then the bytes of each writable buffer, device after device.
fn driver_devices(k: &KernelFn, name: &str, devices: &[Vec<KernelArg>]) -> String {
    let np = k.params.len();
    let mut s = String::from("#include <stdio.h>\n#include <string.h>\nint main(void) {\n");
    let (mut call, mut outs) = (Vec::new(), Vec::new());
    for (d, args) in devices.iter().enumerate() {
        for (i, (p, a)) in k.params.iter().zip(args).enumerate() {
            let KernelArg::Buffer(b) = a else {
                panic!("a device argument the driver cannot bind: {a:?}")
            };
            let id = d * np + i;
            s.push_str(&bind_buffer(id, b));
            call.push(format!("b{id}, s{id}, t{id}, {}LL", b.base_offset));
            if matches!(&p.kind, ParamKind::Buffer { role, .. } if *role != BufferRole::Read) {
                outs.push((id, b.bytes.len() / width(b.dtype) * width(b.dtype)));
            }
        }
    }
    let _ = writeln!(
        s,
        "  int st = {name}({});\n  printf(\"%d\\n\", st);",
        call.join(", ")
    );
    for (id, len) in outs {
        let _ = writeln!(
            s,
            "  for (int j = 0; j < {len}; ++j) printf(\"%02x\", ((unsigned char*)b{id})[j]);\n  printf(\"\\n\");"
        );
    }
    s.push_str("  return 0;\n}\n");
    s
}

/// Lower `k`, apply `mutate` to the C, compile and run it on `devices`; `None` without a host C compiler.
pub(crate) fn run_c_devices(
    tag: &str,
    k: &KernelFn,
    devices: &[Vec<KernelArg>],
    mutate: impl FnOnce(String) -> String,
) -> Option<Ran> {
    let g = CpuC
        .lower_kernel(k)
        .unwrap_or_else(|e| panic!("{tag}: lowering refused: {e}"));
    let Some(compiler) = cc::find_compiler() else {
        eprintln!("SKIP {tag}: no host C compiler");
        return None;
    };
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("kernel_emit");
    std::fs::create_dir_all(&dir).unwrap();
    let (c, exe) = (dir.join(format!("{tag}.c")), dir.join(format!("{tag}.exe")));
    let source = mutate(g.source.clone());
    std::fs::write(
        &c,
        format!("{source}\n{}", driver_devices(k, &g.name, devices)),
    )
    .unwrap();
    compiler
        .compile(&c, &exe, true)
        .unwrap_or_else(|e| panic!("{tag}: {e}\n{}", g.source));
    let out = Command::new(&exe).output().unwrap();
    assert!(out.status.success(), "{tag}: the program crashed");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let mut lines = text.lines();
    let status = lines.next().unwrap().trim().parse().unwrap();
    let outs = lines
        .map(|l| {
            (0..l.len() / 2)
                .map(|i| u8::from_str_radix(&l[2 * i..2 * i + 2], 16).unwrap())
                .collect()
        })
        .collect();
    Some(Ran { status, outs })
}

/// The bytes of every writable buffer of every device, device after device, and their dtypes.
pub(crate) fn written_devices(
    k: &KernelFn,
    devices: &[Vec<KernelArg>],
) -> (Vec<Vec<u8>>, Vec<unpopped_vocab::ElementKind>) {
    let (mut bytes, mut dtypes) = (Vec::new(), Vec::new());
    for args in devices {
        let (b, d) = crate::launch_run::written(k, args);
        bytes.extend(b);
        dtypes.extend(d);
    }
    (bytes, dtypes)
}

/// Run `k` as serial C and as the multi-device interpreter on `devices`; both must finish, with the same bytes in every
/// written buffer of every device.
pub(crate) fn agree_devices(tag: &str, k: &KernelFn, devices: Vec<Vec<KernelArg>>) {
    let Some(c) = run_c_devices(tag, k, &devices, |s| s) else {
        return;
    };
    let mut interp = devices;
    run_launch_devices(k, &mut interp).unwrap_or_else(|e| {
        panic!("{tag}: the interpreter trapped ({e:?}); pick a kernel it finishes")
    });
    assert_eq!(
        c.status, 0,
        "{tag}: the interpreter finished, the C reported {}",
        c.status
    );
    let (want, dtypes) = written_devices(k, &interp);
    same_bytes(tag, &c.outs, &want, &dtypes, 0);
}
