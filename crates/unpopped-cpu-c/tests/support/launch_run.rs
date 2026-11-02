// SPDX-License-Identifier: MIT OR Apache-2.0
//! The launched side of the kernel-emitter differential: the serialized thread-loop C against
//! `oracle::kernel_launch::run_launch` (`docs/ir-kernel-layer.md`, "Emitters").

use unpopped::kernel::{BufferRole, KernelFn, ParamKind};
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

use crate::kernel_run::{run_c, same_bytes};

/// The written buffers of `args` after a run, with their dtypes.
pub(crate) fn written(k: &KernelFn, args: &[KernelArg]) -> (Vec<Vec<u8>>, Vec<ElementKind>) {
    k.params
        .iter()
        .zip(args)
        .filter_map(|(p, a)| match (&p.kind, a) {
            (ParamKind::Buffer { role, .. }, KernelArg::Buffer(b)) if *role != BufferRole::Read => {
                Some((b.bytes.clone(), b.dtype))
            }
            _ => None,
        })
        .unzip()
}

/// Run the launched `k` as serialized C and as the launch interpreter on `args`; both must finish,
/// with the same bytes in every written buffer. (A launch trap is not compared: the C reference
/// only claims the kernels the interpreter finishes.)
pub(crate) fn agree_launch(tag: &str, k: &KernelFn, args: Vec<KernelArg>) {
    let Some(c) = run_c(tag, k, &args) else {
        return;
    };
    let mut interp = args;
    match run_launch(k, &mut interp) {
        Ok(()) => {
            assert_eq!(
                c.status, 0,
                "{tag}: the interpreter finished, the C reported {}",
                c.status
            );
            let (want, dtypes) = written(k, &interp);
            same_bytes(tag, &c.outs, &want, &dtypes, 0);
        }
        Err(KernelTrap::OutOfBounds { .. }) => {
            assert_eq!(c.status, 1, "{tag}: interpreter OutOfBounds");
        }
        Err(e) => panic!("{tag}: the interpreter trapped ({e:?}); pick a kernel it finishes"),
    }
}
