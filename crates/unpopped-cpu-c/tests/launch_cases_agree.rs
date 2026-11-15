// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`unpopped::launch_cases`, the shared cross-emitter corpus, is run by the reference emitter** (ROADMAP U20 part a):
//! every case is lowered by `CpuC::lower_kernel`, compiled with a host C compiler, run, and its written buffers compared
//! byte for byte with the interpreter. Also pinned: each case's `KernelInterface` carries the core-derived half, which is
//! what an external emitter must report too.

use unpopped::backend::Backend;
use unpopped::kernel_interface::KernelInterface;
use unpopped::launch_cases::launch_cases;
use unpopped_cpu_c::CpuC;

#[path = "support/cc.rs"]
mod cc;
// The shared compile-and-run driver; this file uses the part that runs a launch.
#[allow(dead_code)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[allow(dead_code)]
#[path = "support/launch_run.rs"]
mod launch_run;

#[test]
fn every_launch_case_agrees_with_the_interpreter() {
    for case in launch_cases() {
        launch_run::agree_launch_within(&case.name, &case.kernel, case.args.clone(), case.ulps);
    }
}

#[test]
fn every_launch_case_reports_the_core_interface() {
    for case in launch_cases() {
        let g = CpuC.lower_kernel(&case.kernel).expect("the case lowers");
        let core = KernelInterface::of(&case.kernel);
        let got = g.interface().expect("an interface is reported");
        assert_eq!(got.launch, core.launch, "{}", case.name);
        assert_eq!(got.params, core.params, "{}", case.name);
    }
}
