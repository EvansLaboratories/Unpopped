// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The Slang entry model in the walker** (ROADMAP U19 census slice 4, `docs/options-u19-slang-entry-model.md`). A Slang compute
//! kernel has file-scope resources and semantic entry parameters; the language's crate reads them (fishstringer's
//! `fishstringer-slang-parse` is the real reader) and hands the walker an `EntryModel`. These tests use a **test-only reader**
//! over the source text, parsed by `tree-sitter-slang`, to prove the walker's half: resources become parameters in slot order,
//! `numthreads` wins over the context's block, the semantics are computed from the execution ids, and the declines name what is wrong.

use unpopped::exec::LaunchShape;
use unpopped::kernel::{KernelFn, ParamKind};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{
    EntryModel, EntryModelError, Frontend, KernelLiftError, LiftContext, Resource, ResourceKind,
    Semantic, lift_kernel, lift_kernel_in,
};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_slang::LANGUAGE_SLANG.into())
        .ok()?;
    p.parse(src, None)
}

/// The test reader: resources from `[[vk::binding(N, 0)]] KIND<T> name;` lines, the block from `numthreads(x, y, z)`, semantics from
/// `name : SV_*` in the entry's signature. (fishstringer's reader works over the CST; this one only has to be right for these sources.)
fn read(_tree: &tree_sitter::Tree, src: &str, _entry: &str) -> Result<EntryModel, EntryModelError> {
    let mut resources = Vec::new();
    for line in src.lines() {
        let Some(rest) = line.trim().strip_prefix("[[vk::binding(") else {
            continue;
        };
        let (slot, rest) = rest
            .split_once(',')
            .ok_or_else(|| EntryModelError::new("a binding"))?;
        let decl = rest
            .split_once("]]")
            .ok_or_else(|| EntryModelError::new("a binding"))?
            .1
            .trim();
        let (kind, after) = decl
            .split_once('<')
            .ok_or_else(|| EntryModelError::new("a resource type"))?;
        let (elem, name) = after
            .split_once('>')
            .ok_or_else(|| EntryModelError::new("a resource type"))?;
        let kind = match kind.trim() {
            "StructuredBuffer" => ResourceKind::StructuredRO,
            "RWStructuredBuffer" => ResourceKind::StructuredRW,
            "ConstantBuffer" => ResourceKind::ConstantBuffer,
            other => return Err(EntryModelError::new(format!("the resource type `{other}`"))),
        };
        let slot: u32 = slot
            .trim()
            .parse()
            .map_err(|_| EntryModelError::new("a slot"))?;
        resources.push(Resource::new(
            name.trim().trim_end_matches(';'),
            kind,
            elem.trim(),
            slot,
        ));
    }
    let block = src.split_once("numthreads(").map(|(_, r)| {
        let nums: Vec<u32> = r
            .split(')')
            .next()
            .unwrap_or("")
            .split(',')
            .filter_map(|n| n.trim().parse().ok())
            .collect();
        [nums[0], nums[1], nums[2]]
    });
    let mut semantics = Vec::new();
    for part in src.split("SV_").skip(1) {
        let sem = part
            .split(|c: char| !c.is_alphanumeric())
            .next()
            .unwrap_or("");
        let before = &src[..src.find(&format!("SV_{sem}")).unwrap_or(0)];
        let name = before
            .trim_end()
            .trim_end_matches(':')
            .trim_end()
            .rsplit(|c: char| !c.is_alphanumeric() && c != '_')
            .next()
            .unwrap_or("");
        let s = match sem {
            "DispatchThreadID" => Semantic::DispatchThreadId,
            "GroupID" => Semantic::GroupId,
            "GroupThreadID" => Semantic::GroupThreadId,
            "GroupIndex" => Semantic::GroupIndex,
            other => return Err(EntryModelError::new(format!("the semantic `SV_{other}`"))),
        };
        semantics.push((name.to_string(), s));
    }
    Ok(EntryModel::new(block, semantics, resources))
}

const SLANG: Frontend =
    Frontend::new("slang", "numthreads", &[], "out", "in", parse).with_entry_model(read);

fn lift(src: &str, grid: [u32; 3], block: [u32; 3]) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_in(
        &SLANG,
        src,
        "main",
        &LiftContext::new(LaunchShape::new(block, grid)),
    )
}

fn f32s(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::from_f32(&[v.len() as i64], v))
}

fn out_of(args: &[KernelArg], i: usize) -> Vec<f64> {
    let KernelArg::Buffer(b) = &args[i] else {
        panic!()
    };
    b.to_f64_vec()
}

fn not_yet(r: Result<KernelFn, KernelLiftError>, needle: &str) {
    match r {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
        }
        other => panic!("expected NotYetLifted({needle}), got {other:?}"),
    }
}

/// The shape of fishstringer's corpus: two buffers, a constant buffer, a `uint3 gid : SV_DispatchThreadID`.
const SCALE: &str = "[[vk::binding(0, 0)]] StructuredBuffer<float> input;\n\
    [[vk::binding(1, 0)]] RWStructuredBuffer<float> output;\n\
    struct Params { uint out_size; float scale; };\n\
    [[vk::binding(2, 0)]] ConstantBuffer<Params> p;\n\
    [shader(\"compute\")] [numthreads(4, 1, 1)]\n\
    void main(uint3 gid : SV_DispatchThreadID) { uint idx = gid.x; if (idx >= p.out_size) return; output[idx] = input[idx] * p.scale; }";

#[test]
fn a_slang_entry_lifts_with_its_resources_as_parameters_and_runs_on_the_oracle() {
    // the context's block (32) is not the entry's `numthreads(4, 1, 1)`: the attribute wins; the caller's grid is two blocks
    let k = lift(SCALE, [2, 1, 1], [32, 1, 1]).unwrap();
    assert_eq!(k.launch.map(|l| l.block), Some([4, 1, 1]));
    let names: Vec<_> = k.params.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["input", "output", "p"], "slot order");
    assert!(matches!(k.params[2].kind, ParamKind::Struct { .. }));
    let input: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let mut args = vec![
        f32s(&input),
        f32s(&[0.0; 8]),
        KernelArg::Struct(vec![KernelArg::Int(6), KernelArg::Float(2.0)]),
    ];
    run_launch(&k, &mut args).unwrap();
    // eight threads, `out_size` 6: the last two return before writing
    assert_eq!(
        out_of(&args, 1),
        vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 0.0, 0.0]
    );
}

#[test]
fn group_and_group_thread_ids_and_the_group_index_are_the_blocks_own_ids() {
    let src = "[[vk::binding(0, 0)]] RWStructuredBuffer<float> output;\n\
        [numthreads(4, 2, 1)]\n\
        void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID, uint li : SV_GroupIndex) {\n\
          output[li + gid.x * 8u] = (float)(gid.x * 100u + lid.y * 10u + lid.x); }";
    let k = lift(src, [2, 1, 1], [1, 1, 1]).unwrap();
    let mut args = vec![f32s(&[0.0; 16])];
    run_launch(&k, &mut args).unwrap();
    let want: Vec<f64> = (0..2)
        .flat_map(|g| {
            (0..2).flat_map(move |y| (0..4).map(move |x| f64::from(g * 100 + y * 10 + x)))
        })
        .collect();
    assert_eq!(
        out_of(&args, 0),
        want,
        "the linear index is x + y * 4 within each group"
    );
}

#[test]
fn a_dispatch_thread_id_is_block_times_block_dim_plus_thread_in_every_axis() {
    let src = "[[vk::binding(0, 0)]] RWStructuredBuffer<float> output;\n\
        [numthreads(2, 2, 1)]\n\
        void main(uint3 t : SV_DispatchThreadID) { output[t.y * 4u + t.x] = (float)(t.x + 10u * t.y); }";
    let k = lift(src, [2, 2, 1], [1, 1, 1]).unwrap();
    let mut args = vec![f32s(&[0.0; 16])];
    run_launch(&k, &mut args).unwrap();
    let want: Vec<f64> = (0..4)
        .flat_map(|y| (0..4).map(move |x| f64::from(x + 10 * y)))
        .collect();
    assert_eq!(out_of(&args, 0), want);
}

#[test]
fn the_declines_name_the_slots_the_parameter_and_the_missing_launch() {
    // a gap in the slots
    let gap = SCALE.replace("binding(2, 0)", "binding(3, 0)");
    not_yet(lift(&gap, [1, 1, 1], [1, 1, 1]), "binding slots [0, 1, 3]");
    // a repeated slot
    let dup = SCALE.replace("binding(1, 0)", "binding(0, 0)");
    not_yet(lift(&dup, [1, 1, 1], [1, 1, 1]), "binding slots [0, 0, 2]");
    // a non-zero start
    let start = "[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;\n[numthreads(1, 1, 1)]\nvoid main(uint3 g : SV_DispatchThreadID) { output[g.x] = 1.0f; }";
    not_yet(lift(start, [1, 1, 1], [1, 1, 1]), "binding slots [1]");
    // an entry parameter with no semantic
    let bare = "[[vk::binding(0, 0)]] RWStructuredBuffer<float> output;\n[numthreads(1, 1, 1)]\nvoid main(uint3 g : SV_DispatchThreadID, uint3 extra) { output[g.x] = 1.0f; }";
    not_yet(lift(bare, [1, 1, 1], [1, 1, 1]), "`extra`, which carries no execution-id semantic");
    // a constant buffer of a struct that is not defined
    let nostruct = SCALE.replace("struct Params { uint out_size; float scale; };\n", "");
    not_yet(
        lift(&nostruct, [1, 1, 1], [1, 1, 1]),
        "unknown struct `Params`",
    );
    // the block is declared but the caller gave no launch: the grid is the caller's
    not_yet(lift_kernel(&SLANG, SCALE, "main"), "gave no launch");
}

#[test]
fn the_parameter_order_is_the_slot_order_not_the_declaration_order() {
    // declared output first (slot 1), then input (slot 0): the kernel's parameters are input, output
    let src = "[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;
        [[vk::binding(0, 0)]] StructuredBuffer<float> input;
        [numthreads(2, 1, 1)]
        void main(uint3 g : SV_DispatchThreadID) { output[g.x] = input[g.x] + 1.0f; }";
    let k = lift(src, [1, 1, 1], [1, 1, 1]).unwrap();
    let names: Vec<_> = k.params.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["input", "output"]);
    let mut args = vec![f32s(&[10.0, 20.0]), f32s(&[0.0, 0.0])];
    run_launch(&k, &mut args).unwrap();
    assert_eq!(out_of(&args, 1), vec![11.0, 21.0]);
}
