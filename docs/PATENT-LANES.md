# Charter: the two Unpopped patent sub-lanes (`unpopped-patents-1`, `unpopped-patents-2`)

Written 2026-11-25 by the Unpopped lane on the PM's direction (relayed by pm-kernels; CireSnave 2026-11-25: *"If it would benefit from additional lanes to assist it with things like searching and reading patents, lets get those going."*). Docs only. Nothing here is legal advice, and no one in this process is an attorney.

## What the lanes are for

The union re-score (`docs/ir-union-rescore.md`, "Third measurement") found 38 rows that wait on a patent read: the capability sits in an area `docs/PATENT-SWEEP.md` records as NOT CLEARED, unread, or with a re-read trigger. Each row's survey file (`docs/ir-union-evidence/rescore-2026-11-25/<survey>.md`) names the PATENT-SWEEP section. The two lanes read the independent claims of the documents those rows depend on, chart each claim against Unpopped's constructs, and record the result, so the Unpopped lane can build the statement or know exactly what it must not build.

## Rules

1. **Model: Sonnet** (`--model sonnet`). Never Opus. Pass `model: sonnet` on any subagent.
2. **Verdicts, and only these three:** `outside as read` (the independent claims were read in full and each recites at least one limb Unpopped's construct lacks; quote the limb), `needs more` (a limb is unclear or the specification, a dependent claim or a family member matters; quote the limb and say what to read next), or `NOT CLEARED, unread` (the claims were not read). **Never "clear".** No lane writes a freedom-to-operate opinion.
3. **Method:** Google Patents with plain `curl` or `python C:\Projects\tools\patent-claims.py <number>` (public page, no key, full claims, machine-converted text, one request per number). Espacenet and PATENTSCOPE only through a person; if a document is only there, write the number and "needs a person" and move on. Read every independent claim in full, and the dependent claims where one could change the answer. Record the number as Google Patents spells it, the assignee, the status line and the priority date as Google shows them (Google's status and expiry are its own assumptions; say so).
4. **Chart:** for each independent claim, list its limbs one per line, and for each limb say whether Unpopped's construct (name the `Stmt` or type and the file) has it, lacks it, or does not apply. Quote the claim language; never paraphrase a limb.
5. **Where the result goes:** `docs/PATENT-SWEEP.md`, as notes under the section the row names (add a dated subsection; do not edit a verdict already there, add a line saying what changed). Work on a lane branch off `origin/main` in your own worktree, commit freely, and open **one PR per day** at most (CireSnave's rule: one PR per project per day, more only on a release from pm-assist). The PR is docs only. Sign every comment and PR body `— lane: unpopped-patents-1` (or `-2`).
6. **Report:** to `unpopped` (the primary lane), one line per patent: `<number> | verdict | rows it gates | the limb that decides it`. Findings, never conclusions about the product. If a document reads as covering an Unpopped statement, report it at once as `[FINDING]` with the limb quoted: that is a hold on the statement, not a decision.
7. **Authority:** you are a Sonnet research lane. You do not change code, ROADMAP or any verdict beyond the three above, you do not file anything anywhere (no advisory, no issue upstream, no contact with a patent holder), and you do not run `gh auth switch` or act as `ciresnave`. Every absence needs a positive control: "0 hits for X; the same query finds Y, which is present."
8. **Load:** reading is light, but never run a cargo build; the Unpopped lane's gates use the machine's one cargo slot.

## Work order (by rows gated)

**`unpopped-patents-1`:**
1. US11803380B2 (NVIDIA, asynchronous barriers; family US20210124627A1). Gates 7 rows: CUTLASS 62, 67; CuTe 34, 45, 52; Mojo 33; Triton 57. The PM's review of the `mbarrier` and `CopyBulk` claims is the open item; compare with `Mbar*` and `CopyBulk` in `crates/unpopped/src/kernel.rs` and PATENT-SWEEP's `mbarrier` section.
2. US12699602 (named barriers, cluster arrive). Claims read 2026-11-25 (was "claims unread"); narrowed 2026-11-26 by the PM, basis the `docs/PATENT-SWEEP.md` subsection "US12699602B1: file-wrapper chart of the warp-role statements, 2026-11-26": warp roles within ONE thread block on one multiprocessor (`Named`, `NamedArrive`, `Mbar*` inside a block, no cross-SM producer/consumer) are outside as read, conditional on core = streaming multiprocessor (the spec's wider wording for "core" is a claim-construction matter for a person); the hold STAYS, needs more, for cluster-scope and device-scope warp roles with a producer warp and a consumer warp on separate multiprocessors; claims 17 and 16 (memory release) outside as read; the examiner's 2025-12-05 action (scanned PDF) and SRNT NOT CLEARED, unread. Gates CuTe 49 and Triton 58; compare with `BarrierKind::Named` and `NamedArrive`.
3. The U9c and U15 unread claims: US11262989B2, US9052888B2, WO2013070616A1 (vector load/store and predication), then US20230289242A1 and US20210124627A1 as the sweep's `mbarrier` section lists them. Compare with `Place::Raw`, `MaskedLoad`/`MaskedStore` and the packed values (U15).
4. Then the rows that name no number (a search first, then a read): CuTe 44, 46, 64; Mojo 6, 10, 27, 28, 30, 34, 35, 55; Slang 6, 35; Triton 8. Record the queries, the source and the date for each.

**`unpopped-patents-2`:**
1. US12645455 / US12645455B2, US12657024, US12669996 (tensor-map and prefetch family). Granted claims, specification not read. Gates Mojo 33, ThunderKittens 24, CuTe 43; compare with `CopyBulk` and the planned descriptor operand kind (ROADMAP U8c).
2. US12204897B2 and US12710961B2 (`wgmma`) and US20230289398A1 (`tcgen05`): already read as "outside as read" for the `Wgmma` and `Tcgen05Mma` statements. Re-read only the descriptor, swizzle, TMEM allocation and CTA-pair limbs (the sweep's re-read triggers) against the planned slices. Gates CUDA 40, CUTLASS 67, 70, CuTe 33, Triton 19.
3. The remaining named documents: US12020035B2, US20230315655A1, US12499052 (continuation), US10338919B2, WO2018217360A1, EP2659362A2 (CUDA 40, 7, 76).
4. Then the CUTLASS rows that name no number: CUTLASS 49, 59, 63, 64 (a search first, then a read).

## What the primary lane provides

The row-to-section map is in each survey file's not-yet-Yes table (the Blocker column names the PATENT-SWEEP section). The Unpopped constructs are in `crates/unpopped/src/kernel.rs` (statements), `mma.rs` (atoms) and `docs/PATENT-SWEEP.md` itself (what was already read and by whom). Ask `unpopped` for anything else; do not guess a construct's behaviour from its name.
