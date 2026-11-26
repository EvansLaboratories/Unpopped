kind: Docs: tcgen05.cp placement, the two sources (design note 5l) and ROADMAP row U61
---
Note 5l of `docs/design-wgmma-descriptors.md`: PTX ISA 9.5 Figures 186-190 are tcgen05.ld/st fragments, not cp; PTX has no cp placement figure; CUTLASS `copy_traits_sm100.hpp` gives lane/bit mapping per shape, and the multicast pairing is an open disagreement between the two. Design only; row U61 stays blocked on the descriptor walk. Docs only, no gate.
