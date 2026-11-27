kind: `VoteKind::Uniform` (PTX `vote.sync.uni`)
---
Added `VoteKind::Uniform`: `1` when every lane's predicate agrees, else `0` (PTX ISA `vote.sync.uni.pred`). The oracle and the cpu-c emitter spell it explicitly (both had a `_ =>` arm that would have read a new variant as `Ballot`); pinned by `votes_are_predicates_over_the_scope` and the cpu-c differential.
