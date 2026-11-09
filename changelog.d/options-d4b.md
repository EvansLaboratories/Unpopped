kind: docs-only, options note for a tuple state in a combining collective (R5.7 D4 slice b)
---
Adds `docs/options-d4b-tuple-state-combine.md`: why the tuple state cannot be a packed local today (the kernel evaluator refuses a packed value; a callee has one result), three representations, and a recommendation (a struct of scalars: multi-result callees and a `CollectiveTuple` statement). No code.
