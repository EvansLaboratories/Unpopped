kind: `unpopped-cpu-c` spells an atomic `Add`/`Exchange` on `F16`/`Bf16` (ROADMAP U26 third slice)
---
A launched `AtomicRmw` `Add` or `Exchange` on a narrow float is no longer a typed refusal in `unpopped-cpu-c`. `Add` decodes the stored pattern and the operand to `float`, adds, and encodes once, so every add rounds once to the element (ties to even) exactly as the interpreter does; `Exchange` moves the operand's bits, a NaN payload included. `Max`, `Min` and the bitwise combines on a narrow float stay refused (the checker admits none). The atomic and multicast-reduce operand now goes through `value_expr`, so a narrow operand is the pattern and not a widened `float`.

Verified: `kernel_narrow_float_atomic_agrees` compiles the C and compares a 4 x 3 launch with the interpreter for the tie case (`2048 + 1` leaves 2048), exact adds and exchanges carrying NaN payloads, on both types; a mutant that adds the raw patterns and a mutant that round-trips the exchange through the codec each fail it.

Open under U26: a narrow compute dtype and the MMA fragments. The packed `half2`/`bf16x2` atomic forms (U12b) are not spelled.
