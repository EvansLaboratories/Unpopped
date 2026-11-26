kind: Docs: the warp-role patent hold is narrowed to cross-multiprocessor roles (US12699602B1)
---
The stale "claims unread" statements for US12699602 (`docs/ir-union-rulings.md`, `docs/PATENT-LANES.md`, CuTe 49 and Triton 58 in the 2026-11-25 rescore) are replaced by the PM narrowing of 2026-11-26: warp roles within one thread block on one multiprocessor are outside as read, conditional on core = streaming multiprocessor; the hold stays for cluster- and device-scope roles on separate multiprocessors. Docs only, no gate.
