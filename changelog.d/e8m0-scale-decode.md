kind: the oracle decodes an `F8E8M0` MX block scale (CUDA audit row 60)
---
The oracle reads an `F8E8M0` buffer as the scale `2^(code - 127)` with `0xFF` as NaN (no zero, no infinity), through `cast_ref::decode_e8m0` and an arm in `raw_to_f64`; `types::elem_size` gives it its storage width, one byte. It is still a scale type, never a compute dtype: the plan gate refuses it as one. Sources: OCP Microscaling Formats (MX) Specification v1.0 (section number not re-read) and the ONNX float8 note's E8M0 table. Additive: no signature or IR change.
