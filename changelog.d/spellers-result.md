kind: changed
---
`cfamily::unary_f32` and `cfamily::unary_f64` return `Result<String, SpellError>` instead of panicking on `Popc`/`Clz`/`Brev` (BREAKING; B1 batch). `SpellError` is `non_exhaustive`. cpu-c turns the error into a named decline.
