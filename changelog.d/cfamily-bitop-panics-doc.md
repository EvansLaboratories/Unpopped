kind: documented panics of the c-family unary spellers on integer bit operations
---
`cfamily::unary_f32` and `unary_f64` now document, under `# Panics`, that `UnaryOp::Popc`, `Clz` and `Brev` panic (message "c-family lowering: {op:?} is an integer bit operation, refused by the typing gate"), and that type checking refuses them first (`TypeError::IntegerOnlyOperand`). Tests pin the message. No signature or behaviour change.
