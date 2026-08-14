//! Cross-emitter conformance evidence — no library surface, only tests.
//!
//! # Why this crate exists
//!
//! Unpopped is a standard with a normative reference emitter per target. Most
//! tests belong to one side of that: a plan-gate test belongs in `unpopped`, an
//! emitted-C test belongs in `unpopped-cpu-c`. A few belong to **neither**,
//! because their subject is the relationship *between* implementations —
//! "how much of the §6.1 dtype vocabulary does each reference emitter lower"
//! is a fact about the standard's coverage, not about C or about Slang.
//!
//! Those tests need to see every emitter at once. Putting them in one emitter's
//! crate would make that emitter dev-depend on its siblings, which says something
//! false about the dependency graph: `unpopped-cpu-c` has no business knowing
//! `unpopped-slang` exists.
//!
//! So they live here, in a crate that depends on all of them and is depended on
//! by none.
