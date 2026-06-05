//! `cp-sim` — a faithful, headless Rust port of the *Colonizing Pirkanmaa*
//! game simulation, built for fast neuroevolution training.
//!
//! ## Source of truth
//! This is a 1:1 port of the TypeScript game in the repo's `src/` directory
//! (which itself ports the original C++/Qt sources in `reference/`). When in
//! doubt about behaviour, the TS `src/` and the C++ `reference/` are
//! authoritative — see the repo root `CLAUDE.md`.
//!
//! Exception: the Mine / Hydroelectric / Nuclear economy values were
//! deliberately rebalanced in the TS port; for those, the TS `src/` is the only
//! source of truth (the C++ values are intentionally not matched).
//!
//! ## Determinism contract
//! `cp-sim` performs **no I/O, no rendering, no threading, and no
//! nondeterministic RNG**. The only randomness is the explicit MSVCRT-compatible
//! [`rng::Rng`], which is fully seeded and reproducible. This is what lets
//! training run identical games across machines and lets a Rust trace be checked
//! byte-for-byte against a TS golden trace.
//!
//! ## Conventions (followed by all later milestones)
//! - **Naming:** idiomatic Rust `snake_case`. Type and field names stay
//!   recognizably mapped to the TS. The TS uses trailing-underscore fields
//!   (`objectManager_`); in Rust we drop the underscore and use snake_case
//!   (`object_manager`). Where the mapping is non-obvious it is noted in a
//!   comment.
//! - **Numbers:** resource amounts are `i64` (see [`resources`] for the
//!   rationale — resources are integral in practice and exact integer math is
//!   required for TS parity). Grid coordinates are `i32`.
//! - **Error handling:** genuine recoverable conditions return `Result`.
//!   "Should never happen" invariants use `panic!`/`unwrap`/`debug_assert!` —
//!   in a deterministic sim a violated invariant is a bug to surface loudly, not
//!   to paper over.
//!
//! ## Milestone status
//! This is MILESTONE 1: workspace scaffold + lowest-level primitives only.
//! Later milestones add the model hierarchy, managers, and the world generator.

pub mod coordinate;
pub mod managers;
pub mod model;
pub mod resources;
pub mod rng;
pub mod world;

// Convenience re-exports for the cp-ai / parity layers.
pub use managers::{EndTurnOutcome, Game, GameSettings, SeatEvents, WinCause};
pub use model::{
    Building, BuildingType, ObjId, Player, PlayerId, Tile, TileId, TileType, Unit, UnitId, UnitType,
};
