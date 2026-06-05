# rust-trainer

A fast, headless Rust reimplementation of the *Colonizing Pirkanmaa* game,
built to run neuroevolution training far faster than the TypeScript version
can. It is a **faithful port** of the existing game.

## Source of truth

- The TypeScript game in the repo's `src/` is the primary source of truth.
- The original C++/Qt sources in `reference/` are the deeper source of truth
  for game logic (the TS itself ports them) — consult them when the TS is
  unclear.
- **Exception:** the Mine / Hydroelectric / Nuclear economy values were
  *deliberately* rebalanced in the TS port (see the repo root `CLAUDE.md`). For
  those three buildings the TS `src/` is the **only** source of truth; do not
  "fix" them back toward the C++ values.

When porting behaviour, read the corresponding TS file before writing Rust.

## Crate layout

```
rust-trainer/
├── Cargo.toml                  # workspace
└── crates/
    ├── cp-sim/                 # the game simulation (pure logic, deterministic)
    │   └── src/
    │       ├── lib.rs
    │       ├── rng.rs          # MSVCRT rand()/srand() LCG replica (FIDELITY-CRITICAL)
    │       ├── resources.rs    # BasicResource, ResourceMap, all economy constants
    │       └── coordinate.rs   # Coordinate + Direction
    ├── cp-ai/                  # neuroevolution AI (depends on cp-sim) — STUB
    │   └── src/lib.rs
    └── cp-train/               # GA training harness + parity tooling (deps: cp-sim, cp-ai)
        └── src/
            ├── lib.rs          # STUB
            └── bin/
                ├── train.rs    # STUB: training loop
                └── parity.rs   # STUB: TS-golden-trace parity check
```

## Build & test

```bash
cd rust-trainer
cargo build            # whole workspace
cargo test             # unit tests (rng/resources/coordinate)
cargo build --release  # optimized build for training
```

## Porting conventions (followed by all milestones)

- **Naming.** Idiomatic Rust `snake_case`. Type and field names stay
  recognizably mapped to the TS. The TS uses trailing-underscore fields
  (`objectManager_`); in Rust drop the underscore and use snake_case
  (`object_manager`). Note the mapping in a comment where it isn't obvious.
- **Numeric representation.**
  - **Resources are `i64`.** The TS uses plain JS numbers, but inspection of the
    model/managers shows resource amounts are only ever integer constants times
    integer worker counts, combined with integer add/subtract. The only
    divisions in the codebase are AI heuristics and UI percentage bars — never
    stored resource state. So resources are integral in practice, and `i64` is
    exact (no float drift), which matters for deterministic parity with the TS
    golden traces.
  - **Coordinates are `i32`** (small grid positions that may go transiently
    negative before clamping).
  - **RNG state is `u32`** with wrapping multiply/add — see below.
- **`ResourceMap` semantics.** Mirrors the TS `Map<BasicResource, number>`
  (itself mirroring C++ `std::map`). Backed by an **insertion-ordered**
  `Vec<(BasicResource, i64)>` (not a `HashMap`) because several algorithms rely
  on iterating only present keys in insertion order, matching JS `Map`.
- **RNG (fidelity-critical).** `rng.rs` replicates the MSVCRT LCG bit-for-bit:
  `state = state * 214013 + 2531011 (mod 2^32)`, return `(state >> 16) & 32767`,
  `RAND_MAX = 32767`. Modeled in `u32` with `wrapping_mul`/`wrapping_add` (this
  is bit-identical to the TS `Math.imul(...) >>> 0`). Unlike the TS global, the
  Rust `Rng` is an explicit value type so the sim has no hidden global RNG
  state. Do **not** reorder, add, or remove `rand()` calls in the world
  generator (later milestone) — the original map seeds depend on the exact call
  order. Test expected sequences were captured by running the actual TS under
  `vite-node`.
- **Error handling.** Recoverable conditions return `Result`. "Should never
  happen" invariants `panic!`/`unwrap`/`debug_assert!` — in a deterministic sim a
  violated invariant is a bug to surface loudly.
- **Purity.** `cp-sim` does **no** I/O, rendering, threading, or
  nondeterministic RNG. The only randomness is the seeded `Rng`.

## What was ported in Milestone 1, and what was skipped

Ported from `src/core/`:

- `rng.ts`        → `cp-sim/src/rng.rs`
- `resources.ts`  → `cp-sim/src/resources.rs`
- `coordinate.ts` → `cp-sim/src/coordinate.rs`

Skipped (intentionally — these are rendering/UI only, not game logic):

- `src/core/images.ts` — image/animation/sprite lookup tables (rendering only).
- `src/core/descriptions.ts` — human-readable tooltip/help strings (UI only).
  Note: `descriptions.ts` *does* reference economy constants
  (`FARM_GROW_TIME`, etc.), but only to interpolate them into help text — no
  logic. The constants themselves live in `resources.rs`.

Not yet ported (later milestones): the model hierarchy
(`BaseObject → GameObject → PlaceableGameObject`, tiles, buildings, units,
players), the managers (`ObjectManager`, `PlayerManager`, `GameSettingsManager`,
`GameEventHandler`), and the `WorldGenerator`.
