# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A 1:1 TypeScript + Phaser 3 browser port of *Colonizing Pirkanmaa*, a turn-based
strategy game originally written in C++/Qt for the TUNI *Programming 3* course.
The original C++/Qt sources are kept in `reference/` **for comparison only** — they
are not built, but they are the source of truth for game logic. When implementing
or fixing behaviour, check the corresponding C++ file before changing TS logic.

## Commands

```bash
npm run dev          # Vite dev server (127.0.0.1:5173)
npm test             # Vitest, single run
npm run test:watch   # Vitest watch mode
npm run build        # tsc --noEmit type-check THEN vite build to dist/
npm run preview      # serve the production build
```

Run a single test: `npx vitest run tests/economy.test.ts` or filter by name with
`npx vitest run -t "starting resources"`. Node version is pinned in `.nvmrc` (22).
There is no separate lint step — `npm run build` runs `tsc --noEmit` as the gate.

## Fidelity constraints (do not break these)

These are the reasons the port exists; changing them silently breaks the game's
correctness contract, which the tests lock down:

- **`src/core/rng.ts`** replicates the MSVCRT `rand()`/`srand()` LCG bit-for-bit
  (RAND_MAX 32767). A seed must reproduce the exact map the original Windows binary
  produced. `WorldGenerator` depends on the **precise order** of `rand()` calls —
  do not reorder, add, or remove RNG calls in `src/world/worldgenerator.ts`.
- Economy values (salaries, growth timers, starting resources 400/200/100/25, and the
  Farm/Village/Outpost costs) mirror the original `resourcemaps.h`. `tests/economy.test.ts`
  locks the starting resources, RNG and ResourceMap helpers.
- **Deliberate balance divergence (industry):** the Mine / Hydroelectric / Nuclear values
  in `src/core/resources.ts` were intentionally rebalanced away from the C++ original so
  the industrial tier is a real choice rather than always-worse-than-farms. Nuclear is the
  late-game engine (cost shifted into money: 2000; output 160/worker ≈ 2.4× a farm per
  unit-slot), Hydro rewards a river (cheaper; 80/worker), Mine is a cheaper material engine
  (build cost lowered, production unchanged to keep the military/metal balance). For these
  three buildings, `reference/` is **no longer** the source of truth. The AI income models
  (`ai.ts` `netMoneyPerRound`, nn `metrics.ts`), the plant build-gates, and the hardcoded
  metal gates in the candidate enumerators (`candidates.rs` / `src/ai/nn/candidates.ts`:
  the Outpost metal-income gate and the soldier metal-cost gates) mirror these numbers —
  keep them in sync if you retune.
- **Deliberate balance divergence (Outpost rebalance, 2026-06-05 — arc bump `sd` → `sd2`):**
  the Outpost cost in `src/core/resources.ts` (`OUTPOST_BUILD_COST`) / `resources.rs`
  (`outpost_build_cost`) was rebalanced **650 money / 300 wood / 300 stone / 300 metal →
  500 / 200 / 200 / 100**. The original 300-metal cost made the Outpost (and thus the
  soldier-cap → army chain; Outpost = +3 soldiers) effectively UNREACHABLE on a normal
  ~1-mine economy (300 metal ≈ 15 mine-rounds of pure hoarding), so the AI never fielded an
  army — the same "always-worse, never-chosen" trap the industry tier above was rebalanced
  out of. For the Outpost, `reference/` is **no longer** the source of truth. Parity-locked
  pair: the NN outpost tile-gate was also lowered **12 → 8** (`candidates.rs` /
  `src/ai/nn/candidates.ts`) to match the HARD bot (`hard_ai.rs`, already 8). The soldier-cap
  formula (Outpost +3) and map-gen RNG are unchanged. This is parity-affecting: any retune
  must edit BOTH the Rust and TS mirror, re-export goldens, keep parity 8/8, and bump the arc.
- **Deliberate balance divergence (military-economy rebalance, 2026-06-06 — arc bump
  `sd2` → `sd3`):** two metal-only knobs were cut so the soldier-cap → army chain is
  actually fundable on a normal ~1-mine economy (the Outpost-cost rebalance above made the
  cap REACHABLE, but per-round metal drain still strangled the army). (1) **Outpost per-round
  METAL upkeep −15 → −5** (`OUTPOST_PRODUCTION` / `outpost_production`): at −15 a single mine
  (≈20 metal/round) could barely carry ONE Outpost; at −5 it comfortably carries 2-3 (cap
  7-10 soldiers). (2) **Soldier METAL build cost −50 → −30** (`SOLDIER_COST` / `soldier_cost`):
  a 4-6 soldier army now costs 120-180 metal instead of 200-300. The money upkeep (−50), money
  build cost (−200), salary (−30), mine output, Outpost build cost, and the Device soldier-cap
  halving are all left UNCHANGED so a pure-economy line stays competitive; for these two knobs
  `reference/` is **no longer** the source of truth. Parity-locked mirrors that MUST move with
  the knobs (else they re-create the unreachability bug): the Outpost metal-income gate
  `(outposts + 1) * 15 → * 5` and the soldier metal-cost gates `/ 50 → / 30`, `>= 50 → >= 30`,
  `< 50 → < 30` in BOTH `candidates.rs` and `src/ai/nn/candidates.ts`, plus the soldier-hire
  metal gates in `hard_ai.rs`. The NN feature scale `(metal − 50) / 500` is DELIBERATELY left
  at 50 (a normalization offset, not a rule — moving it would shift the feature distribution
  mid-arc). This is parity-affecting: any retune must edit BOTH the Rust and TS mirror plus
  all metal gates, re-export goldens, keep parity 8/8, and bump the arc. NOTE: the scripted
  league (`hard_ai.rs` `AiParams` presets — reserve / max_outposts / etc.) must be RE-TUNED
  against this new economy in a separate later phase.

## Architecture

The layering mirrors the original C++ package structure and is the key thing to
understand. **Pure-logic layers have zero Phaser/DOM dependency** — this is what
makes them unit-testable headlessly:

- `core/` — resources (`ResourceMap` = `Map<resource, number>` with merge/reverse/
  negatives/positives helpers), `Coordinate`, image/animation lookup tables, RNG,
  descriptions.
- `model/` — the object hierarchy `BaseObject → GameObject → PlaceableGameObject`,
  then `TileBase`+tiles, `BuildingBase`+buildings, `UnitBase`+units, `PlayerBase`.
  Ported field names keep a **trailing underscore** (`this.objectManager_`) from
  the C++; follow that convention in these files.
- `managers/` (the "DAL" layer) — `ObjectManager`, `PlayerManager`,
  `GameSettingsManager`, `GameEventHandler`. `GameEventHandler` owns turn flow:
  `tileClicked`, `endTurn` (resource production, HQ-connectivity cutting, win/lose
  checks), conquest, player neutralisation.
- `world/` — `WorldGenerator` (seeded, MSVCRT-compatible).
- `scenes/` — Phaser `BootScene` (asset preload, emits `boot-complete`) and
  `GameScene` (rendering + input). The **only** layers allowed to touch Phaser.
- `ui/` — DOM-based menu panel, start dialog, help window (not Phaser).

### Dependency inversion (the important seam)

`model/` and `managers/` never import Phaser or `GameScene` directly. Instead
`src/model/base.ts` declares interfaces — `IGameScene`, `IGameEventHandler`,
`IObjectManager`, `IGameSettingsManager`, `ISceneObjectHandle` — and the concrete
renderer is injected at runtime. `main.ts` wires it all together: it constructs the
managers, calls `objectManager.addDALS(...)`, then `game.scene.start('GameScene', …)`
and on the scene's `onReady` callback injects the scene back into the managers and
runs `WorldGenerator.generateMap`.

Because of this seam, `tests/gameplay.test.ts` drives a **full game headlessly** by
implementing `IGameScene`/menu interfaces as no-op stubs — no browser needed. When
adding logic that needs to draw or read scene state, add a method to the relevant
interface in `base.ts` rather than importing Phaser into a logic file.

### Dev debug handle

In `import.meta.env.DEV`, `main.ts` exposes `window.__cp = { game, objectManager,
playerManager, eventHandler, gsm }` for headless/browser verification. It is not
present in production builds.

## Model management

Every trained model (AlphaZero net, heuristic hard-bot `AiParams` set,
neuroevolution genome) lives under **`models/`** — the single source of truth. Do
NOT scatter `champion.json`s around `rust-trainer/` or elsewhere; register them.

- Layout: `models/<arc>/<type>/<id>/{weights.json, manifest.json, bench.json?}`,
  indexed by `models/registry.jsonl`, with champion/deploy pointers in
  `models/CHAMPION.json`. Full spec in `models/README.md`.
- **id = `<arc>-<type>-<NNN>`** (e.g. `sd-az-001`). `arc` = game-version code —
  **bump it on any game-rules change** so models from different game versions are
  never compared as equivalent (current arc `sd` = the Strange-Device version).
  `type` ∈ `az` / `hardbot` / `ga`. `NNN` = per-(arc,type) incremental.
- Manage with `npm run models -- <list|show|register|promote>`. `register
  <weights.json> --arc <a> --type <t>` imports a model, assigns the id, stamps the
  git commit, and writes the manifest + registry line.
- Versioning is by id + `git_commit`, never by renaming; champion/deployed are
  pointers, so lineage stays stable. Benchmarks recorded per the metric taxonomy in
  `STRANGE-DEVICE-DESIGN.md` §10.
- Pre-Strange-Device models + the old TS/Rust training cruft are in
  `archive/pre-strange-device-2026-06/` (see its README).

## Deployment

Push to `main` → `.github/workflows/deploy.yml` builds and publishes to GitHub
Pages. `vite.config.ts` sets `base: './'` so assets resolve from a project sub-path;
`assetsInlineLimit: 0` keeps the original PNGs as separate files.
