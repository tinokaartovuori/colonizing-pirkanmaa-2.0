# Phase 2 design — spatial/positional per-target local features

_Approved design, ready to implement. Source of truth: TS `src/`; Rust mirrors for parity._
_Goal: raise the AI's ceiling toward beating the hard heuristic ~95% (research-backed: AlphaStar entity/pointer, GNN object-centric per-target features). See `TRAINING-RESEARCH.md`._

## What Phase 1 already did
Expand/Attack emit one candidate per target tile; MLP argmaxes over (intent × target). But `local` (10-dim) has NO positional info → net can't learn spatial strategy. Phase 2 adds it. **This is a DIM-CHANGING change → retrain from scratch.**

## Core change (recommended: ship this first, clean parity-green baseline)
Add **6 spatial per-target features** to `local` (indices 10–15). `LOCAL_DIM 10→16`, `POLICY_INPUT_DIM 57→63`, `DEFAULT_ARCH [63,24,16,1]`, **paramCount = 1953**. (Note: real current 57-dim paramCount is **1825**, not the stale "1809" in comments.) All map-size-invariant; Manhattan distance `|dx|+|dy|` (matches `controller.ts:89`); clamp(v,…) as noted; sentinel `99` for missing HQ/enemy applied BEFORE the /denominator.

| idx | name | formula (tile=target, p=actor) | clamp | for |
|---|---|---|---|---|
|10|enemyNeighbors| (#8-neighbors owned by ∉{null,p}) / 8 | [0,1] | Expand, Attack |
|11|ownNeighbors| (#8-neighbors owned by p) / 8 | [0,1] | Expand, Attack |
|12|neutralNeighbors| (#8-neighbors owner==null) / 8 | [0,1] | Expand |
|13|distOwnHq| Manhattan(tile, getHqTile(p); null→99) / 20 | [0,3] | Expand, Attack |
|14|distNearestEnemyTile| min Manhattan(tile, any enemy-owned tile; none→99) / 20 | [0,3] | Expand, Attack |
|15|attackerSupport / frontier| Attack: (#own Soldiers on 8-neighbors)/3 ; Expand: 1 if enemyNeighbors>0 else 0 ; else 0 | [0,3] | Attack, Expand |

Non-positional intents (Build*/Hire/Stack/Pass) emit the 6 dims as **0.0** → stay value-equivalent → stay single-candidate.

Perf: precompute `enemyCoords` (enemy-owned tile coords) ONCE per `enumerate()` (like the Phase 1 `tileIndexMap`), pass to expand/attack; min-reduction is commutative so order-independent, but iterate `getTiles()`/`g.get_tiles()` order in both. Neighbor counts are O(8).

## Optional (same retrain): grow net to [63,32,24,1] (2865 params). Try as a SECOND retrain only if core doesn't reach target; don't entangle arch experiment with the parity gate.

## Spatial APIs (parity-proven identical TS↔Rust — REUSE, do not reimplement inline)
- 8-neighbors: `TileBase.getNeighbourTiles()` ↔ `Game::neighbour_tiles(tid)` (Coordinate.neighbours(1,w,h), outer-x/inner-y, Chebyshev r=1, clamped).
- owner: `t.getOwner()` ↔ `g.tiles[tid.0].owner`. HQ: `om.getHqTile(p)` ↔ `g.get_hq_tile(p)`. coord: `t.getCoordinate().x()/y()` ↔ `g.tiles[tid.0].x/.y` (pub, model.rs:290-291).
- No new sim accessor needed (neighbour_tiles already uses settings w/h internally).

## File changes
TS: `candidates.ts` — `LOCAL_DIM=16`; extend `localVec()` (append 6 clamped entries in order, new optional params default 0); add `tileSpatial(tile,p,om,enemyCoords)`; in `enumerate()` build `enemyCoords` once and thread into `expandCandidates`/`attackCandidates`; set per-tile slot-15 dual meaning. `policy.ts` auto-derives (optionally bump arch literal line 17). No change to mlp/controller/features.
Rust: `candidates.rs` — `LOCAL_DIM=16`; extend `Local` struct + Default + `local_vec()` (same order, `clamp3`); add `tile_spatial(g,tid,p,enemy_coords)`; thread `enemy_coords` in `enumerate()`. `policy.rs` auto-derives. Add Rust unit test for `tile_spatial` vs hand-computed board.

## Parity / golden / weights (the dim change flows through)
1. **Retire old champion**: rename `training/checkpoints/champion.json` → `champion-v57.json.bak` (preserves the ~26.7% champion) so golden export falls back to deterministic-LCG genome at the NEW dim. Also clear/rename any `hof.json` and old training checkpoints so no 57-dim warm seed corrupts retraining.
2. `export-golden.ts`: `POLICY_ARCH [57,24,16,1]→[63,24,16,1]`; `SCHEMA_VERSION 2→3`; update embedded local-feature table text (document idx 10–15 + slot-15 dual meaning). Re-export; run twice & diff for byte-determinism.
3. `parity.rs` auto-adapts (reads `trace.genome_arch`/`param_count`, checks per-candidate `local.len()`). Run `cargo run -p cp-train --bin parity` → must be ALL PASS. Likely first-divergence culprit = a distance/neighbor mismatch.
4. weights.ts: `emit-weights.ts` is arch-agnostic. During retrain window, stamp a **zero 63-dim placeholder** weights.ts so `npm test` structural test (`arch[0]===POLICY_INPUT_DIM`, paramCount) stays green and the live client compiles. Strength tests are `runIf(TRAINED)` → skipped until a calibrated champion exists.

## Tests
TS add: (a) expand target adjacent to enemy has local[10]>0 vs interior 0; (b) genome weighting local[14] (distNearestEnemy) NEGATIVELY → chosen expand target is closest to enemy (proves spatial strategy LEARNABLE); (c) attack target with more adjacent own soldiers has higher local[15]; (d) Build*/Hire/Stack have all 6 spatial dims 0. Existing `c.local.length===LOCAL_DIM` auto-checks width.
Rust: parity gate is the cross-check; add focused `tile_spatial` unit test.

## Risks
Distance metric (Manhattan int → f64 /20, NOT Chebyshev); sentinel 99 before divide; enemyCoords contents identical (enemy = owner present AND != p, excl. neutral); perf (precompute enemyCoords); MANDATORY retrain (old 57-dim champion unloadable — tighten `evolve.ts` warm-seed arch check, or just use Rust `train.rs` which validates arch); deployed weights.ts migration (zero placeholder during retrain).

## Step-ordered checklist (parity-green is the gate before any retrain)
1. Retire old champion.json + hof + checkpoints (rename).
2. TS: candidates.ts feature change (+ optional arch bump). Stamp zero 63-dim weights.ts. Add TS tests. `npm test` green (strength skipped).
3. export-golden.ts: POLICY_ARCH→63, SCHEMA_VERSION→3, table text. Re-export; verify byte-determinism.
4. Rust: candidates.rs mirror + tile_spatial test.
5. `cargo run -p cp-train --bin parity` → ALL PASS. Do not proceed until green.
6. Retrain from scratch at [63,24,16,1] (Rust `train.rs`, with --pfsp). Confirm no 57-dim warm seed.
7. emit-weights → real weights.ts; benchmark vs hard AI; track win-rate toward target.
8. (Optional) repeat with [63,32,24,1].
