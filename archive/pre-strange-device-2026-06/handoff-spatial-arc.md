# Handoff — Colonizing Pirkanmaa AI (spatial-policy arc)

_Last updated 2026-06-03 (session 2: draw-attractor diagnosis + KL-anchor fix). **Read this first.**
Goal: an AI that beats the CPU **Hard bot 70%+**. Everything is **uncommitted** in the working tree.
**`az13` is training** (the fix; see below). Deep docs:
`rust-trainer/{GAME-COMPLEXITY-AND-TRAINING, EXP-I-DESIGN, TRAINING-RESEARCH, ALPHAZERO-DESIGN, REWARD-DESIGN}.md`._

## TL;DR — honest status

- **Best model = `exp-A` (`rust-trainer/checkpoints-az/champion.json`, 63-dim policy): ~33% vs hard @ 12×12, ~31% @ 14×12.** This is the ceiling everything keeps hitting.
- This session: did a **full game-mechanics study** + built and ran the **spatial/cut-aware policy (Exp I)** — the research's #1 lever. **The representation works and is verified, but pure self-play TRAINING DEGRADES the policy** (drift to passivity / draw-attractor). Spatial sight is **necessary but not sufficient**.
- **70% NOT reached.** The bottleneck moved from *representation* to *training stability*.
- **Session 2 (this update) — ROOT-CAUSED the training instability + shipped a fix, now running as `az13`.**
  The collapse is the **draw-attractor**: `--timeout-penalty` was **0.0** (a draw was valued neutral = 0), so in
  the draw-prone game self-play converged to "turtle to a safe draw." Triple-confirmed: az11 converts wins→timeouts
  (timeout 25→60%), az12 converts losses→timeouts by turtling (loss 40→25% but timeout 42→65%), and in BOTH the
  value-loss collapsed to ~0.10 (= "everything ≈ draw 0", killing the positional gradient). **Fix = `--timeout-penalty 0.4`
  (draw target = −0.4, so winning ≫ drawing) + a new `--kl-anchor` KL trust-region pulling the policy toward the frozen
  warm-start (stops the drift).** Early `az13` read (iter 0→40): **drift ARRESTED** — win holds 30–40%, timeout ~25–30%,
  policyLoss ~0.74 / valueLoss ~0.5 (NOT collapsing). Whether it CLIMBS past exp-A is the open question (still running).
- **The live game's deployed AI was NOT changed** (`src/ai/nn/weights.ts` is still the older `rust-trainer/checkpoints/champion.json`, NOT even exp-A). Deploying exp-A is a pending, easy win (~2× the live AI). See "Deploy".

## What is RUNNING right now
**`az13` is TRAINING** — `rust-trainer/checkpoints-az13`, 250 iters (~3.5h), warm-start transplant +
`--timeout-penalty 0.4` + `--kl-anchor 1.0` (the draw-attractor fix). Dashboard up on :8787 → `checkpoints-az13`.
Best-checkpoint saving on (`champion-best.json` / `value-best.json`). Watch the **timeout rate** (must stay low,
NOT climb to 50–60% like az11/az12) and whether **win climbs past ~33%**.

**Follow-up experiments are PRE-WIRED** in `rust-trainer/launch-next-az.sh <noctrl|strongkl|weakkl>` — pick by az13's read:
az13 still drifts → `strongkl` (λ=2.0); az13 froze ~33% → `weakkl` (λ=0.5, loosen so it can climb — timeout-penalty
alone now blocks passivity); az13 worked → `noctrl` (the clean control: timeout-penalty alone, no KL — does KL even help?).
Do NOT run a successor concurrently with az13 (20 cores can't feed two 16-thread runs).

> **User preference:** let a run breathe ≥2–3h before acting; KEEP a run going rather than stopping early on a
> negative read (they want idle hours used + a result to inspect). Checking the trajectory a few times is fine; stopping is not.

## The big finding this session (firm, two-part)

**Part 1 — the game is a SPATIAL GRAPH game (mechanics audit, 5 parallel deep reads).**
Winning requires OFFENSIVE force-concentration that even the hard bot is only mediocre at:
- **HQ-connectivity cut:** every end-turn, a 4-connected BFS from each player's HQ; any owned tile not connected is confiscated/neutralised. **Take the one articulation tile that severs the enemy HQ → a chunk (or, if you take the HQ, ALL) of their territory collapses.** This is graph/min-cut reasoning a flat feature vector cannot see.
- **Combat:** strict `attackers > defenders` on ONE tile in ONE turn (deterministic, tile cap 3). Piecemeal attacks lose everything. Defended HQ needs massed force.
- **Army economics:** soldier cap = HQ(+1) + Outpost(+3, 650money+300metal); metal only from Mines. Fielding an army is a multi-step tech chain.
- **Why hard only wins ~40% (and draws ~36% vs itself):** the game is **draw-prone**; a passive turtle that just keeps its HQ is hard to eliminate → timeouts. Measured: hard-vs-hard 36% never resolve even at cap 5000. Full study + numbers in `GAME-COMPLEXITY-AND-TRAINING.md`.

**Part 2 — exp-A behaviour (champ_probe, measured):** it DOES try to attack (Attack chosen ~70% when available) but is **cap-locked at ~1.5 soldiers** (BuildOutpost ~0.1% — long-horizon credit problem, NOT a missing feature: the Outpost candidate already exposes `soldier_cap_gain=3.0`). Its wins are mostly the enemy collapsing. **Forcing an army made it WORSE** (champ_probe `--force-military`: 30→25%, upkeep drains economy + it can't coordinate force). → the binding constraint is **② spatial coordination / cut**, not ① army-building.

## Exp I (spatial policy) — built, verified, but training drifts

**Built (all AZ-only, parity 8/8 preserved, live game untouched):**
- `cp-ai/src/spatial.rs::offensive_cut_value(g, attacker, target)` — "fraction of enemy that disconnects if I take tile T" (1.0 if enemy HQ). **Unit-tested** on a known graph.
- `cp-ai/src/policy_spatial.rs` — `candidate_spatial_features` (6 dims: cut-value, enemy-HQ-proximity, is-enemy-HQ, own-cut-vuln, enemy-neighbor-frac, owner-is-enemy), `policy_input_spatial` (= 63 + 6 = 69), `DEFAULT_ARCH_SPATIAL [69,24,16,1]`, `select_index_spatial`, **`warmstart_spatial`** (transplant exp-A 63-dim → spatial net, 6 spatial weights = 0 → init PREDICTS IDENTICALLY to exp-A; unit-tested).
- `cp-ai/src/search.rs` — `SearchConfig.spatial_policy` flag; `make_node` priors + `select_with_pi` recorded inputs branch to `policy_input_spatial` when set (default false → parity byte-identical).
- `cp-ai/src/controller.rs` — both fallback `select_index` calls + the trace-block scores branch to spatial when `spatial_policy` (behind the flag → parity safe).
- `cp-train/src/bin/alphazero.rs` — `--spatial-policy` (warm-start via transplant if `--init-policy` given, else cold-start) + **best-checkpoint saving** (`champion-best.json`/`value-best.json`, so long runs never lose the peak).
- `cp-train/src/bin/champ_probe.rs` — behavioural probe (per-outcome tile%/soldiers/buildings, intent histogram, `--force-military`, `--spatial-policy`, and a **legitimate win-rate** that strips games where hard self-bankrupted-while-healthy — verified 0 such games in real play, so the benchmark is clean).
- `cp-train/src/bin/hard_vs_hard.rs` (game-length study) + `hard_econ_check.rs` (benchmark-integrity / hard self-bankruptcy check).
- **(session 2) `cp-ai/src/policy_train.rs` — KL trust-region anchor:** `PolicyTrainer.ref_genome` (frozen warm-start) + `kl_coeff`; each step adds `kl_coeff*(p_c − q_c)` to the per-candidate upstream (= gradient of `kl·KL(q‖p)`, equivalent to CE against the blended `pi + kl·q`). Default off (None/0.0) → **parity 8/8 still PASS**, all 21 cp-ai tests green incl. new `kl_anchor_pulls_policy_toward_reference`. Wired in `alphazero.rs` as `--kl-anchor <λ>` (reference = the warm-start `init_genome`, cloned + frozen).
- **(session 2) `rust-trainer/launch-next-az.sh`** — fires the pre-wired az13 successors (`noctrl`/`strongkl`/`weakkl`).

**Ran (all FAILED to beat exp-A):**
| run | config | result |
|---|---|---|
| `checkpoints-az10` | cold-start, pure self-play | stuck ~5% (can't learn from random) |
| `checkpoints-az11` | **warm-start** spatial, pure self-play, 14×12 | 31%→drifted ~12%; best = warm-start ≈ exp-A (champ_probe **31%**) |
| `checkpoints-az12` | warm-start + `--vs-hard-frac 0.75` + lr 5e-4 (anchored) | ~25%→drifted ~10%; best champ_probe **21.5%** (below exp-A) |

**Conclusion:** spatial REPRESENTATION is correct + works (parity 8/8, champ_probe runs it), but **self-play TRAINING DYNAMICS drift the policy to passivity** (timeouts 60–65%, tileFrac→0.09). Neither pure self-play NOR 75% fixed-opponent anchoring stops it. **The bottleneck is now training stability, not representation.**

## Next steps (priority order) — the lever is TRAINING STABILITY

0. **WATCH `az13`** (running): does win climb past ~33% or plateau? Decision tree:
   - **plateaus ~33%** (anchor too strong, can't climb) → `bash rust-trainer/launch-next-az.sh weakkl` (λ=0.5).
   - **drifts after all** (timeout climbs) → `... strongkl` (λ=2.0).
   - **works** → `... noctrl` to confirm timeout-penalty ALONE was enough (is KL even needed?), then probe the champion-best with `champ_probe --spatial-policy` to verify it now USES the cut-features (the whole point, ex-#4).
1. ~~**KL / trust-region anchor**~~ — **DONE** (session 2): `--kl-anchor` shipped + verified, `az13` running it together with `--timeout-penalty 0.4`. Early read: drift arrested.
2. **Anchor/freeze the value net:** it co-trains and may collapse to "everything ≈ draw". The `--timeout-penalty` partly addresses this (value targets no longer all ≈0); if value still collapses, try a frozen/slow-updating value next.
3. **BC warm-up + very gentle RL:** keep the policy near exp-A, nudge lightly. (The KL anchor is a continuous form of this.)
4. Once stable + climbing: verify it actually exploits `offensive_cut_value` via `champ_probe` intent histograms.
5. **Deploy exp-A NOW regardless** (independent of the above) — see below; it ~2×'s the live AI.

## Deploy (pending, easy win — independent of the research)

The live game ships an OLD champion (`weights.ts` meta `source: rust-trainer/checkpoints/champion.json`), weaker than exp-A. To deploy exp-A: `emit-weights.ts`'s CLI guard fails under vite-node, so call `writeWeights` directly:
```ts
// training/_deploy.ts  (run: npx vite-node training/_deploy.ts)
import * as fs from 'node:fs';
import { writeWeights, DEFAULT_TIERS } from './emit-weights';
const g = JSON.parse(fs.readFileSync('rust-trainer/checkpoints-az/champion.json','utf8'));
writeWeights(g, DEFAULT_TIERS, {source:'exp-A', date:'2026-06-03'}, 'src/ai/nn/weights.ts');
```
Then `npm run build` (tsc gate) + a Playwright smoke. exp-A plays with static-leaf MCTS (its native mode; `TIER_SEARCH` hard = sims 400 static — already the default). **The spatial champions CANNOT be deployed** until `policy_input_spatial` is ported to TS (`src/ai/nn/`) — a separate task. Backup of current weights: `/tmp/weights-predeploy-backup.ts`.

## How to run (repo root)
```bash
# DASHBOARD (live)
npx vite-node training/serve-dashboard.ts -- --dir rust-trainer/checkpoints-az12 --port 8787   # http://127.0.0.1:8787/
# PARITY GATE (must stay 8/8 after any cp-sim/cp-ai/features/candidates change)
cd rust-trainer && cargo run --release -p cp-train --bin parity
# BUILD trainer
cd rust-trainer && cargo build --release -p cp-train --bin alphazero
# RESUME the spatial line (warm-start transplant; add a stability fix first per Next steps):
rust-trainer/target/release/alphazero --out rust-trainer/checkpoints-az13 \
  --init-policy rust-trainer/checkpoints-az/champion.json --init-value rust-trainer/checkpoints-az4/value.json \
  --spatial-policy --spatial-value --leaf-value --sims 96 --iters 250 --games 32 --epochs 2 \
  --bench-games 40 --bench-every 5 --cap 120 --width 14 --height 12 --seed 7 --threads 16
# EVALUATE a champion honestly (spatial nets NEED --spatial-policy or champ_probe panics):
rust-trainer/target/release/champ_probe --champion <champion-best.json> --value <value-best.json> \
  --spatial-policy --width 14 --height 12 --sims 96 --games 200
# game-length / hard-econ diagnostics:
rust-trainer/target/release/hard_vs_hard --games 1000 --cap 5000 --width 12 --height 12
rust-trainer/target/release/hard_econ_check --mode vshard --games 300 --width 14 --height 12
```
Checkpoint dirs: `checkpoints-az`(exp-A, BEST), `-az4`(41-dim spatial value for warm-start), `-az10/11/12`(spatial runs, failed). Best-of-run saved as `champion-best.json`/`value-best.json` in az11/az12.

## Invariants / cautions (do NOT break)
- **Parity 8/8** is the correctness contract. The whole spatial path is behind `spatial_policy` (default false) → shipped/parity path is byte-identical. Any `cp-sim`/`cp-ai`/`features`/`candidates` change → re-export golden + `cargo run -p cp-train --bin parity`.
- **Do NOT change shipped `LOCAL_DIM`/`GLOBAL_DIM`/`candidates.ts`/`policy.ts`/`weights.ts`** — spatial features are an AZ-only path (Rust). Deploying a spatial champion requires porting `policy_input_spatial` to TS first.
- **champ_probe needs `--spatial-policy`** for 69-dim spatial champions (else mlp panics "len 63 index 63").
- Reward = **pure win/loss** is correct; all dense shaping (positional Φ, decisiveness, aggression) made it worse — don't revisit reward before fixing training stability.
- Deliberate economy divergence (Mine/Hydro/Nuclear) is intentional — see `CLAUDE.md`.

## Memory pointers
`~/.claude/.../memory/`: `training-watch-cadence.md` (let long runs breathe ≥2h; KEEP a run going overnight, don't stop early), `alphazero-pivot.md`, `reward-design-loop.md`, `rust-ai-training.md`, `economy-rebalance.md`, `neural-ai.md`.
