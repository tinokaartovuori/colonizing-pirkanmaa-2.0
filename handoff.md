# Handoff — Colonizing Pirkanmaa AI training (CONTINUE HERE)

_Last updated 2026-06-05. **Read this first.** Supersedes the prior A2-spatial-CNN handoff.
This is the entry point for continuing the CURRENT effort — curing the AI's passivity via a
disciplined, staged training plan — on another device. Self-contained: everything needed is in
this repo (the design docs travel with the commit)._

The deployed game is **TS/Phaser** (`src/`). The **Rust trainer** (`rust-trainer/`) trains the
AlphaZero net that gets deployed back into the TS game. Parity (Rust ⇄ TS) is bit-exact and locked
by `cargo run -p cp-train --bin parity`.

---

## TL;DR — where we are

After a long arc of failed hypotheses (more net capacity, value-head calibration, curriculum,
search horizon), the binding problem was finally diagnosed by reading actual replays + the user's
observation: **the AI is SEVERELY PASSIVE.** It never builds an army (0–3 soldiers all game),
stalls at ~10–17 tiles, Pass = ~45% of decisions, and its **~0.46 win-rate vs the HARD bot is a
MIRAGE** — ~30% of its "wins" are free enemy-self-bankruptcy. True skill ≈ 0.31–0.39 (a loser).

**Root cause = the REWARD**, not capacity and not the value head:
- A 5.5× bigger net (53.7k params) left win-rate flat → **capacity is not binding** (but that test
  was run under the passive reward, so it's only refuted *in the passive regime* — revisit in Step 4).
- The value head was un-squashed (`--record-opp-value`) and win-rate still didn't move → **value
  calibration is fixable but not the bottleneck.**
- The potential Φ rewarded only *static* economic health → sitting on a tiny economy MAXED Φ → **Pass
  was literally Φ-optimal.**

We pivoted to a disciplined, staged plan with an **HONEST metric**. Design doc:
**`rust-trainer/TRAINING-APPROACH.md`** (read it — Steps 0–4 with concrete behavioral gates).

**Steps 0 and 1 are DONE + verified (committed, compiles, parity 8/8). Step 2 is NEXT.**

---

## Key conclusions (established — don't re-litigate)

- **Spatial-CNN representation** was the historical unlock (22% → ~50%). KEEP it.
- **Passivity is the binding constraint; its root is the reward.**
- **THESIS (load-bearing):** aggression / army-building is a **CURRICULUM / terminal-signal**
  problem, NOT a shaping problem. Potential-based shaping (Ng-1999) is policy-invariant — it can
  *accelerate* convergence to the optimum but cannot *create* it. Step 1 confirmed this: the reward-Φ
  redesign grew the ECONOMY but did NOT create the ARMY.
- **Honest headline metric = `trueWinVsHard`** = raw win-rate minus bankruptcy-mirage wins. The raw
  `winRate` has misled us all along — always judge on `trueWinVsHard`.
- **Measurement discipline:** a 60-game bench ≈ ±12.6% CI → never react to a single bench; judge
  aggregated trends over ~30–60 iters. Behavioral metrics (Pass%, Outposts, max-soldiers) are tight
  (~3000 decisions/bench) and are the leading indicators.
- **Use the SMALL net** (`--net-size small`, 9786 params) for fast iteration — capacity is deferred
  to Step 4. The small net is the SAME proven spatial CNN, just without the refuted param-bloat.
- **Leave ~4 CPU cores free** (`--threads`, default cores−4) so the desktop stays usable.

---

## Current code state (committed; `cargo build --release` clean; parity 8/8)

All trainer logic is in `rust-trainer/crates/cp-train/src/bin/cnn_train.rs` (trainer/MCTS/reward/PFSP)
and `rust-trainer/crates/cp-ai/src/` (net: `spatial_net.rs`, `cnn.rs`, `planes.rs`, `candidates.rs`).

- **Step 0 — honest metrics (DONE):** `trueWinVsHard`, `bankruptcyWinShare`, `villagesPerGame`,
  `outpostsPerGame`, `maxSoldiersPerGame`, `deviceDenialRate` added to `benchmark-history.jsonl` + 4
  new dashboard panels (`training/serve-dashboard.ts`). (`tiles-lost-to-rusher` deferred to Step 2.)
- **Step 1 — reward redesign (DONE):** `potential_step1` adds, flag-gated (all default 0 = exact
  bit-identical no-op):
  - `--income-lead-potential w` — growth/lead Φ (signed income vs strongest enemy) — can't be maxed by sitting.
  - `--cap-potential w` — SATURATING soldier-cap term (`clamp(soldier_cap/7)`) → building an Outpost is **+Φ**.
  - `--idle-flow-penalty w` — idle = unused **FLOW** (unstaffed units + unspent affordable income),
    **NOT empty slots** → a fresh Outpost adds 0 idle. Resolves the idle-vs-outpost tension that
    broke earlier runs (test-proven: `building_outpost_does_not_lower_phi_under_step1`).
  - `--net-size small|large` (default large) — small = 9786-param pre-bloat arch (FD-checked). **Cold-start.**
- **Prior levers already shipped (all flag-gated):** B "eyes" = 24 spatial planes incl. correct
  frontier-reachability threat, owned-vs-conquering soldiers, device-defenseless, capacity scalars
  (see `GAME-MECHANICS.md`); C curriculum = `--script-opponents --script-frac --script-grade`
  (army-rush + device-rush scripted opponents), `--record-opp-value` (records the winning opponent
  seat as value-only examples — fixes value-squash), `--device-credit`; A horizon = `--turn-search`
  (each MCTS edge plays a full turn → reaches the round-90 Device), `--turn-search-spend` (spend the
  turn budget instead of break-on-Pass).
- **Speed (all numerically equivalent, parity-safe):** bit-exact conv `get_unchecked` rewrite (~2×),
  eval-phase saturation (`rayon::join` bench+replay into one pool, merged the 2 sequential replay
  batches), `--replay-games` default 5 (10 replay games), `--threads N` (default cores−4).

- **Step 2 — combat curriculum: IMPLEMENTED + test-verified, NOW RUNNING (not yet judged).**
  `--w-army` (FIELDED-soldier emphasis, `clamp(used_soldier/7)`, pays past one Outpost so the
  Outpost→fill chain pays end-to-end) + `--w-cut` (small defense term, `−w·hq_cut_exposure` =
  losing/severing tiles lowers Φ) in `potential_step1`; the **army-rusher** is in the scripted-opponent
  pool (`--script-opponents --script-frac --script-grade`, keep `--record-opp-value`); the
  **`tilesLostToRusher`** metric is in the training log + dashboard. All flag-gated, defaults
  bit-identical no-op, parity 8/8, 35 cp-train + 58 cp-ai tests pass. Coordination (no double-count):
  `--cap-potential` = HAVE cap (/7), `--soldier-cap-potential` = FILLED (/6), `--w-army` continues
  filling past /6 to /7, `--idle-flow-penalty` keys on unused FLOW not empty slots.
  **NEXT: judge the Step-2 gate** (~30–40 iters, aggregated): max-soldier routinely > 3, honest
  conquest wins appear, `tilesLostToRusher` trends down, `vsArmyRush` climbs off ~0.2. Then Step 3.

---

## Step 1 run result (context for the next step)

`checkpoints-cnn-s1` (small net, gen 0–30, the Step-1 launch below): ECONOMY responded — Villages
0.5→0.75/game, `bankruptcyWinShare` 0.31→0.19 (wins got HONEST), Pass% 45→~37, `trueWinVsHard`
steady ~0.35. But the ARMY did NOT materialize — Outposts stuck ~0.10/game, max-soldiers ~0.6 (≈0)
across 30 gens, even with the army-rusher in self-play at frac 0.5. Exactly the thesis: the net
TURTLES; shaping grows economy but can't create the army optimum. → Step 2's job.

(All `checkpoints-*` dirs + `cnn-backups/` ARE committed — full training history + trained nets
travel. `rust-trainer/target/` is the only large thing gitignored.)

---

## How to build & run (on the new device)

```bash
# Prereqs: Rust (stable) + Node 22 (.nvmrc). Then:
npm install
cd rust-trainer && cargo build --release          # first build is slow; target/ is gitignored

# Gates (run after any change):
cargo run -p cp-train --bin parity --release       # MUST be 8/8 (Rust == TS engine)
cargo test -p cp-ai                                # net + planes + FD gradient-checks
cargo test -p cp-train --bin cnn_train             # trainer + reward + metric tests
# If you change candidate gates/costs/rules, re-export goldens FIRST:
#   npx vite-node training/export-golden.ts   (then parity must still be 8/8)

# Dashboard (live metrics incl. the Step-0 honest panels):
npx vite-node training/serve-dashboard.ts -- --dir rust-trainer/checkpoints-<run> --port 8787

# Launch a training run (background):
./rust-trainer/target/release/cnn_train --train --out rust-trainer/checkpoints-<run> <flags>
```

### Step-1 baseline launch (verified — reproduces cnn-s1)
```bash
./rust-trainer/target/release/cnn_train --train --out rust-trainer/checkpoints-s1 \
  --net-size small --threads 16 --turn-search \
  --income-lead-potential 0.5 --tile-potential 0.4 --cap-potential 0.3 --idle-flow-penalty 0.3 \
  --record-opp-value --device-potential 0.2 --device-credit 0.15 \
  --pfsp --vs-hard-frac 0.4 --script-opponents --script-frac 0.5 --script-grade \
  --tie-penalty 0.4 --stall-rounds 80 --build-prior-floor 0.03 --shape-gamma 0.99 --shape-weight 0.3 \
  --sims 48 --cap 150 --games 24 --bench-games 60 --iters 50
```
Throughput on the dev box was ~40–60 s/iter for the small net (16 threads). Adjust `--threads` for
the MacBook's core count (leave ~4 free).

### The immediate next task
**Step 2 is implemented** — RUN it and judge the gate. Launch command (Step-1 flags + `--w-army 0.4
--w-cut 0.15`):
```bash
./rust-trainer/target/release/cnn_train --train --out rust-trainer/checkpoints-s2 \
  --net-size small --threads 16 --turn-search \
  --income-lead-potential 0.5 --tile-potential 0.4 --cap-potential 0.3 --idle-flow-penalty 0.3 \
  --w-army 0.4 --w-cut 0.15 \
  --record-opp-value --device-potential 0.2 --device-credit 0.15 \
  --pfsp --vs-hard-frac 0.4 --script-opponents --script-frac 0.5 --script-grade \
  --tie-penalty 0.4 --stall-rounds 80 --build-prior-floor 0.03 --shape-gamma 0.99 --shape-weight 0.3 \
  --sims 48 --cap 150 --games 24 --bench-games 60 --iters 50
```
Judge the gate (max-soldier > 3, honest conquest wins, `tilesLostToRusher` ↓, `vsArmyRush` ↑) over
~30–40 iters aggregated. Then Step 3 (device reaction + strategic arc), Step 4 (re-test capacity
only after the net plays actively).

---

## Constraints (do not break)

- **Parity 8/8.** Candidate gates/costs are mirrored: `crates/cp-ai/src/candidates.rs` ⇄
  `src/ai/nn/candidates.ts`, `crates/cp-sim/src/resources.rs` ⇄ `src/core/resources.ts`. Any gate/
  cost/rule change → edit BOTH → re-export goldens → parity 8/8.
- **Map-gen MSVCRT RNG** (`src/core/rng.ts`, `src/world/worldgenerator.ts`) must NOT change (bit-exact).
- **Net-input changes** (plane count, value-scalar dim, local dim) → COLD-START + finite-difference
  gradient-check (pattern in `spatial_net.rs` `combined_grad_*` tests).
- `planes.rs` / `spatial_net.rs` / `cnn_train.rs` (Φ, MCTS) are **AZ-only / parity-free** — change freely.
- Reward-Φ changes are NOT net-input changes → no cold-start.

---

## Pointers (all in-repo, travel with the commit)

- **`rust-trainer/TRAINING-APPROACH.md`** — THE plan (Steps 0–4, gates, §1.x reward defs, §2.2 curriculum). **Primary doc.**
- **`rust-trainer/GAME-MECHANICS.md`** — verified, canonical game mechanics (source of truth for the eyes + reward design).
- **`rust-trainer/EXP-M-DESIGN.md`** — implementation log for the eyes / curriculum / horizon / capacity work + launch commands.
- `rust-trainer/{REWARD-DESIGN,TRAINING-RESEARCH,ALPHAZERO-DESIGN,GAME-COMPLEXITY-AND-TRAINING,MCTS-DESIGN}.md` — background.
- `CLAUDE.md` — fidelity constraints + model management (`models/` registry, `npm run models`).
- `reference/` — the original C++/Qt game (canonical game logic).

---

## Operational notes
- On the Linux dev box, `pkill`/`nohup` hit an exit-144 sandbox quirk (the kill still worked) — likely
  absent on macOS.
- Launch trainers detached and inspect via the JSONL files + the dashboard; reads never disturb a run.
- The trainer checkpoints `champion.json` every bench (≤5 iters lost on a kill). Resume with
  `--init <dir>/champion.json --out <newdir>` (warm-starts the net; replay buffer resets).
