# Training Plan — breaking the representation ceiling (Route A)

_Created 2026-06-03. Operational companion to `handoff.md`. Where `handoff.md`
records **what happened**, this records **what to do next and why**, as an
ordered experiment ladder with explicit kill/promote criteria. Read `handoff.md`
and memory `az-pass-collapse-fix` first for the diagnosis; this assumes it._

## 0. The one-paragraph situation

The AlphaZero stack works end-to-end and the game is decisive (Strange Device).
The win-rate vs the HARD bot is **capped at ~22 %** and this session proved the
cap is **representation**, not reward/exploration/opponent: the net Passes ~70 %
of decisions even when productive candidates (Expand in 33–75 %) are on the menu
(`champ_probe --force-military`). The fix is **Route A**: let the network *see the
board* and *choose targets* instead of ranking 12 intents over a spatially-blind
36-dim aggregate. The infrastructure for the cheap first slice already exists
(`policy_spatial.rs`) and is now training (Experiment A).

## 1. What the network currently sees (so we don't re-add it)

Per-decision the policy MLP scores each candidate with a **70-dim** input
(spatial-policy arc):

| block | dims | source | parity |
|---|---|---|---|
| global aggregate | 36 | `features.rs::global_features` | LOCKED |
| intent one-hot | 12 | `candidates.rs` (`INTENT_COUNT`) | LOCKED |
| per-candidate `local` | 16 | `candidates.rs::local_vec` | LOCKED |
| — of which **spatial** | (6) | local 10–15: enemy/own/neutral-neighbor frac, dist-own-HQ, dist-nearest-enemy, frontier | LOCKED |
| **Exp-I spatial block** | 6 | `policy_spatial.rs::candidate_spatial_features` | **AZ-only / additive** |

The Exp-I 6: `offensive_cut_value` (the crux — enemy fraction severed from its HQ
if I take this tile), `enemy_hq_proximity`, `is_enemy_hq`, `own_cut_vulnerability`,
`enemy_neighbor_frac`, `target_owner_is_enemy`.

**The value net** has its own spatial enrichment: 36 global + 5 board summaries =
**41-dim** (`features.rs::value_features_spatial`), used with `--spatial-value`.

Net: the policy already has *some* per-candidate spatial awareness; what was
missing until Exp-I is the **graph/articulation** view (cut-value), and what is
still missing is a **true spatial trunk** (the net sees aggregates and
per-candidate scalars, never the grid itself). That gap is A2 (Experiment C).

## 2. The experiment ladder

Each rung has a **launch command**, a **judge-by** signal, and a **promote/kill**
gate. Judge by the win-cause split + intent profile, **not** the headline number
(`training-watch-cadence`: let a run breathe ≥2 h; early spikes are value-MCTS
lifting a random policy). All runs warm-start so they begin at the parent's
strength and only *add* capability.

### Experiment A — thin spatial policy (RUNNING)

The handoff's cheap-first: train the existing 6-feature `--spatial-policy` path
with this session's full fix stack, warm-started from the registered baseline
`sd-az-001` (policy `[64,24,16,1]` → `[70,24,16,1]`, value `[41,32,16,1]`).

```bash
cd rust-trainer
./target/release/alphazero --out checkpoints-az2 \
  --spatial-policy --init-policy saved-nets/vs-hard-latest-champion.json \
  --spatial-value  --init-value  saved-nets/vs-hard-latest-value.json \
  --shaping 0.3 --combined-shaping --vs-hard-frac 0.75 \
  --timeout-penalty 0.5 --win-speed 0.3 \
  --dirichlet-alpha 0.4 --dirichlet-eps 0.35 --move-temp 1.2 --temp-until-round 120 \
  --leaf-value --cap 300 --width 14 --height 12 \
  --games 48 --sims 64 --epochs 4 --batch 128 --buffer 60000 \
  --bench-every 5 --bench-games 80 --iters 800 --seed 1
```

- **Hypothesis:** the cut-value feature alone lets the net pick severing targets
  → win-rate clears ~22 %.
- **Judge by** (over a ≥2 h / ~iter-150 window, not a single bench):
  - win-rate **trend** vs hard (mean of last ~10 benches);
  - **win-cause split** — we want Device + Conquest up, Bankruptcy/Tiebreak down;
  - **intent profile** — Pass fraction down from ~70 %, Expand/Attack up.
- **Promote** if mean win-rate clears **~30 %** with a non-degenerate cause split
  → register as `sd-az-002`, make it the warm-start parent for B.
- **Kill / escalate to B** if it oscillates around ~22 % with no upward trend
  over a 2 h window (same signature as the non-spatial plateau) — the 6 scalars
  are too thin; go to Experiment B.

### Experiment B — enriched per-candidate spatial block (SUPERSEDED by Experiment C)

> **Not implemented.** The 2026-06-03 decision was to skip the scalar patches and
> go straight to the A2 CNN trunk (Experiment C). This spec is kept only as a
> fallback if the CNN path is abandoned. ⤵

If A caps, the next-cheapest lever is **more orthogonal per-candidate spatial
signal** before paying for a full trunk. Implemented additively in
`policy_spatial.rs` behind `--rich-spatial` (parity-neutral; A's run untouched).
The rich block appends, on top of the Exp-I 6, signals the net cannot currently
infer:

1. `target_econ_potential` — terrain-derived best-building economic value at the
   target tile (river→hydro, mountain→mine/metal, abundant-forest→wood,
   grassland→farm, forest→low). Lets the net prefer *economically* strong build
   targets, which no current feature encodes per-tile.
2. `target_defenders` — enemy soldiers standing on the target / norm → how
   contested an Attack target is (attack viability).
3. `would_strand` — 1 if expanding to / taking this tile would **not** be
   HQ-connected for me (anti-stranding; the inverse of giving the enemy a free
   cut on my own territory).
4. `setup_cut` — the best `offensive_cut_value` available on the target's
   enemy-neighbors *after* I take the target (a one-ply spatial lookahead: "does
   this open a sever next turn"). The cheap stand-in for tree-search target play.

- **Warm-start:** `warmstart_spatial_generic` pads either the 64-dim baseline
  **or** the 70-dim thin champion up to the rich dim (new features at 0 → predicts
  identically at init).
- **Launch:** same command as A with `--rich-spatial` and `--out checkpoints-az3`,
  `--init-policy` = A's champion if A produced a usable active net, else the
  baseline.
- **Promote/kill:** same gates as A. If B also caps → the scalars are saturated;
  the bottleneck is genuinely the missing **trunk** → Experiment C.

### Experiment C — spatial trunk (A2 proper) — **CHOSEN: hand-rolled Rust CNN**

The real fix per `TRAINING-RESEARCH.md` §1B/§2: stop summarizing the board into
scalars; feed the **grid itself**. **Decision (2026-06-03, with the user): go
straight to A2 — the scalar patches (A/B) are de-risking the literature already
settled. Build it as a hand-rolled Rust CNN** (no tensor-crate dependency; keeps
the fast deterministic MCTS loop and the repo's bit-exact philosophy). Rejected:
libtorch/candle (heavy dep, non-deterministic, parity-twin redo) and a GNN
(message-passing backprop to hand-roll) — both reconsiderable if the CNN stalls.

**Architecture (size-agnostic via conv + global-avg-pool):**
```
planes[PC×H×W]  (owner me/enemy/neutral, my-HQ, enemy-HQ, producer-bldg,
   │             military-bldg, device, soldiers, my-HQ-connected mask)
   │  Conv2d(PC→16,3x3) → tanh → Conv2d(16→24,3x3) → tanh
   ▼
board_embed[24×H×W]
   ├─ GlobalAvgPool → value head: Dense(24→24)→tanh→Dense(24→1)→tanh = value
   └─ per candidate c with target tile (x,y):
        concat( board_embed[:,y,x](24), global(24), local(16), intent_onehot(12) )
        → Dense(76→24)→tanh→Dense(24→1) = policy score for c
```
~7.4k params. Trunk runs ONCE per MCTS node (cheap); the per-candidate head reuses
the cached board_embed. Fully board-size-invariant → train at 14×12, deploy anywhere.

**Build phases:**
- **Phase 1 — conv primitives `cnn.rs`** — Conv2d/tanh/GlobalAvgPool/Dense, each
  forward+backward, finite-difference gradient-checked. **DONE ✓** (tests green).
- **Phase 2 — board planes `planes.rs`** — `board_planes(g,player)->(Vec<f64>,h,w)`,
  `PLANE_COUNT=10`, pure + unit-tested. **DONE ✓**.
- **Phase 3 — `spatial_net.rs`** — the dual-head `SpatialNet` above + `train_grad`
  (combined cross-entropy policy + MSE value loss, shared trunk backpropped once) +
  SGD `apply_grad`, finite-difference gradient-checked on the combined loss.
  **DONE ✓** (`SpatialNet::default_for(10,16,12,seed)`, 7434 params).
- **Phase 4 — standalone CNN trainer core. DONE ✓** (user chose a standalone
  binary over retrofitting the generic `search.rs`). `crates/cp-train/src/bin/
  cnn_train.rs`: own PUCT MCTS (c_puct 1.5, priors = softmax of
  `score_candidate`, leaf value = `value_from`, root-perspective so no sign flip;
  non-root seats forced HardAi during expansion), self-play loop mirroring
  `selfplay.rs` (HQ placement, stalemate cut), per-decision examples `(planes,
  candidates[(target(x,y), local16, intent_onehot12)], π=visit dist, z)`, and a
  batched `train_grad`→`apply_grad` SGD step (lr 0.01, l2 1e-5). `--smoke` gate
  passes (loss decreases). **Also added TERRAIN planes** (`PLANE_COUNT` 10→14:
  grassland/forest/mountain/river) — the net now sees terrain, so hydro-on-river /
  mine-on-mountain become visible. Net = 8010 params. Candidate enumeration
  unchanged → **parity-safe / AZ-only**.
  _Left for Phase 5:_ vs-hard win-rate benchmark, self-play→train→evaluate
  iteration driver, checkpoint save/load + dashboard-format logging, model-registry
  integration, Dirichlet+temperature self-play exploration, replay buffer.
- **Phase 4b — distillation warm-start (avoids cold-start ~5%).** Before RL,
  supervise the CNN to imitate the registered MLP champion `sd-az-001` (behaviour-
  clone its candidate scores over self-play states) for a few epochs, THEN continue
  with MCTS self-play. The CNN has no clean weight-transplant from the MLP, so
  distillation is the warm-start.
- **Phase 5 — train + benchmark** vs HARD; judge by §3; register the first CNN
  champion `sd-az-NNN` the moment it clears the scalar ceiling with a decisive split.
- **Phase 6 — A1 (learned target selection), only if the trunk lifts.** Remove the
  heuristic `EXPAND/ATTACK_CANDIDATE_CAP` so the net scores *every* reachable tile.
  This DOES touch `candidates::enumerate` → **parity path**: build the TS twin in
  `src/ai/nn/*` in lockstep, `npx vite-node training/export-golden.ts`, then
  `cargo run -p cp-train --bin parity` must stay **8/8** before training.

### Orthogonal environment levers (apply to any rung)

These improve the *training environment* independent of representation; fold in
once a representation rung is promoted, changing one variable at a time:

- **A3 — PFSP opponent picking.** Replace uniform self/hard mixing with
  prioritized-fictitious-self-play weighting toward opponents the net *loses* to
  (AlphaStar). Cheap; biggest remaining environment win. The `--vs-hard-frac 0.75`
  exploiter is a crude first step; PFSP generalizes it.
- **A4 — potential-based shaping.** DONE (`z − λΦ`, `--combined-shaping`); keep it.
- **Opponent diversity.** Add a frozen *earlier* champion to the self-play mix so
  the net can't collapse into beating only its current twin.

## 3. Evaluation protocol (how we decide, consistently)

- **Primary metric:** mean win-rate vs HARD over the last ~10 benches (80 games
  each, both seats) — not any single bench (±~8 % at n=80).
- **Behaviour gates (must hold for a promote):** Pass fraction trending **down**;
  Device + Conquest wins present (decisive play, not tiebreak/bankruptcy luck);
  `deviceSurvival` healthy.
- **Decisive diagnostic on any promoted champion:** `champ_probe
  --force-military` — if a free army lifts win-rate, army-building was the
  constraint (reward/exploration); if not, target selection is still the cap
  (representation) → next rung.
- **Register the moment it's good** (`npm run models -- register … --arc sd --type
  az --parent <id>`); `/tmp` is volatile and cold-start relaunches overwrite
  `champion-best.json`. Lineage lives in the registry, never in filenames.

## 4. Success definition

A champion is "the AI plays better" when, vs HARD, it **(a)** clears ~30–40 %
win-rate with **(b)** a decisive cause split (Device/Conquest, not
tiebreak/bankruptcy) and **(c)** a Pass fraction well below the ~70 % plateau
signature — i.e. it *acts*, *targets*, and *converts*. The ladder above is the
ordered, lowest-cost-first path to that, with each rung empirically gating the
next so we never pay for a trunk we didn't need (or starve a run that just needed
to breathe).

## 5. Current state (live) — CNN RUN (cold, after a critical bug fix)

> **2026-06-04 — CRITICAL BUG FIXED.** First CNN runs were ~98.5 % Pass with 0
> Expand/Attack/Hire (cold AND warm). Root cause was NOT representation: the
> standalone `cnn_train` turn loop **omitted the economy scaffold**
> (`ensure_income`/`staff_income`) the real controller runs every turn → the CNN
> seat never staffed workers → no income → `enumerate` degenerated to `{Pass}`
> from round 1 → economically dead → conquered. Fixed (scaffold wired into all 4
> CNN turn loops; diagnostic confirms 40 decisions/game, Expand in 33, self-play
> 100–274 examples/game vs 9). Broken `checkpoints-cnn` artifacts discarded;
> **relaunched COLD with the fixed binary** (+ lightened replay: every 10 iters,
> 3+3 games, parallel). The earlier "CNN RUN LAUNCHED" notes below predate this fix.

## 5b. (superseded) Current state (live) — CNN RUN LAUNCHED

- **The A2 CNN run is TRAINING** → `cargo run -p cp-train --bin cnn_train -- --train`
  (defaults), `--out rust-trainer/checkpoints-cnn`, **warm-started from
  `checkpoints-cnn/distilled.json`** (SpatialNet 8010 params). 800 iters, 48 games
  (36 vs HARD), sims 64, exploration on — config matches the thin run for a **clean
  A/B isolating representation** (MLP+6 scalars → full CNN board vision).
- **Phases 1–5 DONE** (all AZ-only/additive, parity 8/8 untouched): `cnn.rs`,
  `planes.rs` (`PLANE_COUNT=14`, incl. terrain), `spatial_net.rs` (`SpatialNet`),
  `cnn_train.rs` (`--distill` + `--train`: own PUCT MCTS, self-play, benchmark,
  dashboard artifacts + `spatial.json`).
- **Distillation finding (important):** the teacher `sd-az-001` is **very passive**
  (~98 action decisions / 3675; picks Pass ~99 % of 2-candidate spots). So the
  warm-start is *passive-champion-equivalent* (top-1 agreement 0.98). This is the
  same warm-start the thin run used → the clean A/B holds. **The CNN trainer has NO
  potential-shaping** (unlike the thin run's `--combined-shaping`); value target is
  raw game-outcome z. If draws/passivity dominate, **add shaping** (next lever) or
  switch the teacher to **HARD-bot imitation** (active, strong — the documented
  fallback if the passive warm-start + RL stays passive).
- **Dashboard** (restarted on the new code) → serving `checkpoints-cnn` on
  **http://127.0.0.1:8787/**, with the new **CNN spatial heatmap panel** (policy /
  value toggle over the board); it populates after the first benchmark (iter 5).
  `npx vite-node training/serve-dashboard.ts -- --dir rust-trainer/checkpoints-cnn
  --port 8787`. record_replay stalemate-cut gap fixed.
- **Control** (thin scalar run) `checkpoints-az2` stopped at iter 180: peak **30 %**,
  plateau ~22 %, Pass ~58 % — the documented number the CNN must beat.
- **Baseline / teacher** registered `sd-az-001`; saved nets in `rust-trainer/saved-nets/`.
- **Judge by §3** after ≥2 h: win-rate trend vs the ~22–30 % control, Pass fraction
  (should drop if vision works), decisive cause split, and the spatial heatmap.
