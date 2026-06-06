# Handoff — Colonizing Pirkanmaa AI training (CONTINUE HERE)

_Last updated 2026-06-06. **Read this first.** Training is paused; continues on a new device.
Self-contained: everything needed is in this repo (design docs travel with the commit)._

The deployed game is **TS/Phaser** (`src/`). The **Rust trainer** (`rust-trainer/`) trains the
AlphaZero net deployed back into the TS game. Parity (Rust ⇄ TS) is bit-exact and locked by
`cargo run -p cp-train --bin parity --release` (must be 8/8).

---

## TL;DR — where we are

**Best result of the entire 14-run effort: `cnn-asym1`** (script-frac 1.0 + lr 0.003 + epochs 2) —
peak trueWin **0.52**, last-6-bench mean **0.44**. THE ONLY run that sustained trueWin > 0.50 across
multiple benches. champion-best.json preserved at `rust-trainer/checkpoints-cnn-asym1/`.

**The structural problem we've diagnosed across all 14 runs**: every run peaks at trueWin
0.43–0.53 around gen 10–25, then regresses. Eight major intervention categories have been tried
(reward shaping, terminal-z mods, action-space expansion, scripted opponents, KL anchors, buffer
size, learning rate, asymmetric self-play). The asymmetric-self-play fix (cnn-asym1) is the only
one to genuinely break the regression attractor, but it still hit the 0.45 ceiling.

**Two open structural gaps from the CLEAN-SLATE redesign memo (`TRAINING-V2-PROPOSAL.md`)**:
1. **`Intent::MarchSoldier` does not exist** — all soldier moves are adjacency-gated (Attack /
   CrackHQ / CrackDevice / Expand-1-tile). The MARCHER scripted opponent demonstrates marching,
   but the learner CANNOT copy it because the candidate enumerator never proposes "move soldier
   N tiles toward enemy". This is the load-bearing missing-action gap.
2. **CNN receptive field is 5×5 (small) / 7×7 (large) on a 14×12 board.** The trunk cannot
   spatially correlate "my soldier here" with "enemy device 10 tiles away" except via the global
   average-pool collapse. The user's question "miten näkee sen vaikka se olisi toisella puolella
   karttaa" has an architectural answer: only via `GlobalAvgPool`, not via the spatial trunk —
   the fix is precomputed distance-to-enemy-HQ/device planes.

---

## ★ START HERE NEXT SESSION ★

**Recommended next move (per `TRAINING-V2-PROPOSAL.md` §7):** implement supervised pretraining +
KL-anchored RL CORRECTLY (the prior attempt failed due to a dominant-intent diff-heuristic bug
that always recorded `Pass` as the target). The full design is in §6 of `TRAINING-V2-PROPOSAL.md`.

**Conservative alternative (cheaper, ~5h):** resume `cnn-asym1`'s setup (already proven best),
plus add the `Intent::MarchSoldier` candidate enumerator and the distance-to-HQ/device planes.
This is the smallest set of changes that addresses the two structural gaps identified above.

**Most-conservative alternative (just resume what worked):** `cnn-asym2` = same flags as asym1,
warmstart from `rust-trainer/checkpoints-cnn-asym1/champion-best.json`. Lets the asymmetric
attractor train for longer.

Launch command for the most-conservative path (asym2):
```bash
./rust-trainer/presets/launch.sh \
  --out rust-trainer/checkpoints-cnn-asym2 \
  --init rust-trainer/checkpoints-cnn-asym1/champion-best.json \
  --iters 300 --bench-every 5 --replay-every 25 \
  --script-frac 1.0 --lr 0.003 --epochs 2 \
  --vs-hard-frac 0.4 \
  --w-army 0.4 --w-expert 0.15 --w-soldier-forward 0.3 \
  --cap-potential 0.3 --idle-flow-penalty 0.3 \
  --device-crack-credit 0.25 --hq-crack-credit 0.25 \
  --turn-search-spend --build-prior-floor 0.06 --sims 64
```

---

## What this session shipped (code, parity-locked)

All changes parity 8/8, no arc bump (game-rules unchanged). Goldens re-exported once when
intents changed (12 → 15).

### Action space (cp-ai/candidates.rs, src/ai/nn/candidates.ts — mirrored)
- **`Intent::BuildBridge`** (idx 12) — Bridge candidate enumerator, gated on owned river + cost
- **`Intent::CrackDevice`** (idx 13) — enumerates when enemy device reachable
- **`Intent::CrackHQ`** (idx 14) — enumerates when enemy un-conquered HQ reachable
- `INTENT_COUNT` 12 → 15; policy-head dim 64 → 67 (cold-start required for nets predating Plan-B)

### Scripted opponents (cp-ai/hard_ai.rs)
The 6-way scripted pool (was 3-way pre-Plan-B):
- **`HQ_RUSH_PARAMS`** (Plan-B addition) — directed HQ-attacks
- **`GARRISON_PARAMS`** — defensive turtle (warmonger=true forces at_war round 1)
- **`EXPERT_PARAMS`** — pure-econ Expert-stacking bot
- **`MARCHER_PARAMS`** — preemptive march-to-enemy-HQ (with bespoke `march_to_enemy_hq` phase
  gated on warmonger; ARMY_RUSH-like AiParams but marches soldiers each turn when no Attack
  is legal)
- Tuned `GARRISON` (reserve 100→300, max_outposts 4→2) and `EXPERT` (reserve 140→200,
  max_outposts 2→1) to keep self-bankruptcy ≤ 5% across 20-game smokes per variant.

### HARD policy fixes (cp-ai/hard_ai.rs)
- **`affordable_after_commit` helper** + drain-vs-income checks on every expensive build/hire
  (Outpost, Soldier×2, Expert, Village). Catches the slow-drain bankruptcy the original
  Device-only safety-buffer missed.
- **`build_bridges` phase**: HARD now builds Bridges on owned river tiles when they would
  unlock ≥1 new neutral tile. Prefers Hydro over Bridge when `experts && nuclear && round > 30`.
- **`claim_value` priority**: HARD prefers Expand-targets that contain a neutral building
  (Mine=7, Mikontalo/Nuclear/Village/Outpost=6, Hydro/Farm=5, Bridge=4) over bare terrain.

### Reward / training-config flags (cp-train/cnn_train.rs)
- **`--w-soldier-forward <f64>`** — Φ term rewarding own soldiers' position-near-enemy-frontier
  (gradient pulling the army forward). `clamp01(Σ(1 − dist/(W+H)) / 7)` × w.
- **`--w-expert <f64>`** — Φ term rewarding staffed Experts on Mine/Hydro/Nuclear.
- **`--bankruptcy-discount <d>`** (Plan-B EXPANDED scope): when winning by Bankruptcy OR
  Conquest with no Outpost built AND peak-soldier < 2, scale terminal z by (1−d).
  Default 0.0 = exact no-op.
- **`--device-crack-credit <c>` / `--hq-crack-credit <c>`** — action-level credit for choosing
  `CrackDevice` / `CrackHQ` intents in a winning trajectory. Default 0.0 = no-op.
- **`--kl-anchor <w> --kl-anchor-net <path>`** — adds `w·KL(π_current || π_anchor)` to policy
  loss. Anchor net is frozen, loaded once. Default 0.0 = no-op.
- **`--supervised-from-hard` / `--supervised`** — supervised-data-gen + supervised-training
  modes. **Currently buggy** (dominant-intent diff-heuristic always falls through to Pass
  because HARD's plan_turn is opaque — see V2 memo §6). Needs per-action HardAi refactor.

### Trainer / scripted dispatch (cnn_train.rs)
- `ScriptKind` extended to 6 variants (HqRush, GarrisonFortress, EconExpert, Marcher added).
- `do_replay` writes 5 games per scripted opponent (was 1), one game per variant becoming 5
  per variant for variance visibility.
- New behavioural metrics in `benchmark-history.jsonl`: `bridgesPerGame`,
  `crackDeviceAttempts/Successes`, `crackHQAttempts/Successes`, `champSoldierBins`, plus
  M1-M9 (unit/soldier-efficiency, win-by-villages/outposts, contact-rate, expert-hires,
  frontier-ratio, rounds-by-outcome).
- Per-iter log adds `spVsGarrison`, `spVsExpert`, `spVsHqRush`, `spVsMarcher`, `spContactRate`.

### Dashboard (training/serve-dashboard.ts)
- 8-button replay viewer: hard, self, vs-armyrush/hqrush/devicerush/garrison/expert/**marcher**.
- 5 games per scripted opponent in the viewer's batch selector.
- New panels: Plan-B intent activity (bridges/crackHQ/crackDevice over time + bar comparison),
  M1-M9 behavioural diagnostics (USEFUL-vs-USELESS unit/soldier bars, contact-vs-no-contact
  count, peak-soldiers distribution histogram, win-by-villages/outposts bars).
- Bridge → 'B' glyph in replay frame decoder (was '?'). `building_code` made exhaustive in
  both cnn_train.rs and alphazero.rs so a future BuildingType variant triggers a compile error.

### Preset system (rust-trainer/presets/)
- `common.sh` — canonical fixed flag set. Per-experiment knobs stripped (--out, --iters,
  --vs-hard-frac, --bankruptcy-discount, --script-frac, --w-army, --w-expert, --w-soldier-forward,
  --cap-potential, --idle-flow-penalty, --build-prior-floor, --sims, --net-size). Always strip
  a knob from common.sh BEFORE sweeping it — `arg_val` uses FIRST occurrence.
- `mac-m2.sh` — `THREADS = perf_cores` (8 on M2 Pro; empirical: 66 s/iter at 8 threads beats
  75 s at 6 — E-core-drag theory falsified).
- `linux-pc.sh` — `THREADS = max(16, nproc − 4)`, override via `THREADS_OVERRIDE=N`.
- `launch.sh` — auto-detects OS, sources right preset, supports `--print-cmd` dry-run.

---

## The empirical record — 14 runs

| Run | Intervention | Peak (gen) | Last-N mean | Notes |
|----:|---|---:|---:|---|
| b1 | Outpost cost rebalance (arc sd2) | 0.45 (30) | 0.41 | First post-fix run |
| i1 | reward retune (idle-flow 0.3→0.05) | 0.47 (15) | 0.40 | refuted: idle-flow not binding |
| s3 | growth-Φ retune | 0.43 (15) | 0.37 | small regression |
| c1 | turn-search-spend ON | 0.40 (25) | 0.32 | Pass% blew up to 53.7% |
| bc1 | --bankruptcy-discount 0.7 | 0.38 (25) | 0.33 | discount too strong; net stalled |
| bc2 | --bankruptcy-discount 0.4 | 0.40 (30) | 0.36 | bankShare moved, trueWin flat |
| r1 | Plan-B: BuildBridge/CrackHQ/CrackDevice + HQ_RUSH | 0.45 (15) | 0.34 | CrackHQ 18/18 became HARD-overfit |
| r2 | Plan-B + KL 0.3 + GARRISON/EXPERT + 400 iters | 0.45 (10) | **0.20** | KL pinned policy; massive regression |
| r3 | full HARD-fix v1 + MARCHER + Φ-forward | 0.42 (10) | 0.35 | old binary in-memory; partial fix |
| r4 | full HARD-fix v2 + variant tunes + clean run | **0.53 (10)** | 0.25 | highest-ever PEAK; deep regression to 0.13 |
| r5 | --buffer 60000 → 15000 | 0.43 (0) | 0.30 | faster cycling → more unstable, not less |
| kl1 | KL anchor 0.3 to r1 gen-15 net | 0.45 (15) | 0.34 | anchor pinned policy at warmstart |
| **asym1** | **--script-frac 1.0 + lr 0.003 + epochs 2** | **0.52 (0)** | **0.44** | **only run to sustain >0.50** |
| aggro1 | asym1 + --w-soldier-forward 0.3 → 1.2 | 0.52 (0) | (paused gen 5) | early bimodal "all-or-nothing" army |

`b1 mean 0.41` was the previous high-water last-N mean. asym1's 0.44 over 6 benches is the new
best. **Every other run regressed below its own warmstart.** Two of the most-comprehensive runs
(r2, r4) regressed catastrophically (-0.25 to -0.40 from peak).

---

## Key design documents (READ THESE)

**Primary** (the V2 redesign — read first):
- `rust-trainer/TRAINING-V2-PROPOSAL.md` — clean-slate redesign across eyes/actions/reward/curriculum,
  ~3700 words. Identifies the two structural gaps (MarchSoldier intent, receptive field) and the
  proposed §7 experiment (supervised + KL-anchor, correctly this time).

**Supporting analyses (chronological)**:
- `rust-trainer/META-ANALYSIS.md` — the 8-run forensic that identified the buffer-cycling
  positive feedback loop as root cause (later: data-quality fixes raised the peak but didn't
  break the loop).
- `rust-trainer/DEEP-REDESIGN-MEMO.md` — the Plan-B action-space proposal (BuildBridge,
  CrackDevice, CrackHQ).
- `rust-trainer/REWARD-FIX-PROPOSAL.md` — terminal-z bankruptcy-coupon diagnosis (led to
  --bankruptcy-discount).
- `rust-trainer/SEARCH-CURRICULUM-FIX-PROPOSAL.md` — turn-search-spend + build-prior-floor.
- `rust-trainer/OVERNIGHT-RUN-PLAN.md` — the comprehensive r2 design (and its failure).
- `rust-trainer/GAME-MECHANICS.md` — USER-VERIFIED canonical game rules.

---

## Diagnoses that have been refuted (don't re-litigate)

- **Net capacity is NOT the binding constraint** — same 9786-param net hit maxSold 0.92 in b1
  and trueWin 0.53 in r4. The architecture represents winning; training degrades it.
- **The bankruptcy mirage was real but NOT the root cause** — bc1/bc2/r4 stripped the coupon
  with no net trueWin lift; Ng-1999 invariance limits Φ-shaping.
- **Action-space gap was real for BuildBridge/CrackHQ/CrackDevice but did NOT lift trueWin** —
  r1 added the intents; CrackHQ became a HARD-loose-garrison exploit (18/18 success),
  bridges = 0.01/g.
- **Buffer-cycling was suspected but the fix (60k → 15k) made things WORSE** — r5 collapsed
  faster than r4. Buffer size is NOT the binding lever.
- **HARD's self-bankruptcies were CORRUPTING training data** — fixed via affordability gates
  (HARD now self-bankrupts ≤ 5% per variant). r4's gen-0 trueWin jumped to 0.45 just from
  fixed HARD. But that didn't stop the regression.

---

## What WORKS (don't break these)

- **`cnn-asym1` config**: `--script-frac 1.0 --lr 0.003 --epochs 2`, warmstart from r4 peak.
  Only run to sustain >0.50.
- **HARD-fix v2** (affordability gates) — cleaner bench signal.
- **Plan-B intents** (BuildBridge, CrackDevice, CrackHQ) — present and used; CrackHQ in
  particular is a real cracker action.
- **The 6-way scripted pool** — diverse opponent set, --script-grade auto-balances by win-rate.
- **MARCHER scripted opponent** — demonstrates preemptive march (even though the learner
  can't copy it without `Intent::MarchSoldier`).
- **Dashboard 8-button replay viewer** — visualises per-opponent behaviour with 5 games each.

---

## Operational notes for resuming on the new device

```bash
# Prereqs: Rust (stable) + Node 22 (.nvmrc).
npm install
cd rust-trainer && cargo build --release
cd ..

# Gates (run after any change):
cargo run -p cp-train --bin parity --release       # MUST be 8/8
cargo test -p cp-ai --release                       # ~70 tests
cargo test -p cp-train --bin cnn_train --release    # ~62 tests
npx tsc --noEmit                                    # exit 0

# Dashboard:
npx vite-node training/serve-dashboard.ts -- --dir rust-trainer/checkpoints-cnn-<run> --port 8787

# Resume training (see "★ START HERE NEXT SESSION ★" above for launch commands)
```

The Linux preset uses `THREADS_OVERRIDE` env var if you need to pin a thread count
(default is `max(16, nproc−4)`). On the Mac (M2 Pro 8P+4E) the preset auto-detects 8 threads.

---

## What's parity-affecting vs free-to-edit

**Parity-affecting (must mirror Rust ⇄ TS + re-export goldens + parity 8/8):**
- `crates/cp-ai/src/candidates.rs` ⇄ `src/ai/nn/candidates.ts`
- `crates/cp-sim/src/resources.rs` ⇄ `src/core/resources.ts`
- `crates/cp-sim/src/managers.rs` (game-engine; only changed for Bridge support which was
  already in place pre-session)

**Free to edit (no parity impact):**
- `crates/cp-ai/src/hard_ai.rs` — HARD's policy is not in golden traces; modify freely.
- `crates/cp-train/src/bin/cnn_train.rs` — Φ shaping, MCTS, training loop, replay-recording,
  metric instrumentation. Parity-free.
- `crates/cp-ai/src/planes.rs` — if you add new planes you need cold-start (PLANE_COUNT
  change → policy/value-head input dim changes).
- `crates/cp-ai/src/spatial_net.rs` — net arch. Cold-start required for shape changes.
- `training/serve-dashboard.ts` — dashboard only.

---

## Known bugs / partial work

- **`--supervised-from-hard` mode is broken**: the dominant-intent diff-heuristic always
  records `Pass` as the target because HardAi's `plan_turn` is opaque and the diff-window
  spans the whole turn. Fix requires refactoring HardAi to expose per-action intent sequence,
  OR inserting recording hooks inside plan_turn. See V2 memo §6 for the cleanest path.
- **`--kl-anchor` works** but the only test (`cnn-kl1` with anchor=0.3) showed the anchor pins
  the policy at the anchor net's level — no progress, no regression. Worth re-testing at
  anchor=0.05 (lighter) once supervised pretraining works.

---

## Models / checkpoint pointers

| Checkpoint | What it is | Use it for |
|---|---|---|
| `rust-trainer/checkpoints-cnn-asym1/champion-best.json` | Best policy of the session (trueWin 0.52 peak) | Warmstart for any new run |
| `rust-trainer/checkpoints-cnn-r4/champion-best.json` | All-time peak gen-10 net (trueWin 0.53) | Alternative warmstart |
| `rust-trainer/checkpoints-cnn-r1/champion-best.json` | First Plan-B 15-intent peak | Anchor for KL experiments |
| `rust-trainer/checkpoints-cnn-b1/champion-best.json` | Pre-Plan-B reference (12 intents — INCOMPATIBLE with current 15-intent net) | NOT compatible — do not load |
