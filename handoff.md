# Handoff — DAgger works; push it to the full army gate (Colonizing Pirkanmaa AI)

_Last updated 2026-06-07 after the DAgger build + a major measurement-bug discovery + the
Pass-collapse fix. **Read this first.** This supersedes the prior "START DAgger" handoff._

> Deployed game = TS/Phaser (`src/`). The **Rust trainer** (`rust-trainer/`) trains the
> AlphaZero net that is redeployed into the TS game. Parity Rust⇄TS is bit-exact, locked by
> `cargo run -p cp-train --bin parity --release` (must be **8/8**). Arc is now **sd3**.

---

## TL;DR — the one job

DAgger + RL produced the **first genuinely-measured competitive neural net**: `models/sd3/az/
sd3-az-002` (= `rust-trainer/checkpoints-cnn-dagger-rl1/champion-best.json`), **MCTS sims=64 vs
HARD: rawWin 0.633 / trueWin 0.567** — competitive with the `strong_army` yardstick (~0.52). DAgger
broke the policy **Pass-collapse** (96%→0.9% greedy); the KL-anchored RL fine-tune then lifted the
MCTS-deploy net to 0.633. **NOTE: every prior "0.55" was the sims=1 candidate-0 artifact, not a net
— sd3-az-002 is the real baseline now.**

**Still NOT at the full army gate** (peakSoldiers ≥1.5): ~41/60 games peak at *exactly 1 soldier*.
The net WANTS an army (HireSoldier is its top intent) and builds *some* outposts, but outposts/game
≈0.2 → most games have ZERO outposts → soldier cap stays 1, and on a ~1-mine economy it can't fund
both outposts and soldiers. **This is the `metal-economy-root-cause` wall — the sd2→sd3 rebalance was
incomplete.** The blocker is now FILLING the cap, not raising it.

**Current job: break the cap-fill wall.** In progress: an RL reward-tuning run (`checkpoints-cnn-
dagger-rl2`, init from sd3-az-002, `--w-army 0.4 --cap-potential 0.6`) to reward fielding soldiers
enough to overcome upkeep. If that still caps at maxSoldiers≈1, the real fix is a further
**metal/upkeep economy rebalance (arc sd3→sd4)** — game-rules change, re-tune league, re-export
goldens, re-run DAgger. Levers, in order:
1. **RL reward retune** (w-army / cap-potential) — cheap, stays in arc sd3 (CURRENT attempt).
2. **Multi-round DAgger with a strict ~300-step budget per round** — BUT aggregation inflates the
   step budget each round and re-creates the Pass attractor (round 2 → Pass 80%); you MUST scale
   epochs DOWN per round to hold steps ~300. Round 1 alone from the prior best (15 epochs, ~295
   steps) gave the best *greedy* net `checkpoints-cnn-dagger-r1best` (Pass 0.9%).
3. **Economy rebalance arc sd3→sd4** — the structural fix if RL can't fund the army.

---

## ⚠️ Two findings that rewrite the prior handoff (do not skip)

### 1. `mcts_select(sims=1)` is NET-INDEPENDENT — never validate with it
The prior handoff said "validate greedy = sims=1". **Wrong.** With `n_sims=1` the PUCT root has
0 edge-visits, so the U-term `prior·√(Σvisits)/(1+N)` is 0 for *every* edge → `chosen` always
falls to candidate 0, **regardless of the net's weights**. Proof: two differently-trained nets
gave bit-identical sims=1 benches. The real deploy uses **sims=64** (`TrainCfg::default`).
→ **Every "trueWin 0.55" number in the project history was the candidate-0 policy, not a net.**

**Validate the policy head honestly with:**
```bash
cnn_train --validate-net --greedy --init <net.json> --bench-games 80 --cap 150 --threads 16
```
`--greedy` = pure policy-head argmax (`net_greedy_choice`: `forward_board_scalars` →
`score_candidate_into` argmax, no MCTS, no value head). MCTS at sims=64 Pass-collapses while the
value head is weak, so it measures the value head, not the policy — `--greedy` is the gate.

### 2. The real wall was a Pass-collapse from a TRAIN/SERVE INPUT SKEW (now FIXED)
Measured honestly, the BC seed (`checkpoints-cnn-sup-p3/champion-supervised.json`) played
**Pass 96%**. Root cause (proven via `--diag-train`/`--diag-pass` in cnn_train.rs): the BC and
DAgger β-turn examples were encoded from states staffed by the **expert's** `staff_buildings`
(captured at `record_turn` phase-start, `hard_ai.rs:891`), but at play the **NN safety scaffold**
(`scaffold_ensure`/`scaffold_staff`, mirroring `NeuralAiController`) staffs workers differently →
the net trained on one input manifold, was evaluated on another. On its EXACT training states the
net Passes 4.8% (it fit fine); on its own play states it Passes 92.6%. Secondary cause:
over-convergence to a global Pass attractor (~680 steps → 94% Pass; ~260 steps → 33%; **sweet
spot ≈300 steps**). The value head was NOT the culprit.

**Fix (all training-only, parity 8/8):**
- `make_example_for_scaffolded` + the β-turn recorder now **scaffold-encode** states to match the
  deploy pipeline (the core fix).
- `--rollout-eps` (ε-random) + force-progress in `dagger_rollout_turn`: a collapsed net still
  generates diverse on-policy states for the expert to relabel (breaks the chicken-and-egg).
- **Policy-only training** path (`train_grad_policy_only_scalars`/`train_grad_cached_policy_only`
  in `spatial_net.rs` — additive, **forward inference UNCHANGED**, parity-neutral; default-on,
  re-enable value with `--dagger-train-value`) — removes value-head trunk interference for the
  noisy imitation z.

**Result (honest policy-greedy, 80 games vs HARD):**
| metric | BC seed | DAgger-fixed |
|---|---|---|
| Pass % | 96% | **27%** |
| top intent | Pass | **HireSoldier 33%** |
| outposts/game | 0.00 | 0.20 |
| peakSoldiers/game | 0.00 | 0.75 |
| trueWin | 0.04 | 0.19 |

Best nets: `rust-trainer/checkpoints-cnn-dagger-win/champion-dagger.json` and
`checkpoints-cnn-dagger-best/champion-dagger.json`.

---

## How DAgger is wired (cnn_train.rs `--dagger` mode)
Per round: (1) roll the current net out **net-greedy** (`dagger_rollout_turn`, with ε-exploration)
vs a league mix, with the **β-mix** (`dagger_play_one_game`) where the strong-army expert drives a
champ turn with prob β_i = `beta0·decay^(i-1)`; record every champ decision-state **scaffold-encoded**
and labelled by the expert (`expert_label` = `strong_army.record_turn`'s first intent) with
army-chain **boosts** (`push_boosted`: outpost/mine/hire ×N); (2) aggregate into D (optionally
seeded + boosted from the BC `dataset.json`); (3) retrain a fresh net (`train_dagger_net`,
policy-only by default); (4) **policy-greedy bench** vs HARD (`bench_net_greedy`).

A known-good command (tune toward the gate; keep ~300 steps/round → games×epochs/(D/batch)):
```bash
cd rust-trainer && ./target/release/cnn_train --dagger \
  --init checkpoints-cnn-sup-p3/champion-supervised.json \
  --seed-dataset checkpoints-cnn-sup-p3/dataset.json \
  --dagger-rounds 5 --dagger-games 300 --dagger-bench-games 60 \
  --dagger-beta0 0.6 --dagger-beta-decay 0.6 \
  --outpost-boost 8 --mine-boost 3 --hire-boost 1 \
  --pass-keep 0.10 --attack-keep 0.35 --rollout-eps 0.1 \
  --epochs 6 --batch 128 --lr 0.01 --cap 150 \
  --net-size small --threads 16 --out checkpoints-cnn-dagger-<name>
```
Full flag list: `cnn_train --dagger --help`.

## After the gate: the KL-anchored RL fine-tune (ALREADY BUILT)
Once the DAgger seed validates as an army-builder (`--validate-net --greedy`: outposts/game ≥0.3
AND peakSoldiers ≥1.5), RL-fine-tune anchored to it:
```bash
cd rust-trainer && RAYON_NUM_THREADS=16 ./target/release/cnn_train --train \
  --turn-search --turn-search-spend --net-size small \
  --init       checkpoints-cnn-dagger-<name>/champion-dagger.json \
  --kl-anchor 0.1 --kl-anchor-net checkpoints-cnn-dagger-<name>/champion-dagger.json \
  --income-lead-potential 0.3 --tile-potential 0.3 --w-cut 0.15 \
  --record-opp-value --device-potential 0.2 --device-credit 0.15 \
  --device-crack-credit 0.2 --hq-crack-credit 0.2 \
  --cap-potential 0.3 --w-army 0.15 --bankruptcy-discount 0.5 \
  --pfsp --script-opponents --script-frac 0.7 --tie-penalty 0.4 \
  --stall-rounds 80 --shape-gamma 0.99 --shape-weight 0.3 \
  --cap 150 --games 24 --bench-games 60 --threads 16 \
  --vs-hard-frac 0.3 --lr 0.003 --epochs 2 \
  --iters 200 --bench-every 5 --replay-every 25 \
  --out checkpoints-cnn-dagger-rl1
```
No `--kl-decay` flag exists. Watch **outpostsPerGame + maxSoldiersPerGame RISE**. Register a
baseline-beater: `npm run models -- register <champion-best.json> --arc sd3 --type az`.

---

## Build / gates / run
```bash
# Prereqs: Rust stable + Node 22 (.nvmrc). From repo root:
npm install
cd rust-trainer && cargo build -p cp-ai -p cp-train --release && cd ..

# Gates after ANY change:
cd rust-trainer && cargo run -p cp-train --bin parity --release   # MUST be 8/8 (DAgger is parity-free)
cargo test -p cp-ai --release                                     # ~74 pass; the ONLY allowed
  # failure is spatial_net::tests::forward_backward_equivalence_golden (pre-existing SIMD drift)
cd .. && npx tsc --noEmit                                         # exit 0
```

## Operational gotchas (learned the hard way)
- **NEVER `pkill -f`/`pgrep -f` with a pattern in the SAME shell command** — it self-matches and
  SIGTERMs the shell (exit 144). Kill training by PID via `ps -C cnn_train -o pid --no-headers`;
  kill the dashboard by port `lsof -ti:8787 | xargs kill`.
- **Keep only ONE 16-thread run live** (machine ~20 cores, **31 GB RAM**; the 1.7 GB BC dataset
  parses to ~3.4 GB, training clones per-batch so memory is fine — but don't load it twice).
- **Validate with `--validate-net --greedy`, never sims=1.** sims=1 is net-independent.
- `spatial_net::tests::forward_backward_equivalence_golden` can fail as pre-existing SIMD/target-cpu
  numeric drift — unrelated to logic; re-stamp if it's only drift.
- `src/ai/nn/weights.ts` is a stale placeholder; its TS arch test fails until a retrained champion
  is exported to TS (deploy step, separate from training).

## Key files
- DAgger + diagnostics + honest bench: `rust-trainer/crates/cp-train/src/bin/cnn_train.rs`
  (`run_dagger`, `dagger_play_one_game`, `dagger_rollout_turn`, `net_greedy_choice`,
  `bench_net_greedy`, `make_example_for_scaffolded`, `--diag-train`/`--diag-pass`,
  `--validate-net --greedy`). Expert hook: `hard_ai.rs::record_turn`.
- Policy-only training: `rust-trainer/crates/cp-ai/src/spatial_net.rs` (additive, parity-neutral).
- Parity-LOCKED (don't break): candidates.rs⇄candidates.ts, resources.rs⇄resources.ts, planes.rs,
  spatial_net.rs **architecture/forward** (the new training fns are fine).
- BC seed + dataset: `rust-trainer/checkpoints-cnn-sup-p3/{champion-supervised.json, dataset.json}`.
- Best DAgger nets: `rust-trainer/checkpoints-cnn-dagger-win/`, `checkpoints-cnn-dagger-best/`.
- Prior-best (pre-fix, sims=1-measured): `checkpoints-cnn-foundation0-prep6/champion-best.json`
  = `models/sd3/az/sd3-az-001` (its 0.55 was the candidate-0 artifact — treat with suspicion).

## Fallbacks / strategic notes
- **If multi-round DAgger caps below the army gate**: add the anti-Pass margin loss, then escalate
  to the KL-anchored RL fine-tune (above). If RL caps at the teacher (~0.52), the reserved lever is
  **PPO+GAE** for long-horizon credit on the Outpost→army payoff.
- **Strongest deployable opponent TODAY**: `HardAi::strong_army()` (beats HARD ~52%, parity-locked
  TS mirror) is shippable now. The neural effort is to *exceed* the scripts (research goal).
