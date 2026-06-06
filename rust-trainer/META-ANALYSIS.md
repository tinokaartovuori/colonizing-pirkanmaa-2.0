# META-ANALYSIS — eight runs, one ceiling, one root cause

_Authored 2026-06-05. Forensic re-aggregation of all 8 cnn-runs directly from
`checkpoints-cnn-{b1,i1,s3,c1,bc1,bc2,r1,r2}/benchmark-history.jsonl` (98 bench
rows, ~5,880 evaluation games) + a fresh read of the trainer
(`cnn_train.rs`, 8021 LOC) and net (`spatial_net.rs`, 1684 LOC; eyes
`planes.rs`, 651 LOC; action space `candidates.rs`, 1659 LOC). Prior memos are
tactical: this one questions architectural assumptions. The user has asked the
larger question — what is fundamentally wrong, and what strategic change is
needed? — and the data answers it._

---

## §1 — What went wrong (per run)

Aggregated from the JSONL directly. `tW` = `trueWinVsHard` last-N mean
(N=5; N=10 for r2). `Peak` = best single bench in the run.

| run | gens | tW last-N | tW peak | peak gen | regress (peak−final) | Pass% | maxSold |
|----:|:----:|---------:|--------:|---------:|---------------------:|------:|--------:|
| b1  | 0-49 | 0.410    | 0.450   | 30       | 0.040 | 33.8% | 0.76 |
| i1  | 0-49 | 0.390    | 0.467   | 15       | 0.077 | 38.9% | 0.75 |
| s3  | 0-49 | 0.370    | 0.433   | 15       | 0.063 | 40.1% | 0.63 |
| c1  | 0-35 | 0.327    | 0.400   | 25       | 0.073 | 53.7% | 0.61 |
| bc1 | 0-30 | 0.337    | 0.383   | 25       | 0.047 | 47.9% | 0.64 |
| bc2 | 0-35 | 0.360    | 0.400   | 30       | 0.040 | 48.0% | 0.64 |
| r1  | 0-49 | 0.343    | 0.450   | 15       | 0.107 | 18.0% | 0.62 |
| r2  | 0-300| **0.202** | 0.450  | 10       | **0.237** | 57.8% | 0.54 |

**b1** — *Intervention:* the Outpost cost rebalance (300 metal → 100 metal,
arc `sd`→`sd2`) opens the army gate. *Gate:* `maxSold > 3` sustained.
*Reality:* maxSold transiently 0.92 @ gen 30, drifted to 0.57; outposts
~0.20/g (vs ~0.10 prior). **Gate opened but cap never filled; peaked, drifted.**

**i1** — *Intervention:* retune `--w-army --cap-potential --idle-flow-penalty`.
*Gate:* trueWin ≥ 0.50. *Reality:* peaked 0.467 @ gen 15, monotone regress to
0.367; maxSold flat ~0.75. **Reward tuning could not push past 0.45.**

**s3** — *Intervention:* growth-Φ retune (`--income-lead-potential 0.5
--tile-potential 0.4`). *Reality:* peaked 0.433 @ gen 15, regressed to 0.37;
maxSold *fell* 0.75 → 0.63. **Larger growth weights slightly hurt the army.**

**c1** — *Intervention:* `--turn-search-spend ON` to stop MCTS rollouts
truncating on the first greedy Pass. *Gate:* Pass% < 30 + trueWin > b1.
*Reality:* **Pass% exploded to 53.7%** (opposite of intended); peaked 0.40 @
gen 25, regressed. The un-trained value head scores non-Pass alternatives
slightly worse, so spend-mode probes Pass-alternatives and *confirms* Pass.

**bc1** — *Intervention:* `--bankruptcy-discount 0.7` strips the opp-bankruptcy
+1. *Reality:* peaked 0.383 @ gen 25, bankShare *rose* to 0.227; **honest
baseline did not lift**. Net change inside 60-game CI (±12.6%).

**bc2** — *Intervention:* re-tuned `--bankruptcy-discount` + expanded scope
(catches low-army Conquest wins). *Reality:* peaked 0.40 @ gen 30; bankShare
0.155 (the only metric that responded); trueWin flat. **Wins shifted
category, ceiling unchanged.**

**r1** — *Intervention (DEEP-REDESIGN Plan-B):* add `Intent::BuildBridge` +
`CrackDevice` + `CrackHQ` (12→15 intents), action-space gap. *Gate:* trueWin
≥ 0.51, bridges/g ≥ 0.3, deviceDenial ≥ 0.45, hardDeviceShare < 0.18.
*Reality:* peaked 0.45 @ gen 15 (matched b1's peak), regressed to 0.343;
**bridges/g = 0.010, ~70× short**; crackDevice attempts = 0; **CrackHQ 18
attempts/18 successes per bench — the new intent fired ONLY for the easy
1-soldier rush of HARD's loose pre-contact garrison**.

**r2** — *Intervention (OVERNIGHT-RUN):* everything stacked +
`--bankruptcy-discount 0.7` + `--vs-hard-frac 0.4→0.2` + GARRISON_PARAMS +
EXPERT_PARAMS + `--w-expert 0.15` + 400 iters. *Gate:* trueWin ≥ 0.50.
*Reality:* **trueWin 0.45 @ gen 10 → 0.202 last-10 mean → 0.183 last bench**
(0.27 point collapse over 290 iters); **Pass% climbed monotonically 9.7% →
63.6%**; bankShare *rose* to 0.31 despite 0.7 discount; CrackDevice attempts
spiked to 37 @ gen 260 with only 2 successes (policy tries to crack but
cap=1 means it can't); experts/g = 0.018 (zero despite EXPERT_PARAMS +
w-expert). **300 iters with every prior intervention stacked ended WORSE
than the warmstart-net itself.**

### The cross-run pattern (the smoking gun)

Every single run peaks at gen 10-30 with `tW ≈ 0.43-0.47`, then regresses by
0.04-0.24. **The peak is always achieved by the warmstart net + the first
~few thousand new self-play examples — i.e., before the replay buffer has
fully turned over** (buffer cap 60k, ~2,300 examples/iter at games=24 means
~26 iters to fill, and ~26 more to fully cycle once). The structural
suspicion this raises: it is **not the warmstart net that is good and the
training that is bad** — the warmstart is just one specific behavioural mix
(seeded from the prior run's last champion), and the *training process
itself* moves AWAY from whatever local maximum that seed was at. This is the
hint §2 must explain.

---

## §2 — Why the AI doesn't learn the game mechanics

The user observes five specific skill failures. For each I walk through
candidate explanations and check them against ALL 8 runs.

### 2.1 Bridge — `bridgesPerGame = 0.022` in r1 despite the new intent

The action exists (`candidates.rs:43`, `Intent::BuildBridge`). bridges/g
went 0.010 (r1) → 0.032 (r2) — the trainer is generating ~6 visits on a
Bridge candidate (64 sims × 0.06 floor) but the *value head never sees a
payoff*: bridge utility is 5-15 turns out, and the self-play opponent ends
games via 1-rush before river-bypassing matters. **No opponent in the
curriculum makes the Bridge necessary.** Refutes the
"action-space-was-the-gap" hypothesis: opening the action without
providing payoff data does nothing.

### 2.2 Outpost-then-army — BuildOutpost oscillates 4-16 across r1 iters

- **Net capacity.** REFUTED: same 9786-param net hit maxSold 0.92 @ b1-gen
  30, 0.83 @ r1-gen 15 — the architecture *represents* "build Outposts and
  fill them." The peaks just don't persist.
- **Search depth.** PARTIAL: 64 sims × 0.06 floor ≈ 4 visits on Outpost,
  enough to build (Outposts 0.4/g @ r2-gen 10) — then collapse to 0.02/g
  @ r2-gen 270. Search depth doesn't explain the **collapse-after-peak**.
- **Reward collision.** REFUTED: `idle_flow_penalty` (`cnn_train.rs:2729-
  2772`) keys on unstaffed units + idle money, not empty slots — a fresh
  Outpost adds 0 idle (test-locked).
- **Self-play attractor.** The 1-soldier-rush works against both HARD's
  loose pre-contact HQ AND the mirror-passive self-twin. Building an
  Outpost is a 100-metal investment whose payoff is "you can field more
  soldiers" — but the policy already wins ~40% of games WITHOUT them.

### 2.3 Expert hiring — ~0 across all 8 runs even with `--w-expert`+EXPERT_PARAMS

- **Gate.** `StackProducer` (the only Expert candidate, `candidates.rs:
  1264-1309`) requires `free_unit_amount > 1`, which requires a Village.
  Villages are 0.6-0.9/g and fill first with Workers. Mechanically gated.
- **Curriculum failure.** r2 added EXPERT_PARAMS; the log shows
  `spVsExpert` ≈ 0 or null in nearly every iter — the learner *lost
  every* Expert-bucket game. The curriculum signal was a uniform −1, so
  the policy learned "don't engage this opponent" not "stack Experts."

### 2.4 Directed offence — 1-rush equilibrium, no army-conquest

**r1's CrackHQ 18 attempts / 18 successes per bench is the smoking gun.**
HARD's HQ holds 0-1 defenders pre-`at_war` (`hard_ai.rs`
`should_militarise()`). The trainer manufactures this exploit and rewards
it ~17/60 × +1 per bench. r2's GARRISON opponent was supposed to close it,
but it played `script_frac 0.5 × (1 − vs_hard_frac 0.2) / 5 buckets ≈ 8%`
of games — too small to override the vs-HARD signal.

### 2.5 Self-defence — `hardWins.conquest` rose r1 → r2 (72 → 91 / 5 benches)

As Pass% climbed, the rare attack-decision exposes the HQ (cap=1 means the
1 soldier is staged on the enemy tile, HQ undefended). The **eyes** see
threat (`C_ENEMY_REACH`, `C_ENEMY_BUDGET`); the policy's only offensive
move at cap=1 *mechanically requires* leaving HQ undefended.

### Common cause

The five skill failures are not five separate problems. **They are one
problem:** the policy converges to a stable, cheap, locally-optimal exploit
(1-soldier-rush HARD's loose pre-contact HQ) because (i) the vs-HARD
bucket teaches it; (ii) the self-twin doesn't defend either; (iii)
Ng-1999 — Φ cannot redirect a terminal label that already says "1-rush =
+1"; (iv) the 60k replay buffer enshrines it once entered, because the
policy generates exploit-heavy data → trains on exploit-heavy data → emits
*more* exploit-heavy data. **The "missing skills" are missing because
their trajectories are absent from the buffer.**

---

## §3 — The fundamental binding constraint

**Root cause: the training data distribution is dominated by a single
exploit (1-soldier-rush HARD's loose pre-contact HQ) that the curriculum
itself manufactures, and a 60k self-cycling replay buffer cannot escape
it. The trainer is correctly fitting that distribution.** Every prior
memo blamed Φ or action-space; the actual lever is the **data the value
head learns from**. Pick (E) **bench-vs-HARD overfit** as the headline,
with (F) **buffer-cycling positive feedback** as the mechanism that locks
it in.

**Justification:**

1. **r1's CrackHQ 18/18 stickiness is a deterministic exploit, not noise.**
   The net learned a near-optimal response to HARD's specific weakness in
   `should_militarise()`. This *is* opponent-overfit (E).

2. **r2 rules out reward shape.** 300 iters with `--cap-potential 0.3`,
   `--w-army 0.4`, `--w-expert 0.15`, `--idle-flow-penalty 0.3`, *and*
   `--bankruptcy-discount 0.7` produced **MORE** passivity (Pass 9.7% →
   63.6%). Ng-1999: Φ cannot create activity the terminal label doesn't
   reward — and the terminal label says "1-rush = +1, anything else =
   coin flip."

3. **r2's CrackDevice 37 attempts / 2 successes shows the policy *trying*
   what shaping rewarded and *failing* mechanically.** The value head
   learned "cracking is good"; the policy can't execute (cap=1, no
   soldier to spare). The conditional "if cap > 1, then crack" is empty
   in the buffer — so the policy cannot learn the *precondition*.

4. **r2 buffer dynamics fingerprint the trap.** `bufferSize` saturates 60k
   at gen ~25, then is a sliding window of the last ~26 iters' self-play.
   After gen 25, the policy at gen 200 trains on data from iters 174-200,
   which it *itself* generated. **This is a positive feedback loop**:
   exploit-heavy policy → exploit-heavy data → more exploit-heavy policy.
   Once entered, the loop cannot exit without an external data source.

5. **The peak-then-regress pattern in all 8 runs is the universal
   signature of (4).** The warmstart carries residual diversity from
   prior training; the first ~25 iters preserve it while the buffer
   fills. Once the buffer is fully cycled, the policy locks onto the
   exploit. **All 8 runs share this shape because the buffer-cycling
   dynamics are the same in all 8.**

**Refuting other candidates:**

- (A) **Net capacity**: REFUTED — the same 9786-param net hit 0.467 @
  i1-gen 15 *and* 0.18 @ r2-gen 300. The architecture represents the win;
  it just doesn't stay at it. Capacity does not bind on a representation
  already demonstrated.
- (B) **Self-play Nash trap as mechanical truth**: PARTIALLY correct
  but downstream of (E)+(F); the game also admits multi-army conquest,
  the trainer just doesn't generate those trajectories.
- (C) **Terminal reward ±1 too coarse**: REFUTED — denser shaping (Φ,
  device-credit, hq-crack-credit) had no effect.
- (D) **Self-play is the wrong paradigm**: this is the *solution* to
  (E)+(F), not the diagnosis.
- (F-as-iters) **Training horizon**: REFUTED — r2 ran 300 iters and got
  worse. More compute on this paradigm makes it worse, not better.

---

## §4 — What strategic changes are needed

If the root cause is **the trainer is correctly fitting a corrupted data
distribution**, then the lever must change the **data distribution**, not
the reward, the architecture, or the search. Three strategic changes,
ranked by leverage / risk / cost:

### Proposal 1 (RECOMMENDED) — Imitation from HARD-army-rush + KL-anchored RL

What AlphaStar did (DEEP-REDESIGN §4.3). Concretely:

1. **Generate ~50k (state, intent, action) tuples** from HARD-vs-HARD with
   `ARMY_RUSH_PARAMS` on the learner seat. One-shot, ~30 min on M2.
2. **Supervised pretrain**: 10 epochs of cross-entropy on intent + MSE on
   (won/lost). No MCTS, no Φ. The net then plays army-conquest because
   it *copies HARD-army verbatim*.
3. **Self-play RL** uses this net as warmstart but adds a **KL-divergence
   anchor** `λ · KL(π_current || π_supervised)` so the policy cannot drift
   far from the demonstrations (AlphaStar: λ=0.5 → 0.1 decay).
4. **vs-HARD is removed from training entirely** (benchmark only). With
   KL anchor, the 1-rush exploit is *far in KL* from the army-rush
   teacher and so cannot be reached.

**Cost:** ~400 LOC, ~1 day. **Leverage:** very high — directly attacks the
data-distribution root cause. **Risk:** army-rush itself has weaknesses
(loses to device-rush); KL must be loose enough to allow refinement.

### Proposal 2 — Curriculum-only training, slow self-play introduction

Iters 0-30: **0% self-play**, 100% rotating scripted opponents
(army-rush, device-rush, garrison, expert, turtle). Iters 30-60: 25%
frozen past-self via PFSP. Iters 60+: self-play fraction climbs linearly.
**vs-HARD is benchmark-only.** Sidesteps the Nash collapse by ensuring
the policy has a robust prior *before* it mirrors itself.

**Cost:** ~100 LOC. **Leverage:** medium-high. **Risk:** scripted bots may
have their own narrow Nash → learner inherits it.

### Proposal 3 — Game-rule deviation: HQ permanent garrison of 2

The 1-rush exploit exists because HARD's HQ holds 0-1 defenders
pre-contact. *Training-only* rule change: HQ permanently provides 2
passive defenders that count for the strict-`>` resolution. Analogous to
the Mine/Hydro/Nuclear and Outpost-cost rebalances — `reference/` is
no longer canonical for this mechanic. Arc bumps `sd2` → `sd3`.

**Cost:** ~150 LOC + parity 8/8 + golden re-export. **Leverage:** medium —
closes one exploit, may shift to another. **Risk:** highest — parity /
model-management churn, may just relocate the attractor.

### Recommendation

**Primary: Proposal 1.** Biggest lever, published precedent (AlphaStar
Grandmaster), reasonable cost. **No prior run has attempted supervised
pretraining** — every one of the 8 runs assumed self-play would
discover army-conquest from random init; six weeks say it won't.
Proposal 2 is the fallback; Proposal 3 is the last resort.

---

## §5 — The concrete next experiment

**Hypothesis.** A policy that begins life *already playing army-conquest*
(from supervised cloning of HARD with ARMY_RUSH_PARAMS) and is then refined
by self-play with a **KL anchor to the supervised baseline** will (a) NOT
regress to the 1-soldier-rush exploit because the exploit is far in KL from
the army-rush demonstrations, (b) demonstrate `maxSoldiersPerGame ≥ 3.0`
sustained over 50 iters, and (c) reach `trueWinVsHard ≥ 0.55` because it
has a *correct* baseline strategy to refine, not a *broken* one to
escape.

**Configuration / code changes.**

1. New binary `cnn_train --supervised-from-hard`: plays ~2,000 HARD-vs-HARD
   games with ARMY_RUSH_PARAMS on both seats (~30 min on M2 Pro), records
   every (state, intent, action, won/lost) tuple. Output:
   `rust-trainer/checkpoints-cnn-sup1/dataset.bin` (~50k examples).
2. New mode `cnn_train --supervised --epochs 10`: trains the small net on
   that dataset, cross-entropy on intent, MSE on z. Output:
   `rust-trainer/checkpoints-cnn-sup1/champion-supervised.json`.
3. Modified `cnn_train --train`: new `--kl-anchor <weight> --kl-anchor-net
   <path>` flags. When set, adds `weight · KL(softmax(policy_logits) ||
   softmax(anchor_policy_logits))` to the policy loss; anchor net is loaded
   once and used to predict on each batch. Default 0 = bit-identical no-op.
4. Launch: `./cnn_train --train --net-size small
   --init rust-trainer/checkpoints-cnn-sup1/champion-supervised.json
   --kl-anchor 0.3 --kl-anchor-net rust-trainer/checkpoints-cnn-sup1/champion-supervised.json
   --vs-hard-frac 0.0 --script-frac 0.5 --pfsp --iters 100 --bench-every 5`.

**Gate (last-10-bench means, gens 50-100, 600 games).**
- PASS: `trueWinVsHard ≥ 0.55` AND `maxSoldiersPerGame ≥ 3.0` AND
  Pass% < 25% AND no regression beyond 0.05 over the last 30 iters.
- FAIL: `trueWinVsHard < 0.40` after gen 30 — supervised baseline doesn't
  hold under self-play; weakens to Proposal 2.

**Expected wall-clock.** Supervised data gen ~30 min + 10 epochs train
~10 min + 100 iters @ 90 s/iter (small net) ≈ 3.5 h. Total: ~4 h.

**What it teaches us regardless of outcome.**
- If PASS: data distribution was the root cause; supervised+RL is the
  paradigm and we proceed to refining the supervised teacher (better
  scripted opponents, multi-teacher blends).
- If FAIL: rules out *both* "self-play discovers army" *and* "KL-anchored
  cloning preserves army" — the next question is whether the small net has
  *capacity* to represent army-conquest stably (revisit Proposal 1 with
  large net), or whether the game's reward structure fundamentally rewards
  1-soldier-rush even when the policy starts elsewhere (then Proposal 3 is
  the only remaining path). Either way, the experiment **separates the
  "doesn't know" from "won't do" hypotheses**, which six prior Φ-tuning
  runs could not. This is the diagnostic value: regardless of outcome, the
  ambiguity that has plagued every prior memo collapses.
