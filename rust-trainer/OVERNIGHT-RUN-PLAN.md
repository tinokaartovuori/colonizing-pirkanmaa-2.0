# OVERNIGHT-RUN-PLAN — cnn-r2 (10-hour run, designed 2026-06-05)

_Designed against cnn-r1 (Plan-B test) read directly from
`checkpoints-cnn-r1/{benchmark-history.jsonl, log.jsonl}`. cnn-r1 broke b1's
plateau at gen 15 (trueWin 0.45) then collapsed (gen 35-40: trueWin
0.30-0.37, bankShare 0.18-0.28). This plan executes the user's four
instructions (A: self-play weight, B: harder opportunistic discount, C-light:
new scripted opponents, Expert fix) and is sized for the 10-h overnight
budget on the M2 Pro._

---

## Section A — Diagnosis of cnn-r1

**Last-6-bench means (gens 10-35, 360 games):**

| metric                     | value   | note                                         |
|---------------------------:|--------:|----------------------------------------------|
| trueWinVsHard              | 0.369   | beat b1's 0.411 only transiently (gen 15)    |
| winRate raw                | 0.456   |                                              |
| bankruptcyWinShare         | 0.189   | 30 of 160 champ wins are mirage              |
| maxSoldiersPerGame         | 0.69    | b1 had 0.77 — army still tiny                 |
| outpostsPerGame (end)      | 0.20    |                                              |
| bridgesPerGame             | 0.022   | new BuildBridge intent: 8 builds/360 games   |
| **expertsHiredPerGame**    | **0.014** | **5 hires across 360 games — silent fail**  |
| deviceDenialRate           | 0.24    | gate target 0.45 — cracker NOT learned       |
| hardDeviceBuildRate        | 0.39    | HARD builds Device every ~2.5 games          |
| hardDeviceSurvival         | 0.76    |                                              |
| BuildOutpost intents/bench | 7.0     | oscillating 4-8-4-5 (peak gen-15: 16)        |
| CrackHQ attempts/bench     | 18.2    | **all 18 succeed — sticky new behavior**     |
| CrackDevice attempts       | 0.0     | cracker dead despite credit=0.25             |
| Champ.conquest wins (6×)   | 108/360 | 67% of champ wins (the opportunistic mirage) |
| HARD.device wins (6×)      | 106/360 | **29% of bench games lost to Device line**   |

**Self-play (last 10 iters):** Pass 33.3%, HireSoldier 15.9%, BuildOutpost
0.16%, **HireExpert 0.000%** (every self-play game across 240 games: zero
Expert hires). Per-iter wall-clock (cumulative-delta): mean 86.6 s.

**Single failure mode (root cause).** **(c) reward shape**, specifically
the interaction of `--bankruptcy-discount 0.4` × `--vs-hard-frac 0.4`. The
discount still pays 60% for 1-soldier opportunistic Conquest wins; HARD's
default garrison only fires once `at_war` triggers, so in 40% of training
games (vs-HARD bucket) HARD's HQ holds 0-1 soldiers early-mid game and falls
to a single staged soldier. The value head sees this branch as a stable
positive — pulling policy back from the gen-15 BuildOutpost peak. The
intent-oscillation (4-16-4-5) is the visible symptom of (a) policy
oscillation, but the OSCILLATION is DRIVEN by (c). CrackHQ is *sticky*
(15-22 every bench) precisely because that same loose-HQ-garrison rewards
it terminally; BuildOutpost is *non-sticky* because no opponent forces the
soldier count above 1. Net capacity (b1 reached 0.45 with same arch) and Φ
shape (income-lead, cap-potential, w-army already on) are not the binding
constraint — the terminal signal is.

---

## Section B — New scripted opponent designs

Two NEW `AiParams` variants in `hard_ai.rs`, plus a 5-way dispatch
extension in `cnn_train.rs` (existing 3-way `ScriptKind::{DeviceRush,
ArmyRush, HqRush}` grows two buckets).

### B.1 GARRISON_PARAMS — "fortress turtle" (closes the 1-soldier-rush hole)

**Role.** User-identified gap: HARD's default `garrison: 3` only fires
under `at_war` (`should_militarise()`), so early-mid game HARD holds 0-1
defenders and falls to a single soldier-rush. GARRISON forces an
unconditional ≥ 3 HQ garrison from round 1, so the learner cannot Conquest-
win without fielding ≥ 4 soldiers (strict-greater + Outpost impregnability).

**AiParams** (delta from DEVICE_RUSH):
- `garrison: 3`, `expand: 2`, `max_outposts: 4`
- `strike_force: 0`, `assaults_per_turn: 0` (assault phase suppressed via
  the existing `if assaults_per_turn <= 1 && !can_buy { return; }` gate in
  `attack()`; counter-cracking an enemy Device stays ON because `attack:
  true` and `can_buy` flips for that path)
- `warmonger: true` (the SINGLE behavior change that's load-bearing — it
  forces `at_war` to be `true` from round 1, so the garrison fires
  immediately rather than waiting for contact)
- `experts: true`, `military: true`, `device: false`, `nuclear: false`,
  `reserve: 100`, `max_actions: 24`

**Hypothesis.** Learner must build a real army (max_soldiers ≥ 3) and
target weak frontier tiles (not the impregnable Outposts) to win Conquest.
Forces sustained BuildOutpost + HireSoldier. **~30 LOC.**

### B.2 EXPERT_PARAMS — "Expert-stacked economy" (the Expert teacher)

**Role.** Closes the user's Expert-handling observation. EXPERT plays a
pure-econ bot that fronts the Expert tier (Mine+Expert doubles output;
Hydro/Nuclear gate production entirely on Expert presence — see
`managers.rs:846-887`). The learner faces an opponent whose per-round
income overtakes farm-only economies by ~r25; terminal pressure is
Domination loss unless the learner ALSO staffs Experts.

**AiParams.**
- `experts: true`, `nuclear: true` (the levers — already wired)
- `military: false` (pure economic teacher — never strikes)
- `attack: true`, `assaults_per_turn: 1` (cracker-only against an enemy
  Device, no offensive assaults)
- `garrison: 1`, `max_outposts: 2`, `strike_force: 0`
- `device: false`, `warmonger: false`
- `expand: 4`, `reserve: 140`, `max_actions: 28`

**Build-side priority.** No code change — HARD's existing
`build_power_plants` + `invest_nuclear` + `boost_mines` + `staff_plant`
already prefer Experts when `experts: true`. The behavior change is
emergent from the `military: false` × `assaults_per_turn: 1` combo.

**Hypothesis.** Without Expert-stacking, the learner's farm-only income
trails EXPERT by r30+ → Domination loss. With B's stronger opportunistic
discount, Conquest of an undefended econ-bot pays 30%, so over-running
EXPERT isn't a free win either. Forces the learner to ECONOMIC-grow, which
requires Experts. **~25 LOC.**

### Why two not three

Five total scripted variants (3 existing + 2 new) at `--script-frac 0.5`
× 24 games/iter = ~12 scripted games/iter ÷ 5 ≈ 2-3/bucket/iter. Adequate
over 400 iters. A third variant would drop bucket sample below
script-grade's noise floor.

---

## Section C — Expert-handling diagnosis + fix

**Diagnosis (cnn-r1 log.jsonl last 10 iters, 240 self-play games):**

| metric                       | value    |
|-----------------------------:|---------:|
| iterIntents.HireExpert sum   | **0**    |
| iterIntents.StackProducer    | 0        |
| iterIntents.BuildMine        | 90       |
| bench expertsHiredPerGame    | 0.014    |

**Mechanic** (`cp-sim/managers.rs:846-887`):
- **Mine:** Expert on tile → `prod` added a 2nd time per worker →
  **2× output** (metal+stone).
- **Hydro / Nuclear:** Production GATED on Expert (zero without).
- Expert cost ≈ 4× a BasicWorker, 5× upkeep.

**Binding constraint.** Only `Intent::StackProducer` ever buys Experts
(`candidates.rs:1264-1309`). Its gate requires `free_unit_amount(p) > 1`
— a free unit slot AND a buffer. cnn-r1 builds Mines (9/iter) but never
proactively builds Villages, so `free_unit_amount` stays 0-1, blocking
the candidate from ever being emitted. Downstream-of-passivity, not a
missing action.

**Proposed fix: `--w-expert` Φ term (NEW).** Reward filled Expert slots
on producer buildings:

```
w_expert · clamp( staffed_experts_on(Mine|Hydro|Nuclear) / EXPERT_TARGET )
EXPERT_TARGET = 3, saturating, signed positive only
```

Mirrors `w_army` (filled-soldier) in shape. ~30 LOC in
`potential_step1` + flag parsing + 1 unit test that `--w-expert 0` is bit-
identical to baseline.

**Why this over alternatives.** EXPERT_PARAMS (B.2) supplies the
terminal pressure; `--w-expert` supplies the gradient. Φ shaping is
policy-invariant (Ng 1999), so it can't create a wrong terminal optimum
— it can only accelerate the chain Village → free slot → Mine + Expert
once it's terminally rewarded. This is the "reward says good + curriculum
says required" template that worked for the Outpost gate-lowering.
Initial weight 0.15 (smaller than w_army's 0.4 — the chain is longer and
we don't want to crowd out the army).

NOT a candidate-priority adjustment (priority isn't the issue; the gate
behind it is). NOT an action-level credit on the Expert purchase
(non-potential, would risk reward-hacking per the memo).

---

## Section D — Run plan

### D.1 Flag changes vs cnn-r1

| flag                     | r1   | r2     | rationale                                  |
|--------------------------|:----:|:------:|--------------------------------------------|
| `--vs-hard-frac`         | 0.4  | **0.2**| Instruction A: less HARD-bucket bias       |
| `--bankruptcy-discount`  | 0.4  | **0.7**| Instruction B: harden opportunistic floor  |
| `--script-frac`          | 0.4  | **0.5**| absorb the freed self-play slot            |
| `--w-expert` (NEW)       | --   | **0.15** | Section C — requires implementation        |
| `--iters`                | 50   | **400**| 10-h budget allocation                     |
| `--bench-every`          | 5    | **10** | reduce bench overhead at scale             |
| `--replay-every`         | 10   | **25** | debug only; not gating                     |

All other flags unchanged from cnn-r1: `--w-army 0.4 --cap-potential 0.3
--idle-flow-penalty 0.3 --device-crack-credit 0.25 --hq-crack-credit 0.25
--turn-search-spend --build-prior-floor 0.06 --sims 64`. Preset supplies
`--net-size small --turn-search --income-lead-potential 0.5
--tile-potential 0.4 --w-cut 0.15 --record-opp-value --device-potential
0.2 --device-credit 0.15 --pfsp --script-opponents --script-grade
--tie-penalty 0.4 --stall-rounds 80 --shape-gamma 0.99 --shape-weight 0.3
--cap 150 --games 24 --bench-games 60 --threads 8`.

### D.2 Wall-clock budget

- cnn-r1 mean per-iter (gens 21-40, cumulative delta): **86.6 s**.
- New Φ term + script dispatches: negligible CPU.
- Bench-every 5→10 saves ~5% at scale.
- Estimate: **90 s/iter × 400 iters = 600 min = 10.0 h**. Margin: at
  100 s/iter → 667 min = 11.1 h (still overnight).
- Alternative rejected: `--sims 96` × 250 iters ≈ same budget, but the
  DEEP-REDESIGN evidence (§3.1) shows lift comes from action-space +
  reward, not search depth. Spend the budget on iterations.

### D.3 Cold-start required?

**No.** `--w-expert` extends Φ (a scalar) — value-head input dim is
unchanged; default 0.0 is bit-identical to baseline (an existing
checkpoint can load it). New scripted opponents only re-skew the
opponent seat's policy — no planes, no scalars, no candidates change.

**Initialize from `checkpoints-cnn-r1/champion-best.json`** (gen-15 peak,
trueWin 0.45) for a ~5-iter head start over random init. NO parity-arc
bump (no game-rule change).

---

## Section E — Gates and contingencies

### PASS gate (last-10-bench means, gens ~310-400 / 600 games)

- **trueWinVsHard ≥ 0.50** (cnn-r1 = 0.369; +0.13 outside CI of ±0.04)
- AND **bankruptcyWinShare ≤ 0.15** (instruction B success; r1 = 0.189)
- AND **maxSoldiersPerGame ≥ 1.0** (r1 = 0.69; army-build locked in)
- AND **expertsHiredPerGame ≥ 0.5** (r1 = 0.014; Expert chain learned)
- AND **deviceDenialRate ≥ 0.40** (r1 = 0.24; cracker improves)

### FAIL gates (signal-only — don't auto-kill an overnight run)

- **gen 100 trueWin < 0.30 sustained ≥ 5 benches**: discount too hard →
  next session, `--bankruptcy-discount 0.55`.
- **gen 150 expertsHiredPerGame < 0.05**: Φ-term too weak → bump
  `--w-expert 0.15 → 0.4` or add a Village-potential term.
- **gen 100 maxSoldiers > 1.5 AND BuildOutpost intent oscillating ≥ 50%
  bench-to-bench**: curriculum bucket imbalance → cap any single script
  bucket at 50% (new flag).

### Mid-run contingency (if user wakes at hour 5, gen ~200)

If `bankShare > 0.20` AND `trueWin < 0.45`: `Ctrl-C`, add `--score-by-army
0.5` (NEW FLAG, scales value-target by `min(1, soldiers/3)` for army-less
wins), restart from `checkpoints-cnn-r2/champion-best.json`.

---

## Section F — Exact launch command

```bash
./rust-trainer/presets/launch.sh \
  --out rust-trainer/checkpoints-cnn-r2 \
  --init rust-trainer/checkpoints-cnn-r1/champion-best.json \
  --iters 400 \
  --bench-every 10 \
  --replay-every 25 \
  --vs-hard-frac 0.2 \
  --bankruptcy-discount 0.7 \
  --script-frac 0.5 \
  --w-army 0.4 \
  --w-expert 0.15 \
  --cap-potential 0.3 \
  --idle-flow-penalty 0.3 \
  --device-crack-credit 0.25 \
  --hq-crack-credit 0.25 \
  --turn-search-spend \
  --build-prior-floor 0.06 \
  --sims 64
```

**Items requiring implementation BEFORE launch (~100 LOC total):**

1. **`--w-expert <f64>`** — NEW Φ term. `TrainCfg.w_expert: f64` default
   0.0; parse in the arg block at `cnn_train.rs:5727`-ish; accumulate
   `w_expert · clamp(staffed_experts_on_producers / 3.0)` in
   `potential_step1`; 1 unit test for the 0.0-noop. **[NEW FLAG, Section
   C.]**
2. **GARRISON_PARAMS** + `HardAi::garrison_fortress()` + `ScriptKind::
   GarrisonFortress` variant + dispatch extension at `cnn_train.rs:
   4815-4821` + 1 sanity test. **[Section B.1.]**
3. **EXPERT_PARAMS** + `HardAi::econ_expert()` + `ScriptKind::
   EconExpert` variant + dispatch + 1 sanity test. **[Section B.2.]**
4. **script-grade 3-way → 5-way weighting** (`grade_*_w/n` arrays grow
   to 5; existing `pfsp_weight` calls unchanged).

No parity change (candidates.rs untouched); parity 8/8 holds. No arc
bump (no game-rule change). No cold-start.

---

## Summary

cnn-r1's binding constraint is **reward shape**: at `--bankruptcy-
discount 0.4` × `--vs-hard-frac 0.4`, the value head still sees
opportunistic 1-soldier-rush conquests of HARD's loose default HQ
garrison as a stable terminal positive (108 of 160 wins, 67%). This
pulled the policy off the gen-15 BuildOutpost peak. cnn-r2 attacks this
with (A) vs-hard-frac 0.4→0.2, (B) discount 0.4→0.7, (C) two new
scripted opponents (GARRISON closes the 1-rush hole, EXPERT supplies
econ-pressure), and an Expert Φ term that rewards filled Expert slots on
producer buildings. 400 iters × ~90 s/iter ≈ 10 h. Initialize from
cnn-r1's `champion-best.json` (gen-15 peak). Gate: `trueWinVsHard ≥ 0.50`
over last-10-bench means with bankShare ≤ 0.15, maxSoldiers ≥ 1.0,
expertsHired ≥ 0.5, deviceDenial ≥ 0.40 as orthogonal checks. ~100 LOC of
Rust implementation + 5 LOC of tests; no parity change, no arc bump, no
cold-start.
