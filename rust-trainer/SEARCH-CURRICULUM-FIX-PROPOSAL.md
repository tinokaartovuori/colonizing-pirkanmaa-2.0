# Search / Curriculum Fix Proposal — why the army-rusher data is NOT teaching armies

_Authored 2026-06-05. Scope: curriculum, search depth, exploration, opponent
distribution, value bootstrapping. NOT Φ design (other agent). Citations to live
`main` (sd2 arc, post-Outpost rebalance)._

## 1. Why the army-rusher data does NOT teach armies

**1.1 The army-rusher works.** `ARMY_RUSH_PARAMS` (`crates/cp-ai/src/hard_ai.rs:145-160`):
`max_outposts 7`, `strike_force 10`, `assaults_per_turn 10`, `warmonger true`,
`device false`. Standard `HardAi::run_turn` (`hard_ai.rs:294-321`) calls
`build_outposts → expand → military → attack`. Unit-tested at
`cnn_train.rs:6453` (`scripted_army_rusher_fields_soldiers_and_assaults`).
Not a broken opponent.

**1.2 The learner DOES lose, AND `tilesLostToRusher` is rising late.**
`checkpoints-cnn-i1/log.jsonl`: `spVsArmyRush` 0.000 (gen 1) → 0.500 (gen 49)
on 3-7 games/iter — cumulative ~0.25-0.35 (CI ±15%). `tilesLostToRusher` 8.7
(gen 0) → 16.5 (gen 49). So the loss signal IS there.

**1.3 Value-bootstrap arithmetic is fine; the policy never sees the gradient.**
Trace `play_one_game_explore` (`cnn_train.rs:2915, 2986-2993`):
`learner_seat = cur.0 == 0; let record = learner_seat;`. Only seat 0's POLICY
is recorded. With `--record-opp-value` ON, seat-1 examples push but
`value_only: true` + `pi: Vec::new()` (line 3108) → routed to
`train_grad_value_only_scalars` (skips policy head entirely). So the value head
un-squashes (gen 0 `valPredWin` 0.069, gen 47 0.099, loss span 0.40 → 0.49) but
**the learner's policy head only ever updates from its own losing trajectories**,
whose π is concentrated where MCTS visits land — which brings us to §2.

## 2. Mechanical search blind spots — the actual binding constraint

**2.1 `--turn-search` rollouts terminate on the first greedy Pass.**
`Mcts::complete_root_turn` (`cnn_train.rs:515-606`) fills the rest of the turn
with the net's greedy argmax. Line 549: `if cands[best].intent == Intent::Pass
{ if !self.turn_search_spend { break; } ... }`. With `--turn-search` ON and
`--turn-search-spend` OFF (the i1 config — flag exists at lines 407-414 / 5019
but DEFAULTS OFF), the FIRST greedy Pass aborts the rest of every rollout's
root turn. Replays show 100% Pass past round ~40 → every leaf-rollout that
reaches round 40 immediately freezes the root for ~50 rounds. The value head
bootstraps off **futures where the root does nothing**, exactly the
`EXP-M-DESIGN §4` "starved one-action-per-turn future" — reintroduced through
`break-on-Pass`.

**2.2 48 sims × prior 0.03 ≈ 1.4 visits on the Outpost edge.** Branching at
the root (`candidates.rs:1022-1043`): ~10-15 candidates mid-game. Dirichlet
`α=0.40 ε=0.35` (`cnn_train.rs:2055-2056`): mean noise ≈ 0.35/15 = 0.023 per
arm. `build_prior_floor 0.03` (line 2063, applied at 2244-2247) floors Outpost
when enumerated. At 48 sims this is ≈ 1.4 root visits — statistically
indistinguishable from never exploring it. PUCT can't re-weight toward Q
because Q is 0 after one noisy sample (and that sample's leaf was starved by
§2.1).

**2.3 The HQ-only ⇒ cap-1 cliff is invisible.** §5 of `GAME-MECHANICS.md`:
HQ+1 + Outpost+3 — building the first Outpost **quadruples soldier cap**, a
discrete cliff with no smooth gradient. Intent histograms show BuildOutpost is
0-7 per ~3000 decisions/iter (<0.2%) → buffer fraction of post-Outpost states
<2%. The value head prices post-Outpost states the same as pre-Outpost ones.

**2.4 The search's forced opponent is HARD, not the scripted opponent.**
`advance_after_root` (`cnn_train.rs:635-671`) uses
`tree.bot = HardAi::hard()` (line 2217, 746) **even in army-rush games**. The
value head learns "vs army-rusher" but the rollouts that bootstrap it imagine
the milder HARD. Hidden curriculum/search mismatch — search systematically
under-estimates the cost of having no army.

**2.5 Outpost gate gating.** `candidates.rs:492-535`: `tile_count ≥ 8`,
`metal_income ≥ 15·(outposts+1)`, sd2 cost 500/200/200/100. Reachable ~r12-15
with one Mine. So Outpost IS enumerated by mid-game — the binding constraint
is search-side visits (§2.1, §2.2), not the gate.

## 3. The proposal — ONE highest-leverage change (+ small companion)

**PRIMARY: `--turn-search-spend ON` + `--build-prior-floor 0.03 → 0.08` +
`--sims 48 → 96`.**

**File:** `crates/cp-train/src/bin/cnn_train.rs` — no code change. The
spend-mode logic already exists at lines 515-606; just flip the flag (line
5019 parses it). Bump `build_prior_floor`/`sims` via existing flags
(lines 5000, 4641-equivalent).

**Why this is the single highest-leverage in-scope change:**
1. (search-depth fix) `--turn-search-spend` cures §2.1 — that one
   `break` at line 550 is doing more damage than any reward tuning. Spend-mode
   keeps acting past a greedy Pass, executing the best non-Pass candidate as
   long as the value head doesn't deem it strictly worse (line 592). The value
   head finally bootstraps from realistic futures.
2. `--build-prior-floor 0.08` (≈ 8 visits at 96 sims) takes the Outpost edge
   from "1 noisy visit" to "5-10 stable visits" — enough that Q(s, Outpost)
   estimates something and π puts ~10% mass on Outpost in the training target.
3. `--sims 96` (2×) buys depth at the round-25 conquest payoff: turn-search
   makes each sim ~1 turn deep, so 96 sims reach turn ~30 along most-visited
   lines. 48 had ~4 effective hops after PUCT reallocated.

Three flag flips. No code change. No cold-start. Parity-free.

**Test:** integration test in `cnn_train.rs` — fixed mid-game state with
Outpost enumerated + a randomly-initialised net + 96 sims + spend-mode →
assert root `edge_visits[BuildOutpost] > edge_visits[Pass]` (sanity for
floor + sims interaction).

**Gate (30-40 iters, judge aggregated):**
- `outpostsPerGame` > 0.5 (i1 baseline 0.10-0.22) — tight 60-game leading metric.
- `maxSoldiersPerGame` > 1.5 (i1 baseline 0.6).
- `tilesLostToRusher` < 8 (i1 baseline 14-17).
- Intent `Pass` < 30% (i1 baseline ~38%).
- Only THEN does `trueWinVsHard` move.

**Wall-clock:** sims 2× + spend-mode rollouts longer (was: break-on-Pass) ≈
2.5× per iter. i1 was ~80 s/iter → ~3.3 min/iter → ~2.7 h / 50 iters. Tractable.

**COMPANION (≤30 LOC, parity-free, training-side):** fix §2.4 — thread
`opp_script` into `Mcts` and use `HardAi::army_rush()` in `advance_after_root`
when the actual opponent is the rusher. Without this the search systematically
under-estimates the cost of no army.

## Lower-priority alternatives explicitly considered

- **(c) Replay-buffer oversampling of post-Outpost states.** Direct but
  self-defeating: if the policy stops building Outposts the buffer drains.
  Useful AFTER §3 generates examples.
- **(b) New scripted "Outpost-defender" opponent.** Adds another HardAi
  variant + grading work; defer until §3 proves the search axis is right.

## Skeptic's check

1. **Army-rusher still too strong with the new search.** Learner builds the
   chain but loses anyway → value gradient says "armies lose."
   *Detect within 10-15 iters:* `maxSoldiersPerGame` ↑ but `spVsArmyRush`
   stays < 0.30. Mitigate: soften `ARMY_RUSH_PARAMS` (strike_force 10→6,
   assaults 10→4) or gate the rusher to gen ≥ 20.

2. **2× sims doesn't reach the conquest window because rollout leaves stay
   passive.** *Detect:* `valPredWin` stays flat near zero even with spend-mode
   → leaves still evaluated as neutral. Mitigation: swap `complete_root_turn`
   to use `HardAi::hard()` (a known-active policy) for rollout completion
   instead of greedy net argmax — one-line at `cnn_train.rs:524`.

3. **Forced Outposts bankrupt the learner.** Outpost upkeep is −50/round
   (`candidates.rs:541`). 2-3 Outposts without supporting Mines → bankruptcy
   inverts the mirage onto the learner. *Detect:* `bankruptcyWinShare`
   inverts — HARD's bankruptcy share falls AND learner-bankrupt games ≥ 0.10.
   Mitigation: raise the metal-income gate (`candidates.rs:506`) from
   `15·(out+1)` to `20·(out+1)` — parity-locked, edits TS mirror too.

## Summary

The army-rusher curriculum + `--record-opp-value` supply a value-head signal,
but the **policy head never sees π with mass on Outpost** because (a) rollouts
truncate on the first greedy Pass (`cnn_train.rs:550`), starving every leaf into
a do-nothing future that prices Outposts at 0, and (b) 48 sims × prior 0.03 ≈
1.4 visits on the Outpost edge — search is blind. **One flag (`--turn-search-spend`)
+ two knob bumps (`--build-prior-floor 0.08 --sims 96`) directly address both
blind spots — no code change, no parity touch, no cold-start.** A small
companion edit makes the search's forced-opponent match the game-loop opponent.
Gate on `outpostsPerGame > 0.5` and `maxSoldiersPerGame > 1.5` over 30-40 iters
before tuning Φ further.
