# TRAINING-V2-PROPOSAL — clean-slate redesign of the AlphaZero pipeline

_Authored 2026-06-05. **Puhtaalta pöydältä.** Twelve runs have hit one ceiling.
This memo questions assumptions the prior twelve memos accepted: the action-space
shape, the receptive field, the value-head's perceptual access, the curriculum
mix, and — most importantly — whether AlphaZero is the correct paradigm at all
for a 7-skill action-rich strategy game with a 24-channel CNN of 9.8k parameters._

_Sources audited end-to-end: `rust-trainer/GAME-MECHANICS.md` (canonical rules);
`crates/cp-ai/src/{planes.rs, candidates.rs, spatial_net.rs, hard_ai.rs}`;
`crates/cp-train/src/bin/cnn_train.rs` (9,169 LOC); 12 `benchmark-history.jsonl`
files (14 runs total — b1/i1/s3/c1/bc1/bc2/r1/r2/r3/r4/r5/s1/s2/asym1, ~120 bench
rows ≈ 7,200 evaluation games); replay traces from cnn-r4
(`replay_vs_armyrush.json`, `replay_vs_hqrush.json`)._

---

## §0 — Executive summary

The twelve runs have plateaued at **trueWinVsHard ≈ 0.32-0.45** with one
exception (asym1: 0.45 last-5 mean / 0.52 peak — the recently-introduced
asymmetric self-play). Replays prove the underlying behavioural failure is
**not** subtle policy mis-calibration: at gen 50 of cnn-r4 the champion
**builds zero soldiers across an entire 91-round game** and loses with 12 owned
tiles vs the opponent's 100+. This is not a Φ-tuning regime; this is a **data-
distribution regime**. The fix is not another Φ retune but a four-pillar
redesign: (1) **AlphaStar-style supervised pretraining** from an army-marching
teacher to seed the policy *outside* the 1-rush/passive attractor; (2) a
**doubled spatial receptive field via dilated convs** so the net can correlate a
soldier-at-HQ with an enemy-device-far-across-the-map within the trunk, not just
the global pool; (3) **explicit perception channels** for `frontier-distance-to-
nearest-enemy-tile` and `my-mobile-soldier-budget` (today the budget plane only
shows the *enemy's* — the asymmetry is the perceptual root of the
defense/offense gap); (4) **drop AlphaZero MCTS for the policy update entirely**
on the long-horizon Device line and replace it with **PPO + GAE** off the same
self-play infrastructure: MCTS at 64 sims with a 100-turn search horizon and a
forward model that costs ~1 ms/sim cannot bridge a 90-turn Device payoff, and
the empirical record (12 runs, no Φ tweak ever moved the gap > 0.05) is
consistent with that horizon limit. This is the single decision that risks
discarding ~6 months of code and demands the strongest justification — §6 makes
the case.

## §1 — Audit the eyes (perception)

The net's perceptual surface is **24 planes × H × W** + 12 **scalar features** at
the value head + 16 **per-candidate local features**. Faithful list of all 24
planes from `crates/cp-ai/src/planes.rs:38-100`:

```
 0  C_MINE                  owned by me
 1  C_ENEMY                 owned by any live enemy
 2  C_NEUTRAL               unowned
 3  C_MY_HQ                 my un-conquered HQ
 4  C_ENEMY_HQ              any live-enemy HQ
 5  C_PRODUCER              Farm/Mine/Village/Hydro/Nuclear (staffing-agnostic)
 6  C_MILITARY              Outpost (impregnable by assault)
 7  C_DEVICE                Strange Device present
 8  C_MY_OWNED_SOLDIERS     my defenders, count/5
 9  C_HQ_CONNECTED          BFS-from-HQ over owned tiles
10  C_T_GRASSLAND           terrain
11  C_T_FOREST              terrain
12  C_T_MOUNTAIN            terrain
13  C_T_RIVER               terrain
14  C_PRODUCING             producer that PRODUCES this turn (staff-gated)
15  C_ENEMY_OWNED_SOLDIERS  enemy defenders
16  C_ENEMY_REACH           union of enemy-staging frontiers (the threat plane)
17  C_MY_REACH              MY staging frontier (my "where I can strike")
18  C_MY_CONQ_SOLDIERS      my staged-attacker soldiers
19  C_ENEMY_CONQ_SOLDIERS   enemy staged-attacker soldiers
20  C_ATT_MINUS_DEF         signed attacker−defender at each cell
21  C_DEVICE_DEFENSELESS    Device tile = 0 owned defenders binary
22  C_RIVER_BLOCK           unbridged owned river (expansion dead-end)
23  C_ENEMY_BUDGET          BROADCAST: max enemy mobile-soldier budget
```

**User concern × perceptual access:**

1. **Defense (proactive HQ garrison).** Eyes are sufficient: planes 3, 9, 16,
   23 expose own-HQ, connectivity, enemy reach, enemy mobile budget. NOT
   perception-limited. The defensive *capability* (a soldier on the HQ) is
   what's missing — see §3 (the policy never has a soldier in the first
   place).
2. **Army-building.** Eyes are sufficient (planes 8/16/17/20 + scalars 3, 9 for
   `used_soldier`, `soldier_headroom`). NOT perception-limited — the scalars 3
   and 9 from `cnn_train.rs:140-141, 195` directly encode "I have an Outpost
   ⇒ headroom rises by 3."
3. **River crossing.** Eyes are sufficient (plane 22 `C_RIVER_BLOCK` + plane 13
   `C_T_RIVER`). The Bridge intent now exists (`candidates.rs:43, 723-782`)
   with `target_value = bridge_unblock_count`. NOT perception-limited.
4. **Device cracking from far across the map.** **This is the interesting
   case.** The net DOES see the enemy device (plane 7 + plane 21 + plane 4 for
   the enemy HQ + scalar 11 `enemy_device_threat = (max_cd − cd)/max_cd`). But
   the **spatial reasoning** "my soldier here can reach the device in 3 moves"
   requires either (a) a receptive field large enough to span "my soldier" and
   "enemy device" — or (b) a board-wide feature exposing distance. From
   `spatial_net.rs:126-129, 285-296`: the small net is conv1(3×3) → conv2(3×3),
   effective receptive field **5×5**. The board is 14×12. A soldier and a
   device 10 tiles apart are NOT in the same RF. The trunk **cannot** correlate
   them position-by-position. **The only correlation mechanism is the
   GlobalAvgPool at line 297**, which collapses board_embed (D=24 channels) to
   a 24-dim vector via mean across all 168 cells — i.e. "is there an enemy
   device anywhere on the board" survives, but "where my soldier is relative
   to it" does not. The big net (D1=32, D=48, +residual conv3 = 7×7 RF) helps
   only by 2 cells. **This is PERCEPTION-LIMITED for any objective that
   requires reasoning over board-spanning distances**, which includes
   cross-map device cracking AND pre-emptive marching to enemy HQ.
5. **Pre-emptive offense (march to enemy HQ).** Same as 4 — perception-limited
   at depth > RF. The candidate-side feature
   `local.spatial.dist_nearest_enemy` (`candidates.rs:292-303`) papers over
   this for `Expand`/`Attack`/`CrackDevice`/`CrackHQ` candidates by computing
   Manhattan distance and feeding it as a scalar at local index 14. But it's
   only set for the candidates that need a target tile — there is no
   `march-toward-enemy` candidate to host it (see §2).

**Conclusion — gap taxonomy:**

| user concern | perception | action | reward |
|---|---|---|---|
| 1 defense | OK | OK (HireSoldier) | REWARD-LIMITED (§3) |
| 2 army-building | OK | OK | REWARD-LIMITED (§3) |
| 3 river crossing | OK | OK (BuildBridge exists) | DATA-LIMITED (no payoff) |
| 4 device cracking | **PERCEPTION-LIMITED** + ACTION-OK + REWARD-OK | CrackDevice exists | distance scalar exists only on the candidate, not on the planes |
| 5 pre-emptive offense | **PERCEPTION-LIMITED** (no marching candidate) | **ACTION-LIMITED** (no march intent) | n/a |

The high-leverage perceptual gap is **a board-wide distance gradient**, not a
new categorical plane. Concrete proposal in §6: add `C_DIST_TO_ENEMY_DEVICE`,
`C_DIST_TO_ENEMY_HQ`, and **dilate the trunk** so the RF spans the board.

## §2 — Audit the action space

15 intents in `candidates.rs:25-54`. Enumeration in `enumerate()`
(`candidates.rs:1326-1357`):

```
 0 BuildFarm           empty owned grassland
 1 BuildMine           empty owned mountain
 2 BuildVillage        empty grassland + forest exists + cash-flow OK
 3 BuildOutpost        cfg.military, tile_count≥8, metal_income gate
 4 BuildHydro          cfg.experts, owned empty river with Hydro-allowed orientation
 5 BuildNuclear        cfg.experts + cfg.nuclear + cash + free unit slot > 1
 6 Expand              neutral, unowned, has room, non-threatened
 7 HireSoldier         cfg.military + free_soldier > 0 + metal ≥ 50
 8 Attack              feasible enemy tile (not Outpost, ≤3 defenders, mover available)
 9 StackProducer       free_unit > 0 + producer with room
10 Pass                always
11 BuildStrangeDevice  cfg.device + no Device standing + rounds ≥ 18 + not-losing
12 BuildBridge         owned River + no building + Bridge orientation + cost
13 CrackDevice         enemy owns Device + reachable + can stage ≥1 soldier
14 CrackHQ             enemy un-conquered HQ + reachable + can stage ≥1 soldier
```

**User concern × action availability:**

1. **Defense.** `HireSoldier` (intent 7, `candidates.rs:1090-1139`) prefers a
   `threatened` tile, then HQ, then any owned tile with room — so it CAN
   garrison the HQ. *Action present.*
2. **Army-building.** `BuildOutpost` (3) + `HireSoldier` (7). *Action present.*
3. **River crossing.** `BuildBridge` (12) added. *Action present.*
4. **Device cracking.** `CrackDevice` (13, `candidates.rs:790-894`) explicitly
   stages soldiers on the device tile. *Action present.*
5. **Pre-emptive offense.** **NO MATCHING INTENT.** The only offensive intents
   are `Attack` (8), `CrackDevice` (13), `CrackHQ` (14) — and ALL THREE require
   the target tile to ALREADY be in `get_available_tiles_for(p)`, i.e.
   adjacent to owned territory. **There is no `MarchSoldier(distant_tile)`
   intent.** A soldier sitting at HQ with the enemy 10 tiles away has no legal
   way to advance toward the enemy via the candidate set — only Expand
   (workers, not soldiers) and Attack (adjacency-gated) exist. The MARCHER
   scripted opponent (`hard_ai.rs:247-262`) was added precisely because the AI
   "sits at home with 3 soldiers and waits"; but the MARCHER's `march_to_
   enemy_hq` phase is in the scripted controller, not in the candidate set,
   so the LEARNER cannot copy what the MARCHER demonstrates even if the buffer
   contained those games. **This is the single largest action-space gap that
   no prior memo has flagged.**

**Per-candidate local features — sufficient to differentiate good targets?**

The 16 local features (`candidates.rs:354-388`) for any candidate are: cost,
net_delta, target_value, unit_cap_gain, soldier_cap_gain, threatened,
money_margin, income_staffing, wood_margin, metal_margin, +6 spatial
(`enemy_neighbors`, `own_neighbors`, `neutral_neighbors`, `dist_own_hq`,
`dist_nearest_enemy`, `frontier`). For Attack/CrackDevice/CrackHQ the spatial
slot 14 (`dist_nearest_enemy`) and slot 15 (`frontier` = own-soldier-neighbors
of target / 3) are populated. **This is fine for ranking targets that the
candidate generator emitted.** It is NOT fine for *initiating* a multi-turn
march, because the only thing close-distance can do is rank tiles I can
already reach — it cannot suggest "move toward". The action-space conceptual
mismatch is: the policy is asked to "score 15 ways to spend one budget unit",
but a march is a 5-turn commitment with no single-step payoff. Until there is
either (a) a literal `MarchSoldier(direction|tile)` intent or (b) a hierarchical
"phase" head, the policy fundamentally cannot express the marching strategy.

**Priority/ordering issues.** Attack enumeration (`candidates.rs:1141-1262`)
sorts HQ-first → fewest defenders → tile-index. Expand sorts by claim_value DESC
then tile-index ASC. Both have `*_CANDIDATE_CAP` (6 and 4 respectively).
`Intent::Pass` is appended LAST. The ordering is reasonable; the problem is
**absence**, not priority.

## §3 — Audit the reward structure

The reward stack is in `cnn_train.rs`. Three layers compose every example's
value target `z`:

1. **Terminal z** (`cnn_train.rs:3474-3496`, the `terminal_z` closure):
   - winner-seat: `opportunistic_discounted_z(mag, cause, built_outpost,
     max_owned_soldiers, bankruptcy_discount)`
   - loser-seat: `-mag`
   - tie: `-tie_penalty`
   where `mag = device_decided ? 1.0 : (1.0 - device_bonus)`. The
   opportunistic-discount fires when `(Bankruptcy | Conquest) AND
   !built_outpost AND max_owned_soldiers < 2` — exactly the "1-rush mirage".
2. **Per-step Φ shaping** (`cnn_train.rs:2767-2924`, `potential_step1`).
   Composes a dozen flag-gated terms: `income_lead_potential`, `cap_potential`
   (saturating soldier-cap at CAP_TARGET=7), `idle_flow_penalty` (unstaffed
   units + un-spent income — NOT empty slots), `w_army` (filled-soldier out
   to ARMY_TARGET=7), `w_cut` (`hq_cut_exposure`), `w_expert` (filled-Expert
   on Mine/Hydro/Nuclear), `w_soldier_forward` (per-soldier 1 − d/diameter, the
   march gradient). The shaped return formula
   (`cnn_train.rs:3656-`, `shaped_returns`) is the Ng-1999 telescoping
   `G_i = w·(γΦ_{i+1} − Φ_i) + γ·G_{i+1}`, clamped to [−1,1].
3. **Per-decision credit bumps** (`cnn_train.rs:3532-3604`):
   `--device-credit` for BuildStrangeDevice + HireSoldier-while-own-device,
   `--device-crack-credit` for CrackDevice, `--hq-crack-credit` for CrackHQ.
   All clamp z to [−1,1].

**User concern × reward signal:**

1. **Defense — does shaping reward garrisoning HQ?** `w_cut · hq_cut_exposure`
   penalises being one cut from losing tiles. But the formula
   (`cnn_train.rs:2621-2674`) is purely TOPOLOGICAL (BFS over owned tiles); it
   ignores soldier presence. Garrisoning HQ with a soldier reduces NO term in
   Φ except via `w_army` (filled cap) — which is the same +Δ whether the
   soldier is on the HQ or anywhere else. **There is no Φ term that rewards a
   soldier ON THE HQ specifically.** The only reward for HQ defense is
   terminal: not losing the HQ ⇒ not losing the game ⇒ +1. With Φ-shaping
   gradient `≈ 0.3·1/cap_target ≈ 0.04` per defender per turn, the gradient
   from terminal z (γ^t · 1 for t = turns-to-loss) is ≈ 0.99^40 · 1 ≈ 0.67 —
   much larger, but only if the policy's value head can actually predict the
   eventual loss conditional on HQ-being-undefended NOW. The state-value head
   sees `enemy_device_threat`, `enemy_reach`, `att_minus_def`, but NOT
   "is my HQ undefended right now" as a discrete feature. **REWARD-LIMITED
   AND VALUE-HEAD-LIMITED.**
2. **Army-building — `w_army` + `cap_potential` + `--device-credit` defending.**
   The Φ terms exist (`cnn_train.rs:2874-2879`) but yield Δ_step ≤ w_army/7 ≈
   0.057 per added soldier — small. The terminal signal pays IFF the army
   eventually wins; the 1-soldier-rush wins so often (asym1 bankShare 0.04,
   r2 bankShare 0.27) that the value head learns "1 soldier already suffices"
   — i.e., the gradient on going from 1 → 2 soldiers is approximately ZERO
   because both win the same ~45% of the time. **REWARD-LIMITED: the marginal
   value of soldier N → N+1 is ε on the existing training distribution.**
3. **River crossing.** `bridge_unblock_count` is a *local* feature, not a Φ
   term. The terminal signal for a Bridge is "Bridge enables expansion through
   river ⇒ +tiles ⇒ +tile_lead ⇒ +Φ" 5-10 turns later. With γ=0.99 and a
   self-play opponent that ends games before the payoff is realised, the value
   head sees E[z | built bridge] ≈ E[z | did not] — bridge-builders win at the
   same rate as bridge-skippers in cnn-r4 self-play (the buffer doesn't
   contain games long enough for the bridge to matter). **DATA-LIMITED.**
4. **Device cracking.** `--device-crack-credit` bumps `z += c · |z|` on a
   CrackDevice decision IF the seat wins by Conquest or Device
   (`cnn_train.rs:3569-3584`). At r1's c=0.25 this is sensible. But CrackDevice
   only fires when cap > 1 (need ≥ 1 free soldier to stage); r2 logged
   "CrackDevice 37 attempts / 2 successes per bench" — the policy tries but
   the **precondition** (army) is missing. The credit thus *rewards* attempts
   the policy can't execute. The value-head learns "wanting to crack is good"
   but the policy has no army to crack with. **REWARD-LIMITED upstream: the
   reward chain rewards a downstream skill before the upstream skill is
   present in the data distribution.**
5. **Pre-emptive offense.** `w_soldier_forward = w · forward_score(seat)`
   exists (`cnn_train.rs:2884-2886`, `forward_score` 2689-2737) and rewards
   "soldiers near enemy". For 1 soldier at distance d in a 14+12=26 diameter
   board, Δ = (1 − d/26) / 7. d=10 ⇒ 0.087, d=20 ⇒ 0.033. Φ-shaping difference
   `γΦ' − Φ` for moving the soldier 1 step closer ≈ (1/26)/7 = 0.0055 per turn
   — vanishingly small relative to e.g. `idle_flow_penalty 0.3 ·
   unspent_income`. **The gradient exists but is dominated by other terms.**

**The KEY question — is the value head learning "I will win" or "this is good
locally"?**

The value loss is `(value − z)^2` where `z` is the shaped return — a mixture of
the terminal label and the Φ-trajectory. For a turn-search game with
`shape_weight = 0.3`, `gamma = 0.99`, the proportion of `z` coming from Φ vs
terminal is: terminal contribution `γ^T · ±1` for T = turns-to-end, plus
Φ-shape contribution `0.3 · Σ γ^k (Φ_{k+1} − Φ_k)` ≈ `0.3 · (Φ_T − Φ_0)`. At
T=80, γ^80 ≈ 0.45 and the Φ swing is bounded (each term ≤ 1, ~5 active terms
⇒ |Φ| ≤ 5 ⇒ shape contribution ≤ 1.5). **So Φ-shape contributes ~3× more
than the terminal label in mid-game examples.** The value head IS predicting
"this state has good local Φ," not "I will win this game." This is the
**Ng-1999 invariance failure mode**: shaping accelerates an optimum that
already exists, but if the value head is anchored to local Φ and not terminal
outcome, MCTS PUCT explores in a *biased* direction that may not align with
the terminal optimum. The asym1 result (peak 0.52, last-5 0.45) is interesting
precisely because asymmetric self-play directly attacks this by making the
terminal label CLEAR — one seat must attack, one must defend, so the terminal
distribution is no longer ambiguous.

## §4 — Audit the curriculum

Eight scripted variants in `hard_ai.rs` (counting HARD itself, used only as
benchmark):

| variant         | `garrison` | `military` | `attack` | `device` | role / what it teaches  |
|-----------------|:---:|:---:|:---:|:---:|--------------------------|
| HARD            | 3 | T | T | F | benchmark only (`should_militarise()` ⇒ 0-1 defenders early) |
| MEDIUM, EASY    | 2,1 | T | T | F | unused (placeholder) |
| DEVICE_RUSH     | 2 | T | T | T | teaches: react to a countdown — but learner has no army to crack |
| ARMY_RUSH       | 3 | T | T | F | teaches: defend / counter-army — but cap=1 ⇒ can't |
| HQ_RUSH         | 2 | T | T | F | teaches: defend HQ — same cap=1 problem |
| GARRISON        | 3 | T | T | F | closes the 1-rush hole (warmonger=T ⇒ at_war from r1) |
| EXPERT          | 1 | F | T | F | teaches: out-grow with Experts |
| MARCHER         | 1 | T | T | F | demonstrates: march soldiers across the map (warmonger+march_to_enemy_hq) |

**Skills covered:**
- "react to device" ← DEVICE_RUSH
- "defend against army" ← ARMY_RUSH, HQ_RUSH
- "defend permanent garrison" ← GARRISON
- "build economy" ← EXPERT (negatively: lose if you don't out-grow)
- "march army" ← MARCHER (**but the learner cannot copy it without a march
  intent — §2**)

**Skills NOT covered by any opponent:**
- **Sustained pressure across multiple fronts.** No scripted opponent attacks
  on two sides — all are mono-strategy.
- **Bridge usage to circumvent a river-blocked map.** No opponent FORCES the
  learner to build a Bridge to win. The river-separated seeds in worldgen
  (DEEP-REDESIGN §3.4) are not selected for in the bench.
- **Combined economy+army.** EXPERT is military:false. ARMY_RUSH does econ
  minimally. There is no "balanced opponent" that the learner must beat by
  matching both lines.

**Mirror PFSP self-play — what's its value given symmetric attractor is
destructive?**

PFSP (Past-Frozen Self-Play) is enabled via `--pfsp`. Empirical evidence
(META-ANALYSIS §2.5; r2's bankShare climb 0.18→0.27 over 290 iters; replays
show 0-soldier games): the learner mirrors itself toward "do nothing →
opponent self-bankrupts → +1." This is the Nash collapse that
`az-pass-collapse-fix` documents and `az-draw-attractor` predicts.

asym1 is the rebuttal — it goes asymmetric (seat 0 must attack, seat 1 must
defend; or scripted vs learner), forcing the value target to be DECISIVE. Its
0.52 peak is the best across all 14 runs. The data says: **PFSP alone is
counterproductive in this game**; PFSP + asymmetric role-forcing is the only
configuration that has produced a > 0.50 peak.

## §5 — The data: what 12 runs prove

Per-run last-5-bench mean, peak, and key behavioural metrics, derived from each
`benchmark-history.jsonl`:

| run    | n  | tW peak @ gen | tW last-5 | maxSold | OP/g | Br/g | Pass% | bnk% | devDeny | Ex/g |
|--------|---:|:-------------:|----------:|--------:|-----:|-----:|------:|-----:|--------:|-----:|
| b1     | 11 | 0.450 @ g30   | 0.410     | 0.76    | 0.17 | 0.00 | 33.8% | 0.21 | 0.28    | 0.00 |
| i1     | 11 | 0.467 @ g15   | 0.390     | 0.75    | 0.17 | 0.00 | 38.9% | 0.18 | 0.24    | 0.00 |
| s1     |  7 | 0.367 @ g30   | 0.347     | 0.61    | 0.10 | 0.00 | 38.5% | 0.23 | 0.32    | 0.00 |
| s2     |  7 | 0.367 @ g15   | 0.347     | 0.62    | 0.10 | 0.00 | 38.8% | 0.26 | 0.35    | 0.00 |
| s3     | 11 | 0.433 @ g15   | 0.370     | 0.63    | 0.14 | 0.00 | 40.1% | 0.22 | 0.28    | 0.00 |
| c1     |  8 | 0.400 @ g25   | 0.327     | 0.61    | 0.17 | 0.00 | 53.5% | 0.21 | 0.25    | 0.00 |
| bc1    |  7 | 0.383 @ g25   | 0.337     | 0.64    | 0.16 | 0.00 | 47.9% | 0.23 | 0.32    | 0.00 |
| bc2    |  8 | 0.400 @ g30   | 0.360     | 0.64    | 0.17 | 0.00 | 47.9% | 0.15 | 0.23    | 0.00 |
| r1     | 11 | 0.450 @ g15   | 0.343     | 0.62    | 0.19 | 0.010| 17.9% | 0.16 | 0.27    | 0.01 |
| r2     | 31 | 0.450 @ g10   | 0.213     | 0.57    | 0.12 | 0.030| 58.0% | 0.27 | 0.27    | 0.01 |
| r3     |  9 | 0.417 @ g10   | 0.310     | 0.54    | 0.34 | 0.373| 49.6% | 0.11 | 0.60    | 0.01 |
| r4     | 14 | 0.533 @ g15   | 0.263     | 0.57    | 0.19 | 0.207| 50.9% | 0.13 | 0.62    | 0.01 |
| r5     |  5 | 0.433 @ g0    | 0.313     | 0.57    | 0.34 | 0.373| 46.5% | 0.11 | 0.55    | 0.02 |
| **asym1** | 5 | **0.517 @ g0** | **0.450** | **0.83** | **0.44** | **0.383** | 38.9% | **0.04** | **0.47** | 0.00 |

**The 0.45 ceiling — what's its source?**

Look at b1, i1, r1, r4, asym1: every one of them peaks at tW = 0.45-0.53 in
the *first 15-30 generations* of training. The warmstart-init runs (r2, asym1)
have their peak at gen 0 or gen 10, i.e. **before any new examples have
cycled through the buffer**. This says the ceiling is a **property of the
warmstart-net's policy distribution**, not a property of the training process.
Trained-from-scratch runs (b1, i1, s1, s2, s3) need 15-30 gens to reach it.
The training process then *moves away* from this ceiling over 30+ generations.

**Why does it ALWAYS regress?** The buffer (60k examples, ~2,300/iter at
`games=24`) fully cycles every ~26 iters. After cycling, ALL training data
is policy-generated. If the policy at iter 30 is "build farms, pass, hope
HARD self-bankrupts," the buffer at iter 60 is 100% that distribution, and
the value head learns "passing is fine, the game ends in our favour ~45% of
the time." This is **buffer-cycling positive feedback** (META-ANALYSIS §3).

**Was net capacity ever the binding constraint?** Refuted across the 12 runs:
the *same* 9.8k-param small net reached tW=0.467 (i1) and tW=0.213 (r2). The
big net (53.7k) tested in a prior round did not move the win-rate. The
representation IS sufficient; the data is not.

**Peak-at-gen-10-then-regress, mechanically.** The mechanism is:
1. warmstart contains residual exploration (Φ-attractive moves vary across
   warmstart seeds).
2. first 5-15 iters: buffer mixes warmstart trajectories + new explore trajectories.
3. gens 15-30: buffer is ~50% warmstart-style, ~50% new policy. Peak coincides
   with the *highest diversity* in the buffer.
4. gens 30-50+: buffer is 100% current policy. The Nash mirror (or 1-rush
   exploit) dominates. Policy locks.

The 12-run signature **is** the buffer-cycling fingerprint. No Φ tweak,
opportunistic-discount, or new intent has broken it. **The data distribution is
the binding constraint.**

## §6 — Design proposal: what changes

The redesign addresses each user concern by attacking ALL FOUR pillars:
perception, action, reward, and data distribution.

### 6.1 Perception changes — three load-bearing edits

**(P1) Add `C_DIST_TO_ENEMY_HQ` and `C_DIST_TO_ENEMY_DEVICE` planes** (channels
24, 25). Each cell holds `1 − clamp01(d / max_distance)` where d is the
Manhattan distance to the nearest enemy HQ (resp. device). With this, the
trunk no longer needs the receptive field to span the board for
device-cracking and pre-emptive offense — the gradient field is baked into
the perceptual layer. This is the single largest perceptual leverage point
because it converts a board-spanning correlation problem into a local one.
**~30 LOC in `planes.rs`, parity-locked with TS mirror.** Bumps PLANE_COUNT
24 → 26, requires cold-start, arc-bump `sd2` → `sd3`.

**(P2) Add a `C_MY_BUDGET` broadcast plane** symmetric to `C_ENEMY_BUDGET`.
Today plane 23 shows `(max enemy mobile-soldier budget / 6).min(1)` but there
is NO matching plane showing the LEARNER's own budget. The asymmetry is a
perception bug. A defender that perceives "enemy budget is 4, mine is 1"
should garrison; a defender perceiving only "enemy budget is 4" cannot
compare. **~10 LOC.**

**(P3) Dilated trunk for the big net.** Replace the residual `conv3` with a
dilation=2, 3×3 conv. The effective receptive field jumps to 9×9 (covers
~60% of a 14×12 board centred). Same parameter count, same forward-pass FLOPs.
For the SMALL net (recommended for fast iteration per TRAINING-APPROACH §2.5),
keep 2-conv but add dilation=2 to conv2: RF goes 5×5 → 7×7. **~30 LOC in
`cnn.rs` + `spatial_net.rs`.**

The combined P1+P2+P3 increase brings the net's perceptual access to a level
where the 5 user concerns are all *fully addressed at the perception layer*.
After this, any remaining gap is in action or reward, not eyes.

### 6.2 Action-space changes — the marching intent

**(A1) Add `Intent::MarchSoldier`** to `candidates.rs`. Enumerates: for each
own soldier not already on the enemy frontier, the candidate is
`Action::MoveSoldier(unit, from, to)` where `to` is the tile in
`get_available_tiles_for(p)` minimising Manhattan distance to the nearest
enemy HQ (or device, with priority device > HQ). Local features: distance
delta achieved by the move, frontier flag at destination, own-soldier-density
at destination (to bias massing). **This is the single most important action-
space addition** — without it the policy cannot LEARN to march even when the
MARCHER scripted opponent demonstrates it.

**~80 LOC + TS parity mirror + golden re-export.** INTENT_COUNT 15 → 16, cold
start, arc bump.

**(A2) Promote `HireSoldier` to multi-candidate**: one candidate per
{my HQ tile, threatened-frontier tile, Outpost tile, generic owned tile}. Today
the candidate's `tile` is chosen heuristically before the policy sees it. This
prevents the policy from EXPRESSING the preference for "garrison HQ vs
reinforce frontier" — the policy gets one HireSoldier candidate and must pick
or pass. Splitting it gives the policy a real choice. **~40 LOC.**

### 6.3 Reward changes — replace shape-weight with terminal-only + advantage

**(R1) Drop `--shape-weight` to 0** (terminal-only return). The §3 calculation
shows mid-game `z` is ~3× more Φ than terminal at shape_weight=0.3. The value
head is learning the WRONG target. With shape_weight=0, the value head learns
to predict the actual game outcome and MCTS can use that signal.

**(R2) Add a `--win-shape` flag**: scale terminal z by
`mag · (1 + α · honest_signal)` where `honest_signal =
clamp01(max_owned_soldiers/3) · (won_by_conquest ? 1.0 : 0) + (built_outpost
? 0.2 : 0) + (cracked_device ? 0.3 : 0)`. This **modifies the underlying
terminal MDP** (NOT a Φ shaping — Ng-1999 doesn't apply) to make "win by
army" terminally MORE rewarding than "win by 1-rush". Recommended α=0.5.

**(R3) Add a discrete `--garrison-credit`** that nudges `+0.1` to the per-turn
shaped target of HireSoldier-onto-own-HQ decisions when the game ends in a
win for that seat. Mirror of `--device-credit` but on the defense side. Tiny,
ablated by default.

### 6.4 Curriculum changes — drop pure self-play, anchor with asymmetric

**(C1) Lock asym1's recipe**: ≥ 50% of training games are asymmetric (one
seat scripted, the other learner). asym1 is the only run that broke the 0.50
peak ceiling. Make this the default.

**(C2) Add a `BalancedHard` opponent** — econ + army + opportunistic device.
Mirror of HARD but with `should_militarise()` always true and a soft device
gate. Fills the "balanced opponent" gap from §4.

**(C3) Drop pure self-play entirely from training**, retain it only for
benchmarking. Or cap PFSP at 10% of training games. The Nash collapse is
mechanically certain; the only escape is data distribution shift.

### 6.5 Training loop changes — supervised pretraining + KL anchor

**(T1) AlphaStar-style supervised pretraining.** Generate 50k (state, intent,
chosen-action, terminal-z) tuples from MARCHER-vs-MARCHER and HARD-vs-HARD
games (~30 min on M2). Train the net 10 epochs cross-entropy on intent + MSE
on z. **This is the centrepiece.** Six weeks of RL has not discovered the
army-marching strategy because it's outside the explored manifold. Imitation
is the proven recipe for an action-rich strategy game where one expert (the
MARCHER) demonstrates a strategy the agent cannot self-discover.

**(T2) KL anchor during RL**: add `λ · KL(π_current || π_supervised)` to the
policy loss. λ=0.5 → 0.1 decay. Prevents the 1-rush attractor from being
reachable in policy space.

**(T3) Sliding buffer with curriculum-rebalancing.** Track per-opponent
fractions in the buffer; downsample when one opponent dominates > 40%.
Prevents the buffer-cycling positive-feedback.

### 6.6 Net architecture changes

Keep the small net for iteration speed (TRAINING-APPROACH §2.5 stands).
**Add a non-spatial "strategic" head**: a Dense layer that takes ONLY the 12
value_scalars + 4 derived (rounds_played, is_device_window, my_max_soldier,
enemy_max_soldier) and outputs a 8-dim "strategy embedding" concatenated onto
both global_embed (value head) and target_embed (policy head). Today the
value_scalars only feed the value head; the policy head doesn't see them
(`spatial_net.rs:318-344` `candidate_input`). This is a perception gap on the
policy side: the policy cannot directly read "I have 1 soldier and the enemy
has 4." **~50 LOC.**

### 6.7 The big question — AlphaZero or pivot?

**My recommendation: keep MCTS for the search-step but train the policy/value
with PPO-style advantage updates, not AlphaZero cross-entropy targets.**

Reasoning:
1. **Search horizon vs payoff horizon mismatch.** MCTS at 64 sims with
   turn-search reaches depth ~30 rounds along the most-visited line. The
   Device payoff is at round 90. **The forward model cannot be unrolled deep
   enough at 64 sims to ground the value head.** PPO bootstraps the value
   head from actual game returns + GAE, with no horizon limit from the search
   tree.
2. **AlphaZero assumes the value head provides reliable leaf evaluations.**
   In a game where the optimal play distribution is dominated by a 1-rush
   exploit, the value head bootstraps off itself. PPO with GAE-λ=0.95 mixes
   bootstrapped V with empirical returns, naturally damping the self-bootstrap
   bias.
3. **Lower compute cost.** PPO needs ~4× fewer environment interactions per
   gradient step (no MCTS overhead). The 100-sim MCTS overhead per decision
   was empirically a 2.5× wall-clock multiplier; PPO removes that.
4. **The empirical hint.** The DeepMind StarCraft II AlphaStar paper used
   imitation + PPO + KL anchor + league, NOT AlphaZero. Strategy games with
   asymmetric, action-rich, long-horizon payoffs have not historically
   produced strong AlphaZero results — AZ won at Chess/Go/Shogi where the
   forward model is cheap and the search horizon matches the decision
   horizon. *Colonizing Pirkanmaa is much more StarCraft-shaped than
   Chess-shaped.*

**Concrete pivot:** keep `record_replay`, the spatial net, value_scalars, the
opponent infrastructure, and `play_one_game_explore` mostly intact. Replace
`train_grad_scalars` for the POLICY head with `train_grad_ppo`: store
`log_prob_old`, compute advantage `A = z_shaped − V(s)` (with z computed by
GAE from the trajectory), apply clipped surrogate objective. The value head
trains as before on `(value − return)²`. MCTS stays for action SELECTION (it
helps target enumeration and ranking) but the policy TARGET is no longer the
MCTS visit distribution — it's the PPO surrogate loss. **~600 LOC.**

If the user is unwilling to bet on a PPO pivot, the secondary recommendation is
**Plan-A1** in the META-ANALYSIS: supervised pretrain + KL anchor + asym1
curriculum. This is the smaller-risk, smaller-cost option, ~400 LOC.

## §7 — The concrete next experiment

**Hypothesis (the most load-bearing assumption in §6):** the existing
perceptual + reward + curriculum infrastructure is *sufficient* to learn army
conquest, provided the policy initialisation is OUTSIDE the 1-rush
attractor. If true, supervised pretraining + KL anchor alone — without the
PPO pivot, without new planes, without the MarchSoldier intent — should reach
trueWinVsHard ≥ 0.55 and sustain it.

**This experiment FALSIFIES one of two:**
- if it PASSES: paradigm shift to PPO is unnecessary; perception/action gaps
  are also second-order. The fix is supervised+KL anchor. We proceed to
  refining the supervised teacher (multi-teacher blends).
- if it FAILS: supervised pretraining alone cannot hold the policy at the
  army strategy under self-play with the current reward. We then know the
  reward distribution itself is corrupted — the §6.5/§6.7 PPO+terminal-only-z
  redesign is justified, AND we have data on what specifically the
  supervised teacher's policy drifts toward (which gives us the §6.4
  curriculum target).

**Exact configuration:**

1. **NEW binary** `cnn_train_supervised`. Plays 2,000 MARCHER-vs-MARCHER games
   + 1,000 ARMY_RUSH-vs-ARMY_RUSH games + 500 HARD-vs-HARD games (parity-
   varied seeds). Records every (state, value_scalars, intent_chosen,
   action_chosen, terminal_z) tuple. Dataset ~75k examples.
2. **Supervised training**: 10 epochs, cross-entropy on intent + MSE on z.
   Output: `checkpoints-cnn-sup2/champion-supervised.json`.
3. **RL fine-tuning** from that warmstart with KL anchor:
   ```
   --train --net-size small
   --init checkpoints-cnn-sup2/champion-supervised.json
   --kl-anchor 0.5 --kl-anchor-net checkpoints-cnn-sup2/champion-supervised.json
   --kl-decay-iters 50
   --vs-hard-frac 0.0
   --script-frac 0.6
   --asym-frac 0.5
   --shape-weight 0.1
   --bankruptcy-discount 0.7
   --iters 100
   --bench-every 5
   ```
   No new planes, no new intents, no PPO. Uses the existing infrastructure
   end-to-end. Cost ~500 LOC for the supervised binary + KL-anchor flags.

**Gate (last-10 bench means, gens 50-100, 600 games):**
- **PASS**: trueWinVsHard ≥ 0.55 AND maxSoldiersPerGame ≥ 3.0 AND Pass% < 25%
  AND no regression > 0.05 over the last 30 iters.
- **FAIL**: trueWinVsHard < 0.45 by gen 50 — supervised army-rush warmstart
  does not survive self-play with the existing reward; the reward itself is
  corrupted, escalate to §6.5/§6.7.

**Expected wall-clock:** 30 min supervised data gen + 10 min supervised train
+ 100 iters × ~120 s (small net + spend-mode) ≈ 3.5 h. Total ≈ 4 h.

**What we LEARN regardless of outcome.** PASS or FAIL both *separate the
"doesn't know" from "won't do" hypotheses* — the ambiguity that has plagued
every prior memo collapses. PASS says the existing eyes / actions / rewards
are sufficient given good initialisation; the failure was always the data
distribution. FAIL is more interesting: it rules out the cheapest fix, gives
direct evidence that the AZ paradigm itself is the binding constraint on this
game, and unlocks justifying the larger §6.5/§6.7 PPO + perception redesign
with empirical data the prior 12 runs do not contain.

The currently-running `cnn-asym1` trainer is unaffected — supervised data
generation runs against a fresh binary on a separate process; the existing
trainer keeps writing to `checkpoints-cnn-asym1/`.
