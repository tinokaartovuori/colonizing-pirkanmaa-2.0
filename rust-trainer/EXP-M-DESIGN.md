# Exp M — design doc (the radical-structural plateau attack)

_Authored 2026-06-05. Builds on the 7-agent forensic + Exp L. Exp L is a tuning attack
(PFSP + device-potential + action-level reward, NO architecture change); Exp M is the
**structural** attack on the same plateau: new EYES (representation), a richer CURRICULUM,
and a first step toward LONGER LOOK-AHEAD. The three levers are A (horizon), B (eyes),
C (curriculum). Implementation priority was **B → C → A** (highest-confidence root cause
first); as of 2026-06-05 **all three are IMPLEMENTED** (A = parity-safe turn-granularity search,
`--turn-search`; the macro-action policy-head remainder is staged). All new behaviour is **flag-gated with no-op defaults** so the existing pipeline,
parity, and any warm-start are unaffected until Exp M is explicitly launched._

---

## 0. Status of the live run (read first)

At authoring time, **the `cnn_train --train` process for Exp L was NOT running** — only the
dashboard server (`serve-dashboard.ts --dir checkpoints-cnn-l --port 8787`) was alive. The Exp L
checkpoint dir last advanced at ~00:08. So Exp L appears to have stopped (crashed / killed / done)
before this session. This doc and the Exp M code were prepared per the hard constraints regardless:
nothing wrote into `checkpoints-cnn-l/`, no new training run was launched. Judge Exp L from its
existing `benchmark-history.jsonl` before launching Exp M.

---

## 1. Diagnosis (one paragraph — established, not re-derived)

The spatial-CNN AlphaZero net is stuck at a **fake ~0.51** win-rate vs HARD. The champ converged
to a pure **conquest-rush** and treats the Strange Device as a trap. The win-rate is propped by ~77
free enemy-self-bankruptcy wins; strip them and the champ is a ~0.45 **loser**. The value head is
**squashed** (valPredWin ≈ +0.04, often negative; span ~0.42 vs healthy ~2.0) — a *symptom* of
self-play being symmetric, passive, ~15%-tied coin flips, not a value-head capacity problem
(valueLoss declines fine). Mechanistic chain: `build_farm` has **no free-worker-slot gate** (dead
farms); the net never builds **Outpost/Village** so soldier cap stays ~1 and worker cap can't grow
(capacity-blindness); the **soldier plane is owner-agnostic** (`planes.rs` C_SOLDIERS counts all
soldiers regardless of owner) so the net can't perceive relative army strength; and **--sims 48**
MCTS is only ~1–2 decisions deep while conquest pays ~round 35 and the device ~round 90, so neither
the search nor the collapsed value head can see the decisive long-horizon outcome.

---

## 2. Lever B — EYES (representation). PRIORITY 1. **IMPLEMENTED (corrected eyes).**

> **Grounded in `rust-trainer/GAME-MECHANICS.md`** (the verified, USER-CONFIRMED mechanics spec).
> Every plane/scalar below cites the § it is faithful to. The prior B revision shipped a **WRONG
> threat model** (`C_THREAT` = a tile orthogonally adjacent to an enemy soldier's CURRENT cell). That
> is incorrect: there is NO movement range / move-budget / has-moved flag (§1) — a unit moves in one
> action to ANY tile in `getAvailableTiles()`, so a soldier anywhere in enemy territory threatens the
> WHOLE enemy frontier. Threat is **frontier-reachability × mobile-army budget** (§4), not
> soldier-cell adjacency. This revision REPLACES `C_THREAT` and adds the missing distinctions.

### What was REMOVED / REPLACED vs the prior B
- **REMOVED `C_THREAT` (soldier-cell adjacency).** Replaced by `C_ENEMY_REACH` (plane 16): the union
  over live enemies of each enemy's `getAvailableTiles()` — computed EXACTLY as the engine does
  (owned ∪ orthogonal-4 border, minus unbridged-owned-river expansion, minus that enemy's own
  un-conquered HQ). This is every cell an enemy can stage attackers on next turn (§1, §4).
- **RENAMED** `C_MY_SOLDIERS`/`C_ENEMY_SOLDIERS` → `C_MY_OWNED_SOLDIERS`/`C_ENEMY_OWNED_SOLDIERS`
  (these only ever counted **owned defenders** in `tile.units`), and **ADDED** the missing
  **conquering (staged-attacker)** soldier planes per side — they are a distinct list with a distinct
  combat role (§2, §3).

### Final plane layout (`crates/cp-ai/src/planes.rs`, `PLANE_COUNT = 24`)

| idx | name | semantics | faithful to § |
|----:|------|-----------|---------------|
| 0 | `C_MINE` | owned by me | ownership |
| 1 | `C_ENEMY` | owned by any live enemy | ownership |
| 2 | `C_NEUTRAL` | unowned | ownership |
| 3 | `C_MY_HQ` | my un-conquered HQ (special; its loss = death) | §7 |
| 4 | `C_ENEMY_HQ` | any live-enemy HQ | — |
| 5 | `C_PRODUCER` | producer building present (staffing-agnostic) | — |
| 6 | `C_MILITARY` | **Outpost = impregnable** by assault (binary) | §3 |
| 7 | `C_DEVICE` | Strange Device present | §6 |
| 8 | `C_MY_OWNED_SOLDIERS` | my **owned defenders** `(n/5)∧1` | §2, §5 |
| 9 | `C_HQ_CONNECTED` | my HQ-connected mask (BFS, orthogonal-4) | §7 |
| 10 | `C_T_GRASSLAND` | terrain | §8 |
| 11 | `C_T_FOREST` | terrain (Forest ∪ AbundantForest) | §8 |
| 12 | `C_T_MOUNTAIN` | terrain | §8 |
| 13 | `C_T_RIVER` | terrain | §8 |
| 14 | `C_PRODUCING` | my producer actually generating this turn | §5 |
| 15 | `C_ENEMY_OWNED_SOLDIERS` | live-enemy **owned defenders** `(n/5)∧1` | §2 |
| 16 | `C_ENEMY_REACH` | **enemy frontier-reachability** = ∪ enemy `getAvailableTiles()` (REPLACES `C_THREAT`) | §1, §4 |
| 17 | `C_MY_REACH` | my `getAvailableTiles()` (where I strike/expand) | §1 |
| 18 | `C_MY_CONQ_SOLDIERS` | my **conquering** (staged-attacker) soldiers `(n/5)∧1` | §2, §3 |
| 19 | `C_ENEMY_CONQ_SOLDIERS` | enemy **conquering** soldiers `(n/5)∧1` | §2, §3 |
| 20 | `C_ATT_MINUS_DEF` | signed attacker−defender `(diff/5)` from my perspective (strict `>`, tie→defender) | §3 |
| 21 | `C_DEVICE_DEFENSELESS` | device tile holds ZERO owned defenders → one attacker cracks it (binary) | §2, §6 |
| 22 | `C_RIVER_BLOCK` | my unbridged owned river = expansion dead-end (binary) | §8 |
| 23 | `C_ENEMY_BUDGET` | **BROADCAST** strongest enemy's mobile-soldier budget `(b/6)∧1` — gates plane 16 | §4, §5, §6 |

**Mobile-army gating (plane 23).** `C_ENEMY_REACH` is only a *real* threat if the enemy can field
`(my soldiers there)+1`. Plane 23 broadcasts the strongest live enemy's deployable-soldier budget =
its already-fielded soldiers (all mobile — no move budget, §1) + new soldiers it can afford
(`min(Money/200, Metal/50)`) capped by its REMAINING **Device-aware** soldier slots (the cap-halving
is baked into the cached `free_soldier_amount`, §5/§6). The conv trunk can multiply reachability ×
budget to weight where the enemy can actually act.

**Device double-edge (§6).** Plane 7 (present) + plane 21 (defenseless: a standing Device has
`hasSpaceForUnits()==false` → zero owned defenders) + the broadcast budget (which reflects the
builder's own halved cap) let the net see that building a device weakens its own defense.

### Value/global scalars (`value_scalars` in `cnn_train.rs`, `VALUE_SCALAR_DIM = 12`)
Unchanged from the prior B (already faithful and reconciled — NOT duplicated by the new planes):
8 base scalars + `rel_army` (signed relative army), `soldier_headroom` (REMAINING soldier cap —
capacity-blindness fix, §5), `worker_headroom` (remaining unit cap, §5), `enemy_device_threat`
(enemy countdown progress, mirror of `my_countdown`, §6). The capacity-blindness scalars 9/10 satisfy
the brief's per-side free-slot requirement for the VALUE head; the new planes add the SPATIAL eyes.

### Touch-points (B)
- `crates/cp-sim/src/managers.rs`: add read-only `get_available_tiles_for(player)` (refactor of
  `get_available_tiles`, identical logic) so the planes extractor can compute each enemy's frontier.
  Parity-free (no rule/cost/gate change; never on the parity path).
- `crates/cp-ai/src/planes.rs`: `PLANE_COUNT 17 → 24`; remove `C_THREAT`; rename owned-soldier
  planes; add reach (16/17), conquering soldiers (18/19), att−def (20), device-defenseless (21),
  river-block (22), broadcast enemy budget (23); new unit tests assert the reachability/owned-vs-
  conquering/device-defenseless/river-block semantics.
- `crates/cp-ai/src/spatial_net.rs`: FD gradient-check `combined_grad_fd_expm_widths` updated to
  `plane_count=24`, `value_scalar_dim=12` (RUN, passes).
- `crates/cp-train/src/bin/cnn_train.rs`: no edit needed — `PLANE_COUNT`/`VALUE_SCALAR_DIM` flow
  through the constant.

### Parity / cold-start (B)
- **PARITY-FREE.** `planes.rs`, `value_scalars`, `spatial_net.rs` are AZ-only; the cp-sim addition is
  a read-only query that changes no candidate gate / cost / game rule. `candidates.rs` was NOT
  touched. **Parity 8/8 is unaffected** (no golden re-export needed).
- **COLD-START FORCED.** Plane count 17→24 changes the net input dims; the dim-guard auto-cold-starts
  when `--init`'s dims mismatch, but Exp M is launched **without `--init` (cold)** to be explicit.
  FD gradient-check extended to the new width + run green.

### Metrics that prove B works
- `valPredWin` rises toward +0.5..+1.0, `valPredLoss` toward −0.5..−1.0; **win−loss SPAN widens**
  past ~1.0.
- champ `deviceBuildRate` rises and `deviceSurvival` rises (it can now SEE the frontier threat to a
  device + the device's defenselessness, and garrison/raid accordingly).
- Bench intent histogram: `BuildOutpost` / `HireSoldier` rise; soldier-cap utilization climbs.

---

## 3. Lever C — CURRICULUM (training rules). PRIORITY 2. **IMPLEMENTED (2026-06-05).**

> **History note:** an earlier revision of this doc marked C.1 "IMPLEMENTED" before the code
> existed (the flags `--script-opponents`/`--script-frac` were referenced but unimplemented, and no
> `ScriptKind`/`Opponent::Script` was in the binary). The 2026-06-05 Lever-C pass below is the
> ACTUAL implementation. C.2 is now also implemented as a per-decision credit (`--device-credit`).

### C.1 Scripted strategy opponents in self-play — **IMPLEMENTED (device-rusher + army-rusher).**

Self-play vs the self-twin produces symmetric, ~15–20%-tied coin-flips, which is why the value head
stays squashed (no clean ±1 signal). Lever C injects **decisive structure**: two *scripted* strategy
opponents the learner MUST learn to beat/defend. Both are `HardAi` with skewed `AiParams` — NOT new
agents and NOT new game rules, so they play the full game legally through the EXISTING candidate
enumeration and are parity-irrelevant (training-side only).

- **Device-rusher** (`ScriptKind::DeviceRush` → `cp_ai::hard_ai::DEVICE_RUSH_PARAMS`): banks a
  minimal economy (`reserve 120`, `expand 3`, `nuclear off`), and the existing
  `build_strange_device` gate (`rounds ≥ 18`, not-losing, affordable — GAME-MECHANICS §6) makes it
  build the Device as early as allowed; `max_outposts 1` lays the single +3-cap Outpost the Device
  precursor needs, and a tiny strike force (`strike_force 1`, `assaults_per_turn 1`) keeps it on
  defense (the `military` phase already rings the Device's approaches). `attack` stays on so it can
  still crack an enemy Device. Faithful to §6: the device tile holds zero defenders and the build
  halves its own soldier cap, so this rush is *defensively fragile* — the learner can punish an
  over-extended / undefended device-rush, and otherwise must out-race or raid it.
- **Army-rusher** (`ScriptKind::ArmyRush` → `cp_ai::hard_ai::ARMY_RUSH_PARAMS`): maxes soldier
  capacity (`max_outposts 7` — each Outpost is +3 cap, §5), expands hard (`expand 6`), and presses
  the assault every turn (`strike_force 10`, `assaults_per_turn 10`, `warmonger true`). `device off`
  — it commits to the army win. The `attack` phase only targets non-Outpost tiles where it
  out-numbers the defender (strict `>`, §3). The learner must build defensive capacity (the exact
  capacity-blindness gap) to survive it.

Wiring: new `Opponent::Script(ScriptKind)` variant in `cnn_train.rs` (alongside `SelfTwin`/`Hard`/
`Frozen`); the per-game opponent assignment draws a scripted opponent for a fraction `--script-frac`
of the NON-vs-hard games (deterministically per game-seed, even split between the two strategies),
coexisting with PFSP/Frozen/SelfTwin (script is tried first, then PFSP, then self-twin). Only seat 0
(the learner) records, exactly like Hard/Frozen. Gated behind **`--script-opponents`** (presence,
default OFF) and **`--script-frac <f>`** (default 0.0 = no-op, clamped [0,1]) — so with both at their
defaults behaviour is byte-identical. Per-strategy learner win-rate is logged as `spVsDeviceRush` /
`spVsArmyRush` (+ their game counts) in `log.jsonl` for the dashboard.

> A fully hand-scripted (non-HardAi) policy remains future work; the biased-`AiParams` form is the
> cheapest faithful realisation that stays legal and parity-irrelevant.

### C.2 Action-level device credit — **IMPLEMENTED (`--device-credit`).**

Replaces the diffuse whole-game |z| reweight (`--device-bonus`, which scales *every* decision's |z|
in a device-decided game) with PER-DECISION credit. Each recorded example now carries the **chosen
intent** and whether the acting seat **owned a standing device** at that state. In a game that ENDS
in a Device win, after the terminal/shaping `z` is assigned, `--device-credit c` (default 0 = exact
no-op):
- nudges the WINNER's device-COMMIT (`BuildStrangeDevice`) and device-DEFEND (`HireSoldier` while
  owning a standing device) decisions by `+c` toward +1; and
- nudges a seat that OWNED a standing device but LOST the device race by `−c` toward −1 on its
  PASSIVE decisions (anything that is neither committing to nor defending its own device) — teaching
  it not to throw a winnable device.

Each adjusted `z` is re-clamped to [-1, 1]. This is a value-TARGET adjustment on the exact decisive
decisions (not a potential term), so unlike `--device-bonus` it gives *local* credit. It is
independent of and composes with `--device-potential` (the Ng-1999 telescoping Φ term from Exp L)
and `--device-bonus` (still available). Default-off → no change.

### C.3 Handicap / asymmetric start — **DESIGN-ONLY.**

A clean value signal wants some games to be decisively won/lost rather than symmetric. A
`--handicap-start F` flag would, in a fraction F of self-play games, give one seat a small starting
resource/tile edge so the game resolves decisively and the value head gets a clean ±1 signal. Left
design-only: it touches game setup (must not perturb the MSVCRT map-gen RNG order — the handicap must
be applied AFTER `generate_map`/HQ placement as a post-hoc resource grant, which is feasible but
needs its own determinism check). Documented for a follow-up.

### Touch-points (C) — what was ACTUALLY edited (2026-06-05)
- `crates/cp-ai/src/hard_ai.rs`: add `pub const DEVICE_RUSH_PARAMS` / `ARMY_RUSH_PARAMS` (biased
  `AiParams`) + `HardAi::device_rush()` / `HardAi::army_rush()` constructors.
- `crates/cp-ai/src/lib.rs`: re-export the two new params constants.
- `crates/cp-train/src/bin/cnn_train.rs`:
  - `enum Opponent` add `Script(ScriptKind)`; new `enum ScriptKind { DeviceRush, ArmyRush }` with
    `make_bot()`; inner `enum OppKind` add `Script(ScriptKind)`.
  - `play_one_game_explore`: opponent-seat bot is the scripted variant when `opp` is `Script`;
    `ExploreOutcome` gains `script_opp: Option<ScriptKind>`.
  - per-game opponent assignment draws a scripted opponent (deterministic per seed, even split)
    for `--script-frac` of the non-vs-hard games, before the PFSP/self-twin fallback.
  - `TrainCfg` add `script_opponents: bool`, `script_frac: f64`, `device_credit: f64` (+ defaults,
    flag parsing `--script-opponents` / `--script-frac` / `--device-credit`, startup println, usage).
  - per-iter log adds `spVsDeviceRush`/`spVsDeviceRushN`/`spVsArmyRush`/`spVsArmyRushN`.
  - `Example` gains `chosen_intent` + `owned_standing_device` (captured in the harvest); a post-`z`
    action-level **device-credit** pass implements C.2 (default-off no-op).
  - 4 new unit tests (device-rusher builds a device; army-rusher fields soldiers + assaults;
    `--script-opponents` routing tag; device-credit no-op-at-0 + clamp).
- (C.3 handicap — no code; documented below.)

### Parity / cold-start (C)
- **PARITY-FREE, no cold-start.** Scripted opponents are training-side HardAi variants; they do not
  touch candidate gates / costs / rules / net inputs. Default-off → existing behaviour byte-identical.

### Metrics that prove C works
- New per-iter log fields `spVsDeviceRush` / `spVsArmyRush` learner win-rates (dashboard panel).
- A learner that's truly improving beats BOTH scripted strategies (>0.6) — not just self-twin.
- Self-play tie-rate drops; decisive-by-cause shows real device/defense games.
- The value head un-squashes: `valPredWin` rises toward +0.5…+1.0 and the win−loss SPAN widens past
  ~1.0 (the scripted games supply the clean ±1 signal the symmetric coin-flips lacked).

---

## 4. Lever A — HORIZON / LOOK-AHEAD. PRIORITY 3. **IMPLEMENTED (turn-granularity search, parity-safe) + remainder STAGED.**

The decisive events (conquest ~r35, device ~r90) are far beyond the reach of the 48-sim MCTS, so
neither the search nor the squashed value head sees them. The chosen fix is **A.1 turn-granularity
search** in its cleanest parity-safe form — implemented 2026-06-05, flag-gated `--turn-search`.

### Root cause re-stated precisely (corrected the diagnosis)

The CNN MCTS (`cnn_train.rs`) node granularity is **one candidate (one intent)** and the expansion
path (`simulate` → `execute_action(one intent)` → `advance_after_root` → `end_turn()`) **ends the
root's turn after a SINGLE intent**. So the search tree did NOT just under-reach in depth — within
each searched line it modelled a **crippled root that takes exactly one action per turn**. A root
that can only do one intent/turn can never afford an Outpost (650) or the Device (1300) inside the
search, so every depth-N leaf the value head saw was a starved, one-action-per-turn future — utterly
unlike real play (a turn = many intents, GAME-MECHANICS §9). The value gradient therefore learned
from an impoverished lookahead that structurally cannot reach the long-horizon device payoff, while
the *real* self-play turn loop spends the full budget. Fixing the **fidelity of the lookahead's turn
model** is what lets the search/value reach the decisive outcomes.

### A.1 (IMPLEMENTED) — turn-granularity search via "complete the root's turn"

In MCTS-DESIGN terms this keeps the **per-candidate PUCT node/edge structure** (priors = softmax over
`score_candidate_into`, leaf value = the value head, opponents = forced deterministic HARD turns —
all unchanged) and changes ONLY what an *expanded* root edge does:

- **OFF (default, pre-Lever-A):** `execute_action(searched intent)` → `advance_after_root` (= immediate
  `end_turn` + forced opponent turns). One intent, then the turn ends.
- **ON (`--turn-search`):** `execute_action(searched intent)` → **`Mcts::complete_root_turn`** → then
  `advance_after_root`. `complete_root_turn` runs the root through the REST of its turn with the net's
  own greedy (temperature-0) policy — the **same** `enumerate` → `score_candidate_into` argmax →
  `execute_action` + `scaffold_staff` loop the deployed controller (`controller.rs::plan_turn`) runs —
  up to the remaining budget (`cfg.budget - 1`), stopping on Pass / no multi-candidate decision /
  budget exhausted / a mid-turn elimination. So **one MCTS edge now advances a FULL, FAITHFUL turn**:
  the root develops its economy/army within the searched line exactly as it will in play, and the
  value head evaluates *realistic* turn-boundary states many rounds deep.

The searched (branched) decision is still the FIRST intent of the turn at the root state; the rest of
the turn is filled deterministically by the policy (a "rollout-within-the-edge"). This is the smallest
increment that gives turn-granularity without re-architecting the node/edge type or the policy target.

Applies to BOTH `mcts_select_explore` (self-play) and `mcts_select` (greedy bench/deploy MCTS), so the
benchmark measures the same horizon as self-play. The diagnostic/smoke MCTS callers pass `false`.

### What was NOT done (honest, staged remainder)

- **Macro-action policy head** (`{InvestEconomy, BuildArmy, StartDevice, Defend, Raid}` as the policy
  TARGET) — NOT implemented. That would change the net's output space → force a cold-start AND, since
  the DEPLOYED TS net does its own turn loop over the 12 intents, a TS mirror + golden re-export. Out
  of scope for a parity-safe increment. The turn-search above gives the horizon benefit at the SEARCH
  level without touching the 12-intent policy head, so it is the right first increment.
- **Searched intra-turn decisions** (branching on the 2nd, 3rd … intent of the same turn) — NOT done;
  the completion is greedy, not searched. A future step could let the first K intra-turn intents
  branch (still parity-free), at K× the node cost.
- **A.2 root macro-prior flooring** / `--macro-bias` — NOT implemented (superseded as the primary
  lever by the turn-search above; `--build-prior-floor` already props starved builds).

### Parity / cold-start (A)
- **PARITY-FREE, NO cold-start.** `--turn-search` changes only the training-time SEARCH internals
  (what an expanded edge simulates). The net input/output (24 planes / 12 value scalars / 12-intent
  policy head), `candidates.rs`, `enumerate`, costs/gates, map-gen RNG, and the recorded-example shape
  (root state + π over the root candidates) are all untouched. Search is never on the parity path.
  **Verified parity 8/8** (1600 decisions, 4800 fingerprints) with the code in place. Default OFF →
  B+C-only runs are byte-identical, so B+C vs B+C+A is a clean A/B.

### A flag (no-op default)

| flag | default | effect |
|------|---------|--------|
| `--turn-search` | off (false) | each MCTS edge advances a FULL turn (root completes its turn via greedy policy after the searched intent) → tree depth = rounds; `--sims` reaches conquest ~r35 / device ~r90. OFF = one-intent edges (pre-Lever-A). |

`--sims` (existing) remains the orthogonal depth dial; with turn-search ON each sim is one full turn
deep, so the SAME 48 sims now reach ~tens of rounds of faithful lookahead instead of one-intent stubs.

---

## 5. Flags added (all no-op defaults)

| flag | default | lever | effect |
|------|---------|-------|--------|
| (none — B is structural) | — | B | plane/scalar dims change → cold-start on launch |
| `--script-opponents` | off | C.1 | enable scripted device/army-rush opponents in self-play |
| `--script-frac F` | 0.0 | C.1 | fraction of non-vs-hard games using a scripted opponent (0 = no-op) |
| `--device-credit C` | 0.0 | C.2 | per-decision device-build/defend credit (±C) in device-decided games (0 = no-op) |
| `--turn-search` | off | A | each MCTS edge advances a FULL turn (root completes its turn via greedy policy) → tree depth = rounds (off = one-intent edges, pre-Lever-A) |
| `--record-opp-value` | off | C (round-2) | record the SCRIPTED opponent SEAT's trajectory as VALUE-ONLY examples (value head only, no policy grad) → the value head sees the WINNING side's +1 (the round-1 value-squash fix) |
| `--script-grade` | off | C (round-2) | grade the device-rush↔army-rush split by the learner's running per-strategy win-rate `(1−p_win)²` (off = even 50/50) |

Note: `--script-frac` DEFAULT is **0.0** (a true no-op even if `--script-opponents` is passed) — set
it explicitly (the launch command below uses 0.5). `--turn-search` is a presence flag, default OFF.
`--record-opp-value` / `--script-grade` are presence flags, default OFF (only seat-0 / even-split, as
round 1). (C.3's `--handicap-start` is documented but NOT yet added as a flag.)

---

## 6. Gates run for Exp M code

### Lever B (eyes) — prior pass
- `cargo build --release` clean.
- `cargo test -p cp-ai` — FD gradient-checks at the new dims (planes 24, value_scalar_dim 12) +
  planes tests for the corrected reachability / owned-vs-conquering / device-defenseless / river-
  block semantics.

### Lever C (curriculum) — 2026-06-05 pass
- `cargo build --release` — clean (only a pre-existing unrelated `cut_vs_hard` unused-import warning).
- `cargo test -p cp-ai` — green (10 lib + 10 selfplay + 3 hard_ai unit tests).
- `cargo test -p cp-train --bin cnn_train` — **14/14 pass**, incl. the 4 new Lever-C tests:
  `scripted_device_rusher_builds_a_device`, `scripted_army_rusher_fields_soldiers_and_assaults`,
  `script_opponents_flag_routes_games`, `device_credit_no_op_at_zero_and_clamps`.
- `cargo run -p cp-train --bin parity --release -j 4` = **8/8** (1600 decisions, 4800 fingerprints) —
  Lever C is training-only, parity unaffected as required.
- NO `cnn_train --train` run was launched (per the hard constraint).

### Lever A (horizon / turn-search) — 2026-06-05 pass
- `cargo build --release -p cp-train` — clean (only the pre-existing unrelated `cut_vs_hard`
  unused-import warning).
- `cargo test --release -p cp-train --bin cnn_train` — **16/16 pass**, incl. the 2 new Lever-A tests:
  `turn_search_completes_a_full_turn`, `turn_search_default_is_noop_and_on_is_legal`.
- `cargo test --release -p cp-ai` — green (10 lib + 10 selfplay + 3 hard_ai).
- `cargo run -p cp-train --bin parity --release -j 4` = **8/8** (1600 decisions, 4800 fingerprints) —
  turn-search is search-side only, NO cold-start, parity unaffected as required.
- Files touched: `crates/cp-train/src/bin/cnn_train.rs` only (new `Mcts::complete_root_turn`,
  `turn_search`/`turn_budget` on `Mcts`, `turn_search` on `TrainCfg` + flag parse + startup println +
  usage string, threaded through `mcts_select` / `mcts_select_explore` / `cnn_plan_turn`, 2 tests).
  No cp-ai/cp-sim/TS edits.
- NO `cnn_train --train` run was launched (per the hard constraint).

---

## 7. Exp M launch command (COLD-START — do NOT run until Exp L is judged)

```bash
# COLD start (no --init): plane count 15→24 (corrected eyes) and value_scalar_dim 8→12 changed the
# net input, so Exp M MUST cold-start. The dim-guard would force it even if --init were passed.
./rust-trainer/target/release/cnn_train --train \
  --out rust-trainer/checkpoints-cnn-m \
  --pfsp --vs-hard-frac 0.4 \
  --script-opponents --script-frac 0.5 \
  --device-potential 0.2 --device-credit 0.3 \
  --turn-search \
  --tie-penalty 0.5 --stall-rounds 80 \
  --build-prior-floor 0.03 --shape-gamma 0.99 --shape-weight 0.3 \
  --sims 48 --cap 150 --games 32 --bench-games 60
```

`--turn-search` (Lever A) makes every MCTS edge advance a full turn — with B's eyes and C's decisive
curriculum, the search can now actually reach the conquest (~r35) / Strange-Device (~r90) payoffs, so
the value head can un-squash on a horizon that contains the decisive outcome. To run a clean B+C vs
B+C+A A/B, launch a twin WITHOUT `--turn-search` (every other flag identical). **NOTE:** turn-search
makes each sim ~`budget`× more expensive (each edge now plays a full turn, not one intent), so expect
a throughput hit — consider lowering `--sims` or `--games` for the A run, or measure the slowdown
first.

`--device-credit 0.3` (Lever C.2, per-decision device-build/defend credit) REPLACES the prior
`--device-bonus 0.2` here as the better-targeted device signal; `--device-bonus` remains available
if you want the diffuse whole-game reweight instead (the two compose — both default to no-op).
`--script-frac 0.5` means half of the (1 − 0.4) = 0.6 non-vs-hard games are scripted (≈30% of all
games), split evenly device-rush / army-rush.

Dashboard: `npx vite-node training/serve-dashboard.ts -- --dir rust-trainer/checkpoints-cnn-m --port 8788`
(use a DIFFERENT port than the still-running 8787 dashboard).

This is a **NEW game-net generation** (cold-start, new input representation). It is NOT a
candidate-gate / cost / rule change, so the **`arc` code stays `sd`** (no game-rules change). Register
the resulting champion as the next `sd-az-NNN` per `models/README.md` when it's benched.

---

## 8. Staged / remaining work (honest)

- **A — DONE (2026-06-05, parity-safe turn-granularity search).** `--turn-search`: an expanded MCTS
  edge now completes the root's turn via the net's greedy policy (the deployed turn loop) before the
  opponents move, so one edge = one FULL faithful turn (tree depth = rounds). Parity-free (search-side
  only, no net-I/O / candidate / gate change), NO cold-start, verified **parity 8/8**. Default OFF =
  byte-identical to the pre-Lever-A one-intent-per-edge search, so B+C vs B+C+A is a clean A/B. 2 new
  unit tests: `turn_search_completes_a_full_turn` (the completion executes more than one intent) and
  `turn_search_default_is_noop_and_on_is_legal` (OFF is deterministic/unchanged; ON is a legal,
  in-range, normalised decision).
- **A remainder — STAGED (not done):** (1) a macro-action POLICY head (would change the net output →
  cold-start + TS mirror + golden re-export; deliberately out of scope for parity-safety); (2)
  branching the intra-turn 2nd/3rd… intents (the completion is greedy, not searched); (3) the old A.2
  `--macro-bias` root prior-flooring (superseded by turn-search as the primary horizon lever).
- **C.1 scripted opponents — DONE (2026-06-05).** device-rusher + army-rusher as biased `HardAi`,
  wired behind `--script-opponents`/`--script-frac`, per-strategy win-rate logged, 2 unit tests
  prove they build a device / field soldiers + assault.
- **C.2 action-level device credit — DONE (2026-06-05).** `--device-credit` gives per-decision ±C
  credit to device-commit/defend (win) and passive-device-loss decisions, replacing the diffuse
  `--device-bonus` reweight. Default-off no-op; unit-tested for no-op-at-0 + clamp.
- **C.3 handicap-start** — still DESIGN-ONLY; needs its own MSVCRT map-gen RNG-determinism check
  (apply the resource grant AFTER `generate_map`/HQ placement). Skipped this pass as the brief
  allows ("skip if risky").
- **B — DONE (corrected).** Frontier-reachability (full `getAvailableTiles()` union) + mobile-army
  budget gating now ship, REPLACING the wrong 1-step soldier-adjacency `C_THREAT`. A future
  refinement could make the budget gating per-cell (local enemy strength vs my local defenders)
  rather than a single broadcast scalar, but the broadcast budget × reachability is the dominant,
  faithful term.

---

## 9. ROUND 2 — the VALUE-SQUASH fix (2026-06-05, after round-1 B+C+A did NOT break the plateau)

### 9.1 Round-1 result + diagnosis (CODE-VERIFIED)

Round 1 (B+C+A, dir `checkpoints-cnn-m2`, sims 32 / games 24, gens 0–45) left `valPredWin` stuck
≈ 0 to −0.18 for the entire run even though games were decisive (low `spTie`), `vsDeviceRush`/
`vsArmyRush` win-rates noisy ~0.2, winVsHard flat ~0.40, trueWin ~0.27/60. The value head **never
un-squashed**.

**Root cause, confirmed in `crates/cp-train/src/bin/cnn_train.rs`:**
- `play_one_game_explore` records examples **ONLY for the learner (seat 0)** — `learner_seat = cur.0 == 0`,
  `let record = learner_seat;` (lines ~2447-2456 of round-1 code). In a `Script` game the scripted
  opponent plays seat 1 via `hard.plan_turn` in the `else` branch and **records nothing**.
- The terminal target is `terminal_z(seat) = if winner==seat {mag} else {-mag}` (lines ~2552-2558),
  `mag ≈ 1`. So EVERY recorded example of a LOST scripted game gets z ≈ −1.
- The device-rush / army-rush scripts beat the (still-weak) learner ~70-80% of the time, so the
  scripted-curriculum examples are dominated by z = −1. The value head therefore almost never
  experiences WINNING these decisive games → `valPredWin` cannot rise → the squash persists toward
  negative. **The decisive curriculum we added supplied mostly LOSING examples, not a balanced ±1
  signal.** (Hypothesis CONFIRMED, not refuted.)

### 9.2 Fixes implemented (all flag-gated, no-op defaults, parity-safe)

**Fix 2 (PRIMARY) — `--record-opp-value`: salvage the WINNING side's value signal.**
In a scripted game, ALSO record the scripted opponent SEAT's board states (one per opponent turn,
captured before `plan_turn` mutates) as **VALUE-ONLY** examples. The scripted move is not a usable
policy target, but the board *evaluated from that seat* is a clean ±1 VALUE example once terminal z is
known — and that seat WINS most lopsided games, so it supplies the +1 targets learner-only recording
lacked. New `SpatialNet::train_grad_value_only_scalars` trains the value head only (the policy head is
skipped entirely — an all-zero `pi` would NOT be safe because the softmax grad `p_c − 0 = p_c` is
non-zero; FD-checked + zero-policy-grad-checked). `Example` gains `value_only`; the main training path
(`train_batch_lr`) dispatches on it; the device-credit pass skips value-only examples; observability
(`vpred_win`) naturally now includes the winning opponent's +1 predictions. Default OFF → no
value-only examples → byte-identical to round 1.

**Fix 1 — `--script-grade`: balance the curriculum (graded split).**
The device-rush↔army-rush split is graded by the learner's running per-strategy win-rate using the
same AlphaStar `(1−p_win)²` weighting as PFSP, so the trainer samples MORE of the matchup the learner
beats LESS (tracking it toward ~50% on each). Cumulative counts decay ×0.8/iter so the split tracks
RECENT performance (a sliding window), not the whole-run average. Default OFF → even 50/50 (round 1).

**Fix 3 — `--device-credit`: assessed, NO code change.** The brief worried it rewards devices that
then die. Verified in code: the positive credit is already gated on `won_by_device(seat)` =
`device_decided && winner == seat`, so it credits a device-build ONLY in a game that seat WON by
device — it never rewards a dead/losing device. The magnitude is already a tunable. Recommendation:
LOWER it from round-1's 0.3 to **0.15** for round 2 (let the now-balanced value signal do the heavy
lifting; keep the per-decision nudge gentle).

### 9.3 Why Fix 2 is the highest-leverage choice

`--script-grade` / lowering `--script-frac` only rebalance *how many* losing games the learner sees;
they cannot make a weak learner WIN device/army rushes, so `valPredWin` would still be starved of +1
until the policy improves (chicken-and-egg). `--record-opp-value` breaks the deadlock directly: it
hands the value head the winning side's +1 from day one, independent of the learner's strength, so the
value head can calibrate (`valPredWin` → positive, span widens) and then GUIDE the search toward those
winning states. It is the cleanest fix that attacks the confirmed root cause head-on.

### 9.4 Gates run (round 2)
- `cargo build --release` — clean (only the pre-existing unrelated `cut_vs_hard` unused-import warning).
- `cargo test --release -p cp-ai` — **40 + 10 + 3 pass**, incl. the new
  `value_only_grad_zero_policy_and_fd_value` (zero policy grad + FD value-grad check).
- `cargo test --release -p cp-train --bin cnn_train` — **18/18 pass**, incl. the 2 new round-2 tests:
  `record_opp_value_default_off_and_records_winning_side` (OFF = learner-only no-op; ON records
  well-formed seat-1 value-only examples + the winning side yields +1) and
  `script_grade_split_off_is_even_on_biases_to_weaker`.
- `cargo run -p cp-train --bin parity --release -j 4` = **8/8** (1600 decisions, 4800 fingerprints) —
  all round-2 changes are training-side / AZ-only; NO candidate-gate / cost / rule / map-gen / net-I/O
  change → NO cold-start, parity unaffected.
- NO `cnn_train --train` run launched (per the hard constraint).

### 9.5 Files touched (round 2)
- `crates/cp-ai/src/spatial_net.rs`: `train_grad_value_only_scalars` + `train_grad_cached_inner`
  (shared impl with a `value_only` flag that skips the policy head); `value_only` FD/zero-grad test.
- `crates/cp-train/src/bin/cnn_train.rs`: `Example.value_only`; opponent-seat value-only recording in
  `play_one_game_explore` (gated `--record-opp-value`); graded split in the seed/opponent assignment
  (gated `--script-grade`) + decayed per-strategy win counters; `train_batch_lr` value-only dispatch;
  device-credit pass skips value-only; `TrainCfg.record_opp_value`/`script_grade` + flag parse +
  startup println + usage; 2 new tests.

### 9.6 RECOMMENDED round-2 launch command (FULL compute)

```bash
# COLD start (no --init) — same 24-plane / vsd-12 net as round 1 (no net-I/O change in round 2).
./rust-trainer/target/release/cnn_train --train \
  --out rust-trainer/checkpoints-cnn-m3 \
  --pfsp --vs-hard-frac 0.4 \
  --script-opponents --script-frac 0.5 --script-grade \
  --record-opp-value \
  --device-potential 0.2 --device-credit 0.15 \
  --turn-search \
  --tie-penalty 0.5 --stall-rounds 80 \
  --build-prior-floor 0.03 --shape-gamma 0.99 --shape-weight 0.3 \
  --sims 48 --cap 150 --games 32 --bench-games 60
```

Watch `valPredWin` (target: rises positive, span widens past ~1.0 within ~10–15 gens — the round-2
success signal) and `spVsDeviceRush`/`spVsArmyRush` (should climb as the graded split + the calibrated
value head let the learner actually contest the rushes). For a clean A/B, a twin WITHOUT
`--record-opp-value` (everything else identical) isolates the value-salvage effect.

## 10. ROUND 3 — the CAPACITY bump (2026-06-05, after round-2 fixed the value-squash but NOT the win-rate)

### 10.1 Round-2 result + diagnosis (the binding constraint is NET CAPACITY)

Round 2 (full compute, 100 iters, with `--record-opp-value`) produced a clear, important result:
- **The value head DID un-squash** — `valPredWin−valPredLoss` span: 1st-half 0.32 → 2nd-half 0.45
  (monotonic); `valWin` went −0.07 → +0.15..0.28. `--record-opp-value` works as designed.
- Late-game device denial improved (HARD's device wins dropped 15 → 9 over training).
- **BUT win-rate vs HARD was DEAD FLAT ~0.46 across ALL 100 iters** (1st-half 0.464, 2nd-half 0.462,
  never reached 0.55).

CONCLUSION: value-squash was **NOT** the binding constraint — it resolved and win-rate did not move.
The ~0.46–0.55-vs-HARD ceiling is persistent across the WHOLE project (Exp L, B-only, round 1,
round 2) and is **raw playing strength / NET CAPACITY**, not the eyes (B), the curriculum (C), the
horizon (A) or the value target (round 2). The round-2 `SpatialNet` had only **9786 params** — a tiny
CNN for a 14×12, 24-plane spatial strategy game with a 12-intent policy. "Representation/capacity
ceiling" is the recurring confirmed theme of this project. Round 3 attacks it directly.

### 10.2 The new architecture (implemented)

The trunk gained a **depth-preserving residual block** and ALL widths grew. I/O contract UNCHANGED
(24 planes in, 12-intent policy + scalar value out, per-candidate local-feature dim 18, value_scalar
dim 12) → the candidate/parity layer is untouched.

| layer | round-2 (old) | round-3 (new) | new params @ trainer I/O |
|---|---|---|---|
| conv1 (PC→D1, 3×3) | PC=24 → D1=16 | PC=24 → **D1=32** | 32·24·9 + 32 = 6 944 |
| conv2 (D1→D, 3×3) | 16 → D=24 | 32 → **D=48** | 48·32·9 + 48 = 13 872 |
| **conv3 (D→D, 3×3, RESIDUAL)** | — (none) | **48 → 48 + identity skip** | 48·48·9 + 48 = 20 784 |
| value_d1 (D+vsd → HV) | (24+12)→24 | (48+12)→**HV=64** | (48+12)·64 + 64 = 3 904 |
| value_d2 (HV→1) | 24→1 | 64→1 | 65 |
| policy_d1 (2D+local+intent → HP) | 78→24 | 126→**HP=64** | (2·48+18+12)·64 + 64 = 8 128 |
| policy_d2 (HP→1) | 24→1 | 64→1 | 65 |
| **total** | **9 786** | | **53 762** (≈ 5.5×) |

`board_embed = tanh(conv3(trunk2)) + trunk2` (identity skip; `trunk2 = tanh(conv2(...))`). The
residual block adds depth (a third 3×3 conv → effective receptive field 7×7, enough to relate an HQ
to its frontier across the board) WITHOUT the vanishing-gradient risk of a plain stacked tanh-conv,
and is exactly the AlphaZero-Go trunk idiom (conv blocks with skips). Widening conv2's channels to
48 and the heads to 64 gives the policy/value MLPs room to combine the 24 eyes + pooled embedding +
18 local + 12 intent features non-trivially (the old 24-wide heads were a clear bottleneck on a
126-wide policy input). 53.7k params is squarely in the target 30–80k band — big enough to break the
representation ceiling, small enough to stay CPU-trainable for ~100 iters.

The residual block is OPTIONAL (`conv3: Option<Conv2d>`, `#[serde(default)]`) so old checkpoints still
deserialise as the legacy 2-conv trunk; the two `default_*` constructors now build the residual arch,
so the deployed trainer (`SpatialNet::default_with_value_scalars`) gets the new arch with no call-site
changes. This is a **COLD-START** change (new weight shapes) — expected and required.

### 10.3 FD gradient-check (the non-negotiable gate) — PASS

New tests in `spatial_net.rs`, all passing (`cargo test --release -p cp-ai spatial_net`):
- `combined_grad_fd_residual_block` — FD-checks `conv3` weights+bias AND that conv1/conv2/value/policy
  grads route correctly through BOTH the conv3 path and the identity skip (use_residual=true, D=5).
- `round3_default_arch_param_count` — pins `default_with_value_scalars(24,18,12,12)` = D1 32 / D 48 /
  HV 64 / HP 64 / residual present / **param_count == 53 762**.
- All prior FD checks still pass at the new shapes (`combined_grad_finite_difference`,
  `…_local_dim_18`, `…_value_scalars`, `…_expm_widths`, `value_only_grad_zero_policy_and_fd_value`).

```
test spatial_net::tests::combined_grad_fd_residual_block ... ok
test spatial_net::tests::round3_default_arch_param_count ... ok
test spatial_net::tests::combined_grad_fd_expm_widths ... ok
test spatial_net::tests::combined_grad_fd_value_scalars ... ok
test spatial_net::tests::value_only_grad_zero_policy_and_fd_value ... ok
test result: ok. 42 passed; 0 failed; 2 ignored   (cp-ai full suite)
```

### 10.4 Gates run (round 3)
- `cargo build --release` — clean (only the pre-existing unrelated `cut_vs_hard` unused-import warning).
- `cargo test --release -p cp-ai` — **42 pass** (40 prior + 2 new), incl. the residual FD check.
- `cargo test --release -p cp-train --bin cnn_train` — **18/18 pass** (the trainer now builds + trains
  the 53.7k net end-to-end through every test path).
- `cargo run -p cp-train --bin parity --release -j 4` = **8/8** (1600 decisions, 4800 fingerprints).
  The net is AZ-only / parity-free; NO candidate-gate / cost / rule / map-gen / net-I/O change → parity
  unaffected (the cold-start is purely in trunk/head weight shapes).
- NO `cnn_train --train` run launched (per the hard constraint).

### 10.5 Throughput (measured) + launch recommendation

`bench_mcts_forward` (24 planes, 14×12 board, one node = trunk forward + 6 candidate scores + value,
N=20000) measured the new arch at **5.67× slower per node** than the old tiny net:
`OLD 9786 params 857 µs/node → NEW 53.7k params 4.86 ms/node`. The conv trunk dominates (conv2 went
4× wider in FLOPs, conv3 is a brand-new 48×48 block), and turn-search already multiplies forward
passes, so the per-iter cost scales close to this factor on the net-bound parts (self-play MCTS +
training backward, both already rayon-parallel across games/batch).

Estimate: round-2 baseline was ~70 s/iter at sims48 / games32. The net-bound work is the bulk of an
iter, so naively ~70 s × ~5 ≈ **~350 s/iter → ~10 h for 100 iters** at the old game count — too slow
for tight iteration. **Recommendation: keep sims at 48 (do NOT starve search — a stronger net needs
search depth, and turn-search makes each sim a full-turn rollout; sims is a plausible co-bottleneck,
so if anything raise it later, never cut it now), and trade game COUNT for the bigger net: drop
`--games 32 → 24`.** That recovers ~25% wall-time (more parallel slack per core) for ~270–290 s/iter
→ **~7.5–8 h for 100 iters** — tractable, while keeping per-game search quality high. If a core-count
audit shows games already under-fills the cores, games can stay at 32 (the parallelism hides the cost);
24 is the safe default. Bench can be re-run any time: `cargo test -p cp-ai --release -- --ignored
--nocapture bench_mcts_forward`.

### 10.6 RECOMMENDED round-3 launch command (COLD start — NEW 53.7k arch)

```bash
# COLD start (no --init): the residual arch is the new default → fresh weight shapes.
# All round-1/2 levers ON (record-opp-value + script-opponents + script-grade + turn-search).
./rust-trainer/target/release/cnn_train --train \
  --out rust-trainer/checkpoints-cnn-m4 \
  --pfsp --vs-hard-frac 0.4 \
  --script-opponents --script-frac 0.5 --script-grade \
  --record-opp-value \
  --device-potential 0.2 --device-credit 0.15 \
  --turn-search \
  --tie-penalty 0.5 --stall-rounds 80 \
  --build-prior-floor 0.03 --shape-gamma 0.99 --shape-weight 0.3 \
  --sims 48 --cap 150 --games 24 --bench-games 60
```

Startup line will print `params=53762` (confirms the new arch loaded). **Success signal for round 3:
`winVsHard` finally CLIMBS above the ~0.46 floor (target > 0.55) — that is the capacity hypothesis
confirmed.** `valPredWin` should stay un-squashed (round-2 gain retained); if win-rate moves while
`valPredWin` holds, capacity was the binding constraint. If win-rate is STILL flat at 53.7k params,
the ceiling is not capacity at this scale and the next lever is search/representation (raise sims, or
spatial policy output), not more width.

### 10.7 Implemented-vs-designed summary + files touched (round 3)

HONEST status: the capacity bump is **fully implemented, FD-gated, and build/test/parity-clean** —
ready to launch. NOT done (out of scope, parent runs it): the actual 100-iter training run and the
empirical win-rate verdict. The recommended games-cut (32→24) is a throughput recommendation, not yet
A/B-validated against keeping games=32.

Files touched:
- `crates/cp-ai/src/spatial_net.rs`:
  - `SpatialNet.conv3: Option<Conv2d>` (residual block, `#[serde(default)]` for back-compat);
    `BoardCache.trunk2` + `BoardCache.res_act` (retained for the residual backward).
  - `new_seeded_arch(... use_residual ...)` (full constructor); `new_seeded` now delegates with
    `use_residual=false`; **both `default_for` / `default_with_value_scalars` now build the residual
    arch (D1 32 / D 48 / HV 64 / HP 64)** — the deployed trainer net.
  - `forward_board_scalars` applies the residual block (`board_embed = tanh(conv3(trunk2)) + trunk2`);
    `train_grad_cached_inner` backprops it (conv3 tanh + identity-skip grad sum) BEFORE the conv2/conv1
    backward; `param_count`, `SpatialGrad` (+`conv3_w`/`conv3_b`), `zeros_like`/`add`/`scale`/
    `apply_grad` all handle conv3 (empty vecs when absent → no-ops).
  - New tests `combined_grad_fd_residual_block`, `round3_default_arch_param_count`; `bench_mcts_forward`
    rewritten to time OLD-vs-NEW arch at the real trainer I/O.
- NO changes to `cnn_train.rs` (the constructor swap propagates via the `default_*` widths), `planes.rs`,
  `candidates.rs`, `resources.rs`, or map-gen → **parity 8/8 unaffected**.
