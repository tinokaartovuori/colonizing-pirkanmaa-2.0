# Strange Device — game-mechanic design spec

_Status: DESIGN (not yet implemented). Authored 2026-06-03._
_Purpose: add a decisive, draw-eliminating win condition to Colonizing Pirkanmaa,
which also removes the two structural obstacles that cap the trained AI at ~33%._

---

## 1. Why this exists — the problem it solves

Measured this session (Rust sim, hard-vs-hard and exp-A-vs-hard, 14×12):

- The game is **structurally draw-prone**. Two competent bots settle into a
  **territorial equilibrium** (~49% / 49% with ~36% of tiles left NEUTRAL as an
  unclaimed buffer) and neither can advance.
- **41.5% of games are unresolved at cap 120; 35.7% are STILL unresolved at cap
  3000** (25× longer). The stalemates are permanent — more rounds do not help.
- The native win conditions (`≥70% tile domination`, or eliminate the opponent by
  reducing them to 0 tiles / bankruptcy) are unreachable against a competent
  defender, because a defended HQ cannot be cracked (combat needs
  `attackers > defenders` on one tile, tile cap 3, and fielding the army drains
  the economy).
- Consequence: **"beat hard 70% of ALL games" is mathematically impossible** — at
  least ~39% of games are unwinnable stalemates. The realistic ceiling for any
  strategy is ~55%.

Two training pathologies stem from the SAME root cause (the game being draw-prone):

1. **Draw-attractor** — under self-play with a neutral draw value, the optimal
   policy is to turtle to a safe draw, so training drifts to passivity.
2. **Long-horizon credit problem** — the only path to a real win (multi-turn army
   buildup → coordinated assault → cut) is too many plies away to be reinforced;
   the AI never builds outposts (BuildOutpost chosen ~0.1% of the time) or fields
   an army (cap-locked at ~1.5 soldiers).

**The Strange Device fixes the game AND both training pathologies at the source:**
it converts the stalemate into a forced confrontation, bounds game length
diegetically (a countdown, not an artificial training cap), and gives the
previously-worthless Outpost a concrete, reachable purpose.

---

## 2. Core mechanic (one paragraph)

A new buildable, the **Strange Device**. Building it starts a **countdown of X
rounds**; if the Device is still standing when the countdown elapses, its owner
**wins immediately**. Only **one Device can exist in the entire game** at a time.
The catch: while you own a Device, your **soldier cap is halved**, so building it
**leaves you defensively exposed** — the enemy is forced to mass an army and
assault the Device (or you) before the timer completes. This turns a passive
two-turtle stalemate into a decisive race: _does the Device survive, or does the
enemy break through in time?_

---

## 3. Detailed rules

### 3.1 Building the Device
- The Device is a building placed on an **owned tile** (like other buildings).
  It may NOT be placed on the HQ tile (it must be a distinct, attackable target).
- **Cost:** a moderate **one-time build cost** (commitment), e.g. on the order of
  a Nuclear plant. **No per-turn economy drain** (see §6 — the soldier-cap halving
  is the balancer; a drain on top would over-nerf it into a dead mechanic).
- **Uniqueness:** at most **one Device exists in the whole game**. While one
  exists, neither player (including the owner) can build another.

### 3.2 The soldier-cap halving (the key balancer)
- While you own a standing Device, your **maximum soldier count is halved**
  (integer floor).
- Soldier cap is `HQ(+1) + 3 × Outposts`. Halved:

  | Outposts | normal cap | with Device (halved) |
  |---|---|---|
  | 0 | 1 | **0** |
  | 1 | 4 | **2** |
  | 2 | 7 | **3** |
  | 3 | 10 | **5** |

  → Building the Device with no Outposts means **zero defenders** (suicidal); the
  strategy is **gated behind having built Outposts first**. This is deliberate:
  it gives the Outpost a purpose and makes vulnerability scale with tech
  investment.

- **CRITICAL — forced disband on build.** The moment the Device is built, the
  owner's current soldiers are **immediately reduced to the new halved cap**
  (excess soldiers are disbanded/removed). Without this, the degenerate line is
  "pre-build a full army (cap 7), THEN build the Device, keep all 7" — which
  defeats the entire mechanic. The halving must bite _immediately_, not only on
  future recruitment.

- When the Device is destroyed (or otherwise ceases to exist), the soldier cap
  **returns to normal**. The owner may then re-recruit up to the full cap.

### 3.3 The countdown & winning
- Building the Device sets a counter to **X rounds** (proposed initial X ≈ 30–40;
  see §7). Each of the owner's end-of-turns decrements it.
- If the counter reaches 0 while the Device still stands → **the owner wins
  immediately** (all other players lose, same resolution as the 70%-domination
  win).
- The countdown is **visible to both players** (tension / drama — everyone knows
  the clock).

### 3.4 Counterplay — destroying the Device
- The Device tile is a normal conquest target: a focused assault that achieves
  `attackers > defenders` on the Device tile (respecting the existing tile cap 3
  / combat rules) **destroys the Device**.
- Destroying the Device **resets the countdown** AND **reopens the
  one-per-game slot** — the player who destroyed it (or anyone) may now build
  their own Device. This keeps both players in the race, makes destruction
  meaningful (not just a timer reset), and turns the Device into a back-and-forth
  focal point instead of a swingy first-mover-wins lock.
- Optionally: destroying a Device grants the attacker a small reward (e.g. a
  resource refund or a setback to the former owner) to further incentivise the
  assault. _Optional — tune later._

### 3.5 Pressure to actually build it (closing the "neither builds" hole)
If the Device is purely a risk, both players might avoid it and the stalemate
persists. Two safeguards:
- **Primary:** the Device should be attractive to the player who is _ahead_
  (e.g. it may grant a small passive benefit while ticking, or simply: the
  economic leader can afford the build + survive the halving with enough Outposts).
  The leader builds it → the trailer is FORCED to attack → confrontation.
- **Safety net (guarantees zero draws):** keep an **absolute round cap with a
  tile-majority tiebreak** — if no Device resolves the game by the cap, the player
  holding **more tiles wins** (ties → coin-flip / seat order). The Device resolves
  most games before the cap; the tile-majority net catches the rest. **Result:
  the game can NEVER end in a draw/timeout.**

---

## 4. Win / loss / resolution order

At end-of-turn, after the existing checks (0 tiles → lost; negative resources →
lost; ≥70% domination → win), add:

1. **Device countdown** — if any standing Device's counter has reached 0 → its
   owner wins.

And at the absolute round cap (training/benchmark + game safety net):

2. **Tile-majority tiebreak** — most tiles wins; this replaces the current
   "timeout / no winner" outcome entirely.

The existing elimination and 70%-domination conditions remain unchanged.

---

## 5. Why this is good for training (not just the game)

- **Kills the draw-attractor at the source.** Games are now always decisive
  (Device win, elimination, 70% domination, or tile-majority at cap). The
  `--timeout-penalty` training hack becomes unnecessary — the game itself is
  decisive, so self-play no longer has a safe-draw equilibrium to collapse into.
- **Shrinks the credit horizon.** "Build Outpost → enables a defendable Device →
  win in X rounds" is a SHORT, reinforced chain, unlike the current distant,
  uncertain "maybe field an army someday." This directly attacks the long-horizon
  credit-assignment problem that left the AI never building Outposts.
- **Gives the Outpost a purpose.** Outposts gate the Device strategy (more
  Outposts = more residual defense), so the AI now has a clear, learnable reason
  to build them — the very building it currently ignores (~0.1%).
- **Bounds game length diegetically.** The countdown caps games as a _rule_, not
  an artificial training cap, so the AI sees a real terminal signal.

---

## 6. Balance recommendation (decisions taken in design)

- **Soldier-cap halving is the PRIMARY balancer.** It is direct, legible
  (players see the tradeoff), deterministic, easy to implement (the soldier cap
  is already computed), and scales vulnerability with tech investment.
- **DROP the per-turn economy drain** that was floated earlier. Stacking a big
  drain on top of the halving risks over-nerfing the Device into a mechanic
  nobody builds → the stalemate returns. Keep only a moderate one-time build cost.
- **DROP the "adjacent tiles decay to neutral" flavour.** Hard to implement
  (per-tile decay + rendering) for marginal benefit; the halving already provides
  the exposure.

---

## 7. Tuning parameters (initial proposals — to be set empirically)

| Parameter | Proposed start | Notes |
|---|---|---|
| Countdown `X` | 30–40 rounds | Decisive games resolve by median round ~62, so this must threaten genuinely; long enough that a prepared enemy CAN mass force and reach the Device. |
| Build cost | ~Nuclear-tier one-time | Commitment; no ongoing drain. |
| Soldier-cap factor | ×0.5 (floor) | Per the table in §3.2. |
| Destroy reward | none initially | Add only if assaults need more incentive. |
| Tick benefit (to leader) | none / small | Add only if "neither builds it" shows up in testing. |
| Tile-majority tiebreak cap | training/benchmark cap (e.g. 120) | Safety net guaranteeing zero draws. |

**The one thing that cannot be settled on paper:** against an opponent that keeps
a standing army, a halved Device-builder (e.g. 3 defenders) may be cracked
immediately. That may be _fine_ (it makes the Device a timing play — build it
when the enemy is NOT military-ready), or it may make the Device too risky vs a
permanently-armed bot. Only a prototype + hard-vs-hard / cut-vs-hard measurement
reveals whether `X` and the halving are balanced. **Prototype, then tune.**

---

## 8. Implementation notes

### 8.1 Fidelity / parity implications (read before coding)
- This is a **deliberate divergence** from the 1:1 C++/Qt port. That is allowed —
  there is precedent (the Mine/Hydro/Nuclear industry rebalance is already a
  conscious fork; see `CLAUDE.md`). But it means the parity contract must be
  **re-baselined** for the changed mechanics: after implementing, re-export the
  golden traces and re-run the parity gate
  (`cargo run --release -p cp-train --bin parity`). The Device is a NEW mechanic,
  so it is not "breaking" parity with the original — it is an intentional
  addition the original never had.
- The **RNG order** (`src/core/rng.ts`, `worldgenerator.ts`) must NOT change — the
  Device touches game logic, not map generation, so keep all map-gen RNG calls
  byte-identical.

### 8.2 Where the code lives (touch all three for a shippable feature)
- **`rust-trainer/crates/cp-sim/`** — the training sim. Add the Device building,
  the per-owner soldier-cap modifier (halving + forced disband), the countdown,
  the destroy-reopens-slot logic, the Device-win check in `end_turn`
  (`managers.rs` ~`end_turn`, win checks ~line 989+), and the soldier-cap
  computation (`max_soldier_amount`).
- **`src/`** (the TS 1:1 port) — mirror the same logic in
  `model/` (a new building type), `managers/gameeventhandler.ts` (the endTurn win
  check, lines ~365–395 where the 70% domination check lives) and the soldier-cap
  helper. Add assets/UI for the new building + a visible countdown.
- **Reward / training** — once games are decisive, the value target is just the
  outcome; the tile-majority tiebreak lives in the **caller / benchmark harness**
  (e.g. `cp-ai/run.rs` where `Timeout` is currently injected, and `bench.rs` /
  `champ_probe.rs`), NOT inside the sim's `end_turn` (keep `end_turn` faithful to
  the natural win conditions; the cap+tiebreak is a harness convention). A
  tile-majority tiebreak probe already exists in `champ_probe.rs`.

### 8.3 Suggested prototype order (sim-first, measure early)
1. cp-sim: Device building + soldier-cap halving + forced disband + countdown +
   Device-win in `end_turn` + destroy-reopens-slot.
2. Add the tile-majority tiebreak in the bench harness (already partly built).
3. Run `hard_vs_hard` and `cut_vs_hard` with the Device enabled → measure: does
   the unresolved/stalemate fraction drop toward 0? Does a "race for the Device"
   emerge? Tune `X` and the halving.
4. Only after the mechanic is balanced in sim: port to `src/` TS + assets/UI, and
   retrain the AI on the now-decisive game.

---

## 9. Open questions to resolve during prototyping
- Exact `X` and build cost (§7).
- Does the Device need a tick-benefit to ensure it gets built, or does the
  economic-leader incentive + tile-majority net suffice?
- Is the halved defender count survivable against hard's standing army at the
  chosen `X`, or does the Device need a small defensive bonus on its own tile?
- Should destroying a Device grant a reward?
- How is "build on an owned non-HQ tile" surfaced in the UI, and how is the
  countdown displayed?

---

## 10. Instrumentation — outcome-type breakdown (dashboard + benchmarks)

The current tooling only reports win / loss / timeout / tile-fraction. With the
Device, games end in several DISTINCT ways, and **the distribution of outcome
TYPES is the primary signal for whether the redesign worked** (e.g. timeout/draw
should collapse toward 0; a healthy share of games should end by Device; the AI
should win by more than one route). So the **dashboard and the benchmark binaries
(`hard_vs_hard`, `cut_vs_hard`, `champ_probe`, `bench`) must record and show the
outcome by CAUSE**, not just win/loss.

Outcome taxonomy to track (per game, attributable to the winner's seat):

| Outcome | Cause |
|---|---|
| **Strange Device win** | a standing Device's countdown reached 0 |
| **Domination win** | reached ≥70% tile domination |
| **Conquest (attack) win** | enemy reduced to 0 tiles by conquest |
| **Bankruptcy win** | enemy went bankrupt (negative resources) — track separately from conquest; it is often the enemy self-destructing, not our doing (cf. the old `hard_self_bankrupt` benchmark-integrity check) |
| **Tile-majority tiebreak win** | no natural resolution by the round cap → most tiles wins (this REPLACES the old "timeout"; it is a *win*, not a draw) |
| **Timeout / true tie** | should be ~0 once the tiebreak is in — track anyway to confirm it stays near zero |

What to surface:
- A **stacked breakdown** (counts + %) of these outcome types, for the champion
  and ideally split by who won (champ vs hard).
- The fraction of games ending **non-decisively** (timeout/true-tie) — the headline
  "did we kill draws" number; target ≈ 0.
- Mean rounds-to-resolution per outcome type (Device wins vs conquest vs tiebreak)
  — tells us whether the Device is doing its job of bounding game length.

Implementation hooks: the sim's `EndTurnOutcome` (or the harness wrapper) needs to
carry the WIN CAUSE, not just `Win(PlayerId)` — add a cause enum (Device /
Domination / Conquest / Bankruptcy) so the harness can tally it; the tiebreak +
timeout are resolved harness-side (where the cap lives). Then thread the cause
into the dashboard's JSON log and render the stacked breakdown.

### 10.1 Wider metric catalogue (for the dashboard + benchmark binaries)

Beyond the outcome breakdown, collect the following. Many of these would have
turned multi-day diagnostics in the OLD arc into a single glance — the **★ TOP-5
priority** items are the highest value-per-effort and should go in first.

**A. Training-health signals** (catch policy-freeze / value-collapse early)
- **★ Policy entropy** — how spread the policy distribution is. Collapsing entropy
  = an over-confident / frozen policy. Would have flagged the old az13 freeze and
  az11 collapse instantly (we only caught them by hand, late). Cheap.
- **★ Value calibration** — predicted value vs actual game outcome (a reliability
  curve, or just mean predicted-value per true-outcome bucket). If the value net
  predicts ≈0 everywhere → the draw-collapse; shows it directly rather than
  inferring it from `valueLoss → 0.1`. Cheap.
- **Policy drift from the warm-start / reference** (e.g. mean KL to a frozen
  reference). Flat = not learning; runaway = unstable. The flat `policyLoss` was
  the "smoking gun" last arc — make it an explicit, plotted signal.
- Keep the existing `policyLoss` / `valueLoss` curves and games/sec throughput.

**B. Behavioural telemetry** (what the AI actually DOES — on the dashboard, not
just a manual `champ_probe` run)
- **★ Intent histogram** — build types, Expand, Attack, HireSoldier, **BuildDevice**,
  Pass — both raw % and "% when that intent was available". This is what surfaced
  "BuildOutpost ~0.1%, Attack ~70% when available" last arc.
- **Army size + Outpost count over the game** — last arc was cap-locked at ~1.5
  soldiers; Outpost count now matters even more (it gates the Device).
- Economy trajectory (net income, money) — solvent vs self-bankrupting.
- End-game building composition (farms / mines / outposts / villages / Device),
  ideally split by outcome type.

**C. New-mechanic balance** (essential for tuning `X` and the soldier-cap halving)
- **★ Device survival rate** — built → won vs built → destroyed. The core signal
  for whether `X` / the halving are balanced.
- **Device build rate** — fraction of games in which a Device is built at all. If
  ≈0, the mechanic is dead (over-nerfed) → the stalemate returns.
- Mean rounds from Device-build to resolution (does it bound game length?).
- Defense-success with a halved army — how often the builder holds vs is cracked.

**D. Benchmark robustness** (avoid misreading noise)
- **★ Win-rate with confidence intervals** — a 40-game bench is ±~15%; last arc we
  repeatedly mistook noise spikes (e.g. a 45% blip) for signal. Show CI bars or
  widen the bench.
- Win-rate split **by seat** — there is a measured first-mover/seat advantage;
  surface it so it can be corrected for.
- Win-rate vs a **fixed past champion** (Elo-style self-improvement check), not
  only vs hard — tells us if the AI is improving against itself even when the
  vs-hard number is flat.
- Opponent self-destruct flag (generalise the old `hard_self_bankrupt` integrity
  check) — keeps the benchmark honest.

**E. Game-shape** (complements the §10 outcome breakdown)
- Rounds-to-resolution distribution **per outcome type** (Device vs conquest vs
  tiebreak resolve very differently).
- Mean **neutral-tile fraction** at end — the stalemate fingerprint (was ~36%);
  should drop as the game becomes decisive.

**★ Top-5 to implement first:** (1) policy entropy + (2) value calibration
[training health], (3) intent histogram on the dashboard [behaviour], (4) Device
survival + build rate [mechanic balance], (5) win-rate with CI bars + vs a fixed
past champion [robust benchmarking]. (5b) rounds-to-resolution per outcome type
rounds out the §10 breakdown.
