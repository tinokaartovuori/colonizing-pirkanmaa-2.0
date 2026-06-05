# TRAINING-APPROACH — teaching the full skill repertoire

_Authored 2026-06-05. A mechanics-grounded design for how to train the Colonizing
Pirkanmaa AlphaZero AI so it reliably acquires every skill a competent player needs.
This is a DESIGN document — no code is changed and no run is launched here. It
distinguishes **what our data supports** from **hypothesis**, and ends with a
prioritized, gated plan._

Grounding sources (read before acting): `GAME-MECHANICS.md` (verified rules),
`REWARD-DESIGN.md` (the Φ signal mapping, esp. N5 idle), `EXP-M-DESIGN.md`, the
memories `exp-m-radical-redesign`, `plateau-forensics`, `capacity-blindness`,
`az-pass-collapse-fix`, `az-draw-attractor`, `reward-design-loop`. Reward code:
`crates/cp-train/src/bin/cnn_train.rs` (`potential_full`/`potential_econ`/
`potential_dev`, PFSP, script opponents, `bench_vs_hard`). Eyes: `crates/cp-ai/src/
{planes.rs,candidates.rs}`.

---

## 0. The one-paragraph diagnosis we are designing against

The champion is **severely passive**: Pass = 45% of decisions, 0–3 soldiers all
game, BuildOutpost 0.2%, Device 0.1%, stalls at 10–17 tiles while HARD grows to
50–90. Its ~0.46 win-rate vs HARD is a **mirage**: 28–31% of "wins" are free enemy
self-bankruptcy; strip those and the champion is a **~0.31–0.39 loser**. Two prior
hypotheses are **empirically refuted as the binding constraint**: net capacity (5.5×
params, flat) and value-head squash (un-squashed via `--record-opp-value`, win-rate
didn't move). The surviving, well-supported root cause is the **reward/learning
signal**: the potential Φ (`potential_econ`) rewards only *static economic health*
(income, staffed-ratio, filled-cap, in-window banking) with **no territory term and
no cost of inaction** → a champion sitting on ~10 staffed tiles already MAXES Φ, so
Pass is Φ-optimal and risk-free. Historically the **only** config that broke
passivity honestly was a *territory* Φ (lead/dom) which reached **45% genuinely**
(`az-pass-collapse-fix`) before being dropped for an economy Φ. **That regression is
the smoking gun.**

The design below is built on a single thesis: **the winning line must be made more
Φ-attractive than safe-Pass, and the curriculum must make the ABSENCE of each skill
actively lose games.** Reward says "this is good"; curriculum says "not doing this
kills you." Both are needed because potential-based shaping is policy-invariant by
construction (Ng 1999) — it can only *speed* learning of what terminal outcomes
already favour, so the terminal-outcome distribution (who you play) must itself
reward the skills.

---

## 1. The seven capabilities

For each: **(a)** the mechanic, **(b)** why it fails to emerge now, **(c)** the
reward/curriculum mechanism to teach it, **(d)** the behavioral metric that proves it.

### 1.1 Grow economy (Villages → staffed farms → rising income)

- **(a) Mechanic.** Worker/expert cap = HQ(+3) + Village(+3) + Mikontalo(+2)
  (`GAME-MECHANICS §5`). A producer outputs **0 unless staffed** by a worker/expert on
  its tile. So income = f(producers that are *staffed*), and staffing headroom is
  gated by Villages. A Village is a **net drain** (−10/−10/−10 per round,
  `capacity-blindness`) whose only payoff is +3 worker cap that must then be *filled*
  and *put to work* — a ~6-action, multi-turn investment.
- **(b) Why it fails.** Two confirmed reasons. (i) Φ's economy core already saturates
  on a small staffed economy, so there's no gradient to *grow*. (ii) The Village line
  is a delayed, multi-step payoff: build (Φ drops via drain) → hire workers (fills cap)
  → build farms on them → 4-turn growth. With 48-sim MCTS this payoff is **beyond the
  search horizon** and the squashed/short-horizon value can't bridge it, so the net
  abandons Villages (Exp G: 0→4→5→0→1). The `potential_econ` "filled capacity"
  rebalance (Exp H) removed the *punishment* for building empty cap but did **not** add
  a *pull* toward growth.
- **(c) Mechanism.** Replace static econ-health with a **growth/lead potential**, not
  an absolute-health one. Concretely Φ should reward **income LEAD over the opponent**
  and **tile/producer LEAD**, both signed and saturating slowly, so that there is
  *always* a positive gradient to out-grow the enemy (you can't max it by sitting —
  the enemy keeps growing). This is the historical territory-Φ that worked, generalized
  to economy. Crucially keep the **filled-capacity** formulation (Exp H) so building a
  Village is never punished, and let the *lead* term supply the forward pull the
  abandoned-Village problem needs. Pair with **PFSP + a turtle opponent** (§2.2) so the
  terminal signal also says "out-expand or you fall behind and lose."
- **(d) Metric.** `realized_income_per_round` trend per game (should rise past ~r30,
  not plateau at the starting cluster); **Village build count > 0 and SUSTAINED**
  (built and not abandoned: count standing Villages at game end, not just builds);
  staffed-producer count climbing; `tile_lead` and `income_lead` trending positive vs
  HARD. These are already computable from `value_scalars` and the producer scan in
  `potential_econ`.

### 1.2 Build outposts — the army prerequisite (the idle/outpost tension)

- **(a) Mechanic.** Soldier cap = HQ(+1) + Outpost(+3 each) (`§5`). **No Outpost ⇒ ≤1
  soldier the entire game.** Outpost is also **impregnable by assault** (`§3`) — both an
  army enabler AND the best defensive tile. The Device line *requires* an Outpost
  (`candidates.rs`).
- **(b) Why it fails.** The capacity-blindness chain: with BuildOutpost ≈ 0, soldier cap
  ≈ 1, so the net literally cannot field an army and never sees the payoff of one. The
  **key open tension we discovered**: a high `idle_penalty` (which drives activity, Pass
  45→38%) *penalizes the fresh empty soldier slots an Outpost adds* — so building an
  Outpost momentarily LOWERS Φ (it creates 3 empty soldier slots that the idle term
  punishes). A low idle penalty stops punishing Outposts but reverts to passivity (Pass
  72%). This is a **double-counting collision** between the idle term (penalizes empty
  slots) and the implicit cost of building.
- **(c) Mechanism — resolve the tension by SEPARATING "do something" from "hoard empty
  capacity," and by making capacity-growth itself a potential.** Three coordinated
  changes:
  1. **Redefine idle, anti-double-count.** The idle penalty must NOT be a function of
     *empty soldier/worker slots*. Empty slots are the *intended transient state*
     immediately after building capacity — penalizing them directly punishes the build.
     Instead define idle as **unused FLOW**: idle = (workers/experts that exist but are
     NOT staffing a producer) + (un-spent income accumulating with affordable builds
     available). This penalizes *failing to use what you have*, not *having room to
     grow*. Building an Outpost adds 0 idle by this definition (it added capacity, not
     idle units). This is the precise fix for the tension.
  2. **Make capacity a potential with a CEILING, not the soldiers themselves.** Add a
     small Φ term `+ w · clamp(soldier_cap, 0, CAP_TARGET)/CAP_TARGET` where CAP_TARGET
     ≈ 7 (HQ + 2 outposts). This rewards *having the cap* (so building the first
     Outposts is immediately Φ-positive), saturates so the net doesn't outpost-spam, and
     is orthogonal to the *filled-soldier* term (§1.3) which rewards actually fielding
     the army. Empty cap is rewarded only up to the small ceiling; beyond it, only
     filling pays. This breaks the "build-Outpost-drops-Φ" problem at the source.
  3. **MCTS build-prior floor stays ON** for BuildOutpost (it's in `is_starved_build`)
     so search actually explores the line during self-play; **apply it at eval too**
     (the train/eval prior-floor mismatch was a documented bug, `plateau-forensics`) —
     but for HONEST measurement keep an ablation bench with `--eval-prior-floor 0`.
- **(d) Metric.** **Standing Outpost count > 0 at game end** (not just transient
  builds); `max_soldier_amount` reached per game (target: routinely > 4); and the
  *conditional* metric that proves the tension is resolved: **Outpost build rate must
  NOT fall when idle_penalty is raised** (run the two-point sweep that previously showed
  the inversion — if the redefinition worked, raising anti-idle no longer suppresses
  Outposts).

### 1.3 Field & command multiple soldiers (a real army)

- **(a) Mechanic.** Soldiers are the *only* combat unit (workers/experts = 0 combat,
  `§3`). Up to 3 owned defenders + 3 conquering attackers per tile (`§2`). Movement is
  unrestricted within `getAvailableTiles` — a soldier reaches any border tile in one
  action (`§1`), so "commanding" an army is target-selection, not pathing.
- **(b) Why it fails.** Downstream of §1.2: cap ≈ 1 ⇒ no army possible. Even with cap,
  the soldier plane was **owner-agnostic** until Exp M (the net couldn't perceive
  relative army strength) — Exp M's eyes (owned vs conquering soldier planes,
  `C_ATT_MINUS_DEF`) fixed perception but were "necessary not sufficient" under the
  passive reward.
- **(c) Mechanism.** The `soldier_cap_potential` term (FIX 3, rewarding FILLED soldier
  slots) is the right shape — keep it, but it only bites once §1.2 supplies the cap.
  The real teacher is the **curriculum**: an **army-rusher** opponent (already scripted,
  `HardAi::army_rush`) that attacks you. Against it, 1 soldier loses tiles every turn →
  the terminal signal rewards fielding more. With the Exp-M eyes in place the net can
  finally *see* attacker−defender and learn to mass. Add **PFSP** so once the learner
  builds an army, frozen past-selves that also have armies force it to keep scaling.
- **(d) Metric.** **Mean and max owned-soldier count per game** (target: max > 3
  routinely, the documented failure threshold); soldiers fielded *before* combat (not
  just emergency). Use the self-play intent histogram (`HireSoldier` share) and the
  per-game soldier-count from `value_scalars`.

### 1.4 Attack (assault with strict soldier superiority)

- **(a) Mechanic.** Assault wins **iff attacker conquering-soldiers > defender owned-
  soldiers (strict; tie → defender) AND no Outpost on the tile** (`§3`). Loss destroys
  all the attacker's conquering units there. Capturing the enemy HQ = Conquest win.
- **(b) Why it fails.** No army (§1.3) ⇒ attacks are 1-soldier mop-ups only after HARD
  self-collapses (the mirage). The strict-`>` rule means a *successful* attack needs a
  local majority the net never has the soldiers to assemble.
- **(c) Mechanism.** Once §1.2–1.3 give an army, attacking emerges through **MCTS +
  terminal reward** (capturing tiles/HQ is a real terminal win the forward model
  already implements — no shaping needed for the *event*). To *bias* toward the
  decisive line, keep the existing **distance-to-enemy-HQ** spatial intuition (P10 in
  REWARD-DESIGN) and a *small* tile-conquest tactical bonus, kept tiny to avoid
  optimum skew. Curriculum: a **passive economic turtle** opponent (§2.2) that does
  NOT attack — this rewards the learner for *converting* a lead into a kill instead of
  drawing (the historical "draw-happy 45%" failure was exactly *not converting*). The
  turtle punishes non-conversion; the army-rusher punishes non-defense; together they
  force two-sided competence.
- **(d) Metric.** **Conquest-win share that is NOT bankruptcy-propped** (the honest-win
  metric, §3); attacks launched per game with attacker>defender (successful assault
  rate); enemy HQ captures. `bench_vs_hard` already separates `champ_cause.conquest`
  from `champ_cause.bankruptcy`.

### 1.5 Defend (own HQ and Device, using Outpost impregnability + positioning)

- **(a) Mechanic.** A tile is threatened iff it's on the enemy frontier, not an Outpost,
  conquering slots not full, and the enemy can muster (your soldiers+1) (`§4`). Outpost
  = impregnable (`§3`). HQ-connectivity: tiles not BFS-connected to your HQ are
  neutralized/confiscated end of turn (`§7`) — a cut chokepoint prunes everything
  behind it.
- **(b) Why it fails.** With ≤1 soldier the net cannot garrison anything; it has no
  defensive concept because it never had units to position. The threat plane
  (`C_ENEMY_REACH`) exists in the Exp-M eyes but is inert without an army to respond.
- **(c) Mechanism.** Defense is taught **reactively by the army-rusher curriculum**:
  losing HQ-adjacent tiles (and games) to the rusher creates terminal pressure to
  garrison the frontier and to value Outposts as impregnable anchors. A *small*
  potential term for **own-HQ-connectivity health** (penalize being one cut from
  losing tiles — articulation-point exposure, REWARD-DESIGN N3 `own_tiles_lost_via_cut`)
  gives a denser defensive gradient than waiting for the terminal loss. Keep it small;
  the event (actually losing tiles) is in the forward model and will show in value.
- **(d) Metric.** Vs the army-rusher: **tiles lost per game trending down**, HQ
  survival rate up, soldiers positioned on frontier/HQ-adjacent tiles before the
  enemy strikes. `vsArmyRush` win-rate (Exp M tracked ~0.2 — should climb).

### 1.6 React to an enemy Strange Device

- **(a) Mechanic.** A standing Device runs a countdown; survive to 0 at the owner's
  end-of-turn → all others lose (`§6`). But the Device tile holds **zero owned
  defenders** and **halves the builder's soldier cap** → a single enemy soldier staged
  on the device tile destroys it (`§2`, `§6`). So the *counter* is cheap: rush one
  soldier onto the undefended device tile before the countdown expires.
- **(b) Why it fails.** Two reasons. (i) No army ⇒ no soldier to send. (ii) Horizon: the
  Device decides ~round 90, far beyond 48-sim MCTS and the squashed value, so the net
  treats device dynamics as a trap (HARD out-devices it ~5:1; champ device-build ≈ 0).
  Exp-M's eyes include device features (owner, countdown, defenselessness) but the net
  never learned to act on them under the passive reward + short horizon.
- **(c) Mechanism.** This is where **macro/turn-search (Exp M's `--turn-search`)** is
  load-bearing: turn-granularity MCTS makes tree depth = rounds, so search reaches the
  round-90 device outcome and the value head gets a *grounded* target for "enemy device
  standing ⇒ I lose unless I crack it." Pair with the **scripted device-rusher**
  opponent (`HardAi::device_rush`) in the PFSP pool: it builds a device every game, so
  the learner faces the countdown constantly and is forced to learn the cheap counter or
  lose. Add an **action-level device-reaction credit** (small) for staging a soldier on
  an enemy device tile, since that single action's payoff (averting a loss) is diffuse
  over the whole game — `--device-credit` already exists for the *building* side; mirror
  it for the *cracking* side.
- **(d) Metric.** Vs the device-rusher: **device-denial rate** (fraction of enemy
  devices cracked before countdown), `vsDeviceRush` win-rate (Exp M: climbed 0.11→0.31
  — should approach/exceed 0.5); HARD's device-wins-against-us falling (Exp M hDevWin
  15→9 is the right direction).

### 1.7 The strategic arc (choose & execute, and counter)

- **(a) Mechanic.** Two winning lines: economy+army → destroy enemy HQ (Conquest), OR
  economy → Outpost → bank → Device → defend the countdown (Device). Each has a
  counter: rush the enemy before its device, or crack its device; out-expand a turtle.
- **(b) Why it fails.** The net learned exactly ONE degenerate line (1-soldier conquest
  mop-up of a self-collapsed HARD) and treats the other (Device) as a trap. It cannot
  *choose* because it never had the components of either line.
- **(c) Mechanism.** This is the **emergent capstone** — it should appear once §1.1–1.6
  are individually present, *provided* the curriculum contains BOTH a device-rusher and
  an army-rusher and a turtle (so neither pure line dominates and the net must read the
  opponent and counter). PFSP against past-selves that have *learned* lines forces
  counter-play. **Do not shape the arc directly** (that risks a brittle scripted
  strategy and optimum skew) — let MCTS + turn-search + diverse curriculum compose it.
- **(d) Metric.** **Win-cause DIVERSITY** in honest (non-bankruptcy) wins: the champion
  should win by Conquest *and* Device depending on opponent, and its win-cause
  distribution should *shift* to counter each scripted opponent (beat the device-rusher
  by cracking → Conquest/Domination; beat the turtle by out-expanding → Domination/
  Conquest). A champion stuck on one cause vs all opponents has not learned the arc.

---

## 2. Meta / approach decisions

### 2.1 Why self-play collapses to the passive/bankruptcy equilibrium — and the escape

Three compounding mechanisms, all evidenced:

1. **Symmetric passivity is a Nash-stable attractor.** Two identical passive nets that
   both Pass reach a stall/tie; the value target for a tie ≈ 0, so the value head learns
   "≈0 everywhere," MCTS gets no gradient, visits concentrate on the highest-prior move
   (Pass), policy reinforces Pass (`az-pass-collapse-fix`, `az-draw-attractor`). Self-
   play *manufactures* the passive opponent that makes passivity safe.
2. **The reward made passivity OPTIMAL, not just safe.** `potential_econ` maxes on a
   small static economy with no territory term and no inaction cost → Pass is literally
   Φ-optimal (`exp-m-radical-redesign` deep verdict). Shaping is policy-invariant, so it
   couldn't have *caused* a wrong optimum — but it *failed to break the tie* toward the
   active line, and the *terminal* signal was corrupted by the bankruptcy mirage.
3. **The bankruptcy mirage corrupts the terminal signal.** ~30% of "wins" are HARD self-
   bankrupting; the net is *rewarded for doing nothing* while the opponent self-destructs
   → reinforces Pass as a winning policy. This is the deepest issue: **the terminal label
   itself rewards passivity.**

**Escape (the design's spine):**
- **Fix the reward optimum**: growth/lead Φ (§1.1) + capacity-as-potential (§1.2) + the
  redefined-idle anti-passivity term (§1.2c) so the active line is Φ-attractive and Pass
  is not free.
- **Fix the terminal signal**: a curriculum of opponents that **never self-bankrupt and
  actively punish each missing skill** — so winning *requires* doing the thing. An
  aggressive army-rusher forces defense+army; a device-rusher forces device-reaction; a
  turtle forces out-expansion and conversion. **Honest-win metric** (§3) removes the
  bankruptcy reward from measurement (and, ideally, we down-weight self-play games that
  end in opponent bankruptcy so the data doesn't teach "Pass and wait").
- **Fix the horizon**: turn-search (Exp M `--turn-search`) so the decisive long-horizon
  outcomes (Device, late conquest) are inside the search tree and produce grounded value
  targets — without this, no reward can teach the round-90 Device line.
- **Break self-play symmetry**: PFSP + scripted opponents so the learner rarely faces a
  mirror of its own current passivity.

### 2.2 The curriculum (the highest-leverage lever)

A **PFSP league** (already built, `--pfsp`, cap 8 frozen champions, win-rate-weighted)
PLUS three **scripted teaching opponents** (two already exist):
- **Army-rusher** (`HardAi::army_rush`, exists): builds Outposts + soldiers + attacks.
  Teaches §1.2 (cap), §1.3 (army), §1.5 (defense). Punishes "no army."
- **Device-rusher** (`HardAi::device_rush`, exists): builds Outpost → Device every game.
  Teaches §1.6 (device reaction). Punishes "ignore the countdown."
- **Economic turtle** (NEW, small addition): expands economy, never attacks, never
  devices. Teaches §1.1 (out-expand) and §1.4 (convert a lead into a kill — the cure for
  the historical "draw-happy" non-conversion). Punishes "sit on a lead."

**Critical curriculum-design fix from Exp M:** scripted opponents were *lopsidedly
losing* for the learner (it won ~20–30%), so value targets were dominated by −1 and the
value head squashed. The fix already shipped (`--record-opp-value`: record the winning
scripted seat as value-only examples) — **keep it on.** Also **anneal the scripted
fraction**: start high (the learner needs the lessons) and decay toward more PFSP/self-
play as skills appear, so the final policy isn't overfit to scripted quirks. Hard AI
stays **held out of the training pool** (it's the honest benchmark, `reward-design-loop`).

### 2.3 Incremental curriculum vs joint training — RECOMMENDATION: **scaffolded-joint**

Evidence for *not* doing strict sequential staging: the skills are **mutually
dependent** (army needs cap needs economy; defense needs army; device-reaction needs
army), and our history shows single-lever fixes in isolation each plateaued
(capacity eyes alone, value un-squash alone, capacity net alone — each "necessary not
sufficient"). A strictly staged curriculum (freeze economy, then add army, …) risks
**catastrophic forgetting** of the earlier stage and brittle hand-offs.

Evidence for *not* doing pure joint-from-scratch: the joint passive attractor is exactly
what's been collapsing for weeks.

**Recommendation: train JOINTLY but with a STAGED CURRICULUM SCHEDULE** (the opponents
and reward-term weights change over training, the *policy* is one continuously-trained
net):
- **Stage A (econ + cap):** turtle-heavy curriculum + growth/lead Φ + capacity-potential.
  Gate: economy grows, Villages/Outposts standing, max-soldier > 3. (Validates §1.1–1.3.)
- **Stage B (combat):** add army-rusher to the pool. Gate: honest conquest wins appear,
  tiles-lost-to-rusher falls. (Validates §1.4–1.5.)
- **Stage C (device + arc):** add device-rusher + turn-search emphasis + device-reaction
  credit. Gate: device-denial rate up, win-cause diversity, honest win-rate vs HARD
  climbs. (Validates §1.6–1.7.)

This is "scaffolded joint": one net, joint objective, but the *environment difficulty
and reward emphasis ramp* so each skill has a window where it's the dominant pressure
without erasing the others. It's the standard resolution of the joint-vs-staged tension
in curriculum RL and matches our evidence that isolated levers don't stick.

### 2.4 Reward architecture (term set, interactions, anti-double-count)

Proposed Φ (potential-based, `γΦ(s')−Φ(s)`, Φ(terminal)≜0, all signed/bounded):

```
Φ(s) = w_inc  · income_lead         (signed, vs best enemy)      [§1.1, replaces static inc]
     + w_tile · tile_lead           (signed)                     [§1.1, the carrot that worked]
     + w_staff· staffed_ratio       (keep — utilization)         [§1.1]
     + w_cap  · clamp(soldier_cap/CAP_TARGET)  (saturating)      [§1.2 — fixes outpost tension]
     + w_army · filled_soldier_norm (the fielded army)           [§1.3]
     − w_idle · idle_FLOW           (unstaffed units + unspent affordable income)
                                                                 [§1.2c — REDEFINED, no empty-slot term]
     + w_dev  · device_progress     (own ticking device, windowed)[§1.6 builder side]
     − w_cut  · hq_cut_exposure     (small; articulation risk)    [§1.5 defense]
```

**Interaction discipline (the core of the idle/outpost fix):**
- `w_cap` rewards *having* soldier cap (up to a ceiling); `w_army` rewards *filling* it;
  `w_idle` penalizes *unused flow*. These three are **mutually orthogonal**: building an
  Outpost raises `cap` (+), leaves `army` unchanged (empty slots aren't filled army), and
  adds **0 idle** (the redefined idle is about units/income flow, not empty slots). This
  is the precise resolution of "anti-idle vs fresh military capacity" — the previous
  collision came from `w_idle` keying on *empty slots*, which is exactly the transient an
  Outpost creates. Removing the empty-slot term from idle eliminates the double-count.
- **No double-counting with terminal events:** conquest, HQ capture, cuts, device-expiry
  are NOT in Φ (they're terminal/forward-model outcomes seen via value + MCTS). Φ only
  holds *positional* quantities. The two small *action-level* credits (device-build,
  device-crack) are kept tiny and justified solely because their reward is diffuse over a
  long game — flag them off by default and ablate.

**Is potential-based shaping even the right tool for aggression?** Partly. Shaping is
*policy-invariant* — it cannot by itself make attacking optimal if the terminal signal
doesn't already favour it; it can only *accelerate* learning toward the terminal
optimum. Aggression is therefore primarily a **terminal-signal / curriculum** problem
(opponents that punish passivity), with shaping (tile_lead, distance-to-HQ intuition,
small tactical bonus) as an accelerant. The two small action-level credits are the only
non-potential rewards, kept minimal and ablated, precisely because un-bounded action
rewards risk reward-hacking and optimum skew.

### 2.5 Net size

Use the **small (~9.8k-param) net** for all curriculum/reward iteration — it has the
proven representation (the Exp-M eyes), trains fast (~132 s/iter post-speedup), and the
5.5× capacity test was **confounded by passivity** (run under the broken reward), so it
proved nothing. **Capacity is revisited only AFTER** the net plays actively (builds
army, attacks, reacts to devices) and we re-run the capacity ablation under the *fixed*
reward. Hypothesis to test then, not now: capacity may bind on *tactical refinement*
(target selection, multi-front), not on the gross behavioral repertoire.

### 2.6 Measurement discipline

- **Per-skill behavioral metrics** (tighter than the ±12.6% win-rate at 60 games):
  Pass%, standing Village/Outpost count, max-soldier, honest-conquest-win share,
  device-denial rate, win-cause diversity, tiles-lost-to-rusher, `vsArmyRush`/
  `vsDeviceRush`/`vsTurtle` win-rates. Most are already logged or trivially derived from
  `bench_vs_hard` (`champ_cause` already splits bankruptcy) and `value_scalars`.
- **Honest win-rate** = wins − bankruptcy-propped wins, divided by games. `champ_cause.
  bankruptcy` is already tracked; surface a `trueWinVsHard` field and judge by IT, not
  raw win. Flag every bankruptcy win in replays.
- **Convergence horizon:** judge **aggregated trends over ~30–60 iters**, not single
  benches (60-game CI ≈ ±12.6%; any <8% move is noise — `plateau-forensics`). Use
  first-half vs second-half means and monotonic-trend checks (the Exp-M analysis method).
- **Detecting emergence:** each skill has a **leading behavioral indicator** (above) that
  moves *before* win-rate; watch those to catch a skill emerging even while the noisy
  win-rate is flat — and to catch a lever that improves *style* but not *strength* (the
  recurring Exp-M trap: value calibration / device-denial improved while win-rate didn't).

---

## 3. Honesty guardrails (non-negotiable)

1. **`trueWinVsHard` excludes bankruptcy-propped wins** and is the headline metric.
2. **HARD stays held out** of the training pool.
3. **Down-weight (or cut) self-play/curriculum games decided by opponent bankruptcy** in
   the training data — they teach "Pass and wait." (Scripted opponents that don't self-
   bankrupt already mitigate this; verify the bankruptcy share in self-play is low.)
4. **Eval prior-floor ablation:** report both `--eval-prior-floor`-on and -off so we
   never mistake a search-prop for learned skill (the train/eval mismatch bug).
5. **Style-vs-strength check:** for every lever, confirm a *behavioral* metric AND
   `trueWinVsHard` both move; a style-only gain is logged as such, not as progress.

---

## 4. Prioritized, sequenced plan (with decision gates)

The bias is **reward + curriculum first** (well-supported root cause), capacity last
(refuted as current bottleneck). Each step states what it validates and its gate.

**Step 0 — Measurement first (build before training).**
Add `trueWinVsHard` (raw win minus `champ_cause.bankruptcy`) + the per-skill behavioral
counters (standing Villages/Outposts, max-soldier, win-cause diversity, device-denial,
tiles-lost-to-rusher) to the bench/dashboard. **Validates:** we can see real skill, not
the mirage. **Gate:** the headline number is honest before we tune anything. _This is
the recommended FIRST step — every later gate depends on it, and we already nearly have
the data (`champ_cause` splits bankruptcy)._

**Step 1 — Reward optimum: kill safe-Pass.**
Swap `potential_econ`'s static health for the **growth/lead Φ** (income_lead +
tile_lead, keep staffed_ratio) and add **capacity-as-potential** (saturating soldier-cap
term) + the **redefined idle = unused-flow** term (NOT empty slots). Small net,
modest curriculum (PFSP + turtle), turn-search ON. **Validates §1.1–1.2 + the idle/
outpost tension.** **Gate (~30–40 iters):** Pass% < 30; standing Villages > 0 and
Outposts > 0; the idle-sweep no longer suppresses Outposts; `trueWinVsHard` not falling.
_Decision: if Outpost rate still inverts under higher anti-idle → the redefinition is
wrong, revisit §1.2c before proceeding._

**Step 2 — Combat curriculum: army + attack + defense.**
Add the **army-rusher** to the PFSP pool (keep `--record-opp-value`), add `w_army`
(filled-soldier) emphasis and the small `w_cut` defense term. **Validates §1.3–1.5.**
**Gate (~30–40 iters):** max-soldier routinely > 3; honest conquest wins appear;
tiles-lost-to-rusher trends down; `vsArmyRush` climbs off ~0.2.

**Step 3 — Device reaction + the strategic arc.**
Add the **device-rusher** to the pool, emphasize **turn-search depth** (so round-90
outcomes are searched), add the small **device-crack action credit** (ablated).
**Validates §1.6–1.7.** **Gate (~40–60 iters):** device-denial rate up; `vsDeviceRush`
toward 0.5; **win-cause diversity** (champion wins by Conquest vs the turtle AND by
denial/Conquest vs the device-rusher); `trueWinVsHard` climbs past the historical honest
~0.45 (the bar the dropped territory-Φ once cleared) — toward > 0.55.

**Step 4 — Re-test capacity (only now).**
With an *actively playing* net, re-run the bigger-net ablation under the fixed reward +
full curriculum. **Validates:** whether capacity binds on tactical refinement.
**Gate:** bigger net beats the small net's `trueWinVsHard` by > CI over ~60 iters — only
then is capacity the next lever; otherwise keep iterating curriculum/reward.

**Stop/branch rules.** If Step 1 fails its gate (Pass stays high, no expansion), the
growth-lead Φ is insufficient and we escalate to **stronger anti-passivity** (higher
turtle fraction, or down-weighting Pass-heavy self-play data) before adding combat
complexity. If a step improves a behavioral metric but NOT `trueWinVsHard` (the Exp-M
style-vs-strength trap), log it as style and do **not** stack the next reward term on top
— diagnose the strength gap first.

---

## 5. What's well-supported vs hypothesis (skeptic's ledger)

**Well-supported by our data:**
- The win-rate is a bankruptcy mirage; true skill ~0.31–0.39 (deep verdict + `champ_cause`
  accounting).
- The static econ Φ makes Pass optimal; a territory/lead Φ once reached **45% honestly**
  (`az-pass-collapse-fix`) — strongest single evidence for Step 1.
- Net capacity and value-head squash are **NOT** the current binding constraint
  (5.5× flat; un-squash didn't move win-rate).
- The idle/outpost tension is real and is a double-count on empty slots (observed
  inversion).
- Horizon matters for the Device line (round-90 payoff > 48-sim search); turn-search is
  the right tool (Exp M built + verified it cheap).

**Hypothesis (to be validated by the gates, not assumed):**
- That growth/lead Φ + capacity-potential + redefined-idle *together* break passivity
  without a new local optimum (Step 1 gate tests this).
- That the redefined "unused-flow" idle fully resolves the Outpost tension (Step 1 sweep
  tests this).
- That the strategic arc (§1.7) emerges from curriculum diversity rather than needing
  explicit shaping (Step 3 gate tests this; if not, a *small* arc shaping is the fallback,
  accepting some optimum-skew risk).
- That capacity binds on tactical refinement *after* activity (Step 4, explicitly
  deferred).

---

## 6. Summary

The AI is passive because **the reward made doing-nothing optimal and the terminal signal
(bankruptcy mirage + mirror-passive self-play) rewarded it.** Capacity and value-head are
refuted as the bottleneck. The fix is a **growth/lead potential** that makes the active
line Φ-attractive, a **redefined unused-flow idle term** that resolves the idle-vs-Outpost
double-count, **capacity-as-a-saturating-potential** so building Outposts is immediately
positive, and a **scaffolded-joint curriculum** (turtle → army-rusher → device-rusher,
PFSP, `--record-opp-value`, turn-search for the long-horizon Device) where the *absence*
of each skill loses games. Measure with an **honest, bankruptcy-excluded win-rate** plus
per-skill behavioral metrics, judged over 30–60-iter trends. Iterate on the **small net**;
revisit capacity only after the net plays actively. **First step: build the honest metric
(`trueWinVsHard`) + per-skill counters — everything downstream is gated on seeing real
skill instead of the mirage.**
