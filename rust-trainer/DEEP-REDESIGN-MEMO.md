# DEEP-REDESIGN-MEMO — diagnosing AI passivity in Colonizing Pirkanmaa AlphaZero

_Authored 2026-06-05. Premise: every potential-Φ retune (s1/s2/s3/i1) and the
bankruptcy-discount terminal-fix (bc1/bc2) have plateaued the champion at
`trueWinVsHard ≈ 0.32-0.41`. This memo re-derives the binding constraints
from the JSONL history + literature + a fresh read of the code, and proposes
ONE concrete training experiment with materially better odds than another Φ
tweak._

---

## §3 — Quantitative findings from the JSONL history

Numbers re-derived from `checkpoints-cnn-{b1,i1,s3,c1,bc1,bc2}/` directly,
not trusting prior agent summaries.

### 3.1 Six-run, last-6-bench means

| run | trueWin | winRate | bnk% | maxSold | outp/g | vill/g | tile% | devDeny |
|----:|--------:|--------:|-----:|--------:|-------:|-------:|------:|--------:|
| b1  | **0.411** | 0.522 | 0.211 | 0.77 | 0.18 | 0.90 | 0.19 | 0.28 |
| i1  | 0.400 | 0.486 | 0.177 | 0.76 | 0.18 | 0.76 | 0.18 | 0.24 |
| s3  | 0.369 | 0.481 | 0.231 | 0.63 | 0.13 | 0.62 | 0.16 | 0.28 |
| c1  | 0.322 | 0.414 | 0.221 | 0.61 | 0.15 | 0.52 | 0.14 | 0.25 |
| bc1 | 0.331 | 0.428 | 0.225 | 0.61 | 0.14 | 0.52 | 0.14 | 0.32 |
| bc2 | 0.358 | 0.436 | 0.175 | 0.66 | 0.16 | 0.57 | 0.15 | 0.23 |

The bankruptcy-discount runs (bc1/bc2) did **not** beat b1. The terminal-z
fix shrank the bankruptcy share (0.21→0.17-0.22) but bled honest win-rate by
an equal amount — confirming agent-B's own skeptic-check (b). The
Outpost-cost rebalance (Option B, arc `sd2`) was the last lever that moved
anything (max-sold 0.7→0.9 transient in b1, the only run that moved a
behavioral metric). **No single lever has produced ≥+0.10 trueWin over b1**.

### 3.2 Self-play contact rate (user obs #2)

User hypothesis "AI doesn't take contact" is **falsified by the data**.
Attack fires 7-10×/game in both bench and self-play (b1 bench last-6:
Attack/game = 9.83; self-play last-20 iters: 9.35). P(≥1 Attack/game) ≈
100% under any reasonable model. The real issue is **contact that doesn't
translate to wins**: most attacks are 1-soldier mop-ups on neutral border
tiles. Honest-conquest wins ~28% (champ) vs ~17% (HARD) — close to parity.

### 3.3 Bench win-cause breakdown (last-6 benches/run, ~360 games)

| run | champ.dev | champ.conq | champ.bnk | HARD.dev | HARD.conq |
|----:|----------:|-----------:|----------:|---------:|----------:|
| b1  | 15 | 121 | 40 | **95**  | 50 |
| i1  | 14 | 121 | 31 | **106** | 47 |
| s3  | 15 | 111 | 40 | **103** | 57 |
| c1  | 10 |  96 | 33 | **112** | 64 |
| bc1 | 16 |  95 | 35 | **102** | 70 |
| bc2 | 15 | 106 | 28 | **111** | 64 |

**The dominant single failure mode is HARD's Device line: ~28% of bench
games (~100/360).** Champ's own Device share is ~4% (15/360).
`hardDeviceBuildRate ≈ 0.40`, `hardDeviceSurvival ≈ 0.72-0.80`,
`deviceDenialRate ≈ 0.22-0.30`. Every ~5th bench game, HARD builds a Device
that survives the countdown unopposed. This **vindicates user obs #4
directly**: whichever AI builds the Device first essentially always wins.
The §6 mechanic says one staged soldier on the device tile (0 defenders)
cracks it — the champ cannot reliably field+deploy that one soldier in
time. `vsDeviceRush` in self-play **DROPS over training** (b1: 0.28→0.17,
i1: 0.31→0.12) — the curriculum signal is going the wrong direction.

### 3.4 The Bridge blindness (user obs #3)

Across **30 saved replay games × 5 runs (~70k frames)**, the AI builds:

- **Bridge: 0 (zero, across every run).**
- Hydro: 0-0.13/game (built only by HARD in some runs).
- Outpost: 0.06/game.

Root cause confirmed by reading `crates/cp-ai/src/candidates.rs:25-40` (the
`Intent` enum) and `:1034` (the candidate dispatch table):

> **`Intent::BuildBridge` does not exist.** The enum has 12 variants
> (BuildFarm/Mine/Village/Outpost/Hydro/Nuclear, Expand, HireSoldier, Attack,
> StackProducer, Pass, BuildStrangeDevice). Building a Bridge is fully
> supported by the simulator (`crates/cp-sim/src/managers.rs:879,:1436`) —
> the NN candidate generator simply never emits a candidate with the
> `Build("Bridge", river_id)` action. The TS mirror
> (`src/ai/nn/candidates.ts`) is identical, parity-locked.

Worldgen places **a single winding river on every map** (~17-18 tiles long;
`src/world/worldgenerator.ts:156-191`). Per-player end-state count of
owned-river-tiles-with-no-building (= expansion dead-ends):

| run | rivers/map | own_river_blocks_avg |
|----:|-----------:|---------------------:|
| b1  | 17.2 | 5.6 |
| i1  | 17.2 | 6.0 |
| s3  | 17.2 | 6.6 |
| c1  | 18.4 | 2.2 |
| bc1 | 18.4 | 3.2 |
| bc2 | 18.4 | 4.0 |

Each unbridged river tile costs ~2 neighbouring expansion-gainers. The AI
leaves **~8-12 tiles of territorial expansion on the table per game** and
cannot rally past the river offensively, even when its army is intact.
(Strict HQ-separation by river in the 5 sampled replay seeds was 0/30, so
the user's "joki välissä → device-builder wins" claim applies to a larger
seed distribution than the saved replays cover — but the within-territory
river-block accounts for the visible structural symptom on every seed.)

---

## §4 — Scientific literature

Six sources extracted, each with the SPECIFIC TECHNIQUE that could apply
here and a transfer judgement.

### 4.1 KataGo: Playout Cap Randomization (Wu 2019)
URL: <https://arxiv.org/abs/1902.10565>, KataGo docs.
**Technique:** On a small fraction p of turns do a deep search (e.g. 600
sims) and record ONLY those for the policy target; other turns use fast
(~100) searches for game-speed. Decouples value training (lots of games)
from policy training (lots of search per recorded position).
**Transfer: HIGH.** Our `--sims 48` × build-prior-floor 0.03 = ~1.4 visits
on the BuildOutpost edge (agent-C §2.2); the same will be true of Bridge.
Playout-cap-randomization gives "192-sim search on the recorded 25% of
positions" at almost no game-count cost.

### 4.2 KataGo: Forced Playouts + Policy Target Pruning
**Technique:** Force each child to receive `n_forced = sqrt(k·P(c)·N(s))`
minimum visits, then compute the policy TARGET on the post-prune tree (so
forced exploration doesn't bias the training target).
**Transfer: HIGH.** Directly addresses the "rare-but-important moves never
visited" problem without distorting policy targets — cleaner than the
current `--build-prior-floor` hack.

### 4.3 AlphaStar: Supervised Pretraining from Replays (Vinyals et al 2019)
URL: <https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/>
**Technique:** Behaviour-clone an expert policy BEFORE any self-play. The
initial supervised agent already beats the in-game Elite AI 95% of the
time; RL then adds an KL-distillation term against the supervised policy to
prevent forgetting.
**Transfer: VERY HIGH** — if HARD/army-rush/device-rush count as "experts."
We have no human replays but three scripted bots that the AI is supposed to
learn to counter, and HARD wins 2/3 of bench games. **A 5-iter BC warmstart
from HARD on candidate distributions would skip the "discover Outpost /
Bridge / Hydro / cracking exists" phase entirely.** Nobody has tried this
in this project.

### 4.4 OpenAI Five: 80/20 PFSP and Selective Scripting
URL: <https://arxiv.org/pdf/1912.06680>
**Technique:** 80% self-play, 20% past selves (win-rate-weighted PFSP).
Hand-script the side-channels the policy isn't meant to learn end-to-end
(hero-item purchase, courier, item reserve).
**Transfer: PARTIAL** — PFSP already adopted (`--pfsp`). The "selective
scripting" idea supports either a scripted device-crack head, or HARD-BC
warmstart (§4.3).

### 4.5 Go-Exploit: Targeted Search Control (Bhatt 2023, AAMAS)
URL: <https://arxiv.org/abs/2302.12359>
**Technique:** Self-play episodes start from an archive of *interesting*
mid-game states, not always from the initial position. Breaks the "AZ only
explores early game" failure mode.
**Transfer: MEDIUM.** Could archive states where HARD has just built a
Device, or where the river is owned but unbridged — exactly the situations
the agent doesn't handle. But requires a state-archive writer + reset
pathway; bigger than a flag.

### 4.6 Asymmetric self-play / role-forcing (Survive-or-Collapse 2025)
URL: <https://arxiv.org/abs/2605.22217>, OpenAI Five
**Technique:** Assign distinct roles to seats so one attacks/proposes, the
other defends/solves. Forces policy distribution to span attack ⨯ defense
rather than collapsing to mirror passivity.
**Transfer: MEDIUM-HIGH.** Concretely: in 50% of self-play games, force
seat-0 to play scripted ARMY_RUSH_PARAMS while seat-1 is the learner. The
learner is *never* mirroring its own passive policy. Differs from current
`--script-opponents`: the latter gates the OPPONENT distribution, this
forces the LEARNER to face aggression in 100% of those games.

**Negative result for skepticism:** intrinsic-motivation / RND / ICM
(curiosity bonuses for under-visited states) is the off-the-shelf answer to
sparse-reward exploration but is **unlikely to transfer** here: our reward
isn't sparse (Φ shaping every turn) and the issue isn't "we never visit
BuildBridge" — it's that **BuildBridge doesn't exist in the action space at
all**. Curiosity over a 12-intent space cannot reward a 13th intent that
doesn't exist.

---

## §5 — Prioritized failure-mode inventory

### F1 — Missing action: BuildBridge is not in the Intent enum (severity 5, conf H)
**Evidence:** `candidates.rs:25-40`, 12 intents, no BuildBridge. Bridge builds
= 0 across 30 replays. Worldgen places a river on every map. ~4-7 owned
unbridged river tiles per player per game.
**Intervention:** add `Intent::BuildBridge` (~80 LOC each side + parity).

### F2 — Missing action: CrackDevice not a first-class intent (severity 5, conf H)
**Evidence:** HARD's Device line = ~28% of bench games (§3.3). Attack intent
CAN crack a device but the candidate-sort doesn't prioritize device tiles;
the value head has no special signal that this single action saves the game.
`--device-credit` exists only for the BUILDER side (`cnn_train.rs:3206`).
**Intervention:** split out `Intent::CrackDevice` + `--device-crack-credit
c` flag.

### F3 — Device-rusher curriculum signal is going BACKWARDS (severity 4, conf H)
**Evidence:** `spVsDeviceRush` drops from ~0.30 to ~0.15 in 4/6 runs. The
learner faces an opponent it cannot crack (F2), value head locks in −1, the
policy converges to "this branch is dead."
**Intervention:** drop device-rusher from script pool until cracker
stabilizes; re-introduce in Step 3 per TRAINING-APPROACH.md.

### F4 — Search horizon collapses on first greedy Pass (severity 4, conf M-H)
**Evidence:** agent-C diagnosis at `cnn_train.rs:549-552`. `--turn-search-spend`
exists but is off by default. c1 turned it on alone → Pass% rose 0.33→0.53
(spend-mode without an active candidate just probes more Pass-alternatives
and finds them slightly negative). Not a standalone fix.
**Intervention:** ON, combined with F1/F2 so a real productive non-Pass
candidate exists.

### F5 — Mirror-passivity Nash attractor (severity 3, conf M)
**Evidence:** `az-pass-collapse-fix` memory + present data. Two passive
nets vs each other → value-head targets ≈0 → PUCT picks Pass → policy
reinforces Pass.
**Intervention:** asymmetric/role-forced self-play (§4.6); deferred to
contingency.

### F6 — No cold-start from a known-good policy (severity 3, conf M)
**Evidence:** AlphaStar/TStarBot-X all BC from expert before self-play
(§4.3). Random init + 50 iters has to rediscover Outpost / Hydro / Device-
crack / (post-F1) Bridge from scratch.
**Intervention:** 5-iter BC warmstart from HARD before self-play. Deferred
to contingency unless F1+F2 fails.

### F7 — Policy-target dilution: 1.4 visits on rare moves (severity 3, conf H)
**Evidence:** agent-C §2.2: 48 sims × prior 0.03 ≈ 1.4 visits. Same for
Bridge once F1 ships.
**Intervention:** `--build-prior-floor 0.03→0.06`, `--sims 48→64`, ideally
KataGo playout-cap-randomization (deferred).

### F8 — No "bridge-opportunity" plane (severity 2, conf M)
**Evidence:** `planes.rs` has `C_RIVER_BLOCK` (channel 22, unbridged owned
river) but no "Bridge here would unblock N tiles" plane. After F1 ships, the
candidate's `local.target_value` needs to encode this.
**Intervention:** `bridge_unblock_count` local feature on the Bridge
candidate (~10 LOC, parity-locked).

### Cross-cut: perception vs incentive (user obs #1)

**Perception is largely fine for attacking** (planes 16/17/19/20: enemy-reach,
my-reach, enemy-conq-soldiers, att-minus-def). **Perception is broken for
the Device-counter and the Bridge-line not because of missing planes but
because of missing ACTIONS**: the C_DEVICE_DEFENSELESS plane (channel 21)
exists, but the policy can't act on it specially — there's no
CrackDevice intent to learn a logit on. The action space, not the eyes, is
the binding gap.

---

## §6 — The single concrete experiment

### 6.1 The intervention

**Add `Intent::BuildBridge` and `Intent::CrackDevice` to the action space.
Add a `bridge_unblock_count` local feature on Bridge candidates. Add
flag-gated `--device-crack-credit c` mirroring `--device-credit` on the
cracker side. Drop the device-rusher from the PFSP script pool until the
cracker stabilizes (set `--script-frac 0.30`, keep army-rusher dominant).
Turn `--turn-search-spend` ON, bump `--build-prior-floor 0.03 → 0.06`,
`--sims 48 → 64`. Cold-start.**

**Why this beats another Φ tune:**

1. **The action-space gap is unexamined.** Every prior experiment stayed
   inside the existing 12 intents. The biggest assumption nobody has tested
   is that 12 intents is the right action space. Data says it isn't:
   Bridge (a buildable building) and CrackDevice (the cheap counter to the
   single biggest loss source) are missing as named intents.
2. **F1 mirrors the Option-B Outpost-gate fix** that finally moved max-sold
   in b1: same evidence pattern (intent histogram = 0 across all runs),
   same fix shape (open the gate-blocked action). The Outpost fix was the
   only intervention that produced any movement in 6 runs.
3. **F2 directly attacks the single largest loss source** (HARD device
   wins ≈ 28%, §3.3). The cracker is mechanically cheap (1 staged soldier
   on a 0-defender tile) — the policy just needs a distinct lever to learn.
4. **Ng-1999 invariance:** shaping cannot escape a wrong terminal-optimum;
   the bankruptcy mirage was the terminal corruption, bc1/bc2 corrected it
   with zero net lift. The remaining gap is action-space + curriculum, not
   Φ. Six Φ retunes have produced no ≥+0.10 lift.

### 6.2 The implementation

- `rust-trainer/crates/cp-ai/src/candidates.rs`:
  - `Intent::BuildBridge = 12`, `Intent::CrackDevice = 13`,
    `INTENT_COUNT 12 → 14`.
  - `fn build_bridge(g, p, cfg)`: cost `bridge_build_cost()`; target = any
    owned River with no building + `river_orientation ∈ {0,1}`; gate by
    early-tier (`cfg.experts == false` allowed — Bridge is the early bridge,
    Hydro is mid-tier); `local.target_value =
    bridge_unblock_count(g, tile, p)`.
  - `fn crack_device(g, p, cfg)`: enumerate when any live enemy has a
    standing StrangeDevice and we can stage ≥1 soldier; emits an
    `Action::Attack` against the device tile with `Intent::CrackDevice`;
    local includes enemy-device-countdown.
  - Update `enumerate()` dispatch. **~80 LOC.**
- `src/ai/nn/candidates.ts`: mirror exactly. **~80 LOC.**
- `rust-trainer/crates/cp-train/src/bin/cnn_train.rs`:
  - `--device-crack-credit c` flag, default 0.0 (bit-identical no-op);
    award +c·|z|·(1 if seat decision was `CrackDevice` else 0) summed at
    terminal (mirror of `--device-credit`).
  - Behavioural metrics: `bridgesPerGame`, `crackAttempts`,
    `crackSucceeds`. **~50 LOC.**
- `training/export-golden.ts` + `golden/` re-export.
- Tests: `bridge_candidate_emits_on_owned_river_no_building`,
  `crack_device_candidate_emits_when_enemy_device_present`,
  `device_crack_credit_zero_is_terminal_only_noop`.

**Parity:** YES (candidates mirror). Golden re-export, parity 8/8.
**Cold-start:** YES (INTENT_COUNT 12→14 changes policy-head dim).
**Arc bump:** NO (no game-rule change, no new gate, no cost change). Model
ID continues as `sd2-az-NNN`, NNN advances.

### 6.3 The launch command

```bash
./rust-trainer/presets/launch.sh \
  --out rust-trainer/checkpoints-cnn-r1 \
  --w-army 0.4 \
  --cap-potential 0.3 \
  --idle-flow-penalty 0.3 \
  --device-crack-credit 0.25 \
  --turn-search-spend \
  --build-prior-floor 0.06 \
  --sims 64 \
  --script-frac 0.30
```

The preset (`presets/common.sh` + `mac-m2.sh`) supplies `--train --net-size
small --turn-search --income-lead-potential 0.5 --tile-potential 0.4
--w-cut 0.15 --record-opp-value --device-potential 0.2 --device-credit 0.15
--pfsp --vs-hard-frac 0.4 --script-opponents --script-grade --tie-penalty
0.4 --stall-rounds 80 --shape-gamma 0.99 --shape-weight 0.3 --cap 150
--games 24 --bench-games 60 --iters 50 --threads 8`. `--bankruptcy-discount`
is intentionally omitted (bc1/bc2 showed it neither helps nor hurts net;
clean read on the new action space). Expected wall-clock: ~3-3.5h on M2 Pro
(sims=64 + spend-mode ≈ 1.5× b1 baseline).

### 6.4 The gate (judge over last-6 benches, gens ~30-49)

**PASS — deploy as new champion:**
- `trueWinVsHard ≥ 0.51` (b1 = 0.41; user threshold +0.10).
- AND `bridgesPerGame ≥ 0.3` (new intent fires).
- AND `deviceDenialRate ≥ 0.45` (was 0.23-0.28; cracker bites).
- AND `hardWins.device / bench_games < 0.18` (was 0.28-0.31; HARD's
  Device line gets cracked).
- AND no regression: `winRate ≥ 0.48`, `bankruptcyWinShare ≤ 0.22`.

**FAIL — kill conditions:**
- `bridgesPerGame < 0.05` after gen 30: the new BuildBridge candidate is
  never selected → F7 is binding harder than estimated → escalate to
  KataGo playout-cap-randomization.
- `deviceDenialRate < 0.30` after gen 30: cracker doesn't learn → escalate
  to HARD-BC warmstart (F6).
- `trueWinVsHard < 0.30` sustained: hard regression from cold-start →
  revert action-space expansion, only ship F3/F4/F7.

### 6.5 The contingency

- **Bridges built but no trueWin lift** → F8: add `bridge_unblock_count`
  as a global PLANE (not just local feature) for spatial CNN reasoning.
  ~30 LOC, cold-start.
- **Cracker fires but devices still win** → search horizon: KataGo
  playout-cap-randomization (§4.1). 25% of decisions at sims=256, only
  those recorded for policy. ~80 LOC.
- **Both still flat** → AlphaStar-style HARD-BC warmstart (§4.3): 5 iters
  of supervised intent-cloning from HARD on random mid-game states before
  self-play, with KL-distillation against the BC policy during RL. ~300
  LOC + 1 day of infra. Highest ceiling, highest cost — but this is the
  proven recipe for action-rich strategy games.

---

## Summary

Six prior interventions targeting Φ, terminal-z, search horizon, and
Outpost cost have produced no ≥+0.10 trueWin lift over b1's 0.41 honest
baseline. The bench data + a fresh read of `candidates.rs` and `planes.rs`
exposes a **mechanical action-space gap nobody has flagged**: two of the
game's most decisive actions — **building a Bridge** and **cracking a
standing enemy Device** — are not first-class intents in the AI's action
space. Bridges: 0 across 30 saved replay games × 6 runs, despite worldgen
placing a river on every map (4-7 owned unbridged dead-ends per player).
HARD's Device line wins 28% of bench games unopposed because the
Attack-on-Device path has no priority signal (the candidate-sort buries it
with everything else). User obs #3 and #4 are mechanically explained.

The experiment: **add `Intent::BuildBridge` + `Intent::CrackDevice`, add
`--device-crack-credit` shaping, drop the device-rusher until the cracker
stabilizes, turn search-spend ON and bump prior-floor/sims so the new
intents accumulate the visits they need to enter the policy target.
Cold-start, 50 iters, judge on the four-condition gate (trueWin ≥ 0.51 +
bridgesPerGame ≥ 0.3 + deviceDenialRate ≥ 0.45 + HARD device share <
0.18).** If bridges build but trueWin is flat, escalate to the planes
fix; if devices still survive, escalate to playout-cap-randomization;
if both, escalate to HARD-BC warmstart. Each failure mode gives diagnostic
signal the previous Φ-only sweeps did not.

---

### EXACT LAUNCH COMMAND

```bash
./rust-trainer/presets/launch.sh \
  --out rust-trainer/checkpoints-cnn-r1 \
  --w-army 0.4 \
  --cap-potential 0.3 \
  --idle-flow-penalty 0.3 \
  --device-crack-credit 0.25 \
  --turn-search-spend \
  --build-prior-floor 0.06 \
  --sims 64 \
  --script-frac 0.30
```

Sources:
- KataGo (Wu 2019): <https://arxiv.org/abs/1902.10565> / <https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md>
- AlphaStar (Vinyals 2019): <https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/>
- OpenAI Five Dota 2 (Berner 2019): <https://arxiv.org/pdf/1912.06680>
- Go-Exploit (Bhatt 2023, AAMAS): <https://arxiv.org/abs/2302.12359>
- Asymmetric self-play (Survive-or-Collapse 2025): <https://arxiv.org/abs/2605.22217>
- AlphaZero passivity / Tablut: <https://arxiv.org/pdf/2604.05476>
