# Handoff — START DAgger here (Colonizing Pirkanmaa AI)

_Last updated 2026-06-07 after a full redesign + overnight ablation session. **Read this first.**
This handoff is purpose-built to START THE DAgger EXPERIMENT — the recommended next lever to break the
one remaining wall. Everything you need (state, code map, exact plan, commands, gotchas) is here._

> Deployed game = TS/Phaser (`src/`). The **Rust trainer** (`rust-trainer/`) trains the AlphaZero net
> that is redeployed into the TS game. Parity Rust⇄TS is bit-exact, locked by
> `cargo run -p cp-train --bin parity --release` (must be **8/8**). Arc is now **sd3**.

---

## TL;DR — the one job

**Build DAgger** (Dataset Aggregation) to fix the *only* thing blocking a strong learned AI: the net
won't learn **economy-patience** (build Mine→Outpost to raise the soldier cap). We proved the army
*intent* is learnable but the economy *discipline* isn't — because plain imitation suffered
distribution-shift (it imitated the strong-army expert's GOOD-economy states, then at play time hit its
OWN poor-economy states and mis-acted). DAgger fixes exactly this: **label the NET'S OWN visited states
with the strong-army expert's action, retrain, iterate.** Then RL-fine-tune from that strong seed with
the KL-anchor (already built). The strong-army scripted bot is genuinely strong (beats HARD ~52%) and is
cheaply *queryable*, so this is the highest-EV path.

**Why DAgger over PPO:** the economy-patience is a *reactive* policy the expert already masters without
search — so it's learnable as a policy mapping, no long-horizon credit needed. DAgger is a moderate hook
+ a loop; PPO is a ~600-LOC rewrite. Do DAgger first; keep PPO as the fallback to push *past* the teacher.

---

## Where we are (state at handoff)

**Best model ever: `rust-trainer/checkpoints-cnn-foundation0-prep6/champion-best.json` — trueWin 0.55
peak** (registered as `models/sd3/az/sd3-az-001`). Best across ALL runs (prev best r4 0.53, asym1 0.52),
on the new sd3 arc. BUT it wins by ≤1-soldier econ-conquest — **no army**. The dashboard (`:8787`) points
to it (restart: `setsid nohup npx vite-node training/serve-dashboard.ts -- --dir <dir> --port 8787 &`).

**Shipped + committed this redesign (all parity 8/8):**
- **P1 eyes**: `Intent::MarchSoldier` + distance-to-enemy-HQ/device planes + my-budget plane.
  (Conv *dilation* was added then DISABLED in the two `spatial_net.rs` production ctors — it routed
  through the slow general conv path, ~2.4× slower; the distance planes already give board-spanning
  vision. The dilation primitive + its grad-check test remain for future use.)
- **P2 league**: 4 strong bankruptcy-proof archetype bots — `HardAi::rusher()/fortress()/device_rush()
  (rebuilt)/strong_army()` in `crates/cp-ai/src/hard_ai.rs`. `STRONG_ARMY` is the yardstick (beats HARD
  ~52%). Quality harness `crates/cp-train/src/bin/league_health.rs`; head-to-head `league_h2h.rs`.
- **P1.5 economy rebalance (THE root-cause fix, arc sd2→sd3)**: Outpost metal upkeep −15→−5, soldier
  metal −50→−30 (mirrored `resources.rs`⇄`src/core/resources.ts` + the candidate metal-gates). Military
  is now FUNDABLE. Documented in `CLAUDE.md` "Fidelity constraints".
- **P6 dashboard**: per-opponent win-rate chart + activity/passivity panel + league replays
  (`training/serve-dashboard.ts`); per-opponent bench metrics `benchVs{Rusher,Fortress,DeviceRush,
  StrongArmy,Hard}` in `cnn_train.rs`.
- **P3 imitation pipeline (the DAgger foundation — already built!)**: the recorder is FIXED and the BC +
  KL-anchor modes exist (details below).

**The full hour-by-hour trail + every negative result is in `rust-trainer/OVERNIGHT-AUTONOMOUS-LOG.md`.**

### The wall (why DAgger)
FIVE approaches all failed to make the net build an army:
| run | result | why |
|---|---|---|
| foundation (eyes+econ+league, min shaping) | 0.55 peak → 0.43 | no army; ≤1-soldier conquest |
| P4 reward-shaping (cap-potential+crack-credits) | 0.33 | outposts flat — reward can't make RL *discover* the investment |
| P3 imitation + KL-anchor λ0.1 | 0.05 (collapse) | anchored to a WEAK economy-blind seed → worse than cold start |
| imit2 (λ0.05 + strong cap) | 0.05 | hires soldiers (112) but BuildOutpost 0 — income/tile Φ out-competes cap |
| capforce1 (cold + cap0.5/income0.1) | 0.38 | outposts only marginally up; still no real army |

→ The army INTENT is learnable; the economy PATIENCE (delayed Mine→Outpost payoff) is the wall.
MCTS's ~30-turn horizon can't credit it, and plain BC mis-acts on its own states. **DAgger is the fix.**

---

## The DAgger plan

### Algorithm (standard DAgger, expert = `HardAi::strong_army()`)
```
π₀ = the existing BC seed  (rust-trainer/checkpoints-cnn-sup-p3/champion-supervised.json)
D  = the existing BC dataset  (states already labelled by the expert's own play)
for round i in 1..K:                       # K ≈ 3-5
    # 1. ROLL OUT the current net π_{i-1} to collect the states IT visits
    play N games (e.g. 200) with π_{i-1} driving (greedy argmax, the deploy policy),
      vs a mix of league opponents; at EVERY net decision-state s_t, record s_t.
    # 2. EXPERT-LABEL each visited state with what strong_army would do there
    for each s_t: a*(s_t) = first intent strong_army takes from s_t   (clone s_t, run record_turn)
    D = D ∪ {(s_t, a*(s_t))}
    # 3. RETRAIN π on the AGGREGATED D (cross-entropy on intent; train the value head too!)
    π_i = supervised-train fresh small net on D  (the existing --supervised path)
    # 4. VALIDATE π_i  (--validate-net): does it now BUILD Outposts + field >1 soldier in PLAY?
final π = π_K → champion-dagger.json
```
The crux of DAgger: π is trained on the distribution of states *it actually visits*, with correct expert
labels there — eliminating the distribution-shift that killed plain BC.

### The ONE new piece you must write: the expert-label-at-state hook
Everything else is reuse. You need: **given an arbitrary `Game` state `s` + player `p`, return the
`Intent` the strong-army expert would choose first from `s`.**

The primitive already exists — `HardAi::record_turn(g, player, sink)` in `crates/cp-ai/src/hard_ai.rs:858`
plays a full expert turn and invokes `sink(intent, &Game)` once per realised action, with the state
captured at that action's phase start. So:
```rust
// expert label = the FIRST action strong_army would take from state s
fn expert_label(s: &Game, p: PlayerId) -> Option<Intent> {
    let mut g = s.clone();                 // record_turn MUTATES g — clone first
    let mut bot = HardAi::strong_army();
    let mut first: Option<Intent> = None;
    bot.record_turn(&mut g, p, &mut |intent, _state| {
        if first.is_none() { first = Some(intent); }
    });
    first                                  // None ⇒ expert would Pass ⇒ label Pass
}
```
That's it — `record_turn` already mirrors `run_turn`'s exact phase order and classifies per-action intents
(the fix that killed the old Pass-collapse bug). No new bot logic needed.

### Reusable infra (file:line) — wire DAgger as a new `--dagger` mode in `cnn_train.rs`
- **Net rollout (collect visited states):** the greedy net-play loop already exists for `--validate-net`
  (`run_validate_net`, `cnn_train.rs:6906`) and the "enumerate → `score_candidate_into` argmax →
  `execute_action`" turn driver (`cnn_train.rs:~510-540`). Reuse it to drive π and, at each decision,
  (a) emit a `SupervisedExample`-shaped state and (b) call `expert_label(state)` for the target.
- **Example type + BC training:** `SupervisedExample`, `supervised_play_one_game` (`cnn_train.rs:6113`),
  and the `--supervised` trainer (`cnn_train.rs:6889`) — feed them the AGGREGATED dataset. The
  state→planes encoding the recorder uses is the same one the net consumes (PLANE_COUNT 27, INTENT_COUNT
  16, LOCAL_DIM 18, value_scalar_dim 12).
- **Dataset gen reference:** `run_supervised_from_hard` (`cnn_train.rs:6316`) shows the existing
  data-gen loop + the class-balance flags `--pass-keep/--attack-keep/--outpost-boost/--hire-boost/
  --mine-boost` and the `--league` weighted mix (`LeagueBot` at `cnn_train.rs:6067`). Reuse the same
  Pass-subsampling so D isn't swamped by Pass.
- **Expert:** `HardAi::strong_army()` (`hard_ai.rs:731`) — also rusher/fortress/device_rush/marcher/hard.
- **Candidates / Intent:** `enumerate` (`candidates.rs:1488`), `execute_action` (`candidates.rs:1527`),
  `Intent` enum (`candidates.rs:25`, INTENT_COUNT 16). The action→Intent classification lives in
  `TurnSnapshot::classify_into` (`hard_ai.rs:547`).

### Implementation notes / pitfalls
1. **TRAIN THE VALUE HEAD PROPERLY.** The P3 seed's value head was weak (value_loss ~0.74) → under full
   MCTS it caused 96% Pass collapse, which is why we validated greedy (sims=1). For the DAgger seed to be
   RL-fine-tunable, label each state with a value target too (terminal z of the rollout game, or the
   expert's win/loss), and train value MSE alongside intent CE. A good value head is what lets the
   subsequent MCTS-RL not collapse.
2. **Granularity:** label the net's *per-action* decision states (mid-turn), matching how `record_turn`
   emits per-action. Clone the state at each net decision; don't reuse a mutated clone.
3. **Opponent mix during rollout:** roll π out vs the league (esp. STRONG_ARMY + HARD) so the visited
   states cover the contested mid/late-game where Outpost decisions matter.
4. **Keep the army-chain classes from being drowned:** apply the same `--pass-keep ~0.15` subsampling.
5. **Validate the RIGHT thing:** `--validate-net` must show **outpostsPerGame rising (>0.3), maxSoldiers
   >1, BuildOutpost/BuildMine intents present in play** — not just trueWin. That's the make-or-break:
   does the economy discipline transfer to the net's OWN play?

### Validation gate for the DAgger seed (before RL)
PASS if, in `--validate-net` greedy play (≥40 games vs HARD): **builds ≥0.3 Outposts/game AND peak
soldiers >1.5 AND trueWin > 0.25 standalone** (a strong standalone seed). If outposts still ~0 after 3
DAgger rounds, the expert-label hook or the state encoding is wrong — debug before RL.

---

## After DAgger: the KL-anchored RL fine-tune (ALREADY BUILT)
Once the DAgger seed validates as an army-builder, run the existing RL fine-tune anchored to it (the P3
flags work; the only reason P3 failed was the seed was weak):
```bash
cd rust-trainer && RAYON_NUM_THREADS=16 ./target/release/cnn_train --train \
  --turn-search --turn-search-spend --net-size small \
  --init       checkpoints-cnn-dagger/champion-dagger.json \
  --kl-anchor 0.1 --kl-anchor-net checkpoints-cnn-dagger/champion-dagger.json \
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
There is NO `--kl-decay` flag (not implemented). If you want decay, either add one (lower λ over iters)
or re-launch at a lower λ partway. Watch: **outpostsPerGame + maxSoldiersPerGame must RISE** and trueWin
break 0.55 with Pass% < 25%. Register a baseline-beating champion: `npm run models -- register
<champion-best.json> --arc sd3 --type az`.

---

## Build / gates / run
```bash
# Prereqs: Rust stable + Node 22 (.nvmrc). From repo root:
npm install
cd rust-trainer && cargo build -p cp-ai -p cp-train --release && cd ..

# Gates after ANY change:
cd rust-trainer && cargo run -p cp-train --bin parity --release   # MUST be 8/8 (DAgger is parity-free)
cargo test -p cp-ai --release                                     # ~88 tests
cd .. && npx tsc --noEmit                                         # exit 0

# DAgger data-gen + train (once implemented) use --threads for parallelism; validate with --validate-net.
# Dashboard: setsid nohup npx vite-node training/serve-dashboard.ts -- --dir rust-trainer/<run> --port 8787 &
```

## Operational gotchas (learned the hard way)
- **NEVER `pkill -f`/`pgrep -f` with a pattern that appears in the SAME shell command** — it self-matches
  the running shell and SIGTERMs it (exit 144). Kill training by PID via `ps -C cnn_train -o pid
  --no-headers`; kill the dashboard by port via `lsof -ti:8787 | xargs kill`.
- **Keep only ONE 16-thread training run live** (machine has ~20 cores; leave ~4 free).
- **The weak-seed lesson:** warm-starting/anchoring RL to a WEAK net is *worse than cold start* (P3
  collapsed to 0.05). The DAgger seed MUST validate as strong standalone before you anchor RL to it.
- **`spatial_net::tests::forward_backward_equivalence_golden`** can fail as a pre-existing SIMD/target-cpu
  numeric-drift golden (hardware-dependent) — unrelated to logic; re-stamp if it's only numeric drift.
- **`src/ai/nn/weights.ts`** is a stale 64-dim placeholder; its TS arch test fails until a retrained
  champion is exported to TS (deploy step, separate from training).
- Disk: `checkpoints-cnn-sup-p3/dataset.json` is ~1.7 GB (the BC dataset) — keep it (re-usable as D₀) or
  gzip; disk is fine (~870 GB free).

## Key files
- DAgger lives in: `rust-trainer/crates/cp-train/src/bin/cnn_train.rs` (new `--dagger` mode) +
  `rust-trainer/crates/cp-ai/src/hard_ai.rs` (`record_turn` — the expert hook; already there).
- Parity-free: hard_ai.rs, cnn_train.rs (training loop). Parity-locked (DON'T break): candidates.rs⇄
  candidates.ts, resources.rs⇄resources.ts, planes.rs, spatial_net.rs (arch).
- Best model: `rust-trainer/checkpoints-cnn-foundation0-prep6/champion-best.json` = `models/sd3/az/sd3-az-001`.
- BC seed + dataset: `rust-trainer/checkpoints-cnn-sup-p3/{champion-supervised.json, dataset.json}`.
- Full session log: `rust-trainer/OVERNIGHT-AUTONOMOUS-LOG.md`. Design: `rust-trainer/TRAINING-V2-PROPOSAL.md`
  (§6/§7 imitation+anchor), `GAME-MECHANICS.md` (verified rules), plan
  `~/.claude/plans/suunniitellaan-koko-koulutusta-uudelleen-delegated-storm.md`.

## Fallbacks / strategic notes
- **If DAgger caps at the teacher's strength (~0.52)**: that's still a qualitative win (a net that BUILDS
  ARMIES + is robust vs the strong league, not just econ-conquest). To EXCEED the teacher, the RL-from-
  strong-seed step (above) is the path; if it caps, escalate to **PPO+GAE** (replace the MCTS policy
  target with PPO+GAE for long-horizon credit on the Outpost→army payoff — the plan's reserved lever).
- **If you just want the strongest deployable opponent TODAY**: the `strong_army` scripted bot already
  beats HARD ~52% and out-plays every neural net we trained, and it has a parity-locked TS mirror — it is
  shippable now. The neural-net effort is for *exceeding* the scripts (research goal).
