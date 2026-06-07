# Overnight autonomous session log

_Started 2026-06-07. Claude is driving the training-redesign plan autonomously while the user sleeps.
Goal: complete the plan → train the best, highest-quality AI. Self-paced ~30-min checks (/loop).
Heavy work is offloaded to separate agents to keep the main context clean. This file collects status,
measurements, decisions, and PROBLEMS for the user to review later._

Dashboard: http://127.0.0.1:8787/ (live). Plan: `~/.claude/plans/suunniitellaan-koko-koulutusta-uudelleen-delegated-storm.md`.

## ☀️ MORNING REPORT (read this first) — autonomous session ended ~06:25, 2026-06-07

**What shipped + committed tonight (all solid, reusable):**
- **P1 — Complete eyes**: `Intent::MarchSoldier` + distance-to-enemy-HQ/device planes + my-budget plane
  (dilation added then DISABLED — it routed through the slow conv path, ~2.4× slower, and the distance
  planes already give board-spanning vision). Parity 8/8.
- **P2 — Strong league**: 4 rebuilt archetype bots (rusher/fortress/device/strong-army) + quality harness.
  STRONG_ARMY retuned from weakest→strongest (beats HARD ~52%). Two real bugs fixed (can't garrison own
  HQ → ring-defense; h2h probe undercount).
- **P1.5 — Economy rebalance (THE root-cause fix, arc sd2→sd3)**: Outpost metal upkeep −15→−5, soldier
  metal −50→−30, so military is finally FUNDABLE. Parity 8/8, CLAUDE.md updated.
- **P6 — Dashboard**: per-opponent win-rate chart, activity/passivity panel, league replays. Live :8787.
- **P3 — Imitation pipeline**: fixed the broken supervised recorder (per-action intent capture; dataset
  now has the real army chain, 0 Pass-fallbacks).
- **11 commits total.** Best model registered: **`models/sd3/az/sd3-az-001`** (= foundation0-prep6).

**BEST MODEL: `checkpoints-cnn-foundation0-prep6/champion-best.json` — trueWin 0.55 (peak).** This is the
BEST across ALL runs ever (prev best r4 0.53, asym1 0.52), on the new sd3 arc. Dashboard points to it.

**THE KEY SCIENTIFIC FINDING (the wall):** The plateau's true cause is the **ECONOMY-PATIENCE skill** —
the multi-turn delayed-payoff investment of building Mine→Outpost to raise the soldier cap (HQ+1 → +3/
Outpost). The AI learns the army INTENT readily (imitation gives 0% Pass, aggressive HireSoldier) but
will NOT build the economy the army needs — it takes Farm/Expand (immediate income/tile reward) over the
Outpost (delayed payoff). FIVE approaches all failed to break it: foundation (0.55 peak→0.43), P4 reward-
shaping (0.33), P3 imitation+KL-anchor (0.05 — anchoring to the weak economy-blind seed is WORSE than
cold start), imit2 lighter-anchor+strong-cap (0.05), capforce1 cold-start+Φ-rebalanced cap 0.5/income 0.1
(0.38, outposts only marginally up). The economy fix made the army AFFORDABLE but the AI still won't
DISCOVER/VALUE the investment — a genuine long-horizon credit-assignment wall (MCTS ~30-turn horizon vs
the multi-turn Outpost payoff).

**RECOMMENDED NEXT (needs your decision — both are big/architectural, deliberately NOT done autonomously):**
1. **DAgger** (likely the cheaper win): the imitation seed failed only from distribution-shift — it imitated
   the strong-army expert's GOOD-economy states but at play time sees its OWN poor-economy states and
   mis-acts. Fix: label the NET'S OWN visited states with the strong_army bot's action, retrain, iterate,
   then RL-fine-tune. Needs a new expert-labeling hook (~moderate). The strong_army scripted bot is
   genuinely strong (0.52 vs HARD), so good imitation → a strong AI.
2. **PPO + GAE** (the plan's reserved final lever): replace the MCTS policy target with PPO+GAE for long-
   horizon credit on the Outpost→army payoff (~600 LOC). Bigger bet.

All experiment dirs preserved (`-baseline`, `-prep6`, `-failed`, `-collapsed`, `-done`) for inspection.
Autonomous loop STOPPED here (reached the safe limit; the remaining levers warrant your call).

---

## State at handoff (start of autonomous session)
Committed this session (newest first): `53d14a7` E-foundation prep (league curriculum + dilation off);
`3d17277` P2 re-tune; `937b3be` P1.5 economy rebalance (root-cause fix); `78edff5` fortress fixes;
`3d08f92` STRONG_ARMY tuned strongest; `e4ec198` P2 league; `6333b57`/`4e18b2e` P1 eyes.

Foundation run (`checkpoints-cnn-foundation1`) LAUNCHED: cold-start small non-dilated net, SD3 league
curriculum, asym1 recipe, minimal army-size shaping. ~1.25s/game, ~2-3h / 200 iters. Tests whether
complete-eyes + viable-military-economy + strong-league break the 14-run trueWin≈0.44 plateau.

## Decision log
- (start) Let the foundation run finish for a clean headline trueWin result rather than killing it to
  add per-opponent bench metrics. The next run uses the metrics-enabled binary + improved dashboard.
- (start) P6 dashboard (incl. per-new-league-opponent win-rates) delegated to a separate agent.

## Gate for the foundation run (Phase-1 decision)
PASS = league/trueWin ≥ 0.55 AND maxSoldiers ≥ 3 AND Pass% < 25% AND no >0.05 regression over last 30
iters → refine. FAIL → escalate: P3 imitation+KL-anchor, then P4 decisive reward, then PPO pivot.

## Iteration notes (appended each ~30-min check)
- T0: run launched, gen-0 policyLoss 0.92, ~1.25s/game. Dashboard live. P6 agent dispatched (background).
- T0+~15min (gen 17): HEALTHY ENGAGEMENT. Self-play decisive & conquest-heavy (spDecisive 6/spConquest 5
  per iter), spBankruptcy 0, contactRate 1.0 (NOT passive). Intents/iter: HireSoldier 55, Attack 20,
  MarchSoldier 5, Pass 101, BuildOutpost 0. Bench @gen15: trueWinVsHard **0.45**, winRate 0.50, maxSoldiers
  0.77, outposts/game 0.35, champSoldierBins {0:19,1:39,2:0,3:1,4+:1}, bankruptcyWinShare 0.10.
  READ: the AI is engaging militarily (hiring/attacking/marching) — a real change from passive prior runs —
  but still fields ≤1 soldier in most games and builds ~0 outposts EARLY, so trueWin sits at the old 0.45.
  THE TEST: does trueWin climb past 0.55 AND maxSoldiers/outposts rise as training continues (the economy
  fix should now make army-building pay)? Watching the trajectory over the next checks.

- T0+~40min: P6 dashboard agent DONE + committed (`P6` commit). Added per-opponent bench metrics
  (benchVsRusher/Fortress/DeviceRush/StrongArmy/Hard) + dashboard per-opponent chart, activity panel,
  league replays, glyphs. Parity 8/8. Pre-P6 foundation run reached **gen 34, trueWin 43.3%** (iter-30:
  conquest-heavy C23, maxSol 0.7/g, outposts 0.17/g, champ-device 5% built / surv 0.67). Preserved as
  `checkpoints-cnn-foundation0-prep6`. RELAUNCHED foundation1 on the metrics binary (PID 3178334) so the
  per-opponent dashboard populates. Dashboard live :8787 on the fresh run.
- KEY EARLY READ: economy+eyes+league with MINIMAL shaping is tracking the SAME ~0.43 plateau and the AI
  still fields ≤1 soldier / ~0 outposts at gen 30. So the economy fix ALONE likely does NOT make the AI
  VALUE army-building — it wins by ≤1-soldier conquest. Expect the gate to FAIL → escalate to P4 decisive
  reward (nudge outpost-build + cap-fill + efficient conquest, NOT army size) then P3 imitation from the
  league (which DOES build armies). Prepping P4 design now (agent) so it's ready when the baseline confirms.

- P4 DESIGN READY (agent done). No win-shape flag needed; existing levers suffice. PLAN: let the
  foundation baseline accumulate a few per-opponent benches (~gen 20-30), capture it, then launch P4-A
  (kill foundation by PID, NOT pkill -f). P4-A = army-ENABLING chain (small/saturating, not army-size):
  ```
  ./target/release/cnn_train --train --turn-search --turn-search-spend \
    --income-lead-potential 0.5 --tile-potential 0.4 --w-cut 0.15 --record-opp-value \
    --device-potential 0.2 --device-credit 0.15 --device-crack-credit 0.2 --hq-crack-credit 0.2 \
    --cap-potential 0.2 --w-army 0.15 --bankruptcy-discount 0.5 \
    --pfsp --script-opponents --script-frac 0.7 --tie-penalty 0.4 --stall-rounds 80 \
    --shape-gamma 0.99 --shape-weight 0.3 --cap 150 --games 24 --bench-games 60 --threads 16 \
    --net-size small --vs-hard-frac 0.3 --lr 0.003 --epochs 2 --iters 200 --bench-every 5 \
    --replay-every 25 --out checkpoints-cnn-p4-decisive1
  ```
  P4-B (aggressive, if P4-A fields-but-stalls): + --w-soldier-forward 0.15 --bankruptcy-discount 0.7
  --cap-potential 0.25 --w-army 0.2 --out checkpoints-cnn-p4-decisive2.
  GATE: trueWin>0.50 AND outposts/game & maxSoldiers RISE AND Pass%<25% AND spVsStrongArmy/Fortress tick up.

- T0+~70min (02:57): FOUNDATION BASELINE captured + preserved (`checkpoints-cnn-foundation1-baseline`):
  gen10 trueWin 0.43, outposts 0.22, maxSold 0.70; per-opp benchVs Rusher 0.25 / Fortress 0.67 /
  Device 0.25 / StrongArmy 0.08 / Hard 0.50. CONFIRMS: economy+eyes+league w/ minimal shaping = same
  ~0.43 plateau, no army-building, strong league (StrongArmy/Rusher) unbeaten. → LAUNCHED **P4-A**
  (`checkpoints-cnn-p4-decisive1`, PID 3181598): army-ENABLING reward active (cap-potential 0.20,
  w-army 0.15, device/hq-crack-credit 0.20, bankruptcy-discount 0.50, turn-search-spend). Dashboard
  re-pointed to p4-decisive1. TEST: does the cap→fill→crack chain emerge (outposts/maxSold RISE) and
  trueWin break >0.50? Gate at ~gen 40-60. If P4-A fields-but-stalls → P4-B; if no army emerges → P3
  imitation from the league.

- T0+~100min (03:29): P4-A @gen27 — MIXED. trueWin oscillating 0.25-0.38 (≈baseline 0.43, not climbing);
  army chain NOT emerging (outposts flat ~0.2, soldierBins {0:32,1:27,4+:1}). BUT much more ACTIVE
  (Attack 150, MarchSoldier 24/iter, Pass 75) and BETTER vs the strong league: benchVsStrongArmy
  0.08→0.33, Device 0.25→0.42, Hard ~0.42. Read: bankruptcy-discount correctly pushed it off the cheap
  ≤1-soldier line, but RL self-play hasn't DISCOVERED the multi-turn outpost→army investment (the MCTS-
  horizon / outside-manifold problem) — the cap-potential nudge alone isn't enough. → Let P4-A run to
  ~gen 55 (decision threshold); PREP P3 imitation in parallel (agent) so it's ready. If army still flat
  @gen55 → launch P3 (seed the policy with army-building demos from the SD3 league, the handoff's
  recommended move).

- T0+~130min (04:02): P4-A @gen52 — **GATE FAILED**. trueWin oscillating 0.25-0.37 across gen 15-50
  (below baseline 0.43, never approaching 0.50); army chain did NOT emerge (outposts 0.1-0.33 oscillating,
  maxSold ~0.55-0.70 flat, soldierBins {0:29,1:29,4+:2}). benchVsHard fell to 0.17. CONCLUSION: the
  cap-potential/w-army reward nudge is INSUFFICIENT — RL self-play cannot DISCOVER the multi-turn
  outpost→army investment (MCTS-horizon / outside-manifold), confirming the need for imitation seeding.
  → P4 reward-shaping alone = does NOT break the plateau (clean ablation result). ESCALATE to P3.
  P3 prep agent still running (building champion-supervised.json in checkpoints-cnn-sup-p3). PLAN: on the
  P3-agent completion, IF its supervised net validates as an army-builder → kill P4-A by PID, launch
  P3 RL fine-tune (--init champion-supervised.json --kl-anchor), re-point dashboard. Keeping P4-A running
  as the P4 ablation record until then (can't run two 16-thread jobs).

- T0+~160min (04:34): P4-A @gen75 still failed (trueWin 0.33, outposts 0.18, maxSold 0.55 — flat,
  confirms the negative result). P3 prep agent STILL running (~1h): produced champion-supervised.json
  (04:13) + a 1.8GB dataset.json (04:16), process alive — likely validating/training. Waiting for its
  completion to validate (does the supervised net build army?) then launch P3 RL. No interference.
  NOTE: 1.8GB dataset is large — watch disk; the agent may have over-generated. Will review on completion.

- T0+~190min (~04:40): P3 imitation agent DONE + committed (`8c9179e`). RECORDER FIXED (per-action
  HardAi::record_turn → army-chain dataset: BuildMine 21% / BuildOutpost 14% / HireSoldier 11% /
  Attack 24%, 0 Pass-fallbacks). Supervised net (10314 params, 12 epochs) VALIDATION: imitates army
  INTENT (hires 32%, attacks, **0% Pass in greedy play** — out of the attractor) BUT economy-gate-
  blocked (builds ~0 Mines/Outposts in play → cap stuck at 1, trueWin 0.05 standalone). KEY INSIGHT:
  the army chain = (a) army INTENT [imitation-learned ✓] + (b) ECONOMY-PATIENCE [build Mine→Outpost to
  raise cap — NOT learned ✗]. → LAUNCHED P3 RL fine-tune (checkpoints-cnn-p3-imitation1, PID 3208639):
  warm-start supervised + KL-anchor λ=0.1 (no decay flag) + cap-potential 0.2 + the P4 recipe. TEST:
  does RL + anchor + cap-reward now teach (b) so outposts/maxSoldiers RISE and trueWin breaks 0.50?
  P4-A preserved as -failed. Dashboard re-pointed to p3-imitation1. Disk fine (876G free; 1.7G dataset
  kept for possible re-train). WATCH: KL-anchor pulls toward the no-economy supervised policy — if it
  prevents economy-building, lower λ or improve the supervised seed (DAgger / stronger Mine/Outpost
  emphasis). Supervised value head weak (vloss 0.74) → 96% Pass under full MCTS; watch for early Pass
  collapse in the RL run.

- T0+~220min (05:13): P3 RL (imitation1) **COLLAPSED** — gen30 trueWin 0.02-0.07 (WORSE than baseline
  0.43 and P4 0.33), soldierBins {0:59,1:1} = ~0 soldiers, 0 outposts. The KL-anchor λ=0.1 PINNED the
  policy to the weak economy-blind supervised seed (standalone trueWin 0.05) + the weak supervised value
  head dragged MCTS. Anchoring RL to a 0.05 net caps it at ~0.05. Preserved as -collapsed. → relaunched
  **imit2** (PID 3216026): lighter anchor λ=0.05 + STRONGER cap-enabling reward (cap-potential 0.2→0.4)
  + w-soldier-forward 0.1, to free the policy from the weak seed while forcing the Outpost step.

  *** STRATEGIC STATE (for the user) ***: THREE methods now fail at the SAME wall — the ECONOMY-PATIENCE
  skill (the delayed-payoff Mine→Outpost investment that raises the soldier cap): (1) foundation
  (eyes+economy+league, minimal shaping) → plateau 0.43, ≤1 soldier; (2) P4 reward shaping → 0.33, no
  army; (3) P3 imitation+KL-anchor → 0.05 collapse. The army INTENT is learnable (imitation gave 0%
  Pass, hires+attacks) but the AI won't BUILD THE ECONOMY that the army needs, even though the rebalance
  made it affordable. This is a genuine long-horizon credit-assignment wall (MCTS ~30-turn horizon vs the
  multi-turn Outpost payoff). If imit2 (strong cap reward) also fails, the remaining levers are BIG and
  warrant a user decision: (A) DAgger — fix the imitation distribution-shift by labeling the NET's OWN
  visited states with the strong-army expert's action (the seed failed because it only saw expert states,
  not its own poor-economy states); needs a new expert-labeling hook, ~complex. (B) PPO+GAE — the plan's
  reserved final lever for exactly this horizon problem (~600 LOC rewrite). Both are multi-hour, uncertain.
  CURRENT BEST DEPLOYABLE: foundation1-baseline (~0.43 vs HARD, active conquest play). Everything committed.

- T0+~255min (05:48): imit2 ALSO FAILED (gen29 trueWin 0.05, outposts 0). Sharpened diagnosis: it HIRES
  aggressively (HireSoldier 112) but BuildOutpost 0 — builds Farms(32)/Expands(66) instead → the
  income-lead(0.5)+tile(0.4) Φ OUT-COMPETES cap-potential, so the policy takes the easy-expansion line
  over the army-enabling Outpost. Also confirmed: warm-start from the weak seed is STRICTLY WORSE than
  cold start (0.05 vs 0.43). → preserved -failed. LAUNCHED capforce1 (PID 3222817): COLD start + Φ
  REBALANCED so the army step dominates — cap-potential 0.5, income-lead/tile cut to 0.1, w-army 0.2,
  w-soldier-forward 0.15, bankruptcy-discount 0.5. Last cheap shot; next check = WIND-DOWN decision.
- ★ BEST MODEL OF THE NIGHT: **checkpoints-cnn-foundation0-prep6** PEAKED **trueWin 0.55** (champion-best.json
  there) — the eyes+economy+league foundation paradigm DID hit the gate's trueWin threshold at peak
  (0.43 was the regressed mean; classic peak-then-regress). maxSoldiers≥3 NOT met (still ≤1-soldier
  conquest), so not a full gate pass, but a legit decent model + the recommended deploy candidate.

## PROBLEMS / risks collected
- P4 RESULT: reward shaping (cap-potential 0.2 + w-army 0.15 + crack-credits + bankruptcy-discount 0.5)
  did NOT make army-building emerge or break trueWin past baseline. The army line is undiscoverable by
  RL self-play here → imitation (P3) or PPO is required. Logged as a definitive ablation finding.
- WATCH: outposts NOT rising under cap-potential 0.2 → the army investment may be undiscoverable by RL
  self-play alone (MCTS can't credit the delayed payoff). Strengthens the case for P3 imitation. If P3
  also caps, the structural fix is PPO+GAE (longer-horizon credit) per the plan.
- LESSON (recurring, now firmly noted): NEVER use `pkill -f`/`pgrep -f` with a pattern that appears
  literally in the SAME shell command — it self-matches the running shell and SIGTERMs it (exit 144).
  Hit it for both `cnn_train.*foundation1` and `serve-dashboard.ts`. Kill by PID, by `-C cnn_train`
  (process name), or `ps`-derived pids; launch servers without an inline kill of their own pattern.
  No data lost; baseline preserved; P4-A + dashboard healthy.
- WATCH: BuildOutpost ~0 and soldierBins concentrated at 0-1 → army-building not emerging despite the
  economy fix. Tentatively confirmed at gen 30 (outposts 0.17/g). → P4/P3 escalation likely needed.
- LESSON (tooling): `pkill -f "cnn_train.*foundation1"` SELF-MATCHES the running shell (its argv contains
  the pattern) → SIGTERM kills the shell (exit 144) before the relaunch. Kill by PID / `pgrep -af
  release/cnn_train` instead. Caused one messy restart (recovered, no data lost — old run preserved).
