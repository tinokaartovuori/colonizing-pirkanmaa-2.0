# Archive — pre-Strange-Device era (2026-06-03)

Everything here was produced for the **old game version** (no Strange Device).
The game is being changed (a new "Strange Device" win condition, see
`/STRANGE-DEVICE-DESIGN.md`), which makes all trained models below **obsolete** —
they were trained on a game whose dynamics no longer apply. Archived, not deleted,
for historical reference and in case any data is useful later.

**The engine itself (Rust `cp-sim` / `cp-ai` / `cp-train`, the TS game in `src/`,
the design/architecture docs in `rust-trainer/*.md`) was NOT archived — it is the
reusable foundation for the new arc.** Only trained weights, run logs, the old
status doc, and old experiment scripts moved here.

## What's here

- `models/` — all old trained checkpoints + their logs/benchmarks/run.out:
  - `checkpoints/` — the original neuroevolution (GA) champions (3.5 MB; the live
    game still ships one of these via `src/ai/nn/weights.ts`).
  - `checkpoints-az/` — **`champion.json` = "exp-A"**, the best old model: ~33% vs
    the heuristic hard bot @ 12×12, ~31% @ 14×12. This was the ceiling everything
    kept hitting.
  - `checkpoints-az2 … az9/` — earlier AlphaZero (AZ) experiments.
  - `checkpoints-az4/value.json` — the 41-dim spatial value net used to warm-start
    later runs.
  - `checkpoints-az10/11/12/` — spatial-policy (Exp I) runs that drifted.
  - `checkpoints-az13/` — the timeout-penalty + KL-anchor run (drift arrested but
    plateaued at exp-A level; `champion-best.json` saved).
  - `checkpoints-az14-noklctrl/` — timeout-penalty alone (no anchor), drifted down.
  - `champion-v57.json.bak` — an old TS-era champion backup.
- `handoff-spatial-arc.md` — the old "read this first" status doc for the
  spatial-policy arc (goal: beat hard 70%). **Superseded** by the new `/handoff.md`.
- `scripts/launch-next-az.sh` — launcher for the old AZ follow-up experiments.

## Key findings carried forward (why the arc ended + why the game is changing)

Measured 2026-06-03 (the empirical case for the Strange Device redesign):

1. **The game is structurally draw-prone.** Hard-vs-hard: **41.5% of games
   unresolved at cap 120, still 35.7% unresolved at cap 3000** (25× longer). Two
   competent bots settle at ~49%/49% with ~36% tiles left neutral. Permanent
   stalemates. → **"beat hard 70% of ALL games" is mathematically impossible**
   (~39% are unwinnable stalemates); realistic ceiling ~55%.
2. **More MCTS sims HURT** (exp-A: 31.7% @ 96 sims → 23.3% @ 1400). The binding
   constraint was never search depth.
3. **The AI never builds an army** (BuildOutpost ~0.1%, soldiers ~0–1.2 at every
   sim level). It wins only when the enemy self-collapses; it loses whenever the
   enemy fields even ~2.7 soldiers. The long-horizon credit problem (army buildup
   → assault → cut) is unreachable under sparse reward + shallow search.
4. **exp-A (best neural AI) is WEAKER than the heuristic hard bot** (head-to-head
   29% win / 41% loss). The neural approach never surpassed the hand-coded bot.
5. The two training pathologies (draw-attractor + long-horizon credit) both stem
   from the game being draw-prone → **fixing the GAME (Strange Device) addresses
   both at the source**, which is the new direction.

Full detail in `handoff-spatial-arc.md` and `rust-trainer/*.md` (kept in place).
