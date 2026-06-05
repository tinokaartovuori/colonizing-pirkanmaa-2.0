# Neural-network opponents for *Colonizing Pirkanmaa*

This document specifies the three **neural-network-driven CPU opponents**
(`nn-easy`, `nn-medium`, `nn-hard`) added alongside the existing three
heuristic CPUs (`easy`, `medium`, `hard`). It is the source-of-truth for the
design; the code lives in `src/ai/nn/` (ships to the browser) and `training/`
(headless self-play training, run with `vite-node`).

> **Hard constraint.** The existing game logic and the existing heuristic
> `AiController` are **not modified**. The neural AI is purely additive: new
> files plus the minimal wiring needed to make the new opponents *selectable*
> (one extra union member in `Difficulty`, three extra dropdown options, and a
> per-player controller factory in `main.ts`).

## Goals

- **Hard** — should beat a good human; objectively, it must beat the strongest
  existing opponent (the `hard` heuristic) by a clear margin across map sizes
  and game lengths.
- **Medium** — a good human beats it, but only just.
- **Easy** — naturally easy to beat.
- **Runs in the client.** Inference is a tiny dense-MLP forward pass in pure
  TypeScript with **zero runtime dependencies**; trained weights are baked into
  `src/ai/nn/weights.ts` as plain number arrays.
- **Generalises** to large maps and very long games — guaranteed by a
  **board-size-invariant** feature representation (global aggregates, all
  normalised, no per-cell grid) and a training **curriculum** spanning small→
  large maps, 2–4 players, and long round caps.
- **The objective is winning** (70 % tile domination, or last player standing).

## Why neuroevolution, not deep RL

The reward is sparse and long-horizon (win/lose after dozens of turns), the
action space is structured, and we need identical maths for *training* and
*client inference* with no ML runtime. **Evolution Strategies (ES)** over a
compact policy network fits all three:

- No autodiff, no framework — the forward pass *is* the trained artifact, and
  it is trivially portable to the browser.
- Robust to sparse terminal rewards; no credit-assignment machinery.
- Self-play + a Hall-of-Fame league produces strong, non-cyclic strategies.

The policy chooses among **legal, non-suicidal macro-actions** (the same
repertoire a human/heuristic uses), so the network learns *strategy and timing*
rather than re-learning legality. A safety layer makes bankruptcy/illegal
spends unreachable, which both stabilises learning and preserves the engine's
"a CPU never bankrupts itself / never crashes" invariant.

## Inference architecture (`src/ai/nn/`)

| file | role |
|------|------|
| `mlp.ts` | Dense MLP (tanh hidden, linear output), `forward()`, flat (de)serialisation. Pure, deterministic. |
| `features.ts` | `globalFeatures(player, om, pm, round)` → fixed-length normalised vector describing *my* economy/military, the board, and the strongest opponent. Board-size-invariant. |
| `candidates.ts` | Enumerate currently-legal macro-action **intents**, each with a small local-feature vector and a realised executor closure. |
| `safety.ts` | Reserve / projected-net / resource-floor guards. Filters the candidate list so no choice can bankrupt or be illegal. |
| `policy.ts` | `score(candidate) = MLP(concat(global, intent-onehot, local))`; argmax (hard) or temperature-sampled (weaker tiers). |
| `controller.ts` | `NeuralAiController` — `placeHeadquarters` + `planTurn()` generator. Loops *enumerate → score → execute (`eh.aiBuild/aiBuy/aiMove`) → yield* until `Pass` or the action budget is spent. |
| `weights.ts` | Generated artifact: arch + per-tier `{ weights, config }`. |

### Macro-action intents

`StaffBuilding`, `EnsureWoodHarvester`, `BuildFarm`, `BuildMine`,
`BuildVillage`, `BuildOutpost`, `BuildHydro`, `BuildNuclear`, `Expand`,
`HireSoldier`, `Attack`, `StackProducer`, `Pass`.

Targets (which tile) are chosen by cheap, strong heuristics (best grassland,
weakest takeable enemy tile, highest-value safe neutral, …); the **network
decides which intent fires and when to stop** (`Pass`). This keeps the action
dimension fixed and board-size-independent.

### Difficulty tiers

A single trained network powers all three; tiers differ only in a `TierConfig`:

- **hard** — full action budget, greedy (temperature 0), no handicap.
- **medium** — reduced budget, softmax temperature > 0 (occasional sub-optimal
  intent), small economic handicap, calibrated to ~50 % vs the `hard` heuristic.
- **easy** — small budget, high temperature, larger handicap (skips some
  economy/military intents), calibrated well below the `medium` heuristic.

Tiers are **calibrated**, not guessed: `training/calibrate.ts` measures
win-rates vs the heuristic `easy/medium/hard` baselines over many seeds and map
sizes and picks configs that satisfy the monotonicity + target-strength goals.

## Training (`training/`, run with `vite-node` — no new deps)

| file | role |
|------|------|
| `harness.ts` | Headless full-game runner (extracted from the test setup) accepting an arbitrary per-player controller factory; plays to terminal state; returns winner + per-player metrics. |
| `evaluate.ts` | Fitness = win (primary) + end-of-game dominance margin + solvency + speed bonus, averaged over a curriculum batch; opponents = `hard` heuristic + sampled Hall-of-Fame genomes. |
| `evolve.ts` | ES loop: population → evaluate → truncation-select elites → Gaussian-mutate offspring (annealed σ) → update Hall of Fame → checkpoint best genome JSON each generation. |
| `calibrate.ts` | Sets the three `TierConfig`s and emits `src/ai/nn/weights.ts`. |

**Curriculum** (board-size & length generalisation): each evaluation batch
samples map sizes from ~10×10 up to 25×15, 2–4 players, varied seeds, and a
high round cap so the policy is exercised in deep late-game states with full
maps. Fitness explicitly rewards *closing out* games (domination), so the
policy keeps a winning plan deep into long matches rather than stalling.

## Verification

`tests/nnai.test.ts` (headless, deterministic):
1. Weights load with the right shape; `nn-hard` plays full games on **small and
   large** maps without throwing and without ever going resource-negative.
2. The difficulty tiers are ordered by design (budget desc, blunder asc) and the
   behavioural ladder holds (`nn-hard` wins more than `nn-easy` vs a baseline).

`npm run build` (`tsc --noEmit` + vite) and `npm test` stay green.

## Measured results (honest)

Trained via three self-play neuroevolution phases (≈16+ generations each, island
model). Win-rates are over a fixed battery (3 map sizes × 30 seeds, neural at
seat 0, which is the slightly-favoured seat):

| matchup | nn-hard win-rate |
|---|---|
| nn-hard vs **heuristic-hard** | ~25 % |
| nn-medium vs heuristic-hard | ~14 % |
| nn-easy vs heuristic-hard | ~10 % |

The difficulty **ordering is clean and calibrated**, and the neural opponents are
fully playable (solvent, crash-free, client-runnable). **However, the trained
network does NOT surpass the existing `hard` heuristic** — it plays at roughly
the heuristic easy/medium level. The original goal ("hard nearly unbeatable") is
**not met by the neural net**; the strongest CPU in the game remains the
untouched `hard` heuristic.

### Why — the "closing" barrier

Winning requires **70 % tile domination** or **eliminating** every opponent.
Elimination means conquering the enemy HQ, but a tile holds at most 3 units and
conquest needs *attackers > defenders*, so a **3-soldier-garrisoned HQ is
impregnable**. Reaching 70 % demands relentless, coordinated expansion +
military. The macro-action policy (which reuses the heuristic's target-selection
and so is bounded near the heuristic's own play) plus Evolution-Strategies
training within the available compute plateaus *below* the hand-tuned heuristic:
diagnostics show the net stops expanding around 15–26 % of the map and times out
even against a passive opponent. The heuristic itself only closes ~55 % of
games, so this is partly a property of the game.

### Path to a genuinely "nearly unbeatable" agent

- **Search at decision time** (the high-value lever): a shallow look-ahead that
  simulates candidate turns and picks the best would likely overtake the
  reactive heuristic. Blocked here only because the engine has no cheap
  state-clone (snapshot/restore regenerates the whole world); adding a
  lightweight deep-clone would unlock 1–2 ply search / MCTS.
- **A finer action space** (which tile, how many soldiers, micro-timing) instead
  of heuristic-chosen targets, with a much larger / longer training budget
  (AlphaZero-style policy+value + self-play).
- **More generations**: the pipeline supports `--resume`; longer runs help, but
  the plateau suggests architecture/search is the real lever, not just time.
