# Training the neural CPU opponents

Self-play **neuroevolution** of the policy network that drives the `nn-easy` /
`nn-medium` / `nn-hard` opponents. Runs headlessly on the real game engine via
`vite-node` (no extra dependencies). See `docs/ai-neural.md` for the design.

## Pipeline

```bash
# 1. Train N island runs in parallel (one CPU core each). Checkpoints land in
#    training/checkpoints/island-<i>/best.json after every generation.
npm run ai:train -- 12 50 14 12        # islands gens pop games

# 2. Pick the champion across islands (battery vs the hard heuristic).
npm run ai:tournament                  # -> training/checkpoints/champion.json

# 3. Calibrate the three difficulty tiers and bake src/ai/nn/weights.ts.
npm run ai:calibrate                   # reads champion.json by default

# 4. Verify.
npm test                               # strength tests activate once weights are trained
npm run build
```

Each step is deterministic (fixed seeds). `evolve.ts` accepts `--resume` to
continue an interrupted island from its `best.json`.

## How it works

- `harness.ts` — controller-agnostic headless match runner; reads outcomes from
  the game's own win/tie/loss logic (never reimplemented). `sim/` is untouched.
- `factories.ts` — build a seat as either the heuristic `AiController` or a
  `NeuralAiController` from a genome.
- `evaluate.ts` — curriculum (varied map sizes, player counts, seeds, round
  caps) + fitness (win, dominance margin, speed; crushes bankruptcy/crash).
- `evolve.ts` — Evolution Strategies: elitism + annealed Gaussian mutation +
  Hall-of-Fame self-play. Checkpoints best genome + per-gen log each generation.
- `tournament.ts` — robust cross-island selection of the champion.
- `calibrate.ts` / `emit-weights.ts` — set tier knobs to hit the difficulty
  targets and serialise `src/ai/nn/weights.ts` (the dependency-free client
  artifact).
