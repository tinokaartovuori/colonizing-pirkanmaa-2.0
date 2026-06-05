# Search-based AI upgrade — MCTS / AlphaZero (staged design)

_Approved, staged plan. The deterministic, cheaply-cloneable `cp-sim` forward model IS the enabler.
Honest target: Stage A ~40-60% vs hard, +Stage B ~70-90%, 95% a stretch (set ~80% as "success")._

## Verdict / feasibility
- `Game` (`cp-sim/src/managers.rs:81`) is a pure index-arena of `#[derive(Clone)]` types, NO `Rc`/`RefCell`.
  Adding `#[derive(Clone)]` to `Game` makes branching/rollout possible; clone ≈ single-digit µs (parity-neutral).
- Real per-rollout hot path = `get_tile_at` O(n) scan (`managers.rs:212`) and `end_turn` (BFS connectivity + conquest).
  Clone is NOT the bottleneck. Optional later optimization: a coord→TileId index (parity-neutral, speeds normal play too).
- Branching factor per decision ≈ 6-12 candidates (Build×6 + Expand≤6 + Attack≤4 + Hire + Stack + Pass). Turn = a
  short sequence of ~budget candidate-decisions.

## STAGE A — test-time MCTS (NO retraining; do this first)
- **Node granularity = one candidate** (one `plan_turn` loop iteration), value backed up per turn. (Whole-turn nodes
  explode to (6-12)^budget.) Safety scaffold stays a deterministic transition between candidate nodes.
- **Multi-player = determinized single-agent:** only the root player's candidate choices branch; opponent turns are
  forced deterministic transitions (run their policy via `plan_turn` with search off). No max-n/paranoid expansion
  (opponents are deterministic at temperature 0). Value from root player's perspective.
- **New module `cp-ai/src/search.rs`:** arena of nodes; state re-derived by cloning root `Game` + replaying edge
  `Action`s down the path (don't store a Game per node initially). PUCT:
  `a* = argmax Q(s,a) + c_puct·P(s,a)·sqrt(N(s))/(1+N(s,a))`, `Q=W/N` (0 if unvisited), `c_puct≈1.25-2.5`.
  - **Prior P:** softmax over existing `policy::score_candidate(genome,&gvec,c)` (the quantity argmax already uses → free).
  - **Leaf eval (Stage A):** short rollout (horizon ~8-12 turns) with current net argmax for all seats via
    `plan_turn`+`end_turn`, then score with existing metrics (`run.rs` telemetry: tile-frac lead / net_money /
    total_wealth / soldier lead) mapped to [-1,1]; exact ±1/0 if rollout hits Win/Tie (`managers.rs:1006`).
  - **Budget:** N≈100-400 sims/decision, then pick most-visited edge.
- **Integration:** add `SearchConfig` to controller (or `MctsController` wrapper); when enabled, replace the
  `policy::select_index` call in `plan_turn` (`controller.rs:143`) with `search::select(...)`. Everything else
  (scaffold, execute, retry, re-staff) identical. Rollouts/opponent turns recurse into `plan_turn` with search OFF.
  Search OFF in the parity/golden path → **parity gate byte-identical, untouched.**
- **Verify (Rust self-play A/B):** new bin `cp-train/src/bin/bench.rs` — `MctsController(champion)` seat0 vs
  `NeuralAiController(champion)` seat1 over the `game_spec` curriculum → head-to-head win-rate. Success = search
  ≫ 50% (e.g. >65%). The hard heuristic is NOT in Rust, so "vs hard" still needs TS `benchmark.ts` (or port hard to Rust).
- Effort: SMALL-MEDIUM (~days). `derive(Clone)` + search.rs (~300-400 LOC) + controller hook + bench bin.

## STAGE B — AlphaZero training (high ceiling, big compute; AFTER Stage A proves out)
- **Value: a SEPARATE value net** (input = 36-dim global features, output tanh∈[-1,1]) — keeps the POLICY net
  byte-identical → parity/golden/weights untouched. (Two-head shared trunk would break parity; not worth it.)
- Self-play with MCTS drives data: policy target = visit-count distribution π=N(s,a)/ΣN; value target = game outcome z.
- Training: **B1 (recommended first)** = keep GA (`cp-train`), fitness = win-rate of MCTS-wrapped genome + value net
  for leaf eval (drop rollouts → faster). **B2 (optional, highest ceiling)** = gradient policy distillation
  (cross-entropy softmax(scores) vs π) — new backprop stack for the ~2k-param net.
- **Throughput collapse is the headline risk:** search in-loop is ~50-300× slower → 280 g/s → ~1-6 g/s → a GA gen goes
  from seconds to minutes; full runs are days-to-week. Mitigate: value-net leaf eval (no rollouts), small N annealed up,
  smaller maps, prior-softmax caching, coord index, more cores.

## Shipping (deployed game is TS, zero runtime search)
- **Distill search into the policy net** (Stage B2 imitates MCTS visit distributions) → shipped net plays better with
  plain argmax, no runtime search, existing `emit-weights.ts`→`weights.ts` path untouched. RECOMMENDED.
- TS-side MCTS only if a search-powered "insane" difficulty is wanted (TS model is Rc/RefCell shared-mutable → cloning
  is hard/slow; high divergence risk). WASM-ship Rust search = overkill unless runtime search is a product need.

## Immediate next step (smallest viable, parity-safe, reversible)
1. `#[derive(Clone)]` on `Game` (`managers.rs:81`).
2. `cp-ai/src/search.rs`: per-candidate PUCT MCTS, priors from `score_candidate`, short-rollout metric leaf eval.
   Hook into `plan_turn` behind a flag; no-search path byte-identical.
3. `cp-train/src/bin/bench.rs`: MCTS(champion) vs no-search(champion) over curriculum → win-rate.
4. Gate: if search >~65% vs no-search → proceed to port hard-AI to Rust (or TS-bench) + plan Stage B. If marginal →
   fix leaf eval / priors before spending Stage B compute.
