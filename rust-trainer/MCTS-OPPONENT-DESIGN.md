# In-browser MCTS opponent + difficulty tiers — design (approved)

_Ship MCTS as the in-game (TS) opponent with hard/medium/easy tiers. Recommendation: **Option A, TS-native MCTS** (mirror `rust-trainer/crates/cp-ai/src/search.rs`), NOT WASM._

## Why A not B
WASM's killer is the state-transfer boundary: running the Rust search needs a `cp_sim::Game` byte-faithful to the live mid-turn TS state every move — `cp-sim` has no "import arbitrary mid-game snapshot" entry, so B requires a NEW faithful snapshot→Game importer in Rust + build pipeline = MORE divergence. A is a near-mechanical port: every Rust call (`enumerate`, `scoreCandidate`, `execute()`, `staffIncome`, `endTurn`, `globalFeatures`, `mlp.forward`) already has a parity-designed TS twin. Turn-based latency tolerance removes B's only advantage (raw speed).

## Save/restore (branching) — verdict
`src/managers/persistence.ts buildSnapshot` + `gameeventhandler.ts restoreSnapshot` are lossless for what matters (proven by `tests/restore.test.ts`: per-player resources/workers/staffed-farms/tiles after 24 rounds, no self-collapse). NOT cheap enough to clone per MCTS node. So mirror Rust: build ONE headless sandbox engine per search (like restore.test.ts's StubScene+stub-menu setup), and **replay edge actions** down each simulation (Rust `search.rs replay`), not per-node snapshots.

## Files (Option A)
NEW:
- `src/ai/nn/sandbox.ts` — `createSandbox(snapshot, settings)`: headless engine (ObjectManager/PlayerManager/GameEventHandler + StubScene + stub menu + WorldGenerator.generateMap + eh.restoreSnapshot), root player = current. Extract StubScene/stub-menu from restore.test.ts into a shared non-test module.
- `src/ai/nn/value.ts` — `ValueNet{arch,params}` + `valueForward(net,gvec)=tanh(forward)` (reuse mlp.ts; mirror value.rs).
- `src/ai/nn/search.ts` — 1:1 port of search.rs: `softmax`, `puctSelect`, `Node`, `makeNode` (enumerate→score→softmax priors; terminal=all-Pass), `replay`, `metricValue` (4-lead blend from metrics.ts), `evaluateLeaf{static|value|rollout}`, `advanceRoundAfterRootTurn` (endTurn → each non-root seat planTurn(search off)+endTurn → until current==root), `simulate`, `select(...)` → most-visited root edge + tier blunder/temperature on the FINAL choice (mirror policy.ts select). Port XorShift32.
MODIFY:
- `controller.ts planTurn`: optional `search?:SearchConfig`/`valueNet?`. When set, snapshot LIVE mid-turn state → createSandbox → search.select → execute chosen index on the LIVE engine. Unset ⇒ byte-identical to today (preserves nn tiers + tests). Scaffold runs on live engine first (Rust captures root after scaffold).
- `index.ts` createNeuralController: pass per-tier SearchConfig + valueNet.
- `weights.ts` + `emit-weights.ts`: add `valueNet?` + `search:{hard,medium,easy}`; emit Rust-trained value.json (static fallback if absent).
- difficulty seam: reuse nn-easy/medium/hard (route to search-enabled tiers), or add additive mcts-* labels in model/player.ts + menu.

CRITICAL parity: sandbox must keep root player as current during search; candidate execute() acts on getCurrentPlayer(). Search must NEVER mutate the live engine (only read chosen index; controller executes on live). Use stubs so endTurn/restoreSnapshot never touch live UI.

## Tiers (reuse TierConfig + new SearchConfig + wall-clock cap)
- hard: HARD_CONFIG; leaf=value (fallback static); high sims (~800-1500, cap ~2500ms); final=argmax visits.
- medium: MEDIUM_CONFIG; leaf=static; ~150-300 sims, cap ~1200ms; final=temp~0.5 over visit counts + blunder 0.05.
- easy: EASY_CONFIG; leaf=static; ~20-40 sims, cap ~500ms; final=temp~1.2 + blunder ~0.25.
Blunder/temperature applied to visit-count choice in search.select (mirror policy.ts:46-74). Medium/easy need NO value net (static) → shippable immediately; value_leaf falls back to metricValue when valueNet absent.

## Per-move time & responsiveness
Leaf cost dominates: value/static ≈ one forward (µs) → thousands of sims/move feasible in JS; rollout ≈ tens-hundreds only (too slow in JS). Default hard→value. Enforce `timeBudgetMs` cap in select (break loop on performance.now). runCpuTurn (main.ts) steps the generator with setTimeout between actions → run MCTS for one decision synchronously within the cap; Web Worker only if janky.

## Verify
1. TS-vs-Rust parity unit test on a FIXED static-leaf scenario (deterministic) → chosen index matches Rust search::select.
2. Strength: TS bench (mirror training/benchmark.ts) NN-MCTS vs TS-hard AiController → confirm ~20-33% band; bench_hard is the Rust reference.
3. Playwright in-game smoke (nn-hard = MCTS): CPU turns complete within cap, no errors, game progresses; measure per-move latency.
4. No-mutation-leak test: live state byte-identical after a search.

## Open questions to verify FIRST
(a) conquering-unit conquest is resolved per-turn from presence (no hidden multi-turn counter) so snapshot/restore is lossless for sieges — read conquerTile/endTurn; add deep-siege restore test if needed.
(b) all IMenuObjectManager/IGameScene methods used by endTurn/restoreSnapshot are safely stubbable (e.g. setWinMenu).

## Risks
JS perf ceiling (mitigate: value/static leaf + time cap); sandbox/UI entanglement (stubs); save/restore completeness for sieges; TWO search impls to keep in sync (parity test in CI; document search.ts mirrors search.rs); value-net emission (ship static first); opponent model inside rollouts (model opponents as NeuralAiController search-off or hard — document).
