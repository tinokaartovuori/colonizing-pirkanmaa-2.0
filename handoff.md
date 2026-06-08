# Handoff — strength ceiling reached + champion deployed (Colonizing Pirkanmaa AI)

_Last updated 2026-06-08, end of the "deploy + device + ceiling" session. **Read this first.**
Supersedes the prior "DAgger" handoff._

> Deployed game = TS/Phaser (`src/`). The **Rust trainer** (`rust-trainer/`) trains the CNN net
> that is redeployed into the TS game. Parity Rust⇄TS is bit-exact, locked by
> `cargo run -p cp-train --bin parity --release` (must be **8/8**). Arc is now **sd5**.
> Branch: `dagger-passcollapse-fix` (not yet merged to main).

---

## TL;DR — where we are

**The project goal is essentially met and the strength frontier is mapped.** The champion is
**`models/sd5/az/sd5-az-001`** (= `sd4-az-002` weights = `sd3-az-004` weights — the same PPO net
re-registered across economy arcs; the net architecture is arc-independent).

- **Strength:** trueWin vs HARD **0.65–0.72**, peak army **~1.6** (army gate ≥1.5 cleared),
  villages ~4.8, mines ~2.4/game, metal income ~108. Plays the full
  **wood→mine→expert→outpost→soldier** chain. (Exact number depends on the bench's HARD-league
  strength: ~0.70 vs the old league, ~0.65 vs the re-tuned sharper league. Both honest.)
- **Deployed in-browser:** YES — the CNN champion actually plays in the TS game now (this was the
  big deliverable; before this session the browser shipped only ancient models + a stale MLP).
- **The ~0.70 plateau is a REAL ceiling for the current paradigm** (PPO + this net + this opponent),
  proven three ways this session. Not an anchor artifact, not a capacity ceiling. Beating it needs a
  genuinely new method, not another knob.

**Nothing is currently running.** No training, no agents. Dashboard is up (see Operational).

---

## What this session accomplished (the arc)

1. **Closed the deploy-debt (the headline win).** The 0.70 bench number was previously unshippable.
   Now the champion plays in-browser:
   - `b4c384f` — ported the Rust army-economy **scaffold** into `src/ai/nn/controller.ts`
     (mine builder staffing 2 workers + 1 expert = 80 metal, village/unit-cap builder, expert
     placement, wood accumulation). Fixes the train/serve manifold skew that would otherwise
     Pass-collapse a deployed net.
   - `f17d981` — built **`src/ai/nn/spatial_net.ts`** from scratch (there was NO CNN forward in TS
     before): Conv2d/Dense/GlobalAvgPool/tanh, 27-plane extractor, `scoreCandidate`,
     `selectSpatialIndex` (greedy). Forward parity Rust⇄TS to **10 decimals** (via a `cnn_fwd_parity`
     bin). Bundled the champion in `src/ai/nn/models_spatial.ts`, routed `model:sd4-az-002` through
     the CNN controller, surfaced as the headline "CPU (Neural Champion)" in `startdialog.ts`.
   - `971528b` — ported the **value head** (`valueFrom` + the 12 value_scalars, parity ≥6 dec) and a
     **spatial PUCT MCTS** (`src/ai/nn/spatial_search.ts`, c_puct=1.5, action=most-visited). Deploy
     runs **sims=32** (sims=64 is ~2.5s/decision in single-threaded JS — too slow; 32 is ~1s and
     beats greedy). Raising to 64 needs a perf refactor — see Open Levers.

2. **sd4 league re-tune** (`64a7904`) — the sd3-tuned cash reserves were over-banked on the cheaper
   sd4 economy; cut them. STRONG_ARMY yardstick 2/4→3/4 PASS (vs-HARD 49→61%). This sharpened the
   bench opponents, which is why the honest champion number reads ~0.65 vs ~0.70.

3. **Strange Device rebalance → arc sd4→sd5** (`fde4c25`). Diagnosed (from source) that the Device
   was **balance-dominated**: device tile held **0 defenders** (1 raider cracks it), owning it
   **halved** the soldier cap, and it won ~50 rounds later than conquest. The rebalance made it a
   real choice: **1 defender allowed** (crack needs 2), **cap penalty ÷2 → −2**, **countdown
   18→12 / 0.12→0.10** (win ~round 45–50). Build cost 1300 unchanged. Parity-affecting; goldens
   re-exported, parity 8/8. **Also fixed a latent pre-existing parity break** (the scaffold port had
   added `ensureMetalIncome` to TS `planTurn` but not Rust `plan_turn` → stale goldens).

4. **Device-as-win-path: tested and CONCLUDED.** Even with the viable sd5 device + a *fixed*
   reward-relevant incentive (Lever-C `device-credit` was a **silent no-op in `--ppo` mode** — only
   AZ had it; fixed in `697ebe2`) + 33% device-rush opponents, the AI's device-win share plateaued
   ~5% and a device-trained net benched **worse** (0.60 vs champion 0.65, army 1.2 vs 1.6). Verdict:
   **the device is viable-but-inferior; conquest genuinely dominates this map.** The AI correctly
   treats the device as a minority/situational option. This is correct play, not a bug. (Memory:
   `device-win-lever.md`.)

5. **Plateau diagnosed as a REAL ceiling** (memory `anchor-not-capacity.md`):
   - A decision agent argued the plateau was an **anchor/auto-revert artifact** (every PPO run was
     KL-tethered to the champion + an unconditional auto-revert that halves lr on any dip).
   - Tested directly: added `--ppo-no-revert` (`e51f70d`), ran free exploration (anchor 0, no
     revert, entropy 0.03). Result: trueWin **monotonically DECLINED 66→49** with lr healthy. So the
     anchor was a **stabilizer holding the net at its peak**, NOT a cap below a higher one.
   - Therefore: **NOT an anchor artifact, NOT a capacity ceiling** (the net represents the good
     policy fine — it IS the champion; free optimization just can't *hold* it). It's a
     PPO-optimization/opponent ceiling. **Capacity-scaling is NO-GO** (a bigger net hits the same
     instability + huge parity/retrain cost).

6. **Dashboard fixed + verified.** A user-facing PPO instrumentation gap: `--ppo` runs wrote a
   reduced metric set (no per-game economy fields, no replay files), so the dashboard's economy panel
   showed 0 mines/experts and the replay viewer was empty — even though the AI builds them.
   `865bd9f` refactored the AZ bench-row + replay writers into shared fns called by BOTH `--train`
   and `--ppo`, so future PPO runs emit the full 61-key metric set + 11 replay files. (A verify agent
   confirmed the dashboard renders its source correctly — no display bug.)

---

## Hard-won findings (don't re-derive these)

- **PPO+GAE broke the project-long ~0.55 AZ-MCTS plateau** (earlier arc) → ~0.66, then the sd4
  economy rebalance → 0.70. PPO is the proven paradigm. `--ppo` mode in `cnn_train.rs`, spec in
  `rust-trainer/PPO-SPEC.md`.
- **Raw-strength PPO is EXHAUSTED at ~0.65–0.72.** League-PPO, device-PPO, and free-exploration all
  cap/decline here. KL-anchor + auto-revert are load-bearing for *stability*, not crutches.
- **`--device-potential` is Ng-1999 potential shaping = provably policy-INVARIANT** — it cannot make
  a dominated strategy chosen. Use the **non-potential** Lever-C `--device-credit`/`--device-crack-credit`
  if you ever push the device again (but the device is concluded inferior).
- **Bench variance is large:** 80-game ±0.1, 160-game ±0.06. **Use ≥160-game fixed-seed benches for
  champion selection** — noisy 80-game selection has repeatedly picked lucky-but-worse nets.
- **`--validate-net` is the honest policy-strength bench** (net-greedy/MCTS over candidates).
  sims=1 MCTS is net-INDEPENDENT (always candidate 0) — never trust it.
- The device is viable-but-inferior; **bridges are fully covered** (BuildBridge intent + planes; the
  champion builds ~1/game — no lever there).

---

## Operational notes (READ before touching the trainer)

- **SIGBUS hazard:** `cargo build` overwrites `target/release/cnn_train` and **kills any live run**
  using it (replaced-ELF SIGBUS). If a run is live: build with `CARGO_TARGET_DIR=target-agent`, OR
  launch runs from a **copied binary** (`cp target/release/cnn_train /tmp/cnn_train-X`). All this
  session's runs used copied binaries. Right now nothing is live, so builds are safe.
- **Leave ~4 cores free:** the trainer hardcodes 16 threads (20-core box). One ~16-thread run at a
  time.
- **Parity gate:** after ANY change near the sim/candidate/planes/spatial-forward, run
  `cargo run -p cp-train --bin parity --release` → must be **8/8**. Parity-FREE (safe to edit
  without golden re-export): `cnn_train.rs`, `hard_ai.rs`, `controller.rs/.ts`, `spatial_search.ts`,
  `spatial_net.ts` (its *forward* is parity-locked; added backward/value fns are free).
  Parity-AFFECTING (need Rust⇄TS mirror + `npx vite-node training/export-golden.ts` + 8/8 +
  arc bump): `resources.*`, `model.*`, `managers.*`, `candidates.*`, `planes.rs`, spatial forward.
- **Dashboard:** `setsid npx vite-node training/serve-dashboard.ts -- --dir <run-dir> --port 5199`
  (the `for p in $(pgrep…); kill` one-liner trips exit 144 under the harness — kill and start in
  separate calls; `setsid … < /dev/null &` works). Currently serving
  `rust-trainer/checkpoints-cnn-champ-view` (a populated champion bench: economy panel + 11 replays).
- **Known pre-existing test failures** (NOT from this session — confirm via `git stash`): vitest
  `nnai` "structure" arch mismatch, `restore` seed-19, `spatial-mcts-strength`; cargo
  `spatial_net::forward_backward_equivalence_golden`. Don't chase these.

---

## Open levers / next steps (in rough EV order)

1. **Set the model pointers + merge.** `models/CHAMPION.json` is empty `{}` — the champion/deploy
   pointers were never written. Set them to `sd5-az-001` (`npm run models -- promote …` / per
   `models/README.md`). Then consider merging `dagger-passcollapse-fix` → `main` (16 commits;
   deploy + sd5 + dashboard fix all live there).
2. **Deploy at full strength (bounded, zero training risk):** the in-browser MCTS runs sims=32
   because each sim re-rolls the opponent's full HARD turns (`advanceAfterRoot`) for the whole path
   on every expansion. **Cache that rollout per node** → sims=64 becomes latency-viable → the
   *shipped* AI matches the 0.70 bench. (A measurement agent stalled on the slow in-engine bench but
   pinpointed this bottleneck. First confirm 64>32 strength, then optimize.)
3. **To beat 0.72 you need a NEW method, not a knob.** Candidates, all uncertain:
   - AZ-style **MCTS policy-improvement** training now that the value head works (caveat: AZ-MCTS
     historically capped *lower* than PPO ~0.55 — the value head being good is the new variable).
   - A different opponent/curriculum, or accept ~0.70 vs a strong scripted bot as a fine ceiling.
   - Capacity-scaling is **NO-GO** unless free-exploration is re-run and the net *caps* (not
     declines) ≤0.70 — this session it *declined*, which implicates optimization stability, not size.
4. **Re-tune the league for sd5** if you train again — the sd4 reserve re-tune is mostly fine
   (economy unchanged by sd5) but `DEVICE_RUSH_PARAMS` could be sharpened for the new device rules.

---

## Key files & commits (this session, branch `dagger-passcollapse-fix`)

- Deploy: `src/ai/nn/{spatial_net.ts, spatial_search.ts, controller.ts, models_spatial.ts,
  candidates.ts (+target field), index.ts}`, `src/ui/startdialog.ts`. Commits `b4c384f`, `f17d981`,
  `971528b`.
- sd5 rules: `rust-trainer/crates/cp-sim/src/{resources.rs, model.rs, managers.rs}` ⇄
  `src/core/resources.ts`, `src/model/{tile.ts, player.ts}`. Commit `fde4c25`. Arc doc in `CLAUDE.md`
  + `models/README.md`.
- PPO instrumentation/levers: `rust-trainer/crates/cp-train/src/bin/cnn_train.rs` —
  `write_bench_history_row` + `write_replays` shared fns, `--ppo-no-revert`, Lever-C wired into PPO,
  DeviceRush oversampling. Commits `ab869a8`, `697ebe2`, `e51f70d`, `865bd9f`.
- Champion: `models/sd5/az/sd5-az-001/` (commit `55b9f50`). Registry `models/registry.jsonl`.
- Memory (the durable record): `~/.claude/.../memory/` — `champion-deployed.md`, `device-win-lever.md`,
  `anchor-not-capacity.md`, `ppo-broke-plateau.md`, plus the index `MEMORY.md`.

---

## One-command sanity checks for the next session

```bash
cd rust-trainer && cargo run -p cp-train --bin parity --release            # must be 8/8
cd rust-trainer && cp target/release/cnn_train /tmp/cnn_train-chk && \
  /tmp/cnn_train-chk --validate-net --init ../models/sd5/az/sd5-az-001/weights.json \
  --games 80 --seed 4242 --sims 64                                          # ~0.65 trueWin, army ~1.6
npm run build                                                               # tsc + vite, champion bundles in
```
