// NeuralAiController — a drop-in sibling of the heuristic AiController, driven
// by a trained policy network. It is purely additive: the existing AiController
// is untouched, and this class exposes the same surface (placeHeadquarters,
// planTurn generator, playTurn) so main.ts can route nn-* players to it.
//
// A turn has two parts:
//   1. survive() — a deterministic safety scaffold that guarantees solvency
//      (wood income to cover upkeep; staff income buildings, relocating workers
//      when capped). This is the same machinery the heuristic relies on, so a
//      neural CPU inherits the "never bankrupts, never crashes" invariant.
//   2. an NN decision loop — among the currently-legal, currently-affordable
//      discretionary intents (build / expand / militarise / stack / pass), the
//      network scores each and one is chosen, until it elects to Pass or the
//      action budget runs out. THIS is where strategy is learned.

import { TileBase } from '../../model/tile';
import { Forest, AbundantForest, Mountain, Grassland } from '../../model/tiles';
import { UnitBase } from '../../model/unit';
import { PlayerBase } from '../../model/player';
import {
  BASIC_WORKER_COST, EXPERT_COST, MINE_BUILD_COST, VILLAGE_BUILD_COST, ResourceMap,
} from '../../core/resources';
import type { GameEventHandler } from '../../managers/gameeventhandler';
import type { ObjectManager } from '../../managers/objectmanager';
import type { PlayerManager } from '../../managers/playermanager';
import { Genome } from './mlp';
import { globalFeatures } from './features';
import { enumerate, AiCtx, TierConfig, Intent, Candidate } from './candidates';
import { select, scoreCandidate } from './policy';
import { SearchConfig, select as searchSelect } from './search';
import { ValueNet } from './value';
import { SpatialNetTS, selectSpatialIndex } from './spatial_net';
import { SpatialSearchConfig, selectSpatialMcts } from './spatial_search';
import { buildSnapshot } from '../../managers/persistence';
import { MctsWorkerClient } from './mcts-worker-client';
import * as M from './metrics';
import * as S from './safety';

/**
 * Optional MCTS wiring for a controller. When present, the discretionary
 * decision loop runs a test-time PUCT search (search.ts) instead of the direct
 * policy argmax/softmax: the LIVE mid-turn state is snapshotted, a headless
 * sandbox is built from it, the search picks the best candidate INDEX, and that
 * candidate is executed on the LIVE engine. The search never mutates the live
 * engine. `mapInfo` supplies the seed/dimensions the sandbox needs to regenerate
 * the deterministic terrain.
 */
export interface SearchWiring {
  config: SearchConfig;
  valueNet: ValueNet | null;
  mapInfo: { width: number; height: number; seed: number };
}

/**
 * Optional spatial-net MCTS wiring. When present (with a `spatialNet`), the
 * discretionary decision loop runs the deploy/bench PUCT search (spatial_search.ts)
 * using the CNN's policy as prior + its value head at the leaves — the FULL-strength
 * deploy mode of the AlphaZero champion (e.g. sd4-az-002), instead of the greedy
 * policy argmax. `mapInfo` supplies the seed/dimensions the sandbox needs to
 * regenerate the deterministic terrain. The army-economy scaffold still runs first.
 */
export interface SpatialSearchWiring {
  config: SpatialSearchConfig;
  mapInfo: { width: number; height: number; seed: number };
}

/**
 * Optional, purely-additive tracing hook for the golden-trace exporter
 * (training/export-golden.ts → rust-trainer parity harness). When omitted
 * (the default everywhere in the game/tests/training) `planTurn` behaves
 * exactly as before — nothing in this hook influences control flow. It is
 * invoked once per discretionary decision in the NN loop, BEFORE the chosen
 * intent is executed, with the exact vectors/candidates/scores the policy saw.
 */
export interface DecisionTrace {
  round: number;
  globalVec: number[];
  candidates: { intent: number; local: number[]; label: string }[];
  scores: number[];
  chosenCandidateIndex: number;
  chosenIntent: number;
}
export type DecisionSink = (d: DecisionTrace) => void;

export class NeuralAiController {
  private budget = 0;
  /**
   * Lazily-created Web Worker client for the spatial MCTS search. Built on the
   * first spatial-search turn (only when `spatialNet` + `spatialSearch` are wired)
   * and reused across every turn/sim. Running the search off the main thread keeps
   * the UI rendering at 60fps while a neural CPU thinks. `false` once we've decided
   * to fall back permanently (worker unavailable / errored); `null` = not yet built.
   */
  private workerClient: MctsWorkerClient | null | false = null;
  /** One-time fallback-warning latch (so the console isn't spammed per turn). */
  private warnedWorkerFallback = false;

  constructor(
    private eh: GameEventHandler,
    private om: ObjectManager,
    private pm: PlayerManager,
    private genome: Genome,
    private cfg: TierConfig,
    /** Seedable RNG for reproducible training; defaults to Math.random in the client. */
    private rand: () => number = Math.random,
    /** Optional test-time MCTS. When omitted, planTurn is byte-identical to today. */
    private search?: SearchWiring,
    /**
     * Optional trained spatial CNN. When present, the discretionary decision loop
     * scores candidates with the CNN's `score_candidate` (board planes + per-tile
     * target embed + global pool) and picks the greedy argmax — the deployed
     * (non-MCTS) policy mode of the AlphaZero champion (e.g. sd4-az-002). The
     * `genome` MLP is then unused for scoring. Takes precedence over `search`.
     */
    private spatialNet?: SpatialNetTS,
    /**
     * Optional spatial-net deploy MCTS. When present (alongside `spatialNet`), the
     * discretionary loop runs PUCT search (policy prior + value head leaves) for
     * the CNN champion's FULL deploy strength instead of the greedy argmax. Omitted
     * ⇒ greedy spatial policy (byte-identical to the prior deploy).
     */
    private spatialSearch?: SpatialSearchWiring,
  ) {}

  // --- first round ----------------------------------------------------------

  /** Choose and claim a starting tile (same strong heuristic as AiController). */
  placeHeadquarters(player: PlayerBase): void {
    // Candidates must be BUILDABLE: unowned AND empty (first-round HQ placement is
    // refused on a tile that already holds a building, e.g. an unowned Mikontalo —
    // picking one left the player with 0 tiles → instant loss). Prefer grassland,
    // then any non-river land, then any tile.
    const empty = (t: TileBase) => t.getOwner() === null && t.getBuilding() === null;
    let candidates = this.om.getTiles().filter((t) => t.getType() === 'Grassland' && empty(t));
    if (candidates.length === 0) candidates = this.om.getTiles().filter((t) => empty(t) && t.getType() !== 'River');
    if (candidates.length === 0) candidates = this.om.getTiles().filter((t) => empty(t));
    if (candidates.length === 0) return;
    let best = candidates[0];
    let bestScore = -Infinity;
    for (const tile of candidates) {
      const ns = tile.getNeighbourTiles();
      const free = ns.filter((n) => n.getOwner() === null).length;
      const forests = ns.filter((n) => n.getType() === 'Forest').length;
      const mountains = ns.filter((n) => n.getType() === 'Mountain').length;
      const grass = ns.filter((n) => n.getType() === 'Grassland').length;
      const distance = Math.min(this.distanceToNearestOwned(tile), 8);
      const score = free * 3 + grass * 2 + forests * 2 + mountains * 3 + distance;
      if (score > bestScore) { bestScore = score; best = tile; }
    }
    this.eh.tileClicked(best);
  }

  private distanceToNearestOwned(tile: TileBase): number {
    let min = Infinity;
    const c = tile.getCoordinate();
    for (const other of this.om.getTiles()) {
      if (other.getOwner() === null) continue;
      const oc = other.getCoordinate();
      const d = Math.abs(oc.x() - c.x()) + Math.abs(oc.y() - c.y());
      if (d < min) min = d;
    }
    return min === Infinity ? 99 : min;
  }

  // --- turn -----------------------------------------------------------------

  /** Synchronous full turn (used by tests / headless training). */
  playTurn(player: PlayerBase): void {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    for (const _ of this.planTurn(player)) { /* drain */ }
  }

  *planTurn(player: PlayerBase, trace?: DecisionSink): Generator<void> {
    this.budget = this.cfg.budget;
    try {
      // 1. Economy scaffold — mirrors the Rust `ensure_income_pub` the champion
      //    net was trained on. Secure WOOD income, staff producers (placing the
      //    Expert + 2nd mine worker that the net learned to act ON TOP OF), expand
      //    the unit CAP (villages) when it blocks full staffing, then GUARANTEE the
      //    first metal source (mine) as a leftover-resource backstop, then re-staff
      //    (mans the new mine). This is the EXACT pre-net state distribution the
      //    champion saw in training — without it the browser CPU plays on an
      //    unseen, under-developed economy (train/serve skew).
      yield* this.ensureWoodIncome(player);
      yield* this.staffIncome(player);
      yield* this.ensureUnitCap(player);
      yield* this.ensureMetalIncome(player);
      yield* this.staffIncome(player);

      // 2. Learned decision loop.
      const ctx: AiCtx = { eh: this.eh, om: this.om, pm: this.pm, player, cfg: this.cfg };
      const round = this.pm.getRoundsPlayed();
      while (this.budget > 0) {
        const gvec = globalFeatures(player, this.om, this.pm, round);
        let cands = enumerate(ctx);
        let choice: ReturnType<typeof select>;
        if (this.spatialNet && this.spatialSearch) {
          // Trained spatial CNN deploy WITH MCTS, run IN-THREAD (the sync path used
          // by tests/training and the worker-unavailable fallback). The browser uses
          // the async `planTurnAsync` which offloads this same search to a Worker.
          const idx = this.resolveSpatialMctsIndexSync(player, cands, gvec);
          choice = cands[idx] ?? cands[cands.length - 1];
        } else if (this.spatialNet) {
          // Trained spatial CNN deploy: greedy argmax of the net's per-candidate
          // score over the LIVE state (board planes + target-tile embed). Mirrors
          // the deployed net-greedy turn loop the champion was benchmarked with.
          let idx: number;
          try {
            idx = selectSpatialIndex(this.spatialNet, player, this.om, this.pm, cands);
          } catch {
            choice = select(this.genome, gvec, cands, this.cfg, this.rand);
            idx = cands.indexOf(choice);
          }
          choice = cands[idx] ?? cands[cands.length - 1];
        } else if (this.search) {
          // Test-time MCTS: snapshot the LIVE mid-turn state, branch in a
          // sandbox, and pick the best candidate INDEX (into THIS enumerate(),
          // same order/state). Execute it on the LIVE engine below. The search
          // never mutates the live engine.
          const snap = buildSnapshot(this.om, this.pm, this.search.mapInfo);
          let idx: number;
          try {
            idx = searchSelect(
              this.genome, snap, player.getPlayerNum(), this.cfg,
              this.search.config, this.search.valueNet, this.rand,
            );
          } catch {
            // On any search failure, fall back to the direct policy choice so a
            // CPU turn never stalls.
            choice = select(this.genome, gvec, cands, this.cfg, this.rand);
            idx = cands.indexOf(choice);
          }
          choice = cands[idx] ?? cands[cands.length - 1];
        } else {
          choice = select(this.genome, gvec, cands, this.cfg, this.rand);
        }
        if (trace) {
          trace({
            round,
            globalVec: gvec,
            candidates: cands.map((c) => ({ intent: c.intent, local: c.local.slice(), label: c.label })),
            scores: cands.map((c) => scoreCandidate(this.genome, gvec, c)),
            chosenCandidateIndex: cands.indexOf(choice),
            chosenIntent: choice.intent,
          });
        }
        if (choice.intent === Intent.Pass) break;
        let ok = false;
        try { ok = choice.execute(); } catch { ok = false; }
        if (!ok) {
          // The selected intent failed (rare race). Retry once with it removed.
          cands = cands.filter((c) => c !== choice);
          if (cands.length <= 1) break;
          choice = select(this.genome, gvec, cands, this.cfg, this.rand);
          if (choice.intent === Intent.Pass) break;
          try { ok = choice.execute(); } catch { ok = false; }
          if (!ok) break;
        }
        this.budget -= 1;
        yield;
        // Realize the obvious follow-up: staff, expand the unit cap if it now
        // blocks staffing, then staff the new slots (mirrors the Rust loop tail).
        yield* this.staffIncome(player);
        yield* this.ensureUnitCap(player);
        yield* this.staffIncome(player);
      }
    } catch {
      /* never let a CPU crash the game */
    }
  }

  /**
   * Resolve the chosen candidate INDEX for the spatial-CNN-with-MCTS branch
   * IN-THREAD (synchronous): run the full PUCT search on a snapshot of the LIVE
   * mid-turn state, falling back to the greedy spatial argmax and finally the MLP
   * policy on any failure so a CPU turn never stalls. Shared by the sync `planTurn`
   * and as the fallback for the async/worker path. The live engine is never mutated.
   */
  private resolveSpatialMctsIndexSync(player: PlayerBase, cands: Candidate[], gvec: number[]): number {
    try {
      const snap = buildSnapshot(this.om, this.pm, this.spatialSearch!.mapInfo);
      return selectSpatialMcts(
        this.spatialNet!, snap, player.getPlayerNum(), this.cfg, this.spatialSearch!.config,
      );
    } catch {
      try {
        return selectSpatialIndex(this.spatialNet!, player, this.om, this.pm, cands);
      } catch {
        const choice = select(this.genome, gvec, cands, this.cfg, this.rand);
        return cands.indexOf(choice);
      }
    }
  }

  /**
   * Run the spatial-CNN MCTS search OFF the main thread via a Web Worker and
   * return the chosen candidate INDEX. The worker runs the SAME `selectSpatialMcts`
   * on the SAME snapshot, so its index aligns with `cands` (enumeration is
   * deterministic for a given state). On ANY problem — Workers unavailable, the
   * worker errors, or it's been disabled — it falls back to the synchronous
   * in-thread search and logs a one-time warning. The live engine is never mutated
   * by the worker (it's READ-ONLY: snapshot → index).
   */
  private async resolveSpatialMctsIndexViaWorker(player: PlayerBase, cands: Candidate[], gvec: number[]): Promise<number> {
    if (this.workerClient === false) return this.resolveSpatialMctsIndexSync(player, cands, gvec);
    try {
      if (this.workerClient === null) {
        this.workerClient = new MctsWorkerClient(
          this.spatialNet!.w, this.cfg, this.spatialSearch!.config,
        );
      }
      const snap = buildSnapshot(this.om, this.pm, this.spatialSearch!.mapInfo);
      return await this.workerClient.searchViaWorker(snap, player.getPlayerNum());
    } catch (e) {
      this.workerClient = false; // disable the worker for the rest of the match
      if (!this.warnedWorkerFallback) {
        this.warnedWorkerFallback = true;
        // eslint-disable-next-line no-console
        console.warn('[neural-ai] MCTS Web Worker unavailable — falling back to in-thread search.', e);
      }
      return this.resolveSpatialMctsIndexSync(player, cands, gvec);
    }
  }

  /**
   * Async twin of `planTurn` for the browser: byte-identical control flow, except
   * the spatial-CNN-with-MCTS branch AWAITS the search in a Web Worker (with the
   * in-thread fallback) so the main thread keeps rendering at 60fps while a neural
   * CPU thinks. The chosen candidate's `execute()` (live-engine mutation + Phaser
   * animations) STAYS on the main thread after the await. main.ts drives this with
   * `await steps.next()`, keeping the `setTimeout(…, CPU_ACTION_MS)` pacing between
   * actions. The sync `planTurn` (above) remains the path for tests/training and
   * for non-spatial-search controllers. The move chosen is identical to `planTurn`
   * (same search, same snapshot, same deterministic enumeration).
   */
  async *planTurnAsync(player: PlayerBase, trace?: DecisionSink): AsyncGenerator<void> {
    this.budget = this.cfg.budget;
    try {
      yield* this.ensureWoodIncome(player);
      yield* this.staffIncome(player);
      yield* this.ensureUnitCap(player);
      yield* this.ensureMetalIncome(player);
      yield* this.staffIncome(player);

      const ctx: AiCtx = { eh: this.eh, om: this.om, pm: this.pm, player, cfg: this.cfg };
      const round = this.pm.getRoundsPlayed();
      while (this.budget > 0) {
        const gvec = globalFeatures(player, this.om, this.pm, round);
        let cands = enumerate(ctx);
        let choice: ReturnType<typeof select>;
        if (this.spatialNet && this.spatialSearch) {
          // The ONLY async difference vs. planTurn: offload the heavy PUCT search to
          // the Web Worker (await), so the main thread keeps animating. Everything
          // after the await — execute() and the scaffold follow-up — runs on the
          // main thread exactly as in planTurn.
          const idx = await this.resolveSpatialMctsIndexViaWorker(player, cands, gvec);
          choice = cands[idx] ?? cands[cands.length - 1];
        } else if (this.spatialNet) {
          let idx: number;
          try {
            idx = selectSpatialIndex(this.spatialNet, player, this.om, this.pm, cands);
          } catch {
            choice = select(this.genome, gvec, cands, this.cfg, this.rand);
            idx = cands.indexOf(choice);
          }
          choice = cands[idx] ?? cands[cands.length - 1];
        } else if (this.search) {
          const snap = buildSnapshot(this.om, this.pm, this.search.mapInfo);
          let idx: number;
          try {
            idx = searchSelect(
              this.genome, snap, player.getPlayerNum(), this.cfg,
              this.search.config, this.search.valueNet, this.rand,
            );
          } catch {
            choice = select(this.genome, gvec, cands, this.cfg, this.rand);
            idx = cands.indexOf(choice);
          }
          choice = cands[idx] ?? cands[cands.length - 1];
        } else {
          choice = select(this.genome, gvec, cands, this.cfg, this.rand);
        }
        if (trace) {
          trace({
            round,
            globalVec: gvec,
            candidates: cands.map((c) => ({ intent: c.intent, local: c.local.slice(), label: c.label })),
            scores: cands.map((c) => scoreCandidate(this.genome, gvec, c)),
            chosenCandidateIndex: cands.indexOf(choice),
            chosenIntent: choice.intent,
          });
        }
        if (choice.intent === Intent.Pass) break;
        let ok = false;
        try { ok = choice.execute(); } catch { ok = false; }
        if (!ok) {
          cands = cands.filter((c) => c !== choice);
          if (cands.length <= 1) break;
          choice = select(this.genome, gvec, cands, this.cfg, this.rand);
          if (choice.intent === Intent.Pass) break;
          try { ok = choice.execute(); } catch { ok = false; }
          if (!ok) break;
        }
        this.budget -= 1;
        yield;
        yield* this.staffIncome(player);
        yield* this.ensureUnitCap(player);
        yield* this.staffIncome(player);
      }
    } catch {
      /* never let a CPU crash the game */
    }
  }

  // --- search support (public scaffold hooks) -------------------------------

  /**
   * Public re-staffing hook used by the MCTS replay (search.ts): after a
   * candidate edge executes on a sandbox, re-staff income buildings exactly like
   * the per-iteration `staffIncome` call in the decision loop. Mirrors the Rust
   * `search.rs staff_after_action` → `staff_income_pub`.
   */
  staffIncomePub(player: PlayerBase): void {
    try {
      for (const _ of this.staffIncome(player)) { /* drain */ }
    } catch {
      /* never crash the search */
    }
  }

  // --- action plumbing ------------------------------------------------------

  private *doAction(fn: () => boolean): Generator<void, boolean, unknown> {
    let ok = false;
    try { ok = fn(); } catch { ok = false; }
    if (ok) yield;
    return ok;
  }

  private addWorker(player: PlayerBase, tile: TileBase): boolean {
    if (player.getFreeUnitAmount() <= 0) return false;
    if (!S.affords(player, BASIC_WORKER_COST, S.STAFF_RESERVE)) return false;
    return this.eh.aiBuyAndPlaceUnit('BasicWorker', tile);
  }

  /**
   * Buy + place an Expert on `tile` keeping at least `reserve` money buffered.
   * Mirrors the Rust `add_expert_reserve`: staffing a producer is MECHANICAL, so
   * it uses INCOME-BUILD affordability (raw resources + a modest money FLOOR of
   * `reserve + ~1 round of drain`) rather than the strategic 5-rounds-of-drain
   * buffer of `affords` — a mine Expert returns METAL not money, so the 5-round
   * buffer it can never earn back permanently blocked experts as the economy grew
   * (the metal-economy starvation root cause). Staffing callers pass the low
   * STAFF_RESERVE so the strategic `cfg.reserve` never starves mechanical staffing.
   */
  private addExpertReserve(player: PlayerBase, tile: TileBase, reserve: number): boolean {
    if (player.getFreeUnitAmount() <= 0) return false;
    if (!tile.hasSpaceForUnits()) return false;
    const floor = reserve + Math.ceil(M.moneyDrainPerRound(player));
    if (!S.affordsIncomeBuild(player, EXPERT_COST, floor)) return false;
    return this.eh.aiBuyAndPlaceUnit('Expert', tile);
  }

  // --- safety scaffold: staffing & wood (ported from the heuristic) ---------

  private findIdleOnPlain(player: PlayerBase): { unit: UnitBase; tile: TileBase } | null {
    for (const tile of M.ownedTiles(player)) {
      if (tile.getBuilding() || tile instanceof Forest || tile instanceof AbundantForest) continue;
      const w = tile.getUnits().find((u) => u.getType() === 'BasicWorker');
      if (w) return { unit: w, tile };
    }
    return null;
  }

  private findSpareWorker(player: PlayerBase, exclude: TileBase): { unit: UnitBase; tile: TileBase } | null {
    for (const tile of M.ownedTiles(player)) {
      if (tile === exclude) continue;
      const type = tile.getBuilding()?.getType();
      if (type === 'Farm' || type === 'Mine' || type === 'Nuclear Power Plant' || type === 'Hydroelectric Power Plant') continue;
      const w = tile.getUnits().find((u) => u.getType() === 'BasicWorker');
      if (w) return { unit: w, tile };
    }
    return null;
  }

  private findExpendableWorker(player: PlayerBase): { unit: UnitBase; tile: TileBase } | null {
    const idle = this.findIdleOnPlain(player);
    if (idle) return idle;
    // surplus producer worker
    for (const tile of M.ownedTiles(player)) {
      const type = tile.getBuilding()?.getType();
      if (type === 'Mine' || type === 'Nuclear Power Plant' || type === 'Hydroelectric Power Plant') {
        const ws = tile.getUnits().filter((u) => u.getType() === 'BasicWorker');
        if (ws.length > 1) return { unit: ws[ws.length - 1], tile };
      }
    }
    const farms = M.ownedTiles(player).filter((t) => t.getBuilding()?.getType() === 'Farm' && M.hasType(t, 'BasicWorker'));
    if (farms.length >= 2) {
      const tile = farms[farms.length - 1];
      const u = tile.getUnits().find((x) => x.getType() === 'BasicWorker');
      if (u) return { unit: u, tile };
    }
    return null;
  }

  /** Guarantee enough staffed forest harvesters to cover wood upkeep. */
  private *ensureWoodIncome(player: PlayerBase): Generator<void> {
    const upkeep = M.woodUpkeep(player);
    if (upkeep <= 0) return;
    const harvesters = () =>
      M.ownedTiles(player).filter((t) => t instanceof Forest && t.getBuilding() === null && M.hasType(t, 'BasicWorker')).length;
    let need = Math.max(1, Math.ceil(upkeep / 40));
    if (M.wood(player) < upkeep * 4) need += 1;
    let guard = 0;
    while (harvesters() < need && guard++ < 8) {
      const f = M.ownedTiles(player).find(
        (t) => t instanceof Forest && t.getBuilding() === null && t.hasSpaceForUnits() && !M.hasType(t, 'BasicWorker'),
      );
      if (!f) break;
      let did = false;
      if (player.getFreeUnitAmount() > 0 && S.affords(player, BASIC_WORKER_COST, S.STAFF_RESERVE)) {
        did = yield* this.doAction(() => this.addWorker(player, f));
      } else {
        const spare = this.findExpendableWorker(player);
        if (spare && spare.tile !== f) did = yield* this.doAction(() => this.eh.aiMoveUnit(spare.unit, spare.tile, f));
      }
      if (!did) break;
    }
  }

  private workerCount(tile: TileBase): number {
    return tile.getUnits().filter((u) => u.getType() === 'BasicWorker').length;
  }

  /**
   * `staffIncome` — staff every income building toward OPTIMAL output (the Rust
   * `staff_income_inner(place_experts=true)` port). The previous TS scaffold
   * under-staffed: it put ONE worker on a mine (20 metal) when the optimum is 2
   * workers + 1 Expert (80 metal), and never placed a mine Expert — starving the
   * metal economy so the AI could never fund an army. This rewrite fully staffs
   * producers, gated by the shared unit cap (`getFreeUnitAmount`), per-tile space,
   * and the LOW STAFF_RESERVE so mechanical staffing is never starved by the
   * strategic `cfg.reserve`.
   *
   * Pass 0 (mines/plants first — metal is the army bottleneck): each Mine → 1
   * worker → Expert (×2) → 2nd worker = 80 metal/round; each plant → worker +
   * Expert (else produces 0). Pass 1: minimum-viable for farms / abundant forest.
   * Pass 2: a 2nd hydro worker (hydro = 80 × workers).
   */
  private *staffIncome(player: PlayerBase): Generator<void> {
    const producers = (kinds: string[]): TileBase[] =>
      M.ownedTiles(player).filter((t) => {
        const k = t.getBuilding()?.getType();
        return k !== undefined && kinds.includes(k);
      });

    // --- Pass 0: MINES + PLANTS to OPTIMAL (metal/energy fund the army) -------
    for (const tile of producers(['Mine'])) {
      yield* this.ensureWorker(player, tile); // 1st worker
      if (this.cfg.experts && M.hasType(tile, 'BasicWorker') && !M.hasType(tile, 'Expert')) {
        yield* this.doAction(() => this.addExpertReserve(player, tile, S.STAFF_RESERVE)); // ×2
      }
      if (this.workerCount(tile) < 2 && tile.hasSpaceForUnits()) {
        yield* this.doAction(() => this.addWorker(player, tile)); // 2nd worker
      }
    }
    for (const tile of producers(['Nuclear Power Plant', 'Hydroelectric Power Plant'])) {
      if (this.cfg.experts && !M.hasType(tile, 'Expert')) {
        yield* this.doAction(() => this.addExpertReserve(player, tile, S.STAFF_RESERVE));
      }
      if (M.hasType(tile, 'Expert') && !M.hasType(tile, 'BasicWorker')) {
        yield* this.doAction(() => this.addWorker(player, tile));
      }
    }

    // --- Pass 1: minimum-viable staffing for the rest (each producer >0) ------
    for (const tile of M.ownedTiles(player)) {
      const type = tile.getBuilding()?.getType();
      if (type === 'Farm') {
        if (!M.hasType(tile, 'BasicWorker')) yield* this.doAction(() => this.addWorker(player, tile));
      } else if (tile instanceof AbundantForest && !M.hasType(tile, 'BasicWorker')) {
        yield* this.doAction(() => this.addWorker(player, tile));
      }
    }
    // --- Pass 2: a 2nd hydro worker if cap/space allow (hydro = 80 × workers) -
    for (const tile of producers(['Hydroelectric Power Plant'])) {
      if (M.hasType(tile, 'Expert') && this.workerCount(tile) < 2 && tile.hasSpaceForUnits()) {
        yield* this.doAction(() => this.addWorker(player, tile));
      }
    }
  }

  /** Guarantee one worker on a key building, relocating an idle/forest worker if capped. */
  private *ensureWorker(player: PlayerBase, tile: TileBase): Generator<void> {
    if (M.hasType(tile, 'BasicWorker')) return;
    if (player.getFreeUnitAmount() > 0) {
      yield* this.doAction(() => this.addWorker(player, tile));
      return;
    }
    const spare = this.findIdleOnPlain(player) ?? this.findSpareWorker(player, tile);
    if (spare && spare.tile !== tile) yield* this.doAction(() => this.eh.aiMoveUnit(spare.unit, spare.tile, tile));
  }

  // --- economy scaffold: unit cap (villages) + metal source (mines) ----------

  /**
   * `ensureUnitCap` — MECHANICAL cap-expansion: build a Village when the shared
   * unit cap is the only thing blocking `staffIncome` from fully staffing the
   * existing producers (2 workers + Expert per Mine, worker + Expert per plant).
   * Port of the Rust `ensure_unit_cap`.
   *
   * Root cause this fixes: `staffIncome`'s coverage pass exhausts the free unit
   * cap putting 1 worker on each producer, so the Expert / 2nd-worker upgrade pass
   * never fires (experts = 0, mines stuck at 20 metal). Nothing else expands the
   * cap, so the metal economy could never fund an army. A Village (+3 unit slots)
   * is the only cap source the AI controls; the learned policy may still ignore
   * villages — this guarantees the economy fills.
   */
  private *ensureUnitCap(player: PlayerBase): Generator<void> {
    if (!this.cfg.experts) return; // no experts tier => no 3-unit producers; cap rarely binds
    // The unit cap is cached; refresh so getFreeUnitAmount reflects any village /
    // tile change earlier this turn (the learned loop may have built one).
    player.updateUnitAmounts();
    let deficit = 0;
    let anyUnderfilledTileHasSpace = false;
    for (const tile of M.ownedTiles(player)) {
      const kind = tile.getBuilding()?.getType();
      const optimal =
        kind === 'Mine' ? 3 // 2 workers + 1 expert = 80 metal
        : (kind === 'Nuclear Power Plant' || kind === 'Hydroelectric Power Plant') ? 2 // 1 worker + 1 expert
        : kind === 'Farm' ? 1 // 1 worker
        : 0;
      if (optimal === 0) continue;
      const current = this.workerCount(tile) + (M.hasType(tile, 'Expert') ? 1 : 0);
      const want = Math.max(0, optimal - current);
      if (want > 0) {
        deficit += want;
        if (tile.hasSpaceForUnits()) anyUnderfilledTileHasSpace = true;
      }
    }
    // Only expand when the cap is what's blocking us.
    const free = player.getFreeUnitAmount();
    if (deficit <= free || !anyUnderfilledTileHasSpace) return;
    // Solvency: a Village costs -5 money/round upkeep (arc sd4). Require net money
    // to stay non-negative after that upkeep alone. The new workers go on PRODUCERS
    // (they fund their own salary) so we do NOT pre-charge their salaries here.
    if (M.netMoneyPerRound(player) - 5 < 0) return;
    // With 0 villages there is no wood upkeep, so ensureWoodIncome harvests nothing
    // and wood sits at the starting level forever — the village's 200-wood cost can
    // never be afforded (the deepest layer of the starvation trap). When we genuinely
    // want a village but can't afford its wood, run a forest harvester to ACCUMULATE
    // wood toward the cost; the village is built on a later turn once the buffer is there.
    if (!S.affords(player, VILLAGE_BUILD_COST, this.cfg.reserve)) {
      yield* this.accumulateWoodFor(player, VILLAGE_BUILD_COST);
      return;
    }
    yield* this.doAction(() => this.buildVillage(player));
  }

  /**
   * `ensureMetalIncome` — MECHANICAL metal-source guarantee: build a Mine on an
   * owned buildable Mountain when the player has ZERO mines. Port of the Rust
   * `ensure_metal_income` (gate = 0). A SAFETY NET for the metal-starved tail
   * (games where the policy builds no mine), NOT a competitor — the policy still
   * owns mine COUNT past the first. Sequenced AFTER the cap/staff flow so it never
   * steals the early budget the village→cap chain needs.
   */
  private *ensureMetalIncome(player: PlayerBase): Generator<void> {
    const mountains = M.ownedTiles(player).filter(
      (t) => t instanceof Mountain && t.getBuilding() === null && t.getBuildableBuildings().includes('Mine'),
    );
    if (mountains.length === 0) return; // no metal source available
    const mines = M.ownedTiles(player).filter((t) => t.getBuilding()?.getType() === 'Mine').length;
    if (mines >= 1) return; // guarantee only the FIRST metal source; hand off to the policy
    if (!S.affords(player, MINE_BUILD_COST, this.cfg.reserve)) return;
    const costMoney = -(MINE_BUILD_COST.get(1 /* MONEY */) ?? 0);
    if (M.money(player) < costMoney + this.cfg.reserve + 100) return; // headroom — retry next turn
    // Wood is the early blocker (200 wood up-front, no wood income with 0 villages).
    if (!S.hasWoodBuffer(player, MINE_BUILD_COST)) {
      yield* this.accumulateWoodFor(player, MINE_BUILD_COST);
      return;
    }
    yield* this.doAction(() => this.buildMine(player, mountains[0]));
  }

  /**
   * Ensure at least one forest harvester runs so wood accumulates toward a
   * wood-costed build. With no villages there is no wood upkeep, so the normal
   * `ensureWoodIncome` no-ops and wood never grows — this proactive harvest
   * unblocks the first wood-blocked build (cap-expanding Village OR the first Mine).
   * Prefers a free unit slot; when capped, borrows an EXPENDABLE worker (idle /
   * surplus producer / farm) onto a forest. One placement per call. Port of the
   * Rust `accumulate_wood_for`.
   */
  private *accumulateWoodFor(player: PlayerBase, _cost: ResourceMap): Generator<void> {
    const haveHarvester = M.ownedTiles(player).some(
      (t) => t instanceof Forest && t.getBuilding() === null && M.hasType(t, 'BasicWorker'),
    );
    if (haveHarvester) return; // wood is already growing — just wait for the buffer
    const forest = M.ownedTiles(player).find(
      (t) => t instanceof Forest && t.getBuilding() === null && t.hasSpaceForUnits() && !M.hasType(t, 'BasicWorker'),
    );
    if (!forest) return; // no harvestable forest
    if (player.getFreeUnitAmount() > 0 && S.affords(player, BASIC_WORKER_COST, S.STAFF_RESERVE)) {
      yield* this.doAction(() => this.addWorker(player, forest));
      return;
    }
    // Capped with all producers minimally staffed (the trap): borrow an expendable
    // worker (idle / surplus producer), else least-critical farm worker. Mine
    // workers are NOT touched. staffIncome re-fills the farm once the cap rises.
    let borrow = this.findExpendableWorker(player);
    if (!borrow) {
      for (const t of M.ownedTiles(player)) {
        if (t.getBuilding()?.getType() === 'Farm') {
          const w = t.getUnits().find((u) => u.getType() === 'BasicWorker');
          if (w) { borrow = { unit: w, tile: t }; break; }
        }
      }
    }
    if (borrow && borrow.tile !== forest) {
      yield* this.doAction(() => this.eh.aiMoveUnit(borrow!.unit, borrow!.tile, forest));
    }
  }

  /**
   * Buy + place a Mine on an owned empty buildable Mountain. Solvency/wood gated by
   * the caller; uses `cfg.reserve` so it never dips into the strategic buffer.
   * Port of the Rust `build_mine`.
   */
  private buildMine(player: PlayerBase, spot: TileBase): boolean {
    if (!S.affords(player, MINE_BUILD_COST, this.cfg.reserve) || !S.hasWoodBuffer(player, MINE_BUILD_COST)) return false;
    if (!(spot instanceof Mountain) || spot.getBuilding() !== null || !spot.getBuildableBuildings().includes('Mine')) return false;
    return this.eh.aiBuildBuilding('Mine', spot);
  }

  /**
   * Buy + place a Village on the first empty owned buildable grassland (the
   * mechanical cap-fill path — already solvency-gated by `ensureUnitCap`). Uses
   * `cfg.reserve`; refreshes the cached unit cap so the following `staffIncome` can
   * spend the +3 new slots. Port of the Rust `build_village`.
   */
  private buildVillage(player: PlayerBase): boolean {
    if (!S.affords(player, VILLAGE_BUILD_COST, this.cfg.reserve) || !S.hasWoodBuffer(player, VILLAGE_BUILD_COST)) return false;
    const spot = M.ownedTiles(player).find(
      (t) => t instanceof Grassland && t.getBuilding() === null && t.getBuildableBuildings().includes('Village'),
    );
    if (!spot) return false;
    const built = this.eh.aiBuildBuilding('Village', spot);
    if (built) player.updateUnitAmounts(); // refresh the cached cap for the next staffIncome
    return built;
  }
}
