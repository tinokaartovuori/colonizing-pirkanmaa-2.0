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
import { Forest, AbundantForest } from '../../model/tiles';
import { UnitBase } from '../../model/unit';
import { PlayerBase } from '../../model/player';
import { BASIC_WORKER_COST, EXPERT_COST } from '../../core/resources';
import type { GameEventHandler } from '../../managers/gameeventhandler';
import type { ObjectManager } from '../../managers/objectmanager';
import type { PlayerManager } from '../../managers/playermanager';
import { Genome } from './mlp';
import { globalFeatures } from './features';
import { enumerate, AiCtx, TierConfig, Intent } from './candidates';
import { select, scoreCandidate } from './policy';
import { SearchConfig, select as searchSelect } from './search';
import { ValueNet } from './value';
import { buildSnapshot } from '../../managers/persistence';
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
      // 1. Safety scaffold — guarantees solvency before any discretionary play.
      yield* this.ensureWoodIncome(player);
      yield* this.staffIncome(player);

      // 2. Learned decision loop.
      const ctx: AiCtx = { eh: this.eh, om: this.om, pm: this.pm, player, cfg: this.cfg };
      const round = this.pm.getRoundsPlayed();
      while (this.budget > 0) {
        const gvec = globalFeatures(player, this.om, this.pm, round);
        let cands = enumerate(ctx);
        let choice: ReturnType<typeof select>;
        if (this.search) {
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
        // Realize the obvious follow-up: staff anything left unstaffed.
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
  private addExpert(player: PlayerBase, tile: TileBase): boolean {
    if (!S.affords(player, EXPERT_COST, this.cfg.reserve)) return false;
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

  /** Ensure each income building has the worker(s)/expert it needs to produce. */
  private *staffIncome(player: PlayerBase): Generator<void> {
    for (const tile of M.ownedTiles(player)) {
      const type = tile.getBuilding()?.getType();
      if (type === 'Farm') {
        if (!M.hasType(tile, 'BasicWorker')) yield* this.doAction(() => this.addWorker(player, tile));
      } else if (type === 'Mine') {
        yield* this.ensureWorker(player, tile);
      } else if (type === 'Nuclear Power Plant') {
        if (this.cfg.experts && !M.hasType(tile, 'Expert')) yield* this.doAction(() => this.addExpert(player, tile));
      } else if (type === 'Hydroelectric Power Plant') {
        if (this.cfg.experts && !M.hasType(tile, 'Expert')) yield* this.doAction(() => this.addExpert(player, tile));
        if (M.hasType(tile, 'Expert') && !M.hasType(tile, 'BasicWorker')) yield* this.doAction(() => this.addWorker(player, tile));
      } else if (tile instanceof AbundantForest && !M.hasType(tile, 'BasicWorker')) {
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
}
