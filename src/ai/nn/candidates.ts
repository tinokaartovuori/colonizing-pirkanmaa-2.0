// Macro-action *intents* the neural policy chooses among.
//
// Each turn the controller enumerates the currently-legal, currently-affordable
// (per safety.ts) intents, the network scores each, and the best one is
// executed via the engine's AI primitives (aiBuildBuilding / aiBuyAndPlaceUnit
// / aiMoveUnit — the same hooks the heuristic uses). Target selection (which
// tile) uses cheap strong heuristics; the *network* decides which intent fires
// and when to stop (Pass). Because only safe/legal intents are emitted, the
// policy cannot bankrupt or make an illegal move — it can only play well or
// badly.

import { TileBase } from '../../model/tile';
import { Grassland, Forest, AbundantForest, Mountain, River } from '../../model/tiles';
import { UnitBase } from '../../model/unit';
import { PlayerBase } from '../../model/player';
import {
  ResourceMap,
  FARM_BUILD_COST, MINE_BUILD_COST, VILLAGE_BUILD_COST, OUTPOST_BUILD_COST,
  NUCLEARPP_BUILD_COST, HEPP_BUILD_COST, BASIC_WORKER_COST, EXPERT_COST, SOLDIER_COST,
  STRANGE_DEVICE_BUILD_COST, BRIDGE_BUILD_COST,
} from '../../core/resources';
import type { ObjectManager } from '../../managers/objectmanager';
import type { PlayerManager } from '../../managers/playermanager';
import * as M from './metrics';
import * as S from './safety';

/** Engine action surface used by the candidates (GameEventHandler implements it). */
export interface IAiActions {
  aiBuildBuilding(buildingString: string, tile: TileBase): boolean;
  aiBuyAndPlaceUnit(type: string, tile: TileBase): boolean;
  aiMoveUnit(unit: UnitBase, fromTile: TileBase, toTile: TileBase): boolean;
}

/** Tunable per-difficulty behaviour (the only thing that differs between tiers). */
export interface TierConfig {
  /** Max discretionary actions per turn. */
  budget: number;
  /** Softmax temperature for intent choice (0 = greedy argmax). */
  temperature: number;
  /** Cash reserve kept before discretionary spend (higher = more cautious). */
  reserve: number;
  /** Probability of a deliberate blunder (pick a uniformly random legal intent). */
  blunder: number;
  /** Allow hiring experts (power plants / mine boosters). */
  experts: boolean;
  /** Allow building / fielding any military (outposts, soldiers, attacks). */
  military: boolean;
  /** Allow nuclear plants. */
  nuclear: boolean;
  /** Allow the Strange Device endgame (the decisive closing move). */
  device: boolean;
}

export enum Intent {
  BuildFarm = 0,
  BuildMine,
  BuildVillage,
  BuildOutpost,
  BuildHydro,
  BuildNuclear,
  Expand,
  HireSoldier,
  Attack,
  StackProducer,
  Pass,
  // Added in the Strange-Device arc. Kept AFTER Pass so the existing intent
  // values are unchanged; only the one-hot width (INTENT_COUNT) grows 11→12,
  // which is network-breaking (policy input 63→64) — intended, we retrain.
  BuildStrangeDevice,
  // Plan-B action-space expansion (DEEP-REDESIGN-MEMO §6.2). Build a Bridge on
  // an owned River tile. Parity-locked with the Rust mirror.
  BuildBridge,
  // Plan-B action-space expansion (DEEP-REDESIGN-MEMO §6.2). Attack-on-Device
  // as a FIRST-CLASS intent: same Action as Attack but a distinct label so the
  // value head can learn the cracker line.
  CrackDevice,
  // Plan-B addendum. Attack-on-HQ as a FIRST-CLASS intent (same idea as
  // CrackDevice but for un-conquered enemy Headquarters).
  CrackHQ,
}
export const INTENT_COUNT = Intent.CrackHQ + 1;
export const LOCAL_DIM = 16;

/** Max distinct Expand target tiles emitted as candidates per turn (after sort). */
export const EXPAND_CANDIDATE_CAP = 6;
/** Max distinct feasible Attack target tiles emitted as candidates per turn (after sort). */
export const ATTACK_CANDIDATE_CAP = 4;

export interface Candidate {
  intent: Intent;
  local: number[];
  execute: () => boolean;
  label: string;
}

export interface AiCtx {
  eh: IAiActions;
  om: ObjectManager;
  pm: PlayerManager;
  player: PlayerBase;
  cfg: TierConfig;
}

// --- small helpers ---------------------------------------------------------

const claimValue = (t: TileBase): number => {
  const b = t.getBuilding();
  if (b && b.getType() === 'Mikontalo') return 6;
  switch (t.getType()) {
    case 'Mountain': return 5;
    case 'Grassland': return 4;
    case 'Forest': return 3;
    case 'Abundant Forest': return 2;
    default: return 1;
  }
};

const moneyCost = (c: ResourceMap): number => -(c.get(1) ?? 0); // BasicResource.MONEY = 1

function tileThreatened(tile: TileBase, p: PlayerBase): boolean {
  for (const n of tile.getNeighbourTiles()) {
    const o = n.getOwner();
    if (o !== null && o !== p && n.getUnits().some((u) => u.getType() === 'Soldier')) return true;
  }
  return false;
}

function findIdleWorker(p: PlayerBase): { unit: UnitBase; tile: TileBase } | null {
  for (const tile of M.ownedTiles(p)) {
    if (tile.getBuilding() || tile instanceof Forest || tile instanceof AbundantForest) continue;
    const w = tile.getUnits().find((u) => u.getType() === 'BasicWorker');
    if (w) return { unit: w, tile };
  }
  return null;
}

function findSurplusProducerWorker(p: PlayerBase): { unit: UnitBase; tile: TileBase } | null {
  for (const tile of M.ownedTiles(p)) {
    const type = tile.getBuilding()?.getType();
    if (type === 'Mine' || type === 'Nuclear Power Plant' || type === 'Hydroelectric Power Plant') {
      const ws = tile.getUnits().filter((u) => u.getType() === 'BasicWorker');
      if (ws.length > 1) return { unit: ws[ws.length - 1], tile };
    }
  }
  if (M.wood(p) >= 350) {
    for (const tile of M.ownedTiles(p)) {
      if (!(tile instanceof Forest)) continue;
      const w = tile.getUnits().find((u) => u.getType() === 'BasicWorker');
      if (w) return { unit: w, tile };
    }
  }
  return null;
}

function findFreeSoldier(p: PlayerBase, exclude: TileBase): { unit: UnitBase; tile: TileBase } | null {
  for (const tile of M.ownedTiles(p)) {
    if (tile === exclude) continue;
    const s = tile.getUnits().find((u) => u.getType() === 'Soldier');
    if (s) return { unit: s, tile };
  }
  return null;
}

/**
 * The 6 spatial/positional per-target features (local indices 10–15). All
 * map-size-invariant; see `tileSpatial`. Non-positional intents pass all zeros
 * (the defaults), keeping them value-equivalent + single-candidate.
 */
export interface SpatialFeatures {
  enemyNeighbors: number;      // 10: (#8-neighbors owned by ∉{null,p}) / 8, clamp[0,1]
  ownNeighbors: number;        // 11: (#8-neighbors owned by p) / 8,        clamp[0,1]
  neutralNeighbors: number;    // 12: (#8-neighbors with no owner) / 8,     clamp[0,1]
  distOwnHq: number;           // 13: Manhattan(tile, HQ; null→99) / 20,    clamp[0,3]
  distNearestEnemyTile: number;// 14: min Manhattan(tile, enemy tile; none→99)/20, clamp[0,3]
  frontier: number;            // 15: Attack=(#own Soldiers on 8-neighbors)/3; Expand=enemyNeighbors>0?1:0; else 0, clamp[0,3]
}

const ZERO_SPATIAL: SpatialFeatures = {
  enemyNeighbors: 0, ownNeighbors: 0, neutralNeighbors: 0,
  distOwnHq: 0, distNearestEnemyTile: 0, frontier: 0,
};

/** Generic local-feature vector for a candidate (see ai-neural.md). */
function localVec(opts: {
  p: PlayerBase;
  cost?: ResourceMap;
  netDelta?: number;
  targetValue?: number;
  unitCapGain?: number;
  soldierCapGain?: number;
  threatened?: boolean;
  incomeStaffing?: boolean;
  spatial?: SpatialFeatures;
}): number[] {
  const { p, cost } = opts;
  const cm = cost ? moneyCost(cost) : 0;
  const woodNeed = cost ? -(cost.get(2) ?? 0) : 0;
  const buffer = M.woodUpkeep(p) > 0 ? Math.max(100, M.woodUpkeep(p) * 5) : 0;
  const clamp = (v: number, lo = -3, hi = 3) => (v < lo ? lo : v > hi ? hi : v);
  const sp = opts.spatial ?? ZERO_SPATIAL;
  return [
    clamp(cm / 1000),
    clamp((opts.netDelta ?? 0) / 100),
    clamp((opts.targetValue ?? 0) / 6),
    clamp((opts.unitCapGain ?? 0) / 3),
    clamp((opts.soldierCapGain ?? 0) / 3),
    opts.threatened ? 1 : 0,
    clamp((M.money(p) - 120 - M.moneyDrainPerRound(p) * 5) / 1000),
    opts.incomeStaffing ? 1 : 0,
    clamp((M.wood(p) - woodNeed - buffer) / 500),
    clamp((M.metal(p) - 50) / 500),
    // --- spatial/positional (indices 10–15) ---
    clamp(sp.enemyNeighbors, 0, 1),
    clamp(sp.ownNeighbors, 0, 1),
    clamp(sp.neutralNeighbors, 0, 1),
    clamp(sp.distOwnHq, 0, 3),
    clamp(sp.distNearestEnemyTile, 0, 3),
    clamp(sp.frontier, 0, 3),
  ];
}

/**
 * Compute the 6 spatial features for a target tile. Reuses the parity-proven
 * `getNeighbourTiles()` (8-neighbours), `getOwner()`, `om.getHqTile(p)` and
 * `getCoordinate().x()/y()` — no inline neighbour/coordinate math.
 *
 * "enemy" = owner is non-null AND ≠ the acting player (neutral excluded).
 * Manhattan distance |dx|+|dy| matches controller.ts placement metric; sentinel
 * 99 (missing HQ / no enemy tiles) is applied BEFORE dividing by 20. Slot 15
 * (frontier) is filled by the caller per intent (Attack vs Expand). Clamps are
 * applied in `localVec`.
 */
export function tileSpatial(
  tile: TileBase,
  p: PlayerBase,
  om: ObjectManager,
  enemyCoords: { x: number; y: number }[],
): SpatialFeatures {
  const neighbours = tile.getNeighbourTiles();
  let enemyN = 0, ownN = 0, neutralN = 0;
  for (const n of neighbours) {
    const o = n.getOwner();
    if (o === null) neutralN++;
    else if (o === p) ownN++;
    else enemyN++;
  }
  const tc = tile.getCoordinate();
  const tx = tc.x(), ty = tc.y();

  const hq = om.getHqTile(p);
  let distHq = 99;
  if (hq) {
    const hc = hq.getCoordinate();
    distHq = Math.abs(tx - hc.x()) + Math.abs(ty - hc.y());
  }

  let distEnemy = 99;
  for (const e of enemyCoords) {
    const d = Math.abs(tx - e.x) + Math.abs(ty - e.y);
    if (d < distEnemy) distEnemy = d;
  }

  return {
    enemyNeighbors: enemyN / 8,
    ownNeighbors: ownN / 8,
    neutralNeighbors: neutralN / 8,
    distOwnHq: distHq / 20,
    distNearestEnemyTile: distEnemy / 20,
    frontier: 0, // set per-intent by the caller
  };
}

/**
 * Coordinates of all enemy-owned tiles (owner present AND ≠ p; neutral excluded).
 * Precomputed ONCE per `enumerate()` and threaded into expand/attack so the
 * nearest-enemy min-reduction is over a fixed list (commutative → order-independent;
 * iterated in `om.getTiles()` order for cleanliness).
 */
export function enemyTileCoords(om: ObjectManager, p: PlayerBase): { x: number; y: number }[] {
  const out: { x: number; y: number }[] = [];
  for (const t of om.getTiles()) {
    const o = t.getOwner();
    if (o !== null && o !== p) {
      const c = t.getCoordinate();
      out.push({ x: c.x(), y: c.y() });
    }
  }
  return out;
}

// --- intent builders -------------------------------------------------------

function emptyGrassland(p: PlayerBase): TileBase[] {
  return M.ownedTiles(p).filter(
    (t) => t instanceof Grassland && t.getBuilding() === null && t.getBuildableBuildings().includes('Farm'),
  );
}

function buildFarm(ctx: AiCtx): Candidate | null {
  const { player: p } = ctx;
  const spots = emptyGrassland(p);
  if (spots.length === 0) return null;
  if (!S.affordsIncomeBuild(p, FARM_BUILD_COST) || !S.hasWoodBuffer(p, FARM_BUILD_COST)) return null;
  // Prefer a grassland already holding an idle worker (instant staffing, no slot cost).
  const staffed = spots.find((t) => M.hasType(t, 'BasicWorker'));
  const spot = staffed ?? spots[0];
  return {
    intent: Intent.BuildFarm,
    local: localVec({ p, cost: FARM_BUILD_COST, netDelta: 44, targetValue: 4, incomeStaffing: !!staffed }),
    label: 'BuildFarm',
    execute: () => ctx.eh.aiBuildBuilding('Farm', spot),
  };
}

function buildMine(ctx: AiCtx): Candidate | null {
  const { player: p } = ctx;
  if (M.wood(p) < 300) return null;
  const mountain = M.ownedTiles(p).find((t) => t instanceof Mountain && t.getBuilding() === null);
  if (!mountain) return null;
  if (!S.affords(p, MINE_BUILD_COST, ctx.cfg.reserve) || !S.hasWoodBuffer(p, MINE_BUILD_COST)) return null;
  return {
    intent: Intent.BuildMine,
    local: localVec({ p, cost: MINE_BUILD_COST, netDelta: 20, targetValue: 5 }),
    label: 'BuildMine',
    execute: () => ctx.eh.aiBuildBuilding('Mine', mountain),
  };
}

function buildVillage(ctx: AiCtx): Candidate | null {
  const { player: p } = ctx;
  const spot = emptyGrassland(p)[0];
  if (!spot) return null;
  // Sustainability gates (mirror the heuristic's anti-bleed conditions).
  if (!M.ownedTiles(p).some((t) => t instanceof Forest && (t.getBuilding() === null || M.hasType(t, 'BasicWorker')))) return null;
  if (M.netMoneyPerRound(p) - 15 < 0) return null;
  const postUpkeep = M.woodUpkeep(p) + 10;
  if (M.wood(p) - 100 < Math.max(100, postUpkeep * 5)) return null;
  if (!S.affords(p, VILLAGE_BUILD_COST, ctx.cfg.reserve)) return null;
  return {
    intent: Intent.BuildVillage,
    local: localVec({ p, cost: VILLAGE_BUILD_COST, netDelta: -10, targetValue: 4, unitCapGain: 3 }),
    label: 'BuildVillage',
    execute: () => ctx.eh.aiBuildBuilding('Village', spot),
  };
}

function buildOutpost(ctx: AiCtx): Candidate | null {
  const { player: p, om, cfg } = ctx;
  if (!cfg.military) return null;
  // Tile gate lowered 12→8 to match the HARD bot — the old ≥12 was an asymmetric handicap
  // that delayed the NN's army past HARD's. Parity-locked with build_outpost in
  // rust-trainer/crates/cp-ai/src/candidates.rs.
  if (om.getTileCountForPlayer(p) < 8) return null;
  if (M.netMoneyPerRound(p) < 0) return null;
  const outposts = M.buildingCounts(p).Outpost;
  if (M.metalIncomePerRound(p) - (outposts + 1) * 15 < 0) return null;
  const spot = M.ownedTiles(p).find(
    (t) => t instanceof Grassland && t.getBuilding() === null && t.getBuildableBuildings().includes('Outpost'),
  );
  if (!spot) return null;
  // Affordability: the LIGHT, terminal-style standard (mirrors `buildStrangeDevice`)
  // rather than `affords`' `reserve + 5×drain` buffer. The strict buffer made the
  // Outpost UNREACHABLE for a reinvesting economy (after paying 650 money the residual
  // fell below `reserve + 5×drain`, so the candidate was never offered to the net). We
  // offer it whenever the player can (a) literally pay the raw cost, (b) carry the
  // Outpost's −50 money/round upkeep without going net-negative, and (c) keep a small
  // cash floor. The metal-income + net-income (pre-build) + tiles≥12 gates above still
  // guard sustainability. Mirrors `build_outpost` in
  // rust-trainer/crates/cp-ai/src/candidates.rs (parity-locked).
  if (!p.hasEnoughResources(OUTPOST_BUILD_COST)) return null;
  if (M.netMoneyPerRound(p) - 50 < 0) return null;
  if (M.money(p) - moneyCost(OUTPOST_BUILD_COST) < 50) return null;
  return {
    intent: Intent.BuildOutpost,
    local: localVec({ p, cost: OUTPOST_BUILD_COST, netDelta: -50, targetValue: 3, soldierCapGain: 3 }),
    label: 'BuildOutpost',
    execute: () => ctx.eh.aiBuildBuilding('Outpost', spot),
  };
}

function buildHydro(ctx: AiCtx): Candidate | null {
  const { player: p, cfg } = ctx;
  if (!cfg.experts) return null;
  if (M.netMoneyPerRound(p) <= 0) return null;
  const river = M.ownedTiles(p).find(
    (t) => t instanceof River && t.getBuilding() === null && t.getBuildableBuildings().includes('Hydroelectric Power Plant'),
  );
  if (!river) return null;
  if (!S.affords(p, HEPP_BUILD_COST, Math.min(cfg.reserve, 80)) || !S.hasWoodBuffer(p, HEPP_BUILD_COST)) return null;
  return {
    intent: Intent.BuildHydro,
    local: localVec({ p, cost: HEPP_BUILD_COST, netDelta: 80, targetValue: 3 }),
    label: 'BuildHydro',
    execute: () => ctx.eh.aiBuildBuilding('Hydroelectric Power Plant', river),
  };
}

function buildNuclear(ctx: AiCtx): Candidate | null {
  const { player: p, cfg } = ctx;
  if (!cfg.experts || !cfg.nuclear) return null;
  if (M.money(p) <= 2600 || p.getFreeUnitAmount() <= 1) return null;
  const spot = emptyGrassland(p).find((t) => !M.hasType(t, 'BasicWorker'));
  if (!spot) return null;
  if (!S.affords(p, NUCLEARPP_BUILD_COST, cfg.reserve) || !S.hasWoodBuffer(p, NUCLEARPP_BUILD_COST)) return null;
  return {
    intent: Intent.BuildNuclear,
    local: localVec({ p, cost: NUCLEARPP_BUILD_COST, netDelta: 160, targetValue: 5 }),
    label: 'BuildNuclear',
    execute: () => ctx.eh.aiBuildBuilding('Nuclear Power Plant', spot),
  };
}

/** # of a tile's 8-neighbours owned by an enemy (owner present AND ≠ p; neutral excluded). */
function enemyBorderCount(tile: TileBase, p: PlayerBase): number {
  let n = 0;
  for (const nb of tile.getNeighbourTiles()) {
    const o = nb.getOwner();
    if (o !== null && o !== p) n++;
  }
  return n;
}

/**
 * The Strange Device endgame as a neural intent — mirrors the heuristic bot's
 * gating (`ai.ts buildStrangeDevice`): build it only when the strategy is enabled,
 * no Device exists (one per game), the game has matured (≥18 rounds), we are NOT
 * losing on tiles, and the economy can carry the one-time cost as a terminal
 * play (raw resources + non-negative money net + a small cash floor). Placed on the
 * safest interior grassland (fewest enemy-bordering neighbours), which must be empty
 * (the Device never holds units). No Outpost is required (the game allows building the
 * Device on any buildable grassland) — the net is free to consider it without one.
 */
function buildStrangeDevice(ctx: AiCtx): Candidate | null {
  const { player: p, om, pm, cfg } = ctx;
  if (!cfg.device) return null;
  if (om.hasStrangeDevice()) return null; // one per game (counterplay handles an enemy's)
  if (pm.getRoundsPlayed() < 18) return null; // let the game develop first
  const myTiles = om.getTileCountForPlayer(p);
  const notLosing = pm.getPlayers().every((q) => q === p || om.getTileCountForPlayer(q) <= myTiles);
  if (!notLosing) return null;
  // Affordability for a TERMINAL play (the lighter standard the heuristic uses).
  if (!p.hasEnoughResources(STRANGE_DEVICE_BUILD_COST)) return null;
  if (M.netMoneyPerRound(p) < 0) return null;
  if (M.money(p) - moneyCost(STRANGE_DEVICE_BUILD_COST) < 150) return null;
  const spot = M.ownedTiles(p)
    .filter(
      (t) =>
        t instanceof Grassland &&
        t.getBuilding() === null &&
        t.getUnitCount() === 0 && // the Device can't be built on an occupied tile
        t.getBuildableBuildings().includes('Strange Device'),
    )
    .sort((a, b) => enemyBorderCount(a, p) - enemyBorderCount(b, p))[0];
  if (!spot) return null;
  return {
    intent: Intent.BuildStrangeDevice,
    local: localVec({ p, cost: STRANGE_DEVICE_BUILD_COST, netDelta: 0, targetValue: 6 }),
    label: 'BuildStrangeDevice',
    execute: () => ctx.eh.aiBuildBuilding('Strange Device', spot),
  };
}

/**
 * Plan-B `Intent.BuildBridge` (DEEP-REDESIGN-MEMO §6.2). Build a Bridge on an
 * owned River tile (no existing building, orientation allows Bridge), with raw
 * affordability + a small cash floor + wood buffer. Local feature
 * `targetValue = bridgeUnblockCount` — how many additional 4-neighbour tiles
 * would enter `getAvailableTiles()` if the river were bridged (cheap
 * neighbour scan). Mirrors `build_bridge` in
 * `rust-trainer/crates/cp-ai/src/candidates.rs` (parity-locked).
 */
function buildBridge(ctx: AiCtx): Candidate | null {
  const { player: p, cfg } = ctx;
  const river = M.ownedTiles(p).find(
    (t) =>
      t instanceof River && t.getBuilding() === null && t.getBuildableBuildings().includes('Bridge'),
  );
  if (!river) return null;
  // Raw cost affordability + wood buffer.
  if (!p.hasEnoughResources(BRIDGE_BUILD_COST)) return null;
  if (!S.hasWoodBuffer(p, BRIDGE_BUILD_COST)) return null;
  // Cash floor (mirrors the Outpost/Device terminal-style gate).
  if (M.money(p) - moneyCost(BRIDGE_BUILD_COST) < cfg.reserve) return null;
  // bridgeUnblockCount: count orthogonal-4 neighbours of the river tile that are
  // (a) not owned by `p` and (b) not already in availability — the additive gain
  // a Bridge here would yield. Matches Rust `bridge_unblock_count`.
  const preAvail = ctx.om.getAvailableTiles();
  let unblockCount = 0;
  for (const n of river.getNeighbourFourTiles()) {
    if (n.getOwner() === p) continue;
    if (preAvail.includes(n)) continue;
    if (n.hasOpponentHeadquarters(p)) unblockCount++;
  }
  return {
    intent: Intent.BuildBridge,
    local: localVec({ p, cost: BRIDGE_BUILD_COST, netDelta: -5, targetValue: unblockCount }),
    label: 'BuildBridge',
    execute: () => ctx.eh.aiBuildBuilding('Bridge', river),
  };
}

/**
 * Plan-B `Intent.CrackDevice` (DEEP-REDESIGN-MEMO §6.2). Enumerate when ANY
 * enemy owns a standing Strange Device AND the champ can stage ≥1 soldier on it
 * (the tile enters `getAvailableTiles()` and movable + buyable soldiers ≥ needed).
 * Action is functionally an Attack against the device tile; the SEPARATE intent
 * label gives the value head a distinct signal for the single biggest loss
 * source. Mirrors `crack_device` in
 * `rust-trainer/crates/cp-ai/src/candidates.rs` (parity-locked).
 */
function crackDevice(ctx: AiCtx, enemyCoords: { x: number; y: number }[]): Candidate | null {
  const { player: p, om, cfg } = ctx;
  if (!cfg.military) return null;
  const dev = om.findStrangeDeviceTile();
  if (!dev) return null;
  if (dev.getOwner() === p) return null; // we already own it — no crack
  const avail = om.getAvailableTiles();
  if (!avail.includes(dev)) return null;
  if (!dev.hasSpaceForConqueringUnits()) return null;
  const defenders = dev.getUnits().filter((u) => u.getType() === 'Soldier').length;
  if (defenders >= 3) return null;
  const needed = defenders + 1;
  const placed = dev.getConqueringUnits().filter((u) => u.getOwner() === p && u.getType() === 'Soldier').length;
  const toAdd = needed - placed;
  if (toAdd <= 0) return null;
  const canBuy = M.money(p) >= cfg.reserve + 250;
  const movable = M.ownedTiles(p).reduce(
    (n, t) => n + (t === dev ? 0 : t.getUnits().filter((u) => u.getType() === 'Soldier').length),
    0,
  );
  const buyable = canBuy
    ? Math.min(
        p.getFreeSoldierAmount(),
        Math.floor(M.metal(p) / 50),
        Math.floor((M.money(p) - cfg.reserve) / 200),
      )
    : 0;
  if (movable + buyable < toAdd) return null;
  // Countdown urgency: lower countdown = higher target_value (capped at 6).
  // The StrangeDevice subclass carries `getCountdown()`; other building types
  // don't, hence the optional-chain (we already gated on `findStrangeDeviceTile`
  // so in practice this is always present).
  const dBuilding = dev.getBuilding() as { getCountdown?: () => number } | null;
  const countdown = dBuilding?.getCountdown?.() ?? 0;
  const spatial = tileSpatial(dev, p, om, enemyCoords);
  let ownSoldierNeighbors = 0;
  for (const nb of dev.getNeighbourTiles()) {
    ownSoldierNeighbors += nb.getUnits().filter((u) => u.getType() === 'Soldier' && u.getOwner() === p).length;
  }
  spatial.frontier = ownSoldierNeighbors / 3;
  return {
    intent: Intent.CrackDevice,
    local: localVec({ p, netDelta: 0, targetValue: Math.max(0, 6 - Math.min(6, countdown)), spatial }),
    label: 'CrackDevice',
    execute: () => {
      let cur = placed;
      let did = false;
      while (cur < needed) {
        const spare = findFreeSoldier(p, dev);
        let step = false;
        if (spare) step = ctx.eh.aiMoveUnit(spare.unit, spare.tile, dev);
        else if (canBuy && p.getFreeSoldierAmount() > 0 && M.metal(p) >= 50 && S.affords(p, SOLDIER_COST, cfg.reserve))
          step = ctx.eh.aiBuyAndPlaceUnit('Soldier', dev);
        if (!step) break;
        did = true;
        cur++;
      }
      return did;
    },
  };
}

/**
 * Plan-B `Intent.CrackHQ` (Plan-B addendum). Enumerate when ANY enemy owns an
 * un-conquered Headquarters AND the champ can stage ≥1 soldier on it. Action is
 * functionally an Attack; SEPARATE intent label so the value head sees the
 * defender count it needs to beat (§3 strict-greater conquest). Mirrors
 * `crack_hq` in `rust-trainer/crates/cp-ai/src/candidates.rs` (parity-locked).
 */
function crackHQ(ctx: AiCtx, enemyCoords: { x: number; y: number }[]): Candidate | null {
  const { player: p, om, cfg } = ctx;
  if (!cfg.military) return null;
  const avail = om.getAvailableTiles();
  // Find an enemy-owned, un-conquered HQ that's reachable and has space.
  let hqTile: TileBase | null = null;
  for (const t of avail) {
    const b = t.getBuilding();
    if (
      b &&
      b.getType() === 'Headquarters' &&
      // un-conquered: TS uses Building.isConquered() or the renderer flag; we
      // check via `hasOpponentHeadquarters` on the enemy owner (matches the
      // `available_tiles` semantics).
      t.getOwner() !== null &&
      t.getOwner() !== p &&
      t.hasSpaceForConqueringUnits()
    ) {
      // A conquered HQ is the captor's own grassland → b.getType() would not be
      // 'Headquarters' for the captor. So owner!=p + type==Headquarters implies
      // un-conquered for the enemy.
      hqTile = t;
      break;
    }
  }
  if (!hqTile) return null;
  const hq = hqTile;
  const defenders = hq.getUnits().filter((u) => u.getType() === 'Soldier').length;
  if (defenders >= 3) return null;
  const needed = defenders + 1;
  const placed = hq.getConqueringUnits().filter((u) => u.getOwner() === p && u.getType() === 'Soldier').length;
  const toAdd = needed - placed;
  if (toAdd <= 0) return null;
  const canBuy = M.money(p) >= cfg.reserve + 250;
  const movable = M.ownedTiles(p).reduce(
    (n, t) => n + (t === hq ? 0 : t.getUnits().filter((u) => u.getType() === 'Soldier').length),
    0,
  );
  const buyable = canBuy
    ? Math.min(
        p.getFreeSoldierAmount(),
        Math.floor(M.metal(p) / 50),
        Math.floor((M.money(p) - cfg.reserve) / 200),
      )
    : 0;
  if (movable + buyable < toAdd) return null;
  const spatial = tileSpatial(hq, p, om, enemyCoords);
  let ownSoldierNeighbors = 0;
  for (const nb of hq.getNeighbourTiles()) {
    ownSoldierNeighbors += nb.getUnits().filter((u) => u.getType() === 'Soldier' && u.getOwner() === p).length;
  }
  spatial.frontier = ownSoldierNeighbors / 3;
  return {
    intent: Intent.CrackHQ,
    local: localVec({ p, netDelta: 0, targetValue: 6, spatial }),
    label: 'CrackHQ',
    execute: () => {
      let cur = placed;
      let did = false;
      while (cur < needed) {
        const spare = findFreeSoldier(p, hq);
        let step = false;
        if (spare) step = ctx.eh.aiMoveUnit(spare.unit, spare.tile, hq);
        else if (canBuy && p.getFreeSoldierAmount() > 0 && M.metal(p) >= 50 && S.affords(p, SOLDIER_COST, cfg.reserve))
          step = ctx.eh.aiBuyAndPlaceUnit('Soldier', hq);
        if (!step) break;
        did = true;
        cur++;
      }
      return did;
    },
  };
}

/**
 * Tile → index in `om.getTiles()`. `getTiles()` is the worldgen generation order
 * (column-major: index = x*height + y), so this map gives a stable, deterministic
 * total-order tie-break independent of map size. Precomputed ONCE per enumerate
 * call (passed down) to avoid O(n²) indexOf scans.
 */
function tileIndexMap(om: ObjectManager): Map<TileBase, number> {
  const m = new Map<TileBase, number>();
  const tiles = om.getTiles();
  for (let i = 0; i < tiles.length; i++) m.set(tiles[i], i);
  return m;
}

function expandCandidates(ctx: AiCtx, idx: Map<TileBase, number>, enemyCoords: { x: number; y: number }[]): Candidate[] {
  const { player: p, om, cfg } = ctx;
  const neutral = om
    .getAvailableTiles()
    .filter((t) => t.getOwner() === null && t.hasSpaceForUnits() && !tileThreatened(t, p));
  if (neutral.length === 0) return [];

  // Reachability: which mechanism delivers a worker is a per-turn property
  // (idle/hire/surplus), independent of WHICH neutral tile we target, so it is
  // computed once. Bail early if no worker can be delivered to any target.
  const idle = findIdleWorker(p);
  const canHire =
    p.getFreeUnitAmount() > 0 &&
    S.affords(p, BASIC_WORKER_COST, cfg.reserve) &&
    M.netMoneyPerRound(p) - 5 >= 0;
  const surplus = !idle && !canHire ? findSurplusProducerWorker(p) : null;
  if (!idle && !canHire && !surplus) return [];

  // Total order: claimValue DESC, then tile-index ASC (load-bearing for TS↔Rust
  // parity — argmax is strict `>`, lowest index wins on ties). Cap AFTER sorting.
  const sorted = neutral
    .slice()
    .sort((a, b) => claimValue(b) - claimValue(a) || (idx.get(a) ?? 0) - (idx.get(b) ?? 0))
    .slice(0, EXPAND_CANDIDATE_CAP);

  return sorted.map((tile) => {
    const capGain = (tile.getBuilding()?.getType() === 'Mikontalo') ? 2 : 0;
    const spatial = tileSpatial(tile, p, om, enemyCoords);
    spatial.frontier = spatial.enemyNeighbors > 0 ? 1 : 0; // Expand: frontier flag
    return {
      intent: Intent.Expand,
      local: localVec({ p, cost: canHire && !idle ? BASIC_WORKER_COST : undefined, netDelta: idle ? 0 : -5, targetValue: claimValue(tile), unitCapGain: capGain, spatial }),
      label: 'Expand',
      execute: () => {
        if (idle && idle.tile !== tile) return ctx.eh.aiMoveUnit(idle.unit, idle.tile, tile);
        if (canHire) return ctx.eh.aiBuyAndPlaceUnit('BasicWorker', tile);
        if (surplus && surplus.tile !== tile) return ctx.eh.aiMoveUnit(surplus.unit, surplus.tile, tile);
        return false;
      },
    };
  });
}

function hireSoldier(ctx: AiCtx): Candidate | null {
  const { player: p, om, cfg } = ctx;
  if (!cfg.military) return null;
  if (p.getFreeSoldierAmount() <= 0) return null;
  if (M.metal(p) < 50) return null;
  if (!S.affords(p, SOLDIER_COST, cfg.reserve) || !S.canAffordUpkeep(p, 30)) return null;
  const hq = om.getHqTile(p);
  const threatened = M.ownedTiles(p).filter((t) => t !== hq && tileThreatened(t, p));
  const tile = threatened[0] ?? hq ?? M.ownedTiles(p).find((t) => t.hasSpaceForUnits());
  if (!tile || !tile.hasSpaceForUnits()) return null;
  return {
    intent: Intent.HireSoldier,
    local: localVec({ p, cost: SOLDIER_COST, netDelta: -30, soldierCapGain: 0, threatened: threatened.length > 0 }),
    label: 'HireSoldier',
    execute: () => ctx.eh.aiBuyAndPlaceUnit('Soldier', tile),
  };
}

function attackCandidates(ctx: AiCtx, idx: Map<TileBase, number>, enemyCoords: { x: number; y: number }[]): Candidate[] {
  const { player: p, om, cfg } = ctx;
  if (!cfg.military) return [];
  const canBuy = M.money(p) >= cfg.reserve + 250;
  const targets = om
    .getAvailableTiles()
    .filter((t) => {
      const o = t.getOwner();
      return o !== null && o !== p && t.hasSpaceForConqueringUnits();
    })
    .map((t) => ({
      tile: t,
      defenders: t.getUnits().filter((u) => u.getType() === 'Soldier').length,
      isHq: t.getBuilding()?.getType() === 'Headquarters',
      isOutpost: t.getBuilding()?.getType() === 'Outpost',
    }))
    .filter((t) => !t.isOutpost && t.defenders < 3)
    // Total order: HQ-first, then fewest defenders, then tile-index ASC tie-break
    // (load-bearing for TS↔Rust parity — argmax is strict `>`, lowest index wins).
    .sort((a, b) =>
      Number(b.isHq) - Number(a.isHq) ||
      a.defenders - b.defenders ||
      (idx.get(a.tile) ?? 0) - (idx.get(b.tile) ?? 0),
    );

  const out: Candidate[] = [];
  for (const { tile, defenders, isHq } of targets) {
    if (out.length >= ATTACK_CANDIDATE_CAP) break;
    const needed = defenders + 1;
    const placed = tile.getConqueringUnits().filter((u) => u.getOwner() === p && u.getType() === 'Soldier').length;
    const toAdd = needed - placed;
    if (toAdd <= 0) continue;
    const movable = M.ownedTiles(p).reduce(
      (n, t) => n + (t === tile ? 0 : t.getUnits().filter((u) => u.getType() === 'Soldier').length),
      0,
    );
    const buyable = canBuy
      ? Math.min(
          p.getFreeSoldierAmount(),
          Math.floor(M.metal(p) / 50),
          Math.floor((M.money(p) - cfg.reserve) / 200),
        )
      : 0;
    if (movable + buyable < toAdd) continue;
    // Feasible assault on this tile — emit one candidate per feasible target.
    const spatial = tileSpatial(tile, p, om, enemyCoords);
    // Attack frontier (slot 15): my Soldiers on the target's 8-neighbours / 3
    // (counted by soldier owner == p, regardless of the neighbour tile's owner).
    let ownSoldierNeighbors = 0;
    for (const nb of tile.getNeighbourTiles()) {
      ownSoldierNeighbors += nb.getUnits().filter((u) => u.getType() === 'Soldier' && u.getOwner() === p).length;
    }
    spatial.frontier = ownSoldierNeighbors / 3;
    out.push({
      intent: Intent.Attack,
      local: localVec({ p, netDelta: 0, targetValue: isHq ? 6 : 4 - defenders, soldierCapGain: 0, spatial }),
      label: 'Attack' + (isHq ? ':HQ' : ''),
      execute: () => {
        let cur = placed;
        let did = false;
        while (cur < needed) {
          const spare = findFreeSoldier(p, tile);
          let step = false;
          if (spare) step = ctx.eh.aiMoveUnit(spare.unit, spare.tile, tile);
          else if (canBuy && p.getFreeSoldierAmount() > 0 && M.metal(p) >= 50 && S.affords(p, SOLDIER_COST, cfg.reserve))
            step = ctx.eh.aiBuyAndPlaceUnit('Soldier', tile);
          if (!step) break;
          did = true;
          cur++;
        }
        return did;
      },
    });
  }
  return out;
}

function stackProducer(ctx: AiCtx): Candidate | null {
  const { player: p, cfg } = ctx;
  if (p.getFreeUnitAmount() <= 0) return null;
  const tile = M.ownedTiles(p).find((t) => {
    const type = t.getBuilding()?.getType();
    return (type === 'Mine' || type === 'Nuclear Power Plant' || type === 'Hydroelectric Power Plant') && t.hasSpaceForUnits();
  });
  if (!tile) return null;
  const wantExpert = cfg.experts && tile.getBuilding()?.getType() !== 'Hydroelectric Power Plant' && !M.hasType(tile, 'Expert') && p.getFreeUnitAmount() > 1;
  const cost = wantExpert ? EXPERT_COST : BASIC_WORKER_COST;
  if (!S.affords(p, cost, wantExpert ? cfg.reserve : S.STAFF_RESERVE)) return null;
  return {
    intent: Intent.StackProducer,
    local: localVec({ p, cost, netDelta: 20, targetValue: 3, incomeStaffing: true }),
    label: 'StackProducer' + (wantExpert ? ':Expert' : ''),
    execute: () => ctx.eh.aiBuyAndPlaceUnit(wantExpert ? 'Expert' : 'BasicWorker', tile),
  };
}

const PASS: Candidate = { intent: Intent.Pass, local: new Array(LOCAL_DIM).fill(0), label: 'Pass', execute: () => true };

/**
 * All currently-legal, currently-affordable intents (Pass always last).
 *
 * Builder order is fixed and load-bearing (TS↔Rust parity): [BuildFarm, BuildMine,
 * BuildVillage, BuildOutpost, BuildHydro, BuildNuclear, BuildStrangeDevice, Expand,
 * HireSoldier, Attack, StackProducer], then Pass. Build* / StackProducer /
 * HireSoldier / BuildStrangeDevice are single-candidate (0 or 1). Expand and Attack
 * are MULTI-candidate: they emit one Candidate per plausible target tile (each
 * carrying its own per-tile `local` vector), spread into the list in their builders'
 * total-sorted order. (BuildStrangeDevice has intent VALUE 11 but sits at list
 * position 6 — the list is no longer monotonic in intent value, which doesn't
 * matter: the one-hot encodes the value, and the argmax tie-break uses list
 * POSITION, lowest index. Pass is always last in the list.)
 */
export function enumerate(ctx: AiCtx): Candidate[] {
  const idx = tileIndexMap(ctx.om);
  const enemyCoords = enemyTileCoords(ctx.om, ctx.player);
  const out: Candidate[] = [];
  let c: Candidate | null;
  if ((c = buildFarm(ctx))) out.push(c);
  if ((c = buildMine(ctx))) out.push(c);
  if ((c = buildVillage(ctx))) out.push(c);
  if ((c = buildOutpost(ctx))) out.push(c);
  if ((c = buildHydro(ctx))) out.push(c);
  if ((c = buildNuclear(ctx))) out.push(c);
  if ((c = buildStrangeDevice(ctx))) out.push(c);
  if ((c = buildBridge(ctx))) out.push(c);
  out.push(...expandCandidates(ctx, idx, enemyCoords));
  if ((c = hireSoldier(ctx))) out.push(c);
  out.push(...attackCandidates(ctx, idx, enemyCoords));
  // Plan-B Crack candidates: piggy-back on Attack action but with a distinct
  // intent label so the value head learns the cracker line.
  if ((c = crackDevice(ctx, enemyCoords))) out.push(c);
  if ((c = crackHQ(ctx, enemyCoords))) out.push(c);
  if ((c = stackProducer(ctx))) out.push(c);
  out.push(PASS);
  return out;
}
