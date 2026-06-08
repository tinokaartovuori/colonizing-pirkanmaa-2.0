// TS twin of the Rust AlphaZero spatial CNN (`rust-trainer/crates/cp-ai/src/
// {cnn.rs, planes.rs, spatial_net.rs}` + the deploy glue in `cnn_train.rs`).
//
// This is the FORWARD-only inference path for deploying a trained `SpatialNet`
// (the CNN champion, e.g. sd4-az-002) into the browser game. The Rust trainer's
// CNN architecture is:
//   trunk:  planes(PC,H,W) -> Conv2d(PC->D1,3,1) -> tanh -> Conv2d(D1->D,3,1) ->
//           tanh = board_embed (the deployed champion has NO residual conv3).
//   pool:   global_embed = GlobalAvgPool(board_embed)  (length D)
//   policy: per candidate, input =
//           concat( target_embed(D) @ (x,y) | global_embed(D) | local(LOCAL) |
//                   intent_onehot(INTENT) )
//           -> Dense(2D+LOCAL+INTENT -> HP) -> tanh -> Dense(HP->1) = scalar score.
// The value head (Dense over global_embed ⊕ value_scalars) is NOT used by the
// greedy deploy policy (argmax of `score_candidate`), so it is omitted here.
//
// All math is f64 (JS number), mirroring the Rust f64 path so a deployed net
// scores candidates the same way it was trained/benchmarked. The plane builder
// (`boardPlanes`) is a faithful port of `planes.rs board_planes` (27 channels);
// `candFeat` mirrors `cnn_train.rs cand_feat` (shared local 0..15 + 2 CNN-only
// capacity features); `intentOnehot` / `targetXY` mirror the same-named helpers.

import { TileBase } from '../../model/tile';
import { AbundantForest, Forest, Grassland, Mountain, River } from '../../model/tiles';
import { PlayerBase } from '../../model/player';
import { BasicResource, strangeDeviceCountdown } from '../../core/resources';
import type { ObjectManager } from '../../managers/objectmanager';
import type { PlayerManager } from '../../managers/playermanager';
import { Candidate, INTENT_COUNT, LOCAL_DIM } from './candidates';
import { moneyDrainPerRound } from './metrics';
import { StrangeDevice } from '../../model/building';

// ---------------------------------------------------------------------------
// Serialized net (matches the Rust serde JSON of SpatialNet exactly).
// ---------------------------------------------------------------------------

export interface ConvLayer {
  in_ch: number;
  out_ch: number;
  k: number;
  pad: number;
  dilation?: number;
  /** out_ch*in_ch*k*k, layout [oc][ic][ky][kx]. */
  weights: number[];
  /** out_ch. */
  bias: number[];
}
export interface DenseLayer {
  in_dim: number;
  out_dim: number;
  /** out_dim*in_dim, layout [o][i]. */
  weights: number[];
  /** out_dim. */
  bias: number[];
}

/** The serialized SpatialNet weights (subset used for the policy forward). */
export interface SpatialWeights {
  plane_count: number;
  local_dim: number;
  intent_dim: number;
  value_scalar_dim?: number;
  d1: number;
  d: number;
  hv: number;
  hp: number;
  conv1: ConvLayer;
  conv2: ConvLayer;
  conv3?: ConvLayer | null;
  // value head (kept for completeness; unused by the greedy policy path).
  value_d1?: DenseLayer;
  value_d2?: DenseLayer;
  policy_d1: DenseLayer;
  policy_d2: DenseLayer;
}

// ---------------------------------------------------------------------------
// CNN primitives (mirror cnn.rs forward math; allocating reference path).
// ---------------------------------------------------------------------------

/** Flat index into a (C,H,W) map: idx(c,y,x) = (c*H + y)*W + x. */
function idx(c: number, y: number, x: number, h: number, w: number): number {
  return (c * h + y) * w + x;
}

/** Zero-padded same-size cross-correlation. weights [oc][ic][ky][kx]. */
function convForward(layer: ConvLayer, input: number[], h: number, w: number): number[] {
  const { in_ch, out_ch, k, pad, weights, bias } = layer;
  const dil = layer.dilation ?? 1;
  const out = new Array<number>(out_ch * h * w).fill(0);
  for (let oc = 0; oc < out_ch; oc++) {
    const b = bias[oc];
    for (let oy = 0; oy < h; oy++) {
      for (let ox = 0; ox < w; ox++) {
        let sum = b;
        for (let ic = 0; ic < in_ch; ic++) {
          for (let ky = 0; ky < k; ky++) {
            const iy = oy + ky * dil - pad;
            if (iy < 0 || iy >= h) continue;
            for (let kx = 0; kx < k; kx++) {
              const ix = ox + kx * dil - pad;
              if (ix < 0 || ix >= w) continue;
              const wIdx = ((oc * in_ch + ic) * k + ky) * k + kx;
              sum += weights[wIdx] * input[idx(ic, iy, ix, h, w)];
            }
          }
        }
        out[idx(oc, oy, ox, h, w)] = sum;
      }
    }
  }
  return out;
}

function tanhForward(a: number[]): number[] {
  return a.map((v) => Math.tanh(v));
}

/** Global average pool (C,H,W) -> (C,): mean over H*W. */
function globalAvgPool(input: number[], c: number, h: number, w: number): number[] {
  const area = h * w;
  const out = new Array<number>(c).fill(0);
  for (let ch = 0; ch < c; ch++) {
    let sum = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) sum += input[idx(ch, y, x, h, w)];
    out[ch] = sum / area;
  }
  return out;
}

/** Dense forward. weights [o][i]. */
function denseForward(layer: DenseLayer, input: number[]): number[] {
  const { in_dim, out_dim, weights, bias } = layer;
  const out = new Array<number>(out_dim);
  for (let o = 0; o < out_dim; o++) {
    let sum = bias[o];
    const base = o * in_dim;
    for (let i = 0; i < in_dim; i++) sum += weights[base + i] * input[i];
    out[o] = sum;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Plane builder — faithful port of planes.rs board_planes (27 channels).
// ---------------------------------------------------------------------------

export const PLANE_COUNT = 27;

// Channel indices (1:1 with planes.rs C_* constants).
const C_MINE = 0;
const C_ENEMY = 1;
const C_NEUTRAL = 2;
const C_MY_HQ = 3;
const C_ENEMY_HQ = 4;
const C_PRODUCER = 5;
const C_MILITARY = 6;
const C_DEVICE = 7;
const C_MY_OWNED_SOLDIERS = 8;
const C_HQ_CONNECTED = 9;
const C_T_GRASSLAND = 10;
const C_T_FOREST = 11;
const C_T_MOUNTAIN = 12;
const C_T_RIVER = 13;
const C_PRODUCING = 14;
const C_ENEMY_OWNED_SOLDIERS = 15;
const C_ENEMY_REACH = 16;
const C_MY_REACH = 17;
const C_MY_CONQ_SOLDIERS = 18;
const C_ENEMY_CONQ_SOLDIERS = 19;
const C_ATT_MINUS_DEF = 20;
const C_DEVICE_DEFENSELESS = 21;
const C_RIVER_BLOCK = 22;
const C_ENEMY_BUDGET = 23;
const C_DIST_TO_ENEMY_HQ = 24;
const C_DIST_TO_ENEMY_DEVICE = 25;
const C_MY_BUDGET = 26;

const PRODUCER_TYPES = new Set([
  'Farm', 'Mine', 'Village', 'Hydroelectric Power Plant', 'Nuclear Power Plant',
]);

/** Live enemies of `p`: every other player that still owns ≥1 tile (the deploy
 *  proxy for the Rust `live_players()` non-neutralised filter — a player with no
 *  tiles is out of the game). */
function liveEnemies(p: PlayerBase, om: ObjectManager, pm: PlayerManager): PlayerBase[] {
  return pm.getPlayers().filter((q) => q !== p && om.getTileCountForPlayer(q) > 0);
}

function isProducingProducer(tile: TileBase): boolean {
  const b = tile.getBuilding();
  if (!b) return false;
  const has = (type: string) => tile.getUnits().some((u) => u.getType() === type);
  switch (b.getType()) {
    case 'Farm':
      // gen_grassland pays out when stored growth_phase == 4 (engine adds +1 → 5)
      // and a worker is present.
      return (b as { getGrowthPhase?: () => number }).getGrowthPhase?.() === 4 && has('BasicWorker');
    case 'Mine':
      return has('BasicWorker');
    case 'Hydroelectric Power Plant':
    case 'Nuclear Power Plant':
      return has('Expert');
    case 'Village':
    case 'Outpost':
      return true;
    default:
      return false;
  }
}

function countSoldiers(units: { getType(): string }[]): number {
  let n = 0;
  for (const u of units) if (u.getType() === 'Soldier') n++;
  return n;
}

/** A player's mobile-soldier budget (port of planes.rs enemy_mobile_budget). */
function mobileBudget(q: PlayerBase): number {
  const owned = q.getCurrentSoldierAmount();
  const money = q.getResources().get(BasicResource.MONEY) ?? 0;
  const metal = q.getResources().get(BasicResource.METAL) ?? 0;
  const affordable = Math.max(0, Math.min(Math.floor(money / 200), Math.floor(metal / 50)));
  const freeSlots = Math.max(0, q.getFreeSoldierAmount());
  return owned + Math.min(affordable, freeSlots);
}

/** Per-player getAvailableTiles() (port of objectmanager.getAvailableTiles but for
 *  an arbitrary player; mirrors the Rust get_available_tiles_for). */
function availableTilesFor(player: PlayerBase, om: ObjectManager): Set<TileBase> {
  const avail = new Set<TileBase>();
  for (const obj of player.getObjects()) {
    if (!(obj instanceof TileBase)) continue;
    const tile = obj;
    if (tile.getOwner() === player && tile.hasOpponentHeadquarters(player)) avail.add(tile);
    if (tile.getType() === 'River' && tile.getBuilding() === null) continue;
    for (const nTile of tile.getNeighbourFourTiles()) {
      if (avail.has(nTile)) continue;
      if (nTile.hasOpponentHeadquarters(player)) avail.add(nTile);
    }
  }
  return avail;
}

// ---------------------------------------------------------------------------
// Value-head per-state scalar features — port of cnn_train.rs `value_scalars`.
// ---------------------------------------------------------------------------

/** Length of the value-head scalar vector (cnn_train.rs VALUE_SCALAR_DIM). */
export const VALUE_SCALAR_DIM = 12;
const DEVICE_MONEY_COST = 1300; // STRANGE_DEVICE_BUILD_COST money component.
const DEVICE_MIN_ROUND = 18; // build_strange_device rounds≥18 gate.

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function countWorkers(tile: TileBase): number {
  let n = 0;
  for (const u of tile.getUnits()) if (u.getType() === 'BasicWorker') n++;
  return n;
}

/** Growth-aware realized MONEY income/round (port of realized_income_per_round). */
function realizedIncomePerRound(player: PlayerBase): number {
  let income = 0;
  for (const tile of player.getObjects()) {
    if (!(tile instanceof TileBase)) continue;
    const b = tile.getBuilding();
    if (!b) continue;
    if (!isProducingProducer(tile)) {
      // isProducingProducer only covers the planes producer set; here we also
      // need the same gate used by realized_income (Farm/Mine/Hydro/Nuclear/
      // Village/Outpost). isProducingProducer returns true exactly for those
      // when producing, so reuse it directly.
      continue;
    }
    const money = b.getProduction().get(BasicResource.MONEY) ?? 0;
    const kind = b.getType();
    if (kind === 'Mine') {
      const workers = countWorkers(tile);
      const mult = tile.getUnits().some((u) => u.getType() === 'Expert') ? 2 : 1;
      income += money * workers * mult;
    } else if (kind === 'Hydroelectric Power Plant' || kind === 'Nuclear Power Plant') {
      income += money * countWorkers(tile);
    } else {
      income += money;
    }
  }
  return income - moneyDrainPerRound(player);
}

/**
 * VALUE_SCALAR_DIM-length per-state scalar feature vector for the value head, from
 * `player`'s perspective. 1:1 port of cnn_train.rs `value_scalars`. All entries
 * bounded to ≈[-1,1].
 */
export function valueScalars(
  player: PlayerBase, om: ObjectManager, pm: PlayerManager,
): number[] {
  const enemies = liveEnemies(player, om, pm);

  const inc = clamp01(realizedIncomePerRound(player) / 400);

  // Staffed ratio (growth-aware).
  let total = 0, producing = 0;
  for (const tile of player.getObjects()) {
    if (!(tile instanceof TileBase)) continue;
    const b = tile.getBuilding();
    if (!b || !PRODUCER_TYPES.has(b.getType())) continue;
    total += 1;
    if (isProducingProducer(tile)) producing += 1;
  }
  const staffedRatio = producing / Math.max(1, total);

  // Filled (used) capacity.
  const usedUnit = Math.max(0, player.getMaxUnitAmount() - player.getFreeUnitAmount());
  const usedSoldier = Math.max(0, player.getMaxSoldierAmount() - player.getFreeSoldierAmount());
  const usedUnitN = clamp01(usedUnit / 10);
  const usedSoldierN = clamp01(usedSoldier / 6);

  // Treasury toward the Device.
  const money = player.getResources().get(BasicResource.MONEY) ?? 0;
  const bank = clamp01(money / DEVICE_MONEY_COST);

  // Tile lead, signed in [-1,1].
  const myTiles = om.getTileCountForPlayer(player);
  let maxEnemy = 0;
  for (const q of enemies) maxEnemy = Math.max(maxEnemy, om.getTileCountForPlayer(q));
  const totalTiles = Math.max(1, om.getTileCount());
  const tileLead = Math.max(-1, Math.min(1, (myTiles - maxEnemy) / totalTiles));

  // Device-window flag.
  const rounds = pm.getRoundsPlayed();
  const notLosing = enemies.every((q) => om.getTileCountForPlayer(q) <= myTiles);
  const hasDevice = om.hasStrangeDevice();
  const deviceWindow = rounds >= DEVICE_MIN_ROUND && !hasDevice && notLosing ? 1 : 0;

  // My device countdown / 40.
  const devTile = om.findStrangeDeviceTile();
  let myCountdown = 0;
  if (devTile && devTile.getOwner() === player) {
    const b = devTile.getBuilding();
    const cd = b instanceof StrangeDevice ? b.getCountdown() : 0;
    myCountdown = clamp01(cd / 40);
  }

  // Relative army strength (signed): my soldiers vs strongest live enemy.
  const mySol = player.getCurrentSoldierAmount();
  let maxEnemySol = 0;
  for (const q of enemies) maxEnemySol = Math.max(maxEnemySol, q.getCurrentSoldierAmount());
  const relArmy = Math.tanh((mySol - maxEnemySol) / 4);

  // Headroom (capacity-blindness fix).
  const soldierHeadroom = clamp01(player.getFreeSoldierAmount() / 6);
  const workerHeadroom = clamp01(player.getFreeUnitAmount() / 10);

  // Enemy device threat (mirror of my_countdown), progress in [0,1].
  let enemyDeviceThreat = 0;
  if (devTile) {
    const o = devTile.getOwner();
    if (o !== null && o !== player && enemies.includes(o)) {
      const b = devTile.getBuilding();
      const cd = b instanceof StrangeDevice ? Math.max(0, b.getCountdown()) : 0;
      const maxCd = Math.max(1, strangeDeviceCountdown(om.getTileCount()));
      enemyDeviceThreat = clamp01((maxCd - cd) / maxCd);
    }
  }

  return [
    inc, staffedRatio, usedUnitN, usedSoldierN, bank, tileLead,
    deviceWindow, myCountdown, relArmy, soldierHeadroom, workerHeadroom, enemyDeviceThreat,
  ];
}

/** Build the (PLANE_COUNT,H,W) tensor for `player`. Returns {planes,h,w}. */
export function boardPlanes(
  player: PlayerBase, om: ObjectManager, pm: PlayerManager,
): { planes: number[]; h: number; w: number } {
  const tiles = om.getTiles();
  let maxX = 0, maxY = 0;
  for (const t of tiles) {
    const c = t.getCoordinate();
    if (c.x() > maxX) maxX = c.x();
    if (c.y() > maxY) maxY = c.y();
  }
  const w = Math.max(1, maxX + 1);
  const h = Math.max(1, maxY + 1);
  const out = new Array<number>(PLANE_COUNT * h * w).fill(0);

  const enemies = liveEnemies(player, om, pm);
  const enemySet = new Set(enemies);
  const isLiveEnemy = (o: PlayerBase | null) => o !== null && enemySet.has(o);

  const diameter = w + h;
  // Live-enemy HQ coords.
  const enemyHqCoords: { x: number; y: number }[] = [];
  for (const op of enemies) {
    const hq = om.getHqTile(op);
    if (hq) { const c = hq.getCoordinate(); enemyHqCoords.push({ x: c.x(), y: c.y() }); }
  }
  // Enemy-owned standing Device coord (≤1).
  let enemyDevice: { x: number; y: number } | null = null;
  const devTile = om.findStrangeDeviceTile();
  if (devTile && isLiveEnemy(devTile.getOwner())) {
    const c = devTile.getCoordinate();
    enemyDevice = { x: c.x(), y: c.y() };
  }

  for (const t of tiles) {
    const co = t.getCoordinate();
    const x = co.x(), y = co.y();
    if (x < 0 || y < 0 || x >= w || y >= h) continue;
    const cell = (c: number) => idx(c, y, x, h, w);

    // Terrain (one-hot; forest merges Forest + AbundantForest).
    if (t instanceof Grassland) out[cell(C_T_GRASSLAND)] = 1;
    else if (t instanceof Forest || t instanceof AbundantForest) out[cell(C_T_FOREST)] = 1;
    else if (t instanceof Mountain) out[cell(C_T_MOUNTAIN)] = 1;
    else if (t instanceof River) out[cell(C_T_RIVER)] = 1;

    const owner = t.getOwner();
    const ownedByMe = owner === player;
    const ownedByEnemy = isLiveEnemy(owner);
    if (ownedByMe) out[cell(C_MINE)] = 1;
    else if (ownedByEnemy) out[cell(C_ENEMY)] = 1;
    else if (owner === null) out[cell(C_NEUTRAL)] = 1;
    // else: dead/neutralised owner — neither plane set.

    const b = t.getBuilding();
    if (b) {
      const kind = b.getType();
      if (kind === 'Outpost') out[cell(C_MILITARY)] = 1;
      else if (kind === 'Strange Device') { out[cell(C_DEVICE)] = 1; out[cell(C_DEVICE_DEFENSELESS)] = 1; }
      else if (PRODUCER_TYPES.has(kind)) out[cell(C_PRODUCER)] = 1;
      if (ownedByMe && isProducingProducer(t)) out[cell(C_PRODUCING)] = 1;
    }

    if (ownedByMe && t instanceof River && t.getBuilding() === null) out[cell(C_RIVER_BLOCK)] = 1;

    // Soldiers: owned defenders vs conquering attackers, by side.
    const ownedSol = countSoldiers(t.getUnits());
    if (ownedSol > 0) {
      if (ownedByMe) out[cell(C_MY_OWNED_SOLDIERS)] = Math.min(1, ownedSol / 5);
      else if (ownedByEnemy) out[cell(C_ENEMY_OWNED_SOLDIERS)] = Math.min(1, ownedSol / 5);
    }
    let myConq = 0, enemyConq = 0;
    for (const u of t.getConqueringUnits()) {
      if (u.getType() !== 'Soldier') continue;
      const uo = u.getOwner();
      if (uo === player) myConq++;
      else if (isLiveEnemy(uo)) enemyConq++;
    }
    if (myConq > 0) out[cell(C_MY_CONQ_SOLDIERS)] = Math.min(1, myConq / 5);
    if (enemyConq > 0) out[cell(C_ENEMY_CONQ_SOLDIERS)] = Math.min(1, enemyConq / 5);

    let attMinusDef: number;
    if (ownedByMe) attMinusDef = ownedSol - enemyConq;
    else if (ownedByEnemy) attMinusDef = myConq - ownedSol;
    else attMinusDef = myConq;
    if (attMinusDef !== 0) out[cell(C_ATT_MINUS_DEF)] = Math.max(-1, Math.min(1, attMinusDef / 5));

    if (enemyHqCoords.length > 0) {
      let dist = Infinity;
      for (const e of enemyHqCoords) dist = Math.min(dist, Math.abs(x - e.x) + Math.abs(y - e.y));
      out[cell(C_DIST_TO_ENEMY_HQ)] = Math.max(0, Math.min(1, 1 - dist / diameter));
    }
    if (enemyDevice) {
      const dist = Math.abs(x - enemyDevice.x) + Math.abs(y - enemyDevice.y);
      out[cell(C_DIST_TO_ENEMY_DEVICE)] = Math.max(0, Math.min(1, 1 - dist / diameter));
    }
  }

  // Reachability planes.
  for (const t of availableTilesFor(player, om)) {
    const c = t.getCoordinate();
    if (c.x() >= 0 && c.y() >= 0 && c.x() < w && c.y() < h) out[idx(C_MY_REACH, c.y(), c.x(), h, w)] = 1;
  }
  for (const op of enemies) {
    for (const t of availableTilesFor(op, om)) {
      const c = t.getCoordinate();
      if (c.x() >= 0 && c.y() >= 0 && c.x() < w && c.y() < h) out[idx(C_ENEMY_REACH, c.y(), c.x(), h, w)] = 1;
    }
  }

  // Broadcast budget planes.
  let maxEnemyBudget = 0;
  for (const op of enemies) maxEnemyBudget = Math.max(maxEnemyBudget, mobileBudget(op));
  if (maxEnemyBudget > 0) {
    const v = Math.min(1, maxEnemyBudget / 6);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[idx(C_ENEMY_BUDGET, y, x, h, w)] = v;
  }
  const myBudget = mobileBudget(player);
  if (myBudget > 0) {
    const v = Math.min(1, myBudget / 6);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[idx(C_MY_BUDGET, y, x, h, w)] = v;
  }

  // My HQ.
  const myHq = om.getHqTile(player);
  if (myHq) { const c = myHq.getCoordinate(); if (c.x() >= 0 && c.y() >= 0 && c.x() < w && c.y() < h) out[idx(C_MY_HQ, c.y(), c.x(), h, w)] = 1; }
  // Enemy HQs.
  for (const op of enemies) {
    const hq = om.getHqTile(op);
    if (hq) { const c = hq.getCoordinate(); if (c.x() >= 0 && c.y() >= 0 && c.x() < w && c.y() < h) out[idx(C_ENEMY_HQ, c.y(), c.x(), h, w)] = 1; }
  }
  // HQ-connected mask.
  for (const t of om.getHqConnectedTiles(player)) {
    const c = t.getCoordinate();
    if (c.x() >= 0 && c.y() >= 0 && c.x() < w && c.y() < h) out[idx(C_HQ_CONNECTED, c.y(), c.x(), h, w)] = 1;
  }

  return { planes: out, h, w };
}

// ---------------------------------------------------------------------------
// SpatialNet inference (board cache + per-candidate policy score).
// ---------------------------------------------------------------------------

export interface BoardCache {
  h: number;
  w: number;
  /** board_embed = tanh(conv2(tanh(conv1(planes)))) [+ residual]; (D,H,W). */
  boardEmbed: number[];
  /** GlobalAvgPool(boardEmbed); length D. */
  globalEmbed: number[];
  /** Per-state value-head scalar features (length value_scalar_dim), or []. */
  valueScalars: number[];
}

function clamp3(v: number): number {
  return v < -3 ? -3 : v > 3 ? 3 : v;
}

/** INTENT_DIM one-hot of a candidate's intent. */
export function intentOnehot(c: Candidate, intentDim: number): number[] {
  const v = new Array<number>(intentDim).fill(0);
  const i = c.intent as number;
  if (i >= 0 && i < intentDim) v[i] = 1;
  return v;
}

/** The (x,y) board target of a candidate, or null for Pass / no target. */
export function targetXY(c: Candidate): { x: number; y: number } | null {
  if (!c.target) return null;
  const co = c.target.getCoordinate();
  if (co.x() < 0 || co.y() < 0) return null;
  return { x: co.x(), y: co.y() };
}

/** Per-candidate local feature vector (port of cnn_train.rs cand_feat): the
 *  shared c.local (0..LOCAL_DIM-1) plus 2 CNN-only remaining-capacity features. */
export function candLocal(c: Candidate, player: PlayerBase, localDim: number): number[] {
  const local = c.local.slice(0, LOCAL_DIM);
  while (local.length < LOCAL_DIM) local.push(0);
  // index 16/17 — remaining soldier/unit cap, clamp3'd (matches cand_feat).
  if (localDim > LOCAL_DIM) local.push(clamp3(player.getFreeSoldierAmount() / 5));
  if (localDim > LOCAL_DIM + 1) local.push(clamp3(player.getFreeUnitAmount() / 5));
  while (local.length < localDim) local.push(0);
  return local;
}

export class SpatialNetTS {
  readonly d: number;
  constructor(public readonly w: SpatialWeights) {
    this.d = w.d;
  }

  /** Run the shared conv trunk + pool over a board. */
  forwardBoard(planes: number[], h: number, width: number, valueScalars: number[] = []): BoardCache {
    const conv1 = tanhForward(convForward(this.w.conv1, planes, h, width));
    let boardEmbed = tanhForward(convForward(this.w.conv2, conv1, h, width));
    // Optional residual block (champion sd4-az-002 has none).
    if (this.w.conv3) {
      const trunk2 = boardEmbed;
      const resAct = tanhForward(convForward(this.w.conv3, trunk2, h, width));
      boardEmbed = resAct.map((r, i) => r + trunk2[i]);
    }
    const globalEmbed = globalAvgPool(boardEmbed, this.d, h, width);
    return { h, w: width, boardEmbed, globalEmbed, valueScalars };
  }

  /**
   * Scalar value in [-1,1] from a cached board (mirror Rust `value_from` /
   * `value_forward`). value_input = global_embed (D) ⊕ value_scalars; then
   * Dense(value_d1) -> tanh -> Dense(value_d2) -> tanh. Returns 0 if the value
   * head is absent (policy-only weights).
   */
  valueFrom(cache: BoardCache): number {
    const vd1 = this.w.value_d1;
    const vd2 = this.w.value_d2;
    if (!vd1 || !vd2) return 0;
    const vsd = this.w.value_scalar_dim ?? 0;
    let input: number[];
    if (vsd === 0) {
      input = cache.globalEmbed;
    } else {
      input = new Array<number>(this.d + vsd);
      for (let i = 0; i < this.d; i++) input[i] = cache.globalEmbed[i];
      for (let i = 0; i < vsd; i++) input[this.d + i] = cache.valueScalars[i] ?? 0;
    }
    const h1 = tanhForward(denseForward(vd1, input));
    const outPre = denseForward(vd2, h1)[0];
    return Math.tanh(outPre);
  }

  /** Linear policy score for one candidate against a cached board. */
  scoreCandidate(
    cache: BoardCache,
    target: { x: number; y: number } | null,
    local: number[],
    intentOh: number[],
  ): number {
    const d = this.d;
    const input: number[] = new Array<number>(2 * d + local.length + intentOh.length);
    let k = 0;
    // target_embed: board_embed column at (x,y), or zeros for None.
    if (target) {
      for (let c = 0; c < d; c++) input[k++] = cache.boardEmbed[idx(c, target.y, target.x, cache.h, cache.w)];
    } else {
      for (let c = 0; c < d; c++) input[k++] = 0;
    }
    for (let c = 0; c < d; c++) input[k++] = cache.globalEmbed[c];
    for (let i = 0; i < local.length; i++) input[k++] = local[i];
    for (let i = 0; i < intentOh.length; i++) input[k++] = intentOh[i];
    const h1 = tanhForward(denseForward(this.w.policy_d1, input));
    return denseForward(this.w.policy_d2, h1)[0];
  }
}

/**
 * Greedy spatial policy: argmax of `scoreCandidate` over `cands` for the acting
 * `player`. Mirrors the deployed (non-MCTS) net-greedy turn loop in the Rust
 * `cnn_train.rs` completion path. Builds the board once, scores each candidate.
 * Returns the chosen candidate INDEX (0 for an empty list).
 */
export function selectSpatialIndex(
  net: SpatialNetTS,
  player: PlayerBase,
  om: ObjectManager,
  pm: PlayerManager,
  cands: Candidate[],
): number {
  if (cands.length === 0) return 0;
  if (cands.length === 1) return 0;
  const { planes, h, w } = boardPlanes(player, om, pm);
  const cache = net.forwardBoard(planes, h, w);
  const localDim = net.w.local_dim;
  const intentDim = net.w.intent_dim;
  let best = 0;
  let bestScore = -Infinity;
  for (let i = 0; i < cands.length; i++) {
    const c = cands[i];
    const s = net.scoreCandidate(cache, targetXY(c), candLocal(c, player, localDim), intentOnehot(c, intentDim));
    if (s > bestScore) { bestScore = s; best = i; }
  }
  return best;
}
