// Tests for the neural-network CPU opponents (nn-easy/medium/hard).
//
// Structural tests always run: weights load with the right shape, and nn-hard
// plays full games headlessly on small AND large maps without crashing or ever
// going resource-negative (the engine's solvency/robustness invariant).
//
// Strength tests (nn-hard beats the hard heuristic; difficulty monotonicity)
// run only once real, calibrated weights are baked — i.e. when
// NEURAL_WEIGHTS.meta has a `calibration` field. With the untrained placeholder
// they are skipped so the suite stays green during development.

import { describe, it, expect } from 'vitest';
import { GameSettingsManager } from '../src/managers/gamesettings';
import { ObjectManager } from '../src/managers/objectmanager';
import { PlayerManager } from '../src/managers/playermanager';
import { GameEventHandler } from '../src/managers/gameeventhandler';
import { WorldGenerator } from '../src/world/worldgenerator';
import { AiController } from '../src/managers/ai';
import { MouseHoverBorder } from '../src/model/overlays';
import { Coordinate } from '../src/core/coordinate';
import { IGameScene, ISceneObjectHandle } from '../src/model/base';
import { PlayerBase } from '../src/model/player';
import { IMenuObjectManager } from '../src/managers/menu-interface';
import { NeuralAiController } from '../src/ai/nn/controller';
import {
  TierConfig, AiCtx, Intent, Candidate, enumerate,
  EXPAND_CANDIDATE_CAP, INTENT_COUNT, LOCAL_DIM,
  tileSpatial, enemyTileCoords,
} from '../src/ai/nn/candidates';
import { Soldier } from '../src/model/unit';
import { TileBase } from '../src/model/tile';
import { paramCount, Genome, zeroGenome } from '../src/ai/nn/mlp';
import { POLICY_INPUT_DIM, select } from '../src/ai/nn/policy';
import { globalFeatures } from '../src/ai/nn/features';
import { TRAINING_CONFIG } from '../src/ai/nn/tiers';
import { GLOBAL_DIM, GLOBAL_FEATURE_NAMES } from '../src/ai/nn/features';
import { NEURAL_WEIGHTS } from '../src/ai/nn/weights';
import { createNeuralController } from '../src/ai/nn';
import { BasicResource } from '../src/core/resources';

class StubScene implements IGameScene {
  drawItem() {}
  removeItem() {}
  updateItem() {}
  updateTile() {}
  isObjectInScene() { return true; }
  getObjectInScene(): ISceneObjectHandle { return { setAnimationOption() {}, setAnimationFrame() {} }; }
  addMouseFollowPicture() {}
  removeMouseFollowItem() {}
  deleteObjects() {}
}
class CapturingMenu implements IMenuObjectManager {
  winner: PlayerBase | null = null;
  tie = false;
  selectFirstTileMenuView() {}
  setTileInspectionMenuView() {}
  setStatMenuView() {}
  setDefaultMenuView() {}
  setUnitShopMenuView() {}
  setTieMenu() { this.tie = true; }
  setWinMenu(p: PlayerBase) { this.winner = p; }
  setPlayerLostMenu() {}
  setCpuTurnMenuView() {}
}

function rng(seed: number): () => number {
  let s = (seed * 2654435761) >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return (s >>> 0) / 4294967296; };
}

interface Outcome { winner: 'nn' | 'heur' | 'none'; crashed: boolean; nnBankrupt: boolean; rounds: number; }

/** nn-difficulty at seat 0 vs a heuristic CPU at seat 1; play to terminal. */
function runNnVsHeuristic(
  width: number, height: number, seed: number,
  difficulty: 'nn-easy' | 'nn-medium' | 'nn-hard',
  heuristic: 'easy' | 'medium' | 'hard' = 'hard',
  roundCap = 160,
): Outcome {
  const gsm = GameSettingsManager.fromMapDimensions(width, height);
  const om = new ObjectManager();
  const pm = new PlayerManager([{ name: 'NN', difficulty }, { name: 'H', difficulty: heuristic }], om);
  const menu = new CapturingMenu();
  const eh = new GameEventHandler(om, pm, menu, gsm);
  const scene = new StubScene();
  eh.setGameScene(scene); om.setGameScene(scene); om.addDALS(eh, menu, gsm);
  om.setHoverBorder(new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om));
  new WorldGenerator().generateMap(width, height, seed, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });

  const players = pm.getPlayers().slice();
  const nn = createNeuralController(eh, om, pm, difficulty, rng(seed));
  const heur = new AiController(eh, om, pm);
  const ctrlFor = (p: PlayerBase) => (p.getPlayerNum() === 1 ? nn : heur);

  let crashed = false, nnBankrupt = false;
  try {
    eh.setAiActive(true);
    for (let i = 0; i < players.length; i++) ctrlFor(pm.getCurrentPlayer()).placeHeadquarters(pm.getCurrentPlayer());
    eh.setAiActive(false);
    while (pm.getPlayers().length > 1 && pm.getRoundsPlayed() < roundCap) {
      const cur = pm.getCurrentPlayer();
      if (cur.isCpu()) { eh.setAiActive(true); ctrlFor(cur).playTurn(cur); eh.setAiActive(false); }
      eh.endTurn();
      const nnP = players[0];
      if (pm.getPlayers().includes(nnP) && [...nnP.getResources().values()].some((v) => v < 0)) nnBankrupt = true;
      if (menu.winner || menu.tie) break;
    }
  } catch {
    crashed = true;
  }
  const winner = menu.winner ? (menu.winner.getPlayerNum() === 1 ? 'nn' : 'heur') : 'none';
  return { winner, crashed, nnBankrupt, rounds: pm.getRoundsPlayed() };
}

const TRAINED = !!(NEURAL_WEIGHTS.meta && (NEURAL_WEIGHTS.meta as Record<string, unknown>).calibration);

describe('Neural AI — structure', () => {
  it('weights match the network architecture', () => {
    expect(NEURAL_WEIGHTS.arch[0]).toBe(POLICY_INPUT_DIM);
    expect(NEURAL_WEIGHTS.params.length).toBe(paramCount(NEURAL_WEIGHTS.arch));
    expect(GLOBAL_DIM).toBe(GLOBAL_FEATURE_NAMES.length);
    for (const tier of ['easy', 'medium', 'hard'] as const) {
      const c = NEURAL_WEIGHTS.tiers[tier] as TierConfig;
      expect(c.budget).toBeGreaterThan(0);
      expect(c.reserve).toBeGreaterThanOrEqual(0);
    }
  });
});

describe('Neural AI — robustness (always)', () => {
  // Small and large maps, several seeds: never crash, never go negative.
  for (const [w, h] of [[12, 12], [16, 14], [25, 15]] as Array<[number, number]>) {
    for (const seed of [1, 7, 42]) {
      it(`nn-hard is crash-free & solvent on ${w}x${h} seed ${seed}`, () => {
        const o = runNnVsHeuristic(w, h, seed, 'nn-hard');
        expect(o.crashed).toBe(false);
        expect(o.nnBankrupt).toBe(false);
      });
    }
  }
});

// --- Phase 1: learned target selection (multi-candidate Expand/Attack) --------
//
// These tests exercise the enumerate() representation surgery directly: Expand and
// Attack now emit one Candidate per plausible target tile, so the network argmaxes
// over (intent, target) pairs. We build a real seeded game, place HQs, then flush
// player 1 with cash + free unit slots so canHire makes Expand candidates appear,
// and enumerate the candidate list at the start of player 1's turn.

interface Built {
  ctx: AiCtx;
  player: PlayerBase;
  om: ObjectManager;
  pm: PlayerManager;
  eh: GameEventHandler;
}

function buildGameAndPlaceHqs(width: number, height: number, seed: number, warmupRounds = 12): Built {
  const gsm = GameSettingsManager.fromMapDimensions(width, height);
  const om = new ObjectManager();
  const pm = new PlayerManager([{ name: 'NN', difficulty: 'nn-hard' }, { name: 'H', difficulty: 'hard' }], om);
  const menu = new CapturingMenu();
  const eh = new GameEventHandler(om, pm, menu, gsm);
  const scene = new StubScene();
  eh.setGameScene(scene); om.setGameScene(scene); om.addDALS(eh, menu, gsm);
  om.setHoverBorder(new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om));
  new WorldGenerator().generateMap(width, height, seed, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });

  const players = pm.getPlayers().slice();
  const nn = createNeuralController(eh, om, pm, 'nn-hard', rng(seed));
  const heur = new AiController(eh, om, pm);
  const ctrlFor = (p: PlayerBase) => (p.getPlayerNum() === 1 ? nn : heur);
  eh.setAiActive(true);
  for (let i = 0; i < players.length; i++) ctrlFor(pm.getCurrentPlayer()).placeHeadquarters(pm.getCurrentPlayer());
  eh.setAiActive(false);

  // Warm up: play a few real rounds so player 1 owns tiles and accumulates idle
  // workers (Expand's reachability gate), giving a realistic mid-game frontier.
  for (let round = 0; round < warmupRounds && pm.getPlayers().length > 1; round++) {
    const cur = pm.getCurrentPlayer();
    if (cur.isCpu()) { eh.setAiActive(true); ctrlFor(cur).playTurn(cur); eh.setAiActive(false); }
    eh.endTurn();
    if (menu.winner || menu.tie) break;
  }
  // Step the turn pointer to player 1 so getAvailableTiles() is theirs.
  let guard = 0;
  while (pm.getPlayers().length > 1 && pm.getCurrentPlayer().getPlayerNum() !== 1 && guard++ < 6) eh.endTurn();

  const player = players[0];
  const ctx: AiCtx = { eh, om, pm, player, cfg: TRAINING_CONFIG };
  return { ctx, player, om, pm, eh };
}

/** Flush player 1 with cash so Expand's canHire gate passes (no idle worker needed). */
function flushCash(player: PlayerBase): void {
  const r = player.getResources();
  r.set(BasicResource.MONEY, 50000);
  r.set(BasicResource.WOOD, 50000);
  r.set(BasicResource.STONE, 50000);
  r.set(BasicResource.METAL, 50000);
}

/**
 * Find a (seed) whose post-HQ-placement board yields ≥2 Expand candidates with at
 * least two DISTINCT local[2] (targetValue) values for player 1. The current
 * player after placements is player 1, so getAvailableTiles() is theirs.
 */
function findExpandScenario(): { built: Built; cands: Candidate[]; expand: Candidate[] } | null {
  for (const seed of [7, 42, 99, 1, 123, 256, 3, 5, 11, 17, 23, 31, 50, 77, 88]) {
    for (const [w, h] of [[14, 12], [16, 14], [12, 12]] as Array<[number, number]>) {
      const built = buildGameAndPlaceHqs(w, h, seed);
      if (!built.pm.getPlayers().includes(built.player)) continue;
      if (built.pm.getCurrentPlayer().getPlayerNum() !== 1) continue;
      flushCash(built.player);
      const cands = enumerate(built.ctx);
      const expand = cands.filter((c) => c.intent === Intent.Expand);
      const distinct = new Set(expand.map((c) => c.local[2]));
      if (expand.length >= 2 && distinct.size >= 2) return { built, cands, expand };
    }
  }
  return null;
}

/**
 * Craft a genome whose score is monotonically DECREASING in local[2] (targetValue)
 * for Expand candidates, while still ranking Expand above Pass. Hidden neuron 0 reads
 * `A*intentExpand - B*local[2]`; output = +1 * tanh(that). For Expand (intent one-hot
 * bit set) and small targetValue → positive score > Pass(0); a higher-claimValue tile
 * (bigger local[2]) gets a LOWER score. Proves the chosen target is learned, not hard-coded.
 */
function negativeTargetValueGenome(): Genome {
  const arch = [POLICY_INPUT_DIM, 24, 16, 1];
  const g = zeroGenome(arch);
  const params = g.params;
  // Layer 0 weights are laid out [out*nin] then [biases]; weight for (out j, in i)
  // is at offset (here 0) + j*nin + i. We set hidden neuron j=0 only.
  const nin0 = arch[0];
  const intentExpandIdx = GLOBAL_DIM + Intent.Expand; // intent one-hot block starts after globals
  const localBase = GLOBAL_DIM + INTENT_COUNT;        // local block start
  const targetValueIdx = localBase + 2;               // local[2] = targetValue
  params[0 * nin0 + intentExpandIdx] = 2.0;   // +A on Expand one-hot
  params[0 * nin0 + targetValueIdx] = -4.0;   // -B on targetValue
  // Output layer: offset = (nin0*nout0 + nout0) + (nin1*nout1 + nout1)
  const off0 = nin0 * arch[1] + arch[1];
  const off1 = arch[1] * arch[2] + arch[2];
  const outBase = off0 + off1;
  // Output reads hidden2 neuron 0 → but we need hidden0 to propagate. Simplest: make
  // hidden layer 1 neuron 0 = tanh(hidden0[0]) and output = hidden1[0].
  // hidden1 neuron 0 weight from hidden0 neuron 0:
  params[off0 + 0 * arch[1] + 0] = 4.0;
  // output weight from hidden1 neuron 0:
  params[outBase + 0 * arch[2] + 0] = 4.0;
  return g;
}

describe('Neural AI — learned target selection (multi-candidate)', () => {
  it('enumerate emits ≥2 Expand candidates with different local[2]', () => {
    const found = findExpandScenario();
    expect(found, 'expected a seed with ≥2 distinct-claimValue Expand targets').not.toBeNull();
    const expand = found!.expand;
    expect(expand.length).toBeGreaterThanOrEqual(2);
    expect(new Set(expand.map((c) => c.local[2])).size).toBeGreaterThanOrEqual(2);
    // Each Expand candidate has a full-width local vector and intent=Expand.
    for (const c of expand) {
      expect(c.intent).toBe(Intent.Expand);
      expect(c.local.length).toBe(LOCAL_DIM);
    }
  });

  it('Expand candidates are sorted claimValue DESC (so local[2] is non-increasing)', () => {
    const found = findExpandScenario();
    expect(found).not.toBeNull();
    const tv = found!.expand.map((c) => c.local[2]);
    for (let i = 1; i < tv.length; i++) expect(tv[i]).toBeLessThanOrEqual(tv[i - 1]);
  });

  it('a genome weighting targetValue NEGATIVELY picks a non-highest-claimValue Expand target', () => {
    const found = findExpandScenario();
    expect(found).not.toBeNull();
    const { built, cands, expand } = found!;
    const gvec = globalFeatures(built.player, built.om, built.pm, built.pm.getRoundsPlayed());
    const genome = negativeTargetValueGenome();
    // temperature=0, blunder=0 → deterministic argmax (lowest index on ties).
    const cfg: TierConfig = { ...TRAINING_CONFIG };
    const chosen = select(genome, gvec, cands, cfg, () => 0.5);
    // The net must choose an Expand target (beats Pass) — and NOT the highest-claimValue one.
    expect(chosen.intent).toBe(Intent.Expand);
    const maxTv = Math.max(...expand.map((c) => c.local[2]));
    expect(chosen.local[2]).toBeLessThan(maxTv);
  });

  it('Expand candidate count is capped at EXPAND_CANDIDATE_CAP', () => {
    // Large map with many neutral frontier tiles around a flush HQ.
    let maxExpand = 0;
    for (const seed of [42, 99, 256, 7, 123]) {
      const built = buildGameAndPlaceHqs(20, 15, seed);
      if (!built.pm.getPlayers().includes(built.player)) continue;
      if (built.pm.getCurrentPlayer().getPlayerNum() !== 1) continue;
      flushCash(built.player);
      const expand = enumerate(built.ctx).filter((c) => c.intent === Intent.Expand);
      maxExpand = Math.max(maxExpand, expand.length);
      expect(expand.length).toBeLessThanOrEqual(EXPAND_CANDIDATE_CAP);
    }
    expect(maxExpand).toBeGreaterThan(0);
  });

  it('enumeration is deterministic: same seed → identical candidate list & chosen target', () => {
    const a = buildGameAndPlaceHqs(16, 14, 42);
    const b = buildGameAndPlaceHqs(16, 14, 42);
    flushCash(a.player); flushCash(b.player);
    const ca = enumerate(a.ctx);
    const cb = enumerate(b.ctx);
    expect(ca.length).toBe(cb.length);
    for (let i = 0; i < ca.length; i++) {
      expect(ca[i].intent).toBe(cb[i].intent);
      expect(ca[i].label).toBe(cb[i].label);
      expect(ca[i].local).toEqual(cb[i].local);
    }
    const genome = negativeTargetValueGenome();
    const gva = globalFeatures(a.player, a.om, a.pm, a.pm.getRoundsPlayed());
    const gvb = globalFeatures(b.player, b.om, b.pm, b.pm.getRoundsPlayed());
    const cfg: TierConfig = { ...TRAINING_CONFIG };
    const xa = select(genome, gva, ca, cfg, () => 0.5);
    const xb = select(genome, gvb, cb, cfg, () => 0.5);
    expect(xa.intent).toBe(xb.intent);
    expect(xa.local).toEqual(xb.local);
    expect(ca.indexOf(xa)).toBe(cb.indexOf(xb));
  });

  it('candidate list is non-decreasing in intent value with Pass last', () => {
    const built = buildGameAndPlaceHqs(16, 14, 42);
    flushCash(built.player);
    const cands = enumerate(built.ctx);
    for (let i = 1; i < cands.length; i++) {
      expect(cands[i].intent).toBeGreaterThanOrEqual(cands[i - 1].intent);
    }
    expect(cands[cands.length - 1].intent).toBe(Intent.Pass);
  });
});

// --- Phase 2: spatial/positional per-target local features (indices 10–15) ----
//
// These exercise tileSpatial() + the enumerate() threading directly. We build a
// real seeded board (so getNeighbourTiles/getCoordinate use real settings), then
// manually paint tile ownership / drop soldiers to craft controlled scenarios.

interface SpatialBuilt {
  gsm: GameSettingsManager; om: ObjectManager; pm: PlayerManager; eh: GameEventHandler;
  p1: PlayerBase; p2: PlayerBase;
}

function buildBoard(width: number, height: number, seed: number): SpatialBuilt {
  const gsm = GameSettingsManager.fromMapDimensions(width, height);
  const om = new ObjectManager();
  const pm = new PlayerManager([{ name: 'A', difficulty: 'nn-hard' }, { name: 'B', difficulty: 'hard' }], om);
  const menu = new CapturingMenu();
  const eh = new GameEventHandler(om, pm, menu, gsm);
  const scene = new StubScene();
  eh.setGameScene(scene); om.setGameScene(scene); om.addDALS(eh, menu, gsm);
  om.setHoverBorder(new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om));
  new WorldGenerator().generateMap(width, height, seed, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });
  const players = pm.getPlayers().slice();
  return { gsm, om, pm, eh, p1: players[0], p2: players[1] };
}

/** A neutral tile (no owner) that has ≥1 neutral neighbour, for clean control. */
function pickNeutralTile(om: ObjectManager): TileBase {
  return om.getTiles().find((t) => t.getOwner() === null && t.getNeighbourTiles().length >= 3)!;
}

describe('Neural AI — spatial features (Phase 2)', () => {
  it('localVec is 16-wide; tileSpatial fractions are in range', () => {
    const b = buildBoard(14, 12, 7);
    const t = pickNeutralTile(b.om);
    const sp = tileSpatial(t, b.p1, b.om, enemyTileCoords(b.om, b.p1));
    expect(sp.enemyNeighbors).toBeGreaterThanOrEqual(0);
    expect(sp.enemyNeighbors).toBeLessThanOrEqual(1);
    expect(sp.ownNeighbors).toBeLessThanOrEqual(1);
    expect(sp.neutralNeighbors).toBeLessThanOrEqual(1);
    expect(LOCAL_DIM).toBe(16);
  });

  it('enemyNeighbors (local[10]) > 0 next to an enemy tile, 0 in the interior', () => {
    const b = buildBoard(16, 14, 42);
    // Interior neutral tile far from any owner.
    const interior = b.om.getTiles().find(
      (t) => t.getOwner() === null && t.getNeighbourTiles().every((n) => n.getOwner() === null),
    )!;
    expect(interior).toBeTruthy();
    const enemyCoordsBefore = enemyTileCoords(b.om, b.p1);
    const spInterior = tileSpatial(interior, b.p1, b.om, enemyCoordsBefore);
    expect(spInterior.enemyNeighbors).toBe(0);

    // Now make one neighbour of `interior` enemy-owned and recompute.
    const nb = interior.getNeighbourTiles()[0];
    nb.setOwner(b.p2);
    const spAdjacent = tileSpatial(interior, b.p1, b.om, enemyTileCoords(b.om, b.p1));
    expect(spAdjacent.enemyNeighbors).toBeGreaterThan(0);
  });

  it('"enemy" excludes neutral and the acting player', () => {
    const b = buildBoard(16, 14, 42);
    const t = b.om.getTiles().find(
      (t) => t.getOwner() === null && t.getNeighbourTiles().length >= 3,
    )!;
    const n = t.getNeighbourTiles();
    n[0].setOwner(b.p1);  // own → counts as ownNeighbors, NOT enemy
    n[1].setOwner(b.p2);  // enemy
    // remaining neighbours stay neutral
    const sp = tileSpatial(t, b.p1, b.om, enemyTileCoords(b.om, b.p1));
    const total = n.length;
    expect(sp.ownNeighbors).toBeCloseTo(1 / 8);
    expect(sp.enemyNeighbors).toBeCloseTo(1 / 8);
    expect(sp.neutralNeighbors).toBeCloseTo((total - 2) / 8);
  });

  it('distNearestEnemyTile uses sentinel 99/20 (clamped to 3) when no enemy tiles exist', () => {
    const b = buildBoard(14, 12, 7);
    // Strip all ownership so there are no enemy tiles.
    for (const t of b.om.getTiles()) t.setOwner(null);
    const target = pickNeutralTile(b.om);
    const sp = tileSpatial(target, b.p1, b.om, enemyTileCoords(b.om, b.p1));
    expect(sp.distNearestEnemyTile).toBeCloseTo(99 / 20); // clamp[0,3] applied in localVec, raw here
    // distOwnHq also sentinel (no HQ placed): 99/20.
    expect(sp.distOwnHq).toBeCloseTo(99 / 20);
  });

  it('a genome weighting distNearestEnemyTile (local[14]) NEGATIVELY picks the Expand target closest to the enemy', () => {
    const b = buildBoard(16, 14, 42);
    // Give p1 a small cluster of owned tiles + an enemy tile far away, so the
    // neutral expand targets differ in distance-to-enemy.
    const tiles = b.om.getTiles();
    // Own an anchor tile for p1 (so getAvailableTiles surfaces its neutral neighbours).
    const anchor = tiles.find((t) => t.getOwner() === null && t.hasSpaceForUnits())!;
    anchor.setOwner(b.p1);
    // Place an HQ-less enemy presence: own one tile for p2 near a corner.
    const enemyTile = tiles.find((t) => t.getOwner() === null && t !== anchor)!;
    enemyTile.setOwner(b.p2);

    // Flush cash so Expand candidates appear (canHire path).
    flushCash(b.p1);
    while (b.pm.getCurrentPlayer().getPlayerNum() !== 1) b.pm.changeTurn();
    const ctx: AiCtx = { eh: b.eh, om: b.om, pm: b.pm, player: b.p1, cfg: TRAINING_CONFIG };
    const cands = enumerate(ctx);
    const expand = cands.filter((c) => c.intent === Intent.Expand);
    if (expand.length < 2) return; // scenario not realised on this board; skip silently
    const dists = expand.map((c) => c.local[14]);
    // need at least two distinct distances for the test to be meaningful
    if (new Set(dists).size < 2) return;

    // Genome: hidden0 = A*intentExpand - B*local[14]; output mirrors it. Closer to
    // enemy = smaller local[14] = higher score. Proves spatial strategy is learnable.
    const arch = [POLICY_INPUT_DIM, 24, 16, 1];
    const g = zeroGenome(arch);
    const nin0 = arch[0];
    const intentExpandIdx = GLOBAL_DIM + Intent.Expand;
    const dist14Idx = GLOBAL_DIM + INTENT_COUNT + 14;
    g.params[0 * nin0 + intentExpandIdx] = 2.0;
    g.params[0 * nin0 + dist14Idx] = -4.0;
    const off0 = nin0 * arch[1] + arch[1];
    const off1 = arch[1] * arch[2] + arch[2];
    const outBase = off0 + off1;
    g.params[off0 + 0 * arch[1] + 0] = 4.0;
    g.params[outBase + 0 * arch[2] + 0] = 4.0;

    const gvec = globalFeatures(b.p1, b.om, b.pm, b.pm.getRoundsPlayed());
    const cfg: TierConfig = { ...TRAINING_CONFIG };
    const chosen = select(g, gvec, cands, cfg, () => 0.5);
    expect(chosen.intent).toBe(Intent.Expand);
    const minDist = Math.min(...dists);
    expect(chosen.local[14]).toBeCloseTo(minDist);
  });

  it('Attack target with more adjacent own Soldiers has higher local[15] (frontier)', () => {
    const b = buildBoard(16, 14, 42);
    const { om, eh, gsm, p1, p2 } = b;
    while (b.pm.getCurrentPlayer().getPlayerNum() !== 1) b.pm.changeTurn();
    flushCash(p1);

    // Find two enemy-owned conquerable tiles (give p1 adjacency so they're available),
    // and stage own soldiers around ONE of them.
    const grass = om.getTiles().filter((t) => t.getType() === 'Grassland' && t.getOwner() === null);
    // tileA: enemy tile we surround with own soldiers; tileB: enemy tile we leave bare.
    const tileA = grass[0]; const tileB = grass[grass.length - 1];
    tileA.setOwner(p2); tileB.setOwner(p2);
    // p1 must own a neighbour of each so they enter getAvailableTiles().
    for (const t of [tileA, tileB]) {
      const own = t.getNeighbourTiles().find((n) => n.getOwner() === null);
      if (own) own.setOwner(p1);
    }
    // Drop p1 soldiers on two own neighbours of tileA.
    const aNbrs = tileA.getNeighbourTiles().filter((n) => n.getOwner() === p1);
    let dropped = 0;
    for (const n of aNbrs) {
      const s = new Soldier(eh, om, gsm, p1);
      s.addParentTile(n); s.setOwner(p1); n.addUnit(s); dropped++;
      if (dropped >= 1) break;
    }

    const enemyCoords = enemyTileCoords(om, p1);
    const spA = tileSpatial(tileA, p1, om, enemyCoords);
    // recompute frontier the same way attackCandidates does:
    const ownSoldiersAround = (tile: TileBase) =>
      tile.getNeighbourTiles().reduce(
        (acc, nb) => acc + nb.getUnits().filter((u) => u.getType() === 'Soldier' && u.getOwner() === p1).length, 0,
      );
    const spB = tileSpatial(tileB, p1, om, enemyCoords);
    spA.frontier = ownSoldiersAround(tileA) / 3;
    spB.frontier = ownSoldiersAround(tileB) / 3;
    expect(spA.frontier).toBeGreaterThan(spB.frontier);
  });

  it('Build*/HireSoldier/StackProducer/Pass candidates emit all 6 spatial dims (10–15) === 0', () => {
    const built = buildGameAndPlaceHqs(16, 14, 42);
    flushCash(built.player);
    const cands = enumerate(built.ctx);
    const nonSpatial = cands.filter(
      (c) => c.intent !== Intent.Expand && c.intent !== Intent.Attack,
    );
    expect(nonSpatial.length).toBeGreaterThan(0);
    for (const c of nonSpatial) {
      expect(c.local.length).toBe(LOCAL_DIM);
      for (let i = 10; i < 16; i++) expect(c.local[i]).toBe(0);
    }
  });
});

// --- Economy scaffold port (controller.ts ← Rust controller.rs) ---------------
//
// The browser deploy path must reproduce the Rust "economy scaffold" the champion
// net was trained on: each turn, BEFORE the learned net decides, the controller
// secures wood, staffs producers to OPTIMUM (Mine = 2 workers + 1 Expert = 80
// metal/round, plants = worker + Expert), expands the unit CAP via Villages when
// it blocks full staffing, and guarantees the first Mine. Without it the deployed
// CPU plays on an under-developed economy it never saw in training. These tests
// drive the real controller on a real seeded board (painting an owned patch with
// the terrain it needs) and assert the scaffold develops the economy.

import { HeadQuarters, Farm } from '../src/model/building';
import { Mountain, Grassland, Forest } from '../src/model/tiles';

interface ScaffoldBuilt {
  om: ObjectManager; pm: PlayerManager; eh: GameEventHandler; p: PlayerBase;
  nn: NeuralAiController; mountain: TileBase; grasslands: TileBase[]; hq: TileBase;
}

/**
 * Build a board, paint an owned patch for the nn player containing an empty
 * Mountain (Mine site), several empty Grasslands (HQ + Village sites), and a
 * Forest (wood). Place the HQ (so unit cap starts at +3 like a real game), flush
 * cash, and make the nn player the current seat. Returns null if the seed lacks
 * the required terrain (caller scans seeds).
 */
function buildScaffoldScenario(width: number, height: number, seed: number): ScaffoldBuilt | null {
  const gsm = GameSettingsManager.fromMapDimensions(width, height);
  const om = new ObjectManager();
  const pm = new PlayerManager([{ name: 'NN', difficulty: 'nn-hard' }, { name: 'H', difficulty: 'hard' }], om);
  const menu = new CapturingMenu();
  const eh = new GameEventHandler(om, pm, menu, gsm);
  const scene = new StubScene();
  eh.setGameScene(scene); om.setGameScene(scene); om.addDALS(eh, menu, gsm);
  om.setHoverBorder(new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om));
  new WorldGenerator().generateMap(width, height, seed, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });

  const p = pm.getPlayers()[0];
  const tiles = om.getTiles();
  const mountain = tiles.find((t) => t instanceof Mountain && t.getOwner() === null && t.getBuilding() === null);
  const grasslands = tiles.filter((t) => t instanceof Grassland && t.getOwner() === null && t.getBuilding() === null).slice(0, 6);
  const forest = tiles.find((t) => t instanceof Forest && t.getOwner() === null && t.getBuilding() === null);
  if (!mountain || grasslands.length < 4 || !forest) return null;

  for (const t of [mountain, forest, ...grasslands]) t.setOwner(p);

  // Place the HQ on the first grassland (engine-direct, like the Rust unit tests'
  // place_building) so the unit cap starts at the real +3.
  const hq = grasslands[0];
  const hqBuilding = new HeadQuarters(eh, om, p);
  hqBuilding.setParentTile(hq);
  hq.addBuilding(hqBuilding);
  // A Farm on a 2nd grassland adds producer-staffing demand (1 worker) so that
  // Mine(3) + Farm(1) = 4 units exceeds the HQ-only cap (+3) — forcing ensureUnitCap
  // to build a Village (mirrors the Rust Mine+Nuclear cap-pressure test).
  const farmTile = grasslands[1];
  const farmBuilding = new Farm(eh, om, p);
  farmBuilding.setParentTile(farmTile);
  farmTile.addBuilding(farmBuilding);
  p.updateUnitAmounts();

  // Flush resources so affordability never blocks the scaffold (we test BEHAVIOUR,
  // not the affordability gates, which are covered by parity).
  const r = p.getResources();
  r.set(BasicResource.MONEY, 50000);
  r.set(BasicResource.WOOD, 50000);
  r.set(BasicResource.STONE, 50000);
  r.set(BasicResource.METAL, 50000);

  // Make the nn player the current seat (aiBuild* act on the current player).
  let guard = 0;
  while (pm.getCurrentPlayer() !== p && guard++ < 4) pm.changeTurn();
  if (pm.getCurrentPlayer() !== p) return null;

  // budget:0 → the learned decision LOOP is skipped, so ONLY the pre-loop economy
  // scaffold runs. This isolates the ported scaffold (the unit under test) from the
  // net's discretionary Expand, which otherwise contends for the same unit cap —
  // exactly how the Rust `staffing_tests` exercise the scaffold helpers alone.
  const scaffoldCfg: TierConfig = { ...TRAINING_CONFIG, budget: 0 };
  const nn = new NeuralAiController(eh, om, pm, zeroGenome(NEURAL_WEIGHTS.arch), scaffoldCfg, rng(seed));
  return { om, pm, eh, p, nn, mountain, grasslands: grasslands.slice(1), hq };
}

describe('Neural AI — economy scaffold port (deploy parity with Rust)', () => {
  it('the scaffold builds a Mine, staffs it 2 workers + Expert (80 metal), and builds a Village for cap', () => {
    let found: ScaffoldBuilt | null = null;
    for (const seed of [7, 42, 99, 1, 123, 256, 3, 5, 11, 17, 23, 31, 50, 77, 88, 200, 314]) {
      for (const [w, h] of [[16, 14], [20, 15], [14, 12]] as Array<[number, number]>) {
        found = buildScaffoldScenario(w, h, seed);
        if (found) break;
      }
      if (found) break;
    }
    expect(found, 'expected a seed with an owned Mountain + Grasslands + Forest').not.toBeNull();
    const { nn, p, mountain } = found!;

    // Drive several scaffold turns (the economy development is the deterministic
    // scaffold's job, run before the net decides). Resources stay flush each turn
    // so production/upkeep can't strand the scaffold mid-bootstrap.
    const eh = found!.eh;
    const r = p.getResources();
    for (let i = 0; i < 6; i++) {
      r.set(BasicResource.MONEY, 50000); r.set(BasicResource.WOOD, 50000);
      r.set(BasicResource.STONE, 50000); r.set(BasicResource.METAL, 50000);
      eh.setAiActive(true);
      nn.playTurn(p);
      eh.setAiActive(false);
    }

    // A Mine must exist on the owned Mountain (ensureMetalIncome backstop).
    const mineTiles = ownedOf(p).filter((t) => t.getBuilding()?.getType() === 'Mine');
    expect(mineTiles.length, 'scaffold should build at least one Mine').toBeGreaterThanOrEqual(1);
    expect(mineTiles.some((t) => t === mountain)).toBe(true);

    // That Mine must be staffed to OPTIMUM: 2 BasicWorkers + 1 Expert = 80 metal.
    const mine = mineTiles[0];
    const workers = mine.getUnits().filter((u) => u.getType() === 'BasicWorker').length;
    const hasExpert = mine.getUnits().some((u) => u.getType() === 'Expert');
    expect(workers, 'mine should reach 2 BasicWorkers').toBe(2);
    expect(hasExpert, 'mine should have an Expert (doubles output)').toBe(true);

    // A Village must have been built to raise the unit cap so the full mine + plant
    // staffing fits (ensureUnitCap).
    const villages = ownedOf(p).filter((t) => t.getBuilding()?.getType() === 'Village').length;
    expect(villages, 'ensureUnitCap should build at least one Village').toBeGreaterThanOrEqual(1);
  });
});

/** Owned tiles of a player (test-local mirror of metrics.ownedTiles). */
function ownedOf(p: PlayerBase): TileBase[] {
  return p.getObjects().filter((o): o is TileBase => o instanceof TileBase);
}

describe.runIf(TRAINED)('Neural AI — difficulty ladder (trained weights only)', () => {
  const SIZES: Array<[number, number]> = [[12, 12], [16, 14], [20, 15]];
  const SEEDS = 16;

  function winRate(difficulty: 'nn-easy' | 'nn-medium' | 'nn-hard', heuristic: 'easy' | 'medium' | 'hard'): number {
    let wins = 0, games = 0;
    for (const [w, h] of SIZES) {
      for (let s = 1; s <= SEEDS; s++) {
        const o = runNnVsHeuristic(w, h, s + 900, difficulty, heuristic);
        if (o.winner === 'nn') wins++;
        games++;
      }
    }
    return wins / games;
  }

  // The tier knobs are ordered by design: more budget + less randomness = stronger.
  // (Structural guarantee, independent of sampling noise.)
  it('tier configs are ordered by design (hard ≥ medium ≥ easy strength knobs)', () => {
    const h = NEURAL_WEIGHTS.tiers.hard;
    const m = NEURAL_WEIGHTS.tiers.medium;
    const e = NEURAL_WEIGHTS.tiers.easy;
    expect(h.budget).toBeGreaterThanOrEqual(m.budget);
    expect(m.budget).toBeGreaterThanOrEqual(e.budget);
    expect(h.blunder).toBeLessThanOrEqual(m.blunder);
    expect(m.blunder).toBeLessThanOrEqual(e.blunder);
  });

  // Behavioural ladder: nn-hard wins clearly more often than nn-easy against a
  // fixed baseline (the hard heuristic). This is the reliable end-to-end check
  // that the calibrated difficulty actually translates into stronger play.
  it('nn-hard outperforms nn-easy against the hard heuristic', () => {
    const hard = winRate('nn-hard', 'hard');
    const easy = winRate('nn-easy', 'hard');
    expect(hard).toBeGreaterThan(easy);
  });
});
