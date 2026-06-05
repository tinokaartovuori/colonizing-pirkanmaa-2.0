// Save/restore the in-progress game to localStorage so a browser refresh doesn't
// lose it. The map *terrain* is deterministic from the seed (the MSVCRT RNG in
// world generation), so we only persist the mutable state on top of it: who owns
// each tile, the buildings and their state, the units, every player's resources and
// the turn bookkeeping. On load we regenerate the terrain from the seed and then
// re-apply this snapshot (see GameEventHandler.restoreSnapshot).

import type { ObjectManager } from './objectmanager';
import type { PlayerManager } from './playermanager';
import type { PlayerBase, Difficulty } from '../model/player';
import { Forest } from '../model/tiles';
import { Farm, HeadQuarters, StrangeDevice } from '../model/building';
import { BasicResource } from '../core/resources';

const KEY = 'cp-save-v1';

export interface UnitSnap {
  type: string;
  owner: number;
}
export interface BuildingSnap {
  type: string;
  owner: number | null;
  growthPhase?: number;
  conquered?: boolean;
  /** Strange Device: rounds left on its win countdown. */
  countdown?: number;
}
export interface TileSnap {
  x: number;
  y: number;
  owner: number | null;
  b?: BuildingSnap;
  forest?: { wood: number; stumps: number };
  units?: UnitSnap[];
  conq?: UnitSnap[];
}
export interface GameSnapshot {
  v: 1;
  settings: { width: number; height: number; seed: number; players: { name: string; difficulty: Difficulty }[] };
  currentPlayerNum: number;
  roundsPlayed: number;
  lostPlayerNums: number[];
  resources: Record<number, [number, number, number, number]>;
  tiles: TileSnap[];
}

const resTuple = (p: PlayerBase): [number, number, number, number] => {
  const r = p.getResources();
  return [
    r.get(BasicResource.MONEY) ?? 0,
    r.get(BasicResource.WOOD) ?? 0,
    r.get(BasicResource.STONE) ?? 0,
    r.get(BasicResource.METAL) ?? 0,
  ];
};

/** Capture the current game state into a serialisable snapshot. */
export function buildSnapshot(
  om: ObjectManager,
  pm: PlayerManager,
  raw: { width: number; height: number; seed: number },
): GameSnapshot {
  const allPlayers = [...pm.getPlayers(), ...pm.getLostPlayers()].sort((a, b) => a.getPlayerNum() - b.getPlayerNum());

  const resources: Record<number, [number, number, number, number]> = {};
  for (const p of allPlayers) resources[p.getPlayerNum()] = resTuple(p);

  const tiles: TileSnap[] = [];
  for (const tile of om.getTiles()) {
    const owner = tile.getOwner();
    const building = tile.getBuilding();
    const units = tile.getUnits();
    const conq = tile.getConqueringUnits();
    const forestState = tile instanceof Forest ? tile.getHarvestState() : null;
    const harvested = forestState && (forestState.woodLeft !== 600 || forestState.stumps !== 0);

    // Skip pristine, unowned, empty tiles — they're fully reproduced by terrain gen.
    if (owner === null && !building && units.length === 0 && conq.length === 0 && !harvested) continue;

    const c = tile.getCoordinate();
    const snap: TileSnap = { x: c.x(), y: c.y(), owner: owner ? owner.getPlayerNum() : null };
    if (building) {
      const b: BuildingSnap = { type: building.getType(), owner: building.getOwner()?.getPlayerNum() ?? null };
      if (building.getType() === 'Farm') b.growthPhase = (building as Farm).getGrowthPhase();
      if (building.getType() === 'Headquarters' && (building as HeadQuarters).isConquered()) b.conquered = true;
      if (building.getType() === 'Strange Device') b.countdown = (building as StrangeDevice).getCountdown();
      snap.b = b;
    }
    if (forestState && harvested) snap.forest = { wood: forestState.woodLeft, stumps: forestState.stumps };
    if (units.length) snap.units = units.map((u) => ({ type: u.getType(), owner: u.getOwner()!.getPlayerNum() }));
    if (conq.length) snap.conq = conq.map((u) => ({ type: u.getType(), owner: u.getOwner()!.getPlayerNum() }));
    tiles.push(snap);
  }

  return {
    v: 1,
    settings: {
      width: raw.width,
      height: raw.height,
      seed: raw.seed,
      players: allPlayers.map((p) => ({ name: p.getName(), difficulty: p.getDifficulty() })),
    },
    currentPlayerNum: pm.getCurrentPlayer().getPlayerNum(),
    roundsPlayed: pm.getRoundsPlayed(),
    lostPlayerNums: pm.getLostPlayers().map((p) => p.getPlayerNum()),
    resources,
    tiles,
  };
}

export function saveSnapshot(snap: GameSnapshot): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(snap));
  } catch {
    /* storage full / unavailable — saving is best-effort */
  }
}

export function loadSnapshot(): GameSnapshot | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const snap = JSON.parse(raw) as GameSnapshot;
    if (snap && snap.v === 1 && snap.settings && Array.isArray(snap.tiles)) return snap;
  } catch {
    /* corrupt save — ignore */
  }
  return null;
}

export function clearSnapshot(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
}
