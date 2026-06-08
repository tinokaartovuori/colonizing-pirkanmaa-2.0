// Shared, Phaser-free economy/military metrics for the neural AI.
//
// These mirror the steady-state estimates the heuristic AiController computes
// privately, re-derived here (not imported) so the neural stack is fully
// self-contained and can be reused by the feature extractor, the candidate
// generator, and the safety layer without recomputation drift.
//
// Production/upkeep constants are read off resources.ts and the building rules
// in tiles.ts, so they track the original economy automatically.

import { TileBase } from '../../model/tile';
import { Grassland, Forest, AbundantForest, Mountain, River } from '../../model/tiles';
import { PlayerBase } from '../../model/player';
import { BasicResource } from '../../core/resources';
import type { ObjectManager } from '../../managers/objectmanager';
import type { PlayerManager } from '../../managers/playermanager';

export function ownedTiles(p: PlayerBase): TileBase[] {
  return p.getObjects().filter((o): o is TileBase => o instanceof TileBase);
}

export function res(p: PlayerBase, r: BasicResource): number {
  return p.getResources().get(r) ?? 0;
}
export const money = (p: PlayerBase) => res(p, BasicResource.MONEY);
export const wood = (p: PlayerBase) => res(p, BasicResource.WOOD);
export const stone = (p: PlayerBase) => res(p, BasicResource.STONE);
export const metal = (p: PlayerBase) => res(p, BasicResource.METAL);

export function hasType(tile: TileBase, type: string): boolean {
  return tile.getUnits().some((u) => u.getType() === type);
}
export function countWorkers(tile: TileBase): number {
  return tile.getUnits().filter((u) => u.getType() === 'BasicWorker').length;
}

export function salaryPerRound(p: PlayerBase): number {
  return (
    p.getCurrentBasicWorkerAmount() * 5 +
    p.getCurrentExpertAmount() * 25 +
    p.getCurrentSoldierAmount() * 30
  );
}

/** Money leaving the treasury each round: wages + building upkeep. */
export function moneyDrainPerRound(p: PlayerBase): number {
  let upkeep = 0;
  for (const t of ownedTiles(p)) {
    const type = t.getBuilding()?.getType();
    if (type === 'Village') upkeep += 5; // arc sd4 unit-cap rebalance (was 10)
    if (type === 'Outpost') upkeep += 50;
  }
  return salaryPerRound(p) + upkeep;
}

/** Amortised money income minus salaries (farms pay 175 every 4 rounds). */
export function netMoneyPerRound(p: PlayerBase): number {
  let income = 0;
  for (const tile of ownedTiles(p)) {
    const type = tile.getBuilding()?.getType();
    const workers = countWorkers(tile);
    const expert = hasType(tile, 'Expert');
    if (type === 'Farm' && workers > 0) income += 175 / 4;
    else if (type === 'Mine' && workers > 0) income += 20 * workers * (expert ? 2 : 1);
    else if (type === 'Nuclear Power Plant' && workers > 0 && expert) income += 160 * workers;
    else if (type === 'Hydroelectric Power Plant' && workers > 0 && expert) income += 80 * workers;
    else if (tile instanceof AbundantForest && workers > 0) income += 15;
    if (type === 'Village') income -= 5; // arc sd4 unit-cap rebalance (was 10)
    if (type === 'Outpost') income -= 50;
  }
  return income - salaryPerRound(p);
}

export function metalIncomePerRound(p: PlayerBase): number {
  let m = 0;
  for (const tile of ownedTiles(p)) {
    if (tile.getBuilding()?.getType() !== 'Mine') continue;
    m += 20 * countWorkers(tile) * (hasType(tile, 'Expert') ? 2 : 1);
  }
  return m;
}

/** Wood drained each round (Villages -10, Bridges -5). */
export function woodUpkeep(p: PlayerBase): number {
  let w = 0;
  for (const t of ownedTiles(p)) {
    const type = t.getBuilding()?.getType();
    if (type === 'Village') w += 10;
    if (type === 'Bridge') w += 5;
  }
  return w;
}

/** Gross wood produced this round by staffed, non-depleted forests (≈100/worker). */
export function woodIncomePerRound(p: PlayerBase): number {
  let w = 0;
  for (const t of ownedTiles(p)) {
    if (!(t instanceof Forest)) continue;
    // A worker on a forest with wood left yields 100; we can't see woodLeft_
    // here cheaply, so count staffed forests as producing (good enough for a
    // steady-state feature).
    w += countWorkers(t) * 100;
  }
  return w;
}

export interface BuildingCounts {
  Farm: number;
  Mine: number;
  Village: number;
  Outpost: number;
  'Nuclear Power Plant': number;
  'Hydroelectric Power Plant': number;
  Bridge: number;
  staffedFarms: number;
  forestHarvesters: number;
  freeMountains: number;
  freeGrassland: number;
  freeRivers: number;
}

export function buildingCounts(p: PlayerBase): BuildingCounts {
  const c: BuildingCounts = {
    Farm: 0, Mine: 0, Village: 0, Outpost: 0,
    'Nuclear Power Plant': 0, 'Hydroelectric Power Plant': 0, Bridge: 0,
    staffedFarms: 0, forestHarvesters: 0, freeMountains: 0, freeGrassland: 0, freeRivers: 0,
  };
  for (const t of ownedTiles(p)) {
    const type = t.getBuilding()?.getType();
    if (type === 'Farm') { c.Farm++; if (hasType(t, 'BasicWorker')) c.staffedFarms++; }
    else if (type === 'Mine') c.Mine++;
    else if (type === 'Village') c.Village++;
    else if (type === 'Outpost') c.Outpost++;
    else if (type === 'Nuclear Power Plant') c['Nuclear Power Plant']++;
    else if (type === 'Hydroelectric Power Plant') c['Hydroelectric Power Plant']++;
    else if (type === 'Bridge') c.Bridge++;
    if (t instanceof Forest && t.getBuilding() === null && hasType(t, 'BasicWorker')) c.forestHarvesters++;
    if (t instanceof Mountain && t.getBuilding() === null) c.freeMountains++;
    if (t instanceof Grassland && t.getBuilding() === null) c.freeGrassland++;
    if (t instanceof River && t.getBuilding() === null && t.getBuildableBuildings().length > 0) c.freeRivers++;
  }
  return c;
}

/** Soldiers an opponent has invading our tiles or massed on an adjacent tile. */
export function enemyThreat(p: PlayerBase): number {
  let threat = 0;
  for (const tile of ownedTiles(p)) {
    threat += tile.getConqueringUnits().filter((u) => u.getOwner() !== p && u.getType() === 'Soldier').length;
    for (const n of tile.getNeighbourTiles()) {
      const o = n.getOwner();
      if (o !== null && o !== p) threat += n.getUnits().filter((u) => u.getType() === 'Soldier').length;
    }
  }
  return threat;
}

export interface OpponentSummary {
  alive: number;
  maxTiles: number;
  totalTiles: number;
  totalSoldiers: number;
}

export function opponentSummary(p: PlayerBase, om: ObjectManager, pm: PlayerManager): OpponentSummary {
  let alive = 0;
  let maxTiles = 0;
  let totalTiles = 0;
  let totalSoldiers = 0;
  for (const other of pm.getPlayers()) {
    if (other === p) continue;
    alive++;
    const t = om.getTileCountForPlayer(other);
    totalTiles += t;
    if (t > maxTiles) maxTiles = t;
    totalSoldiers += other.getCurrentSoldierAmount();
  }
  return { alive, maxTiles, totalTiles, totalSoldiers };
}

/** True if an enemy-owned tile is reachable to attack (on our border). */
export function hasReachableEnemy(p: PlayerBase, om: ObjectManager): boolean {
  if (enemyThreat(p) > 0) return true;
  return om.getAvailableTiles().some((t) => {
    const o = t.getOwner();
    return o !== null && o !== p;
  });
}
