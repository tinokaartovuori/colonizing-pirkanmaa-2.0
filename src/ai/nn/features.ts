// Board-size-invariant state features for the neural AI.
//
// CRITICAL design choice: features are *global aggregates*, never a per-cell
// grid. A 10×10 opening and a 25×15 late game produce vectors of the same
// length and the same meaning (fractions, per-round rates, ratios-to-caps), so
// one trained network plays well on any map size and at any game stage — the
// "works even if the map is huge and the game has run very long" requirement.
//
// Everything is normalised to roughly [-3, 3] and clamped, which keeps the
// tanh hidden layers in a sane range regardless of absolute resource totals.

import { PlayerBase } from '../../model/player';
import type { ObjectManager } from '../../managers/objectmanager';
import type { PlayerManager } from '../../managers/playermanager';
import * as M from './metrics';

const clamp = (v: number, lo = -3, hi = 3) => (v < lo ? lo : v > hi ? hi : v);

/** Names of the global features, in order — also defines GLOBAL_DIM. */
export const GLOBAL_FEATURE_NAMES = [
  'money', 'wood', 'stone', 'metal',
  'netMoney', 'metalIncome', 'netWood', 'moneyDrain',
  'tileFraction', 'tileAbs',
  'maxUnit', 'freeUnit', 'workers', 'experts',
  'maxSoldier', 'freeSoldier', 'soldiers',
  'staffedFarms', 'mines', 'villages', 'outposts', 'powerplants', 'harvesters',
  'freeGrass', 'freeMountain', 'freeRiver',
  'round', 'threat',
  'oppMaxFraction', 'leadMargin', 'oppSoldiers', 'oppAlive',
  'dominationProgress', 'neutralFraction', 'reachableEnemy',
  'bias',
] as const;

export const GLOBAL_DIM = GLOBAL_FEATURE_NAMES.length;

/**
 * Build the global feature vector for `player`. `round` is the rounds-played
 * counter (so the net can sense game stage); pass `pm.getRoundsPlayed()`.
 */
export function globalFeatures(
  player: PlayerBase,
  om: ObjectManager,
  pm: PlayerManager,
  round: number,
): number[] {
  const totalTiles = Math.max(1, om.getTileCount());
  const myTiles = om.getTileCountForPlayer(player);
  const bc = M.buildingCounts(player);
  const opp = M.opponentSummary(player, om, pm);
  const neutral = om.getNeutralTiles();

  const f = [
    clamp(M.money(player) / 1000),
    clamp(M.wood(player) / 1000),
    clamp(M.stone(player) / 1000),
    clamp(M.metal(player) / 500),
    clamp(M.netMoneyPerRound(player) / 100),
    clamp(M.metalIncomePerRound(player) / 100),
    clamp((M.woodIncomePerRound(player) - M.woodUpkeep(player)) / 300),
    clamp(M.moneyDrainPerRound(player) / 200),
    clamp(myTiles / totalTiles, 0, 1),
    clamp(myTiles / 40),
    clamp(player.getMaxUnitAmount() / 20),
    clamp(player.getFreeUnitAmount() / 10),
    clamp(player.getCurrentBasicWorkerAmount() / 20),
    clamp(player.getCurrentExpertAmount() / 10),
    clamp(player.getMaxSoldierAmount() / 15),
    clamp(player.getFreeSoldierAmount() / 10),
    clamp(player.getCurrentSoldierAmount() / 15),
    clamp(bc.staffedFarms / 15),
    clamp(bc.Mine / 8),
    clamp(bc.Village / 6),
    clamp(bc.Outpost / 4),
    clamp((bc['Nuclear Power Plant'] + bc['Hydroelectric Power Plant']) / 4),
    clamp(bc.forestHarvesters / 4),
    clamp(bc.freeGrassland / 15),
    clamp(bc.freeMountains / 6),
    clamp(bc.freeRivers / 6),
    clamp(round / 60, 0, 3),
    clamp(M.enemyThreat(player) / 8),
    clamp(opp.maxTiles / totalTiles, 0, 1),
    clamp((myTiles - opp.maxTiles) / totalTiles),
    clamp(opp.totalSoldiers / 15),
    clamp(opp.alive / 3, 0, 1),
    clamp(myTiles / (0.7 * totalTiles), 0, 2),
    clamp(neutral / totalTiles, 0, 1),
    M.hasReachableEnemy(player, om) ? 1 : 0,
    1, // bias
  ];
  return f;
}
