// Affordability / solvency guards for the neural AI.
//
// The neural policy only ever chooses among candidate actions that pass these
// guards, so — exactly like the heuristic AiController — a neural CPU can never
// bankrupt itself or make an illegal spend. The network is thereby freed to
// learn *strategy* (what to do, and when to stop) on top of a foundation that
// is always solvent. The thresholds mirror AiController's private helpers.

import { PlayerBase } from '../../model/player';
import { BasicResource, ResourceMap } from '../../core/resources';
import * as M from './metrics';

/** Buffer kept when hiring a worker for a net-positive building. */
export const STAFF_RESERVE = 20;

/** Affordable while keeping `reserve` + ~5 rounds of drain buffered, no resource negative. */
export function affords(p: PlayerBase, cost: ResourceMap, reserve: number): boolean {
  if (!p.hasEnoughResources(cost)) return false;
  const buffer = reserve + M.moneyDrainPerRound(p) * 5;
  return M.money(p) + (cost.get(BasicResource.MONEY) ?? 0) >= buffer;
}

/** Income builds (farms) only need raw resources + a small money floor — they
 *  pay for themselves, so the salary-buffer must not block the bootstrap. */
export function affordsIncomeBuild(p: PlayerBase, cost: ResourceMap, floor = 40): boolean {
  if (!p.hasEnoughResources(cost)) return false;
  return M.money(p) + (cost.get(BasicResource.MONEY) ?? 0) >= floor;
}

/** Safe to take on one more salaried unit without pushing projected net negative. */
export function canAffordUpkeep(p: PlayerBase, salary: number): boolean {
  return M.netMoneyPerRound(p) - salary >= 0;
}

/** Safe to spend `cost`'s wood without risking a wood death during a regrow gap. */
export function hasWoodBuffer(p: PlayerBase, cost: ResourceMap): boolean {
  const need = -(cost.get(BasicResource.WOOD) ?? 0);
  if (need <= 0) return true;
  const upkeep = M.woodUpkeep(p);
  const buffer = upkeep > 0 ? Math.max(100, upkeep * 5) : 0;
  return M.wood(p) - need >= buffer;
}
