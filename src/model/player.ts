// Port of Core/playerbase.{h,cpp}.

import {
  ResourceMap,
  STARTING_RESOURCES,
  RESOURCE_LIMITS,
  UNIT_LIMITS,
  mergeResourceMaps,
  cloneResourceMap,
  BasicResource,
} from '../core/resources';
import type { GameObject, IObjectManager } from './base';
import { TileBase } from './tile';
import { UnitBase } from './unit';

/** Heuristic CPU skill levels (driven by AiController's PARAMS table). */
export type CpuDifficulty = 'easy' | 'medium' | 'hard';
/** Neural-network CPU skill levels (driven by NeuralAiController). */
export type NeuralDifficulty = 'nn-easy' | 'nn-medium' | 'nn-hard';
/**
 * A specific trained model played at full strength, e.g. `model:v57-champion`.
 * The id after the colon indexes the bundled NEURAL_MODELS roster (models.ts).
 */
export type NeuralModelDifficulty = `model:${string}`;
/**
 * CPU skill level. 'human' marks a player that is controlled by a person; the
 * 'nn-*' levels are the tiered neural champion, and `model:<id>` selects one of
 * the named trained opponents — all added alongside the original heuristic
 * easy/medium/hard, never replacing them.
 */
export type Difficulty = 'human' | CpuDifficulty | NeuralDifficulty | NeuralModelDifficulty;

/** True for a `model:<id>` difficulty (a specific bundled trained opponent). */
export function isNeuralModelDifficulty(d: Difficulty): d is NeuralModelDifficulty {
  return typeof d === 'string' && d.startsWith('model:');
}

/** Per-player configuration produced by the start dialog. */
export interface PlayerConfig {
  name: string;
  difficulty?: Difficulty; // 'human' (default) or a CPU level
}

export class PlayerBase {
  private m_name: string;
  private playerNum_: number;
  private objectManager_: IObjectManager;
  private objects_: GameObject[] = [];
  private maxSoldierAmount_ = 0;
  private maxUnitAmount_ = 0;
  private resources_: ResourceMap = new Map();
  private difficulty_: Difficulty;

  constructor(name: string, playerNum: number, objectmanager: IObjectManager, difficulty: Difficulty = 'human') {
    this.m_name = name;
    this.playerNum_ = playerNum;
    this.objectManager_ = objectmanager;
    this.difficulty_ = difficulty;
    this.addOrRemoveResources(STARTING_RESOURCES);
  }

  /** True for any computer-controlled player. */
  isCpu(): boolean {
    return this.difficulty_ !== 'human';
  }

  getDifficulty(): Difficulty {
    return this.difficulty_;
  }

  addObject(object: GameObject): void {
    this.objects_.push(object);
  }

  hasObject(object: GameObject): boolean {
    return this.objects_.includes(object);
  }

  addObjects(objects: GameObject[]): void {
    this.objects_.push(...objects);
  }

  addOrRemoveResources(resources: ResourceMap): void {
    this.resources_ = mergeResourceMaps(this.resources_, resources);
  }

  getResources(): ResourceMap {
    return this.resources_;
  }

  /** Overwrite this player's resource totals (used when restoring a saved game). */
  setResources(resources: ResourceMap): void {
    this.resources_ = cloneResourceMap(resources);
  }

  hasEnoughResources(cost: ResourceMap): boolean {
    const resources = mergeResourceMaps(this.resources_, cost);
    for (const value of resources.values()) {
      if (value < 0) return false;
    }
    return true;
  }

  removeObject(object: GameObject | null): void {
    if (object === null) return;
    const idx = this.objects_.indexOf(object);
    if (idx === -1) {
      throw new Error('Object not found.');
    }
    this.objects_.splice(idx, 1);
  }

  removeObjects(objects: GameObject[]): void {
    for (const object of objects) {
      try {
        this.removeObject(object);
      } catch {
        /* KeyError ignored, matches original */
      }
    }
  }

  getObjects(): GameObject[] {
    return [...this.objects_];
  }

  getName(): string {
    return this.m_name;
  }

  getPlayerNum(): number {
    return this.playerNum_;
  }

  getFreeUnitAmount(): number {
    return this.maxUnitAmount_ - this.getCurrentBasicWorkerAmount() - this.getCurrentExpertAmount();
  }

  getFreeSoldierAmount(): number {
    return this.maxSoldierAmount_ - this.getCurrentSoldierAmount();
  }

  getMaxUnitAmount(): number {
    this.updateUnitAmounts();
    return this.maxUnitAmount_;
  }

  getMaxSoldierAmount(): number {
    this.updateUnitAmounts();
    return this.maxSoldierAmount_;
  }

  updateUnitAmounts(): void {
    let newMaxUnitAmount = 0;
    let newMaxSoldierAmount = 0;
    for (const obj of this.objects_) {
      if (obj instanceof TileBase) {
        newMaxUnitAmount += obj.getMaxUnitsIncrease();
        newMaxSoldierAmount += obj.getMaxSoldiersIncrease();
      }
    }
    if (newMaxUnitAmount >= UNIT_LIMITS) newMaxUnitAmount = UNIT_LIMITS;
    if (newMaxSoldierAmount >= UNIT_LIMITS) newMaxSoldierAmount = UNIT_LIMITS;
    // Owning a standing Strange Device applies a FIXED −2 soldier-cap penalty (arc sd5
    // rebalance, floored at 0): the cost of racing the Device's countdown is being left
    // defensively exposed, but a flat −2 (down from the old halving) lets the builder
    // still field a real ring of defenders. The forced disband of any now-excess
    // soldiers happens on build (GameEventHandler) and at every endTurn via
    // eliminateExcessUnits().
    if (this.ownsStrangeDevice()) newMaxSoldierAmount = Math.max(0, newMaxSoldierAmount - 2);
    this.maxUnitAmount_ = newMaxUnitAmount;
    this.maxSoldierAmount_ = newMaxSoldierAmount;
  }

  /** True while this player owns a tile carrying a standing Strange Device. */
  ownsStrangeDevice(): boolean {
    for (const obj of this.objects_) {
      if (obj instanceof TileBase && obj.getBuilding()?.getType() === 'Strange Device') return true;
    }
    return false;
  }

  eliminateExcessUnits(): void {
    this.updateUnitAmounts();
    // Each removed unit must also leave objects_, or countUnits never drops and the
    // loop spins forever. removeOne returns false when there's nothing left to cull.
    const removeOne = (match: (u: UnitBase) => boolean): boolean => {
      for (const obj of this.objects_) {
        if (obj instanceof UnitBase && match(obj)) {
          this.objectManager_.getGameScene().removeItem(obj);
          obj.getParentTile()?.removeUnit(obj);
          this.removeObject(obj);
          return true;
        }
      }
      return false;
    };
    while (this.getFreeUnitAmount() < 0) {
      if (!removeOne((u) => u.getType() === 'BasicWorker' || u.getType() === 'Expert')) break;
      this.updateUnitAmounts();
    }
    while (this.getFreeSoldierAmount() < 0) {
      if (!removeOne((u) => u.getType() === 'Soldier')) break;
      this.updateUnitAmounts();
    }
  }

  getCurrentUnitAmount(): number {
    return this.getCurrentBasicWorkerAmount() + this.getCurrentExpertAmount();
  }

  getCurrentBasicWorkerAmount(): number {
    return this.countUnits('BasicWorker');
  }
  getCurrentExpertAmount(): number {
    return this.countUnits('Expert');
  }
  getCurrentSoldierAmount(): number {
    return this.countUnits('Soldier');
  }

  private countUnits(type: string): number {
    let amount = 0;
    for (const obj of this.objects_) {
      if (obj instanceof UnitBase && obj.getType() === type) amount++;
    }
    return amount;
  }

  limitResources(): void {
    for (const [key, value] of this.resources_) {
      const limit = RESOURCE_LIMITS.get(key as BasicResource)!;
      if (value >= limit) this.resources_.set(key, limit);
    }
  }
}
