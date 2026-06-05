// Port of Buildings/* (BuildingBase + concrete buildings).

import {
  ResourceMap,
  HQ_UNIT_VALUE,
  HQ_SOLDIER_VALUE,
  OUTPOST_SOLDIER_VALUE,
  VILLAGE_UNIT_VALUE,
  MIKONTALO_UNIT_VALUE,
  FARM_BUILD_COST,
  FARM_PRODUCTION,
  OUTPOST_BUILD_COST,
  OUTPOST_PRODUCTION,
  MINE_BUILD_COST,
  MINE_PRODUCTION,
  HEPP_BUILD_COST,
  HEPP_PRODUCTION,
  NUCLEARPP_BUILD_COST,
  NUCLEARPP_PRODUCTION,
  VILLAGE_BUILD_COST,
  VILLAGE_PRODUCTION,
  BRIDGE_BUILD_COST,
  BRIDGE_PRODUCTION,
  STRANGE_DEVICE_BUILD_COST,
  NO_RESOURCES,
} from '../core/resources';
import * as Desc from '../core/descriptions';
import { ImageVectors } from '../core/images';
import { PlaceableGameObject, IGameEventHandler, IObjectManager } from './base';
import type { TileBase } from './tile';
import type { PlayerBase } from './player';

export class BuildingBase extends PlaceableGameObject {
  readonly BUILD_COST: ResourceMap;
  readonly PRODUCTION_EFFECT: ResourceMap;
  protected basicDescription_: string;
  protected parentTile_: TileBase | null = null;

  constructor(
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    owner: PlayerBase | null,
    buildcost: ResourceMap = new Map(),
    production: ResourceMap = new Map(),
    basic_description = '',
  ) {
    super(eventhandler, objectmanager, owner);
    this.BUILD_COST = buildcost;
    this.PRODUCTION_EFFECT = production;
    this.basicDescription_ = basic_description;
  }

  getType(): string {
    return 'BuildingBase';
  }

  getProduction(): ResourceMap {
    return this.PRODUCTION_EFFECT;
  }

  getCost(): ResourceMap {
    return this.BUILD_COST;
  }

  addBasicDescription(desc: string): void {
    this.basicDescription_ = desc;
  }
  getBasicDescription(): string {
    return this.basicDescription_;
  }

  setParentTile(parentTile: TileBase): void {
    this.parentTile_ = parentTile;
  }

  /** Concrete buildings may override to provide a "<u>Effects:</u>..." string. */
  getExtraDescription(): string {
    return '';
  }
}

// --- Farm ------------------------------------------------------------------

export class Farm extends BuildingBase {
  private growthPhase_ = 1;

  constructor(
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    owner: PlayerBase | null,
    buildcost: ResourceMap = FARM_BUILD_COST,
    production: ResourceMap = FARM_PRODUCTION,
  ) {
    super(eventhandler, objectmanager, owner, buildcost, production, Desc.FARM_DESCRIPTION);
  }

  getType(): string {
    return 'Farm';
  }

  getGrowthPhase(): number {
    return this.growthPhase_;
  }

  setGrowthPhase(phase: number): void {
    this.growthPhase_ = phase;
    if (this.growthPhase_ >= 5) this.growthPhase_ = 1;
  }

  resetFarm(): void {
    this.setGrowthPhase(1);
    if (this.parentTile_) this.lockEventHandler().updateAnimatedTileToStatic(this.parentTile_, 1);
  }
}

// --- HeadQuarters ----------------------------------------------------------

export class HeadQuarters extends BuildingBase {
  private conqured_ = false;

  constructor(eventhandler: IGameEventHandler, objectmanager: IObjectManager, owner: PlayerBase | null) {
    super(eventhandler, objectmanager, owner);
    this.addBasicDescription(Desc.HEADQUARTERS_DESCRIPTION);
  }

  getType(): string {
    return 'Headquarters';
  }

  getExtraDescription(): string {
    if (!this.conqured_) {
      return `<u>Effects:</u><br>+${HQ_UNIT_VALUE} Max Units<br>+${HQ_SOLDIER_VALUE} Max Soldiers`;
    }
    return '';
  }

  setConquered(): void {
    this.conqured_ = true;
    if (this.parentTile_) this.lockEventHandler().updateAnimatedTileToStatic(this.parentTile_, 1);
    this.setImageFiles(ImageVectors.HEADQUARTERSDESTROYED);
    this.addBasicDescription(Desc.BROKEN_HEADQUARTERS_DESCRIPTION);
  }

  isConquered(): boolean {
    return this.conqured_;
  }
}

// --- Outpost ---------------------------------------------------------------

export class Outpost extends BuildingBase {
  constructor(
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    owner: PlayerBase | null,
    buildcost: ResourceMap = OUTPOST_BUILD_COST,
    production: ResourceMap = OUTPOST_PRODUCTION,
  ) {
    super(eventhandler, objectmanager, owner, buildcost, production, Desc.OUTPOST_DESCRIPTION);
  }
  getType(): string {
    return 'Outpost';
  }
  getExtraDescription(): string {
    return `<u>Effects:</u><br>+${OUTPOST_SOLDIER_VALUE} Max Soldiers`;
  }
}

// --- Mine ------------------------------------------------------------------

export class Mine extends BuildingBase {
  constructor(
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    owner: PlayerBase | null,
    buildcost: ResourceMap = MINE_BUILD_COST,
    production: ResourceMap = MINE_PRODUCTION,
  ) {
    super(eventhandler, objectmanager, owner, buildcost, production, Desc.MINE_DESCRIPTION);
  }
  getType(): string {
    return 'Mine';
  }
}

// --- HydroPower ------------------------------------------------------------

export class HydroPower extends BuildingBase {
  constructor(
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    owner: PlayerBase | null,
    buildcost: ResourceMap = HEPP_BUILD_COST,
    production: ResourceMap = HEPP_PRODUCTION,
  ) {
    super(eventhandler, objectmanager, owner, buildcost, production, Desc.HEPP_DESCRIPTION);
  }
  getType(): string {
    return 'Hydroelectric Power Plant';
  }
}

// --- NuclearPlant ----------------------------------------------------------

export class NuclearPlant extends BuildingBase {
  constructor(
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    owner: PlayerBase | null,
    buildcost: ResourceMap = NUCLEARPP_BUILD_COST,
    production: ResourceMap = NUCLEARPP_PRODUCTION,
  ) {
    super(eventhandler, objectmanager, owner, buildcost, production, Desc.NUCLEAR_DESCRIPTION);
  }
  getType(): string {
    return 'Nuclear Power Plant';
  }
}

// --- Village ---------------------------------------------------------------

export class Village extends BuildingBase {
  constructor(
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    owner: PlayerBase | null,
    buildcost: ResourceMap = VILLAGE_BUILD_COST,
    production: ResourceMap = VILLAGE_PRODUCTION,
  ) {
    super(eventhandler, objectmanager, owner, buildcost, production, Desc.VILLAGE_DESCRIPTION);
  }
  getType(): string {
    return 'Village';
  }
  getExtraDescription(): string {
    return `<u>Effects:</u><br>+${VILLAGE_UNIT_VALUE} Max Units`;
  }
}

// --- Bridge ----------------------------------------------------------------

export class Bridge extends BuildingBase {
  constructor(
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    owner: PlayerBase | null,
    buildcost: ResourceMap = BRIDGE_BUILD_COST,
    production: ResourceMap = BRIDGE_PRODUCTION,
  ) {
    super(eventhandler, objectmanager, owner, buildcost, production, Desc.BRIDGE_DESCRIPTION);
  }
  getType(): string {
    return 'Bridge';
  }
}

// --- Mikontalo -------------------------------------------------------------

export class Mikontalo extends BuildingBase {
  constructor(
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    owner: PlayerBase | null,
    buildcost: ResourceMap = NO_RESOURCES,
    production: ResourceMap = NO_RESOURCES,
  ) {
    super(eventhandler, objectmanager, owner, buildcost, production, Desc.MIKONTALO_DESCRIPTION);
  }
  getType(): string {
    return 'Mikontalo';
  }
  getExtraDescription(): string {
    return `<u>Effects:</u><br>+${MIKONTALO_UNIT_VALUE} Max Units`;
  }
}

// --- StrangeDevice ---------------------------------------------------------
// A new, draw-eliminating win condition (not in the C++/Qt original). Building it
// starts a countdown; if the Device still stands when the countdown elapses, its
// owner wins immediately. While owned it halves the owner's soldier cap (enforced in
// PlayerBase.updateUnitAmounts). Only one may exist in the game at a time. See
// STRANGE-DEVICE-DESIGN.md. No per-turn production — the cap halving is the balancer.

export class StrangeDevice extends BuildingBase {
  /** Rounds (owner end-of-turns) left until an undisturbed Device wins. Set on build. */
  private countdown_ = 0;

  constructor(
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    owner: PlayerBase | null,
    buildcost: ResourceMap = STRANGE_DEVICE_BUILD_COST,
  ) {
    super(eventhandler, objectmanager, owner, buildcost, NO_RESOURCES, Desc.STRANGE_DEVICE_DESCRIPTION);
  }

  getType(): string {
    return 'Strange Device';
  }

  getCountdown(): number {
    return this.countdown_;
  }
  setCountdown(rounds: number): void {
    this.countdown_ = rounds;
  }
  /** Tick the clock once (on the owner's end-of-turn). Floors at 0. */
  decrementCountdown(): void {
    if (this.countdown_ > 0) this.countdown_ -= 1;
  }

  getExtraDescription(): string {
    return `<u>Countdown:</u><br>${this.countdown_} rounds to victory`;
  }
}
