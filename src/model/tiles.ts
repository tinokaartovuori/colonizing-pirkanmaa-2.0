// Port of Tiles/{grassland,forest,abundantforest,mountain,river}.

import {
  ResourceMap,
  NO_RESOURCES,
  EMPTY,
  BasicResource,
  FOREST_PRODUCTION,
  FOREST_CAPACITY,
  FOREST_GROW_TIME,
  ABUNDANT_FOREST_PRODUCTION,
  FARM_GROW_TIME,
  HQ_UNIT_VALUE,
  HQ_SOLDIER_VALUE,
  VILLAGE_UNIT_VALUE,
  MIKONTALO_UNIT_VALUE,
  OUTPOST_SOLDIER_VALUE,
  mergeResourceMaps,
  getPositivesMap,
} from '../core/resources';
import { Coordinate } from '../core/coordinate';
import { ImageVectors, AnimationOptions } from '../core/images';
import * as Desc from '../core/descriptions';
import { TileBase } from './tile';
import { Farm, HeadQuarters } from './building';
import type { UnitBase } from './unit';
import type { PlayerBase } from './player';
import type { IGameEventHandler, IObjectManager } from './base';

function resourceMapsEqual(a: ResourceMap, b: ResourceMap): boolean {
  if (a.size !== b.size) return false;
  for (const [k, v] of a) if (b.get(k) !== v) return false;
  return true;
}

// ===========================================================================
// Grassland
// ===========================================================================

export class Grassland extends TileBase {
  constructor(
    location: Coordinate,
    size_x: number,
    size_y: number,
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    max_units = 3,
    production: ResourceMap = EMPTY,
  ) {
    super(location, size_x, size_y, eventhandler, objectmanager, max_units, production, Desc.GRASSLAND_DESCRIPTION);
  }

  getType(): string {
    return 'Grassland';
  }

  getBuildableBuildings(): string[] {
    // Outposts may not sit next to the HQ/another outpost (the original rule).
    let list = ['Farm', 'Village', 'Outpost', 'Nuclear Power Plant'];
    for (const tile of this.getNeighbourTiles()) {
      const b = tile.getBuilding();
      if (
        b !== null &&
        tile.getOwner() === this.getOwner() &&
        (b.getType() === 'Headquarters' || b.getType() === 'Outpost')
      ) {
        list = ['Farm', 'Village', 'Nuclear Power Plant'];
        break;
      }
    }
    // The Strange Device is offered on any owned, UNOCCUPIED grassland (never the HQ
    // tile, which already has a building) while none exists yet — at most one in the
    // whole game. It must be empty because the Device tile can never hold units.
    if (!this.lockObjectManager().hasStrangeDevice() && this.getUnitCount() === 0) {
      list = [...list, 'Strange Device'];
    }
    return list;
  }

  generateResources(): void {
    const building = this.getBuilding();
    if (building === null) return;
    const owner = this.getOwner();
    if (building.getType() === 'Farm') {
      let hasWorker = false;
      for (const unit of this.getUnits()) if (unit.getType() === 'BasicWorker') hasWorker = true;
      const farm = building as Farm;
      const growthPhase = farm.getGrowthPhase() + 1;
      farm.setGrowthPhase(growthPhase);
      if (growthPhase === 5 && hasWorker) {
        owner?.addOrRemoveResources(building.getProduction());
        farm.resetFarm();
      } else if (hasWorker) {
        this.lockEventHandler().updateAnimatedTileToStatic(this, growthPhase);
      } else if (!hasWorker) {
        farm.resetFarm();
      }
    }
    if (building.getType() === 'Nuclear Power Plant') {
      let hasExpert = false;
      for (const unit of this.getUnits()) if (unit.getType() === 'Expert') hasExpert = true;
      if (!hasExpert) return;
      for (const unit of this.getUnits()) {
        if (unit.getType() === 'BasicWorker') owner?.addOrRemoveResources(building.getProduction());
      }
    }
    if (building.getType() === 'Village') owner?.addOrRemoveResources(building.getProduction());
    if (building.getType() === 'Outpost') owner?.addOrRemoveResources(building.getProduction());
  }

  getCurrentRevenue(): ResourceMap {
    let production = NO_RESOURCES;
    const building = this.getBuilding();
    if (building !== null) {
      if (building.getType() === 'Farm') {
        let hasWorker = false;
        for (const unit of this.getUnits()) if (unit.getType() === 'BasicWorker') hasWorker = true;
        const growthPhase = (building as Farm).getGrowthPhase() + 1;
        if (growthPhase === 5 && hasWorker) {
          production = mergeResourceMaps(production, building.getProduction());
        }
      }
      if (building.getType() === 'Nuclear Power Plant') {
        let hasExpert = false;
        for (const unit of this.getUnits()) if (unit.getType() === 'Expert') hasExpert = true;
        if (!hasExpert) return production;
        for (const unit of this.getUnits()) {
          if (unit.getType() === 'BasicWorker') production = mergeResourceMaps(production, building.getProduction());
        }
      }
    }
    return production;
  }

  getExtraDescription(): string {
    const building = this.getBuilding();
    if (building !== null) {
      if (building.getType() === 'Farm') {
        const phase = (building as Farm).getGrowthPhase();
        if (phase === FARM_GROW_TIME) return '<u>Growth:</u><br>Ready next round!';
        return `<u>Growth:</u><br>${phase - 1}/${FARM_GROW_TIME}`;
      }
      if (building.getType() === 'Outpost') return building.getExtraDescription();
      if (building.getType() === 'Village') return building.getExtraDescription();
      if (building.getType() === 'Strange Device') return building.getExtraDescription();
      if (building.getType() === 'Nuclear Power Plant') {
        if (this.getUnitCount() === 0) return '';
        let expert = false;
        for (const unit of this.getUnits()) if (unit.getType() === 'Expert') expert = true;
        return expert ? '' : 'Expert is missing!';
      }
    }
    return '';
  }

  getNetDescription(): string {
    const building = this.getBuilding();
    if (building !== null) {
      if (building.getType() === 'Headquarters') return (building as HeadQuarters).getExtraDescription();
      if (building.getType() === 'Mikontalo') return building.getExtraDescription();
    }
    let functionalString = '<u>Net value:</u>';
    const net = this.getCurrentNet();
    if (resourceMapsEqual(net, NO_RESOURCES) || resourceMapsEqual(net, EMPTY)) {
      if (building !== null && building.getType() === 'Farm') {
        return functionalString + '<br>No money this round.';
      }
      return '';
    }
    functionalString += this.netLines(net);
    return functionalString;
  }

  hasOpponentHeadquarters(player: PlayerBase): boolean {
    const building = this.getBuilding();
    if (building !== null && building.getType() === 'Headquarters' && this.getOwner() === player) {
      if (!(building as HeadQuarters).isConquered()) return false;
    }
    return true;
  }

  getMaxUnitsIncrease(): number {
    const building = this.getBuilding();
    if (building !== null) {
      if (building.getType() === 'Headquarters' && !(building as HeadQuarters).isConquered()) return HQ_UNIT_VALUE;
      if (building.getType() === 'Village') return VILLAGE_UNIT_VALUE;
      if (building.getType() === 'Mikontalo') return MIKONTALO_UNIT_VALUE;
    }
    return 0;
  }

  getMaxSoldiersIncrease(): number {
    const building = this.getBuilding();
    if (building !== null) {
      if (building.getType() === 'Headquarters' && !(building as HeadQuarters).isConquered()) return HQ_SOLDIER_VALUE;
      if (building.getType() === 'Outpost') return OUTPOST_SOLDIER_VALUE;
    }
    return 0;
  }

  updateAnimation(): void {
    const building = this.getBuilding();
    if (building === null) return;
    if (building.getType() === 'Nuclear Power Plant') {
      const scene = this.lockObjectManager().getGameScene();
      const handle = scene.getObjectInScene(building);
      const producing = (this.getCurrentRevenue().get(BasicResource.MONEY) ?? 0) > 0;
      const opt = producing ? AnimationOptions.NUCLEAR : AnimationOptions.EMPTY;
      building.setAnimationOption(opt);
      if (handle) {
        handle.setAnimationOption(opt);
        handle.setAnimationFrame(1);
      }
    }
  }
}

// ===========================================================================
// Forest
// ===========================================================================

export class Forest extends TileBase {
  private woodLeft_: number;
  private roundsStumpsHaveBeen_ = 0;

  constructor(
    location: Coordinate,
    size_x: number,
    size_y: number,
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    max_units = 3,
    production: ResourceMap = EMPTY,
  ) {
    super(location, size_x, size_y, eventhandler, objectmanager, max_units, production, Desc.FOREST_DESCRIPTION);
    this.woodLeft_ = FOREST_CAPACITY.get(BasicResource.WOOD)!;
  }

  getType(): string {
    return 'Forest';
  }

  /** Current harvest progress, for save/restore. */
  getHarvestState(): { woodLeft: number; stumps: number } {
    return { woodLeft: this.woodLeft_, stumps: this.roundsStumpsHaveBeen_ };
  }
  /** Restore harvest progress from a saved game. */
  setHarvestState(woodLeft: number, stumps: number): void {
    this.woodLeft_ = woodLeft;
    this.roundsStumpsHaveBeen_ = stumps;
  }

  getBuildableBuildings(): string[] {
    if (this.woodLeft_ === 0) {
      let list = ['Farm', 'Village', 'Outpost', 'Nuclear Power Plant'];
      for (const tile of this.getNeighbourTiles()) {
        const b = tile.getBuilding();
        if (b !== null && (b.getType() === 'Headquarters' || b.getType() === 'Outpost')) {
          list = ['Farm', 'Village', 'Nuclear Power Plant'];
          break;
        }
      }
      // Offered only on an UNOCCUPIED tile — the Device tile can never hold units.
      if (!this.lockObjectManager().hasStrangeDevice() && this.getUnitCount() === 0) {
        list = [...list, 'Strange Device'];
      }
      return list;
    }
    return [];
  }

  generateResources(): void {
    const owner = this.getOwner();
    for (const unit of this.getUnits()) {
      if (unit.getType() === 'BasicWorker' && this.woodLeft_ > 0) {
        owner?.addOrRemoveResources(FOREST_PRODUCTION);
        this.woodLeft_ -= FOREST_PRODUCTION.get(BasicResource.WOOD)!;
      }
    }

    if (this.roundsStumpsHaveBeen_ === FOREST_GROW_TIME) {
      this.roundsStumpsHaveBeen_ = 0;
      this.woodLeft_ = FOREST_CAPACITY.get(BasicResource.WOOD)!;
      this.lockEventHandler().updateForest('Grow', this);
    } else if (this.woodLeft_ === 0 && this.roundsStumpsHaveBeen_ === 0) {
      this.lockEventHandler().updateForest('Cut', this);
      this.roundsStumpsHaveBeen_++;
    } else if (this.woodLeft_ === 0) {
      this.roundsStumpsHaveBeen_++;
    }
  }

  getCurrentRevenue(): ResourceMap {
    let wl = this.woodLeft_;
    let production = NO_RESOURCES;
    for (const unit of this.getUnits()) {
      if (unit.getType() === 'BasicWorker' && wl > 0) {
        production = mergeResourceMaps(production, FOREST_PRODUCTION);
        wl -= FOREST_PRODUCTION.get(BasicResource.WOOD)!;
      }
    }
    return production;
  }

  /** A pixel-art progress bar (0..1 filled) for the menu's tile-inspection panel. */
  private progressBar(fraction: number, kind: 'cut' | 'grow'): string {
    const pct = Math.max(0, Math.min(100, Math.round(fraction * 100)));
    return `<div class="cp-bar"><div class="cp-bar-fill cp-bar-${kind}" style="width:${pct}%"></div></div>`;
  }

  getExtraDescription(): string {
    const cap = FOREST_CAPACITY.get(BasicResource.WOOD)!;
    if (this.woodLeft_ === cap && this.roundsStumpsHaveBeen_ === 0) return '';
    // Being cut: the bar fills as the forest is felled (full bar = fully cut down).
    if (this.woodLeft_ > 0 && this.roundsStumpsHaveBeen_ === 0) {
      return `<u>Cut down:</u>${this.progressBar((cap - this.woodLeft_) / cap, 'cut')}`;
    }
    // Cut bare and regrowing: the bar fills as it grows back.
    if (this.roundsStumpsHaveBeen_ > 0) {
      return `<u>Regrowing:</u>${this.progressBar((this.roundsStumpsHaveBeen_ - 1) / FOREST_GROW_TIME, 'grow')}`;
    }
    return '';
  }
}

// ===========================================================================
// AbundantForest
// ===========================================================================

export class AbundantForest extends TileBase {
  constructor(
    location: Coordinate,
    size_x: number,
    size_y: number,
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    max_units = 3,
    production: ResourceMap = EMPTY,
  ) {
    super(
      location,
      size_x,
      size_y,
      eventhandler,
      objectmanager,
      max_units,
      production,
      Desc.ABUNDANT_FOREST_DESCRIPTION,
    );
  }

  getType(): string {
    return 'Abundant Forest';
  }

  getBuildableBuildings(): string[] {
    return [];
  }

  generateResources(): void {
    const owner = this.getOwner();
    for (const unit of this.getUnits()) {
      if (unit.getType() === 'BasicWorker') {
        owner?.addOrRemoveResources(ABUNDANT_FOREST_PRODUCTION);
        break;
      }
    }
  }

  getCurrentRevenue(): ResourceMap {
    let production = NO_RESOURCES;
    for (const unit of this.getUnits()) {
      if (unit.getType() === 'BasicWorker') {
        production = mergeResourceMaps(production, ABUNDANT_FOREST_PRODUCTION);
        break;
      }
    }
    return production;
  }

  getExtraDescription(): string {
    return '';
  }
}

// ===========================================================================
// Mountain
// ===========================================================================

export class Mountain extends TileBase {
  constructor(
    location: Coordinate,
    size_x: number,
    size_y: number,
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    max_units = 3,
    production: ResourceMap = EMPTY,
  ) {
    super(location, size_x, size_y, eventhandler, objectmanager, max_units, production, Desc.MOUNTAIN_DESCRIPTION);
  }

  getType(): string {
    return 'Mountain';
  }

  getBuildableBuildings(): string[] {
    return ['Mine'];
  }

  generateResources(): void {
    const building = this.getBuilding();
    const owner = this.getOwner();
    if (building !== null && building.getType() === 'Mine') {
      let hasExpert = false;
      for (const unit of this.getUnits()) if (unit.getType() === 'Expert') hasExpert = true;
      for (const unit of this.getUnits()) {
        if (unit.getType() === 'BasicWorker') {
          owner?.addOrRemoveResources(building.getProduction());
          if (hasExpert) owner?.addOrRemoveResources(building.getProduction());
        }
      }
    }
  }

  getCurrentRevenue(): ResourceMap {
    let production = NO_RESOURCES;
    const building = this.getBuilding();
    if (building !== null && building.getType() === 'Mine') {
      let hasExpert = false;
      for (const unit of this.getUnits()) if (unit.getType() === 'Expert') hasExpert = true;
      for (const unit of this.getUnits()) {
        if (unit.getType() === 'BasicWorker') {
          production = mergeResourceMaps(production, building.getProduction());
          if (hasExpert) production = mergeResourceMaps(production, building.getProduction());
        }
      }
    }
    return production;
  }

  getExtraDescription(): string {
    const building = this.getBuilding();
    if (building !== null && building.getType() === 'Mine') {
      for (const unit of this.getUnits()) {
        if (unit.getType() === 'Expert') return 'Expert doubles the production rate.';
      }
    }
    return '';
  }
}

// ===========================================================================
// River
// ===========================================================================

export class River extends TileBase {
  private riverOrientation_ = 3;
  private riverShape_ = '';

  constructor(
    location: Coordinate,
    size_x: number,
    size_y: number,
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    max_units = 3,
    production: ResourceMap = EMPTY,
  ) {
    super(location, size_x, size_y, eventhandler, objectmanager, max_units, production);
  }

  getType(): string {
    return 'River';
  }

  getBuildableBuildings(): string[] {
    if (this.riverOrientation_ === 1 || this.riverOrientation_ === 0) {
      return ['Bridge', 'Hydroelectric Power Plant'];
    }
    return [];
  }

  generateResources(): void {
    const building = this.getBuilding();
    if (building === null) return;
    const owner = this.getOwner();
    if (building.getType() === 'Hydroelectric Power Plant') {
      let hasExpert = false;
      for (const unit of this.getUnits()) if (unit.getType() === 'Expert') hasExpert = true;
      if (!hasExpert) return;
      for (const unit of this.getUnits()) {
        if (unit.getType() === 'BasicWorker') owner?.addOrRemoveResources(building.getProduction());
      }
    }
    if (building.getType() === 'Bridge') owner?.addOrRemoveResources(building.getProduction());
  }

  getCurrentRevenue(): ResourceMap {
    let production = NO_RESOURCES;
    const building = this.getBuilding();
    if (building !== null) {
      if (building.getType() === 'Hydroelectric Power Plant') {
        let hasExpert = false;
        for (const unit of this.getUnits()) if (unit.getType() === 'Expert') hasExpert = true;
        if (!hasExpert) return NO_RESOURCES;
        for (const unit of this.getUnits()) {
          if (unit.getType() === 'BasicWorker') {
            production = mergeResourceMaps(production, getPositivesMap(building.getProduction()));
          }
        }
        return production;
      }
      if (building.getType() === 'Bridge') {
        return mergeResourceMaps(production, getPositivesMap(building.getProduction()));
      }
    }
    return NO_RESOURCES;
  }

  getExtraDescription(): string {
    const building = this.getBuilding();
    if (building !== null && building.getType() === 'Hydroelectric Power Plant') {
      if (this.getUnitCount() !== 0) {
        let hasExpert = false;
        for (const unit of this.getUnits()) if (unit.getType() === 'Expert') hasExpert = true;
        if (!hasExpert) return 'Expert is missing!';
      }
    }
    return '';
  }

  /** Whether a unit at the given owned/conquering slot index should use the swim sprite. */
  private ownedSwims(index: number): boolean {
    if (this.riverShape_ === 'NS' || this.riverShape_ === 'SE' || this.riverShape_ === 'SW') {
      return index === 1;
    }
    return false;
  }
  private conqueringSwims(index: number): boolean {
    let changeImage = index === 1;
    if (this.riverShape_ === 'EW') changeImage = true;
    if (this.riverShape_ === 'SE' || this.riverShape_ === 'NE') {
      if (index === 2) changeImage = true;
    }
    if (this.riverShape_ === 'NW' || this.riverShape_ === 'SW') {
      if (index === 0) changeImage = true;
    }
    return changeImage;
  }

  private setUnitSprite(unit: UnitBase, swim: boolean): void {
    const type = unit.getType();
    if (swim && this.getBuilding() === null) {
      if (type === 'BasicWorker') unit.setImageFiles(ImageVectors.BASICWORKER_SWIM);
      if (type === 'Expert') unit.setImageFiles(ImageVectors.EXPERT_SWIM);
      if (type === 'Soldier') unit.setImageFiles(ImageVectors.SOLDIER_SWIM);
    } else {
      if (type === 'BasicWorker') unit.setImageFiles(ImageVectors.BASICWORKER);
      if (type === 'Expert') unit.setImageFiles(ImageVectors.EXPERT);
      if (type === 'Soldier') unit.setImageFiles(ImageVectors.SOLDIER);
    }
  }

  addUnit(unit: UnitBase): void {
    if (!unit.isConqueringUnit()) {
      if (this.getUnitCount() + 1 > 3) throw new Error('Tile has no more room for Units!');
      unit.setLocationTile(this);
      const index = this.getUnitCount();
      this.setUnitSprite(unit, this.ownedSwims(index));
      this.units_.push(unit);
    } else {
      if (this.getConqueringUnitCount() + 1 > 3) throw new Error('Tile has no more room for Enemy units!');
      unit.setLocationTile(this);
      const index = this.getConqueringUnitCount();
      this.setUnitSprite(unit, this.conqueringSwims(index));
      this.conqueringUnits_.push(unit);
    }
  }

  updateUnitCoordinates(): void {
    let ind = 0;
    for (const u of this.units_) {
      u.setTileRelatedCoordinates(ind, 1);
      this.setUnitSprite(u, this.ownedSwims(ind));
      ind++;
    }
    ind = 0;
    for (const u of this.conqueringUnits_) {
      u.setTileRelatedCoordinates(ind, 0);
      this.setUnitSprite(u, this.conqueringSwims(ind));
      ind++;
    }
  }

  updateAnimation(): void {
    const building = this.getBuilding();
    if (building === null) return;
    const scene = this.lockObjectManager().getGameScene();
    const handle = scene.getObjectInScene(building);
    const producing =
      building.getType() === 'Hydroelectric Power Plant' && (this.getCurrentRevenue().get(BasicResource.MONEY) ?? 0) > 0;
    const opt = producing ? AnimationOptions.HEPP : AnimationOptions.EMPTY;
    building.setAnimationOption(opt);
    if (handle) {
      handle.setAnimationOption(opt);
      handle.setAnimationFrame(1);
    }
  }

  getRiverOrientation(): number {
    return this.riverOrientation_;
  }
  setRiverOrientation(ori: number): void {
    this.riverOrientation_ = ori;
    this.addBasicDescription(ori === 3 ? Desc.RIVER_DESCRIPTION_2 : Desc.RIVER_DESCRIPTION_1);
  }
  getRiverShape(): string {
    return this.riverShape_;
  }
  setRiverShape(shape: string): void {
    this.riverShape_ = shape;
  }
}

// Re-export so callers have a single import site for concrete tiles.
export type AnyTile = Grassland | Forest | AbundantForest | Mountain | River;
