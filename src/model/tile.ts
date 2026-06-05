// Port of Tiles/tilebase.{h,cpp}.

import {
  ResourceMap,
  NO_RESOURCES,
  EMPTY,
  BasicResource,
  mergeResourceMaps,
  getNegativesMap,
} from '../core/resources';
import { Coordinate, Direction } from '../core/coordinate';
import { ImageVectors } from '../core/images';
import { GameObject, IGameEventHandler, IObjectManager, IGameSettingsManager } from './base';
import type { UnitBase } from './unit';
import type { BuildingBase } from './building';
import type { PlayerBase } from './player';

/** True when two resource maps have identical entries (used for NO_RESOURCES/EMPTY comparisons). */
function resourceMapsEqual(a: ResourceMap, b: ResourceMap): boolean {
  if (a.size !== b.size) return false;
  for (const [k, v] of a) {
    if (b.get(k) !== v) return false;
  }
  return true;
}

export abstract class TileBase extends GameObject {
  readonly MAX_UNITS: number;
  readonly BASE_PRODUCTION: ResourceMap;
  private basicDescription_: string;
  private building_: BuildingBase | null = null;
  protected gameSettingsManager_!: IGameSettingsManager;
  protected units_: UnitBase[] = [];
  protected conqueringUnits_: UnitBase[] = [];

  constructor(
    location: Coordinate,
    size_x: number,
    size_y: number,
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    max_units: number,
    production: ResourceMap,
    basic_description = '',
  ) {
    super(eventhandler, objectmanager, { coordinate: location, width: size_x, height: size_y });
    this.MAX_UNITS = max_units;
    this.BASE_PRODUCTION = production;
    this.basicDescription_ = basic_description;
  }

  getType(): string {
    return 'TileBase';
  }

  addUnit(unit: UnitBase): void {
    if (unit.getType() === 'BasicWorker') unit.setImageFiles(ImageVectors.BASICWORKER);
    if (unit.getType() === 'Expert') unit.setImageFiles(ImageVectors.EXPERT);
    if (unit.getType() === 'Soldier') unit.setImageFiles(ImageVectors.SOLDIER);

    if (!unit.isConqueringUnit()) {
      // A Strange Device tile never holds defenders (see hasSpaceForUnits) — refuse
      // any owner unit so it stays crackable.
      if (this.getBuilding()?.getType() === 'Strange Device') {
        throw new Error('Cannot place units on a Strange Device tile!');
      }
      if (this.getUnitCount() + 1 > this.MAX_UNITS) {
        throw new Error('Tile has no more room for Units!');
      }
      unit.setLocationTile(this);
      this.units_.push(unit);
    } else {
      if (this.getConqueringUnitCount() + 1 > this.MAX_UNITS) {
        throw new Error('Tile has no more room for conquering units!');
      }
      unit.setLocationTile(this);
      this.conqueringUnits_.push(unit);
    }
    // Authoritatively (re)assign every unit's in-tile draw slot by array order, so up to
    // three sit side-by-side and never overlap. Without this, paths that add a unit but
    // forget to renumber (buying & placing a unit, aiBuyAndPlaceUnit, replaceTile) left
    // the newcomer on a stale/duplicate offset — the intermittent visual stacking bug.
    // Mirrors removeUnit(), which already renumbers after a removal.
    this.updateUnitCoordinates();
  }

  /** Add a unit while restoring a saved game — skips the placement-legality check
   *  (which depends on the *current* player and so wrongly rejected other players'
   *  units during restore, leaving the enemy unstaffed and doomed to collapse). */
  addUnitRestored(unit: UnitBase): void {
    if (unit.getType() === 'BasicWorker') unit.setImageFiles(ImageVectors.BASICWORKER);
    if (unit.getType() === 'Expert') unit.setImageFiles(ImageVectors.EXPERT);
    if (unit.getType() === 'Soldier') unit.setImageFiles(ImageVectors.SOLDIER);
    unit.setLocationTileUnchecked(this);
    if (unit.isConqueringUnit()) this.conqueringUnits_.push(unit);
    else this.units_.push(unit);
  }

  removeUnit(unit: UnitBase): void {
    let idx = this.units_.indexOf(unit);
    if (idx !== -1) this.units_.splice(idx, 1);
    idx = this.conqueringUnits_.indexOf(unit);
    if (idx !== -1) this.conqueringUnits_.splice(idx, 1);
    this.updateUnitCoordinates();
    this.lockEventHandler().updateTile(this);
  }

  addBuilding(building: BuildingBase): void {
    if (this.getType() === 'Forest') {
      this.lockEventHandler().updateForest('Grassland', this, building);
      return;
    }
    building.setParentTile(this);
    building.setLocationTile(this);
    this.building_ = building;
    this.updateUnitCoordinates();
  }

  getBuilding(): BuildingBase | null {
    return this.building_;
  }

  /** Direct setter used when an object manager copies a building onto a replacement tile. */
  setBuildingDirect(building: BuildingBase | null): void {
    this.building_ = building;
  }

  conquerTile(currentPlayer: PlayerBase): void {
    // If no one owns the tile that has the player's unit on, the player gets it.
    for (const unit of this.getConqueringUnits()) {
      if (unit.getOwner() === currentPlayer && this.getOwner() === null) {
        this.setOwner(currentPlayer);
        unit.setAsConquering(false);
        for (const u of this.conqueringUnits_) {
          u.setAsConquering(false);
          this.units_.push(u);
        }
        this.conqueringUnits_ = [];
      }
    }

    // If someone else owns the tile, conquer if more soldiers than the opponent.
    if (this.getOwner() !== currentPlayer && this.getOwner() !== null) {
      const ownSoldiers = this.getSoldierCount();
      const opponentSoldiers = this.getOpponentSoldierCount();

      let hasOutpost = false;
      if (this.getBuilding() !== null && this.getBuilding()!.getType() === 'Outpost') {
        hasOutpost = true;
      }

      if (ownSoldiers > opponentSoldiers && !hasOutpost) {
        this.setOwner(currentPlayer);

        if (this.getBuilding() !== null && this.getBuilding()!.getType() === 'Headquarters') {
          // Conquered HQ image/flag update.
          (this.getBuilding() as unknown as { setConquered(): void }).setConquered();
        }

        const units = [...this.units_];
        for (const unit of units) {
          this.lockEventHandler().deleteUnitFromTile(unit, this);
        }
        for (const unit of this.conqueringUnits_) {
          unit.setAsConquering(false);
          this.units_.push(unit);
        }
        this.conqueringUnits_ = [];
      } else {
        const units = [...this.conqueringUnits_];
        for (const unit of units) {
          this.lockEventHandler().deleteUnitFromTile(unit, this);
        }
      }
    }

    this.updateUnitCoordinates();
    this.lockEventHandler().updateTile(this);
  }

  hasOpponentHeadquarters(_player: PlayerBase): boolean {
    return true;
  }

  getMaxUnitsIncrease(): number {
    return 0;
  }
  getMaxSoldiersIncrease(): number {
    return 0;
  }

  getUnitCount(): number {
    return this.units_.length;
  }
  getConqueringUnitCount(): number {
    return this.conqueringUnits_.length;
  }

  getSoldierCount(): number {
    let n = 0;
    for (const unit of this.getConqueringUnits()) if (unit.getType() === 'Soldier') n++;
    return n;
  }
  getOpponentSoldierCount(): number {
    let n = 0;
    for (const unit of this.getUnits()) if (unit.getType() === 'Soldier') n++;
    return n;
  }

  getUnits(): UnitBase[] {
    return this.units_;
  }
  getConqueringUnits(): UnitBase[] {
    return this.conqueringUnits_;
  }

  updateAnimation(): void {
    /* base: no-op */
  }

  getCurrentExpenses(): ResourceMap {
    let expenses = NO_RESOURCES;
    for (const unit of this.getUnits()) {
      expenses = mergeResourceMaps(expenses, unit.getSalary());
    }
    expenses = getNegativesMap(expenses);
    if (this.getBuilding() !== null) {
      expenses = mergeResourceMaps(expenses, getNegativesMap(this.getBuilding()!.getProduction()));
    }
    return expenses;
  }

  getCurrentNet(): ResourceMap {
    const revenue = this.getCurrentRevenue();
    const expenses = this.getCurrentExpenses();
    return mergeResourceMaps(revenue, expenses);
  }

  updateUnitCoordinates(): void {
    let ind = 0;
    for (const u of this.units_) {
      u.setTileRelatedCoordinates(ind, 1);
      ind++;
    }
    ind = 0;
    for (const u of this.conqueringUnits_) {
      u.setTileRelatedCoordinates(ind, 0);
      ind++;
    }
  }

  getNeighbourFourTiles(): TileBase[] {
    const coord = this.getCoordinatePtr()!;
    const width = this.gameSettingsManager_.getMapGridWidth();
    const height = this.gameSettingsManager_.getMapGridHeight();
    const result: TileBase[] = [];
    for (const nc of coord.neighbouringFour(width, height)) {
      const tile = this.lockObjectManager().getTile(nc);
      if (tile) result.push(tile);
    }
    return result;
  }

  getNeighbourTiles(): TileBase[] {
    const coord = this.getCoordinatePtr()!;
    const width = this.gameSettingsManager_.getMapGridWidth();
    const height = this.gameSettingsManager_.getMapGridHeight();
    const result: TileBase[] = [];
    for (const nc of coord.neighbours(1, width, height)) {
      const tile = this.lockObjectManager().getTile(nc);
      if (tile) result.push(tile);
    }
    return result;
  }

  hasSpaceForUnits(): boolean {
    // The Strange Device tile holds NO defending units: otherwise the owner could
    // garrison it to the cap (3) and make it impossible to conquer, defeating the
    // whole mechanic. Conquering (attacking) units may still stage here
    // (hasSpaceForConqueringUnits is unchanged), so with zero defenders a single
    // attacker can crack it.
    if (this.getBuilding()?.getType() === 'Strange Device') return false;
    return 1 + this.getUnitCount() <= this.MAX_UNITS;
  }
  hasSpaceForConqueringUnits(): boolean {
    return 1 + this.getConqueringUnitCount() <= this.MAX_UNITS;
  }

  setGameSettings(manager: IGameSettingsManager): void {
    this.gameSettingsManager_ = manager;
  }

  clickAction(): void {
    this.lockEventHandler().tileClicked(this);
  }

  /**
   * Which compass sides need an owner-colour border drawn (a side gets a border
   * when its neighbour is off-map or owned by someone else). Renderer-facing
   * replacement for getOwnerBorderPixmap().
   */
  getOwnerBorderDirections(): Direction[] {
    if (this.getOwner() === null) return [];
    const coord = this.getCoordinatePtr()!;
    const width = this.gameSettingsManager_.getMapGridWidth();
    const height = this.gameSettingsManager_.getMapGridHeight();
    const om = this.lockObjectManager();
    const sides: Array<{ dir: Direction; guard: boolean; neighbour: Coordinate }> = [
      { dir: Direction.N, guard: coord.y() !== 0, neighbour: coord.neighbour_at(Direction.N, 1) },
      { dir: Direction.E, guard: coord.x() !== width - 1, neighbour: coord.neighbour_at(Direction.E, 1) },
      { dir: Direction.S, guard: coord.y() !== height - 1, neighbour: coord.neighbour_at(Direction.S, 1) },
      { dir: Direction.W, guard: coord.x() !== 0, neighbour: coord.neighbour_at(Direction.W, 1) },
    ];
    const result: Direction[] = [];
    for (const s of sides) {
      const neighbourTile = s.guard ? om.getTile(s.neighbour) : null;
      if (neighbourTile === null || neighbourTile.getOwner() !== this.getOwner()) {
        result.push(s.dir);
      }
    }
    return result;
  }

  addBasicDescription(desc: string): void {
    this.basicDescription_ = desc;
  }
  getBasicDescription(): string {
    return this.basicDescription_;
  }

  getNetDescription(): string {
    let functionalString = '<u>Net value:</u>';
    const net = this.getCurrentNet();
    if (resourceMapsEqual(net, NO_RESOURCES) || resourceMapsEqual(net, EMPTY)) {
      return '';
    }
    functionalString += this.netLines(net);
    return functionalString;
  }

  protected netLines(net: ResourceMap): string {
    let s = '';
    const order: Array<[BasicResource, string]> = [
      [BasicResource.MONEY, 'Money'],
      [BasicResource.WOOD, 'Wood'],
      [BasicResource.STONE, 'Stone'],
      [BasicResource.METAL, 'Metal'],
    ];
    for (const [res, label] of order) {
      const v = net.get(res);
      if (v !== undefined && v !== 0) s += `<br>${v} ${label}/r`;
    }
    return s;
  }

  abstract getCurrentRevenue(): ResourceMap;
  abstract getExtraDescription(): string;
  abstract getBuildableBuildings(): string[];
  abstract generateResources(): void;
}
