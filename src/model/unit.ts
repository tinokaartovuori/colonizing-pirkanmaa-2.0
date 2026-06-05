// Port of Units/* (UnitBase + BasicWorker, Expert, Soldier).

import {
  ResourceMap,
  BASIC_WORKER_COST,
  BASIC_WORKER_SALARY,
  EXPERT_COST,
  EXPERT_SALARY,
  SOLDIER_COST,
  SOLDIER_SALARY,
} from '../core/resources';
import { Coordinate } from '../core/coordinate';
import { PlaceableGameObject, IGameEventHandler, IObjectManager, IGameSettingsManager } from './base';
import type { TileBase } from './tile';
import type { PlayerBase } from './player';

export abstract class UnitBase extends PlaceableGameObject {
  private gameSettingsManager_: IGameSettingsManager;
  private tileRelativeCoordinate_: Coordinate;
  private parentTile_: TileBase | null;
  private isConqueringUnit_ = false;

  constructor(
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    gamesettingsmanager: IGameSettingsManager,
    owner: PlayerBase | null,
    parenttile: TileBase | null = null,
  ) {
    super(eventhandler, objectmanager, owner);
    this.gameSettingsManager_ = gamesettingsmanager;
    this.parentTile_ = parenttile;
    this.tileRelativeCoordinate_ = new Coordinate(0, 0);
    if (parenttile) {
      // Mirrors the parented constructor's (buggy but faithful) branch.
      if (parenttile.getOwner() !== this.getOwner()) {
        this.tileRelativeCoordinate_ = new Coordinate(parenttile.getUnitCount(), 1);
        this.isConqueringUnit_ = false;
      } else {
        this.tileRelativeCoordinate_ = new Coordinate(parenttile.getConqueringUnitCount(), 0);
        this.isConqueringUnit_ = true;
      }
    }
  }

  getType(): string {
    return 'UnitBase';
  }

  addParentTile(tile: TileBase): void {
    this.parentTile_ = tile;
    if (tile.getOwner() === this.getOwner()) {
      this.tileRelativeCoordinate_ = new Coordinate(tile.getUnitCount(), 1);
      this.isConqueringUnit_ = false;
    } else {
      this.tileRelativeCoordinate_ = new Coordinate(tile.getConqueringUnitCount(), 0);
      this.isConqueringUnit_ = true;
    }
  }

  getParentTile(): TileBase | null {
    return this.parentTile_;
  }

  updateParentTile(): void {
    if (!this.parentTile_) return;
    if (this.parentTile_.getOwner() === this.getOwner()) {
      this.setTileRelatedCoordinates(this.parentTile_.getUnitCount(), 1);
      this.isConqueringUnit_ = false;
    } else {
      this.setTileRelatedCoordinates(this.parentTile_.getUnitCount(), 1);
      this.isConqueringUnit_ = true;
    }
  }

  canBePlacedOnTile(target: TileBase): boolean {
    if (
      (this.isConqueringUnit_ && target.hasSpaceForConqueringUnits()) ||
      (!this.isConqueringUnit_ && target.hasSpaceForUnits())
    ) {
      const availableTiles = this.lockObjectManager().getAvailableTiles();
      return availableTiles.includes(target);
    }
    return false;
  }

  getGridSize(): number {
    return this.gameSettingsManager_.getMapGridSize();
  }

  getTileRelatedCoordinates(): Coordinate {
    return this.tileRelativeCoordinate_;
  }

  setTileRelatedCoordinates(x: number, y: number): void {
    this.tileRelativeCoordinate_.set_x(x);
    this.tileRelativeCoordinate_.set_y(y);
  }

  paySalary(): void {
    this.owner_?.addOrRemoveResources(this.getSalary());
  }

  isConqueringUnit(): boolean {
    return this.isConqueringUnit_;
  }

  setAsConquering(isConquering: boolean): void {
    this.isConqueringUnit_ = isConquering;
  }

  abstract getSalary(): ResourceMap;
  abstract getCost(): ResourceMap;
}

export class BasicWorker extends UnitBase {
  getType(): string {
    return 'BasicWorker';
  }
  getSalary(): ResourceMap {
    return BASIC_WORKER_SALARY;
  }
  getCost(): ResourceMap {
    return BASIC_WORKER_COST;
  }
}

export class Expert extends UnitBase {
  getType(): string {
    return 'Expert';
  }
  getSalary(): ResourceMap {
    return EXPERT_SALARY;
  }
  getCost(): ResourceMap {
    return EXPERT_COST;
  }
}

export class Soldier extends UnitBase {
  getType(): string {
    return 'Soldier';
  }
  getSalary(): ResourceMap {
    return SOLDIER_SALARY;
  }
  getCost(): ResourceMap {
    return SOLDIER_COST;
  }
}
