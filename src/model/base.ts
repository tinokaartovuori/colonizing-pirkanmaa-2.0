// Port of Core/baseobject, gameobject, placeablegameobject (+ the manager
// interfaces those objects call back into). References use plain pointers; the
// GC handles the cycles the C++ used weak_ptr for.

import { Coordinate } from '../core/coordinate';
import { AnimationOption, AnimationOptions } from '../core/images';
import { ResourceMap } from '../core/resources';
import type { TileBase } from './tile';
import type { BuildingBase } from './building';
import type { UnitBase } from './unit';
import type { PlayerBase } from './player';

// --- Manager interfaces (implemented by the DAL classes) -------------------

/** Handle to a drawn scene item; the renderer supplies the concrete object. */
export interface ISceneObjectHandle {
  setAnimationOption(opt: AnimationOption): void;
  setAnimationFrame(frame: number): void;
}

export interface IGameScene {
  drawItem(obj: BaseObject): void;
  removeItem(obj: BaseObject): void;
  updateItem(obj: BaseObject): void;
  updateTile(tile: TileBase): void;
  isObjectInScene(obj: BaseObject): boolean;
  getObjectInScene(obj: BaseObject): ISceneObjectHandle | null;
  addMouseFollowPicture(images: string[]): void;
  removeMouseFollowItem(): void;
  deleteObjects(): void;
}

export interface IGameEventHandler {
  updateTile(tile: TileBase): void;
  updateForest(status: string, tile: TileBase, building?: BuildingBase | null): void;
  updateAnimatedTileToStatic(tile: TileBase, frame: number): void;
  deleteUnitFromTile(unit: UnitBase, tile: TileBase): void;
  tileClicked(tile: TileBase): void;
}

export interface IObjectManager {
  getTile(coord: Coordinate): TileBase | null;
  getGameScene(): IGameScene;
  getAvailableTiles(): TileBase[];
  /** True if a Strange Device currently exists anywhere on the map (the one-per-game rule). */
  hasStrangeDevice(): boolean;
}

export interface IGameSettingsManager {
  getMapGridSize(): number;
  getMenuGridSize(): number;
  getMapWidth(): number;
  getMapHeight(): number;
  getMenuWidth(): number;
  getMenuHeight(): number;
  getMapGridWidth(): number;
  getMapGridHeight(): number;
}

// --- Object id counter (BaseObject::c_next_id) -----------------------------

let c_next_id = 0;

// --- BaseObject ------------------------------------------------------------

export class BaseObject {
  readonly ID: number;
  protected readonly EVENTHANDLER: IGameEventHandler;
  protected readonly OBJECTMANAGER: IObjectManager;
  private coordinate_: Coordinate | null;
  private imageFilePaths_: string[] = [];
  private m_animation_option: AnimationOption;
  protected m_width = 1;
  protected m_height = 1;

  constructor(
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    coordinate: Coordinate | null = null,
    width = 1,
    height = 1,
  ) {
    this.ID = c_next_id++;
    this.EVENTHANDLER = eventhandler;
    this.OBJECTMANAGER = objectmanager;
    this.coordinate_ = coordinate ? Coordinate.copy(coordinate) : null;
    this.m_animation_option = new AnimationOption(); // AnimationOptions::EMPTY equivalent
    this.m_width = width;
    this.m_height = height;
  }

  getID(): number {
    return this.ID;
  }

  setCoordinate(coordinate: Coordinate): void {
    this.coordinate_ = Coordinate.copy(coordinate);
  }

  getCoordinatePtr(): Coordinate | null {
    return this.coordinate_ ? Coordinate.copy(this.coordinate_) : null;
  }

  getCoordinate(): Coordinate {
    if (!this.coordinate_) throw new Error('BaseObject has no Coordinate.');
    return Coordinate.copy(this.coordinate_);
  }

  getType(): string {
    return 'BaseObject';
  }

  setImageFiles(imageVector: string[]): void {
    this.imageFilePaths_ = imageVector;
  }
  getImageFiles(): string[] {
    return this.imageFilePaths_;
  }

  setAnimationOption(option: AnimationOption): void {
    this.m_animation_option = option;
  }
  getAnimationOption(): AnimationOption {
    return this.m_animation_option;
  }

  getWidth(): number {
    return this.m_width;
  }
  getHeight(): number {
    return this.m_height;
  }

  protected lockEventHandler(): IGameEventHandler {
    return this.EVENTHANDLER;
  }
  protected lockObjectManager(): IObjectManager {
    return this.OBJECTMANAGER;
  }
}

// --- GameObject ------------------------------------------------------------

export class GameObject extends BaseObject {
  protected owner_: PlayerBase | null = null;

  constructor(
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
    opts: { coordinate?: Coordinate | null; width?: number; height?: number; owner?: PlayerBase | null } = {},
  ) {
    super(eventhandler, objectmanager, opts.coordinate ?? null, opts.width ?? 1, opts.height ?? 1);
    this.owner_ = opts.owner ?? null;
  }

  setOwner(owner: PlayerBase | null): void {
    const current = this.owner_;
    if (current && current !== owner) {
      current.removeObject(this);
    }
    if (owner && !owner.hasObject(this)) {
      owner.addObject(this);
    }
    this.owner_ = owner;
  }

  getOwner(): PlayerBase | null {
    return this.owner_;
  }

  getType(): string {
    return 'GameObject';
  }

  hasSameOwnerAs(other: GameObject): boolean {
    return this.getOwner() === other.getOwner();
  }
}

// --- PlaceableGameObject ---------------------------------------------------

export class PlaceableGameObject extends GameObject {
  private m_location: TileBase | null = null;

  constructor(eventhandler: IGameEventHandler, objectmanager: IObjectManager, owner: PlayerBase | null) {
    super(eventhandler, objectmanager, { owner });
  }

  getType(): string {
    return 'PlaceableGameObject';
  }

  canBePlacedOnTile(target: TileBase): boolean {
    if (target.getOwner() === null || this.getOwner() === null) return true;
    return this.hasSameOwnerAs(target);
  }

  setLocationTile(tile: TileBase | null): void {
    if (tile) {
      if (!this.canBePlacedOnTile(tile)) {
        throw new Error('IllegalAction for ' + this.getType());
      }
      this.setCoordinate(tile.getCoordinate());
      this.m_location = tile;
    }
  }

  /** Place on a tile WITHOUT the legality check — only for restoring a saved game,
   *  where placement legality (which depends on the *current* player) does not apply. */
  setLocationTileUnchecked(tile: TileBase): void {
    this.setCoordinate(tile.getCoordinate());
    this.m_location = tile;
  }

  currentLocationTile(): TileBase | null {
    return this.m_location;
  }

  getCost(): ResourceMap {
    return new Map();
  }
}

export { AnimationOptions };
