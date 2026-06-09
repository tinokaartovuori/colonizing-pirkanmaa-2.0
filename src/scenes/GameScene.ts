// Phaser renderer for the map. Implements the IGameScene contract the model and
// GameEventHandler call into. Mirrors gamescene.cpp draw/update/remove plus the
// 450ms animation timer, hover border, mouse-follow item and tile click hit-test.

import Phaser from 'phaser';
import { BaseObject, IGameScene, ISceneObjectHandle, IGameSettingsManager } from '../model/base';
import { TileBase } from '../model/tile';
import { UnitBase } from '../model/unit';
import { BuildingBase, StrangeDevice } from '../model/building';
import { ClickedTileBorder, BlockedTile, MouseHoverBorder } from '../model/overlays';
import { Coordinate, Direction } from '../core/coordinate';
import { AnimationOption, ImageVectors } from '../core/images';
import type { ObjectManager } from '../managers/objectmanager';

interface RenderItem {
  obj: BaseObject;
  sprite: Phaser.GameObjects.Image;
  frames: string[];
  anim: AnimationOption;
  frame: number; // 1-indexed
  dir: number;
  randomizePending: boolean;
}

const DIR_ANGLE: Record<number, number> = {
  [Direction.N]: 0,
  [Direction.E]: 90,
  [Direction.S]: 180,
  [Direction.W]: 270,
};

export interface GameSceneInit {
  objectManager: ObjectManager;
  settings: IGameSettingsManager;
  onTileClick: (tile: TileBase) => void;
  /** Clicking a unit sprite directly; return true if it was handled (e.g. picked up to move). */
  onUnitClick?: (unit: UnitBase, tile: TileBase) => boolean;
  onReady: (scene: GameScene) => void;
}

export class GameScene extends Phaser.Scene implements IGameScene {
  private objectManager!: ObjectManager;
  private settings!: IGameSettingsManager;
  private onTileClick!: (tile: TileBase) => void;
  private onUnitClick?: (unit: UnitBase, tile: TileBase) => boolean;
  private onReadyCb!: (scene: GameScene) => void;

  private gridSize = 1;
  private items: Map<number, RenderItem> = new Map();
  private borderImages: Phaser.GameObjects.Image[] = [];
  private bordersDirty = true;
  /** Distinct marker (red tile + countdown number) for each Strange Device, keyed by the
   *  building's ID. The Device has no art yet, so this is how it reads on the map. */
  private deviceMarkers: Map<number, { image: Phaser.GameObjects.Image; text: Phaser.GameObjects.Text }> = new Map();

  private mousePicture: string[] = [];
  private mouseDragSprite: Phaser.GameObjects.Image | null = null;
  private hoverBorder: MouseHoverBorder | null = null;
  private animAccum = 0;

  constructor() {
    super({ key: 'GameScene' });
  }

  init(data: GameSceneInit): void {
    this.objectManager = data.objectManager;
    this.settings = data.settings;
    this.onTileClick = data.onTileClick;
    this.onUnitClick = data.onUnitClick;
    this.onReadyCb = data.onReady;
  }

  create(): void {
    // Restarting the scene (New Game / Quit) re-runs create() on the same scene
    // instance, but Phaser's shutdown has already destroyed every sprite. Drop the
    // stale bookkeeping so the animation loop never touches a destroyed sprite
    // (which threw "Cannot read properties of undefined (reading 'sys')").
    this.items.clear();
    this.borderImages = [];
    this.bordersDirty = true;
    this.deviceMarkers.clear();
    this.mousePicture = [];
    this.mouseDragSprite = null;
    this.animAccum = 0;

    this.gridSize = this.settings.getMapGridSize();
    this.cameras.main.setBackgroundColor('#35772c'); // QColor(53,119,44)
    this.hoverBorder = this.objectManager.getBorderTile();

    this.input.on('pointermove', (p: Phaser.Input.Pointer) => this.onPointerMove(p));
    this.input.on('pointerdown', (p: Phaser.Input.Pointer) => this.onPointerDown(p));

    this.onReadyCb(this);
  }

  // --- IGameScene -----------------------------------------------------------

  drawItem(obj: BaseObject): void {
    if (obj instanceof UnitBase) {
      this.drawUnit(obj);
      return;
    }
    const coord = obj.getCoordinatePtr();
    if (!coord) return;
    const depth = this.depthFor(obj);
    const sprite = this.add
      .image(coord.x() * this.gridSize, coord.y() * this.gridSize, this.firstTexture(obj))
      .setOrigin(0, 0)
      .setDisplaySize(obj.getWidth() * this.gridSize, obj.getHeight() * this.gridSize)
      .setDepth(depth);
    if (obj instanceof MouseHoverBorder) sprite.setAlpha(0.5);
    this.register(obj, sprite);
    if (obj instanceof TileBase) this.bordersDirty = true;
  }

  private drawUnit(unit: UnitBase): void {
    const parent = unit.getParentTile();
    if (!parent) return;
    const g = this.gridSize;
    const step = Math.round(g / 6);
    const rel = unit.getTileRelatedCoordinates();
    const absX = parent.getCoordinate().x() * g;
    const absY = parent.getCoordinate().y() * g;
    const x = absX + rel.x() * step * 2;
    const y = absY + rel.y() * step * 3;
    const sprite = this.add
      .image(x, y, this.firstTexture(unit))
      .setOrigin(0, 0)
      .setDisplaySize((g * 2) / 6, (g * 3) / 6)
      .setDepth(3);
    this.register(unit, sprite);
  }

  updateItem(obj: BaseObject): void {
    const item = this.items.get(obj.ID);
    if (!item) return;
    item.frames = obj.getImageFiles();
    item.anim = obj.getAnimationOption();
    if (item.frame > item.frames.length && item.frames.length >= 1) item.frame = 1;
    const coord = obj.getCoordinatePtr();
    if (coord && !(obj instanceof UnitBase)) {
      item.sprite.setPosition(coord.x() * this.gridSize, coord.y() * this.gridSize);
    }
    this.applyTexture(item);
    if (obj instanceof TileBase) this.bordersDirty = true;
  }

  updateTile(tile: TileBase): void {
    const building = tile.getBuilding();
    if (building !== null) {
      if (this.items.has(building.ID)) this.updateItem(building);
      else this.drawItem(building);
    }
    this.refreshDeviceMarker(tile);
    // Redraw units (remove all then re-add), as gamescene.cpp does.
    for (const unit of tile.getUnits()) this.removeItem(unit);
    if (tile.getUnitCount() > 0) for (const unit of tile.getUnits()) this.drawItem(unit);
    for (const unit of tile.getConqueringUnits()) this.removeItem(unit);
    if (tile.getConqueringUnitCount() > 0) for (const unit of tile.getConqueringUnits()) this.drawItem(unit);
    this.bordersDirty = true;
  }

  /** Draw/update the Strange Device tile: the device art (purple dome) plus the
   *  win-countdown number (rounds until its owner wins), positioned over the dome so both
   *  players can read the clock and see where to strike. */
  private refreshDeviceMarker(tile: TileBase): void {
    const building = tile.getBuilding();
    if (!(building instanceof StrangeDevice)) return;
    const g = this.gridSize;
    const c = tile.getCoordinate();
    const x = c.x() * g + g / 2;
    const y = c.y() * g + g / 2;
    // The dome sits a little above the tile centre; place the countdown on its face.
    const domeY = y - g * 0.1;
    const txt = String(building.getCountdown());
    let marker = this.deviceMarkers.get(building.ID);
    if (!marker || !marker.image.scene || !marker.text.scene) {
      // Device tile art (above terrain ≤ depth 1, below units at depth 3).
      const image = this.add.image(x, y, 'strange_device').setDisplaySize(g, g).setDepth(2);
      // Countdown number on top of everything so it stays readable on the dome.
      const text = this.add
        .text(x, domeY, txt, {
          fontFamily: '"PressStart2P", monospace',
          fontSize: `${Math.max(10, Math.round(g / 2.8))}px`,
          color: '#ffffff',
          stroke: '#2a0030',
          strokeThickness: Math.max(3, Math.round(g / 14)),
        })
        .setOrigin(0.5)
        .setDepth(4);
      marker = { image, text };
      this.deviceMarkers.set(building.ID, marker);
    } else {
      marker.image.setPosition(x, y);
      marker.text.setText(txt);
      marker.text.setPosition(x, domeY);
    }
  }

  removeItem(obj: BaseObject): void {
    const item = this.items.get(obj.ID);
    if (item) {
      item.sprite.destroy();
      this.items.delete(obj.ID);
      if (obj instanceof TileBase) this.bordersDirty = true;
    }
    // A destroyed Strange Device takes its marker (device art + countdown) with it.
    const marker = this.deviceMarkers.get(obj.ID);
    if (marker) {
      marker.image.destroy();
      marker.text.destroy();
      this.deviceMarkers.delete(obj.ID);
    }
  }

  isObjectInScene(obj: BaseObject): boolean {
    return this.items.has(obj.ID);
  }

  getObjectInScene(obj: BaseObject): ISceneObjectHandle | null {
    const item = this.items.get(obj.ID);
    if (!item) return null;
    return {
      setAnimationOption: (opt: AnimationOption) => {
        item.anim = opt;
      },
      setAnimationFrame: (frame: number) => {
        item.frame = frame;
        this.applyTexture(item);
      },
    };
  }

  addMouseFollowPicture(images: string[]): void {
    this.mousePicture = [...images];
  }

  removeMouseFollowItem(): void {
    this.mousePicture = [];
    if (this.mouseDragSprite) {
      this.mouseDragSprite.destroy();
      this.mouseDragSprite = null;
    }
  }

  deleteObjects(): void {
    for (const item of this.items.values()) item.sprite.destroy();
    this.items.clear();
    for (const b of this.borderImages) b.destroy();
    this.borderImages = [];
    for (const m of this.deviceMarkers.values()) {
      m.image.destroy();
      m.text.destroy();
    }
    this.deviceMarkers.clear();
    this.removeMouseFollowItem();
  }

  // --- internals ------------------------------------------------------------

  private register(obj: BaseObject, sprite: Phaser.GameObjects.Image): void {
    const anim = obj.getAnimationOption();
    this.items.set(obj.ID, {
      obj,
      sprite,
      frames: obj.getImageFiles(),
      anim,
      frame: 1,
      dir: 1,
      randomizePending: anim.randomFrame,
    });
  }

  private firstTexture(obj: BaseObject): string {
    const f = obj.getImageFiles();
    return f.length > 0 ? f[0] : ImageVectors.GRASSLAND[0];
  }

  private depthFor(obj: BaseObject): number {
    if (obj instanceof BuildingBase) return 1;
    if (obj instanceof ClickedTileBorder) return 10;
    if (obj instanceof MouseHoverBorder) return 10;
    if (obj instanceof BlockedTile) return 8;
    return 0;
  }

  private applyTexture(item: RenderItem): void {
    if (item.frames.length === 0) return;
    // Defensive: a destroyed sprite has no scene; never call setTexture on it.
    if (!item.sprite || !item.sprite.scene) return;
    const idx = Math.min(Math.max(item.frame - 1, 0), item.frames.length - 1);
    if (this.textures.exists(item.frames[idx])) item.sprite.setTexture(item.frames[idx]);
  }

  /** Replica of SceneItem::changeAnimationFrame. */
  private advanceFrame(item: RenderItem): void {
    if (!item.anim.animated) return;
    if (item.randomizePending) {
      item.frame = Math.floor(Math.random() * item.frames.length);
      item.randomizePending = false;
    }
    item.frame += item.dir;
    const n = item.frames.length;
    if (item.anim.style === 'rollover') {
      if (item.frame >= n + 1) item.frame = 1;
    } else {
      // backandforth
      if (item.frame <= 1) {
        item.dir = 1;
        item.frame = 1;
      }
      if (item.frame >= n) {
        item.frame = n;
        item.dir = -1;
      }
    }
    this.applyTexture(item);
  }

  override update(_time: number, delta: number): void {
    this.animAccum += delta;
    if (this.animAccum >= 450) {
      this.animAccum -= 450;
      for (const item of this.items.values()) this.advanceFrame(item);
    }
    if (this.bordersDirty) {
      this.refreshOwnerBorders();
      this.bordersDirty = false;
    }
  }

  private refreshOwnerBorders(): void {
    for (const b of this.borderImages) b.destroy();
    this.borderImages = [];
    for (const tile of this.objectManager.getTiles()) {
      const owner = tile.getOwner();
      if (!owner) continue;
      const key = ImageVectors.TILEOWNERBORDERS[owner.getPlayerNum() - 1];
      const coord = tile.getCoordinate();
      for (const dir of tile.getOwnerBorderDirections()) {
        const img = this.add
          .image(coord.x() * this.gridSize + this.gridSize / 2, coord.y() * this.gridSize + this.gridSize / 2, key)
          .setDisplaySize(this.gridSize, this.gridSize)
          .setAngle(DIR_ANGLE[dir] ?? 0)
          .setAlpha(0.6)
          // Above buildings (1) and the full-bleed Strange-Device art (marker 2) so
          // the territory outline is never hidden by tile art, but below units (3)
          // and the device countdown (4). At 0.5 the opaque device sprite painted
          // over the border, so a conquered region showed no edge around the Device.
          .setDepth(2.5);
        this.borderImages.push(img);
      }
    }
  }

  // --- input ----------------------------------------------------------------

  private pixelToTile(p: Phaser.Input.Pointer): Coordinate {
    return new Coordinate(Math.floor(p.worldX / this.gridSize), Math.floor(p.worldY / this.gridSize));
  }

  private onPointerMove(p: Phaser.Input.Pointer): void {
    // Mouse-follow drag item.
    if (this.mousePicture.length > 0) {
      if (!this.mouseDragSprite) {
        this.mouseDragSprite = this.add
          .image(p.worldX - 10, p.worldY - 15, this.mousePicture[0])
          .setOrigin(0, 0)
          .setDisplaySize(20, 30)
          .setDepth(11);
      } else {
        this.mouseDragSprite.setPosition(p.worldX - 10, p.worldY - 15);
      }
    } else if (this.mouseDragSprite) {
      this.mouseDragSprite.destroy();
      this.mouseDragSprite = null;
    }

    // Hover border.
    const width = this.settings.getMapGridWidth();
    const height = this.settings.getMapGridHeight();
    const pt = this.pixelToTile(p);
    if (!this.hoverBorder) this.hoverBorder = this.objectManager.getBorderTile();
    if (!this.hoverBorder) return;
    if (pt.x() < 0 || pt.x() > width - 1 || pt.y() < 0 || pt.y() > height - 1) {
      this.removeItem(this.hoverBorder);
      this.hoverBorder.setDrawn(false);
      return;
    }
    if (!this.isObjectInScene(this.hoverBorder)) {
      this.hoverBorder.setImageFiles(ImageVectors.MOUSEHOVERBORDER);
      this.drawItem(this.hoverBorder);
    }
    this.hoverBorder.setDrawn(true);
    this.hoverBorder.setCoordinate(pt);
    this.updateItem(this.hoverBorder);
  }

  private onPointerDown(p: Phaser.Input.Pointer): void {
    const width = this.settings.getMapGridWidth();
    const height = this.settings.getMapGridHeight();
    const pt = this.pixelToTile(p);
    if (pt.x() < 0 || pt.x() > width - 1 || pt.y() < 0 || pt.y() > height - 1) return;
    const tile = this.objectManager.getTile(pt);
    if (!tile) return;

    // Clicking a unit sprite directly picks it up to move (no menu needed). If
    // the handler declines (e.g. enemy unit, or already moving), fall through to
    // a normal tile click.
    if (this.onUnitClick) {
      const unit = this.unitAt(p.worldX, p.worldY);
      if (unit && this.onUnitClick(unit, unit.getParentTile() ?? tile)) return;
    }
    this.onTileClick(tile);
  }

  /** Topmost unit sprite under the given world point, if any. */
  private unitAt(worldX: number, worldY: number): UnitBase | null {
    let found: UnitBase | null = null;
    let bestDepth = -Infinity;
    for (const item of this.items.values()) {
      if (!(item.obj instanceof UnitBase)) continue;
      if (item.sprite.getBounds().contains(worldX, worldY)) {
        if (item.sprite.depth >= bestDepth) {
          bestDepth = item.sprite.depth;
          found = item.obj;
        }
      }
    }
    return found;
  }
}
