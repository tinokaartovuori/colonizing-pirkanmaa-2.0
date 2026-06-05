// Port of DAL/objectmanager.{h,cpp}.

import { Coordinate } from '../core/coordinate';
import { ImageVectors } from '../core/images';
import { IObjectManager, IGameScene, IGameSettingsManager } from '../model/base';
import { TileBase } from '../model/tile';
import { HeadQuarters } from '../model/building';
import { PlayerBase } from '../model/player';
import { ClickedTileBorder, BlockedTile, MouseHoverBorder } from '../model/overlays';
import type { IGameEventHandler } from '../model/base';

interface ICurrentPlayerSource {
  getCurrentPlayer(): PlayerBase;
}

export class ObjectManager implements IObjectManager {
  private tiles_: TileBase[] = [];
  private hoverBorder_: MouseHoverBorder | null = null;
  private clickedTileBorder_: ClickedTileBorder | null = null;
  private blockedTileOverlays_: BlockedTile[] = [];
  private gameScene_!: IGameScene;
  private gameEventHandler_!: IGameEventHandler & ICurrentPlayerSource;
  private gameSettingsManager_!: IGameSettingsManager;

  setGameScene(gs: IGameScene): void {
    this.gameScene_ = gs;
  }
  getGameScene(): IGameScene {
    return this.gameScene_;
  }

  addDALS(
    gameeventhandler: IGameEventHandler & ICurrentPlayerSource,
    _menuobjectmanager: unknown,
    gamesettingsmanager: IGameSettingsManager,
  ): void {
    this.gameEventHandler_ = gameeventhandler;
    this.gameSettingsManager_ = gamesettingsmanager;
  }

  addTiles(tiles: TileBase[]): void {
    for (const tile of tiles) this.tiles_.push(tile);
  }

  replaceTile(oldTile: TileBase, newTile: TileBase): void {
    const idx = this.tiles_.indexOf(oldTile);
    if (idx !== -1) {
      this.tiles_.splice(idx, 1);
      this.tiles_.push(newTile);
      newTile.setOwner(oldTile.getOwner());
      for (const unit of oldTile.getUnits()) {
        unit.addParentTile(newTile);
        newTile.addUnit(unit);
      }
    } else {
      console.warn('Error, tile to be replaced was not found.');
    }
  }

  getTile(coordinate: Coordinate): TileBase | null {
    for (const tile of this.tiles_) {
      if (tile.getCoordinate().equals(coordinate)) return tile;
    }
    return null;
  }

  getTiles(): TileBase[] {
    return this.tiles_;
  }

  setHoverBorder(border: MouseHoverBorder): void {
    this.hoverBorder_ = border;
  }

  setClickedTileBorder(tile: TileBase): void {
    this.clickedTileBorder_ = new ClickedTileBorder(
      tile.getCoordinate(),
      1,
      1,
      this.gameEventHandler_,
      this,
    );
    this.clickedTileBorder_.setImageFiles(ImageVectors.CLICKEDTILEBORDER);
    this.gameScene_.drawItem(this.clickedTileBorder_);
  }

  getClickedTileBorder(): ClickedTileBorder | null {
    return this.clickedTileBorder_;
  }

  removeClickedTileBorder(): void {
    if (this.clickedTileBorder_ !== null) {
      this.gameScene_.removeItem(this.clickedTileBorder_);
    }
    this.clickedTileBorder_ = null;
  }

  getBorderTile(): MouseHoverBorder | null {
    return this.hoverBorder_;
  }

  getHqConnectedTiles(player: PlayerBase): TileBase[] {
    const tiles: TileBase[] = [];
    const hq = this.getHqTile(player);
    if (hq !== null) {
      tiles.push(hq);
      for (let i = 0; i < tiles.length; i++) {
        for (const neighbour of tiles[i].getNeighbourFourTiles()) {
          if (tiles.includes(neighbour)) continue;
          if (player === neighbour.getOwner()) tiles.push(neighbour);
        }
      }
    }
    return tiles;
  }

  getHqTile(player: PlayerBase): TileBase | null {
    for (const object of player.getObjects()) {
      if (object instanceof TileBase) {
        const building = object.getBuilding();
        if (building !== null && building.getType() === 'Headquarters') {
          if (!(building as HeadQuarters).isConquered()) {
            return object;
          }
        }
      }
    }
    return null;
  }

  getAvailableTiles(): TileBase[] {
    const player = this.gameEventHandler_.getCurrentPlayer();
    const availableTiles: TileBase[] = [];

    for (const obj of player.getObjects()) {
      if (obj instanceof TileBase) {
        const tile = obj;
        if (tile.getOwner() === player && tile.hasOpponentHeadquarters(player)) {
          if (!availableTiles.includes(tile)) availableTiles.push(tile);
        }

        if (tile.getType() === 'River' && tile.getBuilding() === null) {
          continue;
        } else {
          for (const nTile of tile.getNeighbourFourTiles()) {
            if (availableTiles.includes(nTile)) continue;
            if (nTile.hasOpponentHeadquarters(player)) availableTiles.push(nTile);
          }
        }
      }
    }
    return availableTiles;
  }

  addBlockTileOverlays(): void {
    const availableTiles = this.getAvailableTiles();
    const blockedTiles: TileBase[] = [];
    for (const tile of this.getTiles()) {
      if (!availableTiles.includes(tile)) blockedTiles.push(tile);
    }
    for (const blockedTile of blockedTiles) {
      const overlay = new BlockedTile(blockedTile.getCoordinate(), 1, 1, this.gameEventHandler_, this);
      overlay.setImageFiles(ImageVectors.BLOCKED_TILE);
      this.blockedTileOverlays_.push(overlay);
      this.gameScene_.drawItem(overlay);
    }
  }

  removeBlockTileOverlays(): void {
    for (const overlay of this.blockedTileOverlays_) {
      this.gameScene_.removeItem(overlay);
    }
    this.blockedTileOverlays_ = [];
  }

  /** The tile carrying the single Strange Device, if one exists (the one-per-game rule). */
  findStrangeDeviceTile(): TileBase | null {
    for (const tile of this.tiles_) {
      const b = tile.getBuilding();
      if (b !== null && b.getType() === 'Strange Device') return tile;
    }
    return null;
  }

  hasStrangeDevice(): boolean {
    return this.findStrangeDeviceTile() !== null;
  }

  getTileCount(): number {
    return this.tiles_.length;
  }

  getTileCountForPlayer(player: PlayerBase): number {
    let n = 0;
    for (const tile of this.tiles_) if (tile.getOwner() === player) n++;
    return n;
  }

  getNeutralTiles(): number {
    let n = 0;
    for (const tile of this.tiles_) if (tile.getOwner() === null) n++;
    return n;
  }
}
