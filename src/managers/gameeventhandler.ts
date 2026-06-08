// Port of DAL/gameeventhandler.{h,cpp}.

import {
  ResourceMap,
  NO_RESOURCES,
  mergeResourceMaps,
  BasicResource,
  cloneResourceMap,
  strangeDeviceCountdown,
} from '../core/resources';
import { ImageVectors, AnimationOptions } from '../core/images';
import { Coordinate } from '../core/coordinate';
import { IGameEventHandler, IGameScene, IGameSettingsManager } from '../model/base';
import { TileBase } from '../model/tile';
import { Grassland, River, Forest } from '../model/tiles';
import { UnitBase, BasicWorker, Expert, Soldier } from '../model/unit';
import {
  BuildingBase,
  HeadQuarters,
  Farm,
  Village,
  Outpost,
  NuclearPlant,
  Mine,
  HydroPower,
  Bridge,
  StrangeDevice,
} from '../model/building';
import { PlayerBase } from '../model/player';
import { ObjectManager } from './objectmanager';
import { PlayerManager } from './playermanager';
import { IMenuObjectManager } from './menu-interface';
import type { GameSnapshot } from './persistence';

export class GameEventHandler implements IGameEventHandler {
  private gameScene_!: IGameScene;
  private unitToDeploy_: UnitBase | null = null;
  private unitPreviousTile_: TileBase | null = null;
  /** When true, menu views are suppressed so a CPU can act without UI churn. */
  private aiActive_ = false;
  /** Set once a winner is decided (a sole survivor) or the game ties out. The match is
   *  over: every action entry point below no-ops so play can never continue past the win. */
  private gameOver_ = false;

  /** True once the match has ended (one winner left, or a tie). */
  isGameOver(): boolean {
    return this.gameOver_;
  }

  /** App-supplied restart callback (replaces the Qt restartGameSignal). */
  onRestart: (() => void) | null = null;

  /**
   * Fired after the active player changes (HQ placement or end of turn), so the
   * app layer can drive CPU turns and show the "it's your turn" banner. Not used
   * by the headless tests.
   */
  onTurnChanged: (() => void) | null = null;

  private notifyTurnChanged(): void {
    this.onTurnChanged?.();
  }

  /** Toggle menu suppression while a CPU player takes its turn. */
  setAiActive(active: boolean): void {
    this.aiActive_ = active;
  }

  constructor(
    private objectManager_: ObjectManager,
    private playerManager_: PlayerManager,
    private menuObjectManager_: IMenuObjectManager,
    private gameSettingsManager_: IGameSettingsManager,
  ) {}

  setGameScene(gs: IGameScene): void {
    this.gameScene_ = gs;
  }

  getCurrentPlayer(): PlayerBase {
    return this.playerManager_.getCurrentPlayer();
  }

  /** Build a Headquarters for a player with the right per-colour sprite. */
  private makeHeadquarters(player: PlayerBase): HeadQuarters {
    const HQ = new HeadQuarters(this, this.objectManager_, player);
    switch (player.getPlayerNum()) {
      case 1: HQ.setImageFiles(ImageVectors.HEADQUARTERSONE); break;
      case 2: HQ.setImageFiles(ImageVectors.HEADQUARTERSTWO); break;
      case 3: HQ.setImageFiles(ImageVectors.HEADQUARTERSTHREE); break;
      case 4: HQ.setImageFiles(ImageVectors.HEADQUARTERSFOUR); break;
    }
    HQ.setAnimationOption(AnimationOptions.HEADQUARTERS);
    return HQ;
  }

  // --- save/restore ---------------------------------------------------------

  /** Place a building on a tile without charging for it (used when restoring a save). */
  private placeBuildingDirect(building: BuildingBase, tile: TileBase, owner: PlayerBase | null): void {
    building.setParentTile(tile);
    building.setLocationTile(tile);
    if (owner) building.setOwner(owner);
    tile.setBuildingDirect(building);
  }

  /** Place a unit on a tile without charging for it (used when restoring a save). */
  private placeUnitDirect(type: string, tile: TileBase, owner: PlayerBase, conquering: boolean): void {
    const unit = this.makeUnit(type, owner);
    if (!unit) return;
    unit.setOwner(owner); // also registers the unit with the player (for unit counts)
    unit.setAsConquering(conquering);
    unit.addParentTile(tile);
    // Use the restore path: addUnit() runs a placement-legality check tied to the
    // *current* player, which silently rejected other players' units during restore.
    tile.addUnitRestored(unit);
  }

  /**
   * Re-apply a saved game's mutable state on top of freshly-generated terrain. Assumes
   * every player has already been reconstructed (in original order, so player numbers
   * match) and the map has been generated from the same seed. Restores tile ownership,
   * buildings + their state, units, resources, then the turn bookkeeping.
   */
  restoreSnapshot(snap: GameSnapshot): void {
    const players = this.playerManager_.getPlayers();
    const byNum = (n: number | null): PlayerBase | null =>
      n == null ? null : (players.find((p) => p.getPlayerNum() === n) ?? null);

    for (const ts of snap.tiles) {
      const tile = this.objectManager_.getTile(new Coordinate(ts.x, ts.y));
      if (!tile) continue;
      const owner = byNum(ts.owner);

      if (ts.b) {
        const bOwner = byNum(ts.b.owner) ?? owner;
        let building = tile.getBuilding();
        if (!building) {
          // Mikontalo is placed by terrain generation; everything else we build here.
          building = ts.b.type === 'Headquarters' && bOwner ? this.makeHeadquarters(bOwner) : this.makeBuilding(ts.b.type, tile, bOwner!);
          if (building) this.placeBuildingDirect(building, tile, bOwner);
        } else if (bOwner) {
          building.setOwner(bOwner);
        }
        if (building) {
          if (ts.b.growthPhase !== undefined && building.getType() === 'Farm') (building as Farm).setGrowthPhase(ts.b.growthPhase);
          if (ts.b.conquered && building.getType() === 'Headquarters') (building as HeadQuarters).setConquered();
          if (ts.b.countdown !== undefined && building.getType() === 'Strange Device') (building as StrangeDevice).setCountdown(ts.b.countdown);
        }
      }

      if (tile instanceof Forest && ts.forest) tile.setHarvestState(ts.forest.wood, ts.forest.stumps);
      if (owner) tile.setOwner(owner);
      for (const u of ts.units ?? []) {
        const uo = byNum(u.owner);
        if (uo) this.placeUnitDirect(u.type, tile, uo, false);
      }
      for (const u of ts.conq ?? []) {
        const uo = byNum(u.owner);
        if (uo) this.placeUnitDirect(u.type, tile, uo, true);
      }

      this.updateTile(tile);
      tile.updateUnitCoordinates();
      this.gameScene_.updateTile(tile);
    }

    // Resources.
    for (const [num, vals] of Object.entries(snap.resources)) {
      const p = byNum(Number(num));
      if (!p) continue;
      const map: ResourceMap = new Map([
        [BasicResource.MONEY, vals[0]],
        [BasicResource.WOOD, vals[1]],
        [BasicResource.STONE, vals[2]],
        [BasicResource.METAL, vals[3]],
      ]);
      p.setResources(cloneResourceMap(map));
    }

    // Turn bookkeeping (removes lost players, sets the rounds counter + current turn).
    this.playerManager_.restore(snap.currentPlayerNum, snap.roundsPlayed, snap.lostPlayerNums);
  }

  firstRoundActions(tile: TileBase): void {
    if (tile.getBuilding() !== null) return;

    const current = this.playerManager_.getCurrentPlayer();
    const HQ = this.makeHeadquarters(current);
    tile.addBuilding(HQ);
    tile.setOwner(current);
    this.updateTile(tile);

    const neighbours = tile
      .getCoordinatePtr()!
      .neighbours(1, this.gameSettingsManager_.getMapGridWidth(), this.gameSettingsManager_.getMapGridHeight());

    for (const nc of neighbours) {
      const neighbour = this.objectManager_.getTile(nc);
      if (neighbour && neighbour.getOwner() === null) {
        neighbour.setOwner(current);
        const b = neighbour.getBuilding();
        if (b !== null && b.getType() === 'Mikontalo') {
          b.setOwner(current);
        }
      }
    }

    // Disconnect any tile not connected to the HQ.
    const hqConnected = this.objectManager_.getHqConnectedTiles(current);
    for (const object of current.getObjects()) {
      if (object instanceof TileBase) {
        if (!hqConnected.includes(object)) object.setOwner(null);
      }
    }

    this.playerManager_.changeTurn();

    if (this.playerManager_.getCurrentPlayer().getObjects().length === 0) {
      this.menuObjectManager_.selectFirstTileMenuView(this.playerManager_.getCurrentPlayer());
    } else {
      this.openDefaultMenuView();
    }

    this.notifyTurnChanged();
  }

  tileClicked(tile: TileBase): void {
    if (this.gameOver_) return;
    if (this.playerManager_.getPlayers().length <= 1) return;

    let lastTileCoord = new Coordinate(-1, -1);
    const ctb = this.objectManager_.getClickedTileBorder();
    if (ctb !== null) lastTileCoord = ctb.getCoordinate();
    this.objectManager_.removeClickedTileBorder();

    const current = this.playerManager_.getCurrentPlayer();

    if (current.getObjects().length === 0 && tile.getType() === 'Grassland' && tile.getOwner() === null) {
      this.firstRoundActions(tile);
      this.gameScene_.updateTile(tile);
    } else if (current.getObjects().length !== 0) {
      if (this.unitToDeploy_ !== null) {
        if (this.unitPreviousTile_ !== null) {
          // Dropped back on the tile it came from — just cancel (put the unit back).
          if (tile === this.unitPreviousTile_) {
            const prev = this.unitPreviousTile_;
            this.cancelUnitAction();
            this.setTileInspectionMenuView(prev);
            return;
          }
          // Moving a unit between tiles. (It was lifted off unitPreviousTile_ on pickup,
          // so the removeUnit below is a no-op kept for parity with the original.)
          try {
            this.objectManager_.setClickedTileBorder(this.unitPreviousTile_);
            this.unitToDeploy_.setOwner(current);
            this.unitToDeploy_.setAsConquering(tile.getOwner() !== current);
            tile.addUnit(this.unitToDeploy_);
            this.unitToDeploy_.addParentTile(tile);
            this.unitPreviousTile_.removeUnit(this.unitToDeploy_);
            tile.updateAnimation();
            this.unitPreviousTile_.updateAnimation();
            this.gameScene_.removeMouseFollowItem();
            this.objectManager_.removeBlockTileOverlays();
            tile.updateUnitCoordinates();
            this.unitPreviousTile_.updateUnitCoordinates();
            this.gameScene_.updateTile(tile);
            this.gameScene_.updateTile(this.unitPreviousTile_);
            this.unitToDeploy_ = null;
            this.setTileInspectionMenuView(this.unitPreviousTile_);
            this.unitPreviousTile_ = null;
          } catch {
            return;
          }
        } else {
          // Placing a freshly-bought unit.
          try {
            if (this.canBuyUnitOrBuilding(this.unitToDeploy_)) {
              this.unitToDeploy_.addParentTile(tile);
              this.unitToDeploy_.setOwner(current);
              tile.addUnit(this.unitToDeploy_);
              this.buyUnitOrBuilding(this.unitToDeploy_);
            } else {
              this.unitToDeploy_ = null;
              this.unitPreviousTile_ = null;
              this.gameScene_.removeMouseFollowItem();
              this.objectManager_.removeBlockTileOverlays();
              return;
            }
            this.unitToDeploy_ = null;
            this.unitPreviousTile_ = null;
            this.gameScene_.removeMouseFollowItem();
            this.objectManager_.removeBlockTileOverlays();
            tile.updateAnimation();
            this.gameScene_.updateTile(tile);
            this.menuObjectManager_.setUnitShopMenuView();
          } catch {
            return;
          }
        }
      } else {
        if (!lastTileCoord.equals(tile.getCoordinate())) {
          this.setTileInspectionMenuView(tile);
          this.objectManager_.setClickedTileBorder(tile);
        } else {
          this.openDefaultMenuView();
        }
      }
    }
  }

  endTurn(): void {
    if (this.gameOver_) return; // the match is decided — no further turns
    const losingReasons: string[] = [];
    const current = this.playerManager_.getCurrentPlayer();

    // Generate resources from the current player's tiles.
    for (const object of current.getObjects()) {
      if (object instanceof TileBase) object.generateResources();
    }
    // Pay salaries.
    for (const object of current.getObjects()) {
      if (object instanceof UnitBase) object.paySalary();
    }
    // Conquer.
    for (const tile of this.objectManager_.getTiles()) {
      tile.conquerTile(this.playerManager_.getCurrentPlayer());
    }

    // HQ-connectivity for every opponent. NOTE: both loops iterate *copies* —
    // setOwner() removes the tile from player.getObjects() and deleteUnitFromTile()
    // splices tile.getUnits(), so iterating the live arrays would skip entries and
    // leave stranded units/tiles behind (the "cut units stay on the map" bug).
    for (const player of this.playerManager_.getPlayers()) {
      if (player !== this.playerManager_.getCurrentPlayer()) {
        const hqConnected = this.objectManager_.getHqConnectedTiles(player);
        for (const object of [...player.getObjects()]) {
          if (object instanceof TileBase) {
            const tile = object;
            if (!hqConnected.includes(tile) && hqConnected.length === 0) {
              this.clearTileUnits(tile);
              tile.setOwner(this.playerManager_.getCurrentPlayer());
              const b = tile.getBuilding();
              if (b !== null && b.getType() === 'Farm') (b as Farm).resetFarm();
            } else if (!hqConnected.includes(tile)) {
              this.clearTileUnits(tile);
              tile.setOwner(null);
              const b = tile.getBuilding();
              if (b !== null && b.getType() === 'Farm') (b as Farm).resetFarm();
            }
          }
        }
        player.eliminateExcessUnits();
        player.limitResources();
      }
    }

    const lostPlayersThisRound: PlayerBase[] = [];

    // Lost: no tiles.
    for (const player of this.playerManager_.getPlayers()) {
      if (player.getObjects().length === 0) {
        losingReasons.push('conquered');
        this.playerManager_.setPlayerAsLost(player, this.playerManager_.getCurrentPlayer());
        lostPlayersThisRound.push(player);
      }
    }

    // Lost: negative resources.
    for (const player of this.playerManager_.getPlayers()) {
      for (const value of player.getResources().values()) {
        if (value < 0) {
          if (player.getObjects().length > 0) {
            losingReasons.push('noresources');
            this.playerManager_.setPlayerAsLost(player, this.playerManager_.getCurrentPlayer());
            lostPlayersThisRound.push(player);
            this.neutralizePlayer(player);
            break;
          }
        }
      }
    }

    // Win: a player owns >= 70% of tiles. Iterate SNAPSHOTS — setPlayerAsLost splices the
    // live players array, so iterating it directly skips entries and (with 3+ players) left
    // some opponents un-eliminated, so the match never collapsed to a single winner.
    for (const player of [...this.playerManager_.getPlayers()]) {
      if (
        Math.trunc(
          (this.objectManager_.getTileCountForPlayer(player) * 100) / this.objectManager_.getTileCount(),
        ) >= 70
      ) {
        for (const p of [...this.playerManager_.getPlayers()]) {
          if (player !== p) {
            this.playerManager_.setPlayerAsLost(p);
            lostPlayersThisRound.push(p);
            this.neutralizePlayer(p);
          }
        }
        break; // a winner is decided
      }
    }

    // --- Strange Device -----------------------------------------------------
    // A Device whose tile was captured this turn, or cut off from its builder by the
    // HQ-connectivity rule (or its builder neutralised), is DESTROYED: the one-per-game
    // slot reopens and the countdown is gone. We detect it by the tile no longer being
    // owned by the player who built the Device (its building owner never changes).
    {
      const dt = this.objectManager_.findStrangeDeviceTile();
      if (dt) {
        const device = dt.getBuilding() as StrangeDevice;
        if (dt.getOwner() !== device.getOwner()) this.destroyStrangeDevice(dt);
      }
    }
    // The surviving Device's clock ticks on its owner's end-of-turn; if it reaches zero
    // while still standing, the owner wins immediately — everyone else loses (the same
    // resolution as the 70%-domination win).
    {
      const dt = this.objectManager_.findStrangeDeviceTile();
      if (dt) {
        const device = dt.getBuilding() as StrangeDevice;
        const owner = dt.getOwner();
        if (owner === current) device.decrementCountdown();
        if (owner !== null && device.getCountdown() <= 0) {
          for (const p of [...this.playerManager_.getPlayers()]) {
            if (p !== owner) {
              this.playerManager_.setPlayerAsLost(p);
              lostPlayersThisRound.push(p);
              this.neutralizePlayer(p);
            }
          }
        }
      }
    }

    this.playerManager_.changeTurn();
    this.objectManager_.removeBlockTileOverlays();

    if (this.playerManager_.getPlayers().length <= 1) this.gameOver_ = true; // winner or tie — lock the match

    if (this.playerManager_.getPlayers().length === 0) {
      this.menuObjectManager_.setTieMenu(lostPlayersThisRound, losingReasons);
    } else if (this.playerManager_.getPlayers().length === 1) {
      this.menuObjectManager_.setWinMenu(this.playerManager_.getPlayers()[0]);
    } else if (lostPlayersThisRound.length === 0) {
      this.openDefaultMenuView();
      this.notifyTurnChanged();
    } else {
      this.menuObjectManager_.setPlayerLostMenu(lostPlayersThisRound, losingReasons);
      this.notifyTurnChanged();
    }
  }

  /** Destroy the Strange Device on a tile: remove its sprite + countdown label and detach
   *  it from the tile. The one-per-game slot reopens, so any player may build a new one. */
  private destroyStrangeDevice(tile: TileBase): void {
    const building = tile.getBuilding();
    if (building === null) return;
    this.gameScene_.removeItem(building);
    tile.setBuildingDirect(null);
    this.updateTile(tile);
  }

  /** Remove every unit (owned and conquering) from a tile, destroying their sprites.
   *  Iterates copies because deleteUnitFromTile splices the live arrays. */
  private clearTileUnits(tile: TileBase): void {
    for (const unit of [...tile.getUnits()]) this.deleteUnitFromTile(unit, tile);
    for (const unit of [...tile.getConqueringUnits()]) this.deleteUnitFromTile(unit, tile);
  }

  neutralizePlayer(player: PlayerBase): void {
    for (const object of [...player.getObjects()]) {
      if (object instanceof TileBase) {
        const tile = object;
        this.clearTileUnits(tile);
        tile.setOwner(null);
        const b = tile.getBuilding();
        if (b !== null) {
          if (b.getType() === 'Farm') {
            (b as Farm).resetFarm();
            this.updateTile(tile);
          }
          if (b.getType() === 'Headquarters') {
            (b as HeadQuarters).setConquered();
            this.updateTile(tile);
          }
        }
      }
    }
  }

  updateAnimatedTileToStatic(tile: TileBase, frame: number): void {
    if (this.gameScene_.isObjectInScene(tile)) {
      const b = tile.getBuilding();
      if (b) {
        const handle = this.gameScene_.getObjectInScene(b);
        handle?.setAnimationFrame(frame);
      }
      this.gameScene_.updateTile(tile);
    }
  }

  updateForest(status: string, tile: TileBase, building: BuildingBase | null = null): void {
    if (!this.gameScene_.isObjectInScene(tile)) return;
    if (status === 'Cut') {
      tile.setImageFiles(ImageVectors.FOREST_STUMPS);
    } else if (status === 'Grow') {
      // Original reseeds with time(NULL); the choice is effectively random.
      tile.setImageFiles(Math.random() < 0.5 ? ImageVectors.FOREST_1 : ImageVectors.FOREST_2);
    }
    this.gameScene_.updateItem(tile);

    if (status === 'Grassland') {
      const newTile = new Grassland(tile.getCoordinate(), 1, 1, this, this.objectManager_);
      this.playerManager_.getCurrentPlayer().removeObject(tile);
      newTile.setGameSettings(this.gameSettingsManager_);
      this.objectManager_.replaceTile(tile, newTile);
      newTile.setImageFiles(ImageVectors.GRASSLAND);
      this.gameScene_.removeItem(tile);
      this.gameScene_.drawItem(newTile);
      if (building) newTile.addBuilding(building);
      this.updateTile(newTile);
      this.setTileInspectionMenuView(newTile);
    }
  }

  setTileInspectionMenuView(tile: TileBase, indexForBuildings = 0): void {
    if (this.unitToDeploy_ !== null || this.aiActive_) return;
    this.menuObjectManager_.setTileInspectionMenuView(tile, indexForBuildings);
  }

  openStatsMenuView(): void {
    this.menuObjectManager_.setStatMenuView();
  }

  openDefaultMenuView(): void {
    this.unitToDeploy_ = null;
    this.unitPreviousTile_ = null;
    this.gameScene_.removeMouseFollowItem();
    this.menuObjectManager_.setDefaultMenuView();
    this.objectManager_.removeClickedTileBorder();
    this.objectManager_.removeBlockTileOverlays();
  }

  openUnitBuyMenu(): void {
    this.menuObjectManager_.setUnitShopMenuView();
    this.objectManager_.removeClickedTileBorder();
    this.objectManager_.removeBlockTileOverlays();
  }

  /** Construct a unit of the given type owned by `owner`, with sprites + animation set. */
  private makeUnit(type: string, owner: PlayerBase): UnitBase | null {
    let unit: UnitBase | null = null;
    if (type === 'BasicWorker') {
      unit = new BasicWorker(this, this.objectManager_, this.gameSettingsManager_, owner);
      unit.setImageFiles(ImageVectors.BASICWORKER);
    }
    if (type === 'Expert') {
      unit = new Expert(this, this.objectManager_, this.gameSettingsManager_, owner);
      unit.setImageFiles(ImageVectors.EXPERT);
    }
    if (type === 'Soldier') {
      unit = new Soldier(this, this.objectManager_, this.gameSettingsManager_, owner);
      unit.setImageFiles(ImageVectors.SOLDIER);
    }
    if (unit) unit.setAnimationOption(AnimationOptions.UNIT);
    return unit;
  }

  /**
   * Cancel an in-progress unit move/deploy (e.g. on Esc): drop the held unit, clear
   * the mouse-follow cursor and the legal-tile overlays. Returns true if there was
   * actually something to cancel, so callers can fall back to other behaviour.
   */
  cancelUnitAction(): boolean {
    if (this.unitToDeploy_ === null && this.unitPreviousTile_ === null) return false;
    // If we lifted an existing unit off a tile, put it back where it came from.
    if (this.unitToDeploy_ !== null && this.unitPreviousTile_ !== null) {
      try {
        this.unitPreviousTile_.addUnit(this.unitToDeploy_);
        this.gameScene_.updateTile(this.unitPreviousTile_);
      } catch {
        /* tile somehow full — leave it dropped */
      }
    }
    this.unitToDeploy_ = null;
    this.unitPreviousTile_ = null;
    this.gameScene_.removeMouseFollowItem();
    this.objectManager_.removeBlockTileOverlays();
    return true;
  }

  createUnit(unit: string): void {
    if (this.gameOver_) return;
    if (this.unitToDeploy_ !== null) {
      this.cancelUnitAction();
      return;
    }
    const current = this.playerManager_.getCurrentPlayer();
    const unitToPlace = this.makeUnit(unit, current);
    if (!unitToPlace) return;

    if (unit === 'Soldier' && current.getFreeSoldierAmount() <= 0) return;
    if ((unit === 'BasicWorker' || unit === 'Expert') && current.getFreeUnitAmount() <= 0) return;
    if (!this.canBuyUnitOrBuilding(unitToPlace)) return;

    this.unitToDeploy_ = unitToPlace;
    this.objectManager_.addBlockTileOverlays();
    this.gameScene_.addMouseFollowPicture(unitToPlace.getImageFiles());
  }

  /**
   * Pick up a unit for moving by clicking it directly on the map. Mirrors the
   * MOVE button (moveUnitFromTile) but takes the unit instance instead of an
   * index, and only acts when not already mid-deploy/move.
   */
  selectUnitForMove(unit: UnitBase, tile: TileBase): boolean {
    if (this.gameOver_) return false;
    if (this.unitToDeploy_ !== null || this.unitPreviousTile_ !== null) return false;
    if (unit.getOwner() !== this.playerManager_.getCurrentPlayer()) return false;
    this.unitPreviousTile_ = tile;
    this.unitToDeploy_ = unit;
    this.objectManager_.addBlockTileOverlays();
    if (unit.getType() === 'BasicWorker') this.gameScene_.addMouseFollowPicture(ImageVectors.BASICWORKER);
    if (unit.getType() === 'Expert') this.gameScene_.addMouseFollowPicture(ImageVectors.EXPERT);
    if (unit.getType() === 'Soldier') this.gameScene_.addMouseFollowPicture(ImageVectors.SOLDIER);
    this.liftUnitFromTile(unit, tile); // remove from its tile so it isn't shown twice
    return true;
  }

  /** Lift a picked-up unit off its tile (and destroy its tile sprite) so only the
   *  mouse-follow cursor shows while dragging. Restored by cancelUnitAction. */
  private liftUnitFromTile(unit: UnitBase, tile: TileBase): void {
    tile.removeUnit(unit); // updates the tile (repositions any remaining units)
    this.gameScene_.removeItem(unit); // destroy the lifted unit's own sprite
  }

  moveUnitFromTile(index: number, tile: TileBase): void {
    if (this.gameOver_) return;
    if (this.unitToDeploy_ !== null || this.unitPreviousTile_ !== null) {
      this.cancelUnitAction();
      return;
    }
    this.unitPreviousTile_ = tile;
    if (tile.getOwner() === this.playerManager_.getCurrentPlayer()) {
      this.unitToDeploy_ = tile.getUnits()[index];
    } else {
      this.unitToDeploy_ = tile.getConqueringUnits()[index];
    }
    this.objectManager_.addBlockTileOverlays();
    if (this.unitToDeploy_.getType() === 'BasicWorker') this.gameScene_.addMouseFollowPicture(ImageVectors.BASICWORKER);
    if (this.unitToDeploy_.getType() === 'Expert') this.gameScene_.addMouseFollowPicture(ImageVectors.EXPERT);
    if (this.unitToDeploy_.getType() === 'Soldier') this.gameScene_.addMouseFollowPicture(ImageVectors.SOLDIER);
    this.liftUnitFromTile(this.unitToDeploy_, tile);
  }

  buyUnitOrBuilding(object: { getCost(): ResourceMap }): void {
    const cost = object.getCost();
    const player = this.playerManager_.getCurrentPlayer();
    if (!player.hasEnoughResources(cost)) return;
    player.addOrRemoveResources(cost);
  }

  canBuyUnitOrBuilding(object: { getCost(): ResourceMap }): boolean {
    const cost = object.getCost();
    return this.playerManager_.getCurrentPlayer().hasEnoughResources(cost);
  }

  deleteUnitFromTileByIndex(index: number, tile: TileBase): void {
    if (this.unitToDeploy_ !== null || this.unitPreviousTile_ !== null) return;
    if (tile.getOwner() === this.playerManager_.getCurrentPlayer()) {
      const u = tile.getUnits()[index];
      this.gameScene_.removeItem(u);
      tile.removeUnit(u);
    } else {
      const u = tile.getConqueringUnits()[index];
      this.gameScene_.removeItem(u);
      tile.removeUnit(u);
    }
    this.objectManager_.removeClickedTileBorder();
    this.setTileInspectionMenuView(tile);
    tile.updateAnimation();
  }

  deleteUnitFromTile(unit: UnitBase, tile: TileBase): void {
    this.gameScene_.removeItem(unit);
    tile.removeUnit(unit);
    // A deleted unit is gone for good — drop it from its owner too, otherwise it
    // lingers as a "phantom" that still counts against the cap and is still salaried
    // (which inflated counts and could even deadlock eliminateExcessUnits).
    const owner = unit.getOwner();
    if (owner) {
      try {
        owner.removeObject(unit);
      } catch {
        /* already untracked */
      }
    }
  }

  updateTile(tile: TileBase): void {
    this.gameScene_.updateTile(tile);
  }

  /** Construct a building of the given type for `current`, with sprites set. */
  private makeBuilding(buildingString: string, tile: TileBase, current: PlayerBase): BuildingBase | null {
    let building: BuildingBase | null = null;

    if (buildingString === 'Village') {
      building = new Village(this, this.objectManager_, current);
      building.setImageFiles(ImageVectors.VILLAGE);
      building.setAnimationOption(AnimationOptions.EMPTY);
    }
    if (buildingString === 'Outpost') {
      building = new Outpost(this, this.objectManager_, current);
      building.setImageFiles(ImageVectors.OUTPOST);
      building.setAnimationOption(AnimationOptions.OUTPOST);
    }
    if (buildingString === 'Nuclear Power Plant') {
      building = new NuclearPlant(this, this.objectManager_, current);
      building.setImageFiles(ImageVectors.NUCLEARPLANT);
      building.setAnimationOption(AnimationOptions.NUCLEAR);
    }
    if (buildingString === 'Mine') {
      building = new Mine(this, this.objectManager_, current);
      building.setImageFiles(ImageVectors.MINE);
      building.setAnimationOption(AnimationOptions.EMPTY);
    }
    if (buildingString === 'Hydroelectric Power Plant') {
      building = new HydroPower(this, this.objectManager_, current);
      if (tile instanceof River) {
        const orientation = tile.getRiverOrientation();
        if (orientation === 1) building.setImageFiles(ImageVectors.HYDROPOWERNS);
        if (orientation === 0) building.setImageFiles(ImageVectors.HYDROPOWERWE);
      }
      building.setAnimationOption(AnimationOptions.HEPP);
    }
    if (buildingString === 'Farm') {
      building = new Farm(this, this.objectManager_, current);
      building.setImageFiles(ImageVectors.FARM);
      building.setAnimationOption(AnimationOptions.EMPTY);
    }
    if (buildingString === 'Bridge') {
      building = new Bridge(this, this.objectManager_, current);
      if (tile instanceof River) {
        const orientation = tile.getRiverOrientation();
        if (orientation === 1) building.setImageFiles(ImageVectors.BRIDGEWE);
        if (orientation === 0) building.setImageFiles(ImageVectors.BRIDGENS);
      }
      building.setAnimationOption(AnimationOptions.EMPTY);
    }
    if (buildingString === 'Strange Device') {
      // One Strange Device per game — refuse a second wherever the build is triggered
      // from (the menu's buildable list hides it, but the direct build paths don't).
      if (this.objectManager_.hasStrangeDevice()) return null;
      // The Device must be built on an EMPTY tile (guardrail kept across arc sd5) —
      // otherwise you could pre-stack soldiers and then build on top. After it stands it
      // may garrison at most 1 defender (TileBase.hasSpaceForUnits, arc sd5).
      if (tile.getUnitCount() > 0) return null;
      const device = new StrangeDevice(this, this.objectManager_, current);
      // Countdown scales with map size (bigger map = longer to mass an army + cross it).
      device.setCountdown(strangeDeviceCountdown(this.objectManager_.getTileCount()));
      // No art yet: render as an empty tile (plain grassland sprite) with the countdown
      // number drawn on top by GameScene. The number marks where the Device sits.
      device.setImageFiles(ImageVectors.GRASSLAND);
      device.setAnimationOption(AnimationOptions.EMPTY);
      building = device;
    }

    return building;
  }

  buildBuilding(buildingString: string, tile: TileBase): void {
    if (this.gameOver_) return;
    if (this.unitPreviousTile_ !== null || this.unitToDeploy_ !== null) return;

    const current = this.playerManager_.getCurrentPlayer();
    const building = this.makeBuilding(buildingString, tile, current);

    if (!building) return;
    if (!this.canBuyUnitOrBuilding(building)) return;

    this.buyUnitOrBuilding(building);
    tile.addBuilding(building);
    // Building a Strange Device halves the soldier cap immediately — disband any soldiers
    // now over the new cap (else the degenerate line is "field a full army, THEN build").
    if (building.getType() === 'Strange Device') current.eliminateExcessUnits();
    this.updateTile(tile);
    this.setTileInspectionMenuView(tile);
    tile.updateAnimation();
  }

  // --- AI-driven actions ----------------------------------------------------
  // These mirror the human flows above but skip menu/overlay/mouse-follow churn,
  // so a CPU can act in a single synchronous pass. Each returns success.

  /** Build a building on an owned tile (used by the CPU). */
  aiBuildBuilding(buildingString: string, tile: TileBase): boolean {
    const current = this.playerManager_.getCurrentPlayer();
    const building = this.makeBuilding(buildingString, tile, current);
    if (!building) return false;
    if (!this.canBuyUnitOrBuilding(building)) return false;
    this.buyUnitOrBuilding(building);
    tile.addBuilding(building);
    if (building.getType() === 'Strange Device') current.eliminateExcessUnits();
    this.updateTile(tile);
    tile.updateAnimation();
    return true;
  }

  /** Buy a unit of `type` and place it on `tile` (used by the CPU). */
  aiBuyAndPlaceUnit(type: string, tile: TileBase): boolean {
    const current = this.playerManager_.getCurrentPlayer();
    if (type === 'Soldier' && current.getFreeSoldierAmount() <= 0) return false;
    if ((type === 'BasicWorker' || type === 'Expert') && current.getFreeUnitAmount() <= 0) return false;

    const unit = this.makeUnit(type, current);
    if (!unit) return false;
    if (!this.canBuyUnitOrBuilding(unit)) return false;
    if (!unit.canBePlacedOnTile(tile)) return false;

    try {
      unit.addParentTile(tile);
      unit.setOwner(current);
      tile.addUnit(unit);
    } catch {
      return false;
    }
    this.buyUnitOrBuilding(unit);
    tile.updateAnimation();
    this.gameScene_.updateTile(tile);
    return true;
  }

  /** Move an existing unit from one tile to an adjacent one (used by the CPU). */
  aiMoveUnit(unit: UnitBase, fromTile: TileBase, toTile: TileBase): boolean {
    const current = this.playerManager_.getCurrentPlayer();
    if (unit.getOwner() !== current) return false;
    if (!unit.canBePlacedOnTile(toTile)) return false;
    try {
      unit.setOwner(current);
      unit.setAsConquering(toTile.getOwner() !== current);
      toTile.addUnit(unit);
      unit.addParentTile(toTile);
      fromTile.removeUnit(unit);
    } catch {
      return false;
    }
    toTile.updateAnimation();
    fromTile.updateAnimation();
    toTile.updateUnitCoordinates();
    fromTile.updateUnitCoordinates();
    this.gameScene_.updateTile(toTile);
    this.gameScene_.updateTile(fromTile);
    return true;
  }

  getCurrentRevenue(): ResourceMap {
    let revenue = NO_RESOURCES;
    const current = this.playerManager_.getCurrentPlayer();
    for (const tile of this.objectManager_.getTiles()) {
      if (tile.getOwner() === current) {
        revenue = mergeResourceMaps(revenue, tile.getCurrentRevenue());
      }
    }
    return revenue;
  }

  getCurrentExpences(): ResourceMap {
    let expenses: ResourceMap = new Map();
    const current = this.playerManager_.getCurrentPlayer();
    for (const tile of this.objectManager_.getTiles()) {
      if (tile.getOwner() === current) {
        expenses = mergeResourceMaps(expenses, tile.getCurrentExpenses());
      }
      for (const unit of tile.getConqueringUnits()) {
        if (unit.getOwner() === current) expenses = mergeResourceMaps(expenses, unit.getSalary());
      }
    }
    return expenses;
  }

  getCurrentNet(): ResourceMap {
    let net: ResourceMap = new Map();
    const current = this.playerManager_.getCurrentPlayer();
    for (const tile of this.objectManager_.getTiles()) {
      if (tile.getOwner() === current) {
        net = mergeResourceMaps(net, tile.getCurrentNet());
      }
      for (const unit of tile.getConqueringUnits()) {
        if (unit.getOwner() === current) net = mergeResourceMaps(net, unit.getSalary());
      }
    }
    return net;
  }

  restartGame(): void {
    this.onRestart?.();
  }

  // Convenience accessors used by the menu/scene layers.
  getObjectManager(): ObjectManager {
    return this.objectManager_;
  }
  getPlayerManager(): PlayerManager {
    return this.playerManager_;
  }
  getSettings(): IGameSettingsManager {
    return this.gameSettingsManager_;
  }
}
