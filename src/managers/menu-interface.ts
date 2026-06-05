// Interface the GameEventHandler uses to drive menu views, decoupling the core
// game logic from the concrete (Phaser/DOM) menu renderer.

import type { TileBase } from '../model/tile';
import type { PlayerBase } from '../model/player';

export interface IMenuObjectManager {
  selectFirstTileMenuView(player: PlayerBase): void;
  setTileInspectionMenuView(tile: TileBase, indexForBuildings?: number): void;
  setStatMenuView(): void;
  setDefaultMenuView(): void;
  setUnitShopMenuView(): void;
  setTieMenu(players: PlayerBase[], reasons: string[]): void;
  setWinMenu(player: PlayerBase): void;
  setPlayerLostMenu(players: PlayerBase[], reasons: string[]): void;
  /** Panel shown while a CPU player takes its turn. */
  setCpuTurnMenuView(player: PlayerBase): void;
}
