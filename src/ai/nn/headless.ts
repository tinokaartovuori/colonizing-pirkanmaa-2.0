// Shared, Phaser-free engine harness for headless play (tests, training, and the
// in-browser MCTS sandbox). This is the StubScene + stub-menu pattern that was
// previously duplicated inside tests/restore.test.ts and training/harness.ts,
// lifted into one place so the MCTS search sandbox (src/ai/nn/sandbox.ts) and the
// tests can all build a UI-detached GameEventHandler the exact same way.
//
// NOTHING here touches Phaser or the DOM — every IGameScene / IMenuObjectManager
// method is a safe no-op (or, for the menu, a capturing field) so endTurn /
// restoreSnapshot can run with no renderer attached.

import { IGameScene, ISceneObjectHandle } from '../../model/base';
import { IMenuObjectManager } from '../../managers/menu-interface';
import { PlayerBase } from '../../model/player';

/** A no-op IGameScene: every draw/update/lookup call is safely ignored. */
export class StubScene implements IGameScene {
  drawItem(): void {}
  removeItem(): void {}
  updateItem(): void {}
  updateTile(): void {}
  isObjectInScene(): boolean { return true; }
  getObjectInScene(): ISceneObjectHandle { return { setAnimationOption() {}, setAnimationFrame() {} }; }
  addMouseFollowPicture(): void {}
  removeMouseFollowItem(): void {}
  deleteObjects(): void {}
}

/**
 * A no-op IMenuObjectManager that also CAPTURES the win/tie/loss callbacks the
 * engine fires from endTurn. Search/training read `winner`/`tie` to detect a
 * terminal; UI methods are no-ops so endTurn never touches a real renderer.
 */
export class CapturingMenu implements IMenuObjectManager {
  winner: PlayerBase | null = null;
  tie = false;
  selectFirstTileMenuView(): void {}
  setTileInspectionMenuView(): void {}
  setStatMenuView(): void {}
  setDefaultMenuView(): void {}
  setUnitShopMenuView(): void {}
  setTieMenu(): void { this.tie = true; }
  setWinMenu(p: PlayerBase): void { this.winner = p; }
  setPlayerLostMenu(): void {}
  setCpuTurnMenuView(): void {}
}
