import { describe, it, expect } from 'vitest';
import { GameSettingsManager } from '../src/managers/gamesettings';
import { ObjectManager } from '../src/managers/objectmanager';
import { PlayerManager } from '../src/managers/playermanager';
import { GameEventHandler } from '../src/managers/gameeventhandler';
import { WorldGenerator } from '../src/world/worldgenerator';
import { MouseHoverBorder } from '../src/model/overlays';
import { Coordinate } from '../src/core/coordinate';
import { IGameScene, ISceneObjectHandle } from '../src/model/base';
import { TileBase } from '../src/model/tile';
import { Soldier } from '../src/model/unit';
import { ImageVectors } from '../src/core/images';
import { IMenuObjectManager } from '../src/managers/menu-interface';

class StubScene implements IGameScene {
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
const stubMenu: IMenuObjectManager = {
  selectFirstTileMenuView() {}, setTileInspectionMenuView() {}, setStatMenuView() {},
  setDefaultMenuView() {}, setUnitShopMenuView() {}, setTieMenu() {}, setWinMenu() {},
  setPlayerLostMenu() {}, setCpuTurnMenuView() {},
};

function setup(width: number, height: number, seed: number, names: string[]) {
  const gsm = GameSettingsManager.fromMapDimensions(width, height);
  const om = new ObjectManager();
  const pm = new PlayerManager(names, om);
  const eh = new GameEventHandler(om, pm, stubMenu, gsm);
  const scene = new StubScene();
  eh.setGameScene(scene); om.setGameScene(scene); om.addDALS(eh, stubMenu, gsm);
  om.setHoverBorder(new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om));
  new WorldGenerator().generateMap(width, height, seed, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });
  return { gsm, om, pm, eh };
}

/** Drop a conquering soldier owned by `attacker` onto a tile (e.g. an enemy HQ). */
function placeConqueringSoldier(eh: GameEventHandler, om: ObjectManager, gsm: any, attacker: any, tile: TileBase) {
  // Make the tile reachable: give the attacker a neighbouring tile so the enemy
  // HQ is part of getAvailableTiles() (as it is in a real game).
  const neighbour = tile.getNeighbourFourTiles()[0];
  if (neighbour && neighbour.getOwner() !== attacker) neighbour.setOwner(attacker);
  const s = new Soldier(eh, om, gsm, attacker);
  s.setImageFiles(ImageVectors.SOLDIER);
  s.addParentTile(tile); // owner != tile owner -> conquering
  s.setOwner(attacker);
  tile.addUnit(s);
}

function hqTileOf(om: ObjectManager, player: any): TileBase {
  return om.getTiles().find(
    (t) => t.getOwner() === player && t.getBuilding()?.getType() === 'Headquarters',
  )!;
}

describe('conquering an enemy headquarters', () => {
  for (const n of [2, 3, 4]) {
    it(`does not crash with ${n} players`, () => {
      const names = ['One', 'Two', 'Three', 'Four'].slice(0, n);
      const { gsm, om, pm, eh } = setup(14, 12, 3, names);

      // Every player places an HQ on a free grassland.
      for (let i = 0; i < n; i++) {
        const g = om.getTiles().filter((t) => t.getType() === 'Grassland' && t.getOwner() === null);
        eh.tileClicked(g[Math.floor((g.length / n) * i)] ?? g[0]);
      }

      // PlayerOne (current after the placement round wraps to index 0) attacks
      // PlayerTwo's HQ with two soldiers.
      const p1 = pm.getPlayers()[0];
      const p2 = pm.getPlayers()[1];
      while (pm.getCurrentPlayer() !== p1) pm.changeTurn();
      const enemyHq = hqTileOf(om, p2);
      placeConqueringSoldier(eh, om, gsm, p1, enemyHq);
      placeConqueringSoldier(eh, om, gsm, p1, enemyHq);

      expect(() => eh.endTurn()).not.toThrow();
    });
  }
});
