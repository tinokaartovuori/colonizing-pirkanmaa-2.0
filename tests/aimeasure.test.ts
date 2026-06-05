import { describe, it, expect } from 'vitest';
import { GameSettingsManager } from '../src/managers/gamesettings';
import { ObjectManager } from '../src/managers/objectmanager';
import { PlayerManager } from '../src/managers/playermanager';
import { GameEventHandler } from '../src/managers/gameeventhandler';
import { WorldGenerator } from '../src/world/worldgenerator';
import { AiController } from '../src/managers/ai';
import { MouseHoverBorder } from '../src/model/overlays';
import { Coordinate } from '../src/core/coordinate';
import { IGameScene, ISceneObjectHandle } from '../src/model/base';
import { TileBase } from '../src/model/tile';
import { PlayerConfig, Difficulty, PlayerBase } from '../src/model/player';
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

function setup(width: number, height: number, seed: number, configs: PlayerConfig[]) {
  const gsm = GameSettingsManager.fromMapDimensions(width, height);
  const om = new ObjectManager();
  const pm = new PlayerManager(configs, om);
  const eh = new GameEventHandler(om, pm, stubMenu, gsm);
  const scene = new StubScene();
  eh.setGameScene(scene); om.setGameScene(scene); om.addDALS(eh, stubMenu, gsm);
  om.setHoverBorder(new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om));
  new WorldGenerator().generateMap(width, height, seed, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });
  return { gsm, om, pm, eh, ai: new AiController(eh, om, pm) };
}

function metrics(om: ObjectManager, p: PlayerBase) {
  const tiles = p.getObjects().filter((o): o is TileBase => o instanceof TileBase);
  const buildings: Record<string, number> = {};
  for (const t of tiles) {
    const b = t.getBuilding()?.getType();
    if (b) buildings[b] = (buildings[b] || 0) + 1;
  }
  const byType: Record<string, number> = {};
  for (const t of tiles) byType[t.getType()] = (byType[t.getType()] || 0) + 1;
  return {
    tiles: om.getTileCountForPlayer(p),
    cap: p.getMaxUnitAmount(),
    workers: p.getCurrentBasicWorkerAmount(),
    experts: p.getCurrentExpertAmount(),
    soldiers: p.getCurrentSoldierAmount(),
    money: p.getResources().get(1),
    wood: p.getResources().get(2),
    stone: p.getResources().get(3),
    metal: p.getResources().get(4),
    ownedTypes: byType,
    buildings,
  };
}

describe('AI strength measurement', () => {
  for (const diff of ['medium', 'hard'] as Difficulty[]) {
    it(`${diff} grows a real economy over 40 rounds vs an idle opponent`, () => {
      const { om, pm, eh, ai } = setup(16, 14, 7, [{ name: 'Idle' }, { name: 'Cpu', difficulty: diff }]);
      const cpu = pm.getPlayers().find((p) => p.getName() === 'Cpu')!;
      const fg = () => om.getTiles().filter((t) => t.getType() === 'Grassland' && t.getOwner() === null && t.getBuilding() === null);
      // both place HQs
      eh.tileClicked(fg()[0]);
      eh.setAiActive(true); ai.placeHeadquarters(cpu); eh.setAiActive(false);
      let bankruptAt = -1;
      for (let r = 0; r < 40 && pm.getPlayers().includes(cpu); r++) {
        eh.endTurn(); // idle human
        if (pm.getCurrentPlayer() === cpu) { eh.setAiActive(true); ai.playTurn(cpu); eh.setAiActive(false); eh.endTurn(); }
        if (bankruptAt < 0 && [...cpu.getResources().values()].some((v) => v < 0)) {
          bankruptAt = r;
          // eslint-disable-next-line no-console
          console.log(`[AI ${diff}] BANKRUPT round ${r}`, JSON.stringify(metrics(om, cpu)));
        }
      }
      const m = metrics(om, cpu);
      // eslint-disable-next-line no-console
      console.log(`[AI ${diff}]`, JSON.stringify(m), 'bankruptAt', bankruptAt);
      const realBuildings = Object.entries(m.buildings).filter(([k]) => k !== 'Headquarters').reduce((s, [, n]) => s + n, 0);
      expect(m.tiles).toBeGreaterThan(15); // expanded well beyond the starting 9
      expect(realBuildings).toBeGreaterThanOrEqual(3); // built a real economy
    });
  }
});
