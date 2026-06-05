// CPU-vs-CPU regression guard: two computer players play a full match against each
// other. This exercises the AI against a *live* opponent (expansion conflict, HQ
// connectivity, conquest) — not just the idle-opponent harness in aimeasure — and
// locks down the two properties that matter most: the AI never bankrupts itself and
// never throws. It also confirms a real economy still grows under pressure.
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
import { PlayerConfig, PlayerBase } from '../src/model/player';
import { IMenuObjectManager } from '../src/managers/menu-interface';

class StubScene implements IGameScene {
  drawItem() {}
  removeItem() {}
  updateItem() {}
  updateTile() {}
  isObjectInScene() {
    return true;
  }
  getObjectInScene(): ISceneObjectHandle {
    return { setAnimationOption() {}, setAnimationFrame() {} };
  }
  addMouseFollowPicture() {}
  removeMouseFollowItem() {}
  deleteObjects() {}
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

const tilesOf = (p: PlayerBase) => p.getObjects().filter((o): o is TileBase => o instanceof TileBase).length;

describe('AI duel (CPU vs CPU)', () => {
  for (const seed of [3, 11, 42, 99]) {
    it(`seed ${seed}: two hard CPUs play a full match, solvent and crash-free`, () => {
      const { om, pm, eh, ai } = setup(16, 16, seed, [
        { name: 'A', difficulty: 'hard' },
        { name: 'B', difficulty: 'hard' },
      ]);
      const [A, B] = pm.getPlayers();
      // Both place their HQ back-to-back (placing an HQ advances the turn itself).
      eh.setAiActive(true);
      ai.placeHeadquarters(pm.getCurrentPlayer());
      ai.placeHeadquarters(pm.getCurrentPlayer());
      eh.setAiActive(false);

      let bankrupt = false;
      for (let r = 0; r < 80 && pm.getPlayers().length > 1; r++) {
        const cur = pm.getCurrentPlayer();
        if (cur.isCpu()) {
          eh.setAiActive(true);
          ai.playTurn(cur);
          eh.setAiActive(false);
        }
        eh.endTurn();
        for (const p of [A, B]) {
          if (pm.getPlayers().includes(p) && [...p.getResources().values()].some((v) => v < 0)) bankrupt = true;
        }
      }

      // Never bankrupts itself, even fighting a live opponent.
      expect(bankrupt).toBe(false);
      // At least one player built a substantial economy (didn't just sit on its HQ).
      const biggest = Math.max(...pm.getPlayers().map(tilesOf));
      expect(biggest).toBeGreaterThan(15);
    });
  }
});
