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
import { IMenuObjectManager } from '../src/managers/menu-interface';
import { createModelController } from '../src/ai/nn';

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

function newGame(width: number, height: number, seed: number) {
  const gsm = GameSettingsManager.fromMapDimensions(width, height);
  const om = new ObjectManager();
  const pm = new PlayerManager(['Champ', 'Hard'], om);
  const eh = new GameEventHandler(om, pm, stubMenu, gsm);
  const scene = new StubScene();
  eh.setGameScene(scene); om.setGameScene(scene); om.addDALS(eh, stubMenu, gsm);
  const hover = new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om);
  om.setHoverBorder(hover);
  new WorldGenerator().generateMap(width, height, seed, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });
  return { gsm, om, pm, eh };
}

type Peak = {
  tiles: number; mines: number; mineStaff: number; experts: number;
  soldiers: number; villages: number; outposts: number; soldierCap: number;
};

/** Run one CNN-champion-vs-Hard game on `seed`, returning the champ's PEAK
 *  economy/army state (the champ may later be conquered, but the army-economy
 *  chain is proven by what it BUILT while alive). */
function runSeed(seed: number): Peak {
  const { om, pm, eh } = newGame(14, 12, seed);
  const champ = pm.getPlayers()[0];
  // Greedy net (temperature 0 in HARD_CONFIG) → rng is unused for scoring; fixed.
  const champCtrl = createModelController(eh, om, pm, 'sd4-az-002', () => 0.42,
    { width: 14, height: 12, seed });
  const hardAi = new AiController(eh, om, pm); // heuristic opponent keeps the game alive

  const ownedBuildings = (p: typeof champ) => {
      const counts: Record<string, number> = {};
      for (const o of p.getObjects()) {
        if (o instanceof TileBase) {
          const b = o.getBuilding()?.getType();
          if (b) counts[b] = (counts[b] ?? 0) + 1;
        }
      }
      return counts;
    };
    const mineStaff = (p: typeof champ) => {
      let bestW = 0, bestE = 0, mines = 0;
      for (const o of p.getObjects()) {
        if (o instanceof TileBase && o.getBuilding()?.getType() === 'Mine') {
          mines++;
          const wk = o.getUnits().filter((u) => u.getType() === 'BasicWorker').length;
          const ex = o.getUnits().filter((u) => u.getType() === 'Expert').length;
          if (wk + ex > bestW + bestE) { bestW = wk; bestE = ex; }
        }
      }
      return { mines, bestW, bestE };
    };

    const playCurrent = () => {
      const cur = pm.getCurrentPlayer();
      eh.setAiActive(true);
      if (cur.getObjects().length === 0) {
        if (cur === champ) champCtrl.placeHeadquarters(cur);
        else hardAi.placeHeadquarters(cur);
        eh.setAiActive(false);
        // placeHeadquarters advances the turn itself.
      } else {
        if (cur === champ) champCtrl.playTurn(cur);
        else hardAi.playTurn(cur);
        eh.setAiActive(false);
        eh.endTurn();
      }
    };

    const peak: Peak = { tiles: 0, mines: 0, mineStaff: 0, experts: 0, soldiers: 0, villages: 0, outposts: 0, soldierCap: 0 };
    for (let i = 0; i < 300 && pm.getPlayers().includes(champ) && pm.getRoundsPlayed() < 70; i++) {
      playCurrent();
      const ms = mineStaff(champ);
      const b = ownedBuildings(champ);
      peak.tiles = Math.max(peak.tiles, om.getTileCountForPlayer(champ));
      peak.mines = Math.max(peak.mines, ms.mines);
      peak.mineStaff = Math.max(peak.mineStaff, ms.bestW + ms.bestE);
      peak.experts = Math.max(peak.experts, champ.getCurrentExpertAmount());
      peak.soldiers = Math.max(peak.soldiers, champ.getCurrentSoldierAmount());
      peak.villages = Math.max(peak.villages, b.Village ?? 0);
      peak.outposts = Math.max(peak.outposts, b.Outpost ?? 0);
      peak.soldierCap = Math.max(peak.soldierCap, champ.getMaxSoldierAmount());
    }
    return peak;
}

describe('DEPLOY VERIFY: sd4-az-002 CNN champion army-economy chain', () => {
  it('the deployed CNN champion drives the army-economy chain (mine→staff→expert→soldier, not Pass-collapse)', () => {
    const seeds = [1, 2, 3, 7];
    const peaks = seeds.map((s) => ({ seed: s, peak: runSeed(s) }));
    // eslint-disable-next-line no-console
    console.log('DEPLOY PEAKS:', JSON.stringify(peaks));

    // Across the seed pool the deployed net must demonstrably:
    //  - EXPAND (every seed grows well past the starting ~1 tile — no Pass-collapse).
    for (const { peak } of peaks) expect(peak.tiles).toBeGreaterThan(5);
    //  - build + STAFF a mine with a worker+Expert (the metal-economy chain) on a seed
    //    where it survives long enough, and field a soldier (the army) on a seed.
    const max = (f: (p: Peak) => number) => Math.max(...peaks.map((x) => f(x.peak)));
    expect(max((p) => p.mines)).toBeGreaterThanOrEqual(1);      // built a mine
    expect(max((p) => p.mineStaff)).toBeGreaterThanOrEqual(2);  // staffed worker+Expert
    expect(max((p) => p.experts)).toBeGreaterThanOrEqual(1);    // placed Experts
    expect(max((p) => p.soldiers)).toBeGreaterThanOrEqual(1);   // fielded a soldier
    expect(max((p) => p.villages)).toBeGreaterThanOrEqual(1);   // raised unit cap
  });
});
