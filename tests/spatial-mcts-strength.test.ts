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
import { IMenuObjectManager } from '../src/managers/menu-interface';
import { createModelController } from '../src/ai/nn';

// Deploy STRENGTH test: the CNN champion WITH the deploy MCTS (policy prior +
// value-head leaves, sims≈64) should be at least as strong as the greedy net
// policy vs the Hard bot. We measure outcomes over a small seed pool and assert
// MCTS ≥ greedy on wins (with army/tiles reported for context). This drives the
// SAME in-engine objects an in-browser deploy would, so it validates the wiring
// end-to-end without a browser.

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

interface Outcome { win: boolean; alive: boolean; peakSoldiers: number; peakTiles: number; rounds: number; }

function runSeed(seed: number, mcts: boolean): Outcome {
  const W = 14, H = 12;
  const { om, pm, eh } = newGame(W, H, seed);
  const champ = pm.getPlayers()[0];
  const mapInfo = mcts ? { width: W, height: H, seed } : undefined;
  const champCtrl = createModelController(eh, om, pm, 'sd4-az-002', () => 0.42, mapInfo);
  const hardAi = new AiController(eh, om, pm);

  let peakSoldiers = 0, peakTiles = 0;
  const playCurrent = () => {
    const cur = pm.getCurrentPlayer();
    eh.setAiActive(true);
    if (cur.getObjects().length === 0) {
      if (cur === champ) champCtrl.placeHeadquarters(cur); else hardAi.placeHeadquarters(cur);
      eh.setAiActive(false);
    } else {
      if (cur === champ) champCtrl.playTurn(cur); else hardAi.playTurn(cur);
      eh.setAiActive(false);
      eh.endTurn();
    }
  };

  let i = 0;
  for (; i < 400 && pm.getPlayers().includes(champ) && pm.getPlayers().length > 1 && pm.getRoundsPlayed() < 50; i++) {
    playCurrent();
    if (pm.getPlayers().includes(champ)) {
      peakSoldiers = Math.max(peakSoldiers, champ.getCurrentSoldierAmount());
      peakTiles = Math.max(peakTiles, om.getTileCountForPlayer(champ));
    }
  }
  const alive = pm.getPlayers().includes(champ);
  // Win = sole survivor, OR (game timed out) ahead on tiles.
  let win = false;
  if (alive && pm.getPlayers().length === 1) win = true;
  else if (alive) {
    const myTiles = om.getTileCountForPlayer(champ);
    const enemyTiles = pm.getPlayers()
      .filter((p) => p !== champ)
      .reduce((m, p) => Math.max(m, om.getTileCountForPlayer(p)), 0);
    win = myTiles > enemyTiles;
  }
  return { win, alive, peakSoldiers, peakTiles, rounds: pm.getRoundsPlayed() };
}

describe('DEPLOY STRENGTH: sd4-az-002 CNN champion — MCTS vs greedy', () => {
  it('the deploy MCTS is at least as strong as the greedy policy vs Hard', () => {
    const seeds = [1, 2, 3];
    const t0 = Date.now();
    const greedy = seeds.map((s) => ({ seed: s, o: runSeed(s, false) }));
    const tGreedy = Date.now() - t0;
    const t1 = Date.now();
    const mcts = seeds.map((s) => ({ seed: s, o: runSeed(s, true) }));
    const tMcts = Date.now() - t1;

    const wins = (xs: { o: Outcome }[]) => xs.filter((x) => x.o.win).length;
    const sumSold = (xs: { o: Outcome }[]) => xs.reduce((a, x) => a + x.o.peakSoldiers, 0);
    const sumTiles = (xs: { o: Outcome }[]) => xs.reduce((a, x) => a + x.o.peakTiles, 0);

    // eslint-disable-next-line no-console
    console.log('STRENGTH greedy:', JSON.stringify(greedy.map((x) => x.o)));
    // eslint-disable-next-line no-console
    console.log('STRENGTH mcts:  ', JSON.stringify(mcts.map((x) => x.o)));
    // eslint-disable-next-line no-console
    console.log(`WINS greedy=${wins(greedy)}/${seeds.length}  mcts=${wins(mcts)}/${seeds.length} | ` +
      `peakSoldiers greedy=${sumSold(greedy)} mcts=${sumSold(mcts)} | ` +
      `peakTiles greedy=${sumTiles(greedy)} mcts=${sumTiles(mcts)} | ` +
      `wall greedy=${tGreedy}ms mcts=${tMcts}ms`);

    // The MCTS deploy must not be WEAKER than greedy on wins (it should be ≥).
    expect(wins(mcts)).toBeGreaterThanOrEqual(wins(greedy));
    // Sanity: MCTS still drives the army-economy chain (fields soldiers somewhere).
    expect(sumSold(mcts)).toBeGreaterThanOrEqual(1);
  }, 600_000);

  // Direct, opponent-free discriminator: the SAME champion net plays seat 0 WITH
  // the deploy MCTS and seat 1 with the GREEDY policy, on the same map. If the
  // search adds strength, the MCTS seat should not lose the head-to-head (tiles).
  it('head-to-head: champion-MCTS is at least as strong as champion-greedy', () => {
    const W = 14, H = 12;
    const seeds = [4, 5, 8];
    let mctsAhead = 0, greedyAhead = 0;
    const rows: string[] = [];
    for (const seed of seeds) {
      const { om, pm, eh } = newGame(W, H, seed);
      const mctsP = pm.getPlayers()[0];
      const greedyP = pm.getPlayers()[1];
      const mctsCtrl = createModelController(eh, om, pm, 'sd4-az-002', () => 0.42, { width: W, height: H, seed });
      const greedyCtrl = createModelController(eh, om, pm, 'sd4-az-002', () => 0.42);
      for (let i = 0; i < 400 && pm.getPlayers().length > 1 && pm.getRoundsPlayed() < 55; i++) {
        const cur = pm.getCurrentPlayer();
        eh.setAiActive(true);
        const ctrl = cur === mctsP ? mctsCtrl : greedyCtrl;
        if (cur.getObjects().length === 0) { ctrl.placeHeadquarters(cur); eh.setAiActive(false); }
        else { ctrl.playTurn(cur); eh.setAiActive(false); eh.endTurn(); }
      }
      const mt = pm.getPlayers().includes(mctsP) ? om.getTileCountForPlayer(mctsP) : 0;
      const gt = pm.getPlayers().includes(greedyP) ? om.getTileCountForPlayer(greedyP) : 0;
      if (mt > gt) mctsAhead++; else if (gt > mt) greedyAhead++;
      rows.push(`seed ${seed}: mctsTiles=${mt} greedyTiles=${gt}`);
    }
    // eslint-disable-next-line no-console
    console.log('H2H', rows.join(' | '), `=> mctsAhead=${mctsAhead} greedyAhead=${greedyAhead}`);
    // MCTS should not be beaten by greedy more often than it wins.
    expect(mctsAhead).toBeGreaterThanOrEqual(greedyAhead);
  }, 600_000);
});
