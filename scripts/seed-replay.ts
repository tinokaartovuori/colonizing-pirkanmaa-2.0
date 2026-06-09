// Headless: play one human-vs-Jorma game to completion and upload it to the
// game-records backend, so the replay dashboard has a real recorded game to show.
// The "human" seat is driven by the heuristic AI (we only need a finished game with
// a human-typed seat, which is what makes the recorder upload it). Run with:
//   VITE_CP_SERVER=https://cp-games.cp-2-0.workers.dev npx vite-node scripts/seed-replay.ts
import { GameSettingsManager } from '../src/managers/gamesettings';
import { ObjectManager } from '../src/managers/objectmanager';
import { PlayerManager } from '../src/managers/playermanager';
import { GameEventHandler } from '../src/managers/gameeventhandler';
import { WorldGenerator } from '../src/world/worldgenerator';
import { AiController } from '../src/managers/ai';
import { GameRecorder } from '../src/managers/gamerecorder';
import { MouseHoverBorder } from '../src/model/overlays';
import { Coordinate } from '../src/core/coordinate';
import type { IGameScene, ISceneObjectHandle } from '../src/model/base';
import type { IMenuObjectManager } from '../src/managers/menu-interface';
import type { PlayerConfig } from '../src/model/player';

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

const width = 14, height = 12, seed = 7;
const gsm = GameSettingsManager.fromMapDimensions(width, height);
const om = new ObjectManager();
const players: PlayerConfig[] = [
  { name: 'Tino', difficulty: 'human' },
  { name: 'Jorma', difficulty: 'hard' },
];
const pm = new PlayerManager(players, om);
const eh = new GameEventHandler(om, pm, stubMenu, gsm);
const scene = new StubScene();
eh.setGameScene(scene);
om.setGameScene(scene);
om.addDALS(eh, stubMenu, gsm);
om.setHoverBorder(new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om));
new WorldGenerator().generateMap(width, height, seed, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });

const recorder = new GameRecorder(om, pm, { width, height, seed });
eh.onTurnEnded = (p) => recorder.recordTurn(p);
eh.onGameOver = (info) => recorder.finish(info);

const ai = new AiController(eh, om, pm);

let guard = 0;
while (!eh.isGameOver() && guard++ < 400) {
  const cur = pm.getCurrentPlayer();
  if (cur.getObjects().length === 0) {
    ai.placeHeadquarters(cur); // advances the turn itself
  } else {
    ai.playTurn(cur);
    eh.endTurn();
  }
}

if (!eh.isGameOver()) {
  // Capped before a natural finish — still upload the history we have.
  const survivors = pm.getPlayers();
  recorder.finish({ winner: survivors[0] ?? null, winCause: 'domination', rounds: pm.getRoundsPlayed() });
}

console.log(`done: gameOver=${eh.isGameOver()} rounds=${pm.getRoundsPlayed()} server=${import.meta.env.VITE_CP_SERVER ?? '(default localhost)'}`);
// Give the best-effort async upload a moment to flush before the process exits.
await new Promise((r) => setTimeout(r, 1500));
