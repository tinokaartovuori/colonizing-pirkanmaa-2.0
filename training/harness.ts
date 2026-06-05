// Headless match runner for training the neural AI.
//
// Mirrors sim/harness.ts but is controller-agnostic: each seat is built by a
// factory, so a match can pit a neural genome against the heuristic AI, against
// Hall-of-Fame genomes, or any mix. Outcomes are read from the game's OWN
// win/tie/loss logic via a capturing menu stub (never reimplemented), so
// training optimises against real game rules. sim/ is left untouched.

import { GameSettingsManager } from '../src/managers/gamesettings';
import { ObjectManager } from '../src/managers/objectmanager';
import { PlayerManager } from '../src/managers/playermanager';
import { GameEventHandler } from '../src/managers/gameeventhandler';
import { WorldGenerator } from '../src/world/worldgenerator';
import { MouseHoverBorder } from '../src/model/overlays';
import { Coordinate } from '../src/core/coordinate';
import { IGameScene, ISceneObjectHandle } from '../src/model/base';
import { PlayerBase, PlayerConfig } from '../src/model/player';
import { IMenuObjectManager } from '../src/managers/menu-interface';
import { ICpuController } from '../src/ai/controller-types';

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

class CapturingMenu implements IMenuObjectManager {
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

/** Builds a controller for a seat. `rand` is the match's seeded RNG. */
export type ControllerFactory = (
  eh: GameEventHandler,
  om: ObjectManager,
  pm: PlayerManager,
  seat: number,
  rand: () => number,
) => ICpuController;

export interface MatchSpec {
  width: number;
  height: number;
  seed: number;
  roundCap: number;
  /** One factory per seat (length defines the player count). */
  factories: ControllerFactory[];
}

export interface MatchResult {
  winnerSeat: number | null;
  reason: 'domination' | 'last-standing' | 'tie' | 'timeout';
  rounds: number;
  /** Final tile fraction [0,1] per seat (0 if eliminated). */
  tileFrac: number[];
  bankrupt: boolean[];
  crashed: boolean;
}

/** A small, fast, seedable xorshift32 RNG (independent of the game's MSVCRT RNG). */
export function makeRng(seed: number): () => number {
  let s = (seed * 2654435761) >>> 0;
  if (s === 0) s = 0x9e3779b9;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return (s >>> 0) / 4294967296;
  };
}

export function playMatch(spec: MatchSpec): MatchResult {
  const { width, height, seed, roundCap, factories } = spec;
  const n = factories.length;
  const gsm = GameSettingsManager.fromMapDimensions(width, height);
  const om = new ObjectManager();
  // All seats flagged as CPU so the loop drives every one via its controller.
  const configs: PlayerConfig[] = factories.map((_, i) => ({ name: `P${i + 1}`, difficulty: 'hard' }));
  const pm = new PlayerManager(configs, om);
  const menu = new CapturingMenu();
  const eh = new GameEventHandler(om, pm, menu, gsm);
  const scene = new StubScene();
  eh.setGameScene(scene);
  om.setGameScene(scene);
  om.addDALS(eh, menu, gsm);
  om.setHoverBorder(new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om));
  new WorldGenerator().generateMap(width, height, seed, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });

  const rand = makeRng(seed);
  const players = pm.getPlayers().slice();
  const ctrls = factories.map((f, i) => f(eh, om, pm, i, rand));
  const ctrlFor = (p: PlayerBase) => ctrls[p.getPlayerNum() - 1];

  const bankrupt = new Array<boolean>(n).fill(false);
  let crashed = false;

  try {
    eh.setAiActive(true);
    for (let i = 0; i < n; i++) {
      const cur = pm.getCurrentPlayer();
      ctrlFor(cur).placeHeadquarters(cur);
    }
    eh.setAiActive(false);

    while (pm.getPlayers().length > 1 && pm.getRoundsPlayed() < roundCap) {
      const cur = pm.getCurrentPlayer();
      if (cur.isCpu()) {
        eh.setAiActive(true);
        ctrlFor(cur).playTurn(cur);
        eh.setAiActive(false);
      }
      eh.endTurn();
      for (const p of players) {
        if (pm.getPlayers().includes(p) && [...p.getResources().values()].some((v) => v < 0)) {
          bankrupt[p.getPlayerNum() - 1] = true;
        }
      }
      if (menu.winner || menu.tie) break;
    }
  } catch {
    crashed = true;
  }

  const total = Math.max(1, om.getTileCount());
  const tileFrac = players.map((p) =>
    pm.getPlayers().includes(p) ? om.getTileCountForPlayer(p) / total : 0,
  );

  const survivors = pm.getPlayers();
  let winnerSeat: number | null = null;
  let reason: MatchResult['reason'];
  if (menu.winner) {
    winnerSeat = menu.winner.getPlayerNum() - 1;
    reason = (om.getTileCountForPlayer(menu.winner) * 100) / om.getTileCount() >= 70 ? 'domination' : 'last-standing';
  } else if (menu.tie) {
    reason = 'tie';
  } else if (survivors.length === 1) {
    winnerSeat = survivors[0].getPlayerNum() - 1;
    reason = 'last-standing';
  } else {
    reason = 'timeout';
  }

  return { winnerSeat, reason, rounds: pm.getRoundsPlayed(), tileFrac, bankrupt, crashed };
}
