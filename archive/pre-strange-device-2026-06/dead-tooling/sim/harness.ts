// Headless AI-vs-AI simulation harness.
//
// Drives a full Colonizing Pirkanmaa match with no Phaser/DOM by stubbing
// IGameScene and the menu interface — the same trick tests/aiduel.test.ts uses.
// The capturing menu stub records the game's OWN win/tie/loss determinations so
// we read outcomes from the real GameEventHandler logic, not a reimplementation.
//
// Used to (a) collect data on how the AI plays across thousands of games and
// (b) score AI variants head-to-head (challenger win rate vs baseline).

import { GameSettingsManager } from '../src/managers/gamesettings';
import { ObjectManager } from '../src/managers/objectmanager';
import { PlayerManager } from '../src/managers/playermanager';
import { GameEventHandler } from '../src/managers/gameeventhandler';
import { WorldGenerator } from '../src/world/worldgenerator';
import { AiController, AiParams } from '../src/managers/ai';
import { MouseHoverBorder } from '../src/model/overlays';
import { Coordinate } from '../src/core/coordinate';
import { IGameScene, ISceneObjectHandle } from '../src/model/base';
import { TileBase } from '../src/model/tile';
import { PlayerConfig, PlayerBase, Difficulty } from '../src/model/player';
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

/** Menu stub that captures the win/tie/loss callbacks the game fires. */
class CapturingMenu implements IMenuObjectManager {
  winner: PlayerBase | null = null;
  tie = false;
  lostEvents: { players: PlayerBase[]; reasons: string[] }[] = [];
  selectFirstTileMenuView(): void {}
  setTileInspectionMenuView(): void {}
  setStatMenuView(): void {}
  setDefaultMenuView(): void {}
  setUnitShopMenuView(): void {}
  setTieMenu(players: PlayerBase[], reasons: string[]): void { this.tie = true; this.lostEvents.push({ players, reasons }); }
  setWinMenu(player: PlayerBase): void { this.winner = player; }
  setPlayerLostMenu(players: PlayerBase[], reasons: string[]): void { this.lostEvents.push({ players, reasons }); }
  setCpuTurnMenuView(): void {}
}

export interface PlayerSpec extends PlayerConfig {
  /** Optional AI param override for this seat (challenger experiments). */
  params?: Partial<AiParams>;
}

export interface GameOptions {
  width: number;
  height: number;
  seed: number;
  players: PlayerSpec[];
  /** Hard cap on rounds; a game still undecided at the cap counts as a timeout. */
  roundCap?: number;
}

export interface PlayerResult {
  index: number;            // 0-based seat
  num: number;              // 1-based player number
  name: string;
  difficulty: Difficulty;
  won: boolean;
  /** Round this player was eliminated (-1 if survived to the end / won). */
  lostRound: number;
  lostReason: string | null; // 'conquered' | 'noresources' | null
  bankrupt: boolean;         // ever went to negative resources
  finalTiles: number;
  finalCap: number;
  finalWorkers: number;
  finalExperts: number;
  finalSoldiers: number;
  finalMoney: number;
  buildings: Record<string, number>;
  /** Peak tiles owned at any round end (measures how big it got even if later crushed). */
  peakTiles: number;
}

export interface GameResult {
  seed: number;
  numPlayers: number;
  rounds: number;
  /** 0-based seat of the winner, or null for tie/timeout. */
  winner: number | null;
  reason: 'domination' | 'last-standing' | 'tie' | 'timeout';
  anyBankrupt: boolean;
  crashed: boolean;
  /** A Strange Device was built at some point during the game. */
  deviceBuilt: boolean;
  /** The winner owned a standing Strange Device at game end (a Device-countdown win). */
  deviceWin: boolean;
  players: PlayerResult[];
}

function tilesOf(om: ObjectManager, p: PlayerBase): number {
  return om.getTileCountForPlayer(p);
}

function buildingsOf(p: PlayerBase): Record<string, number> {
  const out: Record<string, number> = {};
  for (const o of p.getObjects()) {
    if (o instanceof TileBase) {
      const b = o.getBuilding()?.getType();
      if (b) out[b] = (out[b] || 0) + 1;
    }
  }
  return out;
}

export function runGame(opts: GameOptions): GameResult {
  const { width, height, seed, players, roundCap = 200 } = opts;
  const gsm = GameSettingsManager.fromMapDimensions(width, height);
  const om = new ObjectManager();
  const pm = new PlayerManager(players, om);
  const menu = new CapturingMenu();
  const eh = new GameEventHandler(om, pm, menu, gsm);
  const scene = new StubScene();
  eh.setGameScene(scene);
  om.setGameScene(scene);
  om.addDALS(eh, menu, gsm);
  om.setHoverBorder(new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om));
  new WorldGenerator().generateMap(width, height, seed, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });

  const allPlayers = pm.getPlayers().slice(); // stable original-seat order
  // One AI controller per seat, so each can carry its own param override.
  const aiBySeat = allPlayers.map((_, i) => new AiController(eh, om, pm, players[i].params));
  const aiFor = (p: PlayerBase) => aiBySeat[p.getPlayerNum() - 1];

  // Per-seat bookkeeping keyed by original seat index.
  const lostRound = new Array(allPlayers.length).fill(-1);
  const lostReason: (string | null)[] = new Array(allPlayers.length).fill(null);
  const bankrupt = new Array(allPlayers.length).fill(false);
  const peakTiles = new Array(allPlayers.length).fill(0);

  let crashed = false;
  let rounds = 0;
  let deviceBuilt = false;

  try {
    // First round: every CPU places its HQ in turn order. Placing an HQ advances
    // the turn itself, so after N placements the turn is back on seat 0.
    eh.setAiActive(true);
    for (let i = 0; i < allPlayers.length; i++) {
      const cur = pm.getCurrentPlayer();
      aiFor(cur).placeHeadquarters(cur);
    }
    eh.setAiActive(false);

    while (pm.getPlayers().length > 1 && pm.getRoundsPlayed() < roundCap) {
      const cur = pm.getCurrentPlayer();
      if (cur.isCpu()) {
        eh.setAiActive(true);
        aiFor(cur).playTurn(cur);
        eh.setAiActive(false);
      }
      eh.endTurn();
      if (om.hasStrangeDevice()) deviceBuilt = true;

      // Record bankruptcy + peak tiles for everyone still tracked.
      for (const p of allPlayers) {
        const seat = p.getPlayerNum() - 1;
        if (pm.getPlayers().includes(p)) {
          if ([...p.getResources().values()].some((v) => v < 0)) bankrupt[seat] = true;
          peakTiles[seat] = Math.max(peakTiles[seat], tilesOf(om, p));
        }
      }
      // Note newly-lost players (those captured by the menu this turn).
      for (const ev of menu.lostEvents) {
        for (let k = 0; k < ev.players.length; k++) {
          const seat = ev.players[k].getPlayerNum() - 1;
          if (lostRound[seat] === -1) {
            lostRound[seat] = pm.getRoundsPlayed();
            lostReason[seat] = ev.reasons[k] ?? ev.reasons[0] ?? null;
          }
        }
      }
      if (menu.winner || menu.tie) break;
    }
    rounds = pm.getRoundsPlayed();
  } catch {
    crashed = true;
    rounds = pm.getRoundsPlayed();
  }

  const survivors = pm.getPlayers();
  let winnerSeat: number | null = null;
  let reason: GameResult['reason'];
  if (menu.winner) {
    winnerSeat = menu.winner.getPlayerNum() - 1;
    // Domination if the winner holds >=70% of tiles, else last-standing.
    const frac = (om.getTileCountForPlayer(menu.winner) * 100) / om.getTileCount();
    reason = frac >= 70 ? 'domination' : 'last-standing';
  } else if (menu.tie) {
    reason = 'tie';
  } else if (survivors.length === 1) {
    winnerSeat = survivors[0].getPlayerNum() - 1;
    reason = 'last-standing';
  } else {
    reason = 'timeout';
  }

  const playerResults: PlayerResult[] = allPlayers.map((p, i) => {
    const seat = i;
    const alive = survivors.includes(p);
    return {
      index: seat,
      num: p.getPlayerNum(),
      name: p.getName(),
      difficulty: p.getDifficulty(),
      won: winnerSeat === seat,
      lostRound: lostRound[seat],
      lostReason: lostReason[seat],
      bankrupt: bankrupt[seat],
      finalTiles: alive ? tilesOf(om, p) : 0,
      finalCap: alive ? p.getMaxUnitAmount() : 0,
      finalWorkers: alive ? p.getCurrentBasicWorkerAmount() : 0,
      finalExperts: alive ? p.getCurrentExpertAmount() : 0,
      finalSoldiers: alive ? p.getCurrentSoldierAmount() : 0,
      finalMoney: alive ? (p.getResources().get(1) ?? 0) : 0,
      buildings: alive ? buildingsOf(p) : {},
      peakTiles: peakTiles[seat],
    };
  });

  const deviceTileAtEnd = om.findStrangeDeviceTile();
  const deviceWin =
    winnerSeat !== null && deviceTileAtEnd !== null && deviceTileAtEnd.getOwner() === allPlayers[winnerSeat];

  return {
    seed,
    numPlayers: allPlayers.length,
    rounds: rounds!,
    winner: winnerSeat,
    reason,
    anyBankrupt: bankrupt.some(Boolean),
    crashed,
    deviceBuilt,
    deviceWin,
    players: playerResults,
  };
}
