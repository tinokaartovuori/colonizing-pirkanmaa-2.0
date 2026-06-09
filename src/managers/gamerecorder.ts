// Game recorder + upload.
//
// A passive observer that records a completed human-vs-AI game (full per-turn
// history) and POSTs it to the analysis backend (server/, default
// http://127.0.0.1:8790). It NEVER changes game rules — it only reads the
// managers and snapshots state at each turn boundary.
//
// Wiring (see main.ts): per match, construct a GameRecorder, hook
// eventHandler.onTurnEnded -> recordTurn and eventHandler.onGameOver -> finish.
// AI-vs-AI games are not uploaded (the backend rejects them with 422, and we
// don't even send them).

import type { ObjectManager } from './objectmanager';
import type { PlayerManager } from './playermanager';
import type { PlayerBase, Difficulty } from '../model/player';
import { TileBase } from '../model/tile';
import { UnitBase } from '../model/unit';
import { BasicResource } from '../core/resources';
import { StrangeDevice } from '../model/building';
import { buildSnapshot, GameSnapshot } from './persistence';

/** Backend base URL; override with VITE_CP_SERVER at build time. */
const SERVER_URL =
  (import.meta.env.VITE_CP_SERVER as string | undefined) ?? 'http://127.0.0.1:8790';

// --- the named opponent roster (single source of truth) --------------------
// Each selectable AI is a CHARACTER with a fixed (locked) name + an algorithm
// label, mapped to the engine difficulty string. Used by the start dialog (to
// lock the seat name) and here (to derive the backend `type` field).

/** Backend player-type tag. Mirrors the server's enum (human + the three AIs). */
export type PlayerTypeTag = 'human' | 'jorma' | 'kalevi' | 'gunnar';

export interface RosterCharacter {
  /** Fixed character name, locked onto the seat when selected. */
  name: string;
  /** Engine difficulty string this character plays as. */
  difficulty: Difficulty;
  /** Backend type tag. */
  type: PlayerTypeTag;
  /** Dropdown label, e.g. "Jorma (Heuristiikka)". */
  label: string;
}

export const AI_ROSTER: RosterCharacter[] = [
  { name: 'Jorma', difficulty: 'hard', type: 'jorma', label: 'Jorma (Heuristiikka)' },
  { name: 'Kalevi', difficulty: 'model:kalevi', type: 'kalevi', label: 'Kalevi (AlphaZero)' },
  { name: 'Gunnar', difficulty: 'model:gunnar', type: 'gunnar', label: 'Gunnar (AlphaZero XL)' },
];

const ROSTER_BY_DIFFICULTY = new Map<Difficulty, RosterCharacter>(
  AI_ROSTER.map((c) => [c.difficulty, c]),
);

/** The locked character for an AI difficulty, or null for a human / unknown seat. */
export function rosterCharacterFor(difficulty: Difficulty): RosterCharacter | null {
  return ROSTER_BY_DIFFICULTY.get(difficulty) ?? null;
}

/** Backend `type` tag for a seat's difficulty (human, or one of the roster AIs). */
export function playerTypeTag(difficulty: Difficulty): PlayerTypeTag {
  return rosterCharacterFor(difficulty)?.type ?? 'human';
}

// --- per-turn metrics + history --------------------------------------------

export interface SeatMetrics {
  seat: number; // 0-based (playerNum - 1)
  money: number;
  wood: number;
  stone: number;
  metal: number;
  tiles: number;
  soldiers: number;
  buildings: number;
  hasDevice: boolean;
  deviceCountdown: number | null;
}

export interface HistoryEntry {
  round: number;
  /** 0-based seat of the player whose turn just ended. */
  seat: number;
  snapshot: GameSnapshot;
  metrics: SeatMetrics[];
}

/** Derive light per-seat metrics from the live managers (read-only). */
function computeMetrics(om: ObjectManager, pm: PlayerManager): SeatMetrics[] {
  const all = [...pm.getPlayers(), ...pm.getLostPlayers()].sort(
    (a, b) => a.getPlayerNum() - b.getPlayerNum(),
  );
  return all.map((p) => {
    const r = p.getResources();
    let tiles = 0;
    let soldiers = 0;
    let buildings = 0;
    let hasDevice = false;
    let deviceCountdown: number | null = null;
    for (const obj of p.getObjects()) {
      if (obj instanceof TileBase) {
        tiles++;
        const b = obj.getBuilding();
        if (b) {
          buildings++;
          if (b instanceof StrangeDevice && b.getOwner() === p) {
            hasDevice = true;
            deviceCountdown = b.getCountdown();
          }
        }
        for (const u of obj.getUnits()) {
          if (u instanceof UnitBase && u.getType() === 'Soldier') soldiers++;
        }
      }
    }
    return {
      seat: p.getPlayerNum() - 1,
      money: r.get(BasicResource.MONEY) ?? 0,
      wood: r.get(BasicResource.WOOD) ?? 0,
      stone: r.get(BasicResource.STONE) ?? 0,
      metal: r.get(BasicResource.METAL) ?? 0,
      tiles,
      soldiers,
      buildings,
      hasDevice,
      deviceCountdown,
    };
  });
}

/**
 * Records one match and uploads it on completion. One instance per match.
 * History is in-memory and starts fresh at construction (so a restored save
 * begins a new history from the restored state — older turns are not recovered).
 */
export class GameRecorder {
  private history: HistoryEntry[] = [];
  private finished = false;

  constructor(
    private readonly om: ObjectManager,
    private readonly pm: PlayerManager,
    private readonly map: { width: number; height: number; seed: number },
  ) {}

  /** Append a history entry for the turn that `endedBy` just played. */
  recordTurn(endedBy: PlayerBase): void {
    if (this.finished) return;
    this.history.push({
      round: this.pm.getRoundsPlayed(),
      seat: endedBy.getPlayerNum() - 1,
      snapshot: buildSnapshot(this.om, this.pm, this.map),
      metrics: computeMetrics(this.om, this.pm),
    });
  }

  /** Assemble + upload the completed game (best-effort). No-op if no human seat. */
  finish(info: { winner: PlayerBase | null; winCause: string; rounds: number }): void {
    if (this.finished) return;
    this.finished = true;

    const seats = [...this.pm.getPlayers(), ...this.pm.getLostPlayers()].sort(
      (a, b) => a.getPlayerNum() - b.getPlayerNum(),
    );

    const players = seats.map((p) => {
      const difficulty = p.getDifficulty();
      const character = rosterCharacterFor(difficulty);
      return {
        seat: p.getPlayerNum() - 1,
        type: playerTypeTag(difficulty),
        name: character ? character.name : p.getName(),
        nameLocked: character !== null,
      };
    });

    // Only upload games with at least one human seat (the backend rejects the rest).
    if (!players.some((p) => p.type === 'human')) return;

    const finalSnapshot = buildSnapshot(this.om, this.pm, this.map);
    const winnerSeat = info.winner ? info.winner.getPlayerNum() - 1 : null;

    const body = {
      map: { width: this.map.width, height: this.map.height },
      players,
      outcome: { winnerSeat, winCause: info.winCause, rounds: info.rounds },
      gameData: {
        seed: this.map.seed,
        history: this.history,
        finalSnapshot,
        winnerSeat,
        winCause: info.winCause,
      },
    };

    void this.upload(body);
  }

  private async upload(body: unknown): Promise<void> {
    try {
      const res = await fetch(`${SERVER_URL}/api/games`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        console.warn(`[gamerecorder] upload rejected: HTTP ${res.status}`);
      }
    } catch (e) {
      // Server down / offline — uploading is best-effort, never block the UI.
      console.warn('[gamerecorder] upload failed (server unreachable?):', e);
    }
  }
}
