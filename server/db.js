// SQLite persistence layer for completed games.
// Uses Node's built-in `node:sqlite` (Node >= 22.5 with --experimental-sqlite,
// stable on Node >= 24) — zero external dependencies.
import { DatabaseSync } from 'node:sqlite';
import { randomUUID } from 'node:crypto';

const AI_TYPES = new Set(['jorma', 'kalevi', 'gunnar']);
const PLAYER_TYPES = new Set(['human', ...AI_TYPES]);
const WIN_CAUSES = new Set([
  'conquest', 'domination', 'device', 'bankruptcy', 'tie', 'resign', 'other',
]);

export function openDb(dbPath) {
  const db = new DatabaseSync(dbPath);
  db.exec('PRAGMA journal_mode = WAL;');
  db.exec('PRAGMA foreign_keys = ON;');
  migrate(db);
  return db;
}

function migrate(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS games (
      id           TEXT PRIMARY KEY,
      created_at   TEXT NOT NULL,
      width        INTEGER NOT NULL,
      height       INTEGER NOT NULL,
      winner_seat  INTEGER,
      win_cause    TEXT,
      rounds       INTEGER,
      human_count  INTEGER NOT NULL,
      matchup      TEXT NOT NULL,
      players      TEXT NOT NULL,   -- JSON array of player objects
      game_data    TEXT NOT NULL    -- JSON blob: full replay / move log
    );

    -- One row per player per game, so we can query "by player type" cheaply
    -- (e.g. per-AI win-rate vs humans) without scanning JSON.
    CREATE TABLE IF NOT EXISTS game_players (
      game_id      TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
      seat         INTEGER NOT NULL,
      type         TEXT NOT NULL,
      name         TEXT,
      is_ai        INTEGER NOT NULL,
      name_locked  INTEGER NOT NULL,
      won          INTEGER NOT NULL,
      PRIMARY KEY (game_id, seat)
    );

    CREATE INDEX IF NOT EXISTS idx_games_created_at ON games(created_at);
    CREATE INDEX IF NOT EXISTS idx_gp_type           ON game_players(type);
    CREATE INDEX IF NOT EXISTS idx_gp_type_won       ON game_players(type, won);
  `);
}

/**
 * Validate + normalize an incoming game record.
 * Returns { ok: true, record } or { ok: false, status, error }.
 */
export function validateRecord(body) {
  if (!body || typeof body !== 'object') {
    return { ok: false, status: 400, error: 'body must be a JSON object' };
  }

  const map = body.map ?? {};
  const width = Number(map.width);
  const height = Number(map.height);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return { ok: false, status: 400, error: 'map.width and map.height must be positive numbers' };
  }

  if (!Array.isArray(body.players) || body.players.length === 0) {
    return { ok: false, status: 400, error: 'players must be a non-empty array' };
  }

  const players = [];
  for (let i = 0; i < body.players.length; i++) {
    const p = body.players[i] ?? {};
    const type = String(p.type ?? '').toLowerCase();
    if (!PLAYER_TYPES.has(type)) {
      return {
        ok: false,
        status: 400,
        error: `players[${i}].type must be one of: human, jorma, kalevi, gunnar (got "${p.type}")`,
      };
    }
    const seat = p.seat ?? i;
    const isAI = type !== 'human';
    players.push({
      seat: Number(seat),
      type,
      name: p.name != null ? String(p.name) : null,
      isAI,
      nameLocked: Boolean(p.nameLocked),
    });
  }

  const humanCount = players.filter((p) => !p.isAI).length;
  if (humanCount === 0) {
    // The core rule: pure AI-vs-AI games are not persisted.
    return {
      ok: false,
      status: 422,
      error: 'game has no human players; only human-vs-AI games are stored',
      code: 'NO_HUMAN',
    };
  }

  const outcome = body.outcome ?? {};
  const winnerSeat = outcome.winnerSeat ?? outcome.winner_seat ?? null;
  let winCause = outcome.winCause ?? outcome.win_cause ?? null;
  if (winCause != null) {
    winCause = String(winCause).toLowerCase();
    if (!WIN_CAUSES.has(winCause)) winCause = 'other';
  }
  const rounds = outcome.rounds != null ? Number(outcome.rounds) : null;

  const matchup = body.matchup
    ? String(body.matchup)
    : buildMatchup(players);

  const record = {
    id: typeof body.id === 'string' && body.id ? body.id : randomUUID(),
    createdAt: new Date().toISOString(),
    width,
    height,
    winnerSeat: winnerSeat != null ? Number(winnerSeat) : null,
    winCause,
    rounds,
    humanCount,
    matchup,
    players,
    gameData: body.gameData ?? body.game_data ?? null,
  };

  if (record.gameData == null) {
    return { ok: false, status: 400, error: 'gameData (the replay/move log blob) is required' };
  }

  return { ok: true, record };
}

function buildMatchup(players) {
  return players
    .slice()
    .sort((a, b) => a.seat - b.seat)
    .map((p) => (p.name ? `${p.type}(${p.name})` : p.type))
    .join(' vs ');
}

export function insertGame(db, record) {
  const insertGameStmt = db.prepare(`
    INSERT INTO games
      (id, created_at, width, height, winner_seat, win_cause, rounds,
       human_count, matchup, players, game_data)
    VALUES
      (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const insertPlayerStmt = db.prepare(`
    INSERT INTO game_players
      (game_id, seat, type, name, is_ai, name_locked, won)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `);

  const tx = () => {
    insertGameStmt.run(
      record.id,
      record.createdAt,
      record.width,
      record.height,
      record.winnerSeat,
      record.winCause,
      record.rounds,
      record.humanCount,
      record.matchup,
      JSON.stringify(record.players),
      typeof record.gameData === 'string'
        ? record.gameData
        : JSON.stringify(record.gameData),
    );
    for (const p of record.players) {
      insertPlayerStmt.run(
        record.id,
        p.seat,
        p.type,
        p.name,
        p.isAI ? 1 : 0,
        p.nameLocked ? 1 : 0,
        record.winnerSeat != null && p.seat === record.winnerSeat ? 1 : 0,
      );
    }
  };

  db.exec('BEGIN');
  try {
    tx();
    db.exec('COMMIT');
  } catch (e) {
    db.exec('ROLLBACK');
    throw e;
  }
  return record.id;
}

export function listGames(db, { limit = 50, offset = 0 } = {}) {
  const lim = Math.min(Math.max(Number(limit) || 50, 1), 200);
  const off = Math.max(Number(offset) || 0, 0);
  const rows = db.prepare(`
    SELECT id, created_at, width, height, winner_seat, win_cause, rounds,
           human_count, matchup
    FROM games
    ORDER BY created_at DESC, id DESC
    LIMIT ? OFFSET ?
  `).all(lim, off);
  const total = db.prepare('SELECT COUNT(*) AS n FROM games').get().n;
  return { total, limit: lim, offset: off, games: rows.map(rowToSummary) };
}

export function getGame(db, id) {
  const row = db.prepare('SELECT * FROM games WHERE id = ?').get(id);
  if (!row) return null;
  return {
    id: row.id,
    createdAt: row.created_at,
    map: { width: row.width, height: row.height },
    outcome: {
      winnerSeat: row.winner_seat,
      winCause: row.win_cause,
      rounds: row.rounds,
    },
    humanCount: row.human_count,
    matchup: row.matchup,
    players: JSON.parse(row.players),
    gameData: safeParse(row.game_data),
  };
}

export function getStats(db) {
  // Per-AI win-rate in games that included at least one human (all stored games
  // are human-vs-AI by construction, so every game counts).
  const rows = db.prepare(`
    SELECT type,
           COUNT(*)        AS games,
           SUM(won)        AS wins
    FROM game_players
    WHERE is_ai = 1
    GROUP BY type
    ORDER BY type
  `).all();
  const perAi = rows.map((r) => ({
    type: r.type,
    games: r.games,
    wins: r.wins,
    winRate: r.games ? r.wins / r.games : 0,
  }));

  const humans = db.prepare(`
    SELECT COUNT(*) AS games, SUM(won) AS wins
    FROM game_players WHERE is_ai = 0
  `).get();

  const totalGames = db.prepare('SELECT COUNT(*) AS n FROM games').get().n;

  return {
    totalGames,
    perAi,
    humans: {
      games: humans.games ?? 0,
      wins: humans.wins ?? 0,
      winRate: humans.games ? (humans.wins ?? 0) / humans.games : 0,
    },
  };
}

function rowToSummary(row) {
  return {
    id: row.id,
    createdAt: row.created_at,
    map: { width: row.width, height: row.height },
    winnerSeat: row.winner_seat,
    winCause: row.win_cause,
    rounds: row.rounds,
    humanCount: row.human_count,
    matchup: row.matchup,
  };
}

function safeParse(s) {
  try { return JSON.parse(s); } catch { return s; }
}
