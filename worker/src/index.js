// Cloudflare Worker backend for storing completed games of Colonizing Pirkanmaa.
//
// Ported 1:1 from server/server.js + server/db.js (Node + node:sqlite) to the
// Workers runtime + D1 (serverless SQLite). Same endpoints, same request/response
// shapes, same SQL schema (worker/schema.sql) — so the deployed Worker and the
// local `npm run server` backend are interchangeable.
//
// Only games with >= 1 human player are stored; pure AI-vs-AI games are rejected
// 422 NO_HUMAN (the core rule), enforced server-side here as well as client-side.
//
// Deploy: see worker/README.md.

const MAX_BODY = 32 * 1024 * 1024; // 32 MB — replays can be large

const AI_TYPES = new Set(['jorma', 'kalevi', 'gunnar']);
const PLAYER_TYPES = new Set(['human', ...AI_TYPES]);
const WIN_CAUSES = new Set([
  'conquest', 'domination', 'device', 'bankruptcy', 'tie', 'resign', 'other',
]);

// --- HTTP helpers -----------------------------------------------------------

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

function json(status, payload) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json', ...CORS },
  });
}

// --- request entrypoint -----------------------------------------------------

export default {
  async fetch(req, env) {
    if (req.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS });
    }

    const url = new URL(req.url);
    const path = url.pathname.replace(/\/+$/, '') || '/';

    try {
      if (req.method === 'GET' && path === '/health') {
        return json(200, { ok: true });
      }

      if (req.method === 'POST' && path === '/api/games') {
        const raw = await readBody(req);
        let body;
        try { body = JSON.parse(raw || '{}'); }
        catch { return json(400, { ok: false, error: 'invalid JSON' }); }

        const v = validateRecord(body);
        if (!v.ok) {
          return json(v.status, { ok: false, error: v.error, code: v.code });
        }
        const id = await insertGame(env.cp_games, v.record);
        return json(200, { ok: true, id, matchup: v.record.matchup });
      }

      if (req.method === 'GET' && path === '/api/games') {
        const result = await listGames(env.cp_games, {
          limit: url.searchParams.get('limit'),
          offset: url.searchParams.get('offset'),
        });
        return json(200, { ok: true, ...result });
      }

      if (req.method === 'GET' && path === '/api/games/stats') {
        return json(200, { ok: true, ...(await getStats(env.cp_games)) });
      }

      const m = path.match(/^\/api\/games\/([^/]+)$/);
      if (req.method === 'GET' && m) {
        const game = await getGame(env.cp_games, decodeURIComponent(m[1]));
        if (!game) return json(404, { ok: false, error: 'not found' });
        return json(200, { ok: true, game });
      }

      return json(404, { ok: false, error: 'not found' });
    } catch (e) {
      const status = e.status || 500;
      return json(status, { ok: false, error: e.message || 'internal error' });
    }
  },
};

async function readBody(req) {
  const buf = await req.arrayBuffer();
  if (buf.byteLength > MAX_BODY) {
    throw Object.assign(new Error('payload too large'), { status: 413 });
  }
  return new TextDecoder().decode(buf);
}

// --- validation (verbatim from server/db.js; randomUUID -> Web Crypto) ------

function validateRecord(body) {
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

  const matchup = body.matchup ? String(body.matchup) : buildMatchup(players);

  const record = {
    id: typeof body.id === 'string' && body.id ? body.id : crypto.randomUUID(),
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

// --- D1 queries (async; batch() gives the atomic game+players insert) -------

async function insertGame(db, record) {
  const gameStmt = db.prepare(`
    INSERT INTO games
      (id, created_at, width, height, winner_seat, win_cause, rounds,
       human_count, matchup, players, game_data)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).bind(
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
    typeof record.gameData === 'string' ? record.gameData : JSON.stringify(record.gameData),
  );

  const playerStmt = db.prepare(`
    INSERT INTO game_players
      (game_id, seat, type, name, is_ai, name_locked, won)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `);
  const playerStmts = record.players.map((p) => playerStmt.bind(
    record.id,
    p.seat,
    p.type,
    p.name,
    p.isAI ? 1 : 0,
    p.nameLocked ? 1 : 0,
    record.winnerSeat != null && p.seat === record.winnerSeat ? 1 : 0,
  ));

  // D1 runs a batch as a single atomic transaction. The game row comes first so
  // the players' foreign key is satisfied.
  await db.batch([gameStmt, ...playerStmts]);
  return record.id;
}

async function listGames(db, { limit = 50, offset = 0 } = {}) {
  const lim = Math.min(Math.max(Number(limit) || 50, 1), 200);
  const off = Math.max(Number(offset) || 0, 0);
  const { results } = await db.prepare(`
    SELECT id, created_at, width, height, winner_seat, win_cause, rounds,
           human_count, matchup
    FROM games
    ORDER BY created_at DESC, id DESC
    LIMIT ? OFFSET ?
  `).bind(lim, off).all();
  const total = (await db.prepare('SELECT COUNT(*) AS n FROM games').first()).n;
  return { total, limit: lim, offset: off, games: results.map(rowToSummary) };
}

async function getGame(db, id) {
  const row = await db.prepare('SELECT * FROM games WHERE id = ?').bind(id).first();
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

async function getStats(db) {
  const { results } = await db.prepare(`
    SELECT type, COUNT(*) AS games, SUM(won) AS wins
    FROM game_players
    WHERE is_ai = 1
    GROUP BY type
    ORDER BY type
  `).all();
  const perAi = results.map((r) => ({
    type: r.type,
    games: r.games,
    wins: r.wins,
    winRate: r.games ? r.wins / r.games : 0,
  }));

  const humans = await db.prepare(`
    SELECT COUNT(*) AS games, SUM(won) AS wins
    FROM game_players WHERE is_ai = 0
  `).first();

  const totalGames = (await db.prepare('SELECT COUNT(*) AS n FROM games').first()).n;

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
