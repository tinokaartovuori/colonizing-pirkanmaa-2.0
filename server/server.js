// Lightweight self-hosted backend for storing completed games of
// Colonizing Pirkanmaa. Node's built-in http + node:sqlite — no deps.
//
// Run:  node server/server.js
// Env:  CP_PORT (default 8790), CP_DB (default server/data/games.db)
import http from 'node:http';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  openDb, validateRecord, insertGame, listGames, getGame, getStats,
} from './db.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

const PORT = Number(process.env.CP_PORT) || 8790;
const DB_PATH = resolve(process.env.CP_DB || resolve(__dirname, 'data', 'games.db'));
const MAX_BODY = 32 * 1024 * 1024; // 32 MB — replays can be large

mkdirSync(dirname(DB_PATH), { recursive: true });
const db = openDb(DB_PATH);

function send(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(body),
  });
  res.end(body);
}

function setCors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

function readBody(req) {
  return new Promise((resolveBody, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (c) => {
      size += c.length;
      if (size > MAX_BODY) {
        reject(Object.assign(new Error('payload too large'), { status: 413 }));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => resolveBody(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  setCors(res);
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  const url = new URL(req.url, `http://${req.headers.host}`);
  const path = url.pathname.replace(/\/+$/, '') || '/';

  try {
    if (req.method === 'GET' && path === '/health') {
      return send(res, 200, { ok: true });
    }

    if (req.method === 'POST' && path === '/api/games') {
      const raw = await readBody(req);
      let body;
      try { body = JSON.parse(raw || '{}'); }
      catch { return send(res, 400, { ok: false, error: 'invalid JSON' }); }

      const v = validateRecord(body);
      if (!v.ok) {
        return send(res, v.status, { ok: false, error: v.error, code: v.code });
      }
      const id = insertGame(db, v.record);
      return send(res, 200, { ok: true, id, matchup: v.record.matchup });
    }

    if (req.method === 'GET' && path === '/api/games') {
      const result = listGames(db, {
        limit: url.searchParams.get('limit'),
        offset: url.searchParams.get('offset'),
      });
      return send(res, 200, { ok: true, ...result });
    }

    if (req.method === 'GET' && path === '/api/games/stats') {
      return send(res, 200, { ok: true, ...getStats(db) });
    }

    const m = path.match(/^\/api\/games\/([^/]+)$/);
    if (req.method === 'GET' && m) {
      const game = getGame(db, decodeURIComponent(m[1]));
      if (!game) return send(res, 404, { ok: false, error: 'not found' });
      return send(res, 200, { ok: true, game });
    }

    return send(res, 404, { ok: false, error: 'not found' });
  } catch (e) {
    const status = e.status || 500;
    return send(res, status, { ok: false, error: e.message || 'internal error' });
  }
});

server.listen(PORT, () => {
  console.log(`[cp-server] listening on http://127.0.0.1:${PORT}  db=${DB_PATH}`);
});
