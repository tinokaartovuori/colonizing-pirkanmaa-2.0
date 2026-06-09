# Games backend (`server/`)

A minimal, self-hosted **Node + SQLite** backend that persists **completed games**
of Colonizing Pirkanmaa for later analysis.

It only stores games that included **at least one human player** (human-vs-AI
matchups). Pure AI-vs-AI games are rejected server-side and never written.

- **Zero external dependencies** — uses Node's built-in `node:http` and
  `node:sqlite`.
- The game engine / client code is **not** touched. This package only defines and
  documents the JSON shape the client should `POST`; wiring the client is a
  separate task.

## Requirements

- Node **22.5+** (run with `--experimental-sqlite`) or Node **24+** (stable, flag
  is a harmless no-op). The repo pins Node 22 in `.nvmrc`.

## Run

```bash
npm run server
# → [cp-server] listening on http://127.0.0.1:8790  db=…/server/data/games.db
```

Or directly:

```bash
node --experimental-sqlite server/server.js
```

### Config (env vars)

| Var       | Default                  | Meaning                |
|-----------|--------------------------|------------------------|
| `CP_PORT` | `8790`                   | HTTP port              |
| `CP_DB`   | `server/data/games.db`   | SQLite file path       |

```bash
CP_PORT=9000 CP_DB=/var/lib/cp/games.db node --experimental-sqlite server/server.js
```

CORS is enabled (`Access-Control-Allow-Origin: *`) so the static GitHub-Pages
frontend can POST from a different origin.

## Endpoints

| Method | Path                  | Purpose                                              |
|--------|-----------------------|------------------------------------------------------|
| `POST` | `/api/games`          | Store a completed game (rejects all-AI games)        |
| `GET`  | `/api/games`          | List games, paginated (`?limit=&offset=`), summaries |
| `GET`  | `/api/games/:id`      | Full record incl. `gameData` blob                    |
| `GET`  | `/api/games/stats`    | Per-AI win-rate vs humans + totals                   |
| `GET`  | `/health`             | `{ "ok": true }`                                      |

### `POST /api/games` — exact request shape (client must match this)

```jsonc
{
  "id": "optional-client-uuid",          // optional; server generates one if absent
  "map":     { "width": 12, "height": 12 },
  "players": [
    { "seat": 0, "type": "human",  "name": "Tino",   "nameLocked": true  },
    { "seat": 1, "type": "kalevi", "name": "Kalevi", "nameLocked": false },
    { "seat": 2, "type": "jorma",  "name": "Jorma",  "nameLocked": false }
  ],
  "outcome": {
    "winnerSeat": 0,                      // seat of winner; null/omit for tie
    "winCause": "conquest",              // conquest|domination|device|bankruptcy|tie|resign|other
    "rounds": 47
  },
  "matchup": "optional override string", // optional; derived from players if absent
  "gameData": { "frames": [ /* … */ ] }  // REQUIRED: full replay / move log, stored verbatim
}
```

Field notes:

- **`players[].type`** must be one of `human`, `jorma`, `kalevi`, `gunnar`.
  `isAI` is **derived** server-side (`type !== "human"`) — the client does not
  need to send it (if it does, it's ignored).
- **`players[].seat`** — integer seat index; defaults to array index if omitted.
- **`players[].nameLocked`** — boolean, stored as-is.
- **`gameData`** is required and stored verbatim as a JSON blob (can be an object
  or a string). Max body size is 32 MB.
- **`outcome.winnerSeat`** is matched against each `players[].seat` to mark the
  per-player `won` flag used by `/stats`.
- snake_case aliases are also accepted on outcome / gameData
  (`winner_seat`, `win_cause`, `game_data`) for convenience.

### Responses

- `200 { "ok": true, "id": "...", "matchup": "..." }` — stored.
- `422 { "ok": false, "error": "...", "code": "NO_HUMAN" }` — **no human player**;
  not stored (the core rule).
- `400 { "ok": false, "error": "..." }` — invalid body (bad/missing map, players,
  type, or `gameData`).

## SQLite schema

```sql
CREATE TABLE games (
  id           TEXT PRIMARY KEY,
  created_at   TEXT NOT NULL,          -- ISO-8601 UTC
  width        INTEGER NOT NULL,
  height       INTEGER NOT NULL,
  winner_seat  INTEGER,
  win_cause    TEXT,
  rounds       INTEGER,
  human_count  INTEGER NOT NULL,
  matchup      TEXT NOT NULL,          -- e.g. "human(Tino) vs kalevi vs jorma"
  players      TEXT NOT NULL,          -- JSON array of player objects
  game_data    TEXT NOT NULL           -- JSON replay / move-log blob (verbatim)
);

-- One row per player per game → cheap "by player type" queries (no JSON scan).
CREATE TABLE game_players (
  game_id      TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
  seat         INTEGER NOT NULL,
  type         TEXT NOT NULL,          -- human|jorma|kalevi|gunnar
  name         TEXT,
  is_ai        INTEGER NOT NULL,       -- 0/1
  name_locked  INTEGER NOT NULL,       -- 0/1
  won          INTEGER NOT NULL,       -- 0/1
  PRIMARY KEY (game_id, seat)
);

CREATE INDEX idx_games_created_at ON games(created_at);
CREATE INDEX idx_gp_type          ON game_players(type);
CREATE INDEX idx_gp_type_won      ON game_players(type, won);
```

## Smoke test

With the server running (`npm run server`):

```bash
server/smoke-test.sh                    # defaults to http://127.0.0.1:8790
# or against a custom URL:
server/smoke-test.sh http://127.0.0.1:9000
```

It (1) POSTs a human-vs-AI game → `200`, (2) POSTs an all-AI game → `422`
`NO_HUMAN`, (3) lists games, (4) prints stats, and exits non-zero on any
mismatch.

Equivalent manual curl:

```bash
# human-vs-AI → 200 stored
curl -s -X POST http://127.0.0.1:8790/api/games -H 'Content-Type: application/json' -d '{
  "map":{"width":12,"height":12},
  "players":[{"seat":0,"type":"human","name":"Tino"},{"seat":1,"type":"kalevi"}],
  "outcome":{"winnerSeat":0,"winCause":"conquest","rounds":47},
  "gameData":{"frames":[]}
}'

# all-AI → 422 rejected (not stored)
curl -s -X POST http://127.0.0.1:8790/api/games -H 'Content-Type: application/json' -d '{
  "map":{"width":12,"height":12},
  "players":[{"seat":0,"type":"kalevi"},{"seat":1,"type":"jorma"}],
  "outcome":{"winnerSeat":0,"winCause":"domination","rounds":30},
  "gameData":{"frames":[]}
}'

curl -s http://127.0.0.1:8790/api/games
curl -s http://127.0.0.1:8790/api/games/stats
```
