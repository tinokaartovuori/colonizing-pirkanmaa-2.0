# Game-records backend on Cloudflare (Worker + D1)

This is the **public, free, always-on** host for the game-records backend — the
deployed twin of `server/` (Node + SQLite). It stores completed **human-vs-AI**
games POSTed from the live GitHub Pages site so they can be reviewed later
(primarily: how the AIs did vs humans). Pure AI-vs-AI games are rejected `422`.

- **Worker** (`src/index.js`) = the HTTP API, ported 1:1 from `server/server.js`
  + `server/db.js`. Same endpoints, same request/response shapes.
- **D1** = Cloudflare's serverless SQLite. Schema in `schema.sql` — identical to
  the local backend's. Free tier: 5 GB storage, 5M rows read/day, 100k Worker
  requests/day. Far more than this game will ever use.

## Endpoints (unchanged from `server/README.md`)

| Method | Path               | Purpose                                       |
|--------|--------------------|-----------------------------------------------|
| `POST` | `/api/games`       | Store a completed game (rejects all-AI → 422) |
| `GET`  | `/api/games`       | List games, paginated (`?limit=&offset=`)     |
| `GET`  | `/api/games/:id`   | Full record incl. `gameData` blob             |
| `GET`  | `/api/games/stats` | Per-AI win-rate vs humans + totals            |
| `GET`  | `/health`          | `{ "ok": true }`                              |

The POST request body shape is documented in `server/README.md` — it is the exact
body `src/managers/gamerecorder.ts` already sends.

---

## One-time deploy (≈5 minutes)

You need a free Cloudflare account. These steps use your account auth, so **you**
run them (e.g. type them here with a leading `!` to run in-session). `wrangler` is
Cloudflare's CLI; `npx wrangler` fetches it on demand — nothing to install.

All commands run from the **repo root**, pointing at this folder's config.

```bash
# 0. Log in (opens a browser to authorize your Cloudflare account)
npx wrangler login

# 1. Create the D1 database. Prints a `database_id = "..."` line.
npx wrangler d1 create cp-games --config worker/wrangler.toml

# 2. Paste that id into worker/wrangler.toml  →  database_id = "..."

# 3. Create the tables in the remote D1 database
npx wrangler d1 execute cp-games --remote --config worker/wrangler.toml --file=worker/schema.sql

# 4. Deploy the Worker. Prints the public URL, e.g.
#    https://cp-games.<your-subdomain>.workers.dev
npx wrangler deploy --config worker/wrangler.toml
```

### Point the live site at it

The deployed frontend reads `VITE_CP_SERVER` at build time. Set it as a **repo
variable** (it's a public URL, not a secret — it ends up in the client bundle):

1. GitHub → repo **Settings → Secrets and variables → Actions → Variables tab →
   New repository variable**
2. Name `VITE_CP_SERVER`, value = the `https://cp-games.<subdomain>.workers.dev`
   URL from step 4 (no trailing slash).
3. Push any commit to `main` (or re-run the deploy workflow) so the bundle rebuilds
   with the URL baked in.

`.github/workflows/deploy.yml` already passes `vars.VITE_CP_SERVER` into the build.

---

## Verify it works

```bash
URL=https://cp-games.<your-subdomain>.workers.dev

# health
curl -s $URL/health                         # → {"ok":true}

# human-vs-AI → 200 stored
curl -s -X POST $URL/api/games -H 'Content-Type: application/json' -d '{
  "map":{"width":12,"height":12},
  "players":[{"seat":0,"type":"human","name":"Tino"},{"seat":1,"type":"kalevi"}],
  "outcome":{"winnerSeat":0,"winCause":"conquest","rounds":47},
  "gameData":{"frames":[]}
}'                                           # → {"ok":true,"id":"...","matchup":"..."}

# all-AI → 422 rejected (the core rule)
curl -s -X POST $URL/api/games -H 'Content-Type: application/json' -d '{
  "map":{"width":12,"height":12},
  "players":[{"seat":0,"type":"kalevi"},{"seat":1,"type":"jorma"}],
  "outcome":{"winnerSeat":0,"winCause":"domination","rounds":30},
  "gameData":{"frames":[]}
}'                                           # → {"ok":false,...,"code":"NO_HUMAN"}

curl -s $URL/api/games                       # list
curl -s $URL/api/games/stats                 # per-AI win-rate vs humans
```

Then the real end-to-end check: play a human-vs-Kalevi game to completion on the
live site → `GET /api/games` shows the row (matchup + per-turn `history` +
`finalSnapshot` in `gameData`).

## Reading the collected data

```bash
# Browse stored games straight from D1 (read-only SQL)
npx wrangler d1 execute cp-games --remote --config worker/wrangler.toml \
  --command "SELECT created_at, matchup, win_cause, rounds FROM games ORDER BY created_at DESC LIMIT 20"
```

Or just `curl $URL/api/games` / `$URL/api/games/stats`.

## Local dev (optional)

`npx wrangler dev --config worker/wrangler.toml` runs the Worker locally against a
local D1 file — apply the schema once with `--local` instead of `--remote`. For
plain local testing the existing `npm run server` (Node + SQLite) is simpler and
serves the identical API.
