# Handoff — game shipped; NEXT: persist human-vs-AI games to a server

_Updated 2026-06-09. Previous handoffs covered training/ceiling; this one covers the
**deployed game** and grounds the next task: make the **human-vs-AI game-saving mechanism
actually work in production**._

---

## NEXT TASK (do this) — host the game-record backend so played games are saved

**Goal:** when a real person plays a game against the AIs on the live site, that completed
game (who played whom + the full per-turn history + outcome) gets saved to a server so it can
be reviewed later — primarily to study **how the AI did vs humans**. Pure AI-vs-AI games are
NOT saved.

**Status: built end-to-end; not yet live (awaiting the user's Cloudflare deploy).** Client +
local backend + the Cloudflare Worker are all done — only the account-auth deploy steps remain
(see "What's left" below). Pieces:
- ✅ **Client side DONE**: `src/managers/gamerecorder.ts` (`GameRecorder`) records a per-turn
  history during the match and, on game end **if ≥1 human seat**, POSTs the full record to
  `${VITE_CP_SERVER}/api/games` (best-effort; never blocks the UI). Hooked via
  `GameEventHandler.onTurnEnded` / `onGameOver` (added this session), wired in `src/main.ts`.
- ✅ **Backend DONE (but only runs locally)**: `server/` — zero-dependency Node + `node:sqlite`.
  `npm run server` → `http://127.0.0.1:8790`. Endpoints: `POST /api/games` (enforces the
  no-human rule server-side → 422 `NO_HUMAN`), `GET /api/games`, `GET /api/games/:id`,
  `GET /api/games/stats` (per-AI win-rate vs humans). CORS `*`. Schema + request shape in
  `server/README.md`. Smoke test: `server/smoke-test.sh`.
- ✅ **Cloudflare Worker + D1 DONE (`worker/`)**: `worker/src/index.js` (API, ported 1:1 from
  `server/`), `worker/schema.sql` (D1 tables, identical schema), `worker/wrangler.toml` (binding;
  `database_id` placeholder filled at deploy), `worker/README.md` (deploy guide), `npm run
  worker:*` scripts. `deploy.yml` now injects `vars.VITE_CP_SERVER` into the build.
- ❌ **THE GAP (remaining)**: nobody has run the Cloudflare deploy yet, so there's no public URL
  and `VITE_CP_SERVER` isn't set → the deployed build still falls back to `http://127.0.0.1:8790`
  and uploads no-op for visitors. Closing it = the account-auth deploy steps below.

**Why it's not trivial:** GitHub Pages is **static** (no backend). The game is public + HTTPS, so
the backend must be a **public HTTPS endpoint** the github.io page can POST to (a localhost or
plain-HTTP backend is blocked by mixed-content + is unreachable to visitors).

**Chosen host: Cloudflare Worker + D1** (free, serverless SQLite, always-on, reachable from the
static Pages site). The Worker is now **built** (`worker/`) — `server/db.js`+`server.js` ported
1:1 to the Workers runtime + D1, same endpoints/shapes/schema. Full deploy walkthrough in
`worker/README.md`. Rejected alternatives: self-host Node+SQLite (needs an always-on box + cert),
Turso (extra managed dep) — both strictly more upkeep than D1.

**What's left = account-auth steps only (need the user's Cloudflare login; ~5 min):**
1. **Deploy the Worker** (run from repo root; `npx` fetches wrangler — nothing to install):
   - `npm run worker:login` → authorize Cloudflare account in browser
   - `npm run worker:create` → prints `database_id`; paste it into `worker/wrangler.toml`
   - `npm run worker:schema` → creates the tables in remote D1
   - `npm run worker:deploy` → prints the public `https://cp-games.<subdomain>.workers.dev` URL
2. **Point the frontend at it**: GitHub → repo Settings → Secrets and variables → Actions →
   **Variables** → add `VITE_CP_SERVER` = that Worker URL (no trailing slash). `deploy.yml` already
   passes `vars.VITE_CP_SERVER` into `npm run build`, so the next push to `main` bakes it in.
   (It's a public URL embedded in the client → a repo *variable*, not a secret.)
3. **Verify**: `curl $URL/health` → `{"ok":true}`; then play a human-vs-Kalevi game on the live
   site to completion → `GET /api/games` shows the row (matchup + per-turn `history` +
   `finalSnapshot` in `gameData`). An AI-vs-AI game must NOT appear (422 `NO_HUMAN`).

**POST body the client sends** (match this on any rehost — full spec in `server/README.md`):
```jsonc
{ "map": {"width","height"},
  "players": [{"seat","type":"human|jorma|kalevi|gunnar","name","nameLocked"}],
  "outcome": {"winnerSeat","winCause","rounds"},
  "gameData": { "seed", "history":[{round,seat,snapshot,metrics[]}], "finalSnapshot", "winnerSeat","winCause" } }
```
`history` = one entry per turn: a full `buildSnapshot` (`src/managers/persistence.ts`; captures the
Strange Device pos + countdown too) + per-player metrics (money/wood/stone/metal, tiles, soldiers,
buildings, hasDevice, deviceCountdown) → fully replayable for analysis.

---

## Current live state

- **Game is LIVE (free, no Cloudflare needed for the frontend):**
  https://tinokaartovuori.github.io/colonizing-pirkanmaa-2.0/
  Repo was made **PUBLIC** (user-approved) so free GitHub Pages works. `.github/workflows/deploy.yml`
  auto-builds + deploys on **push to `main`**. (Free Pages doesn't support private repos — that's why.)
- **AI opponents** (start dialog; locked names, dup → Jorma1/Jorma2; humans editable):
  - **Jorma** = HARD heuristic (`'hard'`, `src/managers/ai.ts`). sd5 soldier-metal gates fixed this session (50→30).
  - **Kalevi** = `model:kalevi` — small AlphaZero net (`rust-trainer/exit2-best.json`), **5-seed mean 0.643 vs HARD** (win-rate pick).
  - **Gunnar** = `model:gunnar` — large AlphaZero net (`rust-trainer/large2-best.json`), 0.585 vs HARD but **more aggressive** (army 1.22); **beats Kalevi head-to-head ~0.55**.
- **Neural MCTS runs in a Web Worker** (`src/ai/nn/worker.ts` + `mcts-worker-client.ts`) — no UI freeze while the AI thinks; read-only (snapshot→index), graceful fallback to in-thread search.
- **Strange Device**: new tile art (`public/assets/images/strange_device.png`) + countdown over the dome (`GameScene.refreshDeviceMarker`); build-menu preview; descriptions corrected to the sd5 **−2 soldier-cap** (was stale "halves"). Snapshot save/restore of the device is covered + tested (`tests/strangedevice.test.ts`).
- **Models**: both champions deployed as a TS roster (`src/ai/nn/models_spatial_roster.ts`; regenerate via `vite-node training/emit-spatial-roster.ts`). Forward parity Rust⇄TS = 0.0 diff for both arches (incl. Gunnar's residual conv3).

## Training conclusion (context; NOT the next task)
~**0.64 is the real win-rate ceiling vs HARD** for this game — confirmed across small-net AZ/PPO
(7 runs) + a large-net cold-start. **Capacity buys aggression, not win-rate** (Gunnar). Two
champions on a Pareto frontier (Kalevi = win-rate, Gunnar = aggression), both banked + deployed.
Detail in memories: `capacity-aggression-not-winrate`, `search-scaling-curve`, `bench-seed-variance`.
The `--multi-eval` harness (`cnn_train --multi-eval --net-a --net-b`) does head-to-head + 3p/4p (eval-only).

## Operational notes
- **Deploy = push to `main`** (auto). Recent commits: 77fc572 (game+roster+backend), 4b1f6a3 (worker+UI),
  a9bf4e2 (name-validation fix), 54cf706 (device text). Build gate: `npm run build` (tsc --noEmit + vite).
- **Pre-existing red tests** (unrelated; fail on a clean stash): `nnai` weights 64-vs-68, `restore` seed-19,
  `spatial-deploy`, `spatial-mcts-strength` — uncommitted weight/checkpoint drift, not deploy bugs.
- **Bash gotcha**: `pkill -f "<pattern>"` self-matches the running shell if the pattern text appears in the
  command → kills the shell (exit 144). Kill by PID, `lsof -ti tcp:<port>`, or `killall <procname>` instead.
- The huge `rust-trainer/` + `models/` working-tree churn is training noise — NOT needed for the game build;
  deploy commits touch only `src/`, `public/`, `server/`, `package.json`, `.github/`.
