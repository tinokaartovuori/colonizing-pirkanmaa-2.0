-- D1 (Cloudflare serverless SQLite) schema for the game-records backend.
-- Mirrors server/db.js 1:1 so the local Node+SQLite backend and the deployed
-- Worker store identical rows. Apply with:
--   wrangler d1 execute cp-games --remote --file=worker/schema.sql

CREATE TABLE IF NOT EXISTS games (
  id           TEXT PRIMARY KEY,
  created_at   TEXT NOT NULL,
  width        INTEGER NOT NULL,
  height       INTEGER NOT NULL,
  winner_seat  INTEGER,
  win_cause    TEXT,
  rounds       INTEGER,
  human_count  INTEGER NOT NULL,
  matchup      TEXT NOT NULL,   -- e.g. "human(Tino) vs kalevi vs jorma"
  players      TEXT NOT NULL,   -- JSON array of player objects
  game_data    TEXT NOT NULL    -- JSON blob: full replay / per-turn history
);

-- One row per player per game, so "by player type" queries (per-AI win-rate
-- vs humans) stay cheap with no JSON scan.
CREATE TABLE IF NOT EXISTS game_players (
  game_id      TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
  seat         INTEGER NOT NULL,
  type         TEXT NOT NULL,   -- human|jorma|kalevi|gunnar
  name         TEXT,
  is_ai        INTEGER NOT NULL,
  name_locked  INTEGER NOT NULL,
  won          INTEGER NOT NULL,
  PRIMARY KEY (game_id, seat)
);

CREATE INDEX IF NOT EXISTS idx_games_created_at ON games(created_at);
CREATE INDEX IF NOT EXISTS idx_gp_type          ON game_players(type);
CREATE INDEX IF NOT EXISTS idx_gp_type_won      ON game_players(type, won);
