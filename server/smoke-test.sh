#!/usr/bin/env bash
# Smoke test for the games backend.
# Usage:  server/smoke-test.sh [BASE_URL]
# Assumes the server is already running (node server/server.js).
set -euo pipefail
BASE="${1:-http://127.0.0.1:8790}"

echo "== 1. POST a human-vs-AI game (expect 200, stored) =="
HUMAN_GAME='{
  "map": { "width": 12, "height": 12 },
  "players": [
    { "seat": 0, "type": "human",  "name": "Tino",  "nameLocked": true },
    { "seat": 1, "type": "kalevi", "name": "Kalevi", "nameLocked": false },
    { "seat": 2, "type": "jorma",  "name": "Jorma",  "nameLocked": false }
  ],
  "outcome": { "winnerSeat": 0, "winCause": "conquest", "rounds": 47 },
  "gameData": { "frames": [ { "round": 1, "note": "demo replay blob" } ] }
}'
RESP=$(curl -s -w '\n%{http_code}' -X POST "$BASE/api/games" \
  -H 'Content-Type: application/json' -d "$HUMAN_GAME")
echo "$RESP"
CODE=$(echo "$RESP" | tail -n1)
[ "$CODE" = "200" ] || { echo "FAIL: expected 200"; exit 1; }

echo
echo "== 2. POST an all-AI game (expect 422, rejected) =="
AI_GAME='{
  "map": { "width": 12, "height": 12 },
  "players": [
    { "seat": 0, "type": "kalevi" },
    { "seat": 1, "type": "jorma" }
  ],
  "outcome": { "winnerSeat": 0, "winCause": "domination", "rounds": 30 },
  "gameData": { "frames": [] }
}'
RESP=$(curl -s -w '\n%{http_code}' -X POST "$BASE/api/games" \
  -H 'Content-Type: application/json' -d "$AI_GAME")
echo "$RESP"
CODE=$(echo "$RESP" | tail -n1)
[ "$CODE" = "422" ] || { echo "FAIL: expected 422"; exit 1; }

echo
echo "== 3. GET list (expect the human game) =="
curl -s "$BASE/api/games" ; echo

echo
echo "== 4. GET stats =="
curl -s "$BASE/api/games/stats" ; echo

echo
echo "ALL SMOKE TESTS PASSED"
