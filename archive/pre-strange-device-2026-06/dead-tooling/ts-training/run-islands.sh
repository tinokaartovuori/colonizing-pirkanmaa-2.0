#!/usr/bin/env bash
# Island-model neuroevolution: launch N independent ES runs (distinct seeds) in
# parallel — one CPU core each — then pick the strongest with tournament.ts.
# Each island checkpoints to training/checkpoints/island-<seed>/best.json.
#
# Usage: bash training/run-islands.sh [N] [GENS] [POP] [GAMES]
set -u
cd "$(dirname "$0")/.."

N="${1:-12}"
GENS="${2:-50}"
POP="${3:-14}"
GAMES="${4:-12}"
ELITE=$(( POP / 3 ))
VN=node_modules/.bin/vite-node
mkdir -p training/checkpoints

echo "launching $N islands: gens=$GENS pop=$POP elite=$ELITE games=$GAMES"
pids=()
for i in $(seq 1 "$N"); do
  out="training/checkpoints/island-$i"
  mkdir -p "$out"
  "$VN" training/evolve.ts -- --gens "$GENS" --pop "$POP" --elite "$ELITE" \
      --games "$GAMES" --seed "$((1000 + i))" --out "$out" \
      > "$out/stdout.log" 2>&1 &
  pids+=($!)
done

echo "island pids: ${pids[*]}"
for p in "${pids[@]}"; do wait "$p"; done
echo "ALL ISLANDS DONE"
