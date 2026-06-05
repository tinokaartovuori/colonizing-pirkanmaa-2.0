#!/usr/bin/env bash
# Resume the Phase-2 GA training and, in parallel, periodically benchmark the
# current champion vs the hard heuristic — writing benchmark.json (latest) AND
# appending a time-series line to benchmark-history.jsonl (gen,winRate,...) that
# the live dashboard plots as the win-rate curve.
set -u
cd /mnt/x/Github/colonizing-pirkanmaa-2.0
CKPT=rust-trainer/checkpoints

append_hist() {
  python3 - "$CKPT" <<'PY'
import json, datetime, sys
ck = sys.argv[1]
try:
    b = json.load(open(ck + '/benchmark.json'))
    g = sum(1 for _ in open(ck + '/log.jsonl'))
    rec = {'gen': g, 'winRate': b.get('winRate'), 'lossRate': b.get('lossRate'),
           'timeoutRate': b.get('timeoutRate'), 'tileFrac': b.get('avgFinalTileFrac'),
           'ts': datetime.datetime.now().isoformat(timespec='seconds')}
    open(ck + '/benchmark-history.jsonl', 'a').write(json.dumps(rec) + '\n')
    print('  [hist] gen', g, 'win', rec['winRate'])
except Exception as e:
    print('  [hist] skip:', e)
PY
}

( cd rust-trainer && target/release/train --resume checkpoints --out checkpoints \
    --gens 450 --pop 96 --games 64 --cap 200 --seed 1 --pfsp 3.0 \
    >> checkpoints/long.train.out 2>&1 ) &
TRAIN_PID=$!
echo "resumed train pid $TRAIN_PID ($(date +%T))"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 300
  if [ -f "$CKPT/champion.json" ]; then
    cp "$CKPT/champion.json" /tmp/bench-champ.json 2>/dev/null
    npx vite-node training/benchmark.ts -- --champion /tmp/bench-champ.json --games 40 --seed 7 >/dev/null 2>&1 || true
    append_hist
  fi
done

echo "TRAIN DONE $(date +%T) — final 200-game benchmark"
npx vite-node training/benchmark.ts -- --champion "$CKPT/champion.json" --games 200 --seed 7 2>&1 \
  | grep -E "win-rate|loss-rate|timeout-rate|avg tile frac"
append_hist
echo "ALL DONE $(date +%T)"
