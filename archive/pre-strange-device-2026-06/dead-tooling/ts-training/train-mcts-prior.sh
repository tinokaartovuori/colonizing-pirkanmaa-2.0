#!/usr/bin/env bash
# Train the GA policy net (the MCTS prior) and, in parallel, benchmark the
# current champion vs the HARD heuristic with the fast Rust `bench_hard`,
# appending the REAL vs-hard win-rate to benchmark-history.jsonl for the live
# dashboard. (bench_hard ~19 games/s → an 80-game vs-hard point costs ~4s.)
set -u
cd /mnt/x/Github/colonizing-pirkanmaa-2.0
CKPT=rust-trainer/checkpoints
BH=rust-trainer/target/release/bench_hard
GAMES=80

bench_hist() {
  [ -f "$CKPT/champion.json" ] || return 0
  cp "$CKPT/champion.json" /tmp/prior-champ.json 2>/dev/null || return 0
  "$BH" --champion /tmp/prior-champ.json --search none --games "$GAMES" \
        --curriculum bench --seed 1 > /tmp/bh.out 2>/dev/null || return 0
  python3 - "$CKPT" "$GAMES" /tmp/bh.out <<'PY'
import sys, re, json, datetime
ck, games, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
out = open(path).read()
def cnt(label):
    m = re.search(label + r'\s+(\d+)', out)
    return (int(m.group(1)) / games) if m else None
tf = re.search(r'tile-frac\s+([0-9.]+)', out)
g = sum(1 for _ in open(ck + '/log.jsonl'))
rec = {'gen': g, 'winRate': cnt('win-rate'), 'lossRate': cnt('loss-rate'),
       'timeoutRate': cnt('timeout'), 'tileFrac': float(tf.group(1)) if tf else None,
       'ts': datetime.datetime.now().isoformat(timespec='seconds')}
open(ck + '/benchmark-history.jsonl', 'a').write(json.dumps(rec) + '\n')
print('  [vs-hard] gen', g, 'win', rec['winRate'])
PY
}

( cd rust-trainer && target/release/train --resume checkpoints --out checkpoints \
    --gens 300 --pop 96 --games 64 --cap 200 --seed 1 --pfsp 3.0 \
    >> checkpoints/mcts-prior.train.out 2>&1 ) &
TP=$!
echo "MCTS-PRIOR TRAIN START $(date +%T) pid $TP"
bench_hist                              # seed an initial vs-hard point
while kill -0 "$TP" 2>/dev/null; do
  sleep 90
  bench_hist
done
echo "TRAIN DONE $(date +%T) — final vs-hard bench"
bench_hist
echo "ALL DONE $(date +%T)"
