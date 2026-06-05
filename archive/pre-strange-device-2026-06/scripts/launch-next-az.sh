#!/usr/bin/env bash
# Ready-to-fire follow-up experiments to az13 (timeout -0.4 + KL-anchor λ=1.0).
# az13 is the "KL ON, λ=1.0" arm. Pick the variant that matches az13's read:
#
#   az13 DRIFTED (timeout still climbed) -> KL too weak  -> run STRONGKL (λ=2.0)
#   az13 FROZE   (stuck ≈ exp-A 33%)      -> KL too strong -> run WEAKKL  (λ=0.5)
#   az13 WORKED  (want the clean control) -> run NOKL (isolates timeout-penalty alone)
#
# Usage:  bash rust-trainer/launch-next-az.sh <noctrl|strongkl|weakkl>
# Run from the repo root. Each logs to its own checkpoints-azNN/run.out.
# NOTE: only launch ONE at a time — az13 already uses ~16 threads; concurrent
#       runs contend for the 20 cores and slow both. Wait for az13 to finish
#       (or stop it) before firing a successor, unless you drop --threads.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

BIN=rust-trainer/target/release/alphazero
COMMON=(--init-policy rust-trainer/checkpoints-az/champion.json \
        --init-value  rust-trainer/checkpoints-az4/value.json \
        --spatial-policy --spatial-value --leaf-value \
        --sims 96 --iters 250 --games 32 --epochs 2 \
        --bench-games 40 --bench-every 5 --cap 120 --width 14 --height 12 \
        --seed 7 --threads 16 --timeout-penalty 0.4)

case "${1:-}" in
  noctrl)   OUT=rust-trainer/checkpoints-az14-noklctrl; KL=(--kl-anchor 0.0) ;;  # control: timeout-penalty alone
  strongkl) OUT=rust-trainer/checkpoints-az15-strongkl; KL=(--kl-anchor 2.0) ;;
  weakkl)   OUT=rust-trainer/checkpoints-az16-weakkl;   KL=(--kl-anchor 0.5) ;;
  *) echo "usage: $0 <noctrl|strongkl|weakkl>"; exit 1 ;;
esac

mkdir -p "$OUT"
echo "launching $1 -> $OUT (timeout-penalty 0.4, ${KL[*]})"
nohup "$BIN" --out "$OUT" "${COMMON[@]}" "${KL[@]}" > "$OUT/run.out" 2>&1 &
echo "pid $!  (tail -f $OUT/run.out)"
