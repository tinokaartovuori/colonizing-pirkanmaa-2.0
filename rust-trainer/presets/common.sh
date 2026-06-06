# shellcheck shell=sh
# Canonical, device-independent trainer flag set.
#
# Source of truth: the Step-2 launch in handoff.md ("The immediate next task"),
# which is the Step-1 baseline + the army knobs (--w-army, --w-cut).
#
# Per-device knobs:  --threads          (in mac-m2.sh / linux-pc.sh)
# Per-experiment knobs (stripped here; pass on the command line):
#   --out --w-army --w-expert --cap-potential --script-frac --idle-flow-penalty
#   --build-prior-floor --sims --iters --vs-hard-frac --bankruptcy-discount --net-size
#   ...and anything else you're sweeping
# (OVERNIGHT-RUN cnn-r2: --iters and --vs-hard-frac stripped so r2's --iters 400
# / --vs-hard-frac 0.2 actually wins over the preset baseline. arg_val uses the
# FIRST occurrence — see note below.)
#
# NOTE: cnn_train's arg parser (`arg_val` in cnn_train.rs:1579) uses the FIRST
# occurrence of a flag, so adding a flag here and then "overriding" it on the
# command line silently has NO effect — the baked-in value wins. Always strip
# a knob from this file before sweeping it.
#
# DO NOT add experiment-specific knobs here. This file is one source of truth
# for the FIXED part of the canonical command, copied verbatim from handoff.md
# minus the stripped knobs above.

# Resolve the repo root (parent of rust-trainer/) from this file's location so
# the preset works regardless of the caller's cwd.
PRESETS_DIR="$(cd "$(dirname "$0")" && pwd)"
RUST_TRAINER_DIR="$(cd "$PRESETS_DIR/.." && pwd)"
REPO_ROOT="$(cd "$RUST_TRAINER_DIR/.." && pwd)"
CNN_TRAIN_BIN="$RUST_TRAINER_DIR/target/release/cnn_train"

# Canonical flags, in handoff.md order, minus the stripped knobs.
# Quoted as one shell word per token so a `set --` style expansion works.
COMMON_FLAGS="\
--train \
--turn-search \
--income-lead-potential 0.5 \
--tile-potential 0.4 \
--w-cut 0.15 \
--record-opp-value \
--device-potential 0.2 \
--device-credit 0.15 \
--pfsp \
--script-opponents \
--script-grade \
--tie-penalty 0.4 \
--stall-rounds 80 \
--shape-gamma 0.99 \
--shape-weight 0.3 \
--cap 150 \
--games 24 \
--bench-games 60"
