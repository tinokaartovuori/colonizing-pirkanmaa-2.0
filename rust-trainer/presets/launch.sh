#!/bin/sh
# Cross-device launcher for cnn_train.
#
# Usage:
#   ./rust-trainer/presets/launch.sh [--print-cmd] [extra cnn_train flags...]
#
# Examples (per-experiment knobs after a `--` are appended verbatim):
#   ./rust-trainer/presets/launch.sh --print-cmd \
#       --out rust-trainer/checkpoints-cnn-b2 --w-army 0.6 --cap-potential 0.4
#
#   ./rust-trainer/presets/launch.sh \
#       --out rust-trainer/checkpoints-cnn-b2 --w-army 0.6 --cap-potential 0.4 \
#       --script-frac 0.5
#
# Auto-detects OS (uname -s) and sources the right device preset:
#   Darwin  -> mac-m2.sh
#   Linux   -> linux-pc.sh
#
# Override the auto-detected device with CP_DEVICE=mac-m2 or CP_DEVICE=linux-pc.
# Override the auto-detected thread count with THREADS_OVERRIDE=N.

set -eu

PRESETS_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- flag parsing -----------------------------------------------------------
PRINT_CMD=0
# Walk argv; pull --print-cmd / -n out, leave the rest for cnn_train.
EXTRA_ARGS=""
for arg in "$@"; do
  case "$arg" in
    --print-cmd|-n)
      PRINT_CMD=1
      ;;
    --help|-h)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *)
      # Re-quote each forwarded arg so values with spaces survive.
      EXTRA_ARGS="$EXTRA_ARGS $(printf '%s' "$arg" | sed "s/'/'\\\\''/g; s/^/'/; s/\$/'/")"
      ;;
  esac
done

# --- device selection -------------------------------------------------------
if [ -n "${CP_DEVICE:-}" ]; then
  DEVICE="$CP_DEVICE"
else
  case "$(uname -s)" in
    Darwin) DEVICE=mac-m2 ;;
    Linux)  DEVICE=linux-pc ;;
    *)
      echo "launch.sh: unknown OS '$(uname -s)'. Set CP_DEVICE=mac-m2 or CP_DEVICE=linux-pc." >&2
      exit 2
      ;;
  esac
fi

PRESET_FILE="$PRESETS_DIR/$DEVICE.sh"
if [ ! -f "$PRESET_FILE" ]; then
  echo "launch.sh: no preset for device '$DEVICE' (expected $PRESET_FILE)." >&2
  exit 2
fi

# shellcheck disable=SC1090
. "$PRESET_FILE"
# After sourcing, the preset has defined:
#   CNN_TRAIN_BIN, COMMON_FLAGS, THREADS, DEVICE_LABEL

# --- binary check -----------------------------------------------------------
if [ ! -x "$CNN_TRAIN_BIN" ]; then
  echo "launch.sh: cnn_train binary not found or not executable:" >&2
  echo "  $CNN_TRAIN_BIN" >&2
  echo "Build it first:  (cd rust-trainer && cargo build --release)" >&2
  exit 1
fi

# --- assemble + run ---------------------------------------------------------
# Note: COMMON_FLAGS expands as multiple words on purpose (unquoted). EXTRA_ARGS
# is built with single-quoting per-token, so `eval` reproduces the user's args
# exactly while still passing them as separate words.
CMD="\"$CNN_TRAIN_BIN\" $COMMON_FLAGS --threads $THREADS$EXTRA_ARGS"

if [ "$PRINT_CMD" -eq 1 ]; then
  printf '# device:   %s\n' "$DEVICE_LABEL"
  printf '# binary:   %s\n' "$CNN_TRAIN_BIN"
  printf '# threads:  %s\n' "$THREADS"
  printf '%s\n' "$CMD"
  exit 0
fi

eval "exec $CMD"
