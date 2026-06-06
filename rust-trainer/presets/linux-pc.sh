# shellcheck shell=sh
# Linux desktop preset (x86_64, no P/E split).
#
# Thread choice rationale:
#   - handoff.md "Step-1 baseline launch" uses --threads 16 on the dev box and
#     reports ~40-60 s/iter for the small net at that setting. Treat 16 as the
#     known-good floor.
#   - CLAUDE.md target: leave ~4 cores free for the desktop. If the box has
#     more than 20 logical cores, prefer nproc - 4 over the floor of 16.
#   - Override with THREADS_OVERRIDE=N ./launch.sh ... if you want to pin it
#     (e.g. headless / batch / want all cores).

. "$(dirname "$0")/common.sh"

FLOOR=16

# Detect logical cores. nproc on Linux; fall back to /proc/cpuinfo; final
# fall back to the floor so we never crash on minimal systems.
if command -v nproc >/dev/null 2>&1; then
  NPROC="$(nproc)"
elif [ -r /proc/cpuinfo ]; then
  NPROC="$(grep -c ^processor /proc/cpuinfo)"
else
  NPROC="$FLOOR"
fi

LEAVE_FREE=$((NPROC - 4))
if [ "$LEAVE_FREE" -gt "$FLOOR" ]; then
  AUTO_THREADS="$LEAVE_FREE"
else
  AUTO_THREADS="$FLOOR"
fi

THREADS="${THREADS_OVERRIDE:-$AUTO_THREADS}"

DEVICE_LABEL="linux-pc (nproc=$NPROC, threads=$THREADS)"
