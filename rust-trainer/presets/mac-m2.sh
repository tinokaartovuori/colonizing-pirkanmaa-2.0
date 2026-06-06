# shellcheck shell=sh
# Apple Silicon (M-series) preset.
#
# Thread choice rationale:
#   - EMPIRICAL (2026-06-05 this Mac, M2 Pro 8P+4E): --threads 8 ran 66s/iter
#     mean (s3), --threads 6 ran 75s/iter mean (i1) -> 8 is ~14% faster. The
#     prior "E-cores drag the pool down" claim is FALSIFIED on this hardware;
#     macOS's scheduler + Rayon work-stealing handle the perf_cores fine.
#   - So: THREADS = perf_cores (use ALL P-cores, leave the E-cores entirely
#     free for OS/desktop/dashboard). 8 P-cores busy + 4 E-cores idle still
#     leaves the desktop responsive.
#     M2 Pro (8P + 4E)  -> 8 threads
#     M1 / M2 base (4P + 4E) -> 4 threads
#     M3 Max (12P + 4E) -> 12 threads
#
# Override at the call site with THREADS_OVERRIDE=N ./launch.sh ... if you know
# better (e.g. probing 10 to fold 2 E-cores in, or backing off for foreground work).

. "$(dirname "$0")/common.sh"

# Detect P-cores. perflevel0 = performance cluster on Apple Silicon.
PERF_CORES="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 0)"

if [ "$PERF_CORES" -ge 2 ]; then
  AUTO_THREADS=$PERF_CORES
else
  # Tiny machine or non-Apple-Silicon Mac: fall back to physicalcpu - 1.
  PHYS="$(sysctl -n hw.physicalcpu 2>/dev/null || echo 2)"
  AUTO_THREADS=$((PHYS - 1))
  [ "$AUTO_THREADS" -lt 1 ] && AUTO_THREADS=1
fi

THREADS="${THREADS_OVERRIDE:-$AUTO_THREADS}"

DEVICE_LABEL="mac-m2 (perf_cores=$PERF_CORES, threads=$THREADS)"
