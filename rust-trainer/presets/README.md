# rust-trainer/presets — device-aware launcher for `cnn_train`

One source of truth for the long canonical flag list, with per-device thread
tuning so switching machines doesn't mean editing the command every time.

## Files

| File           | Role                                                                              |
|----------------|-----------------------------------------------------------------------------------|
| `common.sh`    | The fixed canonical flag set (Step-2 launch, copied verbatim from `handoff.md` minus the per-experiment knobs `--out --threads --w-army --cap-potential --script-frac`). |
| `mac-m2.sh`    | Apple Silicon preset. Detects perf-cores via `sysctl hw.perflevel0.physicalcpu`. Sets `THREADS = perf_cores - 2` (M2 Pro 8P -> **6 threads**), leaving E-cores + 2 P-cores free for the desktop. |
| `linux-pc.sh`  | Linux preset. Floors at `--threads 16` (the value used on the dev box per `handoff.md`); if `nproc - 4 > 16`, uses `nproc - 4`. |
| `launch.sh`    | OS auto-detect (`uname -s`), sources the right preset, runs the binary. Supports `--print-cmd` for a dry run. |

## Usage

```bash
# Dry run — print the resolved command without executing.
./rust-trainer/presets/launch.sh --print-cmd \
  --out rust-trainer/checkpoints-cnn-b2 --w-army 0.6 --cap-potential 0.4

# Real run — same form, no --print-cmd. Per-experiment knobs go at the end.
./rust-trainer/presets/launch.sh \
  --out rust-trainer/checkpoints-cnn-b2 \
  --w-army 0.6 --cap-potential 0.4 --script-frac 0.5
```

## Overrides

- `THREADS_OVERRIDE=N` — pin the thread count manually (skips auto-detect).
- `CP_DEVICE=mac-m2|linux-pc` — force a preset (skips `uname -s`).

```bash
THREADS_OVERRIDE=24 ./rust-trainer/presets/launch.sh --print-cmd --out rust-trainer/checkpoints-x
```

## Exit codes

| Code | Meaning                                                          |
|------|------------------------------------------------------------------|
| 0    | OK (or `--print-cmd` succeeded).                                 |
| 1    | `cnn_train` binary missing — run `cargo build --release` first.  |
| 2    | Unknown OS / unknown `CP_DEVICE`.                                |

## Editing the canonical flags

Edit only `common.sh`. The Step-2 launch from `handoff.md` is the source of
truth; if you sync a new baseline from `handoff.md`, keep the per-experiment
knobs (`--out --threads --w-army --cap-potential --script-frac`) **out** of
`common.sh` so each run can override them on the command line.
