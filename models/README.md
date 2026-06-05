# Models — registry & management

Single source of truth for every trained model (AlphaZero nets, heuristic hard-bot
parameter sets, neuroevolution genomes). Replaces the old scattered storage
(`rust-trainer/checkpoints*`, `training/checkpoints`, ad-hoc `champion.json`s — all
archived under `archive/`).

## Layout

```
models/
  registry.jsonl              # index — one JSON line per model (quick listing)
  CHAMPION.json               # pointers: { champions: {"<arc>/<type>": id}, deployed: {...} }
  manage.ts                   # the CLI (run via `npm run models -- <cmd>`)
  <arc>/<type>/<id>/
    weights.json              # the model itself (genome / net / params)
    manifest.json             # full metadata (see below)
    bench.json                # optional — detailed benchmark output
```

## Naming & versioning

- **id = `<arc>-<type>-<NNN>`** — e.g. `sd-az-001`, `sd-hardbot-001`.
  - **arc** = game-version code. Bump it whenever the GAME's rules change so models
    from different game versions never get compared as if equivalent. Current arc:
    **`sd`** = the Strange-Device game version (see `/STRANGE-DEVICE-DESIGN.md`).
    The pre-Strange-Device models are archived, not in this registry.
  - **type** = `az` (AlphaZero net) · `hardbot` (heuristic `AiParams` set) · `ga`
    (neuroevolution genome).
  - **NNN** = zero-padded incremental, per `(arc, type)`. Assigned by `manage.ts`.
- **Versioning is by id + git_commit**, not by renaming. Once registered, a model's
  files never change; "champion" / "deployed" are *pointers* (in `CHAMPION.json`),
  so lineage stays stable and reproducible.

## manifest.json fields

| field | meaning |
|---|---|
| `id`, `arc`, `type`, `version` | identity |
| `created_utc` | ISO timestamp |
| `git_commit` | repo commit at registration (exact reproducibility) |
| `parent` | id of the warm-start parent, or null (lineage) |
| `training_config` | sims / iters / reward / opponent-mix / map size / seed / … |
| `benchmarks` | `{ vs_hard: { winrate, ci, n, outcome_breakdown, seat_split }, … }` — see `/STRANGE-DEVICE-DESIGN.md` §10 for the metric taxonomy |
| `status` | `experimental` \| `champion` \| `deployed` \| `archived` |
| `notes` | free text |

## CLI (`npm run models -- <cmd>`)

```bash
npm run models -- list                       # table of all registered models
npm run models -- show sd-az-001             # print a manifest
npm run models -- register rust-trainer/checkpoints-X/champion.json \
                  --arc sd --type az --parent sd-az-000 --notes "kl-anchor run"
npm run models -- promote sd-az-003 --deployed   # set champion (+ deploy pointer)
```

`register` copies the weights in, assigns the next id, stamps the git commit, and
appends the registry line. Fill `training_config` + `benchmarks` in the manifest
afterwards — or (preferred, a follow-up) have the trainer write them automatically
at the end of a run so registration is one step.

## Conventions to keep (also in CLAUDE.md)

- Every model that's worth keeping gets **registered** here — no stray
  `champion.json`s outside `models/`.
- The live game's deployed model is whatever `CHAMPION.json.deployed["weights.ts"]`
  points at; deploying = writing that model to `src/ai/nn/weights.ts`.
- Bump the **arc** code on any game-rules change; never benchmark across arcs.
