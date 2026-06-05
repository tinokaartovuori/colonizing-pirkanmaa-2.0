# Reward configs (`fitness_v2`, v3 shaping)

JSON files here tune the GA fitness without recompiling. Pass one with:

```bash
cargo run -p cp-train --bin train --release -- --reward rust-trainer/rewards/v3-default.json
```

Omitting `--reward` uses the **built-in default**, which is byte-for-value
identical to `v3-default.json`. Every field is optional in a JSON file
(`#[serde(default)]`); omitted fields fall back to the built-in default.
Unknown fields are rejected.

## What changed in v3

Trained champions were turtling: in 2p games the champion held only ~15% of
the map (most tiles left neutral) and games ran to the round cap. A
pure-RELATIVE reward alone does not fix this — mutual turtling makes both seats'
leads ~0, so it scores neutral. v3 therefore combines:

- **ABSOLUTE growth** (anti-turtle): grow toward the 70% domination win and grow
  the economy — `w_dom / w_econ / w_prod / w_solv`.
- **RELATIVE advantage** ("more than the opponent is positive", signed): per-round
  lead in tiles, total wealth, income, and military vs the mean living opponent —
  `w_tile_lead / w_wealth_lead / w_income_lead / w_mil_lead`.

Both blocks share the anneal factor `a_t`, whose floor is raised to **0.5** (was
0.3) so the growth + lead signals stay strong late in training. The old
standalone `w_rank` term is dropped (default `0.0`) — the relative leads cover
"ahead of opponent" more richly — but it remains configurable.

## How the fitness is composed

For the evaluated seat `s` of one game (`cap` = round cap, `T` = total tiles):

```
fitness = terminal + dense + tactical + tile_loss + rank
dense   = a_t * (abs_dense + rel_dense)
```

`dense + tactical + tile_loss (+ rank)` apply to ALL outcomes; `terminal` is the
one-shot outcome term and the win case stays dominant.

### terminal (one-shot outcome)
```
bankrupt   -> bankrupt_pen + survive_credit * (survived_rounds / cap)
eliminated -> loss_pen     + survive_credit * (survived_rounds / cap)
won        -> win_base     + win_speed * (1 - win_round / cap)
timeout    -> timeout_base
```

### dense anneal
```
a_t = dense_floor + (1 - dense_floor) * max(0, 1 - gen / (dense_anneal_frac * total_gens))
```
`a_t` decays linearly from 1.0 toward `dense_floor` (v3 default 0.5) over the
first `dense_anneal_frac` of training, then holds.

### abs_dense (ABSOLUTE growth, anti-turtle)
```
abs_dense = w_dom  * mean_domination_progress   // clamp(tile_frac/0.70, 0, 1)
          + w_econ * mean_net_income_norm        // 0.5*(tanh(net/200)+1)
          + w_prod * mean_productive_area         // staffed_producers / owned
          + w_solv * mean_solvency                // clamp(min(m,w,s,me)/400, 0, 1)
```

### rel_dense (RELATIVE advantage, signed)
Each lead is the seat's per-round value minus the MEAN of its LIVING opponents,
normalized by a scale that maps a "meaningful lead" to ~1.0, clamped to [-1, 1],
then averaged over completed game-rounds:
```
rel_dense = w_tile_lead   * mean_tile_lead       // (my_tiles  - opp_mean)/tile_lead_scale
          + w_wealth_lead * mean_wealth_lead      // (my_wealth - opp_mean)/2000
          + w_income_lead * mean_income_lead      // (my_income - opp_mean)/200
          + w_mil_lead    * mean_military_lead    // (my_sol    - opp_mean)/5
```
Lead scales (constants in `cp-ai/run.rs`):
| scale | value | rationale |
|---|---|---|
| `tile_lead_scale` | `max(1, T/5)` | ~1/5 of the whole map is a decisive territory lead |
| `wealth_lead_scale` | `2000` | money-equivalent; ~one extra HQ economy |
| `income_lead_scale` | `200` | net money/round; a few extra farms of income |
| `military_lead_scale` | `5` | ~one extra assault stack |

**`total_wealth(g, p)`** (money-equivalent, telemetry only) sums exactly:
1. liquid resources `money + wood + stone + metal` (raw counts);
2. for each building the player owns (one per owned tile with a building, HQ /
   Mikontalo included): the **money component** of `build_cost()` as a positive
   amount (HQ/Mikontalo contribute 0 — empty build cost);
3. for each owned unit (worker/expert/soldier): the **money component** of
   `cost()` as a positive amount.
Only the money component is summed — metal in costs is intentionally ignored.

### tactical (aggression that wins games; NOT annealed by default)
```
norm(x)  = x / max(1, T)
tactical = tactical_floor * ( w_hq       * enemy_hqs_captured            // raw count
                            + w_cut      * norm(tiles_gained_via_cut)
                            + w_building * norm(enemy_buildings_captured)
                            + w_conquer  * norm(enemy_tiles_conquered)
                            + w_kill     * (enemy_soldiers_killed / max(1, kill_scale)) )
```
With `tactical_floor = 1.0` the block is never annealed. The event counts come
from `cp_sim::Game::seat_events` — observation-only counters incremented at the
exact owner-change / unit-removal points in `end_turn` (soldier conquest +
HQ-connectivity cut). They do not affect game state, decisions, or RNG, so the
parity gate stays green. `enemy_soldiers_killed` counts defender soldiers
destroyed on a successful assault and attacker soldiers destroyed on a
successful defence, attributed to the destroyer.

### tile_loss + legacy rank
```
tile_loss = w_tile_loss * min(0, tile_frac - initial_tile_frac)
rank      = w_rank * clamp(2*(tile_frac - mean_others)/(tile_frac + mean_others + 1e-9), -1, 1)
```
`tile_loss` is <= 0, penalizing a seat that sheds territory below its starting
footprint. `rank` is the legacy v2 term, default `0.0` (skipped entirely when 0).

## Fields

| field | default | meaning |
|---|---|---|
| `bankrupt_pen` | -1.0 | penalty when the seat went bankrupt (negative resource) |
| `loss_pen` | -0.8 | penalty when the seat was eliminated (reduced to zero tiles) |
| `survive_credit` | 0.15 | credit per fraction of `cap` survived (bankrupt/eliminated) |
| `win_base` | 1.0 | base reward for winning |
| `win_speed` | 0.5 | extra reward scaled by win speed (`1 - win_round/cap`) |
| `timeout_base` | 0.0 | terminal reward on a timeout / no-winner outcome |
| `dense_floor` | 0.5 | floor the dense weight `a_t` anneals down to |
| `dense_anneal_frac` | 0.6 | fraction of total gens over which `a_t` decays to the floor |
| `w_dom` | 0.40 | weight on `mean_domination_progress` |
| `w_econ` | 0.10 | weight on `mean_net_income_norm` |
| `w_prod` | 0.08 | weight on `mean_productive_area` |
| `w_solv` | 0.04 | weight on `mean_solvency` |
| `w_tile_lead` | 0.35 | weight on `mean_tile_lead` (signed, [-1,1]) |
| `w_wealth_lead` | 0.20 | weight on `mean_wealth_lead` (signed, [-1,1]) |
| `w_income_lead` | 0.15 | weight on `mean_income_lead` (signed, [-1,1]) |
| `w_mil_lead` | 0.10 | weight on `mean_military_lead` (signed, [-1,1]) |
| `tactical_floor` | 1.0 | multiplier on the tactical block (1.0 = never annealed) |
| `w_hq` | 0.30 | weight per enemy HQ captured (raw count) |
| `w_cut` | 0.50 | weight on `tiles_gained_via_cut / T` |
| `w_building` | 0.40 | weight on `enemy_buildings_captured / T` |
| `w_conquer` | 0.20 | weight on `enemy_tiles_conquered / T` |
| `w_kill` | 0.25 | weight on `enemy_soldiers_killed / max(1, kill_scale)` |
| `kill_scale` | 10.0 | normalizer for `enemy_soldiers_killed` (≈ one decisive war) |
| `w_tile_loss` | 0.30 | weight on tile loss below the seat's initial footprint |
| `w_rank` | 0.0 | legacy rank vs mean of other seats (off by default) |

## Presets

- **`v3-default.json`** — all v3 defaults above (== the built-in default).
- **`v2-default.json`** — the previous shaping (`dense_floor=0.3`, `w_rank=0.2`,
  higher `w_dom/w_econ`, no relative-lead weights). Loading it under the v3 code
  picks up the new relative/kill weights at their v3 defaults via `serde(default)`
  for the fields it omits; kept as a comparison control for the v2 absolute-only
  shaping.
- **`v1-baseline.json`** — control approximating the OLD `fitness_for_game`:
  dense + tactical weights all 0, `w_rank = 0.3`, `w_tile_loss = 0`, both loss
  penalties = -1.0. NOTE: the old fitness applied rank ONLY on a timeout, whereas
  this fitness adds `rank` to all outcomes; so for won/lost games this baseline
  carries a small extra `0.3 * rank` term the original did not.
