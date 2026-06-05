//! Headless self-play runner — drives a full game between N genomes to
//! completion, mirroring `training/export-golden.ts runGame` exactly (HQ
//! placement per seat, then one player-turn + `end_turn` per outer iteration,
//! until one player remains or the round cap is hit).
//!
//! This is what `cp-train` will call to evaluate genomes.

use crate::controller::NeuralAiController;
use crate::policy::XorShift32;
use crate::tiers::TierConfig;
use cp_sim::resources::BasicResource;
use cp_sim::{BuildingType, EndTurnOutcome, Game, PlayerId, TileType, UnitType};
use std::collections::HashSet;

/// Final per-player snapshot (resources + alive + tile count).
#[derive(Debug, Clone)]
pub struct PlayerResult {
    /// 1-based player number.
    pub num: i64,
    pub alive: bool,
    pub money: i64,
    pub wood: i64,
    pub stone: i64,
    pub metal: i64,
    pub tiles: i64,
}

/// Why the game ended (mirrors the golden `Result.reason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameReason {
    Domination,
    LastStanding,
    Tie,
    Timeout,
}

/// Outcome of a full self-play game.
#[derive(Debug, Clone)]
pub struct GameResult {
    /// 1-based winner player number, or `None` on tie/timeout.
    pub winner_num: Option<i64>,
    pub reason: GameReason,
    /// Final `get_rounds_played()`.
    pub rounds: i64,
    pub players: Vec<PlayerResult>,
}

/// Run one self-play game. `genomes[i]` controls seat `i` (player num `i+1`);
/// there must be exactly `genomes.len()` seats. The same `cfg` drives all seats
/// (as in training). Deterministic for fixed inputs.
pub fn run_game(
    seed: u32,
    width: i32,
    height: i32,
    genomes: &[crate::mlp::Genome],
    cfg: &TierConfig,
    round_cap: i64,
) -> GameResult {
    let n = genomes.len();
    let names: Vec<String> = (0..n).map(|i| format!("P{}", i + 1)).collect();
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let mut g = Game::new(width, height, &name_refs);
    g.generate_map(width, height, seed);

    let ctrls: Vec<NeuralAiController> = genomes
        .iter()
        .map(|gen| NeuralAiController::new(gen, *cfg))
        .collect();
    // One RNG per game, shared across turns (matches export-golden's single
    // `makeRng(seed)` handed to every controller). Only consumed at temp>0.
    let mut rng = XorShift32::new(seed);

    // --- round 0: HQ placement for every seat (advance turn after each) ------
    for _ in 0..n {
        let cur = g.current_player();
        ctrls[cur.0].place_headquarters(&mut g, cur);
        g.change_turn();
    }

    let mut winner: Option<PlayerId> = None;
    let mut tie = false;

    while g.live_players().len() > 1 && g.get_rounds_played() < round_cap {
        let cur = g.current_player();
        ctrls[cur.0].plan_turn(&mut g, cur, &mut rng, None);

        match g.end_turn() {
            EndTurnOutcome::Win(p) => {
                winner = Some(p);
                break;
            }
            EndTurnOutcome::Tie => {
                tie = true;
                break;
            }
            _ => {}
        }
    }

    // Resolve outcome (mirrors export-golden's final branch).
    let total = g.get_tile_count();
    let (winner_num, reason) = if let Some(w) = winner {
        let owned = g.get_tile_count_for_player(w);
        let r = if total > 0 && (owned * 100) / total >= 70 {
            GameReason::Domination
        } else {
            GameReason::LastStanding
        };
        (Some(g.players[w.0].player_num), r)
    } else if tie {
        (None, GameReason::Tie)
    } else if g.live_players().len() == 1 {
        let w = g.live_players()[0];
        (Some(g.players[w.0].player_num), GameReason::LastStanding)
    } else {
        (None, GameReason::Timeout)
    };

    let alive: std::collections::HashSet<PlayerId> = g.live_players().iter().copied().collect();
    let players: Vec<PlayerResult> = (0..n)
        .map(|i| {
            let pid = PlayerId(i);
            let res = &g.players[i].resources;
            PlayerResult {
                num: g.players[i].player_num,
                alive: alive.contains(&pid),
                money: res.get(BasicResource::Money).unwrap_or(0),
                wood: res.get(BasicResource::Wood).unwrap_or(0),
                stone: res.get(BasicResource::Stone).unwrap_or(0),
                metal: res.get(BasicResource::Metal).unwrap_or(0),
                tiles: g.get_tile_count_for_player(pid),
            }
        })
        .collect();

    GameResult {
        winner_num,
        reason,
        rounds: g.get_rounds_played(),
        players,
    }
}

// ===========================================================================
// Telemetry variant — a richer run used by `cp-train`'s fitness function.
//
// `run_game` (above) is left UNTOUCHED so the M5 parity gate's loop semantics
// stay identical; `run_game_telemetry` reimplements the same loop but records
// the per-seat signals the GA fitness needs. The two share their step order
// exactly (HQ placement per seat, then plan_turn + end_turn per outer
// iteration, single shared XorShift32(seed)) so a telemetry game and a plain
// game with the same inputs play out identically.
// ===========================================================================

/// Per-seat telemetry captured over a full game (one entry per seat).
#[derive(Debug, Clone)]
pub struct SeatTelemetry {
    /// 1-based player number.
    pub num: i64,
    /// Alive at game end.
    pub alive: bool,
    /// Was neutralised for a negative resource (distinct from `eliminated`).
    pub bankrupt: bool,
    /// Lost by being reduced to zero tiles (not via bankruptcy).
    pub eliminated: bool,
    /// Did this seat win (>=70% tiles domination, or last-standing)?
    pub won: bool,
    /// Round the seat won at, if it won (`get_rounds_played()` at win).
    pub win_round: Option<i64>,
    /// Round the seat was eliminated/bankrupted at, if it lost.
    pub lost_round: Option<i64>,
    /// Rounds the seat survived (lost_round if it lost, else final rounds).
    pub survived_rounds: i64,
    /// Final owned-tile fraction (owned / total).
    pub tile_frac: f64,
    /// Mean of the OTHER seats' final tile fractions.
    pub mean_others_frac: f64,
    /// Min over the whole game of min(money, wood, stone, metal) for this seat,
    /// sampled after every `end_turn`. Captures the tightest the seat ever ran.
    pub min_resource_buffer: i64,
    /// `cp_ai::metrics::net_money_per_round` at game end (0 if dead).
    pub net_money_per_round: f64,
    /// Tiles at game end with an income building staffed by >=1 worker.
    pub staffed_producer_tiles: i64,
    /// Owned tiles at game end.
    pub owned_tiles: i64,

    // --- per-round TRAJECTORY accumulators (M-reward dense signal) -----------
    /// The seat's tile fraction right after setup/HQ placement (first round).
    pub initial_tile_frac: f64,
    /// Average over completed game-rounds (while alive) of
    /// `clamp(tile_frac / 0.70, 0, 1)`.
    pub mean_domination_progress: f64,
    /// Average over rounds of `0.5*(tanh(net_money_per_round/200)+1)` (∈[0,1]).
    pub mean_net_income_norm: f64,
    /// Average over rounds of `(owned>0 ? staffed_producer_tiles/owned : 0)`.
    pub mean_productive_area: f64,
    /// Average over rounds of `clamp(min(money,wood,stone,metal)/400, 0, 1)`.
    pub mean_solvency: f64,
    /// Number of completed game-rounds the seat was counted over.
    pub rounds_counted: i64,

    // --- tactical EVENT counters (read from Game::seat_events at game end) ----
    /// Enemy Headquarters tiles this seat captured by soldier conquest.
    pub enemy_hqs_captured: i64,
    /// Enemy tiles with a real building this seat captured (HQ included).
    pub enemy_buildings_captured: i64,
    /// Enemy-owned tiles this seat took by soldier conquest.
    pub enemy_tiles_conquered: i64,
    /// Disconnected enemy tiles confiscated to this seat during HQ-cut.
    pub tiles_gained_via_cut: i64,
    /// Enemy soldiers this seat destroyed (successful assaults + successful
    /// defences). Read from `Game::seat_events` at game end.
    pub enemy_soldiers_killed: i64,

    // --- per-round RELATIVE-ADVANTAGE accumulators (v3 reward) ---------------
    // Each is the seat's mean per-round lead over the MEAN of its living
    // opponents, normalized so a "meaningful" lead is ~1.0 then clamped to
    // [-1, 1]. Averaged over the same completed game-rounds as the absolute
    // dense signals. See the *_LEAD_SCALE constants below for normalization.
    /// Mean of clamp((my_tiles - opp_mean_tiles)/TILE_LEAD_SCALE, -1, 1).
    pub mean_tile_lead: f64,
    /// Mean of clamp((my_wealth - opp_mean_wealth)/WEALTH_LEAD_SCALE, -1, 1).
    pub mean_wealth_lead: f64,
    /// Mean of clamp((my_income - opp_mean_income)/INCOME_LEAD_SCALE, -1, 1).
    pub mean_income_lead: f64,
    /// Mean of clamp((my_soldiers - opp_mean_soldiers)/MILITARY_LEAD_SCALE, -1, 1).
    pub mean_military_lead: f64,
}

// Relative-lead normalization scales: each maps a "meaningful lead" over the
// mean living opponent to ~1.0 before clamping to [-1, 1]. Observation-only;
// they affect telemetry, never game state or RNG.
/// Tile lead scale: ~1/5 of the whole map is a decisive territory lead.
fn tile_lead_scale(total_tiles: i64) -> f64 {
    (total_tiles as f64 / 5.0).max(1.0)
}
/// Money-equivalent wealth (total_wealth units); ~one extra HQ economy.
const WEALTH_LEAD_SCALE: f64 = 2000.0;
/// Net money per round; ~a few extra farms of income.
const INCOME_LEAD_SCALE: f64 = 200.0;
/// Soldier-count lead; ~one extra assault stack.
const MILITARY_LEAD_SCALE: f64 = 5.0;

/// A full game's telemetry: the same outcome `run_game` reports, plus the
/// per-seat signals the fitness function consumes.
#[derive(Debug, Clone)]
pub struct GameTelemetry {
    pub winner_num: Option<i64>,
    pub reason: GameReason,
    pub rounds: i64,
    /// Total tiles on the map (for `norm()` in fitness). Constant over a game.
    pub total_tiles: i64,
    pub seats: Vec<SeatTelemetry>,
}

/// True if `tid` (owned by anyone) holds an income-producing building staffed
/// by >=1 BasicWorker. Mirrors the positive-income branches of
/// `metrics::net_money_per_round` (Farm/Mine/Nuclear/Hydro) plus staffed
/// forest / abundant-forest harvesting.
fn is_staffed_producer(g: &Game, tid: cp_sim::TileId) -> bool {
    let workers = g
        .tile_units(tid)
        .iter()
        .filter(|&&u| g.units[u.0].kind == UnitType::BasicWorker)
        .count();
    if workers == 0 {
        return false;
    }
    match g.tiles[tid.0].building.as_ref().map(|b| b.kind) {
        Some(BuildingType::Farm)
        | Some(BuildingType::Mine)
        | Some(BuildingType::Nuclear)
        | Some(BuildingType::Hydro) => true,
        // No income building: forests still produce wood, abundant forest money.
        _ => matches!(
            g.tiles[tid.0].tile_type,
            TileType::Forest | TileType::AbundantForest
        ),
    }
}

/// Mean of a summed trajectory signal, guarded against division by zero.
fn mean_of(sum: f64, count: i64) -> f64 {
    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

/// Per-seat min over money/wood/stone/metal — the buffer used for solvency.
fn min_buffer(g: &Game, p: PlayerId) -> i64 {
    let r = &g.players[p.0].resources;
    let m = r.get(BasicResource::Money).unwrap_or(0);
    let w = r.get(BasicResource::Wood).unwrap_or(0);
    let s = r.get(BasicResource::Stone).unwrap_or(0);
    let me = r.get(BasicResource::Metal).unwrap_or(0);
    m.min(w).min(s).min(me)
}

/// Run one self-play game and return rich per-seat telemetry. Plays identically
/// to [`run_game`] for the same inputs.
pub fn run_game_telemetry(
    seed: u32,
    width: i32,
    height: i32,
    genomes: &[crate::mlp::Genome],
    cfg: &TierConfig,
    round_cap: i64,
) -> GameTelemetry {
    let n = genomes.len();
    let names: Vec<String> = (0..n).map(|i| format!("P{}", i + 1)).collect();
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let mut g = Game::new(width, height, &name_refs);
    g.generate_map(width, height, seed);

    let ctrls: Vec<NeuralAiController> = genomes
        .iter()
        .map(|gen| NeuralAiController::new(gen, *cfg))
        .collect();
    let mut rng = XorShift32::new(seed);

    // --- per-seat trackers ----------------------------------------------------
    // Min buffer seeded with the starting resources (sampled before any turn).
    let mut min_buf: Vec<i64> = (0..n).map(|i| min_buffer(&g, PlayerId(i))).collect();
    let mut lost_round: Vec<Option<i64>> = vec![None; n];
    let mut bankrupt: Vec<bool> = vec![false; n];
    let mut eliminated: Vec<bool> = vec![false; n];
    // Liveness from the previous iteration, to detect the transition out.
    let mut was_live: Vec<bool> = vec![true; n];

    // Per-round trajectory accumulators (summed over completed game-rounds
    // while the seat is alive; divided by `rounds_counted` at the end).
    let mut sum_dom: Vec<f64> = vec![0.0; n];
    let mut sum_income_norm: Vec<f64> = vec![0.0; n];
    let mut sum_prod: Vec<f64> = vec![0.0; n];
    let mut sum_solv: Vec<f64> = vec![0.0; n];
    let mut rounds_counted: Vec<i64> = vec![0; n];

    // Per-round relative-advantage accumulators (v3). Summed alongside the
    // absolute signals over the same completed game-rounds.
    let mut sum_tile_lead: Vec<f64> = vec![0.0; n];
    let mut sum_wealth_lead: Vec<f64> = vec![0.0; n];
    let mut sum_income_lead: Vec<f64> = vec![0.0; n];
    let mut sum_mil_lead: Vec<f64> = vec![0.0; n];

    // --- round 0: HQ placement for every seat --------------------------------
    for _ in 0..n {
        let cur = g.current_player();
        ctrls[cur.0].place_headquarters(&mut g, cur);
        g.change_turn();
    }

    // Initial tile fraction right after HQ placement (first round).
    let total0 = g.get_tile_count();
    let initial_tile_frac: Vec<f64> = (0..n)
        .map(|i| {
            if total0 > 0 {
                g.get_tile_count_for_player(PlayerId(i)) as f64 / total0 as f64
            } else {
                0.0
            }
        })
        .collect();

    // Tracks the round counter from the previous iteration so we snapshot
    // trajectory signals exactly once per COMPLETED game-round (when the
    // counter advances after a full turn cycle).
    let mut prev_round = g.get_rounds_played();

    let mut winner: Option<PlayerId> = None;
    let mut tie = false;

    while g.live_players().len() > 1 && g.get_rounds_played() < round_cap {
        let cur = g.current_player();
        ctrls[cur.0].plan_turn(&mut g, cur, &mut rng, None);

        let outcome = g.end_turn();

        // Sample the min buffer for all still-live seats after production.
        let live_now: HashSet<PlayerId> = g.live_players().iter().copied().collect();
        for i in 0..n {
            if live_now.contains(&PlayerId(i)) {
                let b = min_buffer(&g, PlayerId(i));
                if b < min_buf[i] {
                    min_buf[i] = b;
                }
            }
        }

        // Detect seats that just dropped out of the live set; classify why.
        let round = g.get_rounds_played();
        for i in 0..n {
            if was_live[i] && !live_now.contains(&PlayerId(i)) {
                lost_round[i] = Some(round);
                // A bankrupt loss leaves a negative resource on the player;
                // a territorial loss empties its objects (no tiles) instead.
                let has_negative = g.players[i].resources.iter().any(|(_, v)| v < 0);
                if has_negative {
                    bankrupt[i] = true;
                } else {
                    eliminated[i] = true;
                }
            }
            was_live[i] = live_now.contains(&PlayerId(i));
        }

        // Once per COMPLETED game-round (the round counter advanced this
        // iteration), snapshot every still-alive seat's trajectory signals.
        if round != prev_round {
            prev_round = round;
            let total_now = g.get_tile_count();

            // Snapshot per-seat RELATIVE stats for every living seat first, so
            // each seat can compare against the mean of its living opponents.
            // Reading these (tiles, wealth, income, soldiers) mutates nothing.
            let live_ids: Vec<usize> = (0..n)
                .filter(|&i| live_now.contains(&PlayerId(i)))
                .collect();
            let stat_tiles: Vec<f64> = (0..n)
                .map(|i| g.get_tile_count_for_player(PlayerId(i)) as f64)
                .collect();
            let stat_wealth: Vec<f64> = (0..n)
                .map(|i| crate::metrics::total_wealth(&g, PlayerId(i)))
                .collect();
            let stat_income: Vec<f64> = (0..n)
                .map(|i| crate::metrics::net_money_per_round(&g, PlayerId(i)))
                .collect();
            let stat_soldiers: Vec<f64> = (0..n)
                .map(|i| g.current_soldier_amount(PlayerId(i)) as f64)
                .collect();
            let tls = tile_lead_scale(total_now);

            for i in 0..n {
                let pid = PlayerId(i);
                if !live_now.contains(&pid) {
                    continue;
                }

                // Mean of this seat's LIVING opponents for each relative stat.
                let opp: Vec<usize> = live_ids.iter().copied().filter(|&j| j != i).collect();
                if !opp.is_empty() {
                    let k = opp.len() as f64;
                    let mean = |v: &[f64]| -> f64 { opp.iter().map(|&j| v[j]).sum::<f64>() / k };
                    let lead = |mine: f64, om: f64, scale: f64| -> f64 {
                        ((mine - om) / scale).clamp(-1.0, 1.0)
                    };
                    sum_tile_lead[i] += lead(stat_tiles[i], mean(&stat_tiles), tls);
                    sum_wealth_lead[i] +=
                        lead(stat_wealth[i], mean(&stat_wealth), WEALTH_LEAD_SCALE);
                    sum_income_lead[i] +=
                        lead(stat_income[i], mean(&stat_income), INCOME_LEAD_SCALE);
                    sum_mil_lead[i] +=
                        lead(stat_soldiers[i], mean(&stat_soldiers), MILITARY_LEAD_SCALE);
                }

                let frac = if total_now > 0 {
                    g.get_tile_count_for_player(pid) as f64 / total_now as f64
                } else {
                    0.0
                };
                sum_dom[i] += (frac / 0.70).clamp(0.0, 1.0);

                let nmpr = crate::metrics::net_money_per_round(&g, pid);
                sum_income_norm[i] += 0.5 * ((nmpr / 200.0).tanh() + 1.0);

                let owned = g.owned_tiles(pid);
                let owned_count = owned.len() as i64;
                let staffed = owned
                    .iter()
                    .filter(|&&tid| is_staffed_producer(&g, tid))
                    .count() as i64;
                sum_prod[i] += if owned_count > 0 {
                    staffed as f64 / owned_count as f64
                } else {
                    0.0
                };

                sum_solv[i] += (min_buffer(&g, pid) as f64 / 400.0).clamp(0.0, 1.0);
                rounds_counted[i] += 1;
            }
        }

        match outcome {
            EndTurnOutcome::Win(p) => {
                winner = Some(p);
                break;
            }
            EndTurnOutcome::Tie => {
                tie = true;
                break;
            }
            _ => {}
        }
    }

    let final_round = g.get_rounds_played();
    let total = g.get_tile_count();

    let (winner_num, reason) = if let Some(w) = winner {
        let owned = g.get_tile_count_for_player(w);
        let r = if total > 0 && (owned * 100) / total >= 70 {
            GameReason::Domination
        } else {
            GameReason::LastStanding
        };
        (Some(g.players[w.0].player_num), r)
    } else if tie {
        (None, GameReason::Tie)
    } else if g.live_players().len() == 1 {
        let w = g.live_players()[0];
        (Some(g.players[w.0].player_num), GameReason::LastStanding)
    } else {
        (None, GameReason::Timeout)
    };

    let alive: HashSet<PlayerId> = g.live_players().iter().copied().collect();

    // Final tile fractions per seat (for rank scoring and tile_frac).
    let fracs: Vec<f64> = (0..n)
        .map(|i| {
            if total > 0 {
                g.get_tile_count_for_player(PlayerId(i)) as f64 / total as f64
            } else {
                0.0
            }
        })
        .collect();

    let seats: Vec<SeatTelemetry> = (0..n)
        .map(|i| {
            let pid = PlayerId(i);
            let num = g.players[i].player_num;
            let is_alive = alive.contains(&pid);
            let won = winner_num == Some(num);
            let win_round = if won { Some(final_round) } else { None };
            let survived = lost_round[i].unwrap_or(final_round);

            let mean_others = if n > 1 {
                let sum: f64 = (0..n).filter(|&j| j != i).map(|j| fracs[j]).sum();
                sum / (n as f64 - 1.0)
            } else {
                0.0
            };

            let owned: Vec<cp_sim::TileId> = g.owned_tiles(pid);
            let owned_count = owned.len() as i64;
            let staffed = owned
                .iter()
                .filter(|&&tid| is_staffed_producer(&g, tid))
                .count() as i64;
            let nmpr = if is_alive {
                crate::metrics::net_money_per_round(&g, pid)
            } else {
                0.0
            };

            let ev = g.seat_events(pid);

            SeatTelemetry {
                num,
                alive: is_alive,
                bankrupt: bankrupt[i],
                eliminated: eliminated[i],
                won,
                win_round,
                lost_round: lost_round[i],
                survived_rounds: survived,
                tile_frac: fracs[i],
                mean_others_frac: mean_others,
                min_resource_buffer: min_buf[i],
                net_money_per_round: nmpr,
                staffed_producer_tiles: staffed,
                owned_tiles: owned_count,
                initial_tile_frac: initial_tile_frac[i],
                mean_domination_progress: mean_of(sum_dom[i], rounds_counted[i]),
                mean_net_income_norm: mean_of(sum_income_norm[i], rounds_counted[i]),
                mean_productive_area: mean_of(sum_prod[i], rounds_counted[i]),
                mean_solvency: mean_of(sum_solv[i], rounds_counted[i]),
                rounds_counted: rounds_counted[i],
                enemy_hqs_captured: ev.enemy_hqs_captured,
                enemy_buildings_captured: ev.enemy_buildings_captured,
                enemy_tiles_conquered: ev.enemy_tiles_conquered,
                tiles_gained_via_cut: ev.tiles_gained_via_cut,
                enemy_soldiers_killed: ev.enemy_soldiers_killed,
                mean_tile_lead: mean_of(sum_tile_lead[i], rounds_counted[i]),
                mean_wealth_lead: mean_of(sum_wealth_lead[i], rounds_counted[i]),
                mean_income_lead: mean_of(sum_income_lead[i], rounds_counted[i]),
                mean_military_lead: mean_of(sum_mil_lead[i], rounds_counted[i]),
            }
        })
        .collect();

    GameTelemetry {
        winner_num,
        reason,
        rounds: final_round,
        total_tiles: total,
        seats,
    }
}
