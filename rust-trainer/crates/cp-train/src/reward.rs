//! Configurable reward system (`fitness_v2`, now v3 shaping) for the GA
//! self-play loop.
//!
//! Every constant is a field of [`RewardConfig`] so the whole shaping can be
//! retuned from a JSON file with `--reward <FILE>` — NO recompile needed. This
//! enables an empirical reward-search loop.
//!
//! **v3** combines ABSOLUTE growth signals (anti-turtle: expand toward the 70%
//! win, grow the economy) with RELATIVE-advantage signals ("having more than
//! the opponent is positive": tiles, total wealth, income, military), plus
//! tactical events including enemy soldiers killed. The diagnosis driving this:
//! trained champions barely expand and turtle to timeouts; pure-relative reward
//! alone would let MUTUAL turtling score ~0, so we need absolute + relative
//! together (both annealed by `a_t`, which now holds at a strong floor of 0.5).
//!
//! Parts:
//! - **terminal** — one-shot outcome reward (bankrupt / eliminated / won /
//!   timeout). The win term stays dominant.
//! - **abs_dense** — absolute economic growth from per-round trajectory means
//!   (domination progress, income, productive area, solvency).
//! - **rel_dense** — signed per-round lead over the mean living opponent
//!   (tiles, wealth, income, military). "More than enemy = positive."
//! - **tactical** — aggression that wins games (capturing HQs, cutting enemy
//!   territory, taking enemy buildings/tiles, killing enemy soldiers). NOT
//!   annealed by default (`tactical_floor = 1.0`).
//! - **tile_loss** — penalty for shedding territory below the seat's initial
//!   footprint.
//!
//! `dense (= a_t*(abs_dense+rel_dense)) + tactical + tile_loss` apply to ALL
//! outcomes, but the win terminal stays dominant.

use cp_ai::SeatTelemetry;
use serde::Deserialize;

/// Every constant in [`fitness_v2`]. `#[serde(default)]` per field means a JSON
/// reward file may specify any subset; omitted fields fall back to the built-in
/// default (which equals `RewardConfig::default()` == `rewards/v3-default.json`).
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RewardConfig {
    // --- terminal (one-shot outcome) ---
    /// Penalty applied when the seat went bankrupt (negative resource).
    pub bankrupt_pen: f64,
    /// Penalty applied when the seat was eliminated (reduced to zero tiles).
    pub loss_pen: f64,
    /// Credit per fraction of the round-cap survived (bankrupt/eliminated).
    pub survive_credit: f64,
    /// Base reward for winning the game.
    pub win_base: f64,
    /// Extra reward scaled by how fast the win came (`1 - win_round/cap`).
    pub win_speed: f64,
    /// Terminal reward for a timeout / no-winner outcome.
    pub timeout_base: f64,

    // --- dense anneal (shared by abs_dense + rel_dense) ---
    /// Floor the dense weight anneals down to (never below this fraction).
    /// v3 keeps this STRONG at 0.5 so growth/lead signals never fully vanish.
    pub dense_floor: f64,
    /// Fraction of total generations over which the dense weight decays to
    /// `dense_floor` (linearly), then holds.
    pub dense_anneal_frac: f64,

    // --- ABSOLUTE dense trajectory (anti-turtle) ---
    /// Weight on mean domination progress (`clamp(tile_frac/0.70,0,1)`).
    pub w_dom: f64,
    /// Weight on mean normalized net income (`0.5*(tanh(net/200)+1)`).
    pub w_econ: f64,
    /// Weight on mean productive area (`staffed_producers/owned`).
    pub w_prod: f64,
    /// Weight on mean solvency (`clamp(min(m,w,s,me)/400,0,1)`).
    pub w_solv: f64,

    // --- RELATIVE dense ("more than enemy = positive", signed) ---
    /// Weight on mean tile lead over the mean living opponent (∈[-1,1]).
    pub w_tile_lead: f64,
    /// Weight on mean total-wealth lead (∈[-1,1]).
    pub w_wealth_lead: f64,
    /// Weight on mean net-income lead (∈[-1,1]).
    pub w_income_lead: f64,
    /// Weight on mean soldier-count (military) lead (∈[-1,1]).
    pub w_mil_lead: f64,

    // --- tactical events (not annealed by default) ---
    /// Floor multiplier on the tactical block. 1.0 = never annealed.
    pub tactical_floor: f64,
    /// Weight per enemy HQ captured (raw count, not normalized).
    pub w_hq: f64,
    /// Weight on `tiles_gained_via_cut / total_tiles`.
    pub w_cut: f64,
    /// Weight on `enemy_buildings_captured / total_tiles`.
    pub w_building: f64,
    /// Weight on `enemy_tiles_conquered / total_tiles`.
    pub w_conquer: f64,
    /// Weight on `enemy_soldiers_killed / max(1, kill_scale)`.
    pub w_kill: f64,
    /// Normalizer for `enemy_soldiers_killed` (≈ one decisive war's worth).
    pub kill_scale: f64,

    // --- passivity penalty + legacy rank ---
    /// Weight on the (negative) tile loss below the seat's initial footprint.
    pub w_tile_loss: f64,
    /// LEGACY relative-rank term (v2). The v3 relative leads cover "ahead of
    /// opponent" more richly, so this defaults to 0.0. Kept configurable so an
    /// old reward file can still re-enable it.
    pub w_rank: f64,
}

impl Default for RewardConfig {
    fn default() -> Self {
        RewardConfig {
            // terminal
            bankrupt_pen: -1.0,
            loss_pen: -0.8,
            survive_credit: 0.15,
            win_base: 1.0,
            win_speed: 0.5,
            timeout_base: 0.0,
            // dense anneal (strong floor)
            dense_floor: 0.5,
            dense_anneal_frac: 0.6,
            // absolute dense
            w_dom: 0.40,
            w_econ: 0.10,
            w_prod: 0.08,
            w_solv: 0.04,
            // relative dense
            w_tile_lead: 0.35,
            w_wealth_lead: 0.20,
            w_income_lead: 0.15,
            w_mil_lead: 0.10,
            // tactical
            tactical_floor: 1.0,
            w_hq: 0.30,
            w_cut: 0.50,
            w_building: 0.40,
            w_conquer: 0.20,
            w_kill: 0.25,
            kill_scale: 10.0,
            // passivity + legacy
            w_tile_loss: 0.30,
            w_rank: 0.0,
        }
    }
}

impl RewardConfig {
    /// Load a reward config from a JSON file. Omitted fields use the built-in
    /// default; unknown fields are rejected.
    pub fn from_file(path: &std::path::Path) -> Result<Self, String> {
        let s = std::fs::read_to_string(path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        serde_json::from_str(&s).map_err(|e| format!("parsing {}: {e}", path.display()))
    }
}

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// Configurable v3 fitness for the evaluated seat `s` of one game.
///
/// - `cfg` — every reward constant.
/// - `gen` / `total_gens` — drive the dense-weight anneal `a_t`.
/// - `round_cap` — the game's round cap (for survival/win-speed normalization).
/// - `total_tiles` — total tiles on the map (for `norm()` of event counts).
pub fn fitness_v2(
    s: &SeatTelemetry,
    cfg: &RewardConfig,
    gen: usize,
    total_gens: usize,
    round_cap: i64,
    total_tiles: i64,
) -> f64 {
    let cap = round_cap as f64;

    // --- terminal (one-shot, dominant) ---
    let terminal = if s.bankrupt {
        cfg.bankrupt_pen + cfg.survive_credit * (s.survived_rounds as f64 / cap)
    } else if s.eliminated {
        cfg.loss_pen + cfg.survive_credit * (s.survived_rounds as f64 / cap)
    } else if s.won {
        let win_round = s.win_round.unwrap_or(s.survived_rounds) as f64;
        cfg.win_base + cfg.win_speed * (1.0 - win_round / cap)
    } else {
        cfg.timeout_base
    };

    // --- anneal factor a_t (shared by absolute + relative dense) ---
    let anneal_span = (cfg.dense_anneal_frac * total_gens as f64).max(1e-9);
    let a_t = cfg.dense_floor
        + (1.0 - cfg.dense_floor) * (1.0 - gen as f64 / anneal_span).max(0.0);

    // --- ABSOLUTE dense (anti-turtle: grow toward the win + economy) ---
    let abs_dense = cfg.w_dom * s.mean_domination_progress
        + cfg.w_econ * s.mean_net_income_norm
        + cfg.w_prod * s.mean_productive_area
        + cfg.w_solv * s.mean_solvency;

    // --- RELATIVE dense (signed: "more than enemy = positive") ---
    let rel_dense = cfg.w_tile_lead * s.mean_tile_lead
        + cfg.w_wealth_lead * s.mean_wealth_lead
        + cfg.w_income_lead * s.mean_income_lead
        + cfg.w_mil_lead * s.mean_military_lead;

    let dense = a_t * (abs_dense + rel_dense);

    // --- tactical events (not annealed by default) ---
    let denom = total_tiles.max(1) as f64;
    let norm = |x: i64| -> f64 { x as f64 / denom };
    let tactical = cfg.tactical_floor
        * (cfg.w_hq * s.enemy_hqs_captured as f64
            + cfg.w_cut * norm(s.tiles_gained_via_cut)
            + cfg.w_building * norm(s.enemy_buildings_captured)
            + cfg.w_conquer * norm(s.enemy_tiles_conquered)
            + cfg.w_kill * (s.enemy_soldiers_killed as f64 / cfg.kill_scale.max(1.0)));

    // --- passivity penalty + legacy rank ---
    let tile_loss = cfg.w_tile_loss * (s.tile_frac - s.initial_tile_frac).min(0.0);
    let rank = if cfg.w_rank != 0.0 {
        let my = s.tile_frac;
        let others = s.mean_others_frac;
        cfg.w_rank * clamp(2.0 * (my - others) / (my + others + 1e-9), -1.0, 1.0)
    } else {
        0.0
    };

    terminal + dense + tactical + tile_loss + rank
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The built-in default must equal `rewards/v3-default.json` field-for-field,
    /// so `--reward v3-default.json` and no `--reward` produce identical fitness.
    #[test]
    fn builtin_default_matches_v3_default_json() {
        // crates/cp-train -> crates -> rust-trainer
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("rewards/v3-default.json");
        let from_file = RewardConfig::from_file(&path).expect("load v3-default.json");
        let def = RewardConfig::default();

        macro_rules! eq {
            ($f:ident) => {
                assert_eq!(from_file.$f, def.$f, concat!("field ", stringify!($f)));
            };
        }
        eq!(bankrupt_pen);
        eq!(loss_pen);
        eq!(survive_credit);
        eq!(win_base);
        eq!(win_speed);
        eq!(timeout_base);
        eq!(dense_floor);
        eq!(dense_anneal_frac);
        eq!(w_dom);
        eq!(w_econ);
        eq!(w_prod);
        eq!(w_solv);
        eq!(w_tile_lead);
        eq!(w_wealth_lead);
        eq!(w_income_lead);
        eq!(w_mil_lead);
        eq!(tactical_floor);
        eq!(w_hq);
        eq!(w_cut);
        eq!(w_building);
        eq!(w_conquer);
        eq!(w_kill);
        eq!(kill_scale);
        eq!(w_tile_loss);
        eq!(w_rank);
    }
}
