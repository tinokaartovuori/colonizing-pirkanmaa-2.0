//! Exp-I spatial policy input: per-candidate SPATIAL features appended to the
//! standard policy input, so target selection can SEE the board's graph structure
//! — above all the offensive cut-value ("how much of the enemy disconnects from
//! its HQ if I take this tile"), which makes the win condition visible.
//!
//! ADDITIVE / AZ-ONLY: the parity-locked path (`policy::policy_input`,
//! `candidates::enumerate`/`local_vec`, the shipped 63-dim `weights.ts`) is
//! UNTOUCHED. These functions are used only by the AlphaZero spatial-policy
//! training path, so parity stays 8/8 and the live game keeps running on the old
//! net until a spatial champion is trained and explicitly deployed.

use cp_sim::{Game, PlayerId, TileId};

use crate::candidates::{Action, Candidate};
use crate::policy::{policy_input, POLICY_INPUT_DIM};
use crate::spatial;

/// Number of per-candidate spatial features appended after the standard input.
pub const SPATIAL_LOCAL_DIM: usize = 6;
/// Spatial policy input dim = standard policy input + spatial block.
pub const POLICY_INPUT_DIM_SPATIAL: usize = POLICY_INPUT_DIM + SPATIAL_LOCAL_DIM;
/// Default spatial-policy architecture. Hidden layout [24,16,1] MATCHES the
/// shipped 63-dim policy ([63,24,16,1]) so `warmstart_spatial` can transplant the
/// trained champion's weights exactly (only the input layer widens 63→69).
pub const DEFAULT_ARCH_SPATIAL: [usize; 4] = [POLICY_INPUT_DIM_SPATIAL, 24, 16, 1];

/// Warm-start a spatial policy from a trained 63-dim champion: the first 63
/// input weights of layer 0 keep the champion's values, the 6 new SPATIAL input
/// weights start at 0, and every other layer is copied verbatim. The result
/// PREDICTS IDENTICALLY to `base` at init (the 6 spatial features contribute 0),
/// so training begins at the champion's strength (~33%) and only has to LEARN to
/// use the spatial features — instead of relearning the whole game from random
/// (which cold-start proved gets stuck ~5%). Requires `base.arch == [63,24,16,1]`.
pub fn warmstart_spatial(base: &crate::mlp::Genome) -> crate::mlp::Genome {
    let in_old = base.arch[0]; // POLICY_INPUT_DIM (64 in the Strange-Device arc)
    let in_new = POLICY_INPUT_DIM_SPATIAL; // POLICY_INPUT_DIM + 6 (70)
    let h1 = base.arch[1]; // 24
    assert!(in_new >= in_old, "spatial input must be >= base input");
    let mut new_arch = vec![in_new];
    new_arch.extend_from_slice(&base.arch[1..]);
    let mut p: Vec<f64> = Vec::new();
    // Layer 0 weights (row-major by output unit j): first `in_old` kept, rest 0.
    for j in 0..h1 {
        for i in 0..in_new {
            p.push(if i < in_old { base.params[j * in_old + i] } else { 0.0 });
        }
    }
    // Layer 0 biases (unchanged).
    for j in 0..h1 {
        p.push(base.params[h1 * in_old + j]);
    }
    // Every subsequent layer: verbatim copy (same shapes).
    let first_layer_old = h1 * in_old + h1;
    p.extend_from_slice(&base.params[first_layer_old..]);
    crate::mlp::Genome { arch: new_arch, params: p }
}

/// The target tile an action operates on (None for Pass).
pub fn candidate_target_tile(c: &Candidate) -> Option<TileId> {
    match &c.action {
        Action::Build(_, t) => Some(*t),
        Action::Expand { tile, .. } => Some(*tile),
        Action::BuyUnit(_, t) => Some(*t),
        Action::Attack { tile, .. } => Some(*tile),
        Action::March { to, .. } => Some(*to),
        Action::Pass => None,
    }
}

/// Is `t` a live enemy player's (un-conquered) HQ tile?
fn is_enemy_hq(g: &Game, p: PlayerId, t: TileId) -> bool {
    g.live_players()
        .iter()
        .filter(|&&e| e != p)
        .any(|&e| g.get_hq_tile(e) == Some(t))
}

/// The 6-dim spatial feature block for one candidate (all clamped to [0,1]):
///   0 offensive_cut_value  — enemy fraction disconnected if I take the target (CRUX)
///   1 enemy_hq_proximity   — 1 = adjacent to an enemy HQ, 0 = far / none
///   2 is_enemy_hq          — 1 if the target IS a live enemy HQ
///   3 own_cut_vulnerability— if target is mine, how exposed it is to being cut
///   4 enemy_neighbor_frac  — fraction of the target's 4-neighbours owned by an enemy
///   5 target_owner_is_enemy— 1 if an enemy owns the target (an assault), else 0
pub fn candidate_spatial_features(g: &Game, p: PlayerId, c: &Candidate) -> Vec<f64> {
    let t = match candidate_target_tile(c) {
        Some(t) => t,
        None => return vec![0.0; SPATIAL_LOCAL_DIM],
    };
    let cut = spatial::offensive_cut_value(g, p, t);
    let dist = spatial::dist_to_enemy_hq(g, p, t) as f64;
    let enemy_hq_proximity = 1.0 - (dist / 20.0).min(1.0);
    let is_hq = if is_enemy_hq(g, p, t) { 1.0 } else { 0.0 };
    let owner = g.get_tiles().get(t.0).and_then(|tl| tl.owner);
    let own_vuln = if owner == Some(p) { spatial::cut_vulnerability(g, p, t) } else { 0.0 };
    let live = g.live_players().to_vec();
    let enemy_nbrs = g
        .neighbour_four_tiles(t)
        .into_iter()
        .filter(|n| matches!(g.get_tiles().get(n.0).and_then(|tl| tl.owner), Some(o) if o != p && live.contains(&o)))
        .count() as f64;
    let enemy_neighbor_frac = enemy_nbrs / 4.0;
    let owner_is_enemy = match owner {
        Some(o) if o != p && live.contains(&o) => 1.0,
        _ => 0.0,
    };
    vec![cut, enemy_hq_proximity, is_hq, own_vuln, enemy_neighbor_frac, owner_is_enemy]
}

/// Full spatial policy input = [ standard policy_input | spatial(6) ].
pub fn policy_input_spatial(g: &Game, p: PlayerId, global_vec: &[f64], c: &Candidate) -> Vec<f64> {
    let mut v = policy_input(global_vec, c);
    v.extend_from_slice(&candidate_spatial_features(g, p, c));
    v
}

/// Spatial twin of `policy::select_index`: argmax of the spatial-net score over
/// `cands` (deterministic — used only for the AZ fallback retry path, which is
/// off the parity/shipped route). Returns 0 for an empty list.
pub fn select_index_spatial(
    genome: &crate::mlp::Genome,
    g: &Game,
    p: PlayerId,
    global_vec: &[f64],
    cands: &[Candidate],
) -> usize {
    let mut best = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (i, c) in cands.iter().enumerate() {
        let s = crate::mlp::score(genome, &policy_input_spatial(g, p, global_vec, c));
        if s > best_score {
            best_score = s;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidates::Intent;
    use cp_sim::model::BuildingType;

    fn attack_cand(tile: TileId) -> Candidate {
        Candidate {
            intent: Intent::Attack,
            local: vec![0.0; crate::candidates::LOCAL_DIM],
            action: Action::Attack { tile, needed: 1, placed: 0, can_buy: false },
            label: "t".into(),
        }
    }
    fn at(g: &Game, x: i32, y: i32) -> TileId {
        TileId(g.get_tiles().iter().position(|t| t.x == x && t.y == y).unwrap())
    }

    #[test]
    fn spatial_block_dims_and_cut() {
        let mut g = Game::new(6, 3, &["P0", "P1"]);
        g.generate_map(6, 3, 1);
        let enemy = PlayerId(1);
        let line = [at(&g, 0, 0), at(&g, 1, 0), at(&g, 2, 0), at(&g, 3, 0)];
        for &t in &line { g.set_tile_owner(t, Some(enemy)); }
        g.place_building(line[0], BuildingType::Headquarters, Some(enemy));

        let p0 = PlayerId(0);
        let feats = candidate_spatial_features(&g, p0, &attack_cand(line[1]));
        assert_eq!(feats.len(), SPATIAL_LOCAL_DIM);
        // feat[0] = offensive cut on the articulation tile (1,0) -> 2/3.
        assert!((feats[0] - 2.0 / 3.0).abs() < 1e-9, "cut feat {}", feats[0]);
        // feat[2] is_enemy_hq for (1,0) is 0; for the HQ tile it is 1.
        assert_eq!(feats[2], 0.0);
        assert_eq!(candidate_spatial_features(&g, p0, &attack_cand(line[0]))[2], 1.0);
        // feat[5] target_owner_is_enemy = 1 (enemy owns the line).
        assert_eq!(feats[5], 1.0);
        // full input has the right total dim.
        let full = policy_input_spatial(&g, p0, &vec![0.0; crate::features::GLOBAL_DIM], &attack_cand(line[1]));
        assert_eq!(full.len(), POLICY_INPUT_DIM_SPATIAL);
    }

    #[test]
    fn warmstart_predicts_identically_to_base() {
        // A warm-started spatial net must score IDENTICALLY to its 63-dim base for
        // any input, regardless of the 6 spatial features (their weights are 0).
        let base = crate::policy_train::random_genome(&crate::policy::DEFAULT_ARCH.to_vec(), 7);
        let sp = warmstart_spatial(&base);
        assert_eq!(sp.arch, vec![POLICY_INPUT_DIM_SPATIAL, 24, 16, 1]);
        assert_eq!(sp.params.len(), crate::mlp::param_count(&sp.arch));
        let base_inp: Vec<f64> = (0..POLICY_INPUT_DIM).map(|i| (i as f64) * 0.013 - 0.4).collect();
        for spatial6 in [[0.0; 6], [0.5, -0.5, 1.0, 0.2, 0.7, -0.1], [1.0; 6]] {
            let mut sp_inp = base_inp.clone();
            sp_inp.extend_from_slice(&spatial6);
            let sb = crate::mlp::score(&base, &base_inp);
            let ss = crate::mlp::score(&sp, &sp_inp);
            assert!((sb - ss).abs() < 1e-12, "warm-start diverged: base {sb} spatial {ss}");
        }
    }
}
