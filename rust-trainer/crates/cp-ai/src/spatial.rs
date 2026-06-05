//! Spatial / positional features for the AlphaZero representation
//! (design: `rust-trainer/ALPHAZERO-DESIGN.md`, signals: `REWARD-DESIGN.md`).
//!
//! ADDITIVE module: nothing here is wired into the parity-locked feature vectors
//! yet, so the golden/parity path stays byte-identical until `features.rs` /
//! `candidates.rs` are extended (atomically with the TS twins + a golden
//! re-export). This module just provides the computations.
//!
//! The tricky bit — "does removing this tile cut my territory off the HQ?"
//! (signal N3, own-territory-cut) — is a graph articulation measure, written here
//! as PURE functions over an adjacency list so it can be unit-tested without
//! constructing a whole `Game`. The `Game` adapters below are thin glue over
//! `get_hq_tile` / `neighbour_four_tiles` / `live_players` / tile `.x/.y/.owner`,
//! reusing the same orthogonal adjacency the real HQ-cut rule uses
//! (`get_hq_connected_tiles`).

use cp_sim::{Game, PlayerId, TileId};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Pure graph helpers (unit-tested) — no game types.
// ---------------------------------------------------------------------------

/// Size of the connected component containing `root` in an undirected graph
/// given by adjacency list `adj`, optionally treating node `exclude` as removed.
fn component_size(adj: &[Vec<usize>], root: usize, exclude: Option<usize>) -> usize {
    if Some(root) == exclude || root >= adj.len() {
        return 0;
    }
    let mut seen = vec![false; adj.len()];
    let mut stack = vec![root];
    seen[root] = true;
    let mut count = 0usize;
    while let Some(u) = stack.pop() {
        count += 1;
        for &v in &adj[u] {
            if Some(v) == exclude || seen[v] {
                continue;
            }
            seen[v] = true;
            stack.push(v);
        }
    }
    count
}

/// Fraction of the HQ-component (excluding the tile itself) that gets cut off if
/// node `t` is removed — i.e. an articulation measure normalised to [0,1].
///
/// 0 for a leaf or a non-connected node (removing it disconnects nothing); 1 for
/// the HQ node itself; k/(base-1) for an articulation point that severs k tiles.
fn cut_fraction(adj: &[Vec<usize>], hq: usize, t: usize) -> f64 {
    let base = component_size(adj, hq, None);
    if base <= 1 {
        return 0.0;
    }
    if t == hq {
        return 1.0;
    }
    let without = component_size(adj, hq, Some(t));
    // disconnected "others" = base - without - 1 (the removed tile t itself).
    let disconnected = (base as i64 - without as i64 - 1).max(0) as f64;
    disconnected / (base as f64 - 1.0)
}

// ---------------------------------------------------------------------------
// Game adapters.
// ---------------------------------------------------------------------------

/// The owned-tile orthogonal-adjacency graph for player `p`, plus a lookup from
/// `TileId` index → local node index and the HQ's local node (if owned).
pub struct OwnedGraph {
    adj: Vec<Vec<usize>>,
    /// tile index (`TileId.0`) → local node index
    local_of: HashMap<usize, usize>,
    hq: Option<usize>,
    /// local node index → tile index (`TileId.0`)
    pub tiles: Vec<usize>,
}

impl OwnedGraph {
    pub fn build(g: &Game, p: PlayerId) -> OwnedGraph {
        let mut tiles: Vec<usize> = Vec::new();
        let mut local_of: HashMap<usize, usize> = HashMap::new();
        for (i, t) in g.get_tiles().iter().enumerate() {
            if t.owner == Some(p) {
                local_of.insert(i, tiles.len());
                tiles.push(i);
            }
        }
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); tiles.len()];
        for (local, &ti) in tiles.iter().enumerate() {
            for n in g.neighbour_four_tiles(TileId(ti)) {
                if let Some(&nl) = local_of.get(&n.0) {
                    adj[local].push(nl);
                }
            }
        }
        let hq = g.get_hq_tile(p).and_then(|h| local_of.get(&h.0).copied());
        OwnedGraph { adj, local_of, hq, tiles }
    }

    pub fn owned_count(&self) -> usize {
        self.tiles.len()
    }

    /// Cut-vulnerability of one owned tile (signal N3). 0 if the tile isn't owned
    /// by `p` or there's no HQ.
    pub fn cut_fraction_of(&self, tile_index: usize) -> f64 {
        let hq = match self.hq {
            Some(h) => h,
            None => return 0.0,
        };
        match self.local_of.get(&tile_index) {
            Some(&local) => cut_fraction(&self.adj, hq, local),
            None => 0.0,
        }
    }

    /// Mean cut-vulnerability over all owned tiles — a board-invariant
    /// "how fragile is my HQ-connectivity" summary (signal N3, global).
    pub fn mean_cut_risk(&self) -> f64 {
        let hq = match self.hq {
            Some(h) => h,
            None => return 0.0,
        };
        if self.tiles.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0;
        for local in 0..self.tiles.len() {
            sum += cut_fraction(&self.adj, hq, local);
        }
        sum / self.tiles.len() as f64
    }
}

/// Convenience: cut-vulnerability of a single tile (builds the graph once).
/// Prefer `OwnedGraph::build` + `cut_fraction_of` when scoring many tiles.
pub fn cut_vulnerability(g: &Game, p: PlayerId, tid: TileId) -> f64 {
    OwnedGraph::build(g, p).cut_fraction_of(tid.0)
}

/// Convenience: mean HQ-connectivity cut-risk over all owned tiles (signal N3,
/// global summary). Builds the owned graph once.
pub fn mean_cut_risk(g: &Game, p: PlayerId) -> f64 {
    OwnedGraph::build(g, p).mean_cut_risk()
}

/// OFFENSIVE cut-value (the Exp-I policy crux): if `attacker` takes enemy tile
/// `target`, what fraction of that enemy's territory disconnects from its HQ?
/// Computed on the TARGET OWNER's owned-graph via the same articulation measure
/// the HQ-cut rule uses. Returns 1.0 if `target` is the enemy HQ (taking it
/// collapses everything), 0.0 if `target` is neutral or owned by the attacker
/// (no enemy disconnect). This makes the win condition — sever the enemy HQ —
/// visible to per-candidate target selection.
///
/// Prefer [`offensive_cut_value_on`] when scoring many targets of the SAME enemy:
/// it reuses one pre-built graph instead of rebuilding per call.
pub fn offensive_cut_value(g: &Game, attacker: PlayerId, target: TileId) -> f64 {
    match g.get_tiles().get(target.0).and_then(|t| t.owner) {
        Some(owner) if owner != attacker => OwnedGraph::build(g, owner).cut_fraction_of(target.0),
        _ => 0.0, // neutral or own tile: taking it disconnects no enemy
    }
}

/// Offensive cut-value using a pre-built graph of the target's owner (perf path
/// for scoring many candidates against the same enemy).
pub fn offensive_cut_value_on(graph: &OwnedGraph, target: TileId) -> f64 {
    graph.cut_fraction_of(target.0)
}

/// Min Manhattan distance from `tid` to any LIVE enemy player's HQ (signal P10:
/// attack close to the enemy base). Sentinel 99 when no enemy HQ exists.
pub fn dist_to_enemy_hq(g: &Game, p: PlayerId, tid: TileId) -> i32 {
    let tx = g.tiles[tid.0].x;
    let ty = g.tiles[tid.0].y;
    let mut best = 99i32;
    for &op in g.live_players() {
        if op == p {
            continue;
        }
        if let Some(hq) = g.get_hq_tile(op) {
            let d = (tx - g.tiles[hq.0].x).abs() + (ty - g.tiles[hq.0].y).abs();
            if d < best {
                best = d;
            }
        }
    }
    best
}

/// Min Manhattan distance between own HQ and the nearest enemy HQ (sentinel 99).
pub fn hq_to_hq_dist(g: &Game, p: PlayerId) -> i32 {
    let own = match g.get_hq_tile(p) {
        Some(h) => h,
        None => return 99,
    };
    let ox = g.tiles[own.0].x;
    let oy = g.tiles[own.0].y;
    let mut best = 99i32;
    for &op in g.live_players() {
        if op == p {
            continue;
        }
        if let Some(hq) = g.get_hq_tile(op) {
            let d = (ox - g.tiles[hq.0].x).abs() + (oy - g.tiles[hq.0].y).abs();
            if d < best {
                best = d;
            }
        }
    }
    best
}

/// Fraction of own tiles that sit on the enemy frontier (≥1 enemy 8-neighbour)
/// — defence pressure (signals P5/N1). 0 when the player owns nothing.
pub fn frontier_fraction(g: &Game, p: PlayerId) -> f64 {
    let mut owned = 0i64;
    let mut frontier = 0i64;
    for (i, t) in g.get_tiles().iter().enumerate() {
        if t.owner != Some(p) {
            continue;
        }
        owned += 1;
        let touches_enemy = g.neighbour_tiles(TileId(i)).into_iter().any(|n| {
            matches!(g.tiles[n.0].owner, Some(o) if o != p)
        });
        if touches_enemy {
            frontier += 1;
        }
    }
    if owned == 0 {
        0.0
    } else {
        frontier as f64 / owned as f64
    }
}

/// Own tiles within Manhattan `radius` of the nearest enemy HQ, as a fraction of
/// own tiles — how far my front has pushed toward the enemy base (signal P10).
pub fn enemy_hq_push(g: &Game, p: PlayerId, radius: i32) -> f64 {
    // nearest enemy HQ coord
    let mut hq_xy: Option<(i32, i32)> = None;
    let mut best = i32::MAX;
    if let Some(own) = g.get_hq_tile(p) {
        let (ox, oy) = (g.tiles[own.0].x, g.tiles[own.0].y);
        for &op in g.live_players() {
            if op == p {
                continue;
            }
            if let Some(hq) = g.get_hq_tile(op) {
                let (hx, hy) = (g.tiles[hq.0].x, g.tiles[hq.0].y);
                let d = (ox - hx).abs() + (oy - hy).abs();
                if d < best {
                    best = d;
                    hq_xy = Some((hx, hy));
                }
            }
        }
    }
    let (hx, hy) = match hq_xy {
        Some(c) => c,
        None => return 0.0,
    };
    let mut owned = 0i64;
    let mut near = 0i64;
    for t in g.get_tiles() {
        if t.owner != Some(p) {
            continue;
        }
        owned += 1;
        if (t.x - hx).abs() + (t.y - hy).abs() <= radius {
            near += 1;
        }
    }
    if owned == 0 {
        0.0
    } else {
        near as f64 / owned as f64
    }
}

/// Dispersion of own tiles: RMS distance from their centroid, normalised by the
/// RMS of an equal-area compact blob (≈ sqrt(n/π)). ~1 = compact, >1 = spread
/// out. Board-size invariant. 0 for <2 tiles.
pub fn own_dispersion(g: &Game, p: PlayerId) -> f64 {
    let mut xs = 0i64;
    let mut ys = 0i64;
    let mut n = 0i64;
    for t in g.get_tiles() {
        if t.owner == Some(p) {
            xs += t.x as i64;
            ys += t.y as i64;
            n += 1;
        }
    }
    if n < 2 {
        return 0.0;
    }
    let cx = xs as f64 / n as f64;
    let cy = ys as f64 / n as f64;
    let mut var = 0.0;
    for t in g.get_tiles() {
        if t.owner == Some(p) {
            let dx = t.x as f64 - cx;
            let dy = t.y as f64 - cy;
            var += dx * dx + dy * dy;
        }
    }
    let rms = (var / n as f64).sqrt();
    let compact = (n as f64 / std::f64::consts::PI).sqrt().max(1e-9);
    rms / compact
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_size_excludes_node() {
        // line graph 0-1-2-3-4
        let adj = vec![vec![1], vec![0, 2], vec![1, 3], vec![2, 4], vec![3]];
        assert_eq!(component_size(&adj, 0, None), 5);
        // remove node 2 → {0,1} reachable from 0
        assert_eq!(component_size(&adj, 0, Some(2)), 2);
        // exclude the root itself
        assert_eq!(component_size(&adj, 0, Some(0)), 0);
    }

    #[test]
    fn cut_fraction_leaf_is_zero() {
        // line 0-1-2-3-4, hq = 0. Removing the far leaf (4) cuts nothing.
        let adj = vec![vec![1], vec![0, 2], vec![1, 3], vec![2, 4], vec![3]];
        assert_eq!(cut_fraction(&adj, 0, 4), 0.0);
    }

    #[test]
    fn cut_fraction_articulation() {
        // line 0-1-2-3-4, hq = 0. Removing node 2 severs {3,4} (2 of the other 4).
        let adj = vec![vec![1], vec![0, 2], vec![1, 3], vec![2, 4], vec![3]];
        let f = cut_fraction(&adj, 0, 2);
        assert!((f - 2.0 / 4.0).abs() < 1e-9, "expected 0.5, got {f}");
        // node 1 severs {2,3,4} = 3/4
        let f1 = cut_fraction(&adj, 0, 1);
        assert!((f1 - 3.0 / 4.0).abs() < 1e-9, "expected 0.75, got {f1}");
    }

    #[test]
    fn cut_fraction_hq_is_one() {
        let adj = vec![vec![1], vec![0, 2], vec![1, 3], vec![2, 4], vec![3]];
        assert_eq!(cut_fraction(&adj, 0, 0), 1.0);
    }

    #[test]
    fn cut_fraction_star_center() {
        // star: 0 center connected to 1,2,3; hq = 1 (a leaf).
        // removing center 0 severs the other two leaves (2,3) = 2 of the other 3.
        let adj = vec![vec![1, 2, 3], vec![0], vec![0], vec![0]];
        let f = cut_fraction(&adj, 1, 0);
        assert!((f - 2.0 / 3.0).abs() < 1e-9, "expected 0.667, got {f}");
    }

    // --- offensive cut-value (Exp-I crux) integration test over a real Game ----
    use cp_sim::{model::BuildingType, Game, PlayerId, TileId};

    /// Find the tile id at grid (x,y).
    fn at(g: &Game, x: i32, y: i32) -> TileId {
        let i = g.get_tiles().iter().position(|t| t.x == x && t.y == y).expect("tile exists");
        TileId(i)
    }

    #[test]
    fn offensive_cut_value_line() {
        // Build a 6x3 board, then hand ENEMY (P1) a horizontal line of 4 tiles
        // (0,0)-(1,0)-(2,0)-(3,0) with its HQ at (0,0). Attacker = P0.
        let mut g = Game::new(6, 3, &["P0", "P1"]);
        g.generate_map(6, 3, 1);
        // Clear any generated ownership on our line, then assign to P1.
        let enemy = PlayerId(1);
        let line = [at(&g, 0, 0), at(&g, 1, 0), at(&g, 2, 0), at(&g, 3, 0)];
        for &t in &line {
            g.set_tile_owner(t, Some(enemy));
        }
        // HQ at the (0,0) end of the line.
        g.place_building(line[0], BuildingType::Headquarters, Some(enemy));

        let p0 = PlayerId(0);
        // Removing (0,0)=HQ collapses everything -> 1.0
        assert!((offensive_cut_value(&g, p0, line[0]) - 1.0).abs() < 1e-9);
        // Removing (1,0) severs {(2,0),(3,0)} = 2 of the other 3 -> 2/3
        let f1 = offensive_cut_value(&g, p0, line[1]);
        assert!((f1 - 2.0 / 3.0).abs() < 1e-9, "expected 0.667 got {f1}");
        // Removing (2,0) severs {(3,0)} = 1 of the other 3 -> 1/3
        let f2 = offensive_cut_value(&g, p0, line[2]);
        assert!((f2 - 1.0 / 3.0).abs() < 1e-9, "expected 0.333 got {f2}");
        // Removing the far leaf (3,0) severs nothing -> 0
        assert_eq!(offensive_cut_value(&g, p0, line[3]), 0.0);
        // A tile P0 does not face as enemy (neutral / own) -> 0.
        let neutral = at(&g, 5, 2);
        g.set_tile_owner(neutral, None);
        assert_eq!(offensive_cut_value(&g, p0, neutral), 0.0);
        // From the ENEMY's own perspective the line is friendly -> 0 (not an attack).
        assert_eq!(offensive_cut_value(&g, enemy, line[2]), 0.0);
    }
}
