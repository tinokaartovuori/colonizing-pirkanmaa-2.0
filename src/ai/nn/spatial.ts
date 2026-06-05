// Spatial / positional features for the AlphaZero representation — 1:1 TS twin of
// `rust-trainer/crates/cp-ai/src/spatial.rs` (design: ALPHAZERO-DESIGN.md, signals:
// REWARD-DESIGN.md).
//
// ADDITIVE: not wired into the parity-locked feature vectors yet, so the
// golden/parity path stays byte-identical until features.ts/candidates.ts (and
// their Rust twins) are extended atomically. These computations must match
// spatial.rs exactly once wired — every reduction here is order-invariant (set
// membership, min, count, sums) so TS/Rust neighbour-ordering differences don't
// matter.

import type { PlayerBase } from '../../model/player';
import type { TileBase } from '../../model/tile';
import type { ObjectManager } from '../../managers/objectmanager';
import type { PlayerManager } from '../../managers/playermanager';

// --- pure graph helpers (mirror of the Rust unit-tested functions) ---------

/** Component size containing `root` over owned-tile orthogonal adjacency,
 *  optionally treating `exclude` as removed. Order-invariant. */
function componentSize(owned: Set<TileBase>, root: TileBase, exclude: TileBase | null): number {
  if (root === exclude) return 0;
  const seen = new Set<TileBase>([root]);
  const stack: TileBase[] = [root];
  let count = 0;
  while (stack.length) {
    const u = stack.pop() as TileBase;
    count++;
    for (const v of u.getNeighbourFourTiles()) {
      if (v === exclude || !owned.has(v) || seen.has(v)) continue;
      seen.add(v);
      stack.push(v);
    }
  }
  return count;
}

/** Fraction of the HQ-component (excluding the tile itself) cut off if `t` is
 *  removed — articulation measure in [0,1]. Mirrors `cut_fraction` in Rust. */
function cutFraction(owned: Set<TileBase>, hq: TileBase, t: TileBase): number {
  const base = componentSize(owned, hq, null);
  if (base <= 1) return 0;
  if (t === hq) return 1;
  const without = componentSize(owned, hq, t);
  const disconnected = Math.max(0, base - without - 1);
  return disconnected / (base - 1);
}

// --- ObjectManager adapters ------------------------------------------------

/** The set of tiles owned by `p`, plus the HQ tile (if any). */
function ownedSet(p: PlayerBase, om: ObjectManager): { owned: Set<TileBase>; hq: TileBase | null } {
  const owned = new Set<TileBase>();
  for (const t of om.getTiles()) {
    if (t.getOwner() === p) owned.add(t);
  }
  const hq = om.getHqTile(p);
  return { owned, hq: hq && owned.has(hq) ? hq : null };
}

/** Cut-vulnerability of one owned tile (signal N3). */
export function cutVulnerability(tile: TileBase, p: PlayerBase, om: ObjectManager): number {
  const { owned, hq } = ownedSet(p, om);
  if (!hq) return 0;
  return cutFraction(owned, hq, tile);
}

/** Mean cut-vulnerability over all owned tiles — global HQ-fragility (signal N3). */
export function meanCutRisk(p: PlayerBase, om: ObjectManager): number {
  const { owned, hq } = ownedSet(p, om);
  if (!hq || owned.size === 0) return 0;
  let sum = 0;
  for (const t of owned) sum += cutFraction(owned, hq, t);
  return sum / owned.size;
}

/** Min Manhattan distance from `tile` to any enemy player's HQ (signal P10).
 *  Sentinel 99 when no enemy HQ exists. */
export function distToEnemyHq(tile: TileBase, p: PlayerBase, om: ObjectManager, pm: PlayerManager): number {
  const c = tile.getCoordinate();
  const tx = c.x(), ty = c.y();
  let best = 99;
  for (const op of pm.getPlayers()) {
    if (op === p) continue;
    const hq = om.getHqTile(op);
    if (!hq) continue;
    const hc = hq.getCoordinate();
    const d = Math.abs(tx - hc.x()) + Math.abs(ty - hc.y());
    if (d < best) best = d;
  }
  return best;
}

/** Min Manhattan distance between own HQ and the nearest enemy HQ (sentinel 99). */
export function hqToHqDist(p: PlayerBase, om: ObjectManager, pm: PlayerManager): number {
  const own = om.getHqTile(p);
  if (!own) return 99;
  const oc = own.getCoordinate();
  const ox = oc.x(), oy = oc.y();
  let best = 99;
  for (const op of pm.getPlayers()) {
    if (op === p) continue;
    const hq = om.getHqTile(op);
    if (!hq) continue;
    const hc = hq.getCoordinate();
    const d = Math.abs(ox - hc.x()) + Math.abs(oy - hc.y());
    if (d < best) best = d;
  }
  return best;
}

/** Fraction of own tiles on the enemy frontier (≥1 enemy 8-neighbour) — P5/N1. */
export function frontierFraction(p: PlayerBase, om: ObjectManager): number {
  let owned = 0, frontier = 0;
  for (const t of om.getTiles()) {
    if (t.getOwner() !== p) continue;
    owned++;
    const touchesEnemy = t.getNeighbourTiles().some((n) => {
      const o = n.getOwner();
      return o !== null && o !== p;
    });
    if (touchesEnemy) frontier++;
  }
  return owned === 0 ? 0 : frontier / owned;
}

/** Own tiles within Manhattan `radius` of the nearest enemy HQ, as a fraction of
 *  own tiles — how far the front has pushed toward the enemy base (signal P10). */
export function enemyHqPush(p: PlayerBase, om: ObjectManager, pm: PlayerManager, radius: number): number {
  const own = om.getHqTile(p);
  let hx = 0, hy = 0, best = Infinity, found = false;
  if (own) {
    const oc = own.getCoordinate();
    const ox = oc.x(), oy = oc.y();
    for (const op of pm.getPlayers()) {
      if (op === p) continue;
      const hq = om.getHqTile(op);
      if (!hq) continue;
      const hc = hq.getCoordinate();
      const d = Math.abs(ox - hc.x()) + Math.abs(oy - hc.y());
      if (d < best) { best = d; hx = hc.x(); hy = hc.y(); found = true; }
    }
  }
  if (!found) return 0;
  let owned = 0, near = 0;
  for (const t of om.getTiles()) {
    if (t.getOwner() !== p) continue;
    owned++;
    const c = t.getCoordinate();
    if (Math.abs(c.x() - hx) + Math.abs(c.y() - hy) <= radius) near++;
  }
  return owned === 0 ? 0 : near / owned;
}

/** Dispersion of own tiles: RMS distance from centroid / compact-blob RMS
 *  (≈ sqrt(n/π)). ~1 = compact, >1 = spread out. Board-size invariant. */
export function ownDispersion(p: PlayerBase, om: ObjectManager): number {
  let xs = 0, ys = 0, n = 0;
  for (const t of om.getTiles()) {
    if (t.getOwner() === p) {
      const c = t.getCoordinate();
      xs += c.x(); ys += c.y(); n++;
    }
  }
  if (n < 2) return 0;
  const cx = xs / n, cy = ys / n;
  let varSum = 0;
  for (const t of om.getTiles()) {
    if (t.getOwner() === p) {
      const c = t.getCoordinate();
      const dx = c.x() - cx, dy = c.y() - cy;
      varSum += dx * dx + dy * dy;
    }
  }
  const rms = Math.sqrt(varSum / n);
  const compact = Math.max(1e-9, Math.sqrt(n / Math.PI));
  return rms / compact;
}
