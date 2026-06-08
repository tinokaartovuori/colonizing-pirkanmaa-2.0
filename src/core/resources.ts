// Port of Course/basicresources.{h,cpp} and Core/resourcemaps.h.
// All numbers are copied verbatim from the original resourcemaps.h.

export enum BasicResource {
  NONE = 0,
  MONEY = 1,
  WOOD = 2,
  STONE = 3,
  METAL = 4,
}

/**
 * ResourceMap mirrors std::map<BasicResource, int>. We use a Map so that
 * iteration semantics (only present keys are visited) match the C++ original,
 * which several algorithms rely on (mergeResourceMaps, hasEnoughResources).
 */
export type ResourceMap = Map<BasicResource, number>;

export function rmap(entries: Partial<Record<BasicResource, number>>): ResourceMap {
  const m: ResourceMap = new Map();
  for (const [k, v] of Object.entries(entries)) {
    m.set(Number(k) as BasicResource, v as number);
  }
  return m;
}

/** Deep copy. */
export function cloneResourceMap(map: ResourceMap): ResourceMap {
  return new Map(map);
}

/**
 * mergeResourceMaps: start from `right`, then for every key in `left` either
 * insert it (if absent) or add to the existing value. Keys present in only one
 * map survive. (Course::mergeResourceMaps)
 */
export function mergeResourceMaps(left: ResourceMap, right: ResourceMap): ResourceMap {
  const result = new Map(right);
  for (const [key, value] of left) {
    if (!result.has(key)) {
      result.set(key, value);
    } else {
      result.set(key, result.get(key)! + value);
    }
  }
  return result;
}

/** reverseResourceMap: negate every value. */
export function reverseResourceMap(map: ResourceMap): ResourceMap {
  const result = new Map<BasicResource, number>();
  for (const [key, value] of map) result.set(key, value * -1);
  return result;
}

/** getNegativesMap: keep negatives, zero out the rest. */
export function getNegativesMap(map: ResourceMap): ResourceMap {
  const result = new Map<BasicResource, number>();
  for (const [key, value] of map) result.set(key, value < 0 ? value : 0);
  return result;
}

/** getPositivesMap: keep >=0 values, zero out negatives. */
export function getPositivesMap(map: ResourceMap): ResourceMap {
  const result = new Map<BasicResource, number>();
  for (const [key, value] of map) result.set(key, value >= 0 ? value : 0);
  return result;
}

// ---------------------------------------------------------------------------
// ConstResourceMaps (verbatim from resourcemaps.h)
// ---------------------------------------------------------------------------

export const EMPTY: ResourceMap = rmap({});

export const NO_RESOURCES: ResourceMap = rmap({
  [BasicResource.MONEY]: 0,
  [BasicResource.WOOD]: 0,
  [BasicResource.METAL]: 0,
  [BasicResource.STONE]: 0,
});

export const RESOURCE_LIMITS: ResourceMap = rmap({
  [BasicResource.MONEY]: 9999999,
  [BasicResource.WOOD]: 9999999,
  [BasicResource.STONE]: 9999999,
  [BasicResource.METAL]: 9999999,
});

export const UNIT_LIMITS = 999;

export const STARTING_RESOURCES: ResourceMap = rmap({
  [BasicResource.MONEY]: 400,
  [BasicResource.WOOD]: 200,
  [BasicResource.STONE]: 100,
  [BasicResource.METAL]: 25,
});

// Tile - Forest
export const FOREST_PRODUCTION: ResourceMap = rmap({
  [BasicResource.WOOD]: 100,
  [BasicResource.STONE]: 10,
});
export const FOREST_CAPACITY: ResourceMap = rmap({
  [BasicResource.WOOD]: 600,
  [BasicResource.STONE]: 60,
});

// Tile - Abundant Forest
export const ABUNDANT_FOREST_PRODUCTION: ResourceMap = rmap({
  [BasicResource.MONEY]: 15,
});

export const FOREST_GROW_TIME = 5;

// Building - Farm
export const FARM_BUILD_COST: ResourceMap = rmap({
  [BasicResource.MONEY]: -100,
  [BasicResource.WOOD]: -100,
  [BasicResource.METAL]: -5,
});
export const FARM_PRODUCTION: ResourceMap = rmap({
  [BasicResource.MONEY]: 175,
});
export const FARM_GROW_TIME = 4;

// Building - Mine
// NOTE: economy values below DIVERGE from the original C++ resourcemaps.h (a deliberate
// rebalance to make industry a real choice, not a 1:1 port value). See CLAUDE.md.
// Mine is the material engine (metal funds the army/outposts, stone funds villages/plants);
// its build cost was lowered so that engine comes online sooner. Production unchanged so
// the military economy (metal supply) keeps its balance.
export const MINE_BUILD_COST: ResourceMap = rmap({
  [BasicResource.MONEY]: -200,
  [BasicResource.WOOD]: -200,
  [BasicResource.STONE]: 200,
});
export const MINE_PRODUCTION: ResourceMap = rmap({
  [BasicResource.MONEY]: 20,
  [BasicResource.STONE]: 30,
  [BasicResource.METAL]: 20,
});

// Building - Hydroelectric Power Plant
// Rebalanced (diverges from C++): cheaper and far more productive (40 -> 80/worker) so that
// claiming a straight-river tile is genuinely worth it — a mid-game money building that
// beats a farm per unit-slot while sitting on otherwise low-value river land.
export const HEPP_BUILD_COST: ResourceMap = rmap({
  [BasicResource.MONEY]: -280,
  [BasicResource.WOOD]: -150,
  [BasicResource.STONE]: -120,
  [BasicResource.METAL]: -60,
});
export const HEPP_PRODUCTION: ResourceMap = rmap({
  [BasicResource.MONEY]: 80,
});

// Building - Nuclear Power Plant
// Rebalanced (diverges from C++): the late-game economic engine. The cost is shifted into
// MONEY (1200 -> 2000) and away from materials (500/500 -> 250/250) so it is a long money
// *savings goal* rather than a mine-grind, and the payoff is big — 100 -> 160/worker, so an
// expert + 2 workers nets ~285/round (~2.4x a farm per unit-slot). Save up, then dominate.
export const NUCLEARPP_BUILD_COST: ResourceMap = rmap({
  [BasicResource.MONEY]: -2000,
  [BasicResource.WOOD]: -200,
  [BasicResource.STONE]: -250,
  [BasicResource.METAL]: -250,
});
export const NUCLEARPP_PRODUCTION: ResourceMap = rmap({
  [BasicResource.MONEY]: 160,
});

// Building - Outpost (REBALANCED — diverges from C++; this is the source of truth)
// The C++/original cost (650 money + 300 wood + 300 stone + 300 METAL) made the Outpost —
// and therefore the entire soldier-cap → army chain — UNREACHABLE on a normal economy: 300
// metal at once needs ~15 mine-rounds of pure hoarding (a mine yields 20 metal/round) and a
// normal game runs ~1 mine, so the candidate was never enumerated and the soldier cap stayed
// hard-locked at 1 (HQ +1). Same "always-worse, never-chosen" trap the Mine/Hydro/Nuclear
// tier was rebalanced out of (see CLAUDE.md). New cost: metal 300→100 (≈5 mine-rounds, ~3
// with an expert mine), money 650→500, wood/stone 300→200 — the army stays a deliberate
// multi-resource commitment but is now a REACHABLE real choice mid-game. Stays in lockstep
// with outpost_build_cost() in rust-trainer/crates/cp-sim/src/resources.rs.
export const OUTPOST_BUILD_COST: ResourceMap = rmap({
  [BasicResource.MONEY]: -500,
  [BasicResource.WOOD]: -200,
  [BasicResource.STONE]: -200,
  [BasicResource.METAL]: -100,
});
// Per-round Outpost upkeep. METAL upkeep REBALANCED -15 → -5 (arc sd2 → sd3, 2026-06-06,
// "military-economy" pass). The -15 metal/round drain meant a single mine (≈20 metal/round)
// could barely carry ONE Outpost before going metal-negative, so the soldier-cap → army
// chain stalled at 1-2 soldiers and military was never fundable (see CLAUDE.md). At -5/round
// ~1 mine comfortably carries 2-3 Outposts (cap 7-10 soldiers). The -50 money upkeep, the
// outpost build cost, mine output and the Device soldier-cap halving are all left UNCHANGED
// so a pure-economy line stays competitive. Mirrors outpost_production() in
// rust-trainer/crates/cp-sim/src/resources.rs (parity-locked).
export const OUTPOST_PRODUCTION: ResourceMap = rmap({
  [BasicResource.MONEY]: -50,
  [BasicResource.METAL]: -5,
});
export const OUTPOST_SOLDIER_VALUE = 3;

// Building - Bridge
export const BRIDGE_BUILD_COST: ResourceMap = rmap({
  [BasicResource.MONEY]: -100,
  [BasicResource.WOOD]: -300,
  [BasicResource.STONE]: -150,
});
export const BRIDGE_PRODUCTION: ResourceMap = rmap({
  [BasicResource.WOOD]: -5,
});

// Building - Village (Neighborhood)
// MONEY upkeep REBALANCED -10 → -5 (arc sd3 → sd4, 2026-06-08, "unit-cap economy" pass).
// Unit cap = HQ(+3) + Σ Village(+3) + Mikontalo(+2); the ONLY scalable source of cap is
// the Village, at -10 money/round each. Funding BOTH a ~3-4 soldier army AND winning
// expansion needs cap to staff mines (2 workers + expert = 80 metal) + farms + experts
// simultaneously — typically 3 Villages beyond HQ, i.e. -30 money/round of pure cap
// overhead at -10. That overhead made army and winning expansion a genuine tradeoff
// (every PPO/AZ champion capped ~0.62-0.66 trueWin; see CLAUDE.md "unit-cap economy
// rebalance"). At -5 the same 3-Village cap costs -15/round — ~one staffed farm's net
// income (≈+39) restored toward expansion, enough to carry the army's Outpost upkeep
// (-50) + a couple soldier salaries while still expanding. The -10 wood/-10 stone upkeep,
// the build cost, and VILLAGE_UNIT_VALUE are left UNCHANGED so a Village stays a real
// commitment and the cap-per-Village stays put; only the binding money drain is halved.
// Mirrors village_production() in rust-trainer/crates/cp-sim/src/resources.rs
// (parity-locked) and the village net-money model in metrics.ts/metrics.rs + the
// buildVillage netDelta in candidates.ts/candidates.rs.
export const VILLAGE_BUILD_COST: ResourceMap = rmap({
  [BasicResource.MONEY]: -200,
  [BasicResource.WOOD]: -200,
  [BasicResource.STONE]: -100,
  [BasicResource.METAL]: -25,
});
export const VILLAGE_PRODUCTION: ResourceMap = rmap({
  [BasicResource.MONEY]: -5,
  [BasicResource.WOOD]: -10,
  [BasicResource.STONE]: -10,
});
export const VILLAGE_UNIT_VALUE = 3;

// Building - Strange Device
// DELIBERATE divergence from the C++/Qt original (the original had no Device) — a new
// decisive, draw-eliminating win condition. See STRANGE-DEVICE-DESIGN.md and CLAUDE.md.
// The build cost is a one-time nuclear-tier commitment with NO per-turn drain; the
// balancer is the soldier-cap halving (PlayerBase.updateUnitAmounts), not the economy.
// These numbers are TUNABLE — set empirically via the sim/ draw-rate + builder-win-rate
// measurement (target: timeout% → 0 AND builder-vs-non-builder win% ≈ 50%).
export const STRANGE_DEVICE_BUILD_COST: ResourceMap = rmap({
  // Tuned 2026-06-03 to 1300 (from a 1800 first pass): at 1800 the halved-army builder
  // was 56% of all bankruptcies and couldn't afford to DEFEND its own Device; 1300 keeps
  // it a real commitment while dropping bankruptcies (~27%->~21% in hard-vs-hard), making
  // the Device more central (~39% of games end by it) and Device-survival ~coin-flip
  // (~53%). Kept in lockstep with cp-sim strange_device_build_cost().
  [BasicResource.MONEY]: -1300,
  [BasicResource.STONE]: -200,
  [BasicResource.METAL]: -200,
});
// Countdown (in the owner's end-of-turns ≈ rounds) before a standing Device wins.
// Scales with map size: on a bigger map it takes longer to mass an army and cross the
// map to crack the Device, so the countdown must be longer to stay a genuine threat
// rather than an instant win. round(BASE + PER_TILE × tileCount): 10x10→22, 12x12→26,
// 14x12→29, 16x16→38. Trimmed for arc sd5 (was 18 + 0.12) so the device can land before
// a conquest race closes the game — a real alternate win path, not always-too-slow.
// Tunable.
export const STRANGE_DEVICE_COUNTDOWN_BASE = 12;
export const STRANGE_DEVICE_COUNTDOWN_PER_TILE = 0.1;
export function strangeDeviceCountdown(tileCount: number): number {
  return Math.round(STRANGE_DEVICE_COUNTDOWN_BASE + STRANGE_DEVICE_COUNTDOWN_PER_TILE * tileCount);
}

// Building - Mikontalo
export const MIKONTALO_UNIT_VALUE = 2;

// Building - HQ
export const HQ_UNIT_VALUE = 3;
export const HQ_SOLDIER_VALUE = 1;

// Worker
export const BASIC_WORKER_COST: ResourceMap = rmap({ [BasicResource.MONEY]: -50 });
export const BASIC_WORKER_SALARY: ResourceMap = rmap({ [BasicResource.MONEY]: -5 });

// Expert
export const EXPERT_COST: ResourceMap = rmap({ [BasicResource.MONEY]: -250 });
export const EXPERT_SALARY: ResourceMap = rmap({ [BasicResource.MONEY]: -25 });

// Soldier
// METAL build cost REBALANCED -50 → -30 (arc sd2 → sd3, 2026-06-06, "military-economy" pass).
// Paired with the Outpost metal-upkeep cut above so a single-mine economy can actually FIELD
// an army: at 50 metal/soldier a 4-6 soldier army (200-300 metal) drained most of a mine's
// output, so soldiers were never massed; at 30 metal/soldier that army costs 120-180 metal
// and is fundable alongside the (now cheaper) Outpost upkeep. The -200 money build cost and
// the -30 money salary are left UNCHANGED so soldiers stay a money commitment and pure-economy
// stays competitive. Mirrors soldier_cost() in rust-trainer/crates/cp-sim/src/resources.rs
// (parity-locked).
export const SOLDIER_COST: ResourceMap = rmap({
  [BasicResource.MONEY]: -200,
  [BasicResource.METAL]: -30,
});
export const SOLDIER_SALARY: ResourceMap = rmap({ [BasicResource.MONEY]: -30 });
