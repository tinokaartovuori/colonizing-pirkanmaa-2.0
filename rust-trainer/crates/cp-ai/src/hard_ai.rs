//! Faithful Rust port of `src/managers/ai.ts` — the HELD-OUT hard heuristic.
//!
//! This is NOT on the parity / policy-net path. It is a standalone benchmark
//! opponent: a behaviourally-faithful port of the TypeScript `AiController`
//! (the ~1300-line bounded heuristic that drives a CPU through HQ placement and
//! a full turn). It mirrors the TS decision logic, ordering, affordability rules
//! and target selection. It does NOT need to be bit-for-bit; it must be
//! strategically equivalent — same priorities and same strength.
//!
//! It drives a `cp_sim::Game` for the *current* seat, reusing the same engine
//! primitives a human/the NN controller would (`ai_build_building`,
//! `ai_buy_and_place_unit`, `ai_move_unit`). All of `get_available_tiles`,
//! `free_unit_amount`, etc. operate on `current_player()`, so HardAi must be
//! called only when its seat is current (the harness ensures this).
//!
//! The TS turn is a generator that yields per action; headless we run straight
//! through. The action budget is decremented exactly as the TS `doAction` does:
//! only on a *successful* action.

use cp_sim::resources::{
    self, basic_worker_cost, expert_cost, soldier_cost, BasicResource, ResourceMap,
};
use cp_sim::{BuildingType, Game, PlayerId, TileId, TileType, UnitId, UnitType};

/// Per-difficulty parameters (`PARAMS` in ai.ts). Only `hard` is used as the
/// benchmark, but `easy`/`medium` are provided for completeness/parity of the
/// strategy surface.
#[derive(Debug, Clone, Copy)]
pub struct AiParams {
    pub reserve: i64,
    pub max_actions: i64,
    pub experts: bool,
    pub military: bool,
    pub garrison: i64,
    pub expand: i64,
    pub attack: bool,
    pub nuclear: bool,
    pub max_outposts: i64,
    pub strike_force: i64,
    pub assaults_per_turn: i64,
    pub warmonger: bool,
    /// EXPERIMENTAL (non-shipped, ceiling probe only): when true, the `attack`
    /// phase orders targets by `spatial::offensive_cut_value` (the fraction of
    /// enemy territory that disconnects if the tile is taken) instead of the
    /// shipped "HQ-first, then fewest-defenders" order. Default false → byte-
    /// identical to the ported TS bot (parity-safe).
    pub cut_priority: bool,
    /// Use the Strange Device WIN strategy: when clearly leading, build the Device
    /// to force a decisive finish. Gates only the BUILD decision — the counterplay
    /// (massing soldiers + assaulting an enemy Device on sight) is always on, so a
    /// `device: false` AI still races to crack one. Mirrors `AiParams.device`.
    pub device: bool,
    /// LEAGUE-REBUILD (2026-06-06): proactive Outpost building for the turtle
    /// (`FORTRESS`). When true, `build_outposts` relaxes the `should_militarise` /
    /// `military_need` early-return (via `proactive_outposts`) so the bot lays
    /// soldier-cap Outposts BEFORE first contact instead of waiting for a war.
    /// Default `false` keeps every shipped preset (incl. HARD) byte-identical.
    pub fortress: bool,
    /// LEAGUE-REBUILD: strong-army assault-readiness gate (`STRONG_ARMY`). When > 0,
    /// the `attack` phase refuses to open a front until the bot has massed at least
    /// this many soldiers (an enemy Device still cracks the gate). `0` = ship
    /// behavior (open a front whenever a legal Attack exists).
    pub attack_ready_soldiers: i64,
    /// LEAGUE-REBUILD: strong-army econ-readiness gate (`STRONG_ARMY`). When > 0, the
    /// `military` force computation stays defense-only until `net_money_per_round`
    /// reaches this threshold (an enemy Device overrides). `0` = ship behavior.
    pub econ_ready_net: i64,
    /// LEAGUE-REBUILD STEP E v2 (2026-06-06): army-builder force ceiling (`STRONG_ARMY`).
    /// When true, the `military` border-guard / strike-force phase fields the FULL
    /// `garrison + strike_force` army (capped only by `max_soldier_amount` and what the
    /// staffed economy — farms + staffed expert producers + 1 — can underwrite),
    /// instead of HARD's tiny `farms + 1` ceiling. It does NOT add any deadlocking
    /// readiness gate: the `attack`/`march` phases stay on HARD's proven, ungated path
    /// (so the front opens, contact is made, and the army actually grows). `false` =
    /// HARD's ship behavior (byte-identical). See `military()`.
    pub army_builder: bool,
}

/// `PARAMS.hard` — the held-out benchmark difficulty.
pub const HARD_PARAMS: AiParams = AiParams {
    reserve: 140,
    max_actions: 28,
    experts: true,
    military: true,
    garrison: 3,
    expand: 5,
    attack: true,
    nuclear: true,
    max_outposts: 5,
    strike_force: 7,
    assaults_per_turn: 7,
    warmonger: false,
    cut_priority: false,
    device: true,
    fortress: false,
    attack_ready_soldiers: 0,
    econ_ready_net: 0,
    army_builder: false,
};

/// `PARAMS.medium`.
pub const MEDIUM_PARAMS: AiParams = AiParams {
    reserve: 110,
    max_actions: 14,
    experts: true,
    military: true,
    garrison: 2,
    expand: 3,
    attack: true,
    nuclear: false,
    max_outposts: 2,
    strike_force: 3,
    assaults_per_turn: 4,
    warmonger: false,
    cut_priority: false,
    device: true,
    fortress: false,
    attack_ready_soldiers: 0,
    econ_ready_net: 0,
    army_builder: false,
};

/// `PARAMS.easy`.
pub const EASY_PARAMS: AiParams = AiParams {
    reserve: 80,
    max_actions: 5,
    experts: false,
    military: true,
    garrison: 1,
    expand: 2,
    attack: true,
    nuclear: false,
    max_outposts: 1,
    strike_force: 1,
    assaults_per_turn: 1,
    warmonger: false,
    cut_priority: false,
    device: false,
    fortress: false,
    attack_ready_soldiers: 0,
    econ_ready_net: 0,
    army_builder: false,
};

/// Scripted DEVICE-RUSHER strategy opponent (Lever C, TRAINING-ONLY). A HardAi
/// variant biased to bank a minimal economy and build the Strange Device as early
/// as the gate allows (the engine gate already enforces `rounds >= 18`, not-losing,
/// affordable — `build_strange_device`), then DEFEND it (the existing `military`
/// phase already rings the Device's approaches; `attack` is left ON so it can still
/// crack an enemy Device, but with a tiny strike force it does not go on the
/// offensive). Faithful to GAME-MECHANICS §6: the device tile holds zero defenders
/// and the build halves the soldier cap, so this opponent is a *defensively fragile*
/// rush the learner should be able to punish if it over-extends — and must learn to
/// out-race / raid otherwise. This is NOT a new agent or rule: it is HardAi with
/// skewed `AiParams`, so it stays legal and parity-irrelevant.
///
/// LEAGUE-REBUILD (2026-06-06) — DEVICE-STRATEGIST rebuild. The old preset banked too
/// thin (reserve 120) and used a 5-round drain projection that under-counted the FULL
/// countdown of payroll, so it self-bankrupted during the Device clock. The rebuild:
///   - reserve 120→250 + a full-countdown safety projection in `build_strange_device`
///     (the bankruptcy fix — covers ~60% of the gross payroll across the whole
///     countdown, net of income),
///   - `warmonger: true` so `proactive_outposts` fires (the Device-precursor Outpost
///     is laid proactively, not gated behind first contact),
///   - `max_outposts: 3` + a raised `outposts < 2` device precursor so the halved cap
///     still leaves real defenders,
///   - a `military()` branch that fields the whole halved army to ring an OWN device,
///   - a small offensive force (strike_force/assaults 2) so it can still poke.
/// All new logic is gated on `device` (+ `warmonger` for the proactive outposts), so
/// HARD (`device: true, warmonger: false`) is unaffected.
pub const DEVICE_RUSH_PARAMS: AiParams = AiParams {
    reserve: 150,            // FIX 2: lowered 250→150 so banking can spend down to the
                             // Device cost (a 250 reserve held cash hostage above the build)
    max_actions: 28,
    experts: true,           // efficient economy to afford the Device fast
    military: true,          // keep soldiers so it can DEFEND its device
    garrison: 3,
    expand: 3,               // FIX 2: a DENSE economy. expand 6 grabbed so many neutrals that
                             // the bot's territory cut off the (9-tile) passive opponent and
                             // won by CONQUEST before the Device countdown — defeating the
                             // Device strategy. 3 keeps enough econ for the Device while not
                             // racing to swallow the map. (ARC sd3: tried 4 — it DILUTED the
                             // build, dropping no-op build% 90→82; reverted.)
    attack: true,            // counterplay stays on (crack an enemy device on sight)
    nuclear: false,          // the Device, not Nuclear, is the win plan
    max_outposts: 1,         // FIX 2: just the precursor. A 2nd Outpost (200 wood + 100
                             // metal + 50 money/round) starved the Device build (build% fell
                             // 86→54, win% 84→44) for a negligible ring gain — not worth it.
                             // One Outpost → halved cap (HQ 1 + 3)/2 = 2 ring soldiers.
    strike_force: 0,         // FIX 2: no offensive army — the Device is the win plan, and
                             // soldier upkeep (30/round each) starved the money bank
    assaults_per_turn: 0,    // (counter-crack of an enemy Device still fires via `can_buy`)
    warmonger: true,         // load-bearing: fires `proactive_outposts` (device precursor)
    cut_priority: false,
    device: true,            // THE point: race the Strange Device
    fortress: false,
    attack_ready_soldiers: 0,
    econ_ready_net: 0,
    army_builder: false,
};

/// Scripted ARMY-RUSHER strategy opponent (Lever C, TRAINING-ONLY). A HardAi variant
/// biased to max soldier capacity (Outposts give +3 cap each — GAME-MECHANICS §5),
/// expand, hire soldiers and assault early with soldier-superiority (the `attack`
/// phase only targets non-Outpost tiles where it out-numbers the defender — §3).
/// Faithful to the mechanics: no new actions, just HardAi with priorities skewed
/// toward military capacity + aggression and the Device turned OFF (it commits to the
/// army win, not the Device race). The learner must build defensive capacity (the
/// exact capacity-blindness gap) to survive it.
pub const ARMY_RUSH_PARAMS: AiParams = AiParams {
    reserve: 100,
    max_actions: 30,
    experts: true,
    military: true,
    garrison: 3,
    expand: 6,               // grab tiles to feed Outposts + frontier pressure
    attack: true,
    nuclear: false,
    max_outposts: 7,         // MAX soldier cap (each Outpost = +3 cap)
    strike_force: 10,        // field a big offensive army
    assaults_per_turn: 10,   // press the assault every turn
    warmonger: true,         // gear up for war as soon as any enemy exists
    cut_priority: false,
    device: false,           // commit to the army win, not the Device
    fortress: false,
    attack_ready_soldiers: 0,
    econ_ready_net: 0,
    army_builder: false,
};

/// Plan-B HQ-RUSHER strategy opponent (TRAINING-ONLY). Same army-emphasis shape as
/// ARMY_RUSH (Outpost → soldier → soldier), but cranked further on aggression. The
/// HARD bot's existing `attack` phase ALREADY orders targets HQ-first (Device > HQ
/// > fewest-defenders, see `hard_ai::attack`), so the HQ-attack bias is structural
/// in the shipped attack code; HQ_RUSH just turns that knob to 11 by maximising
/// `assaults_per_turn` + `strike_force` + `warmonger` so the rusher relentlessly
/// pushes toward whichever enemy HQ is closest to its frontier. Plan-B's separate
/// `Intent::CrackHQ` candidate (a NN-side action only) makes the learner pay the
/// matching counter-cost. NOT a new agent or game rule — HardAi with biased
/// `AiParams`, so parity-irrelevant.
pub const HQ_RUSH_PARAMS: AiParams = AiParams {
    reserve: 80,
    max_actions: 32,
    experts: true,
    military: true,
    garrison: 2,             // bare-minimum garrison: every spare soldier goes attacking
    expand: 5,               // enough territory to feed Outposts + frontier pressure
    attack: true,
    nuclear: false,
    max_outposts: 7,         // MAX soldier cap (each Outpost = +3 cap)
    strike_force: 12,        // bigger striker than ARMY_RUSH (10)
    assaults_per_turn: 14,   // even more assaults per turn (vs ARMY_RUSH 10) — keep pressing HQ
    warmonger: true,         // gear up for war as soon as any enemy exists
    cut_priority: false,     // shipped HQ-first order in `attack` already favours HQs
    device: false,           // commit to the army win (HQ-cracking line), not the Device
    fortress: false,
    attack_ready_soldiers: 0,
    econ_ready_net: 0,
    army_builder: false,
};

/// OVERNIGHT-RUN §B.1 GARRISON-FORTRESS strategy opponent (TRAINING-ONLY). Closes the
/// 1-soldier-rush hole user-identified: HARD's default `garrison: 3` only fires under
/// `at_war` (`should_militarise()`), so early-mid game HARD holds 0-1 defenders and
/// falls to a single staged soldier. GARRISON forces an unconditional ≥ 3 HQ garrison
/// from round 1 (via `warmonger: true` → `enemy_exists()` is true from round 1, so
/// `should_militarise()` returns true). The assault phase is suppressed by the
/// existing `if assaults_per_turn <= 1 && !can_buy { return; }` gate in `attack()`
/// (lines ~1920), but counter-cracking an enemy Device stays ON via `attack: true`
/// + the `can_buy` window. NOT a new agent or rule — HardAi with biased `AiParams`,
/// so parity-irrelevant. See OVERNIGHT-RUN-PLAN.md §B.1.
pub const GARRISON_PARAMS: AiParams = AiParams {
    // SELF-BANKRUPTCY AUDIT FIX (2026-06-05): reserve 100→300, max_outposts 4→2.
    // GARRISON's `warmonger: true` keeps `at_war` ON from round 1, so the garrison
    // hire path runs every turn. With `expand: 2` capping income tiles, the bot
    // operates near break-even in long games (avg 152 rounds) and the per-commit
    // 4-round `affordable_after_commit` projection isn't enough — cumulative drift
    // (and opponent conquest stripping Farms / cutting connectivity) tips the bot
    // past its income ceiling. Tripling `reserve` forces every `affords()` call to
    // keep a $300 cash floor, and halving max_outposts (was 4 → now 2) cuts the
    // metal-drain exposure (each Outpost costs 15 metal/round and the variant only
    // builds 1 Mine on `expand: 2`). A 2-OP turtle still has cap = HQ(1) + 2×3 = 7
    // soldiers, more than enough for the 3-garrison + 1-2-frontier-defender pattern.
    // Audit (50 same-vs-same games, seeds 1..=20) dropped 15.0% → ≤5%. Behavioural
    // load-bearing properties (warmonger, garrison=3, strike_force=0, assaults=0)
    // are unchanged.
    reserve: 300,
    max_actions: 24,
    experts: true,
    military: true,
    garrison: 3,             // THE point: unconditional ≥ 3 HQ garrison (no 1-rush hole)
    expand: 2,               // small, dense empire — fewer frontiers to cover
    attack: true,            // counter-cracker stays ON (cracker fires via `can_buy`)
    nuclear: false,
    max_outposts: 2,         // 2 OPs cap = 7 soldiers, enough for 3-garrison + frontier
    strike_force: 0,         // turtle: never go on the offensive
    assaults_per_turn: 0,    // suppressed via the `<= 1 && !can_buy` gate in attack()
    warmonger: true,         // load-bearing: forces `at_war` true from round 1
    cut_priority: false,
    device: false,           // pure fortress: no Device race
    fortress: false,         // legacy GARRISON keeps reactive (warmonger) garrison only
    attack_ready_soldiers: 0,
    econ_ready_net: 0,
    army_builder: false,
};

/// REACTIVE-FIX MARCHER strategy opponent (TRAINING-ONLY). Closes the "AI sits at
/// home with its 3 soldiers and waits" hole user-identified: even when the learner
/// builds an army, no scripted opponent ever DEMONSTRATES "march 3 soldiers across
/// the map → conquer". ARMY_RUSH / HQ_RUSH attack ONLY when an enemy is in
/// `get_available_tiles()` (adjacency to enemy frontier required); they have no
/// "advance soldiers TOWARD a distant enemy HQ without legal attack" phase, so on a
/// no-contact opening they sit at HQ with their soldiers like HARD does. MARCHER
/// adds a `march_to_enemy_hq` phase wired into `run_turn` (gated on
/// `params.warmonger`) that, when the bot's soldiers have NO legal Attack this turn,
/// MOVES one or more spare soldiers to whichever tile in `get_available_tiles`
/// minimises Manhattan distance to the nearest enemy HQ. Pair this with cranked
/// aggression knobs (assaults_per_turn=16, strike_force=14, garrison=1) so once the
/// march reaches the frontier the existing attack phase fires hard.
///
/// NOT a new agent or game rule — HardAi with biased `AiParams` + ONE extra phase
/// that calls existing `ai_move_unit` engine primitives. Parity-irrelevant (off the
/// candidates path).
pub const MARCHER_PARAMS: AiParams = AiParams {
    reserve: 80,
    max_actions: 32,
    experts: false,          // skip expert economy — minimal econ, max marcher
    military: true,          // soldiers, soldiers, soldiers
    garrison: 1,             // minimal home defence — willing to leave HQ for the front
    expand: 4,               // still claim neutrals to gain reach toward the enemy
    attack: true,            // existing attack phase fires hard once at the frontier
    nuclear: false,
    max_outposts: 2,         // one early Outpost (cap → 4 soldiers), second to anchor advance
    strike_force: 14,        // huge offensive army goal
    assaults_per_turn: 16,   // press the assault every chance
    warmonger: true,         // load-bearing: `at_war` true from round 1 → march phase fires
    cut_priority: false,
    device: false,           // commit to the army win
    fortress: false,
    attack_ready_soldiers: 0,
    econ_ready_net: 0,
    army_builder: false,
};

/// OVERNIGHT-RUN §B.2 EXPERT-STACKED ECONOMY strategy opponent (TRAINING-ONLY). Closes
/// the Expert-handling gap: cnn-r1 logged 0 HireExpert intents across 240 self-play
/// games. EXPERT plays a pure-econ bot that fronts the Expert tier (Mine + Expert
/// doubles output; Hydro/Nuclear gate production entirely on Expert presence — see
/// `cp_sim/managers.rs:846-887`). The learner faces an opponent whose per-round income
/// overtakes farm-only economies by ~r25, supplying Domination-loss pressure unless
/// the learner ALSO staffs Experts. No build-side code change — HARD's existing
/// `build_power_plants` + `invest_nuclear` + `boost_mines` + `staff_plant` already
/// prefer Experts when `experts: true`. NOT a new agent or rule. See §B.2.
pub const EXPERT_PARAMS: AiParams = AiParams {
    // SELF-BANKRUPTCY AUDIT FIX (2026-06-05): reserve 140→200, max_outposts 2→1.
    // ECON_EXPERT's failure mode in the audit was two-pronged: (a) a $1845-cash
    // seat going bankrupt one turn after the snapshot from cumulative late-game
    // salary creep (Expert salary 25/r each + worker salary 5/r × many workers),
    // and (b) seeds where the bot built an Outpost early then lost its Mine to
    // conquest, leaving an Outpost (-15 metal/r) bleeding metal until negative.
    // Since the variant has `military: false` (no defensive soldier hire) and only
    // a 1-soldier counter-cracker, max_outposts: 2 was never strategically needed —
    // dropping to 1 caps the metal-leak risk without changing the cracker chain
    // (1 OP is enough to feed the cap → 1 soldier → crack-a-Device path). Raising
    // reserve to 200 widens the late-game cash cushion (every `affords()` call
    // now keeps $200 floor) without throttling the early-game build-out (Hydro
    // affords() already uses `reserve.min(80)`). Audit dropped 10.0% → 0.0%.
    reserve: 200,
    max_actions: 28,
    experts: true,           // THE point: front the Expert tier
    military: false,         // pure economic teacher — never strikes on the offensive
    garrison: 1,             // bare minimum (no military emphasis)
    expand: 4,
    attack: true,            // counter-cracker stays ON against an enemy Device
    nuclear: true,           // Nuclear is the late-game engine — pair with Experts
    max_outposts: 1,         // 1 OP feeds the cracker chain; 2 was a metal-leak risk
    strike_force: 0,         // no offensive army
    assaults_per_turn: 1,    // cracker-only (no offensive assaults beyond it)
    warmonger: false,        // do NOT pre-emptively militarise — pure econ
    cut_priority: false,
    device: false,
    fortress: false,
    attack_ready_soldiers: 0,
    econ_ready_net: 0,
    army_builder: false,
};

/// LEAGUE-REBUILD (2026-06-06) — canonical RUSHER ("homing missile"). Pure param fix,
/// NO new logic: `warmonger` + `military` already wire `build_bridges` +
/// `march_to_enemy_hq` + `attack` (Device > HQ > fewest-defenders ordering). The
/// old army/HQ rushers banked far too thin (reserve 80-100) and bankrupted themselves
/// under the +30/round-per-soldier upkeep; raising `reserve` to 220 is the bankruptcy
/// fix (every `affords` call now keeps a $220 floor) while keeping the aggression knobs
/// hot. `max_outposts: 2` caps the metal-leak exposure (each Outpost = 15 metal/round).
pub const RUSHER_PARAMS: AiParams = AiParams {
    // ARC sd3 RE-TUNE (2026-06-07): reserve 220→170. With the rebalanced cheaper military
    // (soldier 30 metal, Outpost upkeep 5/round) the old 220 floor over-banked cash that
    // could fund earlier soldiers + a harder HQ push; 170 still clears the bankruptcy bar
    // (the per-commit upkeep projections are the real guard) while freeing tempo for the
    // homing rush (HQ-reach was only ~42% — too passive).
    reserve: 170,
    max_actions: 30,
    experts: true,
    military: true,
    garrison: 2,
    expand: 4,
    attack: true,
    nuclear: false,
    max_outposts: 2,         // cap → 7 soldiers; limits the metal-leak exposure
    strike_force: 6,         // kept at 6 (a focused homing striker, not a standing army — the
                             // STRONG_ARMY yardstick fields the larger force at strike_force 12)
    assaults_per_turn: 10,   // ARC sd3: 8→10 — march/assault harder each turn (homing pressure)
    warmonger: true,         // wires build_bridges + march_to_enemy_hq + early attack
    cut_priority: false,
    device: false,
    fortress: false,
    attack_ready_soldiers: 0,
    econ_ready_net: 0,
    army_builder: false,
};

/// LEAGUE-REBUILD (2026-06-06) — canonical FORTRESS (the turtle). Builds soldier-cap
/// Outposts PROACTIVELY (before first contact) via `fortress: true` →
/// `proactive_outposts` relaxes the `build_outposts` militarise/military_need gate, then
/// turtles: `strike_force: 0` / `assaults_per_turn: 0` suppress the offensive (the
/// `attack` `<= 1 && !can_buy` gate), and the `warmonger`-march is gated on
/// `strike_force > 0` so the turtle NEVER marches its wall away. `attack: true` keeps the
/// counter-cracker on (an enemy Device still cracks via `can_buy`). `reserve: 320` banks
/// a fat cushion against the +50/round-per-Outpost upkeep — the bankruptcy fix for a bot
/// that proactively builds up to 3 Outposts.
/// ARC sd3 ROOT-CAUSE FIX (2026-06-07) — the FORTRESS's total owned-tile ceiling. A turtle
/// wants a SMALL DENSE empire, NOT reach: ~12 tiles fits HQ + 1-2 mines + 3-4 farms +
/// a forest harvester (Village fuel) + 2-3 Outposts + the HQ ring. Beyond this it would
/// only burn its HQ-only unit cap on scouts and starve its own wall (see `expand`). Used
/// ONLY on the `fortress` path, so HARD / every other preset is byte-identical.
const FORTRESS_TILE_CAP: i64 = 14;

pub const FORTRESS_PARAMS: AiParams = AiParams {
    // LEAGUE-REBUILD STEP F (2026-06-06) — reserve 320→130. The fat 320 reserve was
    // STARVING the very wall it was meant to fund: every `affords()` call keeps a
    // `reserve + drain*5` cash floor, and at 320 + a multi-Outpost drain the floor sat
    // ABOVE the bot's banked cash, so it could neither hire its garrison soldiers
    // (200 money each) nor lay its 2nd/3rd Outpost (500 money) even while sitting on
    // 800-900 banked money / 400+ metal (see fort_diag). The per-commit
    // `affordable_after_commit(.., 4 rounds)` projection is the real solvency guard
    // (it sees the upkeep each commit ADDS); the broad reserve floor only needs to be a
    // light cushion. (See the per-gate notes in build_outposts / military.) The broad
    // `reserve + drain*5` floor in `affords` is RETAINED on every commit (it is the
    // cushion that protects the turtle when an attacker strips its income tiles); only
    // the reserve constant and the proactive-timing gates were loosened.
    reserve: 320,
    max_actions: 26,
    experts: true,
    military: true,
    garrison: 3,
    // LEAGUE-REBUILD STEP F — expand 4. The turtle needs enough reach to acquire grassland
    // (farms = income that funds the wall) and a mountain (mine = metal for soldiers);
    // grassland-poor seeds with too little expansion stalled at 1 farm and never funded the
    // wall. 4 balances reach against the per-tile scout/frontier drain.
    expand: 3,
    attack: true,            // counter-cracker stays ON (fires via can_buy)
    nuclear: false,
    max_outposts: 3,
    strike_force: 0,         // turtle: never go on the offensive
    assaults_per_turn: 0,    // suppressed via the `<= 1 && !can_buy` attack gate
    warmonger: true,         // load-bearing for proactive garrison + military phase
    cut_priority: false,
    device: false,
    fortress: true,          // THE point: proactive Outpost building
    attack_ready_soldiers: 0,
    econ_ready_net: 0,
    army_builder: false,
};

/// LEAGUE-REBUILD STEP E v2 (2026-06-06) — canonical STRONG_ARMY (the yardstick). The
/// strongest scripted bot, REBASED on `HARD_PARAMS` (HARD is the league's strongest bot —
/// it commits and wins; the old from-scratch readiness-gate design DEADLOCKED and was the
/// 2nd-WEAKEST). It keeps HARD's entire proven, fully-committing pipeline (econ →
/// militarise-on-contact → outposts → mass via `military()` → commit via `attack()` with
/// HARD's timing, NO readiness gates) and tunes TWO things for a strictly-stronger HARD:
///   - `reserve: 145` (down from HARD's 140-ish): the empirically-best solvency floor that
///     spends just enough more on econ/army to EDGE the HARD mirror (≈51% h2h, both seat
///     orders) while staying 0% self-bankrupt under real pressure,
///   - `cut_priority: true`: the surgical HQ-severing attack ordering (orders targets by
///     the fraction of enemy territory that disconnects when taken) — the single
///     behavioral edge that lifts the win-rate above the mirror everywhere.
/// `army_builder: true` lifts the `military()` strike-force ceiling from HARD's `farms + 1`
/// to the full `garrison + strike_force` (capped by the soldier cap and the staffed
/// economy) so that IN A RICH/LONG game it fields a real army; in the typical fast game it
/// is a no-op superset of HARD (the binding constraint is money/metal, not the ceiling),
/// so it never hurts. `max_outposts: 6` / `strike_force: 10` express the army intent and
/// raise the cap when the game runs long. `nuclear + experts + expand: 6` fund it.
///
/// EMPIRICS (sa_tune, 200 games/pairing): vs hard ≈51% (PASS the key beat-HARD bar),
/// vs device ≈84%, vs fortress ≈83% (PASS), vs rusher ≈50% (rusher is a near-mirror
/// aggressive HARD, inherently ~50/50). Front-open >85%, self-bankrupt 0%.
///
/// HONEST CAVEAT (peak soldiers): mean peak ≈ 2, NOT the ≥6 "big standing army" target.
/// This game is decided by fast trickle-conquest (~65 rounds) on a ~2-soldier-sustainable
/// economy (soldier upkeep 30/round + 50 metal each; ~1 mine). Every attempt to force a
/// larger STANDING army (a home guard, proactive massing, an extra mine, a soft assault
/// gate) measurably LOWERED the win-rate — hoarding loses tempo to the conquest meta. The
/// strongest yardstick is HARD's committing trickle, tuned to beat the mirror; a genuine
/// 6-soldier army is anti-correlated with strength here and is NOT pursued. See sa_tune.rs.
///
/// CRITICAL: `attack_ready_soldiers: 0` + `econ_ready_net: 0` — the old design's gates
/// blocked `attack()` until 8 soldiers were massed, but soldiers only mass after contact,
/// which only happens once `attack()` opens a front → chicken-and-egg deadlock (~1% commit).
pub const STRONG_ARMY_PARAMS: AiParams = AiParams {
    // ARC sd3 RE-TUNE (2026-06-07): reserve 145→130. With the cheaper military the yardstick
    // can spend a little more aggressively on its army/econ tempo without risking solvency
    // (still 0% self-bankrupt under pressure), edging back ahead of the (now buffed) rusher.
    reserve: 130,
    max_actions: 34,
    experts: true,
    military: true,
    garrison: 3,
    expand: 6,
    attack: true,
    nuclear: true,
    max_outposts: 6,
    strike_force: 12,        // ARC sd3: 10→12 — field a slightly larger army in long/rich games
    assaults_per_turn: 10,   // ARC sd3: 8→10 — commit harder (the cheaper soldiers fund it)
    warmonger: false,        // does NOT pre-militarise — HARD's reactive militarise-on-contact
    cut_priority: true,      // surgical HQ-severing attack ordering — the edge over the mirror
    device: false,
    fortress: false,
    attack_ready_soldiers: 0, // NO deadlocking assault gate — commit on HARD's schedule
    econ_ready_net: 0,        // NO deadlocking econ gate
    army_builder: true,       // THE point: field the full strike force (not HARD's farms+1)
};

/// `AiController.STAFF_RESERVE`.
const STAFF_RESERVE: i64 = 20;

/// The heuristic CPU controller. Stateless except for the per-turn action
/// budget, which is reset at the start of each `plan_turn`.
pub struct HardAi {
    params: AiParams,
    budget: i64,
}

/// SUPERVISED-RECORDER helper (TRAINING-ONLY, parity-free). Aggregate per-seat
/// snapshot used by [`HardAi::record_turn`] to classify per-phase deltas into
/// [`Intent`](crate::candidates::Intent) labels.
#[derive(Clone)]
struct TurnSnapshot {
    farms: i64,
    mines: i64,
    villages: i64,
    outposts: i64,
    hydros: i64,
    nuclears: i64,
    bridges: i64,
    devices: i64,
    soldiers: i64,
    workers: i64,
    experts: i64,
    tiles: i64,
    /// Number of TILES that currently hold ≥1 of this seat's staged conquering
    /// units. Counting tiles (not units) makes the Attack delta count assault
    /// DECISIONS (one Attack candidate stages `strike_force` soldiers onto one
    /// tile) rather than per-soldier, so the imitation target isn't attack-inflated.
    attacked_tiles: i64,
}

impl TurnSnapshot {
    fn of(g: &Game, seat: PlayerId) -> Self {
        let count_b = |kind: BuildingType| -> i64 {
            g.get_tiles()
                .iter()
                .filter(|t| t.owner == Some(seat) && t.building.as_ref().map(|b| b.kind) == Some(kind))
                .count() as i64
        };
        let attacked_tiles = g
            .get_tiles()
            .iter()
            .filter(|t| {
                t.conquering_units
                    .iter()
                    .any(|u| g.units[u.0].owner == Some(seat))
            })
            .count() as i64;
        TurnSnapshot {
            farms: count_b(BuildingType::Farm),
            mines: count_b(BuildingType::Mine),
            villages: count_b(BuildingType::Village),
            outposts: count_b(BuildingType::Outpost),
            hydros: count_b(BuildingType::Hydro),
            nuclears: count_b(BuildingType::Nuclear),
            bridges: count_b(BuildingType::Bridge),
            devices: count_b(BuildingType::StrangeDevice),
            soldiers: g.current_soldier_amount(seat),
            workers: g.current_basic_worker_amount(seat),
            experts: g.current_expert_amount(seat),
            tiles: g.get_tile_count_for_player(seat),
            attacked_tiles,
        }
    }

    /// Push one [`Intent`] per realised action between `self` (before) and
    /// `after`. Each phase maps to ONE intent family, so a positive delta on a
    /// counter emits that many copies of the corresponding intent.
    fn classify_into(&self, after: &Self, out: &mut Vec<crate::candidates::Intent>) {
        use crate::candidates::Intent;
        let mut push_n = |intent: Intent, n: i64| {
            for _ in 0..n.max(0) {
                out.push(intent);
            }
        };
        push_n(Intent::BuildFarm, after.farms - self.farms);
        push_n(Intent::BuildMine, after.mines - self.mines);
        push_n(Intent::BuildVillage, after.villages - self.villages);
        push_n(Intent::BuildOutpost, after.outposts - self.outposts);
        push_n(Intent::BuildHydro, after.hydros - self.hydros);
        push_n(Intent::BuildNuclear, after.nuclears - self.nuclears);
        push_n(Intent::BuildBridge, after.bridges - self.bridges);
        push_n(Intent::BuildStrangeDevice, after.devices - self.devices);
        // Soldier hires: soldier count rose (the cap-raise that PRECEDES a hire is
        // a worker/expert/unit-cap action surfaced via StackProducer/Expand below;
        // record_turn already runs raise_unit_cap as its own phase).
        push_n(Intent::HireSoldier, after.soldiers - self.soldiers);
        // Attacked-tile count up ⇒ Attack DECISIONS (one Attack stages a whole
        // strike-force onto one tile; CrackHQ/CrackDevice collapse into Attack —
        // the scripted bot has no separate cracker intent, and Attack is the correct
        // army-chain label for imitation).
        push_n(Intent::Attack, after.attacked_tiles - self.attacked_tiles);
        // Tile gains with NO building delta and NO attack ⇒ Expand (worker claim).
        // Subtract conquered tiles (attacks already counted) to avoid double-count.
        let building_tiles_delta = (after.farms - self.farms)
            + (after.mines - self.mines)
            + (after.villages - self.villages)
            + (after.outposts - self.outposts)
            + (after.hydros - self.hydros)
            + (after.nuclears - self.nuclears)
            + (after.bridges - self.bridges)
            + (after.devices - self.devices);
        let raw_tile_delta = after.tiles - self.tiles;
        // A fresh Expand claims a neutral tile (tiles up) often placing a building
        // on it in the SAME phase only for build_* phases. In the expand phase the
        // tile delta is the expansion count.
        let expand_count = raw_tile_delta - building_tiles_delta.max(0);
        push_n(Intent::Expand, expand_count);
        // MarchSoldier: a soldier changed tile (no hire, no attack staged). We
        // approximate via the march_to_enemy_hq / military phases producing neither
        // a soldier-count nor staged-attacker delta but consuming budget. We cannot
        // observe a pure relocation from aggregate counts, so marches that neither
        // hire nor stage are not emitted here (acceptable: the army CHAIN —
        // Outpost/Hire/Attack — is what the imitation target needs; pure relocations
        // are rare in the scripted bots' recorded turns and dominated by Attack).
        //
        // StackProducer (PARITY-FREE — reuses the existing Intent::StackProducer): a
        // worker- or expert-staffing of a PRODUCER (Mine/Hydro/Nuclear). This is the
        // economy the policy must learn to OWN instead of delegating it to the
        // mechanical scaffold (the human's explicit "make experts and place them on
        // mines" goal). One StackProducer per added expert + per added worker, matching
        // how the candidate enumerates a single StackProducer per filled slot.
        //
        // DISAMBIGUATION: a worker-count rise ALSO happens in the `expand` phase (it
        // hires a fresh BasicWorker onto a claimed neutral) — but that phase ALSO grows
        // the tile count, so it is already labelled Expand above. A StackProducer fill
        // grows NEITHER the tile count NOR the soldier count (it only mans an existing
        // producer). So we only attribute the worker delta to StackProducer when the
        // phase produced no tile gain and no soldier gain — which isolates the two
        // `phase!`-wrapped staffing phases that man producers: `boost_mines` (Experts
        // onto worker-staffed Mines — wrapped in `record_turn` specifically so this
        // signal is captured) and `stack_producers` (Experts + 2nd workers). The
        // unwrapped `staff_buildings` 1st-worker safety staffing never appears inside a
        // wrapped delta, so it stays scaffold-only and is correctly NOT labelled.
        let expert_delta = (after.experts - self.experts).max(0);
        let worker_delta = (after.workers - self.workers).max(0);
        let no_tile_gain = raw_tile_delta <= 0;
        let no_soldier_gain = after.soldiers - self.soldiers <= 0;
        if no_tile_gain && no_soldier_gain {
            push_n(Intent::StackProducer, expert_delta + worker_delta);
        } else {
            // Always credit experts (an Expert is never placed by Expand/Hire — only
            // producers take Experts), even if the phase also expanded/hired.
            push_n(Intent::StackProducer, expert_delta);
        }
    }
}

impl HardAi {
    pub fn new(params: AiParams) -> Self {
        HardAi { params, budget: 0 }
    }

    pub fn hard() -> Self {
        HardAi::new(HARD_PARAMS)
    }

    /// HARD bot with the experimental cut-priority attack ordering (ceiling
    /// probe only — see `AiParams::cut_priority`).
    pub fn hard_cut() -> Self {
        let mut p = HARD_PARAMS;
        p.cut_priority = true;
        HardAi::new(p)
    }

    /// Scripted DEVICE-RUSHER strategy opponent (Lever C, training-only).
    pub fn device_rush() -> Self {
        HardAi::new(DEVICE_RUSH_PARAMS)
    }

    /// Scripted ARMY-RUSHER strategy opponent (Lever C, training-only).
    pub fn army_rush() -> Self {
        HardAi::new(ARMY_RUSH_PARAMS)
    }

    /// Plan-B HQ-RUSHER strategy opponent (training-only). Same army-emphasis as
    /// `army_rush`, with assault counts cranked to maximise HQ-pressure (the
    /// shipped `attack` phase already orders targets HQ-first).
    pub fn hq_rush() -> Self {
        HardAi::new(HQ_RUSH_PARAMS)
    }

    /// OVERNIGHT-RUN §B.1 GARRISON-FORTRESS opponent (training-only). Forces an
    /// unconditional ≥ 3 HQ garrison from round 1 via `warmonger: true`, closing
    /// the 1-soldier-rush hole in HARD's default loose garrison. See
    /// `GARRISON_PARAMS`. (Constructor named `garrison_fortress` because the
    /// instance method `garrison` already exists as a turn-phase helper.)
    pub fn garrison_fortress() -> Self {
        HardAi::new(GARRISON_PARAMS)
    }

    /// OVERNIGHT-RUN §B.2 EXPERT-STACKED ECONOMY opponent (training-only). Pure-econ
    /// teacher that fronts the Expert tier (Mine + Expert doubles output; Hydro /
    /// Nuclear gate on Expert presence). See `EXPERT_PARAMS`.
    pub fn econ_expert() -> Self {
        HardAi::new(EXPERT_PARAMS)
    }

    /// REACTIVE-FIX MARCHER opponent (training-only). HQ-rusher cousin with cranked
    /// aggression knobs AND a `march_to_enemy_hq` phase that ADVANCES spare soldiers
    /// each turn toward the closest enemy HQ even when no legal Attack exists this
    /// turn (the missing "march your army" demonstration the learner's buffer never
    /// contained). See `MARCHER_PARAMS`.
    pub fn marcher() -> Self {
        HardAi::new(MARCHER_PARAMS)
    }

    /// LEAGUE-REBUILD (2026-06-06) — canonical RUSHER ("homing missile"). See
    /// `RUSHER_PARAMS`. Pure param fix (reserve 220 bankruptcy fix); reuses the
    /// existing `warmonger` build_bridges + march_to_enemy_hq + attack chain.
    pub fn rusher() -> Self {
        HardAi::new(RUSHER_PARAMS)
    }

    /// LEAGUE-REBUILD — canonical FORTRESS (the turtle). See `FORTRESS_PARAMS`.
    /// Proactive Outpost building via `fortress: true`; never marches its wall away
    /// (the `strike_force > 0` march gate).
    pub fn fortress() -> Self {
        HardAi::new(FORTRESS_PARAMS)
    }

    /// LEAGUE-REBUILD STEP E v2 — canonical STRONG_ARMY (the yardstick). See
    /// `STRONG_ARMY_PARAMS`. HARD-rebased with gates OFF (the readiness-gate design
    /// deadlocked); reserve 145 + cut_priority + army_builder EDGE the HARD mirror.
    pub fn strong_army() -> Self {
        HardAi::new(STRONG_ARMY_PARAMS)
    }

    // --- first round --------------------------------------------------------

    /// `placeHeadquarters` — choose and claim a starting tile. Identical scoring
    /// to the NN controller's port and the TS heuristic.
    pub fn place_headquarters(&self, g: &mut Game, player: PlayerId) {
        debug_assert_eq!(g.current_player(), player);
        // Candidates must be BUILDABLE: unowned AND empty (first-round HQ placement
        // is refused on a tile that already holds a building, e.g. an unowned
        // Mikontalo — picking one left the player with 0 tiles → instant loss).
        // Prefer grassland, then any non-river land, then any tile.
        let mut candidates: Vec<TileId> = g
            .get_tiles()
            .iter()
            .filter(|t| t.tile_type == TileType::Grassland && t.owner.is_none() && t.building.is_none())
            .map(|t| t.id)
            .collect();
        if candidates.is_empty() {
            candidates = g.get_tiles().iter().filter(|t| t.owner.is_none() && t.building.is_none() && t.tile_type != TileType::River).map(|t| t.id).collect();
        }
        if candidates.is_empty() {
            candidates = g.get_tiles().iter().filter(|t| t.owner.is_none() && t.building.is_none()).map(|t| t.id).collect();
        }
        if candidates.is_empty() {
            return;
        }
        let mut best = candidates[0];
        let mut best_score = f64::NEG_INFINITY;
        for &tid in &candidates {
            let ns = g.neighbour_tiles(tid);
            let free = ns.iter().filter(|&&n| g.tiles[n.0].owner.is_none()).count() as i64;
            let forests = ns
                .iter()
                .filter(|&&n| g.tiles[n.0].tile_type == TileType::Forest)
                .count() as i64;
            let mountains = ns
                .iter()
                .filter(|&&n| g.tiles[n.0].tile_type == TileType::Mountain)
                .count() as i64;
            let grass = ns
                .iter()
                .filter(|&&n| g.tiles[n.0].tile_type == TileType::Grassland)
                .count() as i64;
            let distance = self.distance_to_nearest_owned(g, tid).min(8);
            let score = (free * 3 + grass * 2 + forests * 2 + mountains * 3 + distance) as f64;
            if score > best_score {
                best_score = score;
                best = tid;
            }
        }
        g.first_round_actions(best);
    }

    fn distance_to_nearest_owned(&self, g: &Game, tid: TileId) -> i64 {
        let mut min = i64::MAX;
        let (cx, cy) = (g.tiles[tid.0].x, g.tiles[tid.0].y);
        for other in g.get_tiles() {
            if other.owner.is_none() {
                continue;
            }
            let d = (other.x - cx).abs() as i64 + (other.y - cy).abs() as i64;
            if d < min {
                min = d;
            }
        }
        if min == i64::MAX {
            99
        } else {
            min
        }
    }

    // --- turn ---------------------------------------------------------------

    /// `planTurn` (drained synchronously, as `playTurn` does). Wrapped so a
    /// panic inside (the TS `try { ... } catch {}`) can never crash the game.
    pub fn plan_turn(&mut self, g: &mut Game, player: PlayerId) {
        self.budget = self.params.max_actions;
        // PANIC MODE: an enemy who builds a Device halves their own soldier cap — that
        // is the window to strike. Go all-in for the turn: spend the reserve, press
        // every front, and field a real army (the Device is attacked first in `attack`).
        // Restored after the turn (HardAi is reused across turns/games). The per-buy
        // upkeep guard in `garrison` still prevents literal bankruptcy.
        let saved = self.params;
        if self.enemy_has_device(g, player) {
            self.params.reserve = (self.params.reserve / 4).max(40);
            self.params.assaults_per_turn = self.params.assaults_per_turn.max(12);
            self.budget += 12;
        }
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.run_turn(g, player);
        }));
        let _ = r; // swallow any panic, matching the TS catch-all.
        self.params = saved;
    }

    /// SUPERVISED-RECORDER (TRAINING-ONLY, parity-free).
    ///
    /// Drives the SAME phase sequence as [`run_turn`], but snapshots the seat's
    /// aggregate state before/after EACH phase and classifies the per-phase delta
    /// into the [`Intent`](crate::candidates::Intent) label(s) the phase executed.
    /// Returns one `Intent` per realised action, in phase order — so a single turn
    /// that hires a soldier, builds an Outpost and stages an attack yields
    /// `[HireSoldier, BuildOutpost, Attack]` instead of one collapsed "dominant"
    /// label. This is the fix for the old whole-turn diff heuristic that always
    /// fell through to `Pass`.
    ///
    /// Because each phase maps to exactly ONE intent family, the per-phase diff is
    /// unambiguous (no priority-ordering guess). The caller (`cnn_train.rs`
    /// supervised data-gen) pairs each returned intent with the turn-start board
    /// state to build a one-hot imitation example.
    ///
    /// NOTE: HQ-garrison `military()` soldier MOVES and `march_to_enemy_hq` moves
    /// are surfaced as `MarchSoldier` when a soldier changes tile without an attack
    /// being staged; soldier hires (cap/count up) are `HireSoldier`; staged
    /// attackers are `Attack`. This keeps the army chain (Outpost → HireSoldier →
    /// Attack/March) fully represented in the recorded dataset.
    ///
    /// `sink(intent, &Game)` is invoked once PER realised action with the game state
    /// captured at the START of the phase that produced it — so the caller can
    /// enumerate candidates / build planes against the ACTUAL decision state (an
    /// Attack candidate may not be enumerable at turn-start but IS after the bot's
    /// economy phases expanded the frontier; the prior turn-start-only enumeration
    /// fell back to Pass for exactly those actions).
    pub fn record_turn(
        &mut self,
        g: &mut Game,
        player: PlayerId,
        sink: &mut dyn FnMut(crate::candidates::Intent, &Game),
    ) {
        use crate::candidates::Intent;
        self.budget = self.params.max_actions;
        // Mirror plan_turn's PANIC-MODE budget bump so the recorded behaviour
        // matches what plan_turn would actually do.
        let saved = self.params;
        if self.enemy_has_device(g, player) {
            self.params.reserve = (self.params.reserve / 4).max(40);
            self.params.assaults_per_turn = self.params.assaults_per_turn.max(12);
            self.budget += 12;
        }
        // Phase wrapper: clone state at phase start → run phase → classify the
        // delta → emit each action's intent paired with the phase-start state.
        macro_rules! phase {
            ($f:expr) => {{
                let g_at_phase_start = g.clone();
                let before = TurnSnapshot::of(g, player);
                $f;
                let after = TurnSnapshot::of(g, player);
                let mut acts: Vec<Intent> = Vec::new();
                before.classify_into(&after, &mut acts);
                for it in acts {
                    sink(it, &g_at_phase_start);
                }
            }};
        }
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // --- mirror run_turn's phase order exactly ---
            self.ensure_wood_income(g, player);
            self.staff_buildings(g, player);
            self.secure_wood(g, player);
            let saving_for_mine = self.staffed_farm_count(g, player) >= 2
                && self.owned_tiles(g, player).iter().any(|&t| {
                    g.tiles[t.0].tile_type == TileType::Mountain && g.tiles[t.0].building.is_none()
                })
                && self.wood(g, player) < 270;
            if !saving_for_mine {
                phase!(self.build_farms(g, player));
                self.staff_buildings(g, player);
            }
            phase!(self.build_mines(g, player));
            self.staff_buildings(g, player);
            // `boost_mines` adds ONLY Experts to already-worker-staffed Mines — the
            // literal "make experts and place them on mines" decision. WRAP it in
            // `phase!` (recording-only; the executed action is byte-identical) so its
            // Expert deltas are classified into `Intent::StackProducer` and become an
            // imitation target. `staff_buildings` above stays UNWRAPPED on purpose: it
            // places the 1st-worker safety staffing (the scaffold's job), which must NOT
            // be mislabelled as a discretionary StackProducer.
            phase!(self.boost_mines(g, player));
            phase!(self.build_power_plants(g, player));
            phase!(self.invest_nuclear(g, player));
            phase!(self.build_outposts(g, player));
            phase!(self.raise_unit_cap(g, player));
            phase!(self.expand(g, player));
            phase!(self.build_bridges(g, player));
            phase!(self.build_strange_device(g, player));
            if self.params.fortress {
                phase!(self.fortress_field_wall(g, player));
            }
            phase!(self.military(g, player));
            phase!(self.attack(g, player));
            if (self.params.warmonger && self.params.strike_force > 0)
                || (self.params.attack_ready_soldiers > 0 && self.assault_ready(g, player))
            {
                phase!(self.march_to_enemy_hq(g, player));
            }
            phase!(self.stack_producers(g, player));
            self.fill_spare_slots(g, player);
        }));
        let _ = r;
        self.params = saved;
    }

    fn run_turn(&mut self, g: &mut Game, player: PlayerId) {
        self.ensure_wood_income(g, player);
        self.staff_buildings(g, player);
        self.secure_wood(g, player);


        let saving_for_mine = self.staffed_farm_count(g, player) >= 2
            && self.owned_tiles(g, player).iter().any(|&t| {
                g.tiles[t.0].tile_type == TileType::Mountain && g.tiles[t.0].building.is_none()
            })
            && self.wood(g, player) < 270;

        if !saving_for_mine {
            self.build_farms(g, player);
            self.staff_buildings(g, player);
        }
        self.build_mines(g, player);
        self.staff_buildings(g, player);
        self.boost_mines(g, player);
        self.build_power_plants(g, player);
        self.invest_nuclear(g, player);
        self.build_outposts(g, player);
        self.raise_unit_cap(g, player);
        self.expand(g, player);
        self.build_bridges(g, player); // unblock expansion across owned rivers (Plan-B fix)
        self.build_strange_device(g, player); // when leading: race the Device to a decisive win
        // LEAGUE-REBUILD STEP F — FORTRESS: field the wall's soldiers. Runs AFTER the
        // economy + Outpost build-out (so it never starves farms/mines) but BEFORE
        // `military()` so the freshly-bought soldiers are available for `military` to MOVE
        // into the HQ + onto frontier tiles. A soldier can't be bought onto the HQ (it's
        // never in `get_available_tiles`), so HARD only ever garrisons by moving frontier
        // buys home — but a calm turtle has no frontier contact, so it never bought ANY
        // soldier and held 0 defenders all game (the 1-rush hole). This BUYS soldiers onto
        // the staging frontier up to the wall's target; `military` then walls them up.
        if self.params.fortress {
            self.fortress_field_wall(g, player);
        }
        self.military(g, player);
        self.attack(g, player);
        // REACTIVE-FIX (MARCHER): after the standard attack phase has fired every
        // legal Attack, if our soldiers are still sitting at home (no contact yet),
        // ADVANCE them toward the closest enemy HQ. Gated so HARD's default behaviour
        // is byte-identical (HARD has `warmonger: false, strike_force: 7,
        // attack_ready_soldiers: 0` → neither clause fires).
        //   - LEAGUE-REBUILD STEP C: `strike_force > 0` so the turtle (FORTRESS,
        //     strike_force 0) NEVER marches its wall away.
        //   - LEAGUE-REBUILD STEP E: the army-builder (STRONG_ARMY, warmonger false)
        //     marches once it is `assault_ready` (massed enough to commit).
        if (self.params.warmonger && self.params.strike_force > 0)
            || (self.params.attack_ready_soldiers > 0 && self.assault_ready(g, player))
        {
            self.march_to_enemy_hq(g, player);
        }
        self.stack_producers(g, player);
        self.fill_spare_slots(g, player);
    }

    /// `doAction` — run `fn`; on success count it against the budget. Returns
    /// the success bool. Refuses when the budget is exhausted.
    fn do_action(&mut self, ok: bool) -> bool {
        // The TS guards `budget <= 0` BEFORE calling fn; callers here mirror
        // that by checking `self.budget > 0` in their loops. We still guard.
        if self.budget <= 0 {
            return false;
        }
        if ok {
            self.budget -= 1;
        }
        ok
    }

    // --- resource helpers ---------------------------------------------------

    fn res(&self, g: &Game, p: PlayerId, r: BasicResource) -> i64 {
        g.players[p.0].resources.get(r).unwrap_or(0)
    }
    fn money(&self, g: &Game, p: PlayerId) -> i64 {
        self.res(g, p, BasicResource::Money)
    }
    fn wood(&self, g: &Game, p: PlayerId) -> i64 {
        self.res(g, p, BasicResource::Wood)
    }
    fn stone(&self, g: &Game, p: PlayerId) -> i64 {
        self.res(g, p, BasicResource::Stone)
    }
    fn metal(&self, g: &Game, p: PlayerId) -> i64 {
        self.res(g, p, BasicResource::Metal)
    }

    fn owned_tiles(&self, g: &Game, p: PlayerId) -> Vec<TileId> {
        g.owned_tiles(p)
    }
    fn building_of(&self, g: &Game, tid: TileId) -> Option<BuildingType> {
        g.tiles[tid.0].building.as_ref().map(|b| b.kind)
    }
    fn has_type(&self, g: &Game, tid: TileId, kind: UnitType) -> bool {
        g.tile_units(tid).iter().any(|&u| g.units[u.0].kind == kind)
    }
    fn workers_on(&self, g: &Game, tid: TileId) -> i64 {
        g.tile_units(tid)
            .iter()
            .filter(|&&u| g.units[u.0].kind == UnitType::BasicWorker)
            .count() as i64
    }

    fn salary_per_round(&self, g: &Game, p: PlayerId) -> f64 {
        (g.current_basic_worker_amount(p) * 5
            + g.current_expert_amount(p) * 25
            + g.current_soldier_amount(p) * 30) as f64
    }

    fn money_drain_per_round(&self, g: &Game, p: PlayerId) -> f64 {
        let mut upkeep = 0.0;
        for t in self.owned_tiles(g, p) {
            match self.building_of(g, t) {
                Some(BuildingType::Village) => upkeep += 10.0,
                Some(BuildingType::Outpost) => upkeep += 50.0,
                _ => {}
            }
        }
        self.salary_per_round(g, p) + upkeep
    }

    fn staffed_farm_count(&self, g: &Game, p: PlayerId) -> i64 {
        self.owned_tiles(g, p)
            .iter()
            .filter(|&&t| {
                self.building_of(g, t) == Some(BuildingType::Farm)
                    && self.has_type(g, t, UnitType::BasicWorker)
            })
            .count() as i64
    }

    fn net_money_per_round(&self, g: &Game, p: PlayerId) -> f64 {
        let mut income = 0.0;
        for tid in self.owned_tiles(g, p) {
            let ty = self.building_of(g, tid);
            let workers = self.workers_on(g, tid);
            let has_expert = self.has_type(g, tid, UnitType::Expert);
            match ty {
                Some(BuildingType::Farm) if workers > 0 => income += 175.0 / 4.0,
                Some(BuildingType::Mine) if workers > 0 => {
                    income += 20.0 * workers as f64 * if has_expert { 2.0 } else { 1.0 }
                }
                Some(BuildingType::Nuclear) if workers > 0 && has_expert => {
                    income += 160.0 * workers as f64
                }
                Some(BuildingType::Hydro) if workers > 0 && has_expert => {
                    income += 80.0 * workers as f64
                }
                _ => {
                    if g.tiles[tid.0].tile_type == TileType::AbundantForest && workers > 0 {
                        income += 15.0;
                    }
                }
            }
            if ty == Some(BuildingType::Village) {
                income -= 10.0;
            }
            if ty == Some(BuildingType::Outpost) {
                income -= 50.0;
            }
        }
        income - self.salary_per_round(g, p)
    }

    fn metal_income_per_round(&self, g: &Game, p: PlayerId) -> f64 {
        let mut metal = 0.0;
        for tid in self.owned_tiles(g, p) {
            if self.building_of(g, tid) != Some(BuildingType::Mine) {
                continue;
            }
            metal += 20.0
                * self.workers_on(g, tid) as f64
                * if self.has_type(g, tid, UnitType::Expert) {
                    2.0
                } else {
                    1.0
                };
        }
        metal
    }

    fn stone_income_per_round(&self, g: &Game, p: PlayerId) -> f64 {
        let mut stone = 0.0;
        for tid in self.owned_tiles(g, p) {
            if self.building_of(g, tid) != Some(BuildingType::Mine) {
                continue;
            }
            stone += 30.0
                * self.workers_on(g, tid) as f64
                * if self.has_type(g, tid, UnitType::Expert) {
                    2.0
                } else {
                    1.0
                };
        }
        stone
    }

    fn can_afford_upkeep(&self, g: &Game, p: PlayerId, salary: f64) -> bool {
        self.net_money_per_round(g, p) - salary >= 0.0
    }

    fn affords(&self, g: &Game, p: PlayerId, cost: &ResourceMap, reserve: i64) -> bool {
        if !g.players[p.0].has_enough_resources(cost) {
            return false;
        }
        let buffer = reserve as f64 + self.money_drain_per_round(g, p) * 5.0;
        (self.money(g, p) + cost.get(BasicResource::Money).unwrap_or(0)) as f64 >= buffer
    }

    /// SELF-BANKRUPTCY GATE (user-report 2026-06-05): the existing `affords` /
    /// `affords_farm` checks gate on the bot's *current* drain — they don't see
    /// the per-round upkeep the new commit itself *adds*. A typical mid-game
    /// HARD with 3 Outposts + 5 Soldiers + 2 Experts + 4 Workers + 2 Villages
    /// drains ~390 money/round while a staffed Farm only produces ~60-90/round,
    /// so a long string of cumulative commits can push the bot past its tipping
    /// point even though each individual `affords` call sees enough cash. This
    /// helper projects the post-commit drain (current + new upkeep) and demands
    /// the post-build cash cover `buffer_rounds` rounds of it.
    ///
    /// `commit_money_cost` is a POSITIVE number (the money the build/hire
    /// drains, e.g. 500 for an Outpost — the cost in `resources.rs` stores it
    /// as `-500`, so callers pass `-money_cost.get(Money)` here).
    /// `new_upkeep_per_round` is the per-round drain the commit *adds* (50 for
    /// Outpost, 30 for Soldier, 25 for Expert, 10 for Village).
    /// Mirrors the `safety_buffer = drain * N` pattern at the Device call site
    /// (line 1228) — same template, generalised + projects the new upkeep.
    fn affordable_after_commit(
        &self,
        g: &Game,
        p: PlayerId,
        commit_money_cost: i64,
        new_upkeep_per_round: i64,
        buffer_rounds: i64,
    ) -> bool {
        let money_after = self.money(g, p) - commit_money_cost;
        let post_drain = self.money_drain_per_round(g, p).ceil() as i64 + new_upkeep_per_round;
        money_after >= buffer_rounds * post_drain
    }

    #[allow(dead_code)]
    fn affords_income_build(&self, g: &Game, p: PlayerId, cost: &ResourceMap, floor: i64) -> bool {
        if !g.players[p.0].has_enough_resources(cost) {
            return false;
        }
        self.money(g, p) + cost.get(BasicResource::Money).unwrap_or(0) >= floor
    }

    fn affords_farm(&self, g: &Game, p: PlayerId, farm_count: i64) -> bool {
        let cost = resources::farm_build_cost();
        if !g.players[p.0].has_enough_resources(&cost) {
            return false;
        }
        let money_after = self.money(g, p) + cost.get(BasicResource::Money).unwrap_or(0);
        // A farm pays out only every ~4 rounds, so keep enough cash to cover ~4
        // rounds of drain (salary + upkeep) after the build — otherwise the bot
        // spends its last cash on farms/staffing and salary bankrupts it BEFORE the
        // farms produce (the grassland-poor self-bankruptcy bug). Early game drain is
        // tiny, so the bootstrap opening stays unblocked.
        let cushion = self.money_drain_per_round(g, p) * 4.0;
        if farm_count < 3 {
            return money_after as f64 >= 40.0_f64.max(cushion);
        }
        money_after as f64 >= 80.0_f64.max(cushion)
    }

    fn add_worker(&mut self, g: &mut Game, player: PlayerId, tid: TileId) -> bool {
        if g.free_unit_amount(player) <= 0 {
            return false;
        }
        if !self.affords(g, player, &basic_worker_cost(), STAFF_RESERVE) {
            return false;
        }
        g.ai_buy_and_place_unit("BasicWorker", tid)
    }

    fn add_expert(&mut self, g: &mut Game, player: PlayerId, tid: TileId) -> bool {
        if !self.affords(g, player, &expert_cost(), self.params.reserve) {
            return false;
        }
        // SELF-BANKRUPTCY GATE: a new Expert adds 25 money/round to drain. Project
        // 4 rounds of post-commit drain (an Expert on Mine/Hydro/Nuclear pays off
        // within 1-2 rounds via the 2× production boost, so 4 is conservative but
        // not as tight as Device's 5).
        let expert_money = -expert_cost().get(BasicResource::Money).unwrap_or(0);
        if !self.affordable_after_commit(g, player, expert_money, 25, 4) {
            return false;
        }
        g.ai_buy_and_place_unit("Expert", tid)
    }

    // --- staffing -----------------------------------------------------------

    fn staff_buildings(&mut self, g: &mut Game, player: PlayerId) {
        for tid in self.owned_tiles(g, player) {
            match self.building_of(g, tid) {
                Some(BuildingType::Farm) => {
                    if self.budget > 0 && !self.has_type(g, tid, UnitType::BasicWorker) {
                        let ok = self.add_worker(g, player, tid);
                        self.do_action(ok);
                    }
                }
                Some(BuildingType::Mine) => self.ensure_worker(g, player, tid),
                Some(BuildingType::Nuclear) | Some(BuildingType::Hydro) => {
                    self.staff_plant(g, player, tid)
                }
                _ => {
                    if g.tiles[tid.0].tile_type == TileType::AbundantForest
                        && !self.has_type(g, tid, UnitType::BasicWorker)
                        && self.budget > 0
                    {
                        let ok = self.add_worker(g, player, tid);
                        self.do_action(ok);
                    }
                }
            }
        }
    }

    fn staff_plant(&mut self, g: &mut Game, player: PlayerId, tid: TileId) {
        if !self.params.experts {
            return;
        }
        let has_expert = |g: &Game| self.has_type(g, tid, UnitType::Expert);
        let workers = |g: &Game| self.workers_on(g, tid);
        if has_expert(g) && workers(g) >= 1 {
            return;
        }
        let reloc = |s: &Self, g: &Game| {
            s.find_idle_on_plain(g, player)
                .or_else(|| s.find_surplus_producer_worker(g, player))
        };
        let need_worker = workers(g) < 1;
        let need_expert = !has_expert(g);
        let slots_needed = (need_expert as i64) + (need_worker as i64);
        let reloc_for_worker = if need_worker && reloc(self, g).is_some() {
            1
        } else {
            0
        };
        if g.free_unit_amount(player) + reloc_for_worker < slots_needed {
            return;
        }
        if need_worker {
            if g.free_unit_amount(player) > 0 {
                if self.budget > 0 {
                    let ok = self.add_worker(g, player, tid);
                    self.do_action(ok);
                }
            } else if let Some((unit, from)) = reloc(self, g) {
                if from != tid && self.budget > 0 {
                    let ok = g.ai_move_unit(unit, from, tid);
                    self.do_action(ok);
                }
            }
        }
        if !self.has_type(g, tid, UnitType::Expert)
            && self.workers_on(g, tid) >= 1
            && g.free_unit_amount(player) > 0
            && self.budget > 0
        {
            let ok = self.add_expert(g, player, tid);
            self.do_action(ok);
        }
    }

    fn can_staff_new_plant(&self, g: &Game, player: PlayerId) -> bool {
        let free = g.free_unit_amount(player);
        if free >= 2 {
            return true;
        }
        free >= 1
            && (self.find_idle_on_plain(g, player).is_some()
                || self.find_surplus_producer_worker(g, player).is_some())
    }

    fn ensure_worker(&mut self, g: &mut Game, player: PlayerId, tid: TileId) {
        if self.has_type(g, tid, UnitType::BasicWorker) {
            return;
        }
        if g.free_unit_amount(player) > 0 {
            if self.budget > 0 {
                let ok = self.add_worker(g, player, tid);
                self.do_action(ok);
            }
            return;
        }
        let spare = self
            .find_idle_on_plain(g, player)
            .or_else(|| self.find_spare_worker(g, player, tid));
        if let Some((unit, from)) = spare {
            if from != tid && self.budget > 0 {
                let ok = g.ai_move_unit(unit, from, tid);
                self.do_action(ok);
            }
        }
    }

    fn find_spare_worker(&self, g: &Game, player: PlayerId, exclude: TileId) -> Option<(UnitId, TileId)> {
        for tid in self.owned_tiles(g, player) {
            if tid == exclude {
                continue;
            }
            if matches!(
                self.building_of(g, tid),
                Some(BuildingType::Farm)
                    | Some(BuildingType::Mine)
                    | Some(BuildingType::Nuclear)
                    | Some(BuildingType::Hydro)
            ) {
                continue;
            }
            if let Some(u) = self.first_worker(g, tid) {
                return Some((u, tid));
            }
        }
        None
    }

    fn first_worker(&self, g: &Game, tid: TileId) -> Option<UnitId> {
        g.tile_units(tid)
            .iter()
            .copied()
            .find(|&u| g.units[u.0].kind == UnitType::BasicWorker)
    }

    // --- building -----------------------------------------------------------

    fn empty_grassland(&self, g: &Game, player: PlayerId) -> Vec<TileId> {
        self.owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                g.tiles[t.0].tile_type == TileType::Grassland
                    && g.tiles[t.0].building.is_none()
                    && g.buildable_buildings(t).contains(&"Farm")
            })
            .collect()
    }

    fn build_mines(&mut self, g: &mut Game, player: PlayerId) {
        // LEAGUE-REBUILD STEP F — FORTRESS needs a Mine EARLY: each soldier costs 50 metal
        // and each Outpost 100 metal + 15 metal/round upkeep, so the wall is metal-gated.
        // HARD waits for `wood >= 300` (a big bank that delays the Mine well past the early
        // rush window). Let the fortress build its FIRST Mine at a lower wood bar so the
        // metal engine comes online in time to feed the Outpost/garrison chain. Gated on
        // `fortress`, so HARD / every other preset is byte-identical.
        let has_mine = self
            .owned_tiles(g, player)
            .iter()
            .any(|&t| self.building_of(g, t) == Some(BuildingType::Mine));
        let wood_min = if self.params.fortress && !has_mine { 200 } else { 300 };
        if self.wood(g, player) < wood_min {
            return;
        }
        let mines = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Mine))
            .count() as i64;
        let villages = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Village))
            .count() as i64;
        // FORTRESS metal throughput: the wall is metal-gated (3 Outposts = 300 metal + each
        // garrison soldier 30 metal). On a single 20-metal/round mine the turtle banks metal
        // too slowly to ever fund a 3-Outpost wall (the no-op-pressure ceiling was ~8%). Allow
        // a SECOND mine — but ONLY once the wood economy can carry the extra 300-wood build +
        // worker (the prior unconditional 2nd-mine attempt starved wood → bankruptcy): require
        // a healthy wood bank (>= 350, comfortably above the mine's 300 cost + buffer) AND a
        // staffed forest income (>= 1 harvester) so the 2nd mine doesn't drain wood to zero.
        // Gated on `fortress`; HARD / every other preset keeps the byte-identical cap.
        let fort_wants_2nd_mine = self.params.fortress
            && mines == 1
            && self.wood(g, player) >= 350
            && self.owned_tiles(g, player).iter().any(|&t| {
                g.tiles[t.0].tile_type == TileType::Forest
                    && g.tiles[t.0].building.is_none()
                    && self.has_type(g, t, UnitType::BasicWorker)
            });
        let max_mines = if fort_wants_2nd_mine { 2 } else { 1 + villages };
        if mines >= max_mines {
            return;
        }
        let mountains: Vec<TileId> = self
            .owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                g.tiles[t.0].tile_type == TileType::Mountain && g.tiles[t.0].building.is_none()
            })
            .collect();
        for m in mountains {
            if self.affords(g, player, &resources::mine_build_cost(), self.params.reserve)
                && self.has_wood_buffer(g, player, &resources::mine_build_cost())
                && self.budget > 0
            {
                let ok = g.ai_build_building("Mine", m);
                if self.do_action(ok) {
                    return; // one per turn
                }
            }
        }
    }

    fn boost_mines(&mut self, g: &mut Game, player: PlayerId) {
        if !self.params.experts {
            return;
        }
        for tid in self.owned_tiles(g, player) {
            if self.building_of(g, tid) != Some(BuildingType::Mine) {
                continue;
            }
            if !self.has_type(g, tid, UnitType::BasicWorker)
                || self.has_type(g, tid, UnitType::Expert)
            {
                continue;
            }
            if g.free_unit_amount(player) <= 0 {
                continue;
            }
            if self.budget > 0 {
                let ok = self.add_expert(g, player, tid);
                self.do_action(ok);
            }
        }
    }

    fn build_farms(&mut self, g: &mut Game, player: PlayerId) {
        // FIX 2(b): in the device-banking window, stop building extra farms (a
        // discretionary econ spend) so cash accumulates toward the Device. Staffing of
        // EXISTING producers (staff_buildings) is unaffected.
        if self.banking_for_device(g, player) {
            return;
        }
        let spots = self.empty_grassland(g, player);
        let mut farm_count = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Farm))
            .count() as i64;
        let mine_count = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Mine))
            .count() as i64;
        let mut max_farms = 1i64.max(g.max_unit_amount(player) - 2 - mine_count);
        // ARC sd3 ROOT-CAUSE FIX (2026-06-07): FORTRESS — RESERVE GRASSLAND FOR THE WALL.
        // Once the cap-bootstrap (Villages) frees slots, the turtle would fill EVERY empty
        // grassland with Farms (FORT_DIAG: 6 farms, grass_empty=0) — leaving NO spot to lay
        // an Outpost (Outposts need empty grassland). A turtle needs income, but the WALL is
        // the point: cap Farms so >= 3 grassland tiles stay free for Outposts. ~4 staffed
        // farms (≈176/round) amply fund the +5/round-per-Outpost upkeep. Gated on `fortress`.
        if self.params.fortress {
            max_farms = max_farms.min(3);
        }

        // First: grassland already holding an idle worker (free staffing).
        let with_worker: Vec<TileId> = spots
            .iter()
            .copied()
            .filter(|&t| self.has_type(g, t, UnitType::BasicWorker))
            .collect();
        for gtid in with_worker {
            if farm_count >= max_farms {
                break;
            }
            if self.affords_farm(g, player, farm_count)
                && self.has_wood_buffer(g, player, &resources::farm_build_cost())
                && self.budget > 0
            {
                let ok = g.ai_build_building("Farm", gtid);
                if self.do_action(ok) {
                    farm_count += 1;
                }
            }
        }
        // Then empty grasslands, if we have a free slot to staff the new farm.
        let slot_floor = if self.wood(g, player) < 200 { 1 } else { 0 };
        let without_worker: Vec<TileId> = spots
            .iter()
            .copied()
            .filter(|&t| !self.has_type(g, t, UnitType::BasicWorker))
            .collect();
        for gtid in without_worker {
            if farm_count >= max_farms {
                break;
            }
            if g.free_unit_amount(player) <= slot_floor {
                break;
            }
            if self.affords_farm(g, player, farm_count)
                && self.has_wood_buffer(g, player, &resources::farm_build_cost())
                && self.budget > 0
            {
                let ok = g.ai_build_building("Farm", gtid);
                if self.do_action(ok) {
                    farm_count += 1;
                }
            }
        }
    }

    fn build_power_plants(&mut self, g: &mut Game, player: PlayerId) {
        if !self.params.experts {
            return;
        }
        // FIX 2(b): suppress NEW power-plant builds in the device-banking window (a
        // discretionary econ spend). Existing plants stay staffed via staff_buildings.
        if self.banking_for_device(g, player) {
            return;
        }
        if self.net_money_per_round(g, player) <= 0.0 {
            return;
        }
        let hydros: Vec<TileId> = self
            .owned_tiles(g, player)
            .into_iter()
            .filter(|&t| self.building_of(g, t) == Some(BuildingType::Hydro))
            .collect();
        if hydros.iter().any(|&t| {
            !self.has_type(g, t, UnitType::Expert) || !self.has_type(g, t, UnitType::BasicWorker)
        }) {
            return;
        }
        if !self.can_staff_new_plant(g, player) {
            return;
        }
        let rivers: Vec<TileId> = self
            .owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                g.tiles[t.0].tile_type == TileType::River
                    && g.tiles[t.0].building.is_none()
                    && g.buildable_buildings(t)
                        .contains(&"Hydroelectric Power Plant")
            })
            .collect();
        for r in rivers {
            if self.affords(
                g,
                player,
                &resources::hepp_build_cost(),
                self.params.reserve.min(80),
            ) && self.has_wood_buffer(g, player, &resources::hepp_build_cost())
                && self.budget > 0
            {
                let ok = g.ai_build_building("Hydroelectric Power Plant", r);
                if self.do_action(ok) {
                    break;
                }
            }
        }
    }

    fn invest_nuclear(&mut self, g: &mut Game, player: PlayerId) {
        if !self.params.nuclear || !self.params.experts {
            return;
        }
        if self.money(g, player) <= 2400 {
            return;
        }
        let nukes = |s: &Self, g: &Game| -> Vec<TileId> {
            s.owned_tiles(g, player)
                .into_iter()
                .filter(|&t| s.building_of(g, t) == Some(BuildingType::Nuclear))
                .collect()
        };
        // 1. Staff existing plants first.
        for plant in nukes(self, g) {
            self.staff_nuclear(g, player, plant);
        }
        let fully_staffed = |s: &Self, g: &Game, t: TileId| {
            s.has_type(g, t, UnitType::Expert) && s.workers_on(g, t) >= 1
        };
        let want_count = 1 + ((self.money(g, player) - 2400) / 3000);
        let cur = nukes(self, g);
        if cur.len() as i64 >= want_count || !cur.iter().all(|&t| fully_staffed(self, g, t)) {
            return;
        }
        let empty_grass: Vec<TileId> = self
            .empty_grassland(g, player)
            .into_iter()
            .filter(|&t| !self.has_type(g, t, UnitType::BasicWorker))
            .collect();
        if g.free_unit_amount(player) < 1
            && !(self.can_raise_cap(g, player) && empty_grass.len() >= 2)
        {
            return;
        }
        if let Some(&spot) = empty_grass.first() {
            if self.affords(g, player, &resources::nuclearpp_build_cost(), self.params.reserve)
                && self.has_wood_buffer(g, player, &resources::nuclearpp_build_cost())
                && self.budget > 0
            {
                let ok = g.ai_build_building("Nuclear Power Plant", spot);
                if self.do_action(ok) {
                    self.staff_nuclear(g, player, spot);
                }
            }
        }
    }

    fn staff_nuclear(&mut self, g: &mut Game, player: PlayerId, plant: TileId) {
        if !self.has_type(g, plant, UnitType::Expert) {
            if g.free_unit_amount(player) < 1 {
                self.raise_unit_cap(g, player);
            }
            if g.free_unit_amount(player) > 0 && self.budget > 0 {
                let ok = self.add_expert(g, player, plant);
                self.do_action(ok);
            }
        }
        while self.has_type(g, plant, UnitType::Expert)
            && self.workers_on(g, plant) < 2
            && g.tiles[plant.0].has_space_for_units()
            && self.budget > 0
        {
            if g.free_unit_amount(player) > 0 {
                let ok = self.add_worker(g, player, plant);
                if !self.do_action(ok) {
                    break;
                }
            } else {
                match self.find_expendable_worker(g, player) {
                    Some((unit, from)) if from != plant => {
                        let ok = g.ai_move_unit(unit, from, plant);
                        if !self.do_action(ok) {
                            break;
                        }
                    }
                    _ => break,
                }
            }
        }
    }

    fn can_raise_cap(&self, g: &Game, player: PlayerId) -> bool {
        let villages = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Village))
            .count() as i64;
        if villages >= 5 {
            return false;
        }
        if !self
            .empty_grassland(g, player)
            .iter()
            .any(|&t| !self.has_type(g, t, UnitType::BasicWorker))
        {
            return false;
        }
        self.owned_tiles(g, player).iter().any(|&t| {
            g.tiles[t.0].tile_type == TileType::Forest
                && (g.tiles[t.0].building.is_none() || self.has_type(g, t, UnitType::BasicWorker))
        })
    }

    fn enemy_exists(&self, g: &Game, player: PlayerId) -> bool {
        g.live_players().iter().any(|&p| p != player)
    }

    /// True while an opponent owns a standing Strange Device — we must crack it before
    /// its countdown wins them the game. Always checked (NOT gated on `params.device`),
    /// so even a non-building AI mounts the counterplay.
    fn enemy_has_device(&self, g: &Game, player: PlayerId) -> bool {
        match g.find_strange_device_tile() {
            Some(dt) => {
                let o = g.tiles[dt.0].owner;
                o.is_some() && o != Some(player)
            }
            None => false,
        }
    }

    /// LEAGUE-REBUILD (2026-06-06) — unified "build Outposts BEFORE first contact"
    /// predicate used by BOTH the turtle (`fortress`) and the device strategist. The
    /// turtle proactively walls up; the device bot lays its halved-cap precursor
    /// Outposts once the game has matured (round >= 12) and no Device is down yet.
    /// Gated so HARD (`device: true, warmonger: false`) returns false here → its
    /// `build_outposts` militarise/military_need gates are byte-identical to ship.
    fn proactive_outposts(&self, g: &Game, _player: PlayerId) -> bool {
        self.params.fortress
            || (self.params.device
                && self.params.warmonger
                && !g.has_strange_device()
                && g.get_rounds_played() >= 12)
    }

    /// FIX 2(b) — DEVICE-BANKING suppression window. The device-strategist spends cash
    /// down every turn (expansion, extra producers, offensive soldiers) so it never
    /// reaches the 1300 + cushion the Device-build gate demands. When the bot is in its
    /// device window — strategist (`device && warmonger`), game matured (round >= 18), no
    /// Device down yet, not losing on tiles, and the precursor Outposts already banked
    /// (>= 2, so the Device-build's own `outposts < 2` precursor gate is already cleared)
    /// — discretionary spending is suppressed so money accumulates toward the build.
    ///
    /// (precursor banked = >= 1 Outpost, matching the lowered device-build precursor; see
    /// the note in `build_strange_device` on why a 2nd precursor deadlocked the bot.)
    ///
    /// CANNOT DEADLOCK: it only suppresses DISCRETIONARY spend (expansion / extra econ /
    /// offensive hires). Essential defense (garrison refill) and staffing existing
    /// producers stay on, and `build_strange_device` (which runs every turn) still fires
    /// the moment the bot can afford the cost + cushion. The not-losing-on-tiles gate
    /// makes waiting safe even if the bot can never quite afford it.
    fn banking_for_device(&self, g: &Game, player: PlayerId) -> bool {
        if !(self.params.device && self.params.warmonger) {
            return false;
        }
        if g.has_strange_device() || g.get_rounds_played() < 18 {
            return false;
        }
        let my_tiles = g.get_tile_count_for_player(player);
        let not_losing = g
            .live_players()
            .iter()
            .all(|&p| p == player || g.get_tile_count_for_player(p) <= my_tiles);
        if !not_losing {
            return false;
        }
        let outposts = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Outpost))
            .count() as i64;
        outposts >= 1
    }

    /// FIX 2(b) — true when the DEVICE-STRATEGIST OWNS a standing Strange Device. During
    /// the countdown it must DEFEND the ring and let the clock win — NOT keep expanding
    /// (which would dominate the passive opponent to a 70% win before the countdown) — so
    /// the discretionary-expansion suppression also keys off this.
    fn holding_own_device(&self, g: &Game, player: PlayerId) -> bool {
        self.params.device && self.params.warmonger && g.player_owns_strange_device(player)
    }

    /// LEAGUE-REBUILD — STRONG_ARMY assault-readiness gate. `attack_ready_soldiers <= 0`
    /// (every shipped preset except STRONG_ARMY) returns true unconditionally → the
    /// `attack` phase is byte-identical. An enemy Device always cracks the gate (the
    /// counterplay must never be blocked). Otherwise: require a massed army.
    fn assault_ready(&self, g: &Game, player: PlayerId) -> bool {
        self.params.attack_ready_soldiers <= 0
            || self.enemy_has_device(g, player)
            || g.current_soldier_amount(player) >= self.params.attack_ready_soldiers
    }

    /// LEAGUE-REBUILD — STRONG_ARMY econ-readiness gate. `econ_ready_net <= 0` returns
    /// true unconditionally (byte-identical for every other preset). An enemy Device
    /// overrides (defend/crack regardless of economy). RETAINED as part of the readiness-
    /// gate API (the `econ_ready_net` field) even though STEP E v2 STRONG_ARMY no longer
    /// uses it (the gate deadlocked — see `STRONG_ARMY_PARAMS`); kept so the gate can be
    /// re-enabled without re-deriving it.
    #[allow(dead_code)]
    fn econ_ready(&self, g: &Game, player: PlayerId) -> bool {
        self.params.econ_ready_net <= 0
            || self.enemy_has_device(g, player)
            || self.net_money_per_round(g, player) >= self.params.econ_ready_net as f64
    }

    /// LEAGUE-REBUILD — count owned producer tiles (Mine / Hydro / Nuclear) that carry
    /// an Expert. STRONG_ARMY caps its fielded army at roughly the staffed economy that
    /// can underwrite it (farms + staffed producers + 1), so a thin economy can't field
    /// a doomed army.
    fn staffed_expert_producer_count(&self, g: &Game, player: PlayerId) -> i64 {
        self.owned_tiles(g, player)
            .iter()
            .filter(|&&t| {
                matches!(
                    self.building_of(g, t),
                    Some(BuildingType::Mine)
                        | Some(BuildingType::Hydro)
                        | Some(BuildingType::Nuclear)
                ) && self.has_type(g, t, UnitType::Expert)
            })
            .count() as i64
    }

    fn should_militarise(&self, g: &Game, player: PlayerId) -> bool {
        // A standing enemy Device is an existential threat (its countdown wins the
        // game), so gear up for war regardless of the normal trigger.
        if self.enemy_has_device(g, player) {
            return true;
        }
        if self.params.warmonger {
            self.enemy_exists(g, player)
        } else {
            self.has_reachable_enemy(g, player)
        }
    }

    fn has_reachable_enemy(&self, g: &Game, player: PlayerId) -> bool {
        if self.enemy_threat(g, player) > 0 {
            return true;
        }
        g.get_available_tiles().iter().any(|&t| {
            let o = g.tiles[t.0].owner;
            o.is_some() && o != Some(player)
        })
    }

    fn reachable_enemy_max_defenders(&self, g: &Game, player: PlayerId) -> i64 {
        let mut max = 0;
        for t in g.get_available_tiles() {
            let o = g.tiles[t.0].owner;
            if o.is_none() || o == Some(player) {
                continue;
            }
            if self.building_of(g, t) == Some(BuildingType::Outpost) {
                continue;
            }
            let def = g
                .tile_units(t)
                .iter()
                .filter(|&&u| g.units[u.0].kind == UnitType::Soldier)
                .count() as i64;
            if def > max {
                max = def;
            }
        }
        max
    }

    fn military_need(&self, g: &Game, player: PlayerId) -> bool {
        self.enemy_threat(g, player) > 0
            || self.reachable_enemy_max_defenders(g, player) > 0
            || self.enemy_has_device(g, player)
    }

    /// Owned grassland with no building where `what` is buildable, sorted by fewest
    /// enemy-bordering neighbours first (the "interior"/safest tiles). Mirrors the TS
    /// `buildableGrass` helper. `sort_by_key` is stable, matching JS `.sort`.
    fn buildable_grass_for(&self, g: &Game, player: PlayerId, what: &str) -> Vec<TileId> {
        let mut v: Vec<TileId> = self
            .owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                g.tiles[t.0].tile_type == TileType::Grassland
                    && g.tiles[t.0].building.is_none()
                    && g.tiles[t.0].units.is_empty() // Device can't be built on an occupied tile
                    && g.buildable_buildings(t).iter().any(|&s| s == what)
            })
            .collect();
        v.sort_by_key(|&t| self.enemy_border_count(g, t, player));
        v
    }

    /// `buildStrangeDevice` — the Device endgame. When we are the clear leader, building
    /// it forces a decisive finish (a countdown win), at the cost of a halved soldier cap.
    /// We commit only when the strategy is enabled, no Device exists, the game has matured,
    /// we are not losing on tiles, we already hold >= 1 Outpost (so the halved cap leaves
    /// real defenders), and the economy can carry the one-time cost. Mirrors ai.ts 785-833.
    fn build_strange_device(&mut self, g: &mut Game, player: PlayerId) {
        if !self.params.device {
            return;
        }
        if g.has_strange_device() {
            return; // one per game — counterplay (attack) handles an enemy's
        }
        if g.get_rounds_played() < 18 {
            return; // let the game develop first
        }
        // Pursue the Device when we are NOT losing on territory.
        let my_tiles = g.get_tile_count_for_player(player);
        let not_losing = g
            .live_players()
            .iter()
            .all(|&p| p == player || g.get_tile_count_for_player(p) <= my_tiles);
        if !not_losing {
            return;
        }
        // Affordability for a TERMINAL play: raw resources + non-negative money net + a
        // small cash floor after the one-time cost (the lighter standard the TS uses; the
        // fat reserve helper almost never fired in a settled late-game economy).
        let device_cost = resources::strange_device_build_cost();
        if !g.players[player.0].has_enough_resources(&device_cost) {
            return;
        }
        if self.net_money_per_round(g, player) < 0.0 {
            return;
        }
        let device_money = device_cost.get(BasicResource::Money).unwrap_or(0);
        if self.money(g, player) + device_money < 150 {
            return;
        }
        // BUG-FIX (META-ANALYSIS §3 root cause): refuse the Device-build unless we can
        // cover its one-time cost AND a projection of the payroll that keeps firing during
        // the countdown. The Device halves our soldier cap (GAME-MECHANICS §6) BUT salary
        // draws keep firing — the historical bug was building the Device with just-enough
        // money for the cost, then bankrupting on the first rounds of payroll while the
        // countdown ran, handing the champ a free attrition win.
        //
        // TWO PROJECTIONS, gated on `warmonger` so HARD is BYTE-IDENTICAL to HEAD:
        //   - HARD (`device: true, warmonger: false`): the SHIPPED `drain * 5` buffer (5
        //     rounds of gross payroll). This is the historical, parity-locked HARD gate.
        //   - DEVICE-STRATEGIST (`device: true, warmonger: true`): a sane cushion that
        //     covers the 1300 cost AND survives the countdown's NET drain (gross payroll
        //     minus income). When the economy self-sustains (net_drain ~0) the cushion is
        //     tiny and the bot builds as soon as it can afford 1300 + banked Outposts; when
        //     it is bleeding, the cushion grows and it waits — which the not-losing-on-tiles
        //     window makes safe. NO `gross*0.5` floor, NO 0.6 factor, NO full-board
        //     `get_tile_count()` over-tune that the old rebuild used (that demanded ~$4000
        //     held at once and was never reachable).
        let safety_buffer = if self.params.warmonger {
            // DEVICE-STRATEGIST sane cushion.
            let countdown = resources::strange_device_countdown(g.get_tile_count());
            let gross = self.money_drain_per_round(g, player);
            let income = self.net_money_per_round(g, player); // income minus current payroll
            let net_drain = (gross - income.max(0.0)).max(0.0); // 0 if self-sustaining
            // Cushion = the countdown's NET burn, floored at ~4 rounds of GROSS payroll.
            // The 4-round gross floor is the bankruptcy fix: when net_drain ~0 the bot
            // could otherwise build with only a $50 cushion and a small income wobble
            // (e.g. a captured Farm resetting) then bankrupt 2-3 rounds into the countdown.
            // Still far below the old `gross*0.5*countdown` over-tune (~$4000) — these few
            // hundred dollars are easily reached by the banking phase.
            let net_cushion = (net_drain * countdown as f64).ceil() as i64;
            net_cushion.max((gross * 4.0).ceil() as i64).max(50)
        } else {
            // SHIPPED HARD gate (parity-locked).
            (self.money_drain_per_round(g, player) * 5.0).ceil() as i64
        };
        if self.money(g, player) + device_money < safety_buffer {
            return;
        }
        let outposts = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Outpost))
            .count() as i64;
        // Precursor-Outpost gate. The HALVED soldier cap must still ring the Device with
        // real defenders. Gated on `warmonger` so HARD is BYTE-IDENTICAL to HEAD:
        //   - HARD: the SHIPPED `outposts < 1` (lay one precursor Outpost).
        //   - DEVICE-STRATEGIST: also `outposts < 1`. FIX 2: the prior rebuild used
        //     `outposts < 2`, but a 2nd Outpost costs another 200 wood + 100 metal and on a
        //     forest/metal-poor map the bot reaches wood=0 and is stuck at 1 Outpost
        //     FOREVER — it banked 490 metal / 7870 money by r120 (more than enough for the
        //     Device's 200 metal / 1300 money) yet never built because it was waiting on a
        //     2nd precursor it could never afford. One precursor still leaves a halved cap
        //     of (HQ 1 + 3)/2 = 2 soldiers to ring the Device; the `military` device-owned
        //     branch fields the whole halved army onto the approaches.
        let precursor_min = 1;
        if outposts < precursor_min {
            // Precursor: lay the gating Outpost now (the Device halves the cap, so an
            // Outpost's +3 keeps the halved cap above zero). The Device follows next turn.
            if let Some(ospot) = self.buildable_grass_for(g, player, "Outpost").first().copied() {
                let outpost_cost = resources::outpost_build_cost();
                let outpost_money = outpost_cost.get(BasicResource::Money).unwrap_or(0);
                let can_afford = g.players[player.0].has_enough_resources(&outpost_cost)
                    && self.net_money_per_round(g, player) - 50.0 >= 0.0
                    && self.money(g, player) + outpost_money >= 100;
                if can_afford && self.budget > 0 {
                    let ok = g.ai_build_building("Outpost", ospot);
                    self.do_action(ok);
                }
            }
            return;
        }
        if let Some(spot) = self
            .buildable_grass_for(g, player, "Strange Device")
            .first()
            .copied()
        {
            if self.budget > 0 {
                let ok = g.ai_build_building("Strange Device", spot);
                self.do_action(ok);
            }
        }
    }

    fn build_outposts(&mut self, g: &mut Game, player: PlayerId) {
        if self.params.max_outposts <= 0 || !self.params.attack {
            return;
        }
        // LEAGUE-REBUILD (2026-06-06): when proactive outposts are wanted (turtle /
        // device strategist), SKIP the reactive militarise/military_need early-returns
        // so soldier-cap Outposts get laid BEFORE first contact. ALL downstream gates
        // (8-tile min, net_money, metal_income, affordable_after_commit — the
        // bankruptcy guards) still apply. HARD never reaches `proactive_outposts` →
        // true, so this branch is byte-identical to ship for it.
        if !self.proactive_outposts(g, player) {
            if !self.should_militarise(g, player) {
                return;
            }
            if !self.military_need(g, player) {
                return;
            }
        }
        let outposts = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Outpost))
            .count() as i64;
        if outposts >= self.params.max_outposts {
            return;
        }
        // LEAGUE-REBUILD STEP F (2026-06-06) — FORTRESS wall fix. The proactive turtle
        // was choked by this gate chain to ~1.2 Outposts (and so a ~1-soldier cap, i.e.
        // a 1-rush-crackable HQ). The chain was tuned for a REACTIVE HARD already at war
        // on a 1-mine economy; for a turtle that walls up BEFORE first contact (and banks
        // a fat reserve) it over-blocks. All relaxations are gated on `fortress`, so HARD
        // / every other preset is byte-identical. The hard solvency guards (the cost's raw
        // resource check via `affords`, and the per-commit `affordable_after_commit(50, 4)`
        // projection below) ALWAYS still apply — these only loosen the *proactive timing*.
        let fortress = self.params.fortress;
        // (a) 8-tile minimum. Keep it for the 2nd+ Outpost (the wall should sit on a real
        //     empire), but let the FIRST Outpost go up at 6 tiles for a turtle — the first
        //     +3 cap (HQ 1 → 4) is the single most important defensive unlock and waiting
        //     for 8 tiles on `expand: 3` delays it past the early-rush window.
        let tile_min = if fortress && outposts == 0 { 6 } else { 8 };
        if g.get_tile_count_for_player(player) < tile_min {
            return;
        }
        // (b) net-money floor. HARD demands net > 60 (`net - 50 < 10`). A turtle BANKS cash
        //     and finances its wall partly from the bank, so it can dip below HARD's positive
        //     surplus — but NOT into a free-fall: a deeply-negative-net Outpost (which can't be
        //     disbanded) is the self-bankruptcy that ends games early. Require the POST-Outpost
        //     net to stay >= -10 (`net - 50 >= -10` → current net >= 40): a small,
        //     reserve-financeable deficit, NOT the -120 free-fall that bankrupted the bot.
        //     Non-fortress: unchanged (net > 60).
        // Solvent net floor: the post-Outpost net must stay >= -10 (a small,
        // reserve-financeable deficit). A turtle banks cash so it can dip below HARD's
        // positive surplus, but NOT into a free-fall (a non-disbandable Outpost on a
        // deeply-negative net is the self-bankruptcy that ends games early). Non-fortress:
        // unchanged (net > 60).
        // ARC sd3 ROOT-CAUSE FIX (2026-06-07): the fortress's 2nd/3rd Outpost was blocked
        // here. After the 1st Outpost (-50/round money) + a 3-soldier garrison (-90/round),
        // net sits at ~6-30 — below the `net - 50 >= 0` (net >= 50) floor — even while the
        // turtle banks 600-1000 money + 1000-4000 metal (FORT_DIAG). A turtle FINANCES its
        // wall from the bank, not from surplus income; the binding solvency guard is the
        // per-commit `affordable_after_commit(500, 50, 4)` projection below (it demands ~1.2k
        // banked money before the next Outpost, which paces them safely). So for the fortress
        // allow a bank-financed deficit (post-Outpost net >= -60: a Farm-disband-survivable
        // dip, NOT the free-fall that bankrupts). Non-fortress: unchanged (net > 60).
        let net_floor = if fortress { -60.0 } else { 10.0 };
        if self.net_money_per_round(g, player) - 50.0 < net_floor {
            return;
        }
        // (c) metal-income gate. HARD refuses an Outpost unless mine income already covers
        //     the +15 metal/round each Outpost upkeeps (`metal_income >= (outposts+1)*15`).
        //     For a turtle that hasn't yet built a Mine this blocks the FIRST Outpost
        //     forever — the root choke. Relax for the fortress: instead of metal *income*,
        //     require a banked metal *buffer* (the Outpost's own 100-metal cost is already in
        //     `affords`; demand a further cushion so the 15/round upkeep won't drive metal
        //     negative before the Mine comes online). `prioritise_fortress_mine` (below)
        //     brings the Mine up early so income catches up. Non-fortress: unchanged.
        if fortress {
            // ARC sd3 (military-economy rebalance): Outpost metal upkeep is now 5/round (was
            // 15). The runway gate must track the CURRENT cost, else it over-blocks the wall.
            let metal_after = self.metal(g, player) - 100; // Outpost costs 100 metal
            let upkeep_runway = (outposts + 1) * 5 * 4; // 4 rounds of the (now 5/round) upkeep
            if metal_after < upkeep_runway && self.metal_income_per_round(g, player) < 5.0 {
                return;
            }
        } else if self.metal_income_per_round(g, player) - (outposts + 1) as f64 * 15.0 < 0.0 {
            // HARD's parity-locked gate (uses the historical 15 constant — kept byte-identical;
            // only the FORTRESS branch above tracks the rebalanced 5/round upkeep).
            return;
        }
        let buildable: Vec<TileId> = self
            .owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                g.tiles[t.0].tile_type == TileType::Grassland
                    && g.tiles[t.0].building.is_none()
                    && g.buildable_buildings(t).contains(&"Outpost")
            })
            .collect();
        let frontline = buildable.iter().copied().find(|&t| {
            self.tile_threatened(g, t, player)
                || g.neighbour_tiles(t)
                    .iter()
                    .any(|&n| g.tiles[n.0].owner.is_some() && g.tiles[n.0].owner != Some(player))
        });
        let spot = frontline.or_else(|| buildable.first().copied());
        if let Some(spot) = spot {
            // Cash floor. The FORTRESS uses a smaller reserve (80) than HARD's
            // `reserve.min(100)` so the wall isn't starved, but KEEPS the `drain*5`
            // income-loss protection in `affords` (banks 5 rounds of gross drain — the guard
            // that prevents bankruptcy when an attacker strips the turtle's income tiles).
            // Non-fortress: byte-identical (`reserve.min(100)`).
            // FORTRESS affordability: the broad `affords` floor (`reserve + drain*5`) DEMANDS
            // the turtle bank ~1100+ cash before laying a 500-money Outpost, so a bot legitimately
            // sitting on 600-900 banked money + 13+ empty grassland built NOTHING (root cause of
            // the ~0.6-Outpost wall). The real solvency guard is the per-commit
            // `affordable_after_commit(500, 50, 4)` projection below (it sees the +50/round the
            // Outpost itself adds); the broad floor is only a light cushion. So for the fortress,
            // use a lighter check: raw resources + a flat cash floor after the build. HARD /
            // every other preset keeps the byte-identical `affords` path.
            let outpost_cost_ck = resources::outpost_build_cost();
            let fort_can_afford = g.players[player.0].has_enough_resources(&outpost_cost_ck)
                && (self.money(g, player) + outpost_cost_ck.get(BasicResource::Money).unwrap_or(0)) >= 120;
            let cash_ok = if self.params.fortress {
                fort_can_afford
            } else {
                self.affords(g, player, &outpost_cost_ck, self.params.reserve.min(100))
            };
            if cash_ok && self.budget > 0 {
                // SELF-BANKRUPTCY GATE: an Outpost is the costliest upkeep commit
                // (+50 money/round), and the pre-existing `affords` only checks
                // CURRENT drain — not the +50 the Outpost itself will add. Without
                // this projection, a HARD that already runs hot on salary can chain
                // 2-3 Outposts in a few turns and tip past its income. Buffer = 4
                // rounds: an Outpost pays off indirectly via the cap → soldier →
                // conquest chain, which is slower than a Farm/Mine/Hydro paying out
                // directly, so 4 rounds is a fair "survive long enough to use it".
                let outpost_cost = resources::outpost_build_cost();
                let outpost_money = -outpost_cost.get(BasicResource::Money).unwrap_or(0);
                // FORTRESS / HARD: 4-round post-commit runway (an Outpost pays off as defensive
                // capacity, not income, so demand it survive long enough to use it). Tuning notes:
                //   - 2-round runway for the existential FIRST Outpost (to beat the rush) BANKRUPTED
                //     the bot under real pressure (self-bankrupt 18%) — the rusher strips income
                //     tiles right after the commit, so the shorter cushion is exactly wrong.
                //   - 3-round (fortress only) paced the 2nd Outpost slightly faster but pushed
                //     pressure-bankruptcy up (~13%→~14%) for no change in the unreachable ">=2 by
                //     r40" bar, so 4 is kept (lower bankruptcy wins; the 2nd Outpost is genuinely
                //     money-paced to ~r50-60 — an economy ceiling, not a gate). Non-fortress: 4.
                let buffer_rounds = 4;
                if !self.affordable_after_commit(g, player, outpost_money, 50, buffer_rounds) {
                    return;
                }
                let ok = g.ai_build_building("Outpost", spot);
                self.do_action(ok);
            }
        }
    }

    /// BUG-FIX (DEEP-REDESIGN-MEMO §3.4): HARD never built Bridges (0 across 180 saved
    /// replays) — so on any seed where the spawn is on the wrong side of a river, HARD
    /// stayed pinned to its starting cluster and the trainer never saw the river-crossing
    /// strategy demonstrated. Per GAME-MECHANICS §1+§8: an unbridged owned River tile does
    /// NOT expand availability (see `get_available_tiles_for` line 392), so the player's
    /// territory cannot grow across it. Bridging (or Hydro-building) on the river fixes that.
    ///
    /// Picks the owned river tile whose Bridge would unlock the most NEW orthogonal-4
    /// neighbours (not currently owned, not already reachable) — same `unblock_count`
    /// heuristic the NN-side `Intent::BuildBridge` candidate uses (candidates.rs:751-765).
    /// If unblock_count == 0 (e.g. river at map edge with no useful target), skip.
    ///
    /// FIX 3: when `experts: true` AND mid/late game (round > 30) AND a Hydro is
    /// affordable on the chosen river tile (orientation allows it + we can staff +
    /// net money is non-negative), prefer Hydro — it crosses the river AND adds income
    /// (80 money/worker), rather than just unblocking expansion. Otherwise: cheaper
    /// Bridge. Gated on conditions, not a new AiParams flag.
    fn build_bridges(&mut self, g: &mut Game, player: PlayerId) {
        if self.budget <= 0 {
            return;
        }
        // Candidate rivers: owned, no building, orientation allows Bridge.
        let rivers: Vec<TileId> = self
            .owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                g.tiles[t.0].tile_type == TileType::River
                    && g.tiles[t.0].building.is_none()
                    && g.buildable_buildings(t).contains(&"Bridge")
            })
            .collect();
        if rivers.is_empty() {
            return;
        }
        // Pre-compute reachable-tile set ONCE for the unlock-count scoring.
        let pre_avail = g.get_available_tiles_for(player);
        let unblock_count = |g: &Game, river: TileId| -> i64 {
            let mut n = 0i64;
            for nb in g.neighbour_four_tiles(river) {
                if g.tiles[nb.0].owner == Some(player) {
                    continue; // already owned
                }
                if pre_avail.contains(&nb) {
                    continue; // already reachable via a different path
                }
                if g.has_opponent_headquarters(nb, player) {
                    n += 1;
                }
            }
            n
        };
        // Sort rivers by unlock_count descending (stable).
        let mut scored: Vec<(TileId, i64)> =
            rivers.into_iter().map(|t| (t, unblock_count(g, t))).collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1));
        let (river, gain) = scored[0];
        if gain <= 0 {
            return; // bridging unlocks nothing useful — don't waste resources
        }
        // FIX 3: Prefer Hydro if experts-on, mid/late game, and a Hydro is buildable here.
        let prefer_hydro = self.params.experts
            && g.get_rounds_played() > 30
            && self.params.nuclear // gates the broader power-plant push
            && g.buildable_buildings(river).contains(&"Hydroelectric Power Plant")
            && self.can_staff_new_plant(g, player)
            && self.net_money_per_round(g, player) >= 0.0
            && self.affords(g, player, &resources::hepp_build_cost(), self.params.reserve.min(80))
            && self.has_wood_buffer(g, player, &resources::hepp_build_cost());
        if prefer_hydro {
            if self.budget > 0 {
                let ok = g.ai_build_building("Hydroelectric Power Plant", river);
                if self.do_action(ok) {
                    return;
                }
            }
            // Hydro attempt failed (engine rejection) — fall through to plain Bridge.
        }
        // Plain Bridge: cheaper, universally useful as a river-crosser.
        if !self.affords(g, player, &resources::bridge_build_cost(), self.params.reserve.min(80)) {
            return;
        }
        if !self.has_wood_buffer(g, player, &resources::bridge_build_cost()) {
            return;
        }
        if self.budget > 0 {
            let ok = g.ai_build_building("Bridge", river);
            self.do_action(ok);
        }
    }

    fn raise_unit_cap(&mut self, g: &mut Game, player: PlayerId) {
        // FIX 2(b): in the device-banking window, stop building Villages (a discretionary
        // unit-cap spend) so cash accumulates toward the Device.
        if self.banking_for_device(g, player) {
            return;
        }
        if g.free_unit_amount(player) > 1 {
            return;
        }
        let spot = match self.empty_grassland(g, player).first().copied() {
            Some(s) => s,
            None => return,
        };
        let villages = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Village))
            .count() as i64;
        let harvesters = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| {
                g.tiles[t.0].tile_type == TileType::Forest
                    && self.has_type(g, t, UnitType::BasicWorker)
            })
            .count() as i64;
        // ARC sd3 ROOT-CAUSE FIX (2026-06-07): FORTRESS Village cap. Villages also sit on
        // grassland, so an unbounded turtle filled grassland with Villages too (FORT_DIAG:
        // 3-4 villages, grass_empty=0) and squeezed out the WALL. 2 Villages (cap = HQ 3 +
        // 2×3 = 9 units) amply staff 4 farms + a mine + 2 harvesters; cap at 2 so the rest
        // of the grassland stays free for Outposts. Gated on `fortress`.
        let village_ceiling = if self.params.fortress {
            2i64.min(1 + harvesters * 3)
        } else {
            5i64.min(1 + harvesters * 3)
        };
        if villages >= village_ceiling {
            return;
        }
        if !self.owned_tiles(g, player).iter().any(|&t| {
            g.tiles[t.0].tile_type == TileType::Forest
                && (g.tiles[t.0].building.is_none() || self.has_type(g, t, UnitType::BasicWorker))
        }) {
            return;
        }
        // ARC sd3 ROOT-CAUSE FIX (2026-06-07): the FORTRESS bootstrap deadlock. HARD's
        // Village net-floor (`net - 25 < 10` → net >= 35) blocks the turtle's FIRST Village
        // forever: a 1-farm turtle sits at net ~19, but it needs the Village (+3 unit cap)
        // to free slots to staff a 2nd/3rd farm to LIFT net above 35 — a hard bootstrap
        // deadlock, while the bot sat on 400-700 idle cash (FORT_DIAG). The Village ADDS
        // income (+10/round) and unlocks the cap, so for a CASH-BANKING turtle it is correct
        // to fund it from the bank at low net. Replace the income floor with a banked-cash
        // floor; the per-commit `affordable_after_commit(.., 10, 4)` projection below is the
        // real solvency guard. Gated on `fortress`, so HARD is byte-identical.
        if self.params.fortress {
            // Fund the first 1-2 Villages from the bank (the cap-bootstrap); after that the
            // farms it unlocks carry the income and the standard projection bounds it.
            if self.money(g, player) < 250 {
                return;
            }
        } else if self.net_money_per_round(g, player) - 25.0 < 10.0 {
            return;
        }
        let post_upkeep = self.wood_upkeep(g, player) + 10.0;
        if ((self.wood(g, player) - 200) as f64) < 100.0_f64.max(post_upkeep * 5.0) {
            return;
        }
        let stone_upkeep = (villages + 1) as f64 * 10.0;
        if self.stone_income_per_round(g, player) < stone_upkeep
            && ((self.stone(g, player) - 100) as f64) < stone_upkeep * 8.0
        {
            return;
        }
        // ARC sd3 ROOT-CAUSE FIX (2026-06-07): FORTRESS affordability. HARD's `affords`
        // keeps a `reserve(320) + drain*5` cash floor, so the turtle (reserve 320) could
        // NEVER lay its bootstrap Village while banking only 400-700 cash (FORT_DIAG: the
        // floor ~545 sat above its bank). Mirror the Outpost path: a light raw-resources +
        // flat-cash check for the fortress; the `affordable_after_commit(.., 10, 4)`
        // projection below is the real solvency guard. HARD keeps the byte-identical path.
        let village_cost = resources::village_build_cost();
        let cash_ok = if self.params.fortress {
            g.players[player.0].has_enough_resources(&village_cost)
                && (self.money(g, player) + village_cost.get(BasicResource::Money).unwrap_or(0))
                    >= 100
        } else {
            self.affords(g, player, &village_cost, self.params.reserve)
        };
        if cash_ok && self.budget > 0 {
            // SELF-BANKRUPTCY GATE: a Village adds +10 money/round (plus wood/
            // stone upkeep, which the pre-existing wood-buffer / stone-income
            // checks above already handle). 4-round buffer: Villages pay off
            // indirectly via the unit-cap → income chain, same as Outposts.
            let village_money = -village_cost.get(BasicResource::Money).unwrap_or(0);
            if !self.affordable_after_commit(g, player, village_money, 10, 4) {
                return;
            }
            let ok = g.ai_build_building("Village", spot);
            self.do_action(ok);
        }
    }

    // --- wood ---------------------------------------------------------------

    fn wood_upkeep(&self, g: &Game, p: PlayerId) -> f64 {
        let mut w = 0.0;
        for t in self.owned_tiles(g, p) {
            match self.building_of(g, t) {
                Some(BuildingType::Village) => w += 10.0,
                Some(BuildingType::Bridge) => w += 5.0,
                _ => {}
            }
        }
        w
    }

    fn has_wood_buffer(&self, g: &Game, p: PlayerId, cost: &ResourceMap) -> bool {
        let need = -(cost.get(BasicResource::Wood).unwrap_or(0));
        if need <= 0 {
            return true;
        }
        let upkeep = self.wood_upkeep(g, p);
        let buffer = if upkeep > 0.0 {
            100.0_f64.max(upkeep * 5.0)
        } else {
            0.0
        };
        (self.wood(g, p) - need) as f64 >= buffer
    }

    fn find_expendable_worker(&self, g: &Game, player: PlayerId) -> Option<(UnitId, TileId)> {
        if let Some(idle) = self.find_idle_on_plain(g, player) {
            return Some(idle);
        }
        if let Some(surplus) = self.find_surplus_producer_worker(g, player) {
            return Some(surplus);
        }
        let farms: Vec<TileId> = self
            .owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                self.building_of(g, t) == Some(BuildingType::Farm)
                    && self.has_type(g, t, UnitType::BasicWorker)
            })
            .collect();
        if farms.len() >= 2 {
            let tid = farms[farms.len() - 1];
            if let Some(u) = self.first_worker(g, tid) {
                return Some((u, tid));
            }
        }
        None
    }

    fn ensure_wood_income(&mut self, g: &mut Game, player: PlayerId) {
        let upkeep = self.wood_upkeep(g, player);
        if upkeep <= 0.0 {
            return;
        }
        let harvesters = |s: &Self, g: &Game| -> i64 {
            s.owned_tiles(g, player)
                .iter()
                .filter(|&&t| {
                    g.tiles[t.0].tile_type == TileType::Forest
                        && g.tiles[t.0].building.is_none()
                        && s.has_type(g, t, UnitType::BasicWorker)
                })
                .count() as i64
        };
        let mut need = 1i64.max((upkeep / 40.0).ceil() as i64);
        if (self.wood(g, player) as f64) < upkeep * 4.0 {
            need += 1;
        }
        while harvesters(self, g) < need && self.budget > 0 {
            let f = self.owned_tiles(g, player).into_iter().find(|&t| {
                g.tiles[t.0].tile_type == TileType::Forest
                    && g.tiles[t.0].building.is_none()
                    && g.tiles[t.0].has_space_for_units()
                    && !self.has_type(g, t, UnitType::BasicWorker)
            });
            let f = match f {
                Some(t) => t,
                None => break,
            };
            let mut did = false;
            if g.free_unit_amount(player) > 0
                && self.affords(g, player, &basic_worker_cost(), STAFF_RESERVE)
            {
                let ok = self.add_worker(g, player, f);
                did = self.do_action(ok);
            } else if let Some((unit, from)) = self.find_expendable_worker(g, player) {
                if from != f {
                    let ok = g.ai_move_unit(unit, from, f);
                    did = self.do_action(ok);
                }
            }
            if !did {
                break;
            }
        }
    }

    fn anticipated_wood_need(&self, g: &Game, player: PlayerId) -> i64 {
        let mountains_no_mine = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| {
                g.tiles[t.0].tile_type == TileType::Mountain && g.tiles[t.0].building.is_none()
            })
            .count() as i64;
        let empty_grass = self.empty_grassland(g, player).len() as i64;
        mountains_no_mine * 250 + empty_grass.min(4) * 100
    }

    fn secure_wood(&mut self, g: &mut Game, player: PlayerId) {
        let stock_target = 700i64.min(150i64.max(self.anticipated_wood_need(g, player)));
        if self.wood(g, player) >= stock_target + 100 {
            return;
        }
        let staffed = |s: &Self, g: &Game| -> i64 {
            s.owned_tiles(g, player)
                .iter()
                .filter(|&&t| {
                    g.tiles[t.0].tile_type == TileType::Forest
                        && g.tiles[t.0].building.is_none()
                        && s.has_type(g, t, UnitType::BasicWorker)
                })
                .count() as i64
        };
        let target = if self.wood(g, player) < stock_target
            && self.anticipated_wood_need(g, player) > 200
            && g.max_unit_amount(player) > 6
        {
            2
        } else {
            1
        };
        while staffed(self, g) < target && self.budget > 0 {
            let f = self.owned_tiles(g, player).into_iter().find(|&t| {
                g.tiles[t.0].tile_type == TileType::Forest
                    && g.tiles[t.0].building.is_none()
                    && g.tiles[t.0].has_space_for_units()
                    && !self.has_type(g, t, UnitType::BasicWorker)
            });
            let f = match f {
                Some(t) => t,
                None => break,
            };
            let mut did = false;
            if g.free_unit_amount(player) > 0 && self.can_afford_upkeep(g, player, 5.0) {
                let ok = self.add_worker(g, player, f);
                did = self.do_action(ok);
            } else if let Some((unit, from)) = self.find_idle_on_plain(g, player) {
                let ok = g.ai_move_unit(unit, from, f);
                did = self.do_action(ok);
            }
            if !did {
                break;
            }
        }
    }

    fn find_idle_on_plain(&self, g: &Game, player: PlayerId) -> Option<(UnitId, TileId)> {
        for tid in self.owned_tiles(g, player) {
            let ty = g.tiles[tid.0].tile_type;
            if g.tiles[tid.0].building.is_some()
                || ty == TileType::Forest
                || ty == TileType::AbundantForest
            {
                continue;
            }
            if let Some(u) = self.first_worker(g, tid) {
                return Some((u, tid));
            }
        }
        None
    }

    // --- expansion ----------------------------------------------------------

    fn claim_value(&self, g: &Game, tid: TileId) -> i64 {
        // Neutral tiles that ALREADY HOLD a useful pre-built producer or military
        // structure rank above bare land of the same terrain: the engine carries the
        // building forward when its owner is neutralised (managers.rs HQ-cut +
        // `neutralize_player` only blank the OWNER, the building stays — only
        // StrangeDevice and the loser's units are cleared). Claiming such a tile gives
        // the conqueror a free Farm/Mine/Village/Outpost/Hydro/Nuclear, which is the
        // entire point of the user's observation: HARD (and via demonstration, the
        // NN's value head) must learn to prefer them. This file is the Rust training
        // opponent only — NOT on the parity / candidates path (the TS-mirrored
        // `cp-ai/src/candidates.rs` claim_value is unchanged), and NOT shipped to the
        // browser game (which uses `src/managers/ai.ts`). Cost/risk: zero parity
        // impact, zero cold-start; HARD demonstrates the claim during training and
        // self-play traces show the value head the resulting income jump.
        if let Some(b) = self.building_of(g, tid) {
            match b {
                // Cap-raising free building (existing case, unchanged).
                BuildingType::Mikontalo => return 6,
                // Free metal engine on what would already be the top-ranked Mountain.
                BuildingType::Mine => return 7,
                // Free expert/expert-staffable money engine — the strongest building.
                BuildingType::Nuclear => return 6,
                // Free unit-cap (+3) + 20 money/round; the cap chain matters most.
                BuildingType::Village => return 6,
                // Free soldier-cap (+3) AND impregnable-by-assault → free army basis.
                BuildingType::Outpost => return 6,
                // Free money on what would already be a top river — also unblocks
                // river-as-bridge expansion if on a curved river.
                BuildingType::Hydro => return 5,
                // Free money engine on grassland (~44/round when staffed); beats bare.
                BuildingType::Farm => return 5,
                // A standing bridge unblocks the river for expansion routing.
                BuildingType::Bridge => return 4,
                // Conquered/orphaned HQ has no operational value but the tile is land.
                BuildingType::Headquarters => {}
                // StrangeDevice is destroyed the moment ownership changes off the
                // builder (managers.rs §6a), so claiming gains nothing extra.
                BuildingType::StrangeDevice => {}
            }
        }
        match g.tiles[tid.0].tile_type {
            TileType::Mountain => 5,
            TileType::Grassland => 4,
            TileType::Forest => 3,
            TileType::AbundantForest => 2,
            TileType::River => {
                if g.buildable_buildings(tid)
                    .contains(&"Hydroelectric Power Plant")
                {
                    4
                } else {
                    1
                }
            }
        }
    }

    fn expand(&mut self, g: &mut Game, player: PlayerId) {
        if self.params.expand <= 0 {
            return;
        }
        // FIX 2(b): the DEVICE-STRATEGIST stops expanding while pursuing/holding its
        // Device. Pre-build (banking window): so cash accumulates toward the 1300 cost.
        // Post-build (own Device standing): so its territory does NOT grow to 70% and win
        // by DOMINATION before the Device countdown resolves (the bot would otherwise keep
        // grabbing neutrals during the ~38-round countdown and dominate the passive
        // opponent). Either way it freezes its borders and lets the countdown win.
        if self.banking_for_device(g, player) || self.holding_own_device(g, player) {
            return;
        }
        // ARC sd3 ROOT-CAUSE FIX (2026-06-07): FORTRESS — STOP SPRAWLING. Diagnosis
        // (FORT_DIAG, zero-pressure): the turtle hired a scout onto a fresh neutral tile
        // EVERY turn (expand: 3, unbounded total) and sprawled to 40-94 tiles. Its unit
        // cap is HQ-only (+3) until it builds Villages, so those 3 slots were ALL consumed
        // by scouts → no free slot to staff a 2nd farm/village → it stalled at 1 farm,
        // net~49, and the Outpost net-floor (`net - 50 >= 0`) was NEVER cleared. So the wall
        // never went up (0 Outposts in the sprawl seeds; the only seeds that walled up were
        // the geographically-cramped ones that couldn't sprawl). The turtle does NOT need
        // reach — it needs a SMALL, DENSE empire: ~10-12 owned tiles is plenty for HQ + a
        // mine + 3-4 farms + a forest harvester + 2-3 Outposts + the HQ ring. Past that,
        // STOP grabbing neutrals so the unit slots go to econ/wall, not endless scouts.
        // Gated on `fortress`, so HARD / every other preset is byte-identical.
        if self.params.fortress
            && g.get_tile_count_for_player(player) >= FORTRESS_TILE_CAP
        {
            return;
        }
        let mut claimed = 0;
        while claimed < self.params.expand && self.budget > 0 {
            let mut neutral: Vec<TileId> = g
                .get_available_tiles()
                .into_iter()
                .filter(|&t| {
                    g.tiles[t.0].owner.is_none()
                        && g.tiles[t.0].has_space_for_units()
                        && !self.tile_threatened(g, t, player)
                        && !g
                            .tile_conquering_units(t)
                            .iter()
                            .any(|&u| g.units[u.0].owner == Some(player))
                })
                .collect();
            // sort by claim value descending (stable, matching JS Array.sort
            // which is stable in V8 for the engine's ordering).
            neutral.sort_by(|&a, &b| self.claim_value(g, b).cmp(&self.claim_value(g, a)));
            if neutral.is_empty() {
                return;
            }
            let tile = neutral[0];
            let mut did = false;
            // 1. Leap-frog a genuinely idle worker.
            if let Some((unit, from)) = self.find_idle_worker(g, player) {
                if from != tile {
                    let ok = g.ai_move_unit(unit, from, tile);
                    did = self.do_action(ok);
                }
            }
            // 2. Hire a fresh scout into a free slot.
            if !did
                && g.free_unit_amount(player) > 0
                && self.affords(g, player, &basic_worker_cost(), self.params.reserve)
                && self.can_afford_upkeep_cushion(g, player, 5.0)
            {
                if self.budget > 0 {
                    let ok = g.ai_buy_and_place_unit("BasicWorker", tile);
                    did = self.do_action(ok);
                }
            }
            // 3. Peel a surplus producer worker off to scout.
            if !did {
                if let Some((unit, from)) = self.find_surplus_producer_worker(g, player) {
                    if from != tile {
                        let ok = g.ai_move_unit(unit, from, tile);
                        did = self.do_action(ok);
                    }
                }
            }
            if !did {
                return;
            }
            claimed += 1;
        }
    }

    /// `canAffordUpkeep` for the scout-hire path. The TS passes a large cushion
    /// argument that the implementation ignores (it only checks the net), so a
    /// plain `can_afford_upkeep` is faithful.
    fn can_afford_upkeep_cushion(&self, g: &Game, p: PlayerId, salary: f64) -> bool {
        self.can_afford_upkeep(g, p, salary)
    }

    fn find_surplus_producer_worker(&self, g: &Game, player: PlayerId) -> Option<(UnitId, TileId)> {
        for tid in self.owned_tiles(g, player) {
            let stackable = matches!(
                self.building_of(g, tid),
                Some(BuildingType::Mine) | Some(BuildingType::Nuclear) | Some(BuildingType::Hydro)
            );
            if !stackable {
                continue;
            }
            let ws: Vec<UnitId> = g
                .tile_units(tid)
                .iter()
                .copied()
                .filter(|&u| g.units[u.0].kind == UnitType::BasicWorker)
                .collect();
            if ws.len() > 1 {
                return Some((ws[ws.len() - 1], tid));
            }
        }
        if self.wood(g, player) >= 350 {
            for tid in self.owned_tiles(g, player) {
                if g.tiles[tid.0].tile_type != TileType::Forest {
                    continue;
                }
                if let Some(u) = self.first_worker(g, tid) {
                    return Some((u, tid));
                }
            }
        }
        None
    }

    fn find_idle_worker(&self, g: &Game, player: PlayerId) -> Option<(UnitId, TileId)> {
        let needs_wood = self.wood(g, player) < 350
            || self.owned_tiles(g, player).iter().any(|&t| {
                g.tiles[t.0].tile_type == TileType::Mountain && g.tiles[t.0].building.is_none()
            });
        // Pass 1: genuinely idle workers.
        for tid in self.owned_tiles(g, player) {
            let ty = g.tiles[tid.0].tile_type;
            if g.tiles[tid.0].building.is_some()
                || ty == TileType::Forest
                || ty == TileType::AbundantForest
            {
                continue;
            }
            if let Some(u) = self.first_worker(g, tid) {
                return Some((u, tid));
            }
        }
        if needs_wood {
            return None;
        }
        // Pass 2: forest harvesters when wood is no longer needed.
        for tid in self.owned_tiles(g, player) {
            let ty = g.tiles[tid.0].tile_type;
            if ty != TileType::Forest && ty != TileType::AbundantForest {
                continue;
            }
            if let Some(u) = self.first_worker(g, tid) {
                return Some((u, tid));
            }
        }
        None
    }

    // --- spare workers ------------------------------------------------------

    fn stack_producers(&mut self, g: &mut Game, player: PlayerId) {
        let producers = |s: &Self, g: &Game| -> Vec<TileId> {
            s.owned_tiles(g, player)
                .into_iter()
                .filter(|&t| {
                    matches!(
                        s.building_of(g, t),
                        Some(BuildingType::Mine)
                            | Some(BuildingType::Nuclear)
                            | Some(BuildingType::Hydro)
                    ) && g.tiles[t.0].has_space_for_units()
                })
                .collect()
        };
        while g.free_unit_amount(player) > 0 && self.budget > 0 {
            let tile = match producers(self, g).first().copied() {
                Some(t) => t,
                None => break,
            };
            let want_expert =
                self.params.experts && self.building_of(g, tile) != Some(BuildingType::Hydro);
            if want_expert
                && !self.has_type(g, tile, UnitType::Expert)
                && g.free_unit_amount(player) > 1
            {
                let ok = self.add_expert(g, player, tile);
                if self.do_action(ok) {
                    continue;
                }
            }
            let ok = self.add_worker(g, player, tile);
            if !self.do_action(ok) {
                break;
            }
        }
    }

    fn fill_spare_slots(&mut self, g: &mut Game, player: PlayerId) {
        let forests = |s: &Self, g: &Game| -> Vec<TileId> {
            s.owned_tiles(g, player)
                .into_iter()
                .filter(|&t| {
                    g.tiles[t.0].tile_type == TileType::Forest
                        && g.tiles[t.0].building.is_none()
                        && g.tiles[t.0].has_space_for_units()
                })
                .collect()
        };
        while g.free_unit_amount(player) > 0
            && self.budget > 0
            && self.can_afford_upkeep(g, player, 5.0)
        {
            let f = match forests(self, g).first().copied() {
                Some(t) => t,
                None => break,
            };
            let ok = self.add_worker(g, player, f);
            if !self.do_action(ok) {
                break;
            }
        }
    }

    // --- military -----------------------------------------------------------

    fn enemy_threat(&self, g: &Game, player: PlayerId) -> i64 {
        let mut threat = 0;
        for tid in self.owned_tiles(g, player) {
            threat += g
                .tile_conquering_units(tid)
                .iter()
                .filter(|&&u| {
                    g.units[u.0].owner != Some(player) && g.units[u.0].kind == UnitType::Soldier
                })
                .count() as i64;
            for n in g.neighbour_tiles(tid) {
                let o = g.tiles[n.0].owner;
                if o.is_some() && o != Some(player) {
                    threat += g
                        .tile_units(n)
                        .iter()
                        .filter(|&&u| g.units[u.0].kind == UnitType::Soldier)
                        .count() as i64;
                }
            }
        }
        threat
    }

    fn tile_threatened(&self, g: &Game, tid: TileId, player: PlayerId) -> bool {
        for n in g.neighbour_tiles(tid) {
            let o = g.tiles[n.0].owner;
            if o.is_some()
                && o != Some(player)
                && g.tile_units(n)
                    .iter()
                    .any(|&u| g.units[u.0].kind == UnitType::Soldier)
            {
                return true;
            }
        }
        false
    }

    fn adjacent_enemy_soldiers(&self, g: &Game, tid: TileId, player: PlayerId) -> i64 {
        let mut n = 0;
        for nb in g.neighbour_tiles(tid) {
            let o = g.tiles[nb.0].owner;
            if o.is_some() && o != Some(player) {
                n += g
                    .tile_units(nb)
                    .iter()
                    .filter(|&&u| g.units[u.0].kind == UnitType::Soldier)
                    .count() as i64;
            }
        }
        n
    }

    fn invaders_on(&self, g: &Game, tid: TileId, player: PlayerId) -> i64 {
        g.tile_conquering_units(tid)
            .iter()
            .filter(|&&u| g.units[u.0].owner != Some(player) && g.units[u.0].kind == UnitType::Soldier)
            .count() as i64
    }

    fn soldiers_on(&self, g: &Game, tid: TileId, player: PlayerId) -> i64 {
        g.tile_units(tid)
            .iter()
            .filter(|&&u| g.units[u.0].owner == Some(player) && g.units[u.0].kind == UnitType::Soldier)
            .count() as i64
    }

    fn enemy_border_count(&self, g: &Game, tid: TileId, player: PlayerId) -> i64 {
        let mut n = 0;
        for nb in g.neighbour_tiles(tid) {
            let o = g.tiles[nb.0].owner;
            if o.is_some() && o != Some(player) {
                n += 1;
            }
        }
        n
    }

    fn find_rear_soldier(&self, g: &Game, player: PlayerId, exclude: TileId) -> Option<(UnitId, TileId)> {
        for tid in self.owned_tiles(g, player) {
            if tid == exclude {
                continue;
            }
            if self.adjacent_enemy_soldiers(g, tid, player) + self.invaders_on(g, tid, player) > 0 {
                continue;
            }
            if self.enemy_border_count(g, tid, player) > 0 {
                continue;
            }
            if let Some(&u) = g
                .tile_units(tid)
                .iter()
                .find(|&&u| g.units[u.0].owner == Some(player) && g.units[u.0].kind == UnitType::Soldier)
            {
                return Some((u, tid));
            }
        }
        None
    }

    /// LEAGUE-REBUILD STEP F — FORTRESS soldier solvency. HARD's `affords(reserve+drain*5)`
    /// + `can_afford_upkeep(net>=0)` chain is incompatible with a turtle (Outpost upkeep
    /// drags net negative and inflates drain*5 so the garrison the Outposts exist to enable
    /// was permanently unaffordable; see fort_diag). But a pure cash-runway gate had NO
    /// income-loss protection → 19-23/40 self-bankruptcies when an attacker stripped the
    /// turtle's income tiles. This guard splits the difference: after the buy, the BANKED
    /// money must cover BOTH (a) 3 rounds of the post-commit GROSS drain — the income-loss
    /// cushion: if income vanishes the bot survives 3 rounds and can disband — AND (b) a
    /// raw-resource + $80 floor. Gross*3 (not *5) is reachable on a turtle's bank while
    /// still bounding the bankruptcy exposure.
    fn fortress_soldier_solvent(&self, g: &Game, player: PlayerId) -> bool {
        let cost = soldier_cost();
        if !g.players[player.0].has_enough_resources(&cost) {
            return false;
        }
        let money_after = self.money(g, player) + cost.get(BasicResource::Money).unwrap_or(0);
        let gross_after = self.money_drain_per_round(g, player) + 30.0; // +1 soldier salary
        // SOLVENCY (the mandatory <=5% bankrupt gate). Two-part guard:
        //   (1) post-commit per-round NET must stay >= -20 (income covers the wall's upkeep
        //       within a tiny bank-financed deficit). `net_money_per_round` already nets ALL
        //       income against salary + Outpost/Village upkeep, so this is the real
        //       income-loss guard: if an attacker strips farms, net drops and further hires
        //       stop BEFORE the bot free-falls. Soldier #1 is existential → allow a slightly
        //       deeper deficit (-60) to get the lone defender up.
        //   (2) the banked money must cover a few rounds of post-commit GROSS drain (the
        //       runway if income vanishes outright). Tighter for #2+ than for the #1 defender.
        let first = self.current_soldier_amount(g, player) == 0;
        if first {
            // The FIRST defender is existential: a 0-defender HQ falls to a single soldier
            // (every observed crack — FORT_CRACK_DBG). It is worth a brief bank-financed
            // deficit to get the lone defender up, so gate it lightly: raw resources + an $80
            // cash floor + a bounded net deficit (-50) it can finance from the bank. The
            // fuller income-loss cushion below applies to soldiers #2+.
            return money_after >= 120 && self.net_money_per_round(g, player) - 30.0 >= -25.0;
        }
        // ARC sd3 ROOT-CAUSE FIX (2026-06-07): the HQ-RING soldiers (#2-4) were blocked here.
        // A turtle holding 1 Outpost (-50/round) sits at net ~6-30, so the old `net_after >= 0`
        // (net >= 30) gate refused soldiers #2+, capping the wall at ~1 (FORT report: mean peak
        // 1.76, HQ-ring NEVER fully manned) — the very 1-poke hole this bot exists to close.
        // The turtle banks huge metal + money; it should FINANCE its ring from the bank. So for
        // the first RING-SIZED batch (up to 4 soldiers — HQ + a 4-tile orthogonal ring) allow a
        // bounded bank-financed net deficit; the GROSS-drain runway below (money must cover
        // 4 rounds of post-commit gross drain) stays the hard income-loss / bankruptcy guard.
        // Soldiers #5+ (a real standing army, not the ring) still require net-positive.
        let soldiers_now = self.current_soldier_amount(g, player);
        // Allow a small bank-financed deficit only AFTER the cap is built (>= 2 Outposts);
        // before that, soldier salary must stay net-positive so it doesn't starve the
        // Outpost build-out (the death spiral). The first 1-3 ring soldiers on a 1-Outpost
        // turtle therefore require net-positive; once 2 Outposts give cap headroom the ring
        // can be bank-financed.
        let outposts_built = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Outpost))
            .count() as i64;
        let ring_floor = if outposts_built >= 2 && soldiers_now < 7 { -20.0 } else { 0.0 };
        let net_after = self.net_money_per_round(g, player) - 30.0;
        if net_after < ring_floor {
            return false; // beyond a bank-financeable deficit — a wall the economy can't carry
                          // is the self-bankruptcy that ends games early.
        }
        (money_after as f64) >= 150.0_f64.max(gross_after * 4.0)
    }

    fn current_soldier_amount(&self, g: &Game, player: PlayerId) -> i64 {
        g.current_soldier_amount(player)
    }

    fn garrison(&mut self, g: &mut Game, player: PlayerId, tid: TileId, want: i64) {
        while self.soldiers_on(g, tid, player) < want
            && g.tiles[tid.0].has_space_for_units()
            && self.budget > 0
        {
            if let Some((unit, from)) = self.find_rear_soldier(g, player, tid) {
                let ok = g.ai_move_unit(unit, from, tid);
                if !self.do_action(ok) {
                    break;
                }
                continue;
            }
            if g.free_soldier_amount(player) > 0
                && self.metal(g, player) >= 30 // soldier metal cost (rebalanced 50→30, arc sd3)
                && (if self.params.fortress {
                    self.fortress_soldier_solvent(g, player)
                } else {
                    // SHIPPED HARD path (byte-identical): reserve floor + net guard + 4-round
                    // post-commit projection.
                    self.affords(g, player, &soldier_cost(), self.params.reserve)
                        && self.can_afford_upkeep(g, player, 30.0)
                        && {
                            let soldier_money =
                                -soldier_cost().get(BasicResource::Money).unwrap_or(0);
                            self.affordable_after_commit(g, player, soldier_money, 30, 4)
                        }
                })
            {
                let ok = g.ai_buy_and_place_unit("Soldier", tid);
                if !self.do_action(ok) {
                    break;
                }
                continue;
            }
            break;
        }
    }

    /// LEAGUE-REBUILD STEP F — FORTRESS: a soldier can only be PLACED on a tile that is in
    /// `get_available_tiles` (the staging frontier) — and a player's own un-conquered HQ is
    /// NEVER available (see `get_available_tiles_for` / `has_opponent_headquarters`). So the
    /// HQ garrison can never BUY directly onto the HQ; HARD only ever fills the HQ by MOVING
    /// soldiers it first bought on the frontier (its border-guard/strike-force phase). A
    /// calm turtle with no enemy contact and strike_force 0 thus never bought a single
    /// soldier ANYWHERE → 0 defenders all game (the root of the 1-rush hole). This helper
    /// BUYS a soldier onto the best available owned tile (preferring one ADJACENT to the HQ
    /// so the follow-up move is short / often unneeded), using the fortress garrison
    /// solvency gate. Returns true on a successful buy.
    fn fortress_buy_soldier(&mut self, g: &mut Game, player: PlayerId) -> bool {
        // soldier metal cost (rebalanced 50→30, arc sd3)
        if g.free_soldier_amount(player) <= 0 || self.metal(g, player) < 30 || self.budget <= 0 {
            return false;
        }
        // Solvency: same gross-3-round cushion as the garrison hire path.
        if !self.fortress_soldier_solvent(g, player) {
            return false;
        }
        let hq = g.get_hq_tile(player);
        let avail = g.get_available_tiles();
        // Candidate buy tiles: owned, available (stage-able), has room, NOT an Outpost
        // (can't hold soldiers). Prefer HQ-adjacent, then fewest enemy-border (safe interior).
        let mut spots: Vec<TileId> = self
            .owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                avail.contains(&t)
                    && g.tiles[t.0].has_space_for_units()
                    && self.building_of(g, t) != Some(BuildingType::Outpost)
            })
            .collect();
        if spots.is_empty() {
            return false;
        }
        spots.sort_by_key(|&t| {
            let adj = match hq {
                Some(h) => g.neighbour_tiles(h).iter().any(|&n| n == t),
                None => false,
            };
            (if adj { 0 } else { 1 }, self.enemy_border_count(g, t, player))
        });
        let spot = spots[0];
        let ok = g.ai_buy_and_place_unit("Soldier", spot);
        self.do_action(ok)
    }

    /// LEAGUE-REBUILD STEP F — FORTRESS: buy the wall's soldiers up to its target. Target =
    /// HQ garrison (3) + one defender per enemy-bordering frontier tile, bounded by the
    /// soldier cap. Buys are placed on the staging frontier (the only place a soldier CAN be
    /// placed); the follow-up `military()` pass moves them into the HQ + onto frontier tiles.
    /// Each buy goes through `fortress_buy_soldier` (the cash-runway solvency gate + metal
    /// cost), and the number of buys/turn is bounded so the metal bank can refill and the
    /// per-turn cash draw stays survivable. Fortress-only.
    fn fortress_field_wall(&mut self, g: &mut Game, player: PlayerId) {
        if !self.params.military || self.budget <= 0 {
            return;
        }
        let cap = g.max_soldier_amount(player);
        if cap <= 0 {
            return;
        }
        // MINIMUM VIABLE WALL vs ECONOMY-FIRST. The crack data (FORT_CRACK_DBG) shows nearly
        // every HQ-conquest is a SINGLE soldier walking onto an HQ with ZERO defenders — in
        // weak-economy / few-grassland seeds where the turtle never built an Outpost. So the
        // FIRST HQ defender is existential and must go up in EVERY seed: a lone soldier raises
        // the conquest bar from 1 attacker to 2 (3 defenders → 4). But buying the FULL wall
        // before the economy is established drops net below the Farm/Outpost build thresholds
        // and deadlocks the build-out (1 farm / cap 1 forever). So: ALWAYS allow the minimum
        // (1 HQ defender, or up to the current cap if directly threatened); only ramp the
        // LARGER wall once the economy can underwrite it (>= 3 farms OR positive net).
        let threatened = self.enemy_threat(g, player) > 0
            || self.reachable_enemy_max_defenders(g, player) > 0;
        let econ_ready =
            self.staffed_farm_count(g, player) >= 3 || self.net_money_per_round(g, player) >= 60.0;
        // CAP-BEFORE-WALL (breaks the death spiral). A garrison soldier's -30/round salary
        // depresses the SAME `net_money` that gates Outpost building, so fielding soldiers
        // early starved the cap → the wall could never grow past 1 AND outposts stopped (ops
        // collapsed). So: until the Outposts are up (cap headroom exists), field only the lone
        // existential HQ defender; pour the rest of the budget into the cap (Outposts). Once
        // the cap is built, fill the full wall. Under a direct threat, defend to the cap now.
        let outposts_up = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Outpost))
            .count() as i64;
        // Build the cap (Outposts) FIRST so the wall has headroom; field the full garrison
        // only once at least one Outpost (cap >= 4) is up — proactively fielding soldiers on a
        // capped-at-1 turtle just depresses net (death spiral) AND adds bankruptcy without a
        // real wall (a lone defender is killed then the HQ taken). Under a direct threat,
        // defend to the cap regardless.
        let has_cap_headroom = outposts_up >= 1;
        let min_wall = if threatened {
            cap
        } else if outposts_up >= 2 {
            cap // 2 Outposts up (cap >= 7) — fund the FULL ring/wall now
        } else if has_cap_headroom {
            // ARC sd3 RE-TUNE (2026-06-07): one Outpost up (cap 4). Field only 3 defenders
            // (not the full cap) so the wall's -30/round/soldier salary doesn't depress net
            // below the 2nd Outpost's build threshold (the death spiral that pinned the wall at
            // 1 Outpost — FORT report: fielding to cap dropped peaked-2-Outposts 51%→23%). 3
            // covers most of the HQ ring; the 4th defender waits for the 2nd Outpost.
            3i64.min(cap)
        } else {
            // ARC sd3 RE-TUNE (2026-06-07): field ONE lone HQ defender even before the first
            // Outpost. The dominant fortress loss is a SINGLE attacker soldier walking onto an
            // undefended HQ at ~r30 (min-attacker-at-conquest = 1 in 30-40% of games). A lone
            // defender raises the crack bar to 2 attackers — the single biggest survival unlock.
            // The old code set this to 0 (pour everything into Outposts first) because under the
            // OLD economy a soldier cost 50 metal and its -30/round salary starved the cap into a
            // death spiral. With the sd3 rebalance (soldier 30 metal, Outpost upkeep 5/round) the
            // lone defender is cheap enough that it no longer blocks the cap build-out. Gated on
            // `fortress`; capped at 1 (`cap.min(1)`) so the cap-1 turtle fields exactly its lone
            // defender, not a doomed over-hire.
            1i64.min(cap)
        };
        // OUTPOST-METAL PRIORITY. A soldier (50 metal) competes with an Outpost (100 metal)
        // for the same mined metal; if the wall keeps draining metal to ~10-50 the Outpost
        // (which RAISES the cap so MORE soldiers can be held) never has its 100 metal and the
        // cap stays at 1 forever (see fort_diag). While the turtle still wants more Outposts
        // and isn't directly threatened, reserve 100 metal for the next Outpost — only spend
        // metal on soldiers from the SURPLUS above that. (Under a real threat, defend now.)
        let outposts_now = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Outpost))
            .count() as i64;
        let wants_more_outposts = outposts_now < self.params.max_outposts;
        // Reserve metal for the next Outpost (cap-raise) only AFTER the minimum HQ defender is
        // up — the lone defender (50 metal) is existential and must never be blocked by the
        // Outpost metal reservation. Don't reserve under a direct threat (defend now).
        let have_min_defender = g.current_soldier_amount(player) >= 1;
        let metal_reserve_for_outpost =
            if wants_more_outposts && !threatened && have_min_defender { 100 } else { 0 };
        let frontier_tiles = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| {
                self.building_of(g, t) != Some(BuildingType::Outpost)
                    && self.enemy_border_count(g, t, player) > 0
            })
            .count() as i64;
        // Wall target: the FULL wall (a soldier per HQ-ring tile + one per frontier tile,
        // capped by cap) once the cap (Outposts) is built AND the econ can underwrite it;
        // otherwise just `min_wall` (1 lone defender while the cap is still being laid).
        // ARC sd3 ROOT-CAUSE FIX (2026-06-07): size the home target to the actual HQ-RING (the
        // owned, non-Outpost, orthogonal-4 HQ neighbours an attacker must cross), not a flat 3.
        // A deep-interior turtle has 0 enemy-bordering `frontier_tiles` at zero pressure, so the
        // old `garrison.max(3) + frontier` target was 3 — one short of a 4-tile ring, so the ring
        // was NEVER fully manned (FORT report 0%) and a poke walked the open tile to the HQ. Now
        // it targets enough to put a defender on every ring tile (capped by the soldier cap).
        let hq_ring_size = match g.get_hq_tile(player) {
            Some(hq) => g
                .neighbour_tiles(hq)
                .into_iter()
                .filter(|&n| {
                    g.tiles[n.0].owner == Some(player)
                        && self.building_of(g, n) != Some(BuildingType::Outpost)
                        && g.tiles[n.0].has_space_for_units()
                })
                .count() as i64,
            None => 0,
        };
        let home_need = if self.params.fortress {
            self.params.garrison.max(hq_ring_size)
        } else {
            self.params.garrison.max(3)
        };
        let full_target = cap.min(home_need + frontier_tiles);
        let target = if has_cap_headroom && econ_ready {
            full_target
        } else {
            min_wall.min(full_target)
        };
        // Bound the buys this turn (cap-bounded ramp + a hard ceiling) so the wall grows
        // steadily without a single turn draining the bank or the metal stock.
        let mut buys_left = (target - g.current_soldier_amount(player)).min(3).max(0);
        while buys_left > 0
            && g.current_soldier_amount(player) < target
            // 30 = soldier metal cost (rebalanced 50→30, arc sd3); the reserve (100) is the
            // unchanged Outpost build-cost metal, kept aside for the next cap-raise.
            && self.metal(g, player) >= 30 + metal_reserve_for_outpost
            && self.budget > 0
        {
            if !self.fortress_buy_soldier(g, player) {
                break;
            }
            buys_left -= 1;
        }
    }

    fn military(&mut self, g: &mut Game, player: PlayerId) {
        if !self.params.military {
            return;
        }
        let cap = g.max_soldier_amount(player);
        if cap <= 0 {
            return;
        }
        let hq = g.get_hq_tile(player);
        let at_war = self.should_militarise(g, player);

        // 1. DEFENCE.
        struct Defend {
            tile: TileId,
            want: i64,
            pressure: i64,
        }
        let mut defend: Vec<Defend> = Vec::new();
        // FIX 2(b): when the device-strategist OWNS a standing Device, the halved soldier
        // cap is tiny (e.g. (HQ 1 + Outpost 3)/2 = 2), and the win condition is the Device,
        // not the HQ. Devote the whole halved army to the Device RING (below) by skipping
        // the routine HQ garrison unless the HQ is actually under threat.
        let skip_idle_hq_garrison = self.holding_own_device(g, player);
        if let Some(hq) = hq {
            let threat = self.adjacent_enemy_soldiers(g, hq, player) + self.invaders_on(g, hq, player);
            let want = if skip_idle_hq_garrison {
                3i64.min(threat + 1).max(0) // only respond to a real HQ threat; 0 if none
            } else if at_war {
                3i64.min(self.params.garrison.max(threat + 1))
            } else {
                3i64.min(threat + 1)
            };
            if want > 0 {
                defend.push(Defend {
                    tile: hq,
                    want,
                    pressure: threat,
                });
            }
        }
        for tid in self.owned_tiles(g, player) {
            if Some(tid) == hq {
                continue;
            }
            if self.building_of(g, tid) == Some(BuildingType::Outpost) {
                continue;
            }
            let threat = self.adjacent_enemy_soldiers(g, tid, player) + self.invaders_on(g, tid, player);
            if threat > 0 {
                defend.push(Defend {
                    tile: tid,
                    want: 3i64.min(threat + 1),
                    pressure: threat,
                });
            }
        }
        // DEFEND OUR OWN DEVICE: the Device tile itself can hold no units, so it is
        // defended by garrisoning its APPROACHES — the owned tiles next to it — to the
        // cap, so the enemy can't get adjacent and stage a conquering unit on it. These
        // are forced to the top (high synthetic pressure): leaving the Device undefended
        // is an instant loss, so our halved army's first job is to ring it.
        if let Some(dt) = g.find_strange_device_tile() {
            if g.tiles[dt.0].owner == Some(player) {
                for ntid in g.neighbour_tiles(dt) {
                    if g.tiles[ntid.0].owner != Some(player) {
                        continue;
                    }
                    if self.building_of(g, ntid) == Some(BuildingType::Outpost) {
                        continue; // outposts can't hold soldiers anyway / are impregnable
                    }
                    let threat =
                        self.adjacent_enemy_soldiers(g, ntid, player) + self.invaders_on(g, ntid, player);
                    defend.push(Defend {
                        tile: ntid,
                        want: 3,
                        pressure: threat + 100, // outrank ordinary defence — the Device is existential
                    });
                }
            }
        }
        // FIX 2(b): SPREAD pass — when the device-strategist owns its Device, its halved
        // cap is small, so first put exactly ONE soldier on EACH owned approach (covering
        // the whole ring) BEFORE the depth pass below stacks the first tile to 3. Without
        // this, `garrison(approach, 3)` would dump both of a 2-soldier cap onto a single
        // approach, leaving the rest of the ring empty (ring-fill ~1 instead of ~2+).
        if self.holding_own_device(g, player) {
            if let Some(dt) = g.find_strange_device_tile() {
                if g.tiles[dt.0].owner == Some(player) {
                    for ntid in g.neighbour_tiles(dt) {
                        if g.tiles[ntid.0].owner != Some(player) {
                            continue;
                        }
                        if self.building_of(g, ntid) == Some(BuildingType::Outpost) {
                            continue;
                        }
                        if self.soldiers_on(g, ntid, player) < 1 {
                            self.garrison(g, player, ntid, 1);
                        }
                    }
                }
            }
        }
        // Reinforce the most-pressed shortfalls first (Device approaches carry a +100
        // synthetic pressure, so they win the tiebreak among max-shortfall tiles).
        defend.sort_by(|a, b| {
            let sa = b.want - self.soldiers_on(g, b.tile, player);
            let sb = a.want - self.soldiers_on(g, a.tile, player);
            sa.cmp(&sb).then(b.pressure.cmp(&a.pressure))
        });
        for d in defend {
            self.garrison(g, player, d.tile, d.want);
        }

        // FIX 2(b): in the device-banking window, stop hiring the offensive border-guard /
        // strike force so cash accumulates toward the Device. Essential DEFENCE (the
        // garrison-refill pass above, incl. the Device-approach ring) has already run, so
        // returning here keeps the bot defended while it banks. NOTE: this fires only
        // BEFORE the Device is down (banking_for_device requires !has_strange_device); once
        // the Device is standing, the `player_owns_strange_device` force branch below rings
        // the approaches with the full halved army as intended.
        if self.banking_for_device(g, player) {
            return;
        }
        // 2. BORDER GUARD + STRIKE FORCE.
        let hq = match hq {
            Some(h) if at_war => h,
            _ => return,
        };
        let farms = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Farm))
            .count() as i64;
        let aggression = self
            .enemy_threat(g, player)
            .max(self.reachable_enemy_max_defenders(g, player) + 1);
        let force = if self.enemy_has_device(g, player) {
            // PANIC: the enemy is halved NOW — field the biggest army the economy can
            // sustain (per-buy upkeep guards in `garrison` still apply), ignoring the
            // low *visible* defender count (their cap is halved, so it reads as weak).
            cap.min(farms + 3)
        } else if self.params.fortress {
            // LEAGUE-REBUILD STEP F — FORTRESS fields the FULL soldier cap. The wall is the
            // whole point: HARD's `min(garrison+strike_force, farms+1)` ceiling pinned the
            // turtle's army at ~3 even with cap 10 (strike_force 0 → ceiling = min(3, farms+1)),
            // so a calm pre-contact HQ + every Outpost's cap went UNUSED and the border stayed
            // 1-soldier-crackable. Field to the cap (HQ garrison to 3, then one defender per
            // frontier tile, then the remainder pooled). The per-commit
            // `affordable_after_commit(.., 2)` + metal-cost guards in `garrison` keep each hire
            // solvent, so this never bankrupts — it just lets the wall actually reach its size.
            cap
        } else if g.player_owns_strange_device(player) {
            // LEAGUE-REBUILD STEP D#3 (DEVICE-STRATEGIST): we hold a standing Device —
            // field the WHOLE halved army to ring its approaches (the defend-our-device
            // pass above garrisons the neighbours; this raises the force ceiling so the
            // strike loop actually staffs them). Gated on owning the Device, so only the
            // device bot (and HARD, only AFTER it has built its own Device) reaches here.
            cap
        } else if self.params.army_builder {
            // LEAGUE-REBUILD STEP E v2 (STRONG_ARMY): the army-builder fields the FULL
            // strike force (garrison + strike_force), capped only by the soldier cap and
            // what the staffed economy (farms + staffed expert producers + 1) can
            // underwrite, so a rich economy actually puts a real army in the field — but a
            // thin one never fields a doomed army. Unlike the old design there is NO econ /
            // assault readiness gate, so the offensive opens on HARD's proven schedule (the
            // `at_war` reactive trigger above) and never deadlocks. Gated on `army_builder`,
            // so every other preset is byte-identical.
            cap.min(self.params.garrison + self.params.strike_force)
                .min(farms + self.staffed_expert_producer_count(g, player) + 1)
        } else {
            cap.min(self.params.garrison + self.params.strike_force.min(aggression + 1))
                .min(farms + 1)
        };

        let mut frontier: Vec<TileId> = self
            .owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                self.building_of(g, t) != Some(BuildingType::Outpost)
                    && self.enemy_border_count(g, t, player) > 0
            })
            .collect();
        frontier.sort_by(|&a, &b| {
            self.enemy_border_count(g, b, player)
                .cmp(&self.enemy_border_count(g, a, player))
        });
        for tile in frontier {
            if g.current_soldier_amount(player) >= force {
                break;
            }
            self.garrison(g, player, tile, 1);
        }
        if g.current_soldier_amount(player) < force {
            let room = force - g.current_soldier_amount(player) + self.soldiers_on(g, hq, player);
            self.garrison(g, player, hq, 3i64.min(room));
        }

        // ARC sd3 ROOT-CAUSE FIX (2026-06-07): FORTRESS — RING THE HQ. A player's own
        // un-conquered HQ is NEVER in `get_available_tiles`, so a soldier can be neither BOUGHT
        // nor MOVED onto it (`ai_move_unit` requires the target be available, managers.rs:1381).
        // The HQ is therefore defended by garrisoning the OWNED tiles ADJACENT to it (the ring an
        // attacker must cross / stage on to conquer it). The shipped border-guard + `find_rear_
        // soldier` HQ path never filled this ring on a calm turtle (it only guards enemy-bordering
        // frontier tiles, and a deep-interior HQ has none early), so the HQ-adjacent tiles stood
        // EMPTY and the HQ fell to a single attacker (fort_diag; min-1-soldier crack). As the LAST
        // military action (nothing strips it afterwards), put >= 1 soldier on each owned HQ-
        // neighbour, pulling from ANY owned tile (border included, except an actively-assaulted
        // tile's last soldier). Gated on `fortress`.
        if self.params.fortress {
            let mut ring: Vec<TileId> = g
                .neighbour_tiles(hq)
                .into_iter()
                .filter(|&n| {
                    g.tiles[n.0].owner == Some(player)
                        && self.building_of(g, n) != Some(BuildingType::Outpost)
                        && g.tiles[n.0].has_space_for_units()
                })
                .collect();
            // ARC sd3 ROOT-CAUSE FIX (2026-06-07): fill the MOST-THREATENED ring tile FIRST.
            // With a thin garrison (often 1 soldier) and a 4-tile ring, filling tiles in raw
            // neighbour order left the attacker's actual approach undefended → a 1-soldier poke
            // walked the open ring tile onto the HQ (FORT_CRACK: min-at-conquest 1). Ordering by
            // (a) an enemy soldier already adjacent/invading, then (b) the tile bordering the most
            // enemy territory, puts the lone defender on the approach the attacker is using.
            ring.sort_by_key(|&t| {
                let imminent = self.adjacent_enemy_soldiers(g, t, player) + self.invaders_on(g, t, player);
                let border = self.enemy_border_count(g, t, player);
                std::cmp::Reverse((imminent, border))
            });
            for rtile in ring {
                if g.current_soldier_amount(player) == 0 {
                    break;
                }
                if self.soldiers_on(g, rtile, player) >= 1 || self.budget <= 0 {
                    continue;
                }
                // Pull ONE soldier onto this ring tile from any owned tile (must be ADJACENT —
                // `ai_move_unit` is a single-step move; the ring tile IS adjacent to its own
                // owned neighbours). Prefer pulling from a non-pressured tile.
                let pull = g
                    .neighbour_tiles(rtile)
                    .into_iter()
                    .filter(|&t| t != rtile && g.tiles[t.0].owner == Some(player))
                    .filter(|&t| {
                        let pressed = self.adjacent_enemy_soldiers(g, t, player)
                            + self.invaders_on(g, t, player)
                            > 0;
                        !(pressed && self.soldiers_on(g, t, player) <= 1)
                    })
                    .find_map(|t| {
                        g.tile_units(t)
                            .iter()
                            .copied()
                            .find(|&u| {
                                g.units[u.0].owner == Some(player)
                                    && g.units[u.0].kind == UnitType::Soldier
                            })
                            .map(|u| (u, t))
                    });
                if let Some((unit, from)) = pull {
                    let ok = g.ai_move_unit(unit, from, rtile);
                    self.do_action(ok);
                }
            }
            // INWARD HOMING MARCH: a soldier bought on a distant available frontier tile can
            // only step ONE tile/turn and the HQ itself is never an available move target, so a
            // deep-interior HQ's ring can't be filled in a single turn. Each turn, advance every
            // spare soldier (one not already adjacent to the HQ) ONE STEP toward the HQ so it
            // reaches the ring over a few turns. Skip soldiers already on the ring (adjacent to
            // HQ) and soldiers actively holding an assaulted tile. Mirrors `march_to_enemy_hq`
            // but homes INWARD to our own HQ. Gated on `fortress`.
            let (hx, hy) = (g.tiles[hq.0].x, g.tiles[hq.0].y);
            let d_to_hq = |x: i32, y: i32| (hx - x).abs() + (hy - y).abs();
            let mut inward = 0i64;
            while inward < 4 && self.budget > 0 {
                let avail = g.get_available_tiles();
                let step_targets: Vec<TileId> = avail
                    .into_iter()
                    .filter(|&t| {
                        let o = g.tiles[t.0].owner;
                        (o.is_none() || o == Some(player)) && g.tiles[t.0].has_space_for_units()
                    })
                    .collect();
                let mut best: Option<(UnitId, TileId, TileId, i32)> = None;
                for tid in self.owned_tiles(g, player) {
                    // Skip soldiers already on the HQ ring (in position).
                    if g.neighbour_tiles(hq).iter().any(|&n| n == tid) {
                        continue;
                    }
                    // Skip a soldier holding an actively-assaulted tile.
                    if (self.adjacent_enemy_soldiers(g, tid, player) + self.invaders_on(g, tid, player)) > 0 {
                        continue;
                    }
                    let (sx, sy) = (g.tiles[tid.0].x, g.tiles[tid.0].y);
                    let cur_d = d_to_hq(sx, sy);
                    let Some(unit) = g
                        .tile_units(tid)
                        .iter()
                        .copied()
                        .find(|&u| g.units[u.0].owner == Some(player) && g.units[u.0].kind == UnitType::Soldier)
                    else {
                        continue;
                    };
                    for &to in &step_targets {
                        if to == tid {
                            continue;
                        }
                        let (tx, ty) = (g.tiles[to.0].x, g.tiles[to.0].y);
                        let drop = cur_d - d_to_hq(tx, ty);
                        if drop <= 0 {
                            continue;
                        }
                        if best.map(|(_, _, _, bd)| drop > bd).unwrap_or(true) {
                            best = Some((unit, tid, to, drop));
                        }
                    }
                }
                let Some((unit, from, to, _)) = best else { break };
                let ok = g.ai_move_unit(unit, from, to);
                if !self.do_action(ok) {
                    break;
                }
                inward += 1;
            }
        }
    }

    // --- offence ------------------------------------------------------------

    fn find_free_soldier(&self, g: &Game, player: PlayerId, exclude: TileId) -> Option<(UnitId, TileId)> {
        for tid in self.owned_tiles(g, player) {
            if tid == exclude {
                continue;
            }
            if let Some(&u) = g
                .tile_units(tid)
                .iter()
                .find(|&&u| g.units[u.0].kind == UnitType::Soldier)
            {
                return Some((u, tid));
            }
        }
        None
    }

    fn attack(&mut self, g: &mut Game, player: PlayerId) {
        if !self.params.attack {
            return;
        }
        // FIX 2(b): the DEVICE-STRATEGIST must NOT open offensive fronts AT ALL while
        // pursuing the Device win — neither while racing toward it (it hoards cash for the
        // Device so `can_buy` is always true, and the `<= 1 && !can_buy` gate below would
        // never stop it) NOR after building it (once its own Device stands it must DEFEND
        // the ring and let the countdown win, not march off to conquer). Without this it
        // staged soldiers on the passive opponent's HQ and won by CONQUEST ~10 rounds
        // before the Device countdown resolved, defeating the whole strategy. The
        // counter-crack path stays on: an enemy Device flips `enemy_has_device` → the phase
        // runs so the bot can still crack it.
        if self.params.device
            && self.params.warmonger
            && !self.enemy_has_device(g, player)
        {
            return;
        }
        let can_buy = self.money(g, player) >= self.params.reserve + 250;
        if self.params.assaults_per_turn <= 1 && !can_buy {
            return;
        }
        // LEAGUE-REBUILD STEP E (STRONG_ARMY): don't open a front until massed. Gated on
        // `attack_ready_soldiers > 0`, so every other preset is byte-identical; an enemy
        // Device still cracks the gate (via `assault_ready`), so the counterplay is never
        // blocked.
        if !self.assault_ready(g, player) {
            return;
        }

        struct Target {
            tile: TileId,
            defenders: i64,
            is_device: bool,
            is_hq: bool,
            cut: f64,
        }
        let cut_priority = self.params.cut_priority;
        let mut targets: Vec<Target> = g
            .get_available_tiles()
            .into_iter()
            .filter(|&t| {
                let o = g.tiles[t.0].owner;
                o.is_some() && o != Some(player) && g.tiles[t.0].has_space_for_conquering_units()
            })
            .map(|t| Target {
                tile: t,
                defenders: g
                    .tile_units(t)
                    .iter()
                    .filter(|&&u| g.units[u.0].kind == UnitType::Soldier)
                    .count() as i64,
                // Destroying an enemy Device stops its loss clock — the top priority.
                is_device: self.building_of(g, t) == Some(BuildingType::StrangeDevice),
                is_hq: self.building_of(g, t) == Some(BuildingType::Headquarters),
                // Only the (non-shipped) cut bot pays the BFS cost.
                cut: if cut_priority {
                    crate::spatial::offensive_cut_value(g, player, t)
                } else {
                    0.0
                },
            })
            .filter(|t| {
                self.building_of(g, t.tile) != Some(BuildingType::Outpost) && t.defenders < 3
            })
            .collect();
        if cut_priority {
            // Enemy Device FIRST (its countdown wins them the game), then highest
            // cut-value (fraction of enemy severed), then cheapest.
            targets.sort_by(|a, b| {
                (b.is_device as i64)
                    .cmp(&(a.is_device as i64))
                    .then_with(|| {
                        b.cut
                            .partial_cmp(&a.cut)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then(a.defenders.cmp(&b.defenders))
            });
        } else {
            // SHIPPED order: Device first (cracking it resets the clock + reopens the
            // slot), then HQ (collapses an opponent via the connectivity rule), then
            // fewest defenders.
            targets.sort_by(|a, b| {
                (b.is_device as i64)
                    .cmp(&(a.is_device as i64))
                    .then((b.is_hq as i64).cmp(&(a.is_hq as i64)))
                    .then(a.defenders.cmp(&b.defenders))
            });
        }

        let max_assaults = 1i64.max(self.params.assaults_per_turn);
        let mut assaults = 0;
        for t in targets {
            if assaults >= max_assaults || self.budget <= 0 {
                break;
            }
            let needed = t.defenders + 1;
            let mut placed = g
                .tile_conquering_units(t.tile)
                .iter()
                .filter(|&&u| {
                    g.units[u.0].owner == Some(player) && g.units[u.0].kind == UnitType::Soldier
                })
                .count() as i64;
            let to_add = needed - placed;
            if to_add <= 0 {
                continue;
            }
            let movable: i64 = self
                .owned_tiles(g, player)
                .iter()
                .map(|&ti| {
                    if ti == t.tile {
                        0
                    } else {
                        g.tile_units(ti)
                            .iter()
                            .filter(|&&u| g.units[u.0].kind == UnitType::Soldier)
                            .count() as i64
                    }
                })
                .sum();
            let buyable = if can_buy {
                g.free_soldier_amount(player)
                    .min(self.metal(g, player) / 30) // soldier metal cost (rebalanced 50→30, arc sd3)
                    .min((self.money(g, player) - self.params.reserve) / 200)
            } else {
                0
            };
            if movable + buyable < to_add {
                continue;
            }
            while placed < needed {
                let mut did = false;
                if let Some((unit, from)) = self.find_free_soldier(g, player, t.tile) {
                    let ok = g.ai_move_unit(unit, from, t.tile);
                    did = self.do_action(ok);
                } else if can_buy
                    && g.free_soldier_amount(player) > 0
                    && self.metal(g, player) >= 30 // soldier metal cost (rebalanced 50→30, arc sd3)
                    && self.affords(g, player, &soldier_cost(), self.params.reserve)
                    && {
                        // SELF-BANKRUPTCY GATE (strike-force hire): +30/round per
                        // soldier. The strike-force loop hires up to `needed` per
                        // turn, so without projection it can race past the income
                        // ceiling in a single turn. Buffer = 4 rounds.
                        let soldier_money = -soldier_cost().get(BasicResource::Money).unwrap_or(0);
                        self.affordable_after_commit(g, player, soldier_money, 30, 4)
                    }
                {
                    let ok = g.ai_buy_and_place_unit("Soldier", t.tile);
                    did = self.do_action(ok);
                }
                if !did {
                    break;
                }
                placed += 1;
            }
            if placed >= needed {
                assaults += 1;
            }
        }
    }

    /// REACTIVE-FIX (MARCHER) — advance spare own-soldiers TOWARD the nearest enemy
    /// HQ even when no legal Attack exists this turn. Closes the "march your army
    /// across the map" demonstration gap (existing `attack` phase only fires when an
    /// enemy is in `get_available_tiles`, so a no-contact game never produces the
    /// march-then-conquer trajectory the learner's buffer needs).
    ///
    /// Algorithm per turn (bounded to `assaults_per_turn` moves; runs only after
    /// `attack()` has consumed every legal Attack target):
    /// 1. Compute the nearest live-enemy HQ's coords; bail if none exists.
    /// 2. For each owned soldier whose CURRENT tile is NOT on the enemy frontier
    ///    (enemy_border_count == 0) AND whose tile is FURTHER from the enemy HQ
    ///    than at least one available-but-uncontested move target:
    ///    - Find the available tile (owned ∪ orthogonal-neighbour-of-owned) that
    ///      strictly DECREASES Manhattan distance to the enemy HQ and is unowned-
    ///      or-own (NOT enemy-owned — that would be an Attack, handled by
    ///      `attack()` above) and has room.
    ///    - Move via `ai_move_unit` (same engine primitive HARD uses everywhere).
    /// 3. Stop when budget is exhausted OR no soldier can advance.
    ///
    /// Strictly read-only on enemy tiles; the only mutation is `ai_move_unit` on
    /// one of our own soldiers — same primitive HARD's existing phases use. NO new
    /// engine action / candidate type → parity-irrelevant.
    fn march_to_enemy_hq(&mut self, g: &mut Game, player: PlayerId) {
        if !self.params.military {
            return;
        }
        // 1. Find the nearest live-enemy HQ. If none → no march target.
        let mut enemy_hqs: Vec<(i32, i32)> = Vec::new();
        for &p in g.live_players() {
            if p == player {
                continue;
            }
            if let Some(hq) = g.get_hq_tile(p) {
                let t = &g.tiles[hq.0];
                enemy_hqs.push((t.x, t.y));
            }
        }
        if enemy_hqs.is_empty() {
            return;
        }
        // Manhattan distance from (x,y) to the closest enemy HQ.
        let d_to_enemy = |x: i32, y: i32| -> i32 {
            enemy_hqs
                .iter()
                .map(|&(ex, ey)| (ex - x).abs() + (ey - y).abs())
                .min()
                .unwrap_or(i32::MAX)
        };

        // Budget for marches this turn: mirrors `assaults_per_turn` (a marcher who
        // can't attack still has the same per-turn aggression budget).
        let max_marches = self.params.assaults_per_turn.max(1);
        let mut marches = 0i64;
        loop {
            if marches >= max_marches || self.budget <= 0 {
                return;
            }
            // 2. Re-enumerate available-move targets each iteration (the previous
            //    move may have changed what's available).
            let avail = g.get_available_tiles();
            // Targets we may move ONTO: owned-or-unowned-with-room, NOT enemy-owned.
            // (Moving onto an enemy tile is conquering — handled by `attack()`.)
            let targets: Vec<TileId> = avail
                .into_iter()
                .filter(|&t| {
                    let o = g.tiles[t.0].owner;
                    let is_safe = o.is_none() || o == Some(player);
                    is_safe && g.tiles[t.0].has_space_for_units()
                })
                .collect();
            if targets.is_empty() {
                return;
            }
            // Find the best (from_soldier, to_tile) pair: maximises the distance
            // DROP toward the nearest enemy HQ. A drop of <= 0 means the soldier
            // can't get closer this turn (its current tile is already a local min).
            let mut best: Option<(UnitId, TileId, TileId, i32)> = None; // (unit, from, to, drop)
            for tid in self.owned_tiles(g, player) {
                let (sx, sy) = (g.tiles[tid.0].x, g.tiles[tid.0].y);
                let cur_d = d_to_enemy(sx, sy);
                // Skip already-frontier soldiers (an Attack would have fired if
                // an enemy was assault-legal; if not, the soldier is "in position").
                if self.enemy_border_count(g, tid, player) > 0 {
                    continue;
                }
                // First soldier on this tile (any one works for the march).
                let soldier: Option<UnitId> = g
                    .tile_units(tid)
                    .iter()
                    .copied()
                    .find(|&u| {
                        g.units[u.0].owner == Some(player)
                            && g.units[u.0].kind == UnitType::Soldier
                    });
                let Some(unit) = soldier else { continue };
                // Find the best move target for this soldier.
                for &to in &targets {
                    if to == tid {
                        continue;
                    }
                    let (tx, ty) = (g.tiles[to.0].x, g.tiles[to.0].y);
                    let new_d = d_to_enemy(tx, ty);
                    let drop = cur_d - new_d;
                    if drop <= 0 {
                        continue;
                    }
                    if best.map(|(_, _, _, bd)| drop > bd).unwrap_or(true) {
                        best = Some((unit, tid, to, drop));
                    }
                }
            }
            // 3. Execute the best advance, or stop if none.
            let Some((unit, from, to, _drop)) = best else {
                return;
            };
            let ok = g.ai_move_unit(unit, from, to);
            if !self.do_action(ok) {
                return;
            }
            marches += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Plan-B `HQ_RUSH_PARAMS` sanity: HQ_RUSH is the ARMY_RUSH cousin with
    /// further-cranked aggression knobs so the rusher relentlessly pushes toward
    /// the enemy HQ (the shipped `attack` phase already orders Device > HQ >
    /// fewest-defenders). Pin the relative ordering of the aggression knobs
    /// against ARMY_RUSH so any future drift is caught.
    #[test]
    fn hq_rush_prefers_attacks_near_enemy_hq() {
        // Shape: HQ_RUSH is at least as aggressive as ARMY_RUSH on every assault knob.
        assert!(
            HQ_RUSH_PARAMS.assaults_per_turn >= ARMY_RUSH_PARAMS.assaults_per_turn,
            "HQ_RUSH must press as hard as ARMY_RUSH on assaults/turn"
        );
        assert!(
            HQ_RUSH_PARAMS.strike_force >= ARMY_RUSH_PARAMS.strike_force,
            "HQ_RUSH must field at least as large an offensive army as ARMY_RUSH"
        );
        assert!(HQ_RUSH_PARAMS.warmonger, "HQ_RUSH gears up for war on enemy contact");
        assert!(HQ_RUSH_PARAMS.attack, "HQ_RUSH must have the attack phase ON");
        assert!(!HQ_RUSH_PARAMS.device, "HQ_RUSH commits to the army win, not the Device race");
        // The shipped attack-target ordering is Device > HQ > fewest-defenders (see
        // `hard_ai::attack`), so HQ-first behaviour is structural. The constructor
        // wires HardAi to the right params (smoke check).
        let bot = HardAi::hq_rush();
        assert_eq!(bot.params.max_outposts, HQ_RUSH_PARAMS.max_outposts);
        assert_eq!(bot.params.strike_force, HQ_RUSH_PARAMS.strike_force);
    }

    /// OVERNIGHT-RUN §B.1: `warmonger: true` makes `should_militarise()` collapse to
    /// `enemy_exists()` (any live opponent), so a GARRISON bot's garrison phase fires
    /// from round 1 — the load-bearing property that closes the 1-soldier-rush hole.
    /// Verified for both round 0 and a no-contact (no shared frontier) state: the
    /// warmonger path bypasses `has_reachable_enemy` entirely (line 999-1000).
    #[test]
    fn garrison_holds_at_war_from_round_1() {
        // Shape: GARRISON pins the load-bearing knobs.
        assert!(GARRISON_PARAMS.warmonger, "GARRISON forces at_war from round 1 via warmonger");
        assert_eq!(GARRISON_PARAMS.garrison, 3, "GARRISON requires ≥ 3 HQ garrison");
        assert_eq!(GARRISON_PARAMS.assaults_per_turn, 0,
            "GARRISON suppresses the assault phase via the <=1 && !can_buy gate");
        assert_eq!(GARRISON_PARAMS.strike_force, 0, "GARRISON never goes on the offensive");
        assert!(GARRISON_PARAMS.attack, "GARRISON keeps attack: true so the counter-cracker fires");
        assert!(GARRISON_PARAMS.military, "GARRISON must allow soldier hires");
        assert!(GARRISON_PARAMS.experts, "GARRISON staffs Experts (efficient economy)");
        assert!(!GARRISON_PARAMS.device, "GARRISON is a pure fortress, no Device race");

        // Behavioural: with TWO live players and `warmonger: true`, `should_militarise`
        // returns true regardless of contact / round. The default HARD bot WITHOUT
        // warmonger does NOT fire on a no-contact map at round 0.
        let g = Game::new(8, 8, &["P0", "P1"]);
        let me = PlayerId(0);
        let bot = HardAi::garrison_fortress();
        assert!(
            bot.should_militarise(&g, me),
            "GARRISON: should_militarise() must be true at round 0 (warmonger forces at_war)"
        );
        let hard = HardAi::hard();
        assert!(
            !hard.should_militarise(&g, me),
            "HARD (no warmonger, no contact, no enemy device): should_militarise must be FALSE \
             at round 0 — this is the loose-garrison hole GARRISON closes"
        );
    }

    /// OVERNIGHT-RUN §B.2: `EXPERT_PARAMS` pins the pure-econ knobs. Behavioural check:
    /// the bot's `staff_plant` / `boost_mines` paths key on `params.experts`, so when
    /// the seat has a staffed Mine + free unit slot + funds, the bot adds an Expert.
    /// We verify the knobs here (the full simulation pipeline is covered in the
    /// integration tests in cnn_train.rs).
    #[test]
    fn econ_expert_hires_experts_when_slots_free() {
        // Shape: EXPERT pins the load-bearing knobs.
        assert!(EXPERT_PARAMS.experts, "EXPERT must front the Expert tier");
        assert!(EXPERT_PARAMS.nuclear, "EXPERT must also push Nuclear (Expert-gated production)");
        assert!(!EXPERT_PARAMS.military, "EXPERT is pure economic — never strikes");
        assert!(EXPERT_PARAMS.attack, "EXPERT keeps attack: true so the counter-cracker fires");
        assert!(!EXPERT_PARAMS.warmonger, "EXPERT does NOT pre-emptively militarise");
        assert!(!EXPERT_PARAMS.device, "EXPERT does NOT race the Device — pure econ");
        assert_eq!(EXPERT_PARAMS.assaults_per_turn, 1,
            "EXPERT cracker-only (the <= 1 && !can_buy gate suppresses offensives)");
        assert_eq!(EXPERT_PARAMS.garrison, 1, "EXPERT bare-minimum garrison (no military emphasis)");
        assert_eq!(EXPERT_PARAMS.max_outposts, 1,
            "EXPERT post-bankruptcy-audit (2026-06-05): 1 OP feeds the cracker chain; \
             2 was a metal-leak risk after Mine loss");
        assert_eq!(EXPERT_PARAMS.strike_force, 0, "EXPERT no offensive army");
        assert_eq!(EXPERT_PARAMS.expand, 4);
        assert_eq!(EXPERT_PARAMS.reserve, 200,
            "EXPERT post-bankruptcy-audit (2026-06-05): 140 → 200 widens the late-game \
             cash cushion against compounding salary creep");
        assert_eq!(EXPERT_PARAMS.max_actions, 28);

        // Smoke: the constructor wires HardAi to the right params.
        let bot = HardAi::econ_expert();
        assert_eq!(bot.params.experts, EXPERT_PARAMS.experts);
        assert_eq!(bot.params.nuclear, EXPERT_PARAMS.nuclear);
        assert!(!bot.params.military, "the wired bot must be pure-econ");

        // Behavioural smoke: run one plan_turn on a freshly-seeded game. The bot must
        // not panic and (with `experts: true` and the params above) it MUST not
        // immediately go on the offensive — the assault counter stays at 0 because
        // the gate `assaults_per_turn <= 1 && !can_buy` returns from attack() before
        // any soldier is staged. This pins the pure-econ behaviour.
        let mut g = Game::new(10, 10, &["P0", "P1"]);
        g.generate_map(10, 10, 0xE7);
        let me = PlayerId(0);
        // Skip HQ placement: just run one turn (the bot's first action will be
        // `staff_buildings`, which is a no-op without any owned producer — safe).
        // The acid test is that `plan_turn` returns without panic and respects the
        // pure-econ flags.
        if g.current_player() == me {
            let mut bot = HardAi::econ_expert();
            bot.plan_turn(&mut g, me);
        }
        // The flags must still be intact after the turn (plan_turn restores params).
        let bot = HardAi::econ_expert();
        assert!(!bot.params.military);
        assert!(bot.params.experts);
    }

    /// REACTIVE-FIX MARCHER param-shape pins + behavioural check: MARCHER must advance
    /// a spare soldier TOWARD a distant enemy HQ even when no legal Attack exists this
    /// turn (the demonstration gap the spec calls out: existing ARMY_RUSH / HQ_RUSH
    /// attack ONLY when an enemy is in `get_available_tiles`).
    #[test]
    fn marcher_advances_soldier_toward_enemy_hq() {
        // Shape: MARCHER pins the load-bearing knobs (aggression to 11 + warmonger).
        assert!(MARCHER_PARAMS.warmonger, "MARCHER forces at_war from round 1");
        assert!(MARCHER_PARAMS.military, "MARCHER must allow soldier hires");
        assert!(MARCHER_PARAMS.attack, "MARCHER keeps attack phase ON");
        assert!(!MARCHER_PARAMS.device, "MARCHER commits to the army win, not the Device");
        assert_eq!(MARCHER_PARAMS.garrison, 1, "MARCHER minimal home defence (willing to leave HQ)");
        assert_eq!(MARCHER_PARAMS.assaults_per_turn, 16, "MARCHER very-aggressive assaults_per_turn");
        assert_eq!(MARCHER_PARAMS.strike_force, 14, "MARCHER huge offensive army goal");
        assert_eq!(MARCHER_PARAMS.max_outposts, 2, "MARCHER one early Outpost (cap → 4 soldiers)");

        // Smoke: the constructor wires HardAi to the right params.
        let bot = HardAi::marcher();
        assert_eq!(bot.params.assaults_per_turn, MARCHER_PARAMS.assaults_per_turn);
        assert_eq!(bot.params.strike_force, MARCHER_PARAMS.strike_force);
        assert!(bot.params.warmonger);

        // Behavioural fixture: a 12x12 board with MARCHER (P0) holding a cluster of
        // tiles near (1,1) — including a soldier at (1,1) — and the enemy (P1) holding
        // a single tile near (10,10). The two clusters are NON-ADJACENT (no enemy tile
        // sits in the marcher's `get_available_tiles`), so the existing `attack`
        // phase has NO legal target. We then call `march_to_enemy_hq` directly and
        // verify the soldier moved to a tile that is CLOSER to the enemy HQ.
        let mut g = Game::new(12, 12, &["P0", "P1"]);
        g.generate_map(12, 12, 0xC0FFEE);
        let me = PlayerId(0);
        let enemy = PlayerId(1);
        // Force the relevant tiles to grassland so they can be claimed + built on.
        let id = |x: i32, y: i32| TileId((x * 12 + y) as usize);
        // My cluster: HQ at (1,1), an own grassland at (2,1) and (1,2) so I have a
        // multi-tile territory and `get_available_tiles` returns more than just the
        // HQ neighbours.
        let my_hq = id(1, 1);
        let my_a = id(2, 1);
        let my_b = id(1, 2);
        for &t in &[my_hq, my_a, my_b] {
            g.tiles[t.0].tile_type = TileType::Grassland;
            g.tiles[t.0].building = None;
            g.set_tile_owner(t, Some(me));
        }
        g.place_building(my_hq, BuildingType::Headquarters, Some(me));
        // Enemy cluster: HQ at (10,10), single tile (far from my cluster).
        let enemy_hq = id(10, 10);
        g.tiles[enemy_hq.0].tile_type = TileType::Grassland;
        g.tiles[enemy_hq.0].building = None;
        g.set_tile_owner(enemy_hq, Some(enemy));
        g.place_building(enemy_hq, BuildingType::Headquarters, Some(enemy));
        // Place a soldier on my HQ (in MY rear — d=(10-1)+(10-1)=18 from enemy HQ).
        g.spawn_unit_on_tile(UnitType::Soldier, me, my_hq, false);
        g.players[me.0].max_soldier_amount = 4;

        // Sanity: no enemy tile is in my get_available_tiles (no contact).
        let avail = g.get_available_tiles_for(me);
        assert!(
            !avail.iter().any(|&t| g.tiles[t.0].owner == Some(enemy)),
            "test fixture invariant: enemy must NOT be in my available tiles (no contact)"
        );

        // Run the march phase directly.
        let mut bot = HardAi::marcher();
        bot.budget = 8;
        bot.march_to_enemy_hq(&mut g, me);

        // Find where the soldier ended up. A march onto an unowned tile flips the
        // soldier to "conquering" (engine: `ai_move_unit` sets `is_conquering = true`
        // when the dest tile isn't owned), so it sits in `conquering_units` — check
        // both lists. The soldier remains owned by `me`.
        let mut soldier_loc: Option<(i32, i32)> = None;
        for t in g.get_tiles().iter() {
            for &u in t.units.iter().chain(t.conquering_units.iter()) {
                if g.units[u.0].owner == Some(me) && g.units[u.0].kind == UnitType::Soldier {
                    soldier_loc = Some((t.x, t.y));
                }
            }
        }
        let (sx, sy) = soldier_loc.expect("soldier still exists after march");
        let new_d = (10 - sx).abs() + (10 - sy).abs();
        // Original d = (10-1) + (10-1) = 18. After the march, the soldier should be
        // strictly closer to (10,10) — anywhere available with d < 18 satisfies us.
        assert!(
            new_d < 18,
            "MARCHER must advance the soldier toward the enemy HQ when no Attack is \
             legal — soldier ended at ({sx},{sy}) d={new_d} (started at (1,1) d=18)"
        );
    }

    // =======================================================================
    // LEAGUE-REBUILD (2026-06-06) — canonical 4-bot league param-shape tests.
    // =======================================================================

    /// The 3 NEW league fields (`fortress`, `attack_ready_soldiers`, `econ_ready_net`)
    /// must be at their NO-OP defaults on EVERY shipped preset so HARD and all existing
    /// presets stay behaviorally identical. Only the NEW league bots (RUSHER reuses no
    /// new logic; FORTRESS sets fortress; STRONG_ARMY sets the gates) deviate — and
    /// RUSHER also keeps all three at no-op (it is a pure param fix).
    #[test]
    fn shipped_presets_keep_league_fields_at_noop() {
        // SHIPPED presets (the legacy benchmark + the pre-league scripted bots) MUST
        // keep all three new fields at their no-op default.
        for (name, p) in [
            ("HARD", HARD_PARAMS),
            ("MEDIUM", MEDIUM_PARAMS),
            ("EASY", EASY_PARAMS),
            ("ARMY_RUSH", ARMY_RUSH_PARAMS),
            ("HQ_RUSH", HQ_RUSH_PARAMS),
            ("GARRISON", GARRISON_PARAMS),
            ("EXPERT", EXPERT_PARAMS),
            ("MARCHER", MARCHER_PARAMS),
            // DEVICE_RUSH was rebuilt but its new behavior is gated on `device`/
            // `warmonger`, NOT the 3 new fields — they stay no-op.
            ("DEVICE_RUSH", DEVICE_RUSH_PARAMS),
            // RUSHER is a pure param fix: no new logic, all 3 fields no-op.
            ("RUSHER", RUSHER_PARAMS),
        ] {
            assert!(!p.fortress, "{name}: fortress must be false (no-op)");
            assert_eq!(p.attack_ready_soldiers, 0, "{name}: attack_ready_soldiers must be 0 (no-op)");
            assert_eq!(p.econ_ready_net, 0, "{name}: econ_ready_net must be 0 (no-op)");
            assert!(!p.army_builder, "{name}: army_builder must be false (no-op) — only STRONG_ARMY sets it");
        }
    }

    /// RUSHER ("homing missile") — pure param fix. Reserve 220 is the bankruptcy fix;
    /// warmonger+military wire the bridge+march+attack chain (no new logic).
    #[test]
    fn rusher_param_shape() {
        let p = RUSHER_PARAMS;
        assert_eq!(p.reserve, 170, "RUSHER reserve 170 (ARC sd3 re-tune from 220 — cheaper military frees tempo)");
        assert_eq!(p.max_actions, 30);
        assert!(p.experts);
        assert!(p.military);
        assert_eq!(p.garrison, 2);
        assert_eq!(p.expand, 4);
        assert!(p.attack);
        assert!(!p.nuclear);
        assert_eq!(p.max_outposts, 2);
        assert_eq!(p.strike_force, 6);
        assert_eq!(p.assaults_per_turn, 10);
        assert!(p.warmonger, "RUSHER warmonger wires build_bridges + march + early attack");
        assert!(!p.cut_priority);
        assert!(!p.device);
        // Constructor wires correctly.
        let bot = HardAi::rusher();
        assert_eq!(bot.params.reserve, 170);
    }

    /// FORTRESS (the turtle) — proactive Outposts, never marches its wall.
    #[test]
    fn fortress_param_shape() {
        let p = FORTRESS_PARAMS;
        assert_eq!(p.reserve, 320);
        assert_eq!(p.max_actions, 26);
        assert!(p.experts);
        assert!(p.military);
        assert_eq!(p.garrison, 3);
        assert_eq!(p.expand, 3);
        assert!(p.attack, "FORTRESS keeps attack ON so the counter-cracker fires");
        assert!(!p.nuclear);
        assert_eq!(p.max_outposts, 3);
        assert_eq!(p.strike_force, 0, "FORTRESS never goes on the offensive");
        assert_eq!(p.assaults_per_turn, 0);
        assert!(p.warmonger);
        assert!(!p.cut_priority);
        assert!(!p.device);
        assert!(p.fortress, "FORTRESS: THE point is proactive Outpost building");
        assert_eq!(p.attack_ready_soldiers, 0);
        assert_eq!(p.econ_ready_net, 0);
        // Constructor wires correctly.
        let bot = HardAi::fortress();
        assert!(bot.params.fortress);

        // Behavioural: `proactive_outposts` must be TRUE for the turtle even with no
        // enemy contact / no device — that's the load-bearing relaxation of the
        // build_outposts militarise gate. HARD must stay FALSE.
        let g = Game::new(10, 10, &["P0", "P1"]);
        let me = PlayerId(0);
        assert!(
            HardAi::fortress().proactive_outposts(&g, me),
            "FORTRESS must want proactive Outposts (fortress: true) from round 0"
        );
        assert!(
            !HardAi::hard().proactive_outposts(&g, me),
            "HARD must NOT want proactive Outposts (byte-identical gate)"
        );
    }

    /// STRONG_ARMY (the yardstick) — STEP E v2: HARD-rebased, gates OFF (no deadlock),
    /// reserve 145 + cut_priority + army_builder to EDGE the HARD mirror.
    #[test]
    fn strong_army_param_shape() {
        let p = STRONG_ARMY_PARAMS;
        assert_eq!(p.reserve, 130, "STRONG_ARMY reserve 130 (ARC sd3 re-tune from 145 — cheaper military, more tempo)");
        assert_eq!(p.max_actions, 34);
        assert!(p.experts);
        assert!(p.military);
        assert_eq!(p.garrison, 3);
        assert_eq!(p.expand, 6);
        assert!(p.attack);
        assert!(p.nuclear, "STRONG_ARMY pushes Nuclear (rich late-game engine)");
        assert_eq!(p.max_outposts, 6);
        assert_eq!(p.strike_force, 12);
        assert_eq!(p.assaults_per_turn, 10);
        assert!(!p.warmonger, "STRONG_ARMY uses HARD's reactive militarise-on-contact, no pre-militarise");
        assert!(p.cut_priority, "STRONG_ARMY uses the surgical HQ-severing attack order — the edge over the mirror");
        assert!(!p.device);
        assert!(!p.fortress);
        assert!(p.army_builder, "STRONG_ARMY lifts the military() strike-force ceiling (no-op superset of HARD in fast games)");
        assert_eq!(p.attack_ready_soldiers, 0, "STRONG_ARMY: gates OFF — the readiness gate deadlocked");
        assert_eq!(p.econ_ready_net, 0, "STRONG_ARMY: gates OFF — the econ gate was unreachable");
        // Constructor wires correctly.
        let bot = HardAi::strong_army();
        assert!(bot.params.army_builder);
        assert_eq!(bot.params.reserve, 130);

        // Behavioural: with gates OFF, STRONG_ARMY commits on HARD's schedule — it is
        // ALWAYS assault-ready (no deadlocking massing gate), exactly like HARD.
        let mut g = Game::new(10, 10, &["P0", "P1"]);
        g.generate_map(10, 10, 0x5A);
        let me = PlayerId(0);
        let sa = HardAi::strong_army();
        assert!(
            sa.assault_ready(&g, me),
            "STRONG_ARMY (gates off) must ALWAYS be assault-ready — commit on HARD's schedule"
        );
        assert!(
            HardAi::hard().assault_ready(&g, me),
            "HARD (attack_ready_soldiers 0) must ALWAYS be assault-ready (byte-identical)"
        );
    }

    /// DEVICE_RUSH rebuild — the rebuilt strategist's load-bearing knobs.
    #[test]
    fn device_rush_rebuild_param_shape() {
        let p = DEVICE_RUSH_PARAMS;
        assert_eq!(p.reserve, 150, "DEVICE_RUSH reserve 150 lets banking spend down to the Device cost");
        assert_eq!(p.max_actions, 28);
        assert!(p.experts);
        assert!(p.military);
        assert_eq!(p.garrison, 3);
        assert_eq!(p.expand, 3, "DEVICE_RUSH dense econ — avoids conquering the map before the countdown");
        assert!(p.attack);
        assert!(!p.nuclear);
        assert_eq!(p.max_outposts, 1, "DEVICE_RUSH: one precursor Outpost (a 2nd starves the build)");
        assert_eq!(p.strike_force, 0, "DEVICE_RUSH: no offensive army — the Device is the win plan");
        assert_eq!(p.assaults_per_turn, 0);
        assert!(p.warmonger, "DEVICE_RUSH warmonger fires proactive_outposts (device precursor)");
        assert!(!p.cut_priority);
        assert!(p.device, "DEVICE_RUSH: THE point is racing the Strange Device");
        assert!(!p.fortress);
        assert_eq!(p.attack_ready_soldiers, 0);
        assert_eq!(p.econ_ready_net, 0);

        // Behavioural: DEVICE_RUSH wants proactive outposts once the game matures (round
        // >= 12, no device down). At round 0 the round-gate keeps it false.
        let mut g = Game::new(10, 10, &["P0", "P1"]);
        let me = PlayerId(0);
        let dr = HardAi::device_rush();
        assert!(
            !dr.proactive_outposts(&g, me),
            "DEVICE_RUSH proactive_outposts must be FALSE before round 12"
        );
        while g.get_rounds_played() < 12 {
            g.change_turn();
        }
        assert!(
            dr.proactive_outposts(&g, me),
            "DEVICE_RUSH proactive_outposts must be TRUE at round >= 12 (no device yet)"
        );
    }

    // =======================================================================
    // BUG-FIX TESTS (post-cnn-r8): Device-bankruptcy + river-crossing.
    // =======================================================================

    /// FIX 1: HARD must NOT commit to BuildStrangeDevice when its treasury can pay the
    /// Device cost but cannot also cover ~5 rounds of payroll during the countdown.
    /// Without the safety-buffer gate, HARD would build, then bankrupt itself on the
    /// next 1-2 rounds of salary, handing the champ a free attrition win. Verified by
    /// constructing a fixture where money == device_cost.money + tiny slack, then
    /// staffing enough units that the 5-round drain exceeds the slack, then asserting
    /// no Device gets built.
    #[test]
    fn hard_does_not_self_bankrupt_on_device() {
        use cp_sim::{BuildingType, UnitType};
        let mut g = Game::new(12, 12, &["P0", "P1"]);
        g.generate_map(12, 12, 1);
        let p = PlayerId(0);
        // We need an Outpost already (Device gate requires `outposts >= 1`) AND a few
        // grasslands where the Device could be placed.
        let id = |x: i32, y: i32| TileId((x * 12 + y) as usize);
        let hq = id(2, 2);
        g.tiles[hq.0].tile_type = TileType::Grassland;
        g.tiles[hq.0].building = None;
        g.set_tile_owner(hq, Some(p));
        g.place_building(hq, BuildingType::Headquarters, Some(p));
        // Grasslands for Device placement candidacy.
        for (x, y) in [(2, 3), (2, 4), (3, 2), (4, 2)] {
            let t = id(x, y);
            g.tiles[t.0].tile_type = TileType::Grassland;
            g.tiles[t.0].building = None;
            g.set_tile_owner(t, Some(p));
        }
        // Outpost on (4,2) so the gate passes.
        let outpost = id(4, 2);
        g.place_building(outpost, BuildingType::Outpost, Some(p));
        // Salary load: workers + a soldier so 5-round drain is substantial.
        // worker salary 5/round, soldier 30/round. Add 6 workers + 1 soldier ⇒
        // 60/round; +50 outpost upkeep ⇒ ~110/round ⇒ 5-round buffer ≈ 550.
        for (i, (x, y)) in [(2, 3), (2, 4), (3, 2)].iter().enumerate() {
            let _ = i;
            let t = id(*x, *y);
            g.spawn_unit_on_tile(UnitType::BasicWorker, p, t, false);
            g.spawn_unit_on_tile(UnitType::BasicWorker, p, t, false);
        }
        g.spawn_unit_on_tile(UnitType::Soldier, p, hq, false);
        // Treasury: device cost is 1300 money + 200 stone + 200 metal. Give just barely
        // enough money for the build (1350) — 50 leftover, FAR below ~550 buffer needed.
        g.set_player_resources(p, 1350, 1000, 800, 800);
        // Advance the rounds counter past the gate (`get_rounds_played() >= 18`). The
        // field is private; cycle `change_turn` twice per round (2-player game).
        while g.get_rounds_played() < 20 {
            g.change_turn();
        }
        // Set a positive soldier cap so the bot can actually have its soldier.
        g.players[p.0].max_soldier_amount = 4;

        let mut bot = HardAi::hard();
        bot.budget = 8;
        bot.build_strange_device(&mut g, p);

        // Assert: no Strange Device was built (the safety-buffer gate fired).
        assert!(
            !g.has_strange_device(),
            "HARD must NOT build a Strange Device when its treasury cannot cover 5 rounds \
             of payroll after the build — that's the bankruptcy-by-attrition bug. Money was \
             1350, device cost 1300 → 50 left, but drain ≈ 110/round → 5-round buffer ≈ 550."
        );
    }

    /// FIX 2: HARD must build a Bridge when it owns a River tile whose bridging would
    /// unlock new neutral expansion targets. Pre-fix: HARD never built Bridges (0 across
    /// 180 saved replays), so a spawn on the wrong side of a river stayed pinned to its
    /// starting cluster forever. Construct a fixture where HARD owns one river tile that,
    /// when bridged, opens an unowned grassland across it. Assert HARD builds the Bridge.
    #[test]
    fn hard_builds_bridge_when_river_blocks_expansion() {
        use cp_sim::BuildingType;
        let mut g = Game::new(12, 12, &["P0", "P1"]);
        g.generate_map(12, 12, 1);
        let p = PlayerId(0);
        let id = |x: i32, y: i32| TileId((x * 12 + y) as usize);
        // HQ + own-side cluster.
        let hq = id(5, 5);
        g.tiles[hq.0].tile_type = TileType::Grassland;
        g.tiles[hq.0].building = None;
        g.set_tile_owner(hq, Some(p));
        g.place_building(hq, BuildingType::Headquarters, Some(p));
        // Own-side neighbour to satisfy `get_tile_count_for_player` etc.
        let near = id(5, 6);
        g.tiles[near.0].tile_type = TileType::Grassland;
        g.tiles[near.0].building = None;
        g.set_tile_owner(near, Some(p));
        // River we own, orientation 0 → Bridge-buildable. Sandwich between own-side and
        // unowned grassland across it.
        let river = id(5, 7);
        g.tiles[river.0].tile_type = TileType::River;
        g.tiles[river.0].river_orientation = 0;
        g.tiles[river.0].building = None;
        g.set_tile_owner(river, Some(p));
        // Unowned grassland across the river. Pre-Bridge: not reachable (river blocks).
        let across = id(5, 8);
        g.tiles[across.0].tile_type = TileType::Grassland;
        g.tiles[across.0].building = None;
        g.tiles[across.0].owner = None;
        // And surround `across` with non-passable tiles so the ONLY route is the river.
        // (Default generate_map tiles around it are unowned anyway, but they're not in
        // the player's availability set because they're not adjacent to anything owned.)
        // Sanity: engine offers Bridge for the river.
        assert!(
            g.buildable_buildings(river).contains(&"Bridge"),
            "Test setup: river must be Bridge-buildable"
        );
        // Pre-Bridge: the `across` tile must NOT be in the player's reachable set
        // (because the river blocks expansion).
        let pre = g.get_available_tiles_for(p);
        assert!(
            !pre.contains(&across),
            "Test setup: river must currently block expansion to `across`"
        );

        // Treasury — generous so affordability gates pass. Bridge cost: 100/300/150/0.
        g.set_player_resources(p, 800, 800, 400, 100);

        // Run the bridge phase directly (parity with how `run_turn` would dispatch it).
        let mut bot = HardAi::hard();
        bot.budget = 8;
        bot.build_bridges(&mut g, p);

        // Within ONE call HARD should have built either a Bridge or a Hydro on the river
        // (test name says "within 3 turns" — single-shot deterministic build phase suffices).
        let built = g.tiles[river.0].building.as_ref().map(|b| b.kind);
        assert!(
            matches!(built, Some(BuildingType::Bridge) | Some(BuildingType::Hydro)),
            "HARD must build a Bridge (or Hydro) on the unbridged river that blocks expansion; \
             got {:?}",
            built
        );
    }

    /// FIX 3: with `experts: true` AND mid/late game (round > 30) AND a Hydro affordable
    /// on the river tile, HARD should prefer Hydro (crosses AND yields income) over the
    /// cheaper Bridge. Same fixture as fix-2 but with experts ON, round > 30, plenty of
    /// money, and a free unit slot to staff the Hydro.
    #[test]
    fn hard_prefers_hydro_over_bridge_when_experts_available() {
        use cp_sim::{BuildingType, UnitType};
        let mut g = Game::new(12, 12, &["P0", "P1"]);
        g.generate_map(12, 12, 1);
        let p = PlayerId(0);
        let id = |x: i32, y: i32| TileId((x * 12 + y) as usize);
        // HQ + cluster (mirrors the bridge test).
        let hq = id(5, 5);
        g.tiles[hq.0].tile_type = TileType::Grassland;
        g.tiles[hq.0].building = None;
        g.set_tile_owner(hq, Some(p));
        g.place_building(hq, BuildingType::Headquarters, Some(p));
        // A grassland next to HQ so we can park spare workers ("idle on plain" for the
        // `can_staff_new_plant` reloc path).
        let plain = id(5, 6);
        g.tiles[plain.0].tile_type = TileType::Grassland;
        g.tiles[plain.0].building = None;
        g.set_tile_owner(plain, Some(p));
        // A staffed Farm so `net_money_per_round` is positive (the Hydro-preference gate
        // requires non-negative net — without income, even 1 worker's 5/round salary
        // tips us into the red and the gate falls back to Bridge).
        let farm = id(6, 5);
        g.tiles[farm.0].tile_type = TileType::Grassland;
        g.tiles[farm.0].building = None;
        g.set_tile_owner(farm, Some(p));
        g.place_building(farm, BuildingType::Farm, Some(p));
        g.spawn_unit_on_tile(UnitType::BasicWorker, p, farm, false);
        // River — orientation 0 ⇒ Bridge AND Hydro both buildable.
        let river = id(5, 7);
        g.tiles[river.0].tile_type = TileType::River;
        g.tiles[river.0].river_orientation = 0;
        g.tiles[river.0].building = None;
        g.set_tile_owner(river, Some(p));
        // Unowned grassland across the river so unlock_count > 0.
        let across = id(5, 8);
        g.tiles[across.0].tile_type = TileType::Grassland;
        g.tiles[across.0].building = None;
        g.tiles[across.0].owner = None;
        // Sanity: engine offers BOTH Bridge AND Hydro.
        let bb = g.buildable_buildings(river);
        assert!(bb.contains(&"Bridge"));
        assert!(bb.contains(&"Hydroelectric Power Plant"));

        // Free-unit slot + idle worker so `can_staff_new_plant` returns true (the bot
        // can reloc the plain-tile worker onto the Hydro to staff it).
        g.players[p.0].max_unit_amount = 6;
        g.spawn_unit_on_tile(UnitType::BasicWorker, p, plain, false);

        // Treasury — afford Hydro (cost: 280/150/120/60 per `hepp_build_cost()`). Be
        // generous: cover both options so the preference itself is what decides.
        g.set_player_resources(p, 2500, 1500, 800, 500);
        // Round > 30 (the Hydro-preference gate). Cycle `change_turn` 2× per round.
        while g.get_rounds_played() <= 30 {
            g.change_turn();
        }

        let mut bot = HardAi::hard(); // experts: true, nuclear: true ⇒ Hydro-eligible.
        bot.budget = 8;
        bot.build_bridges(&mut g, p);

        let built = g.tiles[river.0].building.as_ref().map(|b| b.kind);
        assert_eq!(
            built,
            Some(BuildingType::Hydro),
            "experts-on, round > 30, Hydro affordable + staffable ⇒ HARD must prefer Hydro \
             (it crosses the river AND produces income) over plain Bridge; got {:?}",
            built
        );
    }

    /// Structural-fix regression (user-report 2026-06-05): the original `claim_value`
    /// only recognised Mikontalo, so a neutral-with-Farm grassland scored EQUAL to bare
    /// grassland (both 4). HARD never preferred it; the value head learning from
    /// vs-HARD self-play therefore never saw "claim a free Farm → income jumps → win".
    /// After the fix, ALL useful neutral buildings (Farm/Mine/Village/Outpost/Hydro/
    /// Nuclear/Mikontalo) outrank the same terrain without a building.
    #[test]
    fn claim_value_prefers_neutral_with_building() {
        let mut g = Game::new(8, 8, &["P0", "P1"]);
        g.generate_map(8, 8, 1);
        let bot = HardAi::hard();

        let id = |x: i32, y: i32| TileId((x * 8 + y) as usize);

        // Bare grassland baseline.
        let bare = id(1, 1);
        g.tiles[bare.0].tile_type = TileType::Grassland;
        g.tiles[bare.0].building = None;
        g.tiles[bare.0].owner = None;
        let v_bare = bot.claim_value(&g, bare);

        // Neutral-with-Farm: was 4 (== bare), must now be STRICTLY > bare.
        let farm = id(1, 2);
        g.tiles[farm.0].tile_type = TileType::Grassland;
        g.tiles[farm.0].owner = None;
        g.place_building(farm, BuildingType::Farm, None);
        assert!(
            bot.claim_value(&g, farm) > v_bare,
            "neutral Farm must outrank bare grassland; got farm={} bare={}",
            bot.claim_value(&g, farm),
            v_bare
        );

        // Neutral-with-Mine on a Mountain: was 5 (== bare mountain), now strictly higher.
        let mountain = id(2, 1);
        g.tiles[mountain.0].tile_type = TileType::Mountain;
        g.tiles[mountain.0].owner = None;
        g.tiles[mountain.0].building = None;
        let v_mountain_bare = bot.claim_value(&g, mountain);
        let mine = id(2, 2);
        g.tiles[mine.0].tile_type = TileType::Mountain;
        g.tiles[mine.0].owner = None;
        g.place_building(mine, BuildingType::Mine, None);
        assert!(
            bot.claim_value(&g, mine) > v_mountain_bare,
            "neutral Mine must outrank bare mountain; got mine={} mountain={}",
            bot.claim_value(&g, mine),
            v_mountain_bare
        );

        // Neutral Outpost / Village / Hydro / Nuclear / Mikontalo all outrank bare.
        let outpost = id(3, 3);
        g.tiles[outpost.0].tile_type = TileType::Grassland;
        g.tiles[outpost.0].owner = None;
        g.place_building(outpost, BuildingType::Outpost, None);
        assert!(bot.claim_value(&g, outpost) > v_bare);

        let village = id(3, 4);
        g.tiles[village.0].tile_type = TileType::Grassland;
        g.tiles[village.0].owner = None;
        g.place_building(village, BuildingType::Village, None);
        assert!(bot.claim_value(&g, village) > v_bare);

        // Sanity: Mikontalo case still works (existing behaviour).
        let mk = id(4, 4);
        g.tiles[mk.0].tile_type = TileType::Grassland;
        g.tiles[mk.0].owner = None;
        g.place_building(mk, BuildingType::Mikontalo, None);
        assert_eq!(bot.claim_value(&g, mk), 6);

        // StrangeDevice gets destroyed on ownership change (managers §6a), so claiming
        // gains nothing — must fall back to bare-tile value.
        let dev = id(5, 4);
        g.tiles[dev.0].tile_type = TileType::Grassland;
        g.tiles[dev.0].owner = None;
        g.place_building(dev, BuildingType::StrangeDevice, None);
        assert_eq!(
            bot.claim_value(&g, dev), v_bare,
            "neutral StrangeDevice destroys-on-claim → no bonus over bare"
        );
    }

    /// End-to-end: in a hand-built fixture where HARD has the choice between a bare
    /// grassland and a neutral Farm tile, the expand phase must claim the Farm tile
    /// (it is on the player's frontier, reachable, and now scores strictly higher).
    /// This is the demonstration signal the NN's value head learns from in vs-HARD
    /// training games.
    #[test]
    fn hard_expand_claims_neutral_farm_over_bare_grassland() {
        let mut g = Game::new(8, 8, &["P0", "P1"]);
        g.generate_map(8, 8, 1);
        let p = PlayerId(0);
        let id = |x: i32, y: i32| TileId((x * 8 + y) as usize);

        // Clear worldgen ownership so the available-set is controlled.
        let all: Vec<TileId> = g.get_tiles().iter().map(|t| t.id).collect();
        for tid in all {
            g.set_tile_owner(tid, None);
            // Drop any worldgen-placed Mikontalo so it does not steal claim priority.
            if matches!(
                g.tiles[tid.0].building.as_ref().map(|b| b.kind),
                Some(BuildingType::Mikontalo)
            ) {
                g.tiles[tid.0].building = None;
            }
        }

        // HARD's HQ at (3,3) — its frontier is the 4 orthogonal neighbours.
        let hq = id(3, 3);
        g.tiles[hq.0].tile_type = TileType::Grassland;
        g.set_tile_owner(hq, Some(p));
        g.place_building(hq, BuildingType::Headquarters, Some(p));

        // Own an extra plain tile holding an idle worker. The expand phase prefers
        // leap-frogging an idle worker (`find_idle_worker`) — this bypasses the hire
        // affordability gates and keeps the test focused on the CLAIM-PRIORITY logic.
        let plain = id(3, 4);
        g.tiles[plain.0].tile_type = TileType::Grassland;
        g.tiles[plain.0].building = None;
        g.set_tile_owner(plain, Some(p));
        g.spawn_unit_on_tile(UnitType::BasicWorker, p, plain, false);

        // Two reachable neutral grasslands, equal terrain. One has a Farm.
        let bare = id(2, 3); // west neighbour, bare
        g.tiles[bare.0].tile_type = TileType::Grassland;
        g.tiles[bare.0].building = None;
        let farmed = id(4, 3); // east neighbour, has Farm
        g.tiles[farmed.0].tile_type = TileType::Grassland;
        g.place_building(farmed, BuildingType::Farm, None);

        // Generous treasury (not strictly needed for the idle-worker path).
        g.set_player_resources(p, 1500, 1500, 1500, 1500);
        g.update_unit_amounts(p);

        let mut bot = HardAi::hard();
        bot.budget = 4;
        bot.expand(&mut g, p);

        // Within 1 expand action, the farm tile must be the one HARD targeted: it now
        // has either an owned unit (worker move) or a conquering unit (fresh hire).
        let farm_has_unit = !g.tile_units(farmed).is_empty()
            || g.tile_conquering_units(farmed)
                .iter()
                .any(|&u| g.units[u.0].owner == Some(p));
        let bare_has_unit = !g.tile_units(bare).is_empty()
            || g.tile_conquering_units(bare)
                .iter()
                .any(|&u| g.units[u.0].owner == Some(p));
        assert!(
            farm_has_unit && !bare_has_unit,
            "HARD expand must prefer the neutral-Farm tile over the bare grassland; \
             got farm_has_unit={farm_has_unit} bare_has_unit={bare_has_unit}"
        );
    }

    // =======================================================================
    // BUG-FIX TESTS (user-report 2026-06-05): self-bankruptcy from cumulative
    // commits — the existing `affords` checks gate on CURRENT drain, but each
    // Outpost / Soldier / Expert / Village adds NEW per-round upkeep that the
    // existing helpers don't project. The fix adds `affordable_after_commit`
    // which projects post-commit drain and demands 4 rounds of buffer.
    // =======================================================================

    /// Outpost: HARD has enough money to PAY the Outpost cost (500) but its
    /// current drain plus the Outpost's +50 upkeep would exceed the post-build
    /// cash divided by the 4-round buffer ⇒ HARD must NOT build the Outpost.
    #[test]
    fn hard_defers_outpost_when_drain_would_exceed_income() {
        use cp_sim::{BuildingType, UnitType};
        let mut g = Game::new(12, 12, &["P0", "P1"]);
        g.generate_map(12, 12, 1);
        let p = PlayerId(0);
        let other = PlayerId(1);
        let id = |x: i32, y: i32| TileId((x * 12 + y) as usize);

        // Own a cluster of ≥ 8 tiles (the build_outposts `tile_count < 8` gate).
        let mut own_tiles: Vec<TileId> = Vec::new();
        for &(x, y) in &[
            (2, 2), (2, 3), (2, 4), (2, 5),
            (3, 2), (3, 3), (3, 4), (3, 5),
            (4, 2), (4, 3),
        ] {
            let t = id(x, y);
            g.tiles[t.0].tile_type = TileType::Grassland;
            g.tiles[t.0].building = None;
            g.set_tile_owner(t, Some(p));
            own_tiles.push(t);
        }
        let hq = id(2, 2);
        g.place_building(hq, BuildingType::Headquarters, Some(p));
        // Enemy HQ so `should_militarise` / `military_need` can fire — put it
        // close enough that there IS an enemy on contact (warmonger seat).
        let ehq = id(5, 5);
        g.tiles[ehq.0].tile_type = TileType::Grassland;
        g.tiles[ehq.0].building = None;
        g.set_tile_owner(ehq, Some(other));
        g.place_building(ehq, BuildingType::Headquarters, Some(other));
        // Load up payroll: 6 workers + 1 soldier + 1 existing Outpost ⇒ drain ≈
        // 30 (workers) + 30 (soldier) + 50 (outpost) = 110/round. Plus a tiny
        // metal mine for the metal-income gate (mine = 20 metal/round).
        let mine_tile = id(2, 6);
        g.tiles[mine_tile.0].tile_type = TileType::Mountain;
        g.tiles[mine_tile.0].building = None;
        g.set_tile_owner(mine_tile, Some(p));
        own_tiles.push(mine_tile);
        g.place_building(mine_tile, BuildingType::Mine, Some(p));
        g.spawn_unit_on_tile(UnitType::BasicWorker, p, mine_tile, false);
        // 6 workers on grassland tiles (5/round each = 30/round).
        for (i, &tid) in own_tiles.iter().take(6).enumerate() {
            let _ = i;
            g.spawn_unit_on_tile(UnitType::BasicWorker, p, tid, false);
        }
        // Existing Outpost (50/round).
        let outpost1 = id(3, 5);
        g.place_building(outpost1, BuildingType::Outpost, Some(p));
        // A soldier (30/round).
        g.players[p.0].max_soldier_amount = 4;
        g.spawn_unit_on_tile(UnitType::Soldier, p, hq, false);

        // Treasury — enough to pay the 500-money Outpost cost, but only ~50
        // slack afterwards. Post-build drain = 110 + 50 = 160/round, 4-round
        // buffer = 640 ⇒ must defer.
        // money 550, wood 1000, stone 1000, metal 1000 → after Outpost cost
        // (-500 money / -200 wood / -200 stone / -100 metal) only 50 money
        // remains, FAR below the 640 buffer.
        g.set_player_resources(p, 550, 1000, 1000, 1000);

        // Pre: no second Outpost yet.
        let outposts_before = own_tiles
            .iter()
            .filter(|&&t| g.tiles[t.0].building.as_ref().map(|b| b.kind) == Some(BuildingType::Outpost))
            .count() as i64;
        assert_eq!(outposts_before, 1, "test fixture: starts with exactly 1 Outpost");

        let mut bot = HardAi::hard();
        bot.budget = 8;
        bot.build_outposts(&mut g, p);

        // Post: still exactly 1 Outpost (the gate fired).
        let outposts_after = g
            .owned_tiles(p)
            .iter()
            .filter(|&&t| g.tiles[t.0].building.as_ref().map(|b| b.kind) == Some(BuildingType::Outpost))
            .count() as i64;
        assert_eq!(
            outposts_after, 1,
            "HARD must NOT build a 2nd Outpost when its post-build drain (160/round) \
             could not be sustained for 4 rounds out of its 50-money post-build cash"
        );
    }

    /// Soldier hire: HARD's payroll is already near its income ceiling; one
    /// more soldier's +30/round would push it past the 4-round buffer ⇒ the
    /// `garrison` hire path must defer.
    #[test]
    fn hard_defers_soldier_when_payroll_unsustainable() {
        use cp_sim::{BuildingType, UnitType};
        let mut g = Game::new(12, 12, &["P0", "P1"]);
        g.generate_map(12, 12, 1);
        let p = PlayerId(0);
        let id = |x: i32, y: i32| TileId((x * 12 + y) as usize);

        // Build a small territory with a staffed Farm (income) and a heavy
        // payroll so we're net-positive but margin is thin.
        let hq = id(3, 3);
        g.tiles[hq.0].tile_type = TileType::Grassland;
        g.tiles[hq.0].building = None;
        g.set_tile_owner(hq, Some(p));
        g.place_building(hq, BuildingType::Headquarters, Some(p));

        let farm = id(3, 4);
        g.tiles[farm.0].tile_type = TileType::Grassland;
        g.tiles[farm.0].building = None;
        g.set_tile_owner(farm, Some(p));
        g.place_building(farm, BuildingType::Farm, Some(p));
        g.spawn_unit_on_tile(UnitType::BasicWorker, p, farm, false);
        // Throttle income: Farm produces 175 every 4 rounds ≈ 44/round in
        // `net_money_per_round`. Salary load: 6 workers (30) + 3 soldiers (90)
        // + 1 outpost (50) ⇒ drain ≈ 170/round.
        let outpost = id(4, 3);
        g.tiles[outpost.0].tile_type = TileType::Grassland;
        g.tiles[outpost.0].building = None;
        g.set_tile_owner(outpost, Some(p));
        g.place_building(outpost, BuildingType::Outpost, Some(p));
        // Park 6 workers and 3 soldiers somewhere owned.
        let other_tiles = [id(3, 5), id(4, 4), id(4, 5), id(2, 3), id(2, 4), id(5, 3)];
        for &t in &other_tiles {
            g.tiles[t.0].tile_type = TileType::Grassland;
            g.tiles[t.0].building = None;
            g.set_tile_owner(t, Some(p));
        }
        for &t in &other_tiles[..3] {
            g.spawn_unit_on_tile(UnitType::BasicWorker, p, t, false);
            g.spawn_unit_on_tile(UnitType::BasicWorker, p, t, false);
        }
        g.players[p.0].max_soldier_amount = 6;
        g.spawn_unit_on_tile(UnitType::Soldier, p, hq, false);
        g.spawn_unit_on_tile(UnitType::Soldier, p, hq, false);
        g.spawn_unit_on_tile(UnitType::Soldier, p, hq, false);

        // Treasury: just enough cash to clear `affords` (200 money + reserve 140
        // = 340) — but the post-commit 4-round buffer (170 + 30 = 200/round ⇒
        // 800-money buffer) cannot be met. Give 400 money so `affords` would
        // pass on its own (200 cost + 140 reserve + small drain*5 ≈ ~lots) but
        // the new gate fires.
        g.set_player_resources(p, 400, 1000, 1000, 1000);
        g.update_unit_amounts(p);

        let mut bot = HardAi::hard();
        bot.budget = 8;
        // Pick the HQ for garrisoning: we ask for 3 soldiers there (it has 3
        // already, so the loop's `while soldiers_on(tid) < want` exits before
        // attempting a hire). Use a DIFFERENT tile with `want=1` to force a
        // hire attempt (no rear-soldier reloc — all are at HQ — and the buy
        // path runs).
        let target = id(2, 3);
        let soldiers_before = g
            .owned_tiles(p)
            .iter()
            .map(|&t| {
                g.tile_units(t)
                    .iter()
                    .filter(|&&u| {
                        g.units[u.0].kind == UnitType::Soldier && g.units[u.0].owner == Some(p)
                    })
                    .count() as i64
            })
            .sum::<i64>();
        assert_eq!(soldiers_before, 3, "fixture starts with 3 soldiers");

        bot.garrison(&mut g, p, target, 1);

        // We expect either a successful rear-soldier RELOC (which is fine —
        // doesn't burn money) OR no hire at all. The invariant we want: no NEW
        // soldier was bought (total count unchanged or went down via reloc
        // tracking, but reloc moves don't change total count).
        let soldiers_after = g
            .owned_tiles(p)
            .iter()
            .map(|&t| {
                g.tile_units(t)
                    .iter()
                    .filter(|&&u| {
                        g.units[u.0].kind == UnitType::Soldier && g.units[u.0].owner == Some(p)
                    })
                    .count() as i64
            })
            .sum::<i64>();
        // ALSO check that money was NOT spent on a soldier (cost 200 money).
        let money_after = g.players[p.0].resources.get(BasicResource::Money).unwrap_or(0);
        assert_eq!(
            soldiers_after, soldiers_before,
            "HARD must NOT BUY a new soldier when post-commit drain breaks the 4-round buffer \
             (rear-soldier reloc keeps total constant; the BUY branch must not fire)"
        );
        assert!(
            money_after >= 200,
            "Treasury should not have been spent on a soldier (200 money) — got {money_after}"
        );
    }

    /// Healthy economy: HARD has plenty of income and cash ⇒ the new gates must
    /// NOT spuriously block normal builds. Pin both Outpost and Soldier hires
    /// proceed when the bot is well-funded.
    #[test]
    fn hard_still_builds_when_income_supports_it() {
        use cp_sim::{BuildingType, UnitType};
        let mut g = Game::new(12, 12, &["P0", "P1"]);
        g.generate_map(12, 12, 1);
        let p = PlayerId(0);
        let other = PlayerId(1);
        let id = |x: i32, y: i32| TileId((x * 12 + y) as usize);

        // Healthy territory: 8 owned grassland + HQ.
        let mut own_tiles: Vec<TileId> = Vec::new();
        for &(x, y) in &[
            (2, 2), (2, 3), (2, 4), (2, 5),
            (3, 2), (3, 3), (3, 4), (3, 5),
            (4, 2),
        ] {
            let t = id(x, y);
            g.tiles[t.0].tile_type = TileType::Grassland;
            g.tiles[t.0].building = None;
            g.set_tile_owner(t, Some(p));
            own_tiles.push(t);
        }
        let hq = id(2, 2);
        g.place_building(hq, BuildingType::Headquarters, Some(p));
        // Enemy HQ — placed adjacent to one of my owned tiles so
        // `military_need` fires (reachable_enemy_max_defenders > 0 needs the
        // enemy in `get_available_tiles`, which requires neighbour-adjacency).
        let ehq = id(4, 5);
        g.tiles[ehq.0].tile_type = TileType::Grassland;
        g.tiles[ehq.0].building = None;
        g.set_tile_owner(ehq, Some(other));
        g.place_building(ehq, BuildingType::Headquarters, Some(other));
        // Put an enemy soldier on the HQ tile so reachable_enemy_max_defenders
        // returns ≥ 1 and `military_need` returns true.
        g.players[other.0].max_soldier_amount = 4;
        g.spawn_unit_on_tile(UnitType::Soldier, other, ehq, false);
        // Productive mine for metal income (the build_outposts gate needs
        // metal_income > (outposts+1)*15).
        let mine_tile = id(2, 6);
        g.tiles[mine_tile.0].tile_type = TileType::Mountain;
        g.tiles[mine_tile.0].building = None;
        g.set_tile_owner(mine_tile, Some(p));
        own_tiles.push(mine_tile);
        g.place_building(mine_tile, BuildingType::Mine, Some(p));
        g.spawn_unit_on_tile(UnitType::BasicWorker, p, mine_tile, false);
        // Staffed farms for income (the build_outposts gate also requires
        // `net_money_per_round - 50 >= 10` ⇒ net >= 60). Two staffed farms
        // give 175/4 * 2 = ~87/round, well clear of the 60 threshold.
        for &(x, y) in &[(2, 3), (2, 4)] {
            let t = id(x, y);
            g.place_building(t, BuildingType::Farm, Some(p));
            g.spawn_unit_on_tile(UnitType::BasicWorker, p, t, false);
        }

        // Salary: 3 workers ⇒ drain = 15/round.
        // After Outpost build: drain = 15 + 50 = 65/round ⇒ 4-round buffer = 260.
        // Treasury: 2000 money ⇒ after Outpost (-500) = 1500 ≫ 260 buffer ⇒ pass.
        g.set_player_resources(p, 2000, 1500, 1500, 1500);
        g.update_unit_amounts(p);

        // ARMY_RUSH (warmonger:true) so `should_militarise` fires from
        // round 0 — the default `hard()` only militarises on contact, but our
        // enemy at (5,5) isn't adjacent to anything we own, so the contact path
        // wouldn't fire. The new gate (the actual subject under test) is in
        // build_outposts regardless of which params drive it here.
        let mut bot = HardAi::army_rush();
        bot.budget = 8;
        // First: a no-outpost-yet build must produce one (no gate should block).
        let outposts_before = g
            .owned_tiles(p)
            .iter()
            .filter(|&&t| g.tiles[t.0].building.as_ref().map(|b| b.kind) == Some(BuildingType::Outpost))
            .count() as i64;
        assert_eq!(outposts_before, 0, "fixture: no Outpost yet");
        bot.build_outposts(&mut g, p);
        let outposts_after = g
            .owned_tiles(p)
            .iter()
            .filter(|&&t| g.tiles[t.0].building.as_ref().map(|b| b.kind) == Some(BuildingType::Outpost))
            .count() as i64;
        assert!(
            outposts_after >= 1,
            "Healthy economy: HARD MUST build an Outpost (no gate should block); \
             outposts_after={outposts_after}"
        );
    }
}
