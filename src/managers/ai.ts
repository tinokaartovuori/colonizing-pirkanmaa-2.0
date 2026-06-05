// Computer-player logic. Drives a CPU through HQ placement and a full turn using
// the same GameEventHandler actions a human would trigger.
//
// A turn is a *generator* that yields after every individual action (build, hire,
// move). The app layer advances it one action per tick so the human watches the
// CPU play move-by-move; the tests drain it synchronously via playTurn().
//
// Strategy (a bounded heuristic, not a search): keep every building staffed, grow
// the economy, and — crucially — break the unit-cap ceiling by building Villages
// and grabbing Mikontalo, because more workers is what compounds. It mines for
// metal, stacks workers on mines/power plants (their output scales per worker),
// expands toward the most valuable neutral land, and fields soldiers to defend and
// conquer. Difficulty scales how much it does and how aggressive it is.

import { TileBase } from '../model/tile';
import { Grassland, Forest, AbundantForest, Mountain, River } from '../model/tiles';
import { UnitBase } from '../model/unit';
import { PlayerBase, CpuDifficulty } from '../model/player';
import {
  BasicResource,
  ResourceMap,
  FARM_BUILD_COST,
  MINE_BUILD_COST,
  VILLAGE_BUILD_COST,
  NUCLEARPP_BUILD_COST,
  HEPP_BUILD_COST,
  OUTPOST_BUILD_COST,
  STRANGE_DEVICE_BUILD_COST,
  BASIC_WORKER_COST,
  EXPERT_COST,
  SOLDIER_COST,
} from '../core/resources';
import type { GameEventHandler } from './gameeventhandler';
import type { ObjectManager } from './objectmanager';
import type { PlayerManager } from './playermanager';

export interface AiParams {
  /** Money kept in reserve before discretionary (non-staffing) spending. */
  reserve: number;
  /** Max individual actions per turn (caps turn length / pacing). */
  maxActions: number;
  /** Hire experts to boost mines and run power plants. */
  experts: boolean;
  /** Field soldiers (defence + offence). */
  military: boolean;
  /** Base garrison size near the HQ, scaled up by owned tiles. */
  garrison: number;
  /** Neutral tiles claimed per turn. */
  expand: number;
  /** Send soldiers to conquer weak enemy tiles. */
  attack: boolean;
  /** Invest in a nuclear plant when very wealthy. */
  nuclear: boolean;
  /** Max Outposts to build — each raises the soldier cap by 3 (the gate on armies). */
  maxOutposts: number;
  /** Extra offensive soldiers to field (beyond the garrison) when an enemy is reachable. */
  strikeForce: number;
  /** Max separate enemy tiles to assault in a single turn. 1 = the old "one assault
   *  per turn" behaviour (slow, stalemate-prone); higher lets the whole front advance
   *  each turn, which is what actually ends games by conquest. */
  assaultsPerTurn: number;
  /** Pre-build the army instead of waiting for an enemy on the doorstep. When true,
   *  outposts (the soldier-cap gate) and the strike force are raised as soon as an
   *  opponent is alive and the economy can carry them — so the AI is cap-locked at 1
   *  soldier far less often and can actually close out a won economy by conquest. */
  warmonger: boolean;
  /** Use the Strange Device WIN strategy: when clearly leading, build the Device to force
   *  a decisive finish (it halves our soldier cap, so it's a deliberate gamble). This gates
   *  only the BUILD decision — the counterplay (massing soldiers and assaulting an enemy
   *  Device on sight) is always on, so a device:false AI still races to crack one. */
  device: boolean;
}

export const PARAMS: Record<CpuDifficulty, AiParams> = {
  // Easy: a slow economy and only a token army — enough that games still resolve, but
  // a gentle opponent. Medium: a solid economy, an outpost and a real garrison + raids.
  // Hard: aggressive economy and military — builds several outposts and hunts HQs.
  //
  // assaultsPerTurn is the lever that decides games: with it at 1 the AI grinds at most
  // one enemy tile per turn and ~45% of CPU-vs-CPU matches stalled to a timeout (a huge
  // economy that could never close out). Raising it — so the whole front advances each
  // turn and soldiers attack on hand rather than hoarding cash — measurably beats the
  // old AI head-to-head (hard: 61% of decided 2p games, ~45% of 4p vs a 25% fair share)
  // with no extra bankruptcies. See sim/ for the data. easy stays at 1 on purpose.
  easy: {
    reserve: 80, maxActions: 5, experts: false, military: true,
    garrison: 1, expand: 2, attack: true, nuclear: false, maxOutposts: 1, strikeForce: 1,
    assaultsPerTurn: 1, warmonger: false, device: false,
  },
  medium: {
    reserve: 110, maxActions: 14, experts: true, military: true,
    garrison: 2, expand: 3, attack: true, nuclear: false, maxOutposts: 2, strikeForce: 3,
    assaultsPerTurn: 4, warmonger: false, device: true,
  },
  // Hard is meant to be a genuine challenge: it raises its soldier cap with several
  // outposts (each +3 cap and an impregnable strongpoint), fields a real strike force,
  // garrisons what matters, and presses the whole front — so it actually closes games
  // by conquest instead of stalling out. The extra action budget pays for the larger
  // army; the solvency guards still keep it from ever bankrupting itself.
  hard: {
    reserve: 140, maxActions: 28, experts: true, military: true,
    garrison: 3, expand: 5, attack: true, nuclear: true, maxOutposts: 5, strikeForce: 7,
    assaultsPerTurn: 7, warmonger: false, device: true,
  },
};

export class AiController {
  /** Small buffer kept when hiring a worker for a net-positive building. */
  private static readonly STAFF_RESERVE = 20;
  /** Remaining action budget for the current turn (drives pacing/caps). */
  private budget = 0;

  /**
   * Optional per-difficulty parameter overrides, merged onto the built-in PARAMS.
   * Lets a variant AI be pitted against the baseline in simulations/experiments
   * without forking the class. In production this is left undefined.
   */
  constructor(
    private eh: GameEventHandler,
    private om: ObjectManager,
    private pm: PlayerManager,
    private paramOverride?: Partial<AiParams>,
  ) {}

  // --- first round ----------------------------------------------------------

  /** Choose and claim a starting tile (places the HQ near good resources). */
  placeHeadquarters(player: PlayerBase): void {
    // Candidates must be BUILDABLE: unowned AND empty (first-round HQ placement is
    // refused on a tile that already holds a building, e.g. an unowned Mikontalo —
    // picking one left the player with 0 tiles → instant loss). Prefer grassland,
    // then any non-river land, then any tile.
    const empty = (t: TileBase) => t.getOwner() === null && t.getBuilding() === null;
    let candidates = this.om.getTiles().filter((t) => t.getType() === 'Grassland' && empty(t));
    if (candidates.length === 0) candidates = this.om.getTiles().filter((t) => empty(t) && t.getType() !== 'River');
    if (candidates.length === 0) candidates = this.om.getTiles().filter((t) => empty(t));
    if (candidates.length === 0) return;

    let best = candidates[0];
    let bestScore = -Infinity;
    for (const tile of candidates) {
      const neighbours = tile.getNeighbourTiles();
      const free = neighbours.filter((n) => n.getOwner() === null).length;
      const forests = neighbours.filter((n) => n.getType() === 'Forest').length;
      const mountains = neighbours.filter((n) => n.getType() === 'Mountain').length;
      const grass = neighbours.filter((n) => n.getType() === 'Grassland').length;
      const distance = Math.min(this.distanceToNearestOwned(tile), 8);
      // Want room to expand, building land, and nearby wood + metal.
      const score = free * 3 + grass * 2 + forests * 2 + mountains * 3 + distance;
      if (score > bestScore) {
        bestScore = score;
        best = tile;
      }
    }
    this.eh.tileClicked(best);
  }

  private distanceToNearestOwned(tile: TileBase): number {
    let min = Infinity;
    const c = tile.getCoordinate();
    for (const other of this.om.getTiles()) {
      if (other.getOwner() === null) continue;
      const oc = other.getCoordinate();
      const d = Math.abs(oc.x() - c.x()) + Math.abs(oc.y() - c.y());
      if (d < min) min = d;
    }
    return min === Infinity ? 99 : min;
  }

  // --- turn -----------------------------------------------------------------

  /** Synchronous full turn (used by the tests). */
  playTurn(player: PlayerBase): void {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    for (const _ of this.planTurn(player)) {
      /* drain every action immediately */
    }
  }

  /** A turn as a sequence of actions; yields once per action so a driver can pace it. */
  *planTurn(player: PlayerBase): Generator<void> {
    const base = PARAMS[player.getDifficulty() as CpuDifficulty] ?? PARAMS.medium;
    let params: AiParams = this.paramOverride ? { ...base, ...this.paramOverride } : base;
    // PANIC MODE: an enemy who builds a Device halves their own soldier cap — that is the
    // window to strike. Go all-in for the turn: spend the reserve, press every front, and
    // field a real army (attack() already targets the Device first). The per-buy upkeep
    // guard in garrison() still prevents literal bankruptcy. Always on (not gated on
    // params.device), so even a non-building CPU mounts the counterplay.
    const panic = this.enemyHasDevice(player);
    if (panic) {
      params = {
        ...params,
        reserve: Math.max(40, Math.floor(params.reserve / 4)),
        assaultsPerTurn: Math.max(12, params.assaultsPerTurn),
      };
    }
    this.budget = params.maxActions + (panic ? 12 : 0);
    try {
      // The unit cap (3 from the HQ alone) is the whole game: every worker needs a
      // slot, and only Villages (+3) and Mikontalo (+2) raise it. So the plan is:
      // keep income running, raise the cap, then spend the freed slots on territory.
      //
      // Income first, always — a CPU that stops staffing farms bleeds itself dry on
      // salaries. Then keep one forest worker so wood climbs toward the mine/village
      // thresholds, then build the economy, raise the cap, and expand.
      yield* this.ensureWoodIncome(player); // CRITICAL (first): cover village/bridge wood upkeep
      yield* this.staffBuildings(player, params); // staff existing income buildings
      yield* this.secureWood(player, params); // keep ≥1 forest worker harvesting wood

      // "Saving for a mine": we own a mountain but no mine yet, and wood is below the
      // 250 a mine costs. A mine (metal) unlocks villages and soldiers, so we stop
      // spending wood on *new* farms and let it accumulate — but ONLY once we already
      // have enough staffed farms to stay comfortably solvent (income > salaries).
      const savingForMine =
        this.staffedFarmCount(player) >= 2 &&
        this.ownedTiles(player).some((t) => t instanceof Mountain && t.getBuilding() === null) &&
        this.wood(player) < 270;

      if (!savingForMine) {
        yield* this.buildFarms(player, params); // farms (and convert idle-worker grassland)
        yield* this.staffBuildings(player, params);
      }
      yield* this.buildMines(player, params); // build a mine once wood/money allow
      yield* this.staffBuildings(player, params); // staff the new mine (relocates a worker)
      yield* this.boostMines(player, params); // expert per mine — doubles metal+money (industry)
      yield* this.buildPowerPlants(player, params); // diversify income: hydro plants on rivers
      yield* this.investNuclear(player, params); // late-game: turn a saved-up hoard into nuclear plants
      yield* this.buildOutposts(player, params); // defensive strongpoint + soldier cap
      yield* this.raiseUnitCap(player, params); // villages — the key to growth (frees slots)
      yield* this.expand(player, params); // claim valuable neutral land (Mikontalo, mountains)
      yield* this.buildStrangeDevice(player, params); // when leading: race the Device to a decisive win
      yield* this.military(player, params); // garrison & defence (reacts to threats)
      yield* this.attack(player, params); // conquer weak neighbours with a full strike force
      // Slot-soakers run LAST and leave a slot free for next turn's scout when there
      // is still neutral land to claim — otherwise pile leftovers onto producers.
      yield* this.stackProducers(player, params); // extra workers on mines/plants
      yield* this.fillSpareSlots(player, params); // never leave workers unhired
    } catch {
      /* never let a CPU crash the game */
    }
  }

  /** Perform one action; on success it counts against the budget and yields. */
  private *doAction(fn: () => boolean): Generator<void, boolean, unknown> {
    if (this.budget <= 0) return false;
    let ok = false;
    try {
      ok = fn();
    } catch {
      ok = false;
    }
    if (ok) {
      this.budget -= 1;
      yield;
    }
    return ok;
  }

  // --- resource helpers -----------------------------------------------------

  private res(p: PlayerBase, r: BasicResource): number {
    return p.getResources().get(r) ?? 0;
  }
  private money(p: PlayerBase): number { return this.res(p, BasicResource.MONEY); }
  private wood(p: PlayerBase): number { return this.res(p, BasicResource.WOOD); }
  private stone(p: PlayerBase): number { return this.res(p, BasicResource.STONE); }
  private metal(p: PlayerBase): number { return this.res(p, BasicResource.METAL); }

  /** Total wages owed per round by this player's units. */
  private salaryPerRound(p: PlayerBase): number {
    return p.getCurrentBasicWorkerAmount() * 5 + p.getCurrentExpertAmount() * 25 + p.getCurrentSoldierAmount() * 30;
  }

  /** Total money leaving the treasury each round: wages PLUS building upkeep (Villages
   *  −10, Outposts −50). The cash buffer must cover this — counting only wages used to
   *  under-reserve, so the lumpy farm-payout cycle could dip a village-heavy economy
   *  into the red. */
  private moneyDrainPerRound(p: PlayerBase): number {
    let upkeep = 0;
    for (const t of this.ownedTiles(p)) {
      const type = t.getBuilding()?.getType();
      if (type === 'Village') upkeep += 10;
      if (type === 'Outpost') upkeep += 50;
    }
    return this.salaryPerRound(p) + upkeep;
  }

  /** Number of farms that currently have a worker (i.e. actually producing money). */
  private staffedFarmCount(p: PlayerBase): number {
    return this.ownedTiles(p).filter((t) => t.getBuilding()?.getType() === 'Farm' && this.hasType(t, 'BasicWorker'))
      .length;
  }

  /**
   * Estimated steady-state money change per round: amortised building income minus
   * salaries and building upkeep. Farms pay 175 every 4 rounds (≈44/round); mines pay
   * 20/worker/round (doubled by an expert); villages cost 10/round upkeep. Used to
   * stop the CPU hiring salaried units it can't sustain.
   */
  private netMoneyPerRound(p: PlayerBase): number {
    let income = 0;
    for (const tile of this.ownedTiles(p)) {
      const type = tile.getBuilding()?.getType();
      const workers = tile.getUnits().filter((u) => u.getType() === 'BasicWorker').length;
      const hasExpert = this.hasType(tile, 'Expert');
      if (type === 'Farm' && workers > 0) income += 175 / 4;
      else if (type === 'Mine' && workers > 0) income += 20 * workers * (hasExpert ? 2 : 1);
      else if (type === 'Nuclear Power Plant' && workers > 0 && hasExpert) income += 160 * workers;
      else if (type === 'Hydroelectric Power Plant' && workers > 0 && hasExpert) income += 80 * workers;
      else if (tile instanceof AbundantForest && workers > 0) income += 15;
      if (type === 'Village') income -= 10;
      if (type === 'Outpost') income -= 50;
    }
    return income - this.salaryPerRound(p);
  }

  /** Metal produced per round by staffed mines (an expert doubles a mine's output). */
  private metalIncomePerRound(p: PlayerBase): number {
    let metal = 0;
    for (const tile of this.ownedTiles(p)) {
      if (tile.getBuilding()?.getType() !== 'Mine') continue;
      const workers = tile.getUnits().filter((u) => u.getType() === 'BasicWorker').length;
      metal += 20 * workers * (this.hasType(tile, 'Expert') ? 2 : 1);
    }
    return metal;
  }

  /** Stone produced per round by staffed mines (30/worker, doubled by an expert). The
   *  only stone source — Villages drain 10 stone/round each, so this must cover them. */
  private stoneIncomePerRound(p: PlayerBase): number {
    let stone = 0;
    for (const tile of this.ownedTiles(p)) {
      if (tile.getBuilding()?.getType() !== 'Mine') continue;
      const workers = tile.getUnits().filter((u) => u.getType() === 'BasicWorker').length;
      stone += 30 * workers * (this.hasType(tile, 'Expert') ? 2 : 1);
    }
    return stone;
  }

  /**
   * Safe to take on one more salaried unit (`salary`/round) without risking
   * bankruptcy: either the projected net stays non-negative, or we hold a large
   * enough cash cushion to ride out the deficit while farms mature.
   */
  private canAffordUpkeep(p: PlayerBase, salary: number, _cushion: number): boolean {
    // Strict: never take on a salaried unit that would push projected money income
    // negative. Income (farms) is built first via affordsIncomeBuild, so the bootstrap
    // still works; this is what stops a CPU slowly bleeding its treasury to zero by
    // over-hiring scouts/soldiers it can't sustain.
    return this.netMoneyPerRound(p) - salary >= 0;
  }

  /**
   * Affordable while keeping `reserve` plus a salary cushion in the bank and no
   * resource negative. Farm income is lumpy (a payout every 4 rounds) but wages
   * are due every round, so we keep ~5 rounds of wages buffered — this prevents
   * bankruptcy and self-limits how many salaried units we hire.
   */
  private affords(p: PlayerBase, cost: ResourceMap, reserve: number): boolean {
    if (!p.hasEnoughResources(cost)) return false;
    const buffer = reserve + this.moneyDrainPerRound(p) * 5;
    return this.money(p) + (cost.get(BasicResource.MONEY) ?? 0) >= buffer;
  }

  /**
   * A farm is net-positive (it pays for itself in a few rounds), so it must not be
   * blocked by the salary-buffer logic — doing so created a fatal catch-22: a CPU that
   * leapfrog-expanded with idle scouts had too little cash to ever build the very farm
   * that would fix its income, and slowly bled to death. Income builds only need the
   * raw resources plus a small money floor.
   */
  private affordsIncomeBuild(p: PlayerBase, cost: ResourceMap, floor = 40): boolean {
    if (!p.hasEnoughResources(cost)) return false;
    return this.money(p) + (cost.get(BasicResource.MONEY) ?? 0) >= floor;
  }

  /**
   * Can we build *another* farm without craters? The first few farms bootstrap the
   * economy and only need the small income floor (see affordsIncomeBuild) — blocking
   * them deadlocks the opening. But once a few are up, building many young farms in a
   * row is dangerous: each pays out only every 4 rounds, while its worker's wage and the
   * village/outpost upkeep are due *every* round, so a rapid farm spree can crater money
   * to negative before the new farms mature. Beyond the bootstrap we therefore require a
   * cash cushion covering a few rounds of total drain — throttling expansion farms to
   * when the treasury can absorb the lumpy-payout gap.
   */
  private affordsFarm(p: PlayerBase, farmCount: number): boolean {
    if (!p.hasEnoughResources(FARM_BUILD_COST)) return false;
    const moneyAfter = this.money(p) + (FARM_BUILD_COST.get(BasicResource.MONEY) ?? 0);
    // A farm pays out only every ~4 rounds, so keep enough cash to cover ~4 rounds
    // of drain (salary + upkeep) after the build — otherwise the bot spends its last
    // cash on farms/staffing and salary bankrupts it BEFORE the farms produce
    // (the grassland-poor self-bankruptcy bug). Early game drain is tiny, so the
    // bootstrap opening stays unblocked.
    const cushion = this.moneyDrainPerRound(p) * 4;
    if (farmCount < 3) return moneyAfter >= Math.max(40, cushion);
    return moneyAfter >= Math.max(80, cushion);
  }

  private ownedTiles(p: PlayerBase): TileBase[] {
    return p.getObjects().filter((o): o is TileBase => o instanceof TileBase);
  }
  private hasType(tile: TileBase, type: string): boolean {
    return tile.getUnits().some((u) => u.getType() === type);
  }
  private addWorker(player: PlayerBase, tile: TileBase): boolean {
    if (player.getFreeUnitAmount() <= 0) return false;
    if (!this.affords(player, BASIC_WORKER_COST, AiController.STAFF_RESERVE)) return false;
    return this.eh.aiBuyAndPlaceUnit('BasicWorker', tile);
  }
  private addExpert(player: PlayerBase, tile: TileBase, params: AiParams): boolean {
    if (!this.affords(player, EXPERT_COST, params.reserve)) return false;
    return this.eh.aiBuyAndPlaceUnit('Expert', tile);
  }

  // --- staffing -------------------------------------------------------------

  private *staffBuildings(player: PlayerBase, params: AiParams): Generator<void> {
    for (const tile of this.ownedTiles(player)) {
      const type = tile.getBuilding()?.getType();
      if (type === 'Farm') {
        if (!this.hasType(tile, 'BasicWorker')) yield* this.doAction(() => this.addWorker(player, tile));
      } else if (type === 'Mine') {
        // The metal engine: guarantee one worker — even by pulling one off a forest.
        // Extra workers (which scale its output) are stacked later, in
        // stackProducers, so expansion and villages get first claim on unit slots.
        yield* this.ensureWorker(player, tile);
      } else if (type === 'Nuclear Power Plant' || type === 'Hydroelectric Power Plant') {
        // A power plant produces ONLY with both an expert and ≥1 worker — either one alone
        // is a pure wage drain. staffPlant completes the pair or leaves it be.
        yield* this.staffPlant(player, params, tile);
      } else if (tile instanceof AbundantForest && !this.hasType(tile, 'BasicWorker')) {
        yield* this.doAction(() => this.addWorker(player, tile));
      }
    }
  }

  /**
   * Staff a power plant *profitably*. A Hydro/Nuclear plant produces money only when it
   * has BOTH an expert and at least one worker; an expert with no worker (or a worker
   * with no expert) yields nothing while still drawing a wage — a pure loss, and the
   * reason the CPU's many plants sat idle. So we only ever staff when we can complete the
   * pair this turn: the worker can come from a free slot or a relocatable idle worker,
   * but the expert needs a fresh slot. If we can't finish the pair, we leave the plant
   * alone rather than bleed wages on a half-staffed one — the freed cap (a later Village)
   * will let us complete it.
   */
  private *staffPlant(player: PlayerBase, params: AiParams, tile: TileBase): Generator<void> {
    if (!params.experts) return;
    const workers = () => tile.getUnits().filter((u) => u.getType() === 'BasicWorker').length;
    const hasExpert = () => this.hasType(tile, 'Expert');
    if (hasExpert() && workers() >= 1) return; // already producing (extra workers via stackProducers)

    const reloc = () => this.findIdleOnPlain(player) ?? this.findSurplusProducerWorker(player);
    const needWorker = workers() < 1;
    const needExpert = !hasExpert();
    const slotsNeeded = (needExpert ? 1 : 0) + (needWorker ? 1 : 0);
    const relocForWorker = needWorker && reloc() ? 1 : 0;
    // The expert always needs a brand-new slot; the worker can be relocated. So bail
    // unless free slots (+ a relocatable worker) can complete the pair.
    if (player.getFreeUnitAmount() + relocForWorker < slotsNeeded) return;

    if (needWorker) {
      if (player.getFreeUnitAmount() > 0) yield* this.doAction(() => this.addWorker(player, tile));
      else {
        const sp = reloc();
        if (sp && sp.tile !== tile) yield* this.doAction(() => this.eh.aiMoveUnit(sp.unit, sp.tile, tile));
      }
    }
    if (!hasExpert() && workers() >= 1 && player.getFreeUnitAmount() > 0) {
      yield* this.doAction(() => this.addExpert(player, tile, params));
    }
  }

  /** Whether a fresh power plant could be fully staffed (expert + worker) right now. */
  private canStaffNewPlant(player: PlayerBase): boolean {
    const free = player.getFreeUnitAmount();
    if (free >= 2) return true; // both as fresh hires
    return free >= 1 && !!(this.findIdleOnPlain(player) ?? this.findSurplusProducerWorker(player));
  }

  /** Guarantee one worker on a key building, relocating an idle/forest worker if capped. */
  private *ensureWorker(player: PlayerBase, tile: TileBase): Generator<void> {
    if (this.hasType(tile, 'BasicWorker')) return;
    if (player.getFreeUnitAmount() > 0) {
      yield* this.doAction(() => this.addWorker(player, tile));
      return;
    }
    // Capped: pull from a plain idler first, then from a forest (metal > wood).
    const spare = this.findIdleOnPlain(player) ?? this.findSpareWorker(player, tile);
    if (spare && spare.tile !== tile) yield* this.doAction(() => this.eh.aiMoveUnit(spare.unit, spare.tile, tile));
  }

  /** A worker on a low-value tile (forest / plain) that can be pulled to a key building. */
  private findSpareWorker(player: PlayerBase, exclude: TileBase): { unit: UnitBase; tile: TileBase } | null {
    for (const tile of this.ownedTiles(player)) {
      if (tile === exclude) continue;
      const type = tile.getBuilding()?.getType();
      if (type === 'Farm' || type === 'Mine' || type === 'Nuclear Power Plant' || type === 'Hydroelectric Power Plant') {
        continue; // already doing valuable work
      }
      const worker = tile.getUnits().find((u) => u.getType() === 'BasicWorker');
      if (worker) return { unit: worker, tile };
    }
    return null;
  }

  // --- building -------------------------------------------------------------

  private emptyGrassland(player: PlayerBase): TileBase[] {
    return this.ownedTiles(player).filter(
      (t) => t instanceof Grassland && t.getBuilding() === null && t.getBuildableBuildings().includes('Farm'),
    );
  }

  /**
   * Build a mine on an owned mountain once wood and money allow. A mine costs 250
   * wood; we require a small buffer beyond that (≥300) so building it doesn't strand
   * the farms (which also need wood). Runs even when capped — staffing later
   * relocates a worker onto it.
   */
  private *buildMines(player: PlayerBase, params: AiParams): Generator<void> {
    if (this.wood(player) < 300) return;
    // Each mine wants a worker, which competes with farms/expansion for unit slots.
    // Tie the number of mines to how much cap we've unlocked (one extra mine per
    // village) so the CPU doesn't bury all its workers underground and stop growing.
    const mines = this.ownedTiles(player).filter((t) => t.getBuilding()?.getType() === 'Mine').length;
    const villages = this.ownedTiles(player).filter((t) => t.getBuilding()?.getType() === 'Village').length;
    const maxMines = 1 + villages;
    if (mines >= maxMines) return;
    for (const m of this.ownedTiles(player).filter((t) => t instanceof Mountain && t.getBuilding() === null)) {
      if (this.affords(player, MINE_BUILD_COST, params.reserve) && this.hasWoodBuffer(player, MINE_BUILD_COST)) {
        if (yield* this.doAction(() => this.eh.aiBuildBuilding('Mine', m))) return; // one per turn
      }
    }
  }

  /**
   * Put one Expert on each staffed mine. An expert DOUBLES that mine's money + metal +
   * stone output, so it is the single most efficient industrial upgrade: the extra metal
   * is what pays for Outposts (300 metal each) and the soldier army, and the extra money
   * funds power plants. Gated on a free unit slot and the cash reserve, so it never
   * over-extends. As Villages raise the unit cap, idle slots flow here first.
   */
  private *boostMines(player: PlayerBase, params: AiParams): Generator<void> {
    if (!params.experts) return;
    for (const tile of this.ownedTiles(player)) {
      if (tile.getBuilding()?.getType() !== 'Mine') continue;
      if (!this.hasType(tile, 'BasicWorker') || this.hasType(tile, 'Expert')) continue;
      if (player.getFreeUnitAmount() <= 0) continue;
      yield* this.doAction(() => this.addExpert(player, tile, params));
    }
  }

  private *buildFarms(player: PlayerBase, params: AiParams): Generator<void> {
    const spots = this.emptyGrassland(player);
    // Cap how many farms we run so there's always room for a forest worker (wood) and
    // a scout (expansion): each farm permanently occupies a unit slot. This is what
    // lets the cap-3 opening still afford to harvest the wood it needs for its first
    // Village — without it the CPU spends every slot on farms and never raises its cap.
    let farmCount = this.ownedTiles(player).filter((t) => t.getBuilding()?.getType() === 'Farm').length;
    const mineCount = this.ownedTiles(player).filter((t) => t.getBuilding()?.getType() === 'Mine').length;
    // Leave room (beyond the mines) for a forest worker and a scout, so wood keeps
    // flowing for the next Village and territory keeps growing — otherwise the CPU
    // packs every slot with farms+mines, starves of wood, and stalls.
    const maxFarms = Math.max(1, player.getMaxUnitAmount() - 2 - mineCount);

    // Best value: a grassland that already holds an idle worker — building a farm
    // there staffs it instantly and costs no unit slot. This is how the CPU turns
    // its expansion workers into income even when its unit cap is full.
    for (const g of spots.filter((t) => this.hasType(t, 'BasicWorker'))) {
      if (farmCount >= maxFarms) break;
      if (this.affordsFarm(player, farmCount) && this.hasWoodBuffer(player, FARM_BUILD_COST)) {
        if (yield* this.doAction(() => this.eh.aiBuildBuilding('Farm', g))) farmCount += 1;
      }
    }
    // Then empty grasslands, if we have a free slot to staff the new farm (keep one
    // free for wood/expansion while wood is scarce).
    const slotFloor = this.wood(player) < 200 ? 1 : 0;
    for (const g of spots.filter((t) => !this.hasType(t, 'BasicWorker'))) {
      if (farmCount >= maxFarms) break;
      if (player.getFreeUnitAmount() <= slotFloor) break;
      if (this.affordsFarm(player, farmCount) && this.hasWoodBuffer(player, FARM_BUILD_COST)) {
        if (yield* this.doAction(() => this.eh.aiBuildBuilding('Farm', g))) farmCount += 1;
      }
    }
  }

  /**
   * Diversify income with power plants once the core economy is solvent — this is
   * how the CPU "invests in its own economy". A Hydroelectric plant (+80 money per
   * worker, needs an expert) sits on a straight river; a Nuclear plant (+160 money
   * per worker, needs an expert) is the strongest late-game money maker but costs
   * 2000 money — a long savings goal that then out-produces everything per unit-slot.
   * Both are gated on a healthy cash cushion and a positive projected net, so they are
   * genuine investments rather than gambles.
   */
  private *buildPowerPlants(player: PlayerBase, params: AiParams): Generator<void> {
    if (!params.experts) return; // a plant with no expert produces nothing
    if (this.netMoneyPerRound(player) <= 0) return;
    const hydros = this.ownedTiles(player).filter((t) => t.getBuilding()?.getType() === 'Hydroelectric Power Plant');
    // Don't pile up idle plants: if any existing hydro still lacks its expert+worker,
    // build NO more — staffBuildings will complete them first. Building a second plant
    // while the first sits unstaffed was exactly what produced lots of money-losing hydro.
    if (hydros.some((t) => !this.hasType(t, 'Expert') || !this.hasType(t, 'BasicWorker'))) return;
    // Only start a new plant if we can actually man it (expert + worker) this turn or next.
    if (!this.canStaffNewPlant(player)) return;
    const rivers = this.ownedTiles(player).filter(
      (t) =>
        t instanceof River &&
        t.getBuilding() === null &&
        t.getBuildableBuildings().includes('Hydroelectric Power Plant'),
    );
    for (const r of rivers) {
      // A hydro plant is a strong, cheap-to-run investment, so it earns a smaller
      // reserve than discretionary spending — much like a Village.
      if (this.affords(player, HEPP_BUILD_COST, Math.min(params.reserve, 80)) && this.hasWoodBuffer(player, HEPP_BUILD_COST)) {
        if (yield* this.doAction(() => this.eh.aiBuildBuilding('Hydroelectric Power Plant', r))) break; // one/turn
      }
    }
    // (Nuclear plants are handled by investNuclear — they need slot-room made for them.)
  }

  /**
   * Invest a saved-up treasury into Nuclear plants — the late-game economic engine. A
   * nuclear plant is the best money-per-unit-slot building in the game (160/worker; an
   * expert + 2 workers nets ~285/round, ~2.4x a farm per slot), but it costs 2000 money
   * and needs 3 unit slots staffed (1 expert + 2 workers) to run. A maxed-out economy has
   * neither spare slots nor a reason to keep hoarding cash, so without this the CPU just
   * sits on a growing pile of money. Here, once genuinely rich, it builds plants (scaling
   * the count to the spare cash) and *makes room* for them by raising the unit cap with a
   * Village, then staffs each — converting an idle hoard into compounding income. Every
   * step still goes through the solvency-guarded helpers, so it can't bankrupt itself.
   */
  private *investNuclear(player: PlayerBase, params: AiParams): Generator<void> {
    if (!params.nuclear || !params.experts) return;
    if (this.money(player) <= 2400) return; // only once we've saved up for it
    const nukes = () => this.ownedTiles(player).filter((t) => t.getBuilding()?.getType() === 'Nuclear Power Plant');

    // 1. STAFF what we already have first — never build a second plant while the first
    //    sits empty (that just sinks 2000 money into a dead building).
    for (const plant of nukes()) yield* this.staffNuclear(player, params, plant);

    // 2. BUILD one more (scaled to spare cash) ONLY if we can actually staff it: it needs
    //    an expert (a fresh slot, or one freed by a Village) — workers can always be
    //    relocated off farms. Without this gate the CPU wasted 2000 on plants it could
    //    never man once its unit cap was full.
    const fullyStaffed = (t: TileBase) => this.hasType(t, 'Expert') && t.getUnits().some((u) => u.getType() === 'BasicWorker');
    const wantCount = 1 + Math.floor((this.money(player) - 2400) / 3000);
    if (nukes().length >= wantCount || !nukes().every(fullyStaffed)) return;
    // The expert needs a slot: either one free now, or one a Village will free — and in the
    // latter case we need a SECOND empty grassland (the nuclear takes one, the Village the
    // other), so the plant won't end up built but unmannable.
    const emptyGrass = this.emptyGrassland(player).filter((t) => !this.hasType(t, 'BasicWorker'));
    if (player.getFreeUnitAmount() < 1 && !(this.canRaiseCap(player) && emptyGrass.length >= 2)) return;
    const spot = emptyGrass[0];
    if (spot && this.affords(player, NUCLEARPP_BUILD_COST, params.reserve) && this.hasWoodBuffer(player, NUCLEARPP_BUILD_COST)) {
      if (yield* this.doAction(() => this.eh.aiBuildBuilding('Nuclear Power Plant', spot))) {
        yield* this.staffNuclear(player, params, spot);
      }
    }
  }

  /**
   * Staff one nuclear plant toward expert + 2 workers. The expert is a fresh hire (raise
   * the cap with a Village once if we're full). The workers prefer a free slot but will
   * RELOCATE off a farm when capped — moving a worker from a farm (~44/round) to a nuclear
   * (160/round) is hugely net-positive, so a maxed-out economy can still light up its
   * plants instead of hoarding idle cash.
   */
  private *staffNuclear(player: PlayerBase, params: AiParams, plant: TileBase): Generator<void> {
    if (!this.hasType(plant, 'Expert')) {
      if (player.getFreeUnitAmount() < 1) yield* this.raiseUnitCap(player, params);
      if (player.getFreeUnitAmount() > 0) yield* this.doAction(() => this.addExpert(player, plant, params));
    }
    while (
      this.hasType(plant, 'Expert') &&
      plant.getUnits().filter((u) => u.getType() === 'BasicWorker').length < 2 &&
      plant.hasSpaceForUnits() &&
      this.budget > 0
    ) {
      if (player.getFreeUnitAmount() > 0) {
        if (!(yield* this.doAction(() => this.addWorker(player, plant)))) break;
      } else {
        const fw = this.findExpendableWorker(player); // idle → surplus producer → a spare farm worker
        if (!fw || fw.tile === plant) break;
        if (!(yield* this.doAction(() => this.eh.aiMoveUnit(fw.unit, fw.tile, plant)))) break;
      }
    }
  }

  /** Cheap check: can we still raise the unit cap with another Village (a spot, a forest
   *  to sustain its wood upkeep, and below the village ceiling)? Mirrors raiseUnitCap's
   *  structural gates so investNuclear knows whether an expert slot can be freed. */
  private canRaiseCap(player: PlayerBase): boolean {
    const villages = this.ownedTiles(player).filter((t) => t.getBuilding()?.getType() === 'Village').length;
    if (villages >= 5) return false;
    if (!this.emptyGrassland(player).some((t) => !this.hasType(t, 'BasicWorker'))) return false;
    return this.ownedTiles(player).some((t) => t instanceof Forest && (t.getBuilding() === null || this.hasType(t, 'BasicWorker')));
  }

  /** True if any opponent is still in the game (someone left to fight). */
  private enemyExists(player: PlayerBase): boolean {
    return this.pm.getPlayers().some((p) => p !== player);
  }

  /** Whether to gear up for war. In warmonger mode we don't wait for an enemy on the
   *  doorstep — having any opponent alive is reason enough to raise the soldier cap and
   *  field a strike force, so a won economy isn't left cap-locked at 1 soldier, unable
   *  to convert its lead into a conquest. */
  private shouldMilitarise(player: PlayerBase, params: AiParams): boolean {
    // A standing enemy Device is an existential threat (its countdown wins the game), so
    // gear up for war regardless of the normal "enemy on the doorstep" trigger.
    if (this.enemyHasDevice(player)) return true;
    return params.warmonger ? this.enemyExists(player) : this.hasReachableEnemy(player);
  }

  /** True if an enemy-held tile is on our border (reachable to attack, or a standing threat). */
  private hasReachableEnemy(player: PlayerBase): boolean {
    if (this.enemyThreat(player) > 0) return true;
    return this.om.getAvailableTiles().some((t) => {
      const o = t.getOwner();
      return o !== null && o !== player;
    });
  }

  /** The strongest defended reachable enemy tile — the most soldiers on any enemy tile we
   *  could attack (Outposts excluded, since they can't be taken). 0 when the enemy front
   *  is undefended. This is what decides how big a strike force we actually need: there is
   *  no point fielding (or building outposts for) an army to crush a defenceless neighbour
   *  a single soldier can roll up — over-investing there is what bankrupts a tight economy. */
  private reachableEnemyMaxDefenders(player: PlayerBase): number {
    let max = 0;
    for (const t of this.om.getAvailableTiles()) {
      const o = t.getOwner();
      if (o === null || o === player) continue;
      if (t.getBuilding()?.getType() === 'Outpost') continue;
      const def = t.getUnits().filter((u) => u.getType() === 'Soldier').length;
      if (def > max) max = def;
    }
    return max;
  }

  /** A genuine military reason to spend on outposts/army: we are under threat (defence) or
   *  the reachable enemy front is actually defended (we need the soldier cap to crack it).
   *  Merely bordering a defenceless enemy is NOT a reason to militarise — a lone soldier
   *  conquers undefended tiles, so building a 650-money outpost there only bleeds us. */
  private militaryNeed(player: PlayerBase): boolean {
    return this.enemyThreat(player) > 0 || this.reachableEnemyMaxDefenders(player) > 0 || this.enemyHasDevice(player);
  }

  /**
   * Build an Outpost when there is actually a war to fight: the HQ alone caps soldiers
   * at 1, so an Outpost (+3 soldier cap) is the gateway to fielding any real army, and
   * it doubles as a forward defensive strongpoint. It costs 50 money/round upkeep, so
   * the CPU only commits when an enemy is on its border and the economy can clearly
   * carry it. Outposts may not sit next to the HQ/another outpost (a game rule, which
   * `getBuildableBuildings` already encodes).
   */
  private *buildOutposts(player: PlayerBase, params: AiParams): Generator<void> {
    if (params.maxOutposts <= 0 || !params.attack) return;
    if (!this.shouldMilitarise(player, params)) return; // no reason to militarise yet
    // Only sink money into outposts when there's a real military need — a standing threat,
    // or a defended enemy front we need the extra soldier cap to break. Against a
    // defenceless neighbour a single soldier suffices, so an outpost there is pure waste.
    if (!this.militaryNeed(player)) return;
    const outposts = this.ownedTiles(player).filter((t) => t.getBuilding()?.getType() === 'Outpost').length;
    if (outposts >= params.maxOutposts) return;
    // An Outpost is the gateway to an army (+3 soldier cap) AND an impregnable defensive
    // strongpoint (a tile with an Outpost can never be conquered, however it is out-
    // numbered). Build it as soon as the economy is solid — but only while metal income
    // covers its 15/round upkeep AND the money net stays clearly positive *after* the new
    // outpost's 50/round upkeep, so several of them can never bleed us into the red.
    if (this.om.getTileCountForPlayer(player) < 8) return;
    if (this.netMoneyPerRound(player) - 50 < 10) return;
    if (this.metalIncomePerRound(player) - (outposts + 1) * 15 < 0) return;
    // Prefer a frontline grassland — one bordering enemy land or under threat — so the
    // outpost doubles as the strongpoint that holds the line; fall back to any buildable.
    const buildable = this.ownedTiles(player).filter(
      (t) => t instanceof Grassland && t.getBuilding() === null && t.getBuildableBuildings().includes('Outpost'),
    );
    const frontline = buildable.find(
      (t) => this.tileThreatened(t, player) || t.getNeighbourTiles().some((n) => n.getOwner() !== null && n.getOwner() !== player),
    );
    const spot = frontline ?? buildable[0];
    if (spot && this.affords(player, OUTPOST_BUILD_COST, Math.min(params.reserve, 100))) {
      yield* this.doAction(() => this.eh.aiBuildBuilding('Outpost', spot));
    }
  }

  /** True when an opponent owns a standing Strange Device — we must crack it before its
   *  countdown wins them the game. Always checked (NOT gated on params.device), so even a
   *  non-building AI mounts the counterplay. */
  private enemyHasDevice(player: PlayerBase): boolean {
    const dt = this.om.findStrangeDeviceTile();
    return dt !== null && dt.getOwner() !== null && dt.getOwner() !== player;
  }

  /**
   * The Strange Device endgame. When we are the clear leader, building it forces a
   * decisive finish: a countdown that wins the game if the Device survives. The catch
   * (it halves our soldier cap) makes it a gamble, so we only commit when:
   *   - the device strategy is enabled for this AI (params.device),
   *   - no Device exists yet (one per game; if an enemy owns one we ATTACK instead),
   *   - the game has matured (so it's a closer, not opening cheese),
   *   - we out-tile every opponent (the leader can afford the build + survive the halving),
   *   - we already hold ≥1 Outpost (so the halved cap still leaves real defenders),
   *   - the economy can clearly carry the one-time cost.
   * It is placed on the safest interior grassland (fewest enemy-bordering neighbours).
   */
  private *buildStrangeDevice(player: PlayerBase, params: AiParams): Generator<void> {
    if (!params.device) return;
    if (this.om.hasStrangeDevice()) return; // one per game — counterplay handles an enemy's
    if (this.pm.getRoundsPlayed() < 18) return; // let the game develop first
    // Pursue the Device when we are NOT losing on territory (the leader/co-leader can carry
    // the build and survive the halving). In a stalemate this is exactly who should force
    // the issue — relaxed from "strictly leading", which almost never fired in a ~49/49
    // turtle and left the Device unbuilt (the build-rate bottleneck).
    const myTiles = this.om.getTileCountForPlayer(player);
    const notLosing = this.pm.getPlayers().every((p) => p === player || this.om.getTileCountForPlayer(p) <= myTiles);
    if (!notLosing) return;
    // Affordability for a TERMINAL play. The general affords() helper demands a fat
    // 5-rounds-of-upkeep cash cushion on top of the cost; a settled late-game economy
    // carries heavy upkeep, so that buffer was almost never met — which is exactly why
    // the leader sat on a winning position and never built the Device (measured: 88% of
    // stalemates the leader was otherwise fully able). For a game-ending play we only need
    // to stay solvent through the countdown: raw resources + non-negative money net + a
    // small cash floor left after the one-time cost.
    if (!player.hasEnoughResources(STRANGE_DEVICE_BUILD_COST)) return;
    if (this.netMoneyPerRound(player) < 0) return;
    if (this.money(player) + (STRANGE_DEVICE_BUILD_COST.get(BasicResource.MONEY) ?? 0) < 150) return;
    const interior = (a: TileBase, b: TileBase) => this.enemyBorderCount(a, player) - this.enemyBorderCount(b, player);
    const buildableGrass = (what: string) =>
      this.ownedTiles(player)
        .filter(
          (t) =>
            t instanceof Grassland &&
            t.getBuilding() === null &&
            t.getUnitCount() === 0 && // the Device can't be built on an occupied tile (it never holds units)
            t.getBuildableBuildings().includes(what),
        )
        .sort(interior);
    const outposts = this.ownedTiles(player).filter((t) => t.getBuilding()?.getType() === 'Outpost').length;
    if (outposts < 1) {
      // Precursor: the Device halves our soldier cap, so an Outpost (+3 cap) first keeps the
      // halved cap above zero — i.e. real defenders. The normal buildOutposts step only fires
      // under military need, which a pure turtle never has, so adopt the Device plan here and
      // lay the gating Outpost now (interior, defendable); the Device follows next turn.
      const ospot = buildableGrass('Outpost')[0];
      // Solvency for the precursor: stay money-positive after its 50/round upkeep + a small
      // cash floor (same lighter standard as the Device itself — the fat reserve blocked it).
      const canAffordOutpost =
        ospot &&
        player.hasEnoughResources(OUTPOST_BUILD_COST) &&
        this.netMoneyPerRound(player) - 50 >= 0 &&
        this.money(player) + (OUTPOST_BUILD_COST.get(BasicResource.MONEY) ?? 0) >= 100;
      if (canAffordOutpost) {
        yield* this.doAction(() => this.eh.aiBuildBuilding('Outpost', ospot));
      }
      return;
    }
    const spot = buildableGrass('Strange Device')[0];
    if (!spot) return;
    yield* this.doAction(() => this.eh.aiBuildBuilding('Strange Device', spot));
  }

  /** Build a Village when we are capped out — more unit slots is what compounds. */
  private *raiseUnitCap(player: PlayerBase, params: AiParams): Generator<void> {
    if (player.getFreeUnitAmount() > 1) return; // only when nearly capped
    const spot = this.emptyGrassland(player)[0];
    if (!spot) return;
    // Each Village drains 10 wood/stone/money per round forever. Cap how many we run by
    // how much wood income we sustain (≈3 villages per forest harvester) plus a hard
    // ceiling of 5 — raising it bloats the economy so much that games (and the headless
    // sims) drag on, for little gain, since conquered enemy villages add cap for free
    // anyway. Each village below is still independently gated on the money net, the wood
    // buffer AND the stone guard.
    const villages = this.ownedTiles(player).filter((t) => t.getBuilding()?.getType() === 'Village').length;
    const harvesters = this.ownedTiles(player).filter(
      (t) => t instanceof Forest && this.hasType(t, 'BasicWorker'),
    ).length;
    if (villages >= Math.min(5, 1 + harvesters * 3)) return;
    // Must own a forest to sustain the wood upkeep (or already harvest one).
    if (!this.ownedTiles(player).some((t) => t instanceof Forest && (t.getBuilding() === null || this.hasType(t, 'BasicWorker')))) return;
    // A Village costs 10 money/round upkeep AND triggers a spending cascade — its +3
    // cap slots get filled with workers (≈150 cash + 15/round wages). Require enough
    // money-income headroom to absorb all of that, or the young economy bleeds out
    // a few rounds after a too-eager village (the r34 deaths).
    if (this.netMoneyPerRound(player) - 25 < 10) return;
    // Wood buffer must survive the forest regrow gap *after* this village's extra upkeep.
    const postUpkeep = this.woodUpkeep(player) + 10;
    if (this.wood(player) - 200 < Math.max(100, postUpkeep * 5)) return;
    // Stone guard: a Village also drains 10 stone/round forever, and mines are the only
    // stone source. Only commit if mine output covers the new stone upkeep or we sit on a
    // deep stone buffer — otherwise a village-heavy, mine-light economy bleeds stone to
    // zero and dies to the negative-resource rule (a silent, self-inflicted death).
    const stoneUpkeep = (villages + 1) * 10;
    if (this.stoneIncomePerRound(player) < stoneUpkeep && this.stone(player) - 100 < stoneUpkeep * 8) return;
    if (this.affords(player, VILLAGE_BUILD_COST, params.reserve)) {
      yield* this.doAction(() => this.eh.aiBuildBuilding('Village', spot));
    }
  }

  // --- wood -----------------------------------------------------------------

  /** Wood drained each round by upkeep buildings (Villages −10, Bridges −5). */
  private woodUpkeep(p: PlayerBase): number {
    let w = 0;
    for (const t of this.ownedTiles(p)) {
      const type = t.getBuilding()?.getType();
      if (type === 'Village') w += 10;
      if (type === 'Bridge') w += 5;
    }
    return w;
  }

  /**
   * Safe to spend the wood in `cost` without risking a wood death. A forest harvester
   * produces nothing during its ~5-round regrow gap, so when we carry wood upkeep we
   * must keep a buffer big enough to ride out that gap (≈6× upkeep) on top of the cost.
   */
  private hasWoodBuffer(p: PlayerBase, cost: ResourceMap): boolean {
    const need = -(cost.get(BasicResource.WOOD) ?? 0); // positive wood amount the build costs
    if (need <= 0) return true;
    const upkeep = this.woodUpkeep(p);
    // Enough to ride out a forest's ~5-round regrow gap (upkeep × 5) with a floor.
    const buffer = upkeep > 0 ? Math.max(100, upkeep * 5) : 0;
    return this.wood(p) - need >= buffer;
  }

  /** A worker that can be pulled to cover wood income: an idle one, a surplus producer
   *  worker, or — as a last resort to avoid a wood death — one off a farm (keeping ≥1). */
  private findExpendableWorker(player: PlayerBase): { unit: UnitBase; tile: TileBase } | null {
    const idle = this.findIdleOnPlain(player);
    if (idle) return idle;
    const surplus = this.findSurplusProducerWorker(player);
    if (surplus) return surplus;
    const farms = this.ownedTiles(player).filter(
      (t) => t.getBuilding()?.getType() === 'Farm' && this.hasType(t, 'BasicWorker'),
    );
    if (farms.length >= 2) {
      const tile = farms[farms.length - 1];
      const unit = tile.getUnits().find((u) => u.getType() === 'BasicWorker');
      if (unit) return { unit, tile };
    }
    return null;
  }

  /**
   * Villages and bridges drain wood every single round. Without a forest worker the
   * wood stock bleeds to zero and the player is wiped out by the negative-resource
   * rule — a silent, self-inflicted death. This GUARANTEES enough staffed forest
   * workers to cover the upkeep (one harvester ≈ +54 wood/round, so ~1 per 5 villages),
   * pulling a worker off a spare slot / stacked producer / surplus farm if we're capped.
   */
  private *ensureWoodIncome(player: PlayerBase): Generator<void> {
    const upkeep = this.woodUpkeep(player);
    if (upkeep <= 0) return;
    const harvesters = () =>
      this.ownedTiles(player).filter((t) => t instanceof Forest && t.getBuilding() === null && this.hasType(t, 'BasicWorker')).length;
    // One harvester net-sustains ~5 villages; add one more when the wood stock is
    // critically low so a single forest's regrow gap can't bleed us into the red.
    let need = Math.max(1, Math.ceil(upkeep / 40));
    if (this.wood(player) < upkeep * 4) need += 1;
    while (harvesters() < need && this.budget > 0) {
      const f = this.ownedTiles(player).find(
        (t) => t instanceof Forest && t.getBuilding() === null && t.hasSpaceForUnits() && !this.hasType(t, 'BasicWorker'),
      );
      if (!f) break; // no forest to staff — nothing we can do here
      let did = false;
      if (player.getFreeUnitAmount() > 0 && this.affords(player, BASIC_WORKER_COST, AiController.STAFF_RESERVE)) {
        did = yield* this.doAction(() => this.addWorker(player, f));
      } else {
        const spare = this.findExpendableWorker(player);
        if (spare && spare.tile !== f) did = yield* this.doAction(() => this.eh.aiMoveUnit(spare.unit, spare.tile, f));
      }
      if (!did) break;
    }
  }

  /**
   * Keep workers harvesting forests — wood is the gate on farms (100), mines (250)
   * and villages (200). A single forest worker produces 100 wood/round, so one
   * dedicated harvester is enough to keep wood climbing toward those thresholds.
   * We only ever *add* a harvester from a free slot or an idle expansion worker —
   * never pull a worker off a farm/mine, which would cut income.
   */
  /**
   * Wood the CPU will need *soon*, by what it is about to build: a mine on each owned
   * unbuilt mountain costs 250 wood, and the next few farms/villages on empty grassland
   * ~100 each. Looking ahead like this lets the harvesters stock up BEFORE the build is
   * wanted, instead of the CPU stalling at a mountain with no wood for the mine.
   */
  private anticipatedWoodNeed(player: PlayerBase): number {
    const mountainsNoMine = this.ownedTiles(player).filter((t) => t instanceof Mountain && t.getBuilding() === null).length;
    const emptyGrass = this.emptyGrassland(player).length;
    return mountainsNoMine * 250 + Math.min(emptyGrass, 4) * 100;
  }

  private *secureWood(player: PlayerBase, _params: AiParams): Generator<void> {
    // Gather toward what we're about to spend (anticipated need) plus a small buffer,
    // capped so we never hoard endlessly.
    const stockTarget = Math.min(700, Math.max(150, this.anticipatedWoodNeed(player)));
    if (this.wood(player) >= stockTarget + 100) return; // enough banked for the planned builds

    const forests = () => this.ownedTiles(player).filter((t) => t instanceof Forest && t.getBuilding() === null);
    const staffed = () => forests().filter((t) => this.hasType(t, 'BasicWorker')).length;
    // Two harvesters when we're below target with a real upcoming wood bill (e.g. a mine)
    // and the cap can spare the slot; otherwise one is enough to keep wood climbing.
    const target = this.wood(player) < stockTarget && this.anticipatedWoodNeed(player) > 200 && player.getMaxUnitAmount() > 6 ? 2 : 1;

    while (staffed() < target && this.budget > 0) {
      const f = forests().find((t) => t.hasSpaceForUnits() && !this.hasType(t, 'BasicWorker'));
      if (!f) break;
      let did = false;
      // Hiring a fresh harvester adds a wage; only do it if money income can carry it.
      // Relocating an idle worker (no new wage) is always fine.
      if (player.getFreeUnitAmount() > 0 && this.canAffordUpkeep(player, 5, 0)) {
        did = yield* this.doAction(() => this.addWorker(player, f));
      } else {
        const idle = this.findIdleOnPlain(player); // repurpose an idle expansion worker
        if (idle) did = yield* this.doAction(() => this.eh.aiMoveUnit(idle.unit, idle.tile, f));
      }
      if (!did) break;
    }
  }

  /** A worker idling on a plain owned tile (no building, not a forest) — produces nothing. */
  private findIdleOnPlain(player: PlayerBase): { unit: UnitBase; tile: TileBase } | null {
    for (const tile of this.ownedTiles(player)) {
      if (tile.getBuilding() || tile instanceof Forest || tile instanceof AbundantForest) continue;
      const worker = tile.getUnits().find((u) => u.getType() === 'BasicWorker');
      if (worker) return { unit: worker, tile };
    }
    return null;
  }

  // --- expansion ------------------------------------------------------------

  /** Claim value of a neutral tile: cap-raising Mikontalo and metal mountains first. */
  private claimValue(tile: TileBase): number {
    const b = tile.getBuilding();
    if (b && b.getType() === 'Mikontalo') return 6; // +2 unit cap, free
    if (tile.getType() === 'Mountain') return 5; // unlocks a mine -> metal
    if (tile.getType() === 'Grassland') return 4; // farm / village land
    if (tile.getType() === 'Forest') return 3; // wood
    if (tile.getType() === 'Abundant Forest') return 2;
    // A straight river can host a Hydroelectric plant (a strong money maker), so it is
    // worth claiming — a curved river can't build one and is the lowest-value land.
    if (tile instanceof River && tile.getBuildableBuildings().includes('Hydroelectric Power Plant')) return 4;
    return 1; // curved river
  }

  private *expand(player: PlayerBase, params: AiParams): Generator<void> {
    if (params.expand <= 0) return;
    let claimed = 0;
    while (claimed < params.expand && this.budget > 0) {
      const neutral = this.om
        .getAvailableTiles()
        // Claim only safe neutral land — never park a defenceless worker on a tile an
        // enemy soldier sits next to, where it would just be overrun next turn. AND skip
        // any tile we are ALREADY claiming this turn (one of our units sits on it): a
        // single worker takes an undefended tile, so sending a second is pure waste — the
        // "several workers on one tile" the CPU used to do.
        .filter(
          (t) =>
            t.getOwner() === null &&
            t.hasSpaceForUnits() &&
            !this.tileThreatened(t, player) &&
            !t.getConqueringUnits().some((u) => u.getOwner() === player),
        )
        .sort((a, b) => this.claimValue(b) - this.claimValue(a));
      if (neutral.length === 0) return;
      const tile = neutral[0];
      let did = false;
      // 1. Leap-frog a genuinely idle worker — free, no extra wages.
      const idle = this.findIdleWorker(player);
      if (idle && idle.tile !== tile) {
        did = yield* this.doAction(() => this.eh.aiMoveUnit(idle.unit, idle.tile, tile));
      }
      // 2. Hire a fresh scout into a free slot. Adds a -5/round wage with no immediate
      // income, so only while we can sustain the upkeep (or sit on a cushion).
      if (
        !did &&
        player.getFreeUnitAmount() > 0 &&
        this.affords(player, BASIC_WORKER_COST, params.reserve) &&
        this.canAffordUpkeep(player, 5, params.reserve + 300)
      ) {
        did = yield* this.doAction(() => this.eh.aiBuyAndPlaceUnit('BasicWorker', tile));
      }
      // 3. Capped with no idle worker: peel a *surplus* worker off an over-staffed
      // producer (e.g. a mine's 2nd worker) to scout. Without this the CPU stalls at
      // its starting territory once every slot is committed to the economy — it can
      // never claim the new grassland it needs to build more cap-raising Villages.
      if (!did) {
        const spare = this.findSurplusProducerWorker(player);
        if (spare && spare.tile !== tile) {
          did = yield* this.doAction(() => this.eh.aiMoveUnit(spare.unit, spare.tile, tile));
        }
      }
      if (!did) return; // can't make progress
      claimed += 1;
    }
  }

  /**
   * A worker that can be spared from an over-staffed producer for scouting: a 2nd+
   * worker on a mine or power plant (their first worker stays to keep it running), or
   * a forest harvester once wood is well stocked. Never touches farms (each needs its
   * one worker) or a producer's only worker.
   */
  private findSurplusProducerWorker(player: PlayerBase): { unit: UnitBase; tile: TileBase } | null {
    for (const tile of this.ownedTiles(player)) {
      const type = tile.getBuilding()?.getType();
      const stackable = type === 'Mine' || type === 'Nuclear Power Plant' || type === 'Hydroelectric Power Plant';
      if (!stackable) continue;
      const workers = tile.getUnits().filter((u) => u.getType() === 'BasicWorker');
      if (workers.length > 1) return { unit: workers[workers.length - 1], tile };
    }
    // Otherwise a forest worker, but only when wood is comfortably stocked.
    if (this.wood(player) >= 350) {
      for (const tile of this.ownedTiles(player)) {
        if (!(tile instanceof Forest)) continue;
        const worker = tile.getUnits().find((u) => u.getType() === 'BasicWorker');
        if (worker) return { unit: worker, tile };
      }
    }
    return null;
  }

  /**
   * A worker free to relocate (for expansion): an idle worker on a no-output tile,
   * or — only once wood is comfortably stocked and we aren't still trying to fund a
   * mine — a forest harvester. We must NOT pull harvesters while saving for a mine:
   * doing so starves the wood stockpile and the CPU never breaks its unit-cap
   * ceiling (the classic "48 tiles but no mine" churn).
   */
  private findIdleWorker(player: PlayerBase): { unit: UnitBase; tile: TileBase } | null {
    const needsWood =
      this.wood(player) < 350 ||
      this.ownedTiles(player).some((t) => t instanceof Mountain && t.getBuilding() === null);
    // First pass: genuinely idle workers (no building, not gathering) — always free.
    for (const tile of this.ownedTiles(player)) {
      if (tile.getBuilding() || tile instanceof Forest || tile instanceof AbundantForest) continue;
      const worker = tile.getUnits().find((u) => u.getType() === 'BasicWorker');
      if (worker) return { unit: worker, tile };
    }
    // Second pass: forest harvesters, but only when wood is no longer needed.
    if (needsWood) return null;
    for (const tile of this.ownedTiles(player)) {
      if (!(tile instanceof Forest || tile instanceof AbundantForest)) continue;
      const worker = tile.getUnits().find((u) => u.getType() === 'BasicWorker');
      if (worker) return { unit: worker, tile };
    }
    return null;
  }

  // --- spare workers --------------------------------------------------------

  /**
   * With slots still free after expansion/military, stack extra workers onto mines
   * and power plants — their output scales per worker, so a spare slot is worth more
   * there than idling. Runs last so growth (villages/expansion) always claims slots
   * first.
   */
  private *stackProducers(player: PlayerBase, params: AiParams): Generator<void> {
    const producers = () =>
      this.ownedTiles(player).filter((t) => {
        const type = t.getBuilding()?.getType();
        return (
          (type === 'Mine' || type === 'Nuclear Power Plant' || type === 'Hydroelectric Power Plant') &&
          t.hasSpaceForUnits()
        );
      });
    while (player.getFreeUnitAmount() > 0 && this.budget > 0) {
      const tile = producers()[0];
      if (!tile) break;
      const wantExpert = params.experts && tile.getBuilding()?.getType() !== 'Hydroelectric Power Plant';
      if (wantExpert && !this.hasType(tile, 'Expert') && player.getFreeUnitAmount() > 1) {
        if (yield* this.doAction(() => this.addExpert(player, tile, params))) continue;
      }
      if (!(yield* this.doAction(() => this.addWorker(player, tile)))) break;
    }
  }

  /** Never leave unit slots empty: put leftover workers on forests for wood income. */
  private *fillSpareSlots(player: PlayerBase, _params: AiParams): Generator<void> {
    const forests = () =>
      this.ownedTiles(player).filter((t) => t instanceof Forest && t.getBuilding() === null && t.hasSpaceForUnits());
    // Only soak up spare slots while money income can carry the extra wage — otherwise
    // a capped-out economy slowly bleeds its treasury filling slots with idle harvesters.
    while (player.getFreeUnitAmount() > 0 && this.budget > 0 && this.canAffordUpkeep(player, 5, 0)) {
      const f = forests()[0];
      if (!f) break;
      if (!(yield* this.doAction(() => this.addWorker(player, f)))) break;
    }
  }

  // --- military -------------------------------------------------------------

  /** Rough measure of enemy soldiers menacing us: ones invading our tiles or massed next door. */
  private enemyThreat(player: PlayerBase): number {
    let threat = 0;
    for (const tile of this.ownedTiles(player)) {
      threat += tile.getConqueringUnits().filter((u) => u.getOwner() !== player && u.getType() === 'Soldier').length;
      for (const n of tile.getNeighbourTiles()) {
        const o = n.getOwner();
        if (o !== null && o !== player) threat += n.getUnits().filter((u) => u.getType() === 'Soldier').length;
      }
    }
    return threat;
  }

  /** True if placing a worker on `tile` would expose it to an adjacent enemy soldier. */
  private tileThreatened(tile: TileBase, player: PlayerBase): boolean {
    for (const n of tile.getNeighbourTiles()) {
      const o = n.getOwner();
      if (o !== null && o !== player && n.getUnits().some((u) => u.getType() === 'Soldier')) return true;
    }
    return false;
  }

  /** Enemy soldiers standing on tiles adjacent to `tile` — the force that could assault it next turn. */
  private adjacentEnemySoldiers(tile: TileBase, player: PlayerBase): number {
    let n = 0;
    for (const nb of tile.getNeighbourTiles()) {
      const o = nb.getOwner();
      if (o !== null && o !== player) n += nb.getUnits().filter((u) => u.getType() === 'Soldier').length;
    }
    return n;
  }

  /** Enemy soldiers already standing on `tile` trying to conquer it this turn. */
  private invadersOn(tile: TileBase, player: PlayerBase): number {
    return tile.getConqueringUnits().filter((u) => u.getOwner() !== player && u.getType() === 'Soldier').length;
  }

  /** Our own soldiers garrisoning `tile`. */
  private soldiersOn(tile: TileBase, player: PlayerBase): number {
    return tile.getUnits().filter((u) => u.getOwner() === player && u.getType() === 'Soldier').length;
  }

  /** How many of a tile's neighbours are enemy-owned — its "frontier exposure". A tile
   *  with a positive count is on our border with the enemy. */
  private enemyBorderCount(tile: TileBase, player: PlayerBase): number {
    let n = 0;
    for (const nb of tile.getNeighbourTiles()) {
      const o = nb.getOwner();
      if (o !== null && o !== player) n += 1;
    }
    return n;
  }

  /** A soldier sitting deep in the rear — no enemy pressure AND not manning a border tile —
   *  so it is genuinely idle and free to redeploy without a buy. We deliberately do NOT
   *  pull soldiers off threatened tiles (needed where they are) or off the frontier (they
   *  guard the border), which also stops the pointless back-and-forth shuffling of troops. */
  private findRearSoldier(player: PlayerBase, exclude: TileBase): { unit: UnitBase; tile: TileBase } | null {
    for (const tile of this.ownedTiles(player)) {
      if (tile === exclude) continue;
      if (this.adjacentEnemySoldiers(tile, player) + this.invadersOn(tile, player) > 0) continue; // needed where it is
      if (this.enemyBorderCount(tile, player) > 0) continue; // a border guard — leave it manning the line
      const s = tile.getUnits().find((u) => u.getOwner() === player && u.getType() === 'Soldier');
      if (s) return { unit: s, tile };
    }
    return null;
  }

  /** Bring `tile`'s garrison up to `want` soldiers: relocate a spare rear soldier first
   *  (free), otherwise buy one — always gated on the soldier cap and the solvency rules
   *  so defending never bankrupts us. */
  private *garrison(player: PlayerBase, params: AiParams, tile: TileBase, want: number): Generator<void> {
    while (this.soldiersOn(tile, player) < want && tile.hasSpaceForUnits() && this.budget > 0) {
      const spare = this.findRearSoldier(player, tile);
      if (spare) {
        if (!(yield* this.doAction(() => this.eh.aiMoveUnit(spare.unit, spare.tile, tile)))) break;
        continue;
      }
      if (
        player.getFreeSoldierAmount() > 0 &&
        this.metal(player) >= 50 &&
        this.affords(player, SOLDIER_COST, params.reserve) &&
        this.canAffordUpkeep(player, 30, params.reserve + 600)
      ) {
        if (!(yield* this.doAction(() => this.eh.aiBuyAndPlaceUnit('Soldier', tile)))) break;
        continue;
      }
      break;
    }
  }

  /**
   * Defend first, then build a strike force. The combat rule is simple — an attack only
   * succeeds when the attacker out-numbers the defenders on a tile, and a tile can hold
   * at most 3 units — so 3 soldiers (or an Outpost) make a tile impregnable. So defence
   * means concentrating soldiers on the tiles actually under pressure (and the HQ, whose
   * loss collapses the player via the connectivity rule), bringing each up to out-number
   * the enemy force that can hit it. Rear soldiers are pulled forward for free; fresh
   * hires are still gated on the economy carrying their 30/round wage.
   */
  private *military(player: PlayerBase, params: AiParams): Generator<void> {
    if (!params.military) return;
    const cap = player.getMaxSoldierAmount();
    if (cap <= 0) return;
    const hq = this.om.getHqTile(player);
    const atWar = this.shouldMilitarise(player, params);

    // 1. DEFENCE — garrison every directly-threatened owned tile so its defenders
    //    out-number the enemy force that can reach it (capped at 3 = impregnable). An
    //    Outpost tile is already impregnable, so it never needs a garrison.
    const defend: { tile: TileBase; want: number; pressure: number }[] = [];
    if (hq) {
      const threat = this.adjacentEnemySoldiers(hq, player) + this.invadersOn(hq, player);
      // Hold the HQ at a standing garrison once any enemy can reach us, raised to beat
      // whatever is massing against it — it is the one tile we cannot afford to lose.
      const want = atWar ? Math.min(3, Math.max(params.garrison, threat + 1)) : Math.min(3, threat + 1);
      if (want > 0) defend.push({ tile: hq, want, pressure: threat });
    }
    for (const tile of this.ownedTiles(player)) {
      if (tile === hq) continue;
      if (tile.getBuilding()?.getType() === 'Outpost') continue; // impregnable already
      const threat = this.adjacentEnemySoldiers(tile, player) + this.invadersOn(tile, player);
      if (threat > 0) defend.push({ tile, want: Math.min(3, threat + 1), pressure: threat });
    }
    // DEFEND OUR OWN DEVICE: the Device tile can hold no units, so it is defended by
    // garrisoning its APPROACHES — the owned tiles next to it — to the cap, so the enemy
    // can't get adjacent and stage a conquering unit on it. Forced to the top via a +100
    // synthetic pressure: leaving the Device undefended is an instant loss, so the halved
    // army's first job is to ring it.
    const dt = this.om.findStrangeDeviceTile();
    if (dt && dt.getOwner() === player) {
      for (const n of dt.getNeighbourTiles()) {
        if (n.getOwner() !== player) continue;
        if (n.getBuilding()?.getType() === 'Outpost') continue;
        const threat = this.adjacentEnemySoldiers(n, player) + this.invadersOn(n, player);
        defend.push({ tile: n, want: 3, pressure: threat + 100 });
      }
    }
    // Reinforce the most-pressed shortfalls first (Device approaches carry +100 pressure,
    // so they win the tiebreak among max-shortfall tiles).
    defend.sort((a, b) => (b.want - this.soldiersOn(b.tile, player)) - (a.want - this.soldiersOn(a.tile, player)) || b.pressure - a.pressure);
    for (const d of defend) yield* this.garrison(player, params, d.tile, d.want);

    // 2. BORDER GUARD + STRIKE FORCE — once at war, station soldiers on our frontier
    //    (owned tiles bordering enemy land). A manned border tile can't be taken without
    //    out-numbering it, so it both DETERS attacks and forward-positions the army for a
    //    counter-attack — what the player asked for. Strength scales with the enemy (a
    //    defenceless neighbour needs only a token probe), and is capped by the soldier cap
    //    and what the farms can sustain, so it never over-extends. We keep soldiers already
    //    in place and only pull genuinely-idle rear troops or buy, avoiding troop shuffling.
    if (!atWar || !hq) return;
    const farms = this.ownedTiles(player).filter((t) => t.getBuilding()?.getType() === 'Farm').length;
    const aggression = Math.max(this.enemyThreat(player), this.reachableEnemyMaxDefenders(player) + 1);
    // PANIC: when an enemy holds a Device it is halved NOW — field the biggest army the
    // economy can sustain (garrison()'s per-buy upkeep guard still applies), ignoring the
    // low *visible* defender count (their halved cap reads as weak).
    const force = this.enemyHasDevice(player)
      ? Math.min(cap, farms + 3)
      : Math.min(cap, params.garrison + Math.min(params.strikeForce, aggression + 1), farms + 1);

    const frontier = this.ownedTiles(player)
      .filter((t) => t.getBuilding()?.getType() !== 'Outpost' && this.enemyBorderCount(t, player) > 0)
      .sort((a, b) => this.enemyBorderCount(b, player) - this.enemyBorderCount(a, player));
    // One guard on each frontier tile (most-exposed first), up to our force budget.
    for (const tile of frontier) {
      if (player.getCurrentSoldierAmount() >= force) break;
      yield* this.garrison(player, params, tile, 1);
    }
    // Any remaining budget concentrates on the HQ as the reserve/launch point.
    if (player.getCurrentSoldierAmount() < force) {
      const room = force - player.getCurrentSoldierAmount() + this.soldiersOn(hq, player);
      yield* this.garrison(player, params, hq, Math.min(3, room));
    }
  }

  // --- offence --------------------------------------------------------------

  /**
   * Conquer weak enemy tiles — but only commit to a target we can actually take in a
   * single turn. Dribbling soldiers one at a time just feeds them to the defenders, so
   * we work down the target list (enemy HQ first, then the weakest) and skip any tile
   * we cannot fully overwhelm right now, sending the strike force at the first one we
   * can. One assault per turn keeps the home economy defended.
   */
  /** An owned soldier free to march to the front (not already attacking `exclude`). */
  private findFreeSoldier(player: PlayerBase, exclude: TileBase): { unit: UnitBase; tile: TileBase } | null {
    for (const tile of this.ownedTiles(player)) {
      if (tile === exclude) continue;
      const s = tile.getUnits().find((u) => u.getType() === 'Soldier');
      if (s) return { unit: s, tile };
    }
    return null;
  }

  private *attack(player: PlayerBase, params: AiParams): Generator<void> {
    if (!params.attack) return;
    // Moving an existing soldier onto an undefended enemy tile is free, so we only
    // need the money cushion to *buy* reinforcements. With single-assault AIs we keep
    // the original "need cash to even start" gate, so their behaviour is unchanged.
    const canBuy = this.money(player) >= params.reserve + 250;
    if (params.assaultsPerTurn <= 1 && !canBuy) return;

    const targets = this.om
      .getAvailableTiles()
      .filter((t) => {
        const o = t.getOwner();
        return o !== null && o !== player && t.hasSpaceForConqueringUnits();
      })
      .map((t) => ({
        tile: t,
        defenders: t.getUnits().filter((u) => u.getType() === 'Soldier').length,
        isDevice: t.getBuilding()?.getType() === 'Strange Device', // destroying it stops the loss clock
        isHq: t.getBuilding()?.getType() === 'Headquarters', // conquering an enemy HQ is the prize
        isOutpost: t.getBuilding()?.getType() === 'Outpost', // outposts can never be taken
      }))
      .filter((t) => !t.isOutpost && t.defenders < 3)
      // A reachable enemy Device FIRST (its countdown wins them the game — cracking it
      // resets the clock and reopens the slot), then the HQ (collapses an opponent via the
      // connectivity rule), then the cheapest tiles to flip — undefended border tiles fall
      // to a single soldier and are how a front advances toward the 70% domination line.
      .sort((a, b) => Number(b.isDevice) - Number(a.isDevice) || Number(b.isHq) - Number(a.isHq) || a.defenders - b.defenders);

    const maxAssaults = Math.max(1, params.assaultsPerTurn);
    let assaults = 0;
    for (const { tile, defenders } of targets) {
      if (assaults >= maxAssaults || this.budget <= 0) break;
      const needed = defenders + 1;
      let placed = tile.getConqueringUnits().filter((u) => u.getOwner() === player && u.getType() === 'Soldier').length;
      const toAdd = needed - placed;
      if (toAdd <= 0) continue;

      // How many soldiers can we field for THIS tile = ones we can still move (not
      // already committed elsewhere this turn) + ones we can buy. Recomputed per
      // target so multi-assault never promises the same soldier to two tiles.
      const movable = this.ownedTiles(player).reduce(
        (n, t) => n + (t === tile ? 0 : t.getUnits().filter((u) => u.getType() === 'Soldier').length),
        0,
      );
      const buyable = canBuy
        ? Math.min(
            player.getFreeSoldierAmount(),
            Math.floor(this.metal(player) / 50),
            Math.floor((this.money(player) - params.reserve) / 200),
          )
        : 0;
      if (movable + buyable < toAdd) continue; // can't take it this turn — don't dribble

      while (placed < needed) {
        const spare = this.findFreeSoldier(player, tile);
        let did = false;
        if (spare) {
          did = yield* this.doAction(() => this.eh.aiMoveUnit(spare.unit, spare.tile, tile));
        } else if (canBuy && player.getFreeSoldierAmount() > 0 && this.metal(player) >= 50 && this.affords(player, SOLDIER_COST, params.reserve)) {
          did = yield* this.doAction(() => this.eh.aiBuyAndPlaceUnit('Soldier', tile));
        }
        if (!did) break;
        placed += 1;
      }
      if (placed >= needed) assaults += 1; // count a completed assault; keep pressing the front
    }
  }
}
