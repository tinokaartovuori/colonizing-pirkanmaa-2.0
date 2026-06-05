// Fitness evaluation for neuroevolution.
//
// A genome plays a *curriculum* of matches (varied map sizes, player counts,
// seeds, and round caps) from seat 0, against a mix of the hard heuristic AI
// (the anchor we must beat) and Hall-of-Fame genomes (self-play, prevents
// degenerate exploitation and strategy cycling). Reward rewards winning first,
// then dominance margin and speed, and crushes any bankruptcy/crash — so the
// learned policy is strong, decisive, AND keeps the engine's solvency invariant.

import { Genome } from '../src/ai/nn/mlp';
import { TRAINING_CONFIG } from '../src/ai/nn/tiers';
import { playMatch, MatchResult, MatchSpec } from './harness';
import { heuristicFactory, neuralFactory } from './factories';

/** Opponent for a seat: the hard heuristic, or a Hall-of-Fame genome index. */
export type Opponent = { kind: 'heuristic' } | { kind: 'hof'; index: number };

export interface GameTask {
  width: number;
  height: number;
  seed: number;
  roundCap: number;
  /** Number of seats (players) in this game. */
  players: number;
  /** Which seat the genome under test occupies (rotated to remove seat bias). */
  genomeSeat: number;
  /** Opponents for the non-genome seats, in seat order skipping genomeSeat. */
  opponents: Opponent[];
}

// Map sizes, weighted toward small/medium so most games are cheap; large maps
// are still present so the (size-invariant) policy is exercised on them too.
// Engine tile lookups are O(n) so big maps cost far more per game.
const SIZES: Array<[number, number]> = [
  [12, 12], [12, 12], [12, 12], [14, 12], [14, 12], [14, 12],
  [16, 14], [16, 14], [18, 14], [20, 15], [25, 15],
];
// Favour 2-player games (cleanest signal) but include 3–4p for generality.
const PLAYER_COUNTS = [2, 2, 2, 2, 3, 3, 4];

function pick<T>(arr: T[], rand: () => number): T {
  return arr[Math.floor(rand() * arr.length)];
}

/**
 * Build a generation's curriculum. `heurShare` of the opponent slots are the
 * hard heuristic; the rest are random Hall-of-Fame genomes (if any exist).
 * `longShare` of games use a high round cap to exercise deep late-game play on
 * full maps (kept small — long games are expensive and mostly time out).
 */
export function buildCurriculum(
  rand: () => number,
  opts: { games: number; hofSize: number; heurShare?: number; longShare?: number; cap?: number; longCap?: number },
): GameTask[] {
  const heurShare = opts.heurShare ?? 0.5;
  const longShare = opts.longShare ?? 0.12;
  const cap = opts.cap ?? 80;
  const longCap = opts.longCap ?? 180;
  const tasks: GameTask[] = [];
  for (let g = 0; g < opts.games; g++) {
    const [w, h] = pick(SIZES, rand);
    const n = pick(PLAYER_COUNTS, rand);
    const seed = 1 + Math.floor(rand() * 200);
    const roundCap = rand() < longShare ? longCap : cap;
    const genomeSeat = Math.floor(rand() * n); // rotate seat → no seat-0 bias
    const opponents: Opponent[] = [];
    for (let s = 0; s < n - 1; s++) {
      if (opts.hofSize === 0 || rand() < heurShare) opponents.push({ kind: 'heuristic' });
      else opponents.push({ kind: 'hof', index: Math.floor(rand() * opts.hofSize) });
    }
    tasks.push({ width: w, height: h, seed, roundCap, players: n, genomeSeat, opponents });
  }
  return tasks;
}

/**
 * Reward for the genome at `seat`. Winning DOMINATES (the objective is winning),
 * but we also strongly reward owning a large tile share — that is the path to the
 * 70%-domination win and was the missing signal that left earlier policies
 * economically passive (fewer tiles than the heuristic, timing out instead of
 * closing). Crash/bankruptcy are crushed so solvency is never traded away.
 */
export function scoreMatch(r: MatchResult, seat: number, roundCap: number): number {
  if (r.crashed) return -3;
  if (r.bankrupt[seat]) return -3;
  const mine = r.tileFrac[seat];
  let bestOther = 0;
  for (let i = 0; i < r.tileFrac.length; i++) if (i !== seat && r.tileFrac[i] > bestOther) bestOther = r.tileFrac[i];
  const margin = mine - bestOther; // [-1, 1]
  // Economic dominance is shaping; the dominant term is CLOSING the game. The
  // heuristic chronically times out (it can only make one assault/turn), so a
  // policy that actually wins — using its unrestricted multi-assault — is what
  // beats it. Hence: big win bonus, and a penalty for failing to close (timeout).
  let reward = 1.2 * mine + 0.4 * margin;
  if (r.winnerSeat === seat) {
    reward += 2.5; // a decisive win dwarfs any timeout position
    reward += 0.8 * (1 - Math.min(1, r.rounds / roundCap)); // faster is better
  } else if (r.winnerSeat !== null) {
    reward -= 0.8; // someone else won — the worst outcome
  } else {
    reward -= 0.4; // timeout: failed to close the game out
  }
  return reward;
}

export interface GenomeStats {
  fitness: number;
  games: number;
  wins: number;
  gamesVsHeur: number;
  winsVsHeur: number;
  anyBankrupt: boolean;
  anyCrash: boolean;
}

/** Evaluate a genome over the curriculum against the heuristic + HoF genomes. */
export function evalGenome(genome: Genome, tasks: GameTask[], hof: Genome[]): GenomeStats {
  let fitness = 0;
  let wins = 0;
  let gamesVsHeur = 0;
  let winsVsHeur = 0;
  let anyBankrupt = false;
  let anyCrash = false;

  for (const task of tasks) {
    // Build seat factories, placing the genome at task.genomeSeat and filling the
    // remaining seats with the task's opponents in order.
    const factories = new Array(task.players);
    factories[task.genomeSeat] = neuralFactory(genome, TRAINING_CONFIG);
    let allHeur = true;
    let oi = 0;
    for (let s = 0; s < task.players; s++) {
      if (s === task.genomeSeat) continue;
      const opp = task.opponents[oi++];
      if (!opp || opp.kind === 'heuristic' || hof.length === 0) factories[s] = heuristicFactory('hard');
      else { factories[s] = neuralFactory(hof[opp.index % hof.length]); allHeur = false; }
    }
    const spec: MatchSpec = { width: task.width, height: task.height, seed: task.seed, roundCap: task.roundCap, factories };
    const r = playMatch(spec);
    const seat = task.genomeSeat;
    fitness += scoreMatch(r, seat, task.roundCap);
    if (r.winnerSeat === seat) wins++;
    if (r.bankrupt[seat]) anyBankrupt = true;
    if (r.crashed) anyCrash = true;
    if (allHeur) {
      gamesVsHeur++;
      if (r.winnerSeat === seat) winsVsHeur++;
    }
  }
  return {
    fitness: fitness / Math.max(1, tasks.length),
    games: tasks.length,
    wins,
    gamesVsHeur,
    winsVsHeur,
    anyBankrupt,
    anyCrash,
  };
}
