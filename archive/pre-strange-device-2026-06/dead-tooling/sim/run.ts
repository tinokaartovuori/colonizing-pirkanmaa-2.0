// Batch simulation runner. Two modes:
//
//   baseline  — run many all-equal-AI games (2/3/4 players) and report how the AI
//               plays: decisiveness, game length, win-by-reason, seat bias, economy.
//   h2h       — head-to-head: seat 0 is the "challenger" (param override from
//               --champ '<json>'), the rest are baseline. Reports challenger win
//               rate, which is the fitness signal for tuning.
//
// Usage:
//   vite-node sim/run.ts -- baseline --games 300 --players 2,3,4 --map 16x16
//   vite-node sim/run.ts -- h2h --games 300 --players 2,3,4 --champ '{"strikeForce":6}'

import { runGame, GameResult, PlayerSpec } from './harness';
import { AiParams } from '../src/managers/ai';

function arg(name: string, def?: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}

const mode = process.argv.find((a) => a === 'baseline' || a === 'h2h') ?? 'baseline';
const gamesPer = parseInt(arg('games', '300')!, 10);
const playerCounts = (arg('players', '2,3,4')!).split(',').map((s) => parseInt(s, 10));
const [mw, mh] = (arg('map', '16x16')!).split('x').map((s) => parseInt(s, 10));
const seedStart = parseInt(arg('seedStart', '1')!, 10);
const roundCap = parseInt(arg('roundCap', '200')!, 10);
const champ: Partial<AiParams> = JSON.parse(arg('champ', '{}')!);
// In h2h mode the NON-challenger seats use this override (e.g. '{"device":false}' so a
// device-builder seat-0 plays a non-builder opponent). Empty = stock baseline AI.
const baseParams: Partial<AiParams> = JSON.parse(arg('baseparams', '{}')!);
const hasBase = Object.keys(baseParams).length > 0;

function mean(xs: number[]): number { return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0; }
function pct(n: number, d: number): string { return d ? ((100 * n) / d).toFixed(1) + '%' : '—'; }

function makePlayers(n: number, challenger: boolean): PlayerSpec[] {
  const ps: PlayerSpec[] = [];
  for (let i = 0; i < n; i++) {
    ps.push({
      name: `P${i + 1}`,
      difficulty: 'hard',
      params: challenger && i === 0 ? champ : hasBase ? baseParams : undefined,
    });
  }
  return ps;
}

interface Agg {
  n: number;
  decisive: number;          // reached a winner (not tie/timeout)
  domination: number;
  lastStanding: number;
  tie: number;
  timeout: number;
  crashed: number;
  gamesWithBankrupt: number;
  rounds: number[];
  decisiveRounds: number[];
  winsBySeat: number[];
  // economy of the eventual winner
  winnerTiles: number[];
  winnerCap: number[];
  winnerSoldiers: number[];
  // challenger (seat 0) record, h2h mode
  challWins: number;
  challBankrupt: number;
  // Strange Device
  deviceBuilt: number; // games where a Device was built
  deviceWin: number;   // games won via a standing Device
}

function emptyAgg(seats: number): Agg {
  return {
    n: 0, decisive: 0, domination: 0, lastStanding: 0, tie: 0, timeout: 0, crashed: 0,
    gamesWithBankrupt: 0, rounds: [], decisiveRounds: [], winsBySeat: new Array(seats).fill(0),
    winnerTiles: [], winnerCap: [], winnerSoldiers: [], challWins: 0, challBankrupt: 0,
    deviceBuilt: 0, deviceWin: 0,
  };
}

function record(agg: Agg, r: GameResult): void {
  agg.n++;
  agg.rounds.push(r.rounds);
  if (r.crashed) agg.crashed++;
  if (r.anyBankrupt) agg.gamesWithBankrupt++;
  if (r.reason === 'tie') agg.tie++;
  else if (r.reason === 'timeout') agg.timeout++;
  else {
    agg.decisive++;
    agg.decisiveRounds.push(r.rounds);
    if (r.reason === 'domination') agg.domination++;
    else agg.lastStanding++;
    if (r.winner !== null) {
      agg.winsBySeat[r.winner]++;
      const w = r.players[r.winner];
      agg.winnerTiles.push(w.finalTiles);
      agg.winnerCap.push(w.finalCap);
      agg.winnerSoldiers.push(w.finalSoldiers);
    }
  }
  if (r.winner === 0) agg.challWins++;
  if (r.players[0]?.bankrupt) agg.challBankrupt++;
  if (r.deviceBuilt) agg.deviceBuilt++;
  if (r.deviceWin) agg.deviceWin++;
}

function report(label: string, agg: Agg): void {
  console.log(`\n=== ${label}  (n=${agg.n}) ===`);
  console.log(`  decisive:   ${pct(agg.decisive, agg.n)}  (domination ${pct(agg.domination, agg.n)}, last-standing ${pct(agg.lastStanding, agg.n)})`);
  console.log(`  tie:        ${pct(agg.tie, agg.n)}   timeout: ${pct(agg.timeout, agg.n)}   crashed: ${agg.crashed}`);
  console.log(`  device:     built in ${pct(agg.deviceBuilt, agg.n)} of games, won via Device ${pct(agg.deviceWin, agg.n)}`);
  console.log(`  bankruptcy: ${pct(agg.gamesWithBankrupt, agg.n)} of games had >=1 player go negative`);
  console.log(`  rounds:     mean ${mean(agg.rounds).toFixed(1)}  (decisive games mean ${mean(agg.decisiveRounds).toFixed(1)})`);
  console.log(`  wins/seat:  [${agg.winsBySeat.map((w) => pct(w, agg.decisive)).join(', ')}]`);
  console.log(`  winner econ: tiles ${mean(agg.winnerTiles).toFixed(1)}, cap ${mean(agg.winnerCap).toFixed(1)}, soldiers ${mean(agg.winnerSoldiers).toFixed(1)}`);
}

const t0 = Date.now();
console.log(`mode=${mode} games/config=${gamesPer} players=${playerCounts.join(',')} map=${mw}x${mh} roundCap=${roundCap}`);
if (mode === 'h2h') console.log(`champion override: ${JSON.stringify(champ)}`);

for (const n of playerCounts) {
  const agg = emptyAgg(n);
  for (let g = 0; g < gamesPer; g++) {
    const seed = seedStart + g;
    const players = makePlayers(n, mode === 'h2h');
    const r = runGame({ width: mw, height: mh, seed, players, roundCap });
    record(agg, r);
  }
  if (mode === 'h2h') {
    report(`${n} players (challenger=seat0)`, agg);
    const exp = agg.decisive / n; // win rate of an average seat among decisive games
    console.log(`  >> challenger wins: ${pct(agg.challWins, agg.n)} of ALL games (fair share = ${pct(1, n)});`);
    console.log(`     among decisive games: ${pct(agg.challWins, agg.decisive)};  challenger bankrupt in ${agg.challBankrupt} games`);
    void exp;
  } else {
    report(`${n} players`, agg);
  }
}
console.log(`\nelapsed ${(Date.now() - t0) / 1000}s`);
