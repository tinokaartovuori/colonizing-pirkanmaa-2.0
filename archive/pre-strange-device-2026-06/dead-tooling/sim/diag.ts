// One-off diagnostic: play a few 2p hard-vs-hard games and, at the end, dump each
// player's economy/military shape so we can see WHY games stall (army size, outposts,
// mines, metal). Run: vite-node sim/diag.ts
import { runGame } from './harness';

for (const seed of [1, 2, 3, 4, 5, 6, 7, 8]) {
  const r = runGame({
    width: 16, height: 16, seed, roundCap: 120,
    players: [
      { name: 'P1', difficulty: 'hard' },
      { name: 'P2', difficulty: 'hard' },
    ],
  });
  const line = r.players
    .map((p) => `P${p.num}{tiles:${p.finalTiles},cap:${p.finalCap},sol:${p.finalSoldiers},wk:${p.finalWorkers},${JSON.stringify(p.buildings)}}`)
    .join('  ');
  console.log(`seed ${seed}: ${r.reason} @${r.rounds}r  | ${line}`);
}
