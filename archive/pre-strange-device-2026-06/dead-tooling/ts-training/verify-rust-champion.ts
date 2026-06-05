// Deployment-path verification for a Rust-trained champion (Milestone 8).
//
// Proves a `champion.json` exported by the Rust trainer can be loaded by the
// SAME genome loader the live game uses (src/ai/nn/mlp.ts `Genome`), driven
// through a real `NeuralAiController`, and play a full headless game to a valid
// terminal outcome without crashing or going resource-negative. It then does a
// PROMOTE dry-run: runs emit-weights.ts's `writeWeights` to a TEMP path and
// re-parses the emitted artifact, proving the produced `weights.ts` would load.
//
// It NEVER touches the production src/ai/nn/weights.ts or the real
// training/checkpoints/champion.json — temp paths only.
//
//   vite-node training/verify-rust-champion.ts [-- <champion.json>]
//
// Default input: rust-trainer/checkpoints/smoke/champion.json (the throwaway
// smoke genome — used here only to exercise the pipeline, never deployed).

import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

import { GameSettingsManager } from '../src/managers/gamesettings';
import { ObjectManager } from '../src/managers/objectmanager';
import { PlayerManager } from '../src/managers/playermanager';
import { GameEventHandler } from '../src/managers/gameeventhandler';
import { WorldGenerator } from '../src/world/worldgenerator';
import { AiController } from '../src/managers/ai';
import { MouseHoverBorder } from '../src/model/overlays';
import { Coordinate } from '../src/core/coordinate';
import { IGameScene, ISceneObjectHandle } from '../src/model/base';
import { PlayerBase } from '../src/model/player';
import { IMenuObjectManager } from '../src/managers/menu-interface';

import { Genome, paramCount } from '../src/ai/nn/mlp';
import { NeuralAiController } from '../src/ai/nn/controller';
import { TRAINING_CONFIG } from '../src/ai/nn/tiers';
import { POLICY_INPUT_DIM } from '../src/ai/nn/policy';
import { writeWeights, DEFAULT_TIERS } from './emit-weights';

class StubScene implements IGameScene {
  drawItem(): void {}
  removeItem(): void {}
  updateItem(): void {}
  updateTile(): void {}
  isObjectInScene(): boolean { return true; }
  getObjectInScene(): ISceneObjectHandle { return { setAnimationOption() {}, setAnimationFrame() {} }; }
  addMouseFollowPicture(): void {}
  removeMouseFollowItem(): void {}
  deleteObjects(): void {}
}
class CapturingMenu implements IMenuObjectManager {
  winner: PlayerBase | null = null;
  tie = false;
  selectFirstTileMenuView(): void {}
  setTileInspectionMenuView(): void {}
  setStatMenuView(): void {}
  setDefaultMenuView(): void {}
  setUnitShopMenuView(): void {}
  setTieMenu(): void { this.tie = true; }
  setWinMenu(p: PlayerBase): void { this.winner = p; }
  setPlayerLostMenu(): void {}
  setCpuTurnMenuView(): void {}
}

function rng(seed: number): () => number {
  let s = (seed * 2654435761) >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return (s >>> 0) / 4294967296; };
}

interface Outcome { winner: 'nn' | 'heur' | 'tie' | 'none'; crashed: boolean; nnBankrupt: boolean; rounds: number; }

/** Drive a full headless game: Rust-genome NN (seat 1) vs hard heuristic (seat 0). */
function playFullGame(genome: Genome, width: number, height: number, seed: number, roundCap = 160): Outcome {
  const gsm = GameSettingsManager.fromMapDimensions(width, height);
  const om = new ObjectManager();
  const pm = new PlayerManager([{ name: 'NN', difficulty: 'nn-hard' }, { name: 'H', difficulty: 'hard' }], om);
  const menu = new CapturingMenu();
  const eh = new GameEventHandler(om, pm, menu, gsm);
  const scene = new StubScene();
  eh.setGameScene(scene); om.setGameScene(scene); om.addDALS(eh, menu, gsm);
  om.setHoverBorder(new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om));
  new WorldGenerator().generateMap(width, height, seed, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });

  const players = pm.getPlayers().slice();
  // Construct the controller EXACTLY as the game does (genome + tier + rng),
  // but with the freshly-loaded Rust genome rather than the baked weights.
  const nn = new NeuralAiController(eh, om, pm, genome, TRAINING_CONFIG, rng(seed));
  const heur = new AiController(eh, om, pm);
  const ctrlFor = (p: PlayerBase) => (p.getPlayerNum() === 1 ? nn : heur);

  let crashed = false, nnBankrupt = false;
  try {
    eh.setAiActive(true);
    for (let i = 0; i < players.length; i++) ctrlFor(pm.getCurrentPlayer()).placeHeadquarters(pm.getCurrentPlayer());
    eh.setAiActive(false);
    while (pm.getPlayers().length > 1 && pm.getRoundsPlayed() < roundCap) {
      const cur = pm.getCurrentPlayer();
      if (cur.isCpu()) { eh.setAiActive(true); ctrlFor(cur).playTurn(cur); eh.setAiActive(false); }
      eh.endTurn();
      const nnP = players[0];
      if (pm.getPlayers().includes(nnP) && [...nnP.getResources().values()].some((v) => v < 0)) nnBankrupt = true;
      if (menu.winner || menu.tie) break;
    }
  } catch (e) {
    crashed = true;
    console.error('  game threw:', e);
  }
  const winner = menu.tie ? 'tie' : menu.winner ? (menu.winner.getPlayerNum() === 1 ? 'nn' : 'heur') : 'none';
  return { winner, crashed, nnBankrupt, rounds: pm.getRoundsPlayed() };
}

function main(): void {
  const inPath =
    process.argv[3] ??
    (process.argv.indexOf('--') >= 0 ? process.argv[process.argv.indexOf('--') + 1] : undefined) ??
    'rust-trainer/checkpoints/smoke/champion.json';

  console.log(`[1/3] Loading Rust champion via the game's Genome loader: ${inPath}`);
  const raw = fs.readFileSync(inPath, 'utf8');
  const genome: Genome = JSON.parse(raw);

  // Format compatibility: same fields + correct param count for the arch.
  if (!Array.isArray(genome.arch) || !Array.isArray(genome.params)) {
    throw new Error('champion.json is not a {arch:number[], params:number[]} genome');
  }
  const expected = paramCount(genome.arch);
  console.log(`      arch=${JSON.stringify(genome.arch)} params=${genome.params.length} expected=${expected}`);
  if (genome.params.length !== expected) {
    throw new Error(`param count mismatch: ${genome.params.length} != ${expected}`);
  }
  if (genome.arch[0] !== POLICY_INPUT_DIM) {
    throw new Error(`input dim ${genome.arch[0]} != policy input dim ${POLICY_INPUT_DIM}`);
  }
  console.log('      OK: byte-compatible with the TS Genome loader (mlp.ts).');

  console.log('[2/3] Load-and-play: full headless games with a NeuralAiController on this genome.');
  const matches: Array<[number, number, number]> = [[12, 12, 1], [16, 14, 7], [25, 15, 42]];
  let anyCrash = false, anyBankrupt = false;
  for (const [w, h, seed] of matches) {
    const o = playFullGame(genome, w, h, seed);
    const terminal = o.winner !== 'none' || o.rounds >= 160;
    console.log(
      `      ${w}x${h} seed ${seed}: winner=${o.winner} rounds=${o.rounds} ` +
        `crashed=${o.crashed} nnBankrupt=${o.nnBankrupt} terminal=${terminal}`,
    );
    anyCrash ||= o.crashed;
    anyBankrupt ||= o.nnBankrupt;
    if (!terminal) throw new Error(`game did not reach a valid terminal outcome on ${w}x${h} seed ${seed}`);
  }
  if (anyCrash) throw new Error('a game crashed');
  if (anyBankrupt) throw new Error('the NN player went resource-negative');
  console.log('      OK: every game reached a valid terminal outcome, no crashes, never bankrupt.');

  console.log('[3/3] Promote-to-production DRY RUN: emit weights.ts to a TEMP path (not the real one).');
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'cp-emit-'));
  const tmpOut = path.join(tmpDir, 'weights.ts');
  writeWeights(genome, DEFAULT_TIERS, { source: inPath, generatedFrom: 'verify-rust-champion (DRY RUN)' }, tmpOut);
  const emitted = fs.readFileSync(tmpOut, 'utf8');

  // Well-formedness: must declare NEURAL_WEIGHTS, embed arch, all 3 tiers, and
  // exactly `expected` params. Parse the params array back to confirm count.
  const paramsMatch = emitted.match(/params:\s*\[([^\]]*)\]/);
  if (!paramsMatch) throw new Error('emitted weights.ts has no params array');
  const emittedCount = paramsMatch[1].split(',').filter((s) => s.trim().length > 0).length;
  const checks = [
    ['declares NEURAL_WEIGHTS', emitted.includes('export const NEURAL_WEIGHTS')],
    ['embeds arch', emitted.includes(`arch: ${JSON.stringify(genome.arch)}`)],
    ['has easy/medium/hard tiers', emitted.includes('easy:') && emitted.includes('medium:') && emitted.includes('hard:')],
    [`emits ${expected} params`, emittedCount === expected],
  ] as const;
  for (const [label, ok] of checks) {
    console.log(`      ${ok ? 'OK ' : 'FAIL'}: ${label}`);
    if (!ok) throw new Error(`emitted artifact failed check: ${label}`);
  }
  console.log(`      wrote dry-run artifact to ${tmpOut} (${emittedCount} params) — production weights.ts untouched.`);
  fs.rmSync(tmpDir, { recursive: true, force: true });

  console.log('\nVERIFY OK: Rust champion loads, plays to terminal, and emits a well-formed weights.ts.');
}

main();
