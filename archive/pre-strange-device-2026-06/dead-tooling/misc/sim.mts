// Headless CPU-vs-CPU mass simulator. Plays many full games across difficulties,
// player counts, map sizes and seeds with a stub scene/menu, and aggregates:
//   - exceptions thrown (correctness bugs)
//   - bankruptcies, flagged self-inflicted vs enemy-caused
//   - win types: conquest/elimination vs 70%-territory vs stall vs tie
//   - game length
// Run: npx vite-node scripts/sim.mts [gamesPerConfig]
import { GameSettingsManager } from '../src/managers/gamesettings';
import { ObjectManager } from '../src/managers/objectmanager';
import { PlayerManager } from '../src/managers/playermanager';
import { GameEventHandler } from '../src/managers/gameeventhandler';
import { WorldGenerator } from '../src/world/worldgenerator';
import { AiController } from '../src/managers/ai';
import { MouseHoverBorder } from '../src/model/overlays';
import { Coordinate } from '../src/core/coordinate';
import { IGameScene, ISceneObjectHandle } from '../src/model/base';
import { TileBase } from '../src/model/tile';
import { PlayerConfig, PlayerBase, Difficulty } from '../src/model/player';
import { IMenuObjectManager } from '../src/managers/menu-interface';

class StubScene implements IGameScene {
  drawItem() {}
  removeItem() {}
  updateItem() {}
  updateTile() {}
  isObjectInScene() { return true; }
  getObjectInScene(): ISceneObjectHandle { return { setAnimationOption() {}, setAnimationFrame() {} }; }
  addMouseFollowPicture() {}
  removeMouseFollowItem() {}
  deleteObjects() {}
}

interface LostEvent { names: string[]; reasons: string[]; }
function makeStubMenu(capture: { lost: LostEvent[] }): IMenuObjectManager {
  return {
    selectFirstTileMenuView() {}, setTileInspectionMenuView() {}, setStatMenuView() {},
    setDefaultMenuView() {}, setUnitShopMenuView() {}, setCpuTurnMenuView() {},
    setWinMenu() {},
    setTieMenu(players: PlayerBase[], reasons: string[]) { capture.lost.push({ names: players.map((p) => p.getName()), reasons: [...reasons] }); },
    setPlayerLostMenu(players: PlayerBase[], reasons: string[]) { capture.lost.push({ names: players.map((p) => p.getName()), reasons: [...reasons] }); },
  };
}

function setup(width: number, height: number, seed: number, configs: PlayerConfig[]) {
  const capture = { lost: [] as LostEvent[] };
  const gsm = GameSettingsManager.fromMapDimensions(width, height);
  const om = new ObjectManager();
  const pm = new PlayerManager(configs, om);
  const menu = makeStubMenu(capture);
  const eh = new GameEventHandler(om, pm, menu, gsm);
  const scene = new StubScene();
  eh.setGameScene(scene); om.setGameScene(scene); om.addDALS(eh, menu, gsm);
  om.setHoverBorder(new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om));
  new WorldGenerator().generateMap(width, height, seed, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });
  return { gsm, om, pm, eh, ai: new AiController(eh, om, pm), capture };
}

const MAX_ROUNDS = 200;

interface GameResult {
  outcome: 'conquest' | 'territory' | 'stall' | 'tie' | 'error';
  rounds: number;
  winner: string | null;
  winnerFrac: number;
  bankruptcies: { name: string; round: number; selfInflicted: boolean }[];
  error?: string;
}

function playGame(width: number, height: number, seed: number, configs: PlayerConfig[]): GameResult {
  const bankruptcies: GameResult['bankruptcies'] = [];
  try {
    const { om, pm, eh, ai, capture } = setup(width, height, seed, configs);
    const total = om.getTileCount();
    // Place every HQ (placing advances the turn).
    eh.setAiActive(true);
    for (let i = 0; i < configs.length; i++) ai.placeHeadquarters(pm.getCurrentPlayer());
    eh.setAiActive(false);

    // tile-count history per player name for bankruptcy classification
    const tileHist = new Map<string, number[]>();
    const allNames = pm.getPlayers().map((p) => p.getName());
    for (const n of allNames) tileHist.set(n, []);

    let round = 0;
    for (; round < MAX_ROUNDS && pm.getPlayers().length > 1; round++) {
      const lostBefore = capture.lost.length;
      // one full round: each currently-alive player takes a turn
      const startLen = pm.getPlayers().length;
      let safety = 0;
      const playedThisRound = new Set<string>();
      while (playedThisRound.size < startLen && pm.getPlayers().length > 1 && safety++ < 8) {
        const cur = pm.getCurrentPlayer();
        if (playedThisRound.has(cur.getName())) break;
        playedThisRound.add(cur.getName());
        if (cur.isCpu()) { eh.setAiActive(true); ai.playTurn(cur); eh.setAiActive(false); }
        eh.endTurn();
      }
      // record tile counts
      for (const p of pm.getPlayers()) tileHist.get(p.getName())!.push(om.getTileCountForPlayer(p));
      // classify any new losses
      for (let i = lostBefore; i < capture.lost.length; i++) {
        const ev = capture.lost[i];
        ev.names.forEach((name, idx) => {
          if (ev.reasons[idx] === 'noresources') {
            const hist = tileHist.get(name) ?? [];
            const peak = hist.length ? Math.max(...hist) : 0;
            const current = hist.length ? hist[hist.length - 1] : 0;
            // Self-inflicted only if the player never lost meaningful territory to an
            // enemy — i.e. it bankrupted while still near its peak size. A player that
            // had been cut/conquered down (current well below peak) is an enemy-caused loss.
            const selfInflicted = peak > 0 && current >= peak * 0.85;
            bankruptcies.push({ name, round, selfInflicted });
          }
        });
      }
    }

    if (pm.getPlayers().length === 1) {
      const winner = pm.getPlayers()[0];
      const frac = total > 0 ? om.getTileCountForPlayer(winner) / total : 0;
      return { outcome: frac >= 0.7 ? 'territory' : 'conquest', rounds: round, winner: winner.getName(), winnerFrac: +frac.toFixed(2), bankruptcies };
    }
    if (pm.getPlayers().length === 0) return { outcome: 'tie', rounds: round, winner: null, winnerFrac: 0, bankruptcies };
    return { outcome: 'stall', rounds: round, winner: null, winnerFrac: 0, bankruptcies };
  } catch (e) {
    return { outcome: 'error', rounds: 0, winner: null, winnerFrac: 0, bankruptcies, error: String(e instanceof Error ? e.stack : e).split('\n').slice(0, 3).join(' | ') };
  }
}

// ---- configs ----------------------------------------------------------------
const diffs: Difficulty[] = ['easy', 'medium', 'hard'];
const cfg = (...ds: Difficulty[]): PlayerConfig[] => ds.map((d, i) => ({ name: ['Aa', 'Bb', 'Cc', 'Dd'][i], difficulty: d }));
const CONFIGS: { label: string; players: PlayerConfig[]; w: number; h: number }[] = [
  { label: '2p easy', players: cfg('easy', 'easy'), w: 14, h: 12 },
  { label: '2p medium', players: cfg('medium', 'medium'), w: 14, h: 12 },
  { label: '2p hard', players: cfg('hard', 'hard'), w: 14, h: 12 },
  { label: '2p mixed', players: cfg('easy', 'hard'), w: 14, h: 12 },
  { label: '3p medium', players: cfg('medium', 'medium', 'medium'), w: 18, h: 14 },
  { label: '3p hard', players: cfg('hard', 'hard', 'hard'), w: 18, h: 14 },
  { label: '4p hard', players: cfg('hard', 'hard', 'hard', 'hard'), w: 20, h: 15 },
  { label: '4p mixed', players: cfg('easy', 'medium', 'hard', 'medium'), w: 20, h: 15 },
  { label: '2p hard small', players: cfg('hard', 'hard'), w: 10, h: 10 },
  { label: '2p hard big', players: cfg('hard', 'hard'), w: 25, h: 15 },
];

const gamesPerConfig = Number(process.argv[2] ?? 60);
const grand = { games: 0, conquest: 0, territory: 0, stall: 0, tie: 0, error: 0, selfBankrupt: 0, enemyBankrupt: 0, roundsSum: 0 };
const errors: string[] = [];
const selfBankruptDetail: string[] = [];

for (const c of CONFIGS) {
  const agg = { conquest: 0, territory: 0, stall: 0, tie: 0, error: 0, selfB: 0, enemyB: 0, rounds: 0 };
  for (let g = 0; g < gamesPerConfig; g++) {
    const seed = 1 + ((g * 7 + 3) % 200);
    const r = playGame(c.w, c.h, seed, c.players);
    agg[r.outcome === 'conquest' ? 'conquest' : r.outcome === 'territory' ? 'territory' : r.outcome === 'stall' ? 'stall' : r.outcome === 'tie' ? 'tie' : 'error']++;
    agg.rounds += r.rounds;
    for (const b of r.bankruptcies) { if (b.selfInflicted) { agg.selfB++; if (selfBankruptDetail.length < 12) selfBankruptDetail.push(`${c.label} seed${seed} ${b.name}@r${b.round}`); } else agg.enemyB++; }
    if (r.error && errors.length < 12) errors.push(`${c.label} seed${seed}: ${r.error}`);
    grand.games++; grand.roundsSum += r.rounds;
    grand.conquest += r.outcome === 'conquest' ? 1 : 0;
    grand.territory += r.outcome === 'territory' ? 1 : 0;
    grand.stall += r.outcome === 'stall' ? 1 : 0;
    grand.tie += r.outcome === 'tie' ? 1 : 0;
    grand.error += r.outcome === 'error' ? 1 : 0;
    grand.selfBankrupt += r.bankruptcies.filter((b) => b.selfInflicted).length;
    grand.enemyBankrupt += r.bankruptcies.filter((b) => !b.selfInflicted).length;
  }
  const n = gamesPerConfig;
  console.log(
    `${c.label.padEnd(16)} conquest=${String(agg.conquest).padStart(3)} territory=${String(agg.territory).padStart(3)} stall=${String(agg.stall).padStart(3)} tie=${String(agg.tie).padStart(2)} err=${agg.error} | selfBankrupt=${agg.selfB} enemyBankrupt=${agg.enemyB} | avgRounds=${(agg.rounds / n).toFixed(0)}`,
  );
}

const decided = grand.conquest + grand.territory;
console.log('\n==== TOTAL', grand.games, 'games ====');
console.log(`conquest=${grand.conquest} territory=${grand.territory} stall=${grand.stall} tie=${grand.tie} error=${grand.error}`);
if (decided > 0) console.log(`win split (of decided): conquest=${((grand.conquest / decided) * 100).toFixed(0)}%  territory=${((grand.territory / decided) * 100).toFixed(0)}%`);
console.log(`SELF-INFLICTED bankruptcies=${grand.selfBankrupt}  (enemy-caused=${grand.enemyBankrupt})`);
console.log(`avg rounds=${(grand.roundsSum / grand.games).toFixed(1)}`);
if (errors.length) { console.log('\nERRORS:'); errors.forEach((e) => console.log('  ' + e)); }
if (selfBankruptDetail.length) { console.log('\nSELF-BANKRUPT samples:'); selfBankruptDetail.forEach((e) => console.log('  ' + e)); }
