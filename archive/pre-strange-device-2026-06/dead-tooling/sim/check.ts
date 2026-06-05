// Targeted check for the latest tuning: power-plant staffing (a plant produces only with
// expert + worker), nuclear adoption, border-guard soldiers, double-claim, bankruptcy.
// 4 hard CPUs on a big map for a long game (the user's scenario). Run: vite-node sim/check.ts
import { GameSettingsManager } from '../src/managers/gamesettings';
import { ObjectManager } from '../src/managers/objectmanager';
import { PlayerManager } from '../src/managers/playermanager';
import { GameEventHandler } from '../src/managers/gameeventhandler';
import { WorldGenerator } from '../src/world/worldgenerator';
import { AiController } from '../src/managers/ai';
import { MouseHoverBorder } from '../src/model/overlays';
import { Coordinate } from '../src/core/coordinate';
import { IGameScene, ISceneObjectHandle } from '../src/model/base';
import { PlayerBase, PlayerConfig } from '../src/model/player';
import { TileBase } from '../src/model/tile';
import { IMenuObjectManager } from '../src/managers/menu-interface';

class StubScene implements IGameScene {
  drawItem() {} removeItem() {} updateItem() {} updateTile() {}
  isObjectInScene() { return true; }
  getObjectInScene(): ISceneObjectHandle { return { setAnimationOption() {}, setAnimationFrame() {} }; }
  addMouseFollowPicture() {} removeMouseFollowItem() {} deleteObjects() {}
}
const menu: IMenuObjectManager = {
  selectFirstTileMenuView() {}, setTileInspectionMenuView() {}, setStatMenuView() {},
  setDefaultMenuView() {}, setUnitShopMenuView() {}, setTieMenu() {}, setWinMenu() {},
  setPlayerLostMenu() {}, setCpuTurnMenuView() {},
};
function setup(w: number, h: number, seed: number, configs: PlayerConfig[]) {
  const gsm = GameSettingsManager.fromMapDimensions(w, h);
  const om = new ObjectManager();
  const pm = new PlayerManager(configs, om);
  const eh = new GameEventHandler(om, pm, menu, gsm);
  const scene = new StubScene();
  eh.setGameScene(scene); om.setGameScene(scene); om.addDALS(eh, menu, gsm);
  om.setHoverBorder(new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om));
  new WorldGenerator().generateMap(w, h, seed, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });
  return { gsm, om, pm, eh, ai: new AiController(eh, om, pm) };
}
const has = (t: TileBase, ty: string) => t.getUnits().some((u) => u.getType() === ty);
const workers = (t: TileBase) => t.getUnits().filter((u) => u.getType() === 'BasicWorker').length;

function plantStats(p: PlayerBase) {
  let hydro = 0, hydroStaffed = 0, nuke = 0, nukeStaffed = 0, expertSink = 0;
  for (const o of p.getObjects()) {
    if (!(o instanceof TileBase)) continue;
    const b = o.getBuilding()?.getType();
    if (b === 'Hydroelectric Power Plant') { hydro++; if (has(o, 'Expert') && workers(o) >= 1) hydroStaffed++; if (has(o, 'Expert') && workers(o) === 0) expertSink++; }
    if (b === 'Nuclear Power Plant') { nuke++; if (has(o, 'Expert') && workers(o) >= 1) nukeStaffed++; if (has(o, 'Expert') && workers(o) === 0) expertSink++; }
  }
  return { hydro, hydroStaffed, nuke, nukeStaffed, expertSink };
}

const W = 20, H = 16;
const TURNS = 600;
let agg = { hydro: 0, hydroStaffed: 0, nuke: 0, nukeStaffed: 0, expertSink: 0, bankruptGames: 0, games: 0 };
for (const seed of [1, 2, 3, 4]) {
  const configs: PlayerConfig[] = [1, 2, 3, 4].map((i) => ({ name: `P${i}`, difficulty: 'hard' as const }));
  const g = setup(W, H, seed, configs);
  g.eh.setAiActive(true);
  for (let i = 0; i < configs.length; i++) g.ai.placeHeadquarters(g.pm.getCurrentPlayer());
  g.eh.setAiActive(false);
  let bankrupt = false;
  for (let r = 0; r < TURNS && g.pm.getPlayers().length > 1; r++) {
    const cur = g.pm.getCurrentPlayer();
    if (cur.isCpu()) { g.eh.setAiActive(true); g.ai.playTurn(cur); g.eh.setAiActive(false); }
    g.eh.endTurn();
    for (const p of g.pm.getPlayers()) if ([...p.getResources().values()].some((v) => v < 0)) bankrupt = true;
  }
  let line = '';
  for (const p of g.pm.getPlayers()) {
    const s = plantStats(p);
    agg.hydro += s.hydro; agg.hydroStaffed += s.hydroStaffed; agg.nuke += s.nuke; agg.nukeStaffed += s.nukeStaffed; agg.expertSink += s.expertSink;
    line += `${p.getName()}{H${s.hydroStaffed}/${s.hydro} N${s.nukeStaffed}/${s.nuke} sink${s.expertSink} sol${p.getCurrentSoldierAmount()} $${p.getResources().get(1)}} `;
  }
  agg.games++; if (bankrupt) agg.bankruptGames++;
  console.log(`seed ${seed} @${g.pm.getRoundsPlayed()}r surv${g.pm.getPlayers().length} ${bankrupt ? 'BANKRUPT ' : ''}| ${line}`);
}
console.log(`\nTOTALS: hydro staffed ${agg.hydroStaffed}/${agg.hydro}, nuclear staffed ${agg.nukeStaffed}/${agg.nuke}, expert-only sinks ${agg.expertSink}, bankrupt games ${agg.bankruptGames}/${agg.games}`);
