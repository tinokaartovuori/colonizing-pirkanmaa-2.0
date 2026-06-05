// Regression: units placed on a tile must never share an in-tile draw slot (the
// intermittent "units overlap visually" bug). Root cause: adding a unit did not
// renumber the tile's units, so paths that buy & place a unit (aiBuyAndPlaceUnit, the
// human buy path, replaceTile) could leave a newcomer on a stale/duplicate offset.
// addUnit() now renumbers authoritatively, like removeUnit() — locked down here.
import { describe, it, expect } from 'vitest';
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
import { BasicWorker, Soldier, UnitBase } from '../src/model/unit';
import { PlayerConfig } from '../src/model/player';
import { IMenuObjectManager } from '../src/managers/menu-interface';

class StubScene implements IGameScene {
  drawItem() {} removeItem() {} updateItem() {} updateTile() {}
  isObjectInScene() { return true; }
  getObjectInScene(): ISceneObjectHandle { return { setAnimationOption() {}, setAnimationFrame() {} }; }
  addMouseFollowPicture() {} removeMouseFollowItem() {} deleteObjects() {}
}
const stubMenu: IMenuObjectManager = {
  selectFirstTileMenuView() {}, setTileInspectionMenuView() {}, setStatMenuView() {},
  setDefaultMenuView() {}, setUnitShopMenuView() {}, setTieMenu() {}, setWinMenu() {},
  setPlayerLostMenu() {}, setCpuTurnMenuView() {},
};
function setup(seed: number, configs: PlayerConfig[]) {
  const gsm = GameSettingsManager.fromMapDimensions(16, 16);
  const om = new ObjectManager();
  const pm = new PlayerManager(configs, om);
  const eh = new GameEventHandler(om, pm, stubMenu, gsm);
  const scene = new StubScene();
  eh.setGameScene(scene); om.setGameScene(scene); om.addDALS(eh, stubMenu, gsm);
  om.setHoverBorder(new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om));
  new WorldGenerator().generateMap(16, 16, seed, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });
  return { gsm, om, pm, eh, ai: new AiController(eh, om, pm) };
}

/** The (x,y) in-tile offset key for each unit; a duplicate key means a visual overlap. */
function slotKeys(units: UnitBase[]): string[] {
  return units.map((u) => `${u.getTileRelatedCoordinates().x()},${u.getTileRelatedCoordinates().y()}`);
}

describe('in-tile unit layout (no visual overlap)', () => {
  /** Bring a game to a state where the current player legally owns tiles, and return one
   *  of its owned tiles that still has room — a legal placement target for its units. */
  function ownedPlaceableTile(g: ReturnType<typeof setup>) {
    g.eh.setAiActive(true);
    for (let i = 0; i < g.pm.getPlayers().length; i++) g.ai.placeHeadquarters(g.pm.getCurrentPlayer());
    g.eh.setAiActive(false);
    // A couple of end-turns let ownership/availability settle the way a live game has it.
    g.eh.endTurn();
    g.eh.endTurn();
    const player = g.pm.getCurrentPlayer();
    const avail = g.om.getAvailableTiles();
    const tile = player
      .getObjects()
      .filter((o): o is TileBase => o instanceof TileBase)
      .find((t) => t.getBuilding() === null && t.hasSpaceForUnits() && avail.includes(t))!;
    return { player, tile };
  }

  it('addUnit renumbers units into distinct slots even when they carry duplicate offsets', () => {
    const g = setup(7, [{ name: 'A', difficulty: 'hard' }, { name: 'B', difficulty: 'hard' }]);
    const { player, tile } = ownedPlaceableTile(g);

    // Three workers that all carry the SAME stale offset — exactly the state the bug
    // produced. Adding them to a tile must spread them into three distinct slots.
    const ws = [0, 1, 2].map(() => new BasicWorker(g.eh, g.om, g.gsm, player));
    for (const w of ws) w.setTileRelatedCoordinates(0, 1);
    for (const w of ws) tile.addUnit(w);

    const keys = slotKeys(tile.getUnits());
    expect(keys.length).toBe(3);
    expect(new Set(keys).size).toBe(3); // no two share a slot
  });

  it('three soldiers stacked on a tile occupy three distinct slots', () => {
    const g = setup(11, [{ name: 'A', difficulty: 'hard' }, { name: 'B', difficulty: 'hard' }]);
    const { player, tile } = ownedPlaceableTile(g);
    for (let i = 0; i < 3; i++) {
      const s = new Soldier(g.eh, g.om, g.gsm, player);
      s.setTileRelatedCoordinates(0, 1); // simulate the stale-offset bug
      tile.addUnit(s);
    }
    expect(new Set(slotKeys(tile.getUnits())).size).toBe(3);
  });

  it('a full AI game never leaves two units sharing a slot on any tile', () => {
    const { om, pm, eh, ai } = setup(11, [{ name: 'A', difficulty: 'hard' }, { name: 'B', difficulty: 'hard' }]);
    eh.setAiActive(true);
    ai.placeHeadquarters(pm.getCurrentPlayer());
    ai.placeHeadquarters(pm.getCurrentPlayer());
    eh.setAiActive(false);
    for (let r = 0; r < 14 && pm.getPlayers().length > 1; r++) {
      const cur = pm.getCurrentPlayer();
      if (cur.isCpu()) { eh.setAiActive(true); ai.playTurn(cur); eh.setAiActive(false); }
      eh.endTurn();
    }
    for (const t of om.getTiles().filter((x): x is TileBase => x instanceof TileBase)) {
      for (const list of [t.getUnits(), t.getConqueringUnits()]) {
        const keys = slotKeys(list);
        expect(new Set(keys).size).toBe(keys.length);
      }
    }
  });
});
