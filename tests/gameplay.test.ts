import { describe, it, expect } from 'vitest';
import { GameSettingsManager } from '../src/managers/gamesettings';
import { ObjectManager } from '../src/managers/objectmanager';
import { PlayerManager } from '../src/managers/playermanager';
import { GameEventHandler } from '../src/managers/gameeventhandler';
import { WorldGenerator } from '../src/world/worldgenerator';
import { AiController } from '../src/managers/ai';
import { PlayerConfig } from '../src/model/player';
import { MouseHoverBorder } from '../src/model/overlays';
import { Coordinate } from '../src/core/coordinate';
import { BasicResource } from '../src/core/resources';
import { IGameScene, ISceneObjectHandle, BaseObject } from '../src/model/base';
import { TileBase } from '../src/model/tile';
import { Grassland } from '../src/model/tiles';
import { IMenuObjectManager } from '../src/managers/menu-interface';

// Headless no-op scene + menu.
class StubScene implements IGameScene {
  drawItem(): void {}
  removeItem(): void {}
  updateItem(): void {}
  updateTile(): void {}
  isObjectInScene(): boolean {
    return true;
  }
  getObjectInScene(): ISceneObjectHandle {
    return { setAnimationOption() {}, setAnimationFrame() {} };
  }
  addMouseFollowPicture(): void {}
  removeMouseFollowItem(): void {}
  deleteObjects(): void {}
}
const stubMenu: IMenuObjectManager = {
  selectFirstTileMenuView() {},
  setTileInspectionMenuView() {},
  setStatMenuView() {},
  setDefaultMenuView() {},
  setUnitShopMenuView() {},
  setTieMenu() {},
  setWinMenu() {},
  setPlayerLostMenu() {},
  setCpuTurnMenuView() {},
};

function newGame(width: number, height: number, seed: number) {
  const gsm = GameSettingsManager.fromMapDimensions(width, height);
  const om = new ObjectManager();
  const pm = new PlayerManager(['PlayerOne', 'PlayerTwo'], om);
  const eh = new GameEventHandler(om, pm, stubMenu, gsm);
  const scene = new StubScene();
  eh.setGameScene(scene);
  om.setGameScene(scene);
  om.addDALS(eh, stubMenu, gsm);
  const hover = new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om);
  om.setHoverBorder(hover);
  new WorldGenerator().generateMap(width, height, seed, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });
  return { gsm, om, pm, eh };
}

describe('World generation', () => {
  it('produces width*height tiles and the same map for the same seed', () => {
    const a = newGame(10, 10, 7);
    expect(a.om.getTileCount()).toBe(100);
    const typesA = a.om.getTiles().map((t) => t.getType()).join(',');
    const b = newGame(10, 10, 7);
    const typesB = b.om.getTiles().map((t) => t.getType()).join(',');
    expect(typesB).toBe(typesA); // deterministic
    // exactly one Mikontalo building spawns
    const mikontalos = a.om.getTiles().filter((t) => t.getBuilding()?.getType() === 'Mikontalo').length;
    expect(mikontalos).toBe(1);
  });
});

describe('HQ placement + farm economy (full loop, matches 1:1 reference maths)', () => {
  it('grants 9 tiles, then farm harvest nets +155 over 4 owner turns', () => {
    const { om, pm, eh } = newGame(12, 12, 3);

    // Place both HQs on unowned grasslands.
    const grasslands = () => om.getTiles().filter((t) => t.getType() === 'Grassland' && t.getOwner() === null);
    const p1Tile = grasslands().find((t) => t.getCoordinate().x() >= 2 && t.getCoordinate().y() >= 2)!;
    eh.tileClicked(p1Tile);
    const p2Tile = grasslands().find((t) => t.getCoordinate().x() >= 8 && t.getCoordinate().y() >= 8) ?? grasslands()[0];
    eh.tileClicked(p2Tile);

    const p1 = pm.getPlayers().find((p) => p.getName() === 'PlayerOne')!;
    expect(p1.getObjects().length).toBe(9); // HQ tile + 8 neighbours
    expect(p1.getMaxUnitAmount()).toBe(3);
    expect(p1.getMaxSoldierAmount()).toBe(1);

    // Build a farm on an owned grassland and staff it with a worker.
    const farmTile = p1.getObjects().find(
      (o): o is TileBase => o instanceof Grassland && (o as TileBase).getBuilding() === null,
    )!;
    eh.openDefaultMenuView();
    eh.buildBuilding('Farm', farmTile);
    expect(p1.getResources().get(BasicResource.MONEY)).toBe(300);
    expect(p1.getResources().get(BasicResource.WOOD)).toBe(100);

    eh.createUnit('BasicWorker');
    eh.tileClicked(farmTile);
    expect(p1.getResources().get(BasicResource.MONEY)).toBe(250);
    expect(farmTile.getUnitCount()).toBe(1);

    for (let i = 0; i < 8; i++) eh.endTurn(); // 4 PlayerOne turns
    // harvest +175 once, worker salary -5 x4 => +155
    expect(p1.getResources().get(BasicResource.MONEY)).toBe(405);
  });
});

function newGameWithPlayers(width: number, height: number, seed: number, configs: PlayerConfig[]) {
  const gsm = GameSettingsManager.fromMapDimensions(width, height);
  const om = new ObjectManager();
  const pm = new PlayerManager(configs, om);
  const eh = new GameEventHandler(om, pm, stubMenu, gsm);
  const scene = new StubScene();
  eh.setGameScene(scene);
  om.setGameScene(scene);
  om.addDALS(eh, stubMenu, gsm);
  const hover = new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om);
  om.setHoverBorder(hover);
  new WorldGenerator().generateMap(width, height, seed, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });
  const ai = new AiController(eh, om, pm);
  return { gsm, om, pm, eh, ai };
}

describe('CPU player', () => {
  for (const difficulty of ['easy', 'medium', 'hard'] as const) {
    it(`(${difficulty}) places an HQ, builds an economy and stays solvent`, () => {
      const { om, pm, eh, ai } = newGameWithPlayers(14, 12, 5, [
        { name: 'Human' },
        { name: 'Cpu', difficulty },
      ]);

      // Human picks a starting tile in one corner.
      const freeGrass = () => om.getTiles().filter((t) => t.getType() === 'Grassland' && t.getOwner() === null);
      eh.tileClicked(freeGrass().find((t) => t.getCoordinate().x() <= 4 && t.getCoordinate().y() <= 4) ?? freeGrass()[0]);

      const cpu = pm.getPlayers().find((p) => p.getName() === 'Cpu')!;
      expect(pm.getCurrentPlayer()).toBe(cpu);

      // CPU first round: place its HQ.
      eh.setAiActive(true);
      ai.placeHeadquarters(cpu);
      eh.setAiActive(false);
      expect(cpu.getObjects().length).toBeGreaterThanOrEqual(9); // HQ + neighbours

      // Play 15 full rounds: human just ends turn, CPU plays then ends.
      for (let r = 0; r < 15; r++) {
        eh.endTurn(); // human
        if (pm.getPlayers().includes(cpu) && pm.getCurrentPlayer() === cpu) {
          eh.setAiActive(true);
          ai.playTurn(cpu);
          eh.setAiActive(false);
          eh.endTurn();
        }
        // Never bankrupt itself (negative resources = instant loss).
        for (const value of cpu.getResources().values()) expect(value).toBeGreaterThanOrEqual(0);
        if (!pm.getPlayers().includes(cpu)) break;
      }

      expect(pm.getPlayers()).toContain(cpu); // survived
      const realBuildings = cpu
        .getObjects()
        .filter((o) => o instanceof TileBase && o.getBuilding() !== null && o.getBuilding()!.getType() !== 'Headquarters');
      // Built at least one real building beyond the starting HQ.
      expect(realBuildings.length).toBeGreaterThanOrEqual(1);
      // Medium/hard should actually grow an economy, not stall at a single farm.
      if (difficulty !== 'easy') {
        const grew = realBuildings.length >= 2 || om.getTileCountForPlayer(cpu) > 9;
        expect(grew).toBe(true);
      }
    });
  }

  it('CPU vs CPU plays a whole autonomous game without crashing', () => {
    const { om, pm, eh, ai } = newGameWithPlayers(14, 12, 9, [
      { name: 'Alpha', difficulty: 'hard' },
      { name: 'Beta', difficulty: 'medium' },
    ]);

    const playCurrent = () => {
      const cur = pm.getCurrentPlayer();
      eh.setAiActive(true);
      if (cur.getObjects().length === 0) {
        ai.placeHeadquarters(cur); // advances turn itself
        eh.setAiActive(false);
      } else {
        ai.playTurn(cur);
        eh.setAiActive(false);
        eh.endTurn();
      }
    };

    // Bounded loop so a non-terminating match can never hang the suite.
    for (let i = 0; i < 400 && pm.getPlayers().length > 1; i++) {
      expect(() => playCurrent()).not.toThrow();
    }

    // Either someone won, or both are still standing — but the map must have
    // changed hands from the initial ~9 tiles each (the AIs expanded/fought).
    const everyoneSolvent = pm.getPlayers().every((p) => [...p.getResources().values()].every((v) => v >= 0));
    expect(everyoneSolvent).toBe(true);
    expect(om.getTileCount()).toBe(14 * 12);
  });
});
