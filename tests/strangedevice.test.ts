// Strange Device mechanic — the new, draw-eliminating win condition (see
// STRANGE-DEVICE-DESIGN.md). Drives the real engine headlessly (no Phaser/DOM) the
// same way gameplay.test.ts does, and locks down: countdown scaling, the one-per-game
// uniqueness, the soldier-cap −2 penalty (arc sd5; was a halving) + forced disband, the
// 1-defender garrison (arc sd5; was 0), the win on countdown, and the destroy-on-capture
// (slot reopens, cap restored).

import { describe, it, expect } from 'vitest';
import { GameSettingsManager } from '../src/managers/gamesettings';
import { ObjectManager } from '../src/managers/objectmanager';
import { PlayerManager } from '../src/managers/playermanager';
import { GameEventHandler } from '../src/managers/gameeventhandler';
import { WorldGenerator } from '../src/world/worldgenerator';
import { MouseHoverBorder } from '../src/model/overlays';
import { Coordinate } from '../src/core/coordinate';
import { BasicResource, rmap, strangeDeviceCountdown } from '../src/core/resources';
import { IGameScene, ISceneObjectHandle } from '../src/model/base';
import { TileBase } from '../src/model/tile';
import { Grassland } from '../src/model/tiles';
import { StrangeDevice } from '../src/model/building';
import { PlayerBase } from '../src/model/player';
import { IMenuObjectManager } from '../src/managers/menu-interface';
import { buildSnapshot, GameSnapshot } from '../src/managers/persistence';

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
const stubMenu: IMenuObjectManager = {
  selectFirstTileMenuView() {}, setTileInspectionMenuView() {}, setStatMenuView() {},
  setDefaultMenuView() {}, setUnitShopMenuView() {}, setTieMenu() {}, setWinMenu() {},
  setPlayerLostMenu() {}, setCpuTurnMenuView() {},
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
  om.setHoverBorder(new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om));
  new WorldGenerator().generateMap(width, height, seed, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });
  return { gsm, om, pm, eh };
}

/** Place both HQs on unowned grasslands far apart; returns the two players. */
function placeBothHqs(om: ObjectManager, eh: GameEventHandler, pm: PlayerManager): [PlayerBase, PlayerBase] {
  const grasslands = () => om.getTiles().filter((t) => t.getType() === 'Grassland' && t.getOwner() === null);
  const t1 = grasslands().find((t) => t.getCoordinate().x() >= 2 && t.getCoordinate().y() >= 2)!;
  eh.tileClicked(t1);
  const t2 = grasslands().find((t) => t.getCoordinate().x() >= 8 && t.getCoordinate().y() >= 8) ?? grasslands()[0];
  eh.tileClicked(t2);
  const p1 = pm.getPlayers().find((p) => p.getName() === 'PlayerOne')!;
  const p2 = pm.getPlayers().find((p) => p.getName() === 'PlayerTwo')!;
  return [p1, p2];
}

const grant = (p: PlayerBase) =>
  p.addOrRemoveResources(rmap({
    [BasicResource.MONEY]: 9000, [BasicResource.WOOD]: 2000, [BasicResource.STONE]: 2000, [BasicResource.METAL]: 2000,
  }));

const emptyGrass = (p: PlayerBase): TileBase[] =>
  p.getObjects().filter((o): o is TileBase => o instanceof Grassland && o.getBuilding() === null);

const deviceOf = (om: ObjectManager): StrangeDevice | null => {
  const t = om.findStrangeDeviceTile();
  return t ? (t.getBuilding() as StrangeDevice) : null;
};

describe('Strange Device — countdown scaling', () => {
  it('scales with map size (bigger map = longer countdown)', () => {
    expect(strangeDeviceCountdown(100)).toBe(22); // 10x10  (arc sd5: 12 + 0.10*100)
    expect(strangeDeviceCountdown(144)).toBe(26); // 12x12  (12 + 0.10*144 = 26.4 → 26)
    expect(strangeDeviceCountdown(256)).toBe(38); // 16x16  (12 + 0.10*256 = 37.6 → 38)
    expect(strangeDeviceCountdown(256)).toBeGreaterThan(strangeDeviceCountdown(100));
  });

  it('a built Device gets the map-scaled countdown', () => {
    const { om, pm, eh } = newGame(12, 12, 3);
    const [p1] = placeBothHqs(om, eh, pm);
    grant(p1);
    expect(pm.getCurrentPlayer()).toBe(p1);
    const spot = emptyGrass(p1)[0];
    expect(eh.aiBuildBuilding('Strange Device', spot)).toBe(true);
    expect(deviceOf(om)!.getCountdown()).toBe(strangeDeviceCountdown(om.getTileCount()));
  });
});

describe('Strange Device — uniqueness', () => {
  it('only one can exist; it is removed from buildable lists while it stands', () => {
    const { om, pm, eh } = newGame(12, 12, 5);
    const [p1] = placeBothHqs(om, eh, pm);
    grant(p1);
    const spots = emptyGrass(p1);
    // Before: every owned grassland offers the Device.
    expect(spots[0].getBuildableBuildings()).toContain('Strange Device');
    expect(eh.aiBuildBuilding('Strange Device', spots[0])).toBe(true);
    expect(om.hasStrangeDevice()).toBe(true);
    // After: no other tile offers it, and a second build is refused.
    expect(spots[1].getBuildableBuildings()).not.toContain('Strange Device');
    expect(eh.aiBuildBuilding('Strange Device', spots[1])).toBe(false);
    expect(om.getTiles().filter((t) => t.getBuilding()?.getType() === 'Strange Device').length).toBe(1);
  });
});

describe('Strange Device — soldier-cap −2 penalty + forced disband (arc sd5)', () => {
  it('applies a fixed −2 cap on build and disbands soldiers now over it', () => {
    const { om, pm, eh } = newGame(12, 12, 7);
    const [p1] = placeBothHqs(om, eh, pm);
    grant(p1);
    // HQ alone = soldier cap 1. Field that one soldier on one tile, then build the
    // Device on a DIFFERENT empty tile.
    expect(p1.getMaxSoldierAmount()).toBe(1);
    const spots = emptyGrass(p1);
    expect(eh.aiBuyAndPlaceUnit('Soldier', spots[1])).toBe(true);
    expect(p1.getCurrentSoldierAmount()).toBe(1);
    // Build the Device → cap max(0, 1−2)=0, and the standing soldier is disbanded at once.
    expect(eh.aiBuildBuilding('Strange Device', spots[0])).toBe(true);
    expect(p1.ownsStrangeDevice()).toBe(true);
    expect(p1.getMaxSoldierAmount()).toBe(0);
    expect(p1.getCurrentSoldierAmount()).toBe(0);
  });

  it('on a larger cap the penalty is a flat −2 (not a halving)', () => {
    const { om, pm, eh } = newGame(12, 12, 7);
    const [p1] = placeBothHqs(om, eh, pm);
    grant(p1);
    // Build two Outposts (each +3 soldier cap) → HQ(1) + 6 = cap 7. With a flat −2 the
    // device leaves cap 5 (a halving would have left floor(7/2)=3 — the sd5 difference).
    const spots = emptyGrass(p1);
    expect(eh.aiBuildBuilding('Outpost', spots[1])).toBe(true);
    expect(eh.aiBuildBuilding('Outpost', spots[2])).toBe(true);
    expect(p1.getMaxSoldierAmount()).toBe(7);
    expect(eh.aiBuildBuilding('Strange Device', spots[0])).toBe(true);
    expect(p1.getMaxSoldierAmount()).toBe(5);
  });
});

describe('Strange Device — tile holds at most one defender (arc sd5; stays crackable)', () => {
  it('refuses to build on an occupied tile, then holds exactly one defender', () => {
    const { om, pm, eh } = newGame(12, 12, 7);
    const [p1] = placeBothHqs(om, eh, pm);
    grant(p1);
    expect(p1.getMaxSoldierAmount()).toBe(1); // refresh the (lazily-computed) cap
    const spots = emptyGrass(p1);
    // A tile with a soldier on it cannot host a Device (guardrail: else you could
    // pre-stack defenders and build on top).
    expect(eh.aiBuyAndPlaceUnit('Soldier', spots[1])).toBe(true);
    expect(eh.aiBuildBuilding('Strange Device', spots[1])).toBe(false);
    expect(om.hasStrangeDevice()).toBe(false);
    // Built on an empty tile, the Device tile has room for ONE defender (arc sd5), and an
    // attacker can stage a conquering unit on it, so it stays crackable.
    expect(eh.aiBuildBuilding('Strange Device', spots[0])).toBe(true);
    expect(spots[0].hasSpaceForUnits()).toBe(true);
    expect(spots[0].hasSpaceForConqueringUnits()).toBe(true);
  });

  it('garrisons exactly one soldier, then refuses a second', () => {
    const { om, pm, eh } = newGame(12, 12, 7);
    const [p1] = placeBothHqs(om, eh, pm);
    grant(p1);
    const spots = emptyGrass(p1);
    // Two Outposts so the −2 device penalty still leaves cap room for the 1 defender.
    expect(eh.aiBuildBuilding('Outpost', spots[1])).toBe(true);
    expect(eh.aiBuildBuilding('Outpost', spots[2])).toBe(true);
    const deviceTile = spots[0];
    expect(eh.aiBuildBuilding('Strange Device', deviceTile)).toBe(true);
    expect(deviceTile.hasSpaceForUnits()).toBe(true);
    // Place the one allowed defender on the device tile.
    expect(eh.aiBuyAndPlaceUnit('Soldier', deviceTile)).toBe(true);
    expect(deviceTile.getUnitCount()).toBe(1);
    expect(deviceTile.hasSpaceForUnits()).toBe(false); // full at one defender
    // A second defender is refused.
    expect(eh.aiBuyAndPlaceUnit('Soldier', deviceTile)).toBe(false);
    expect(deviceTile.getUnitCount()).toBe(1);
  });
});

describe('Strange Device — countdown ticks only on the owner\'s turn', () => {
  it('decrements on the owner end-turn, not the opponent\'s', () => {
    const { om, pm, eh } = newGame(12, 12, 3);
    const [p1] = placeBothHqs(om, eh, pm);
    grant(p1);
    eh.aiBuildBuilding('Strange Device', emptyGrass(p1)[0]);
    const start = deviceOf(om)!.getCountdown();
    eh.endTurn(); // P1's turn ends -> owner tick
    expect(deviceOf(om)!.getCountdown()).toBe(start - 1);
    eh.endTurn(); // P2's turn ends -> NOT the owner, no tick
    expect(deviceOf(om)!.getCountdown()).toBe(start - 1);
  });
});

describe('Strange Device — win on countdown', () => {
  it('a Device standing at zero wins the game for its owner', () => {
    const { om, pm, eh } = newGame(12, 12, 3);
    const [p1, p2] = placeBothHqs(om, eh, pm);
    grant(p1);
    eh.aiBuildBuilding('Strange Device', emptyGrass(p1)[0]);
    deviceOf(om)!.setCountdown(1); // about to elapse on P1's end-turn
    eh.endTurn(); // decrement -> 0 -> P1 wins, everyone else loses
    expect(pm.getPlayers()).toEqual([p1]);
    expect(pm.getPlayers()).not.toContain(p2);
  });
});

describe('game ends and locks on a sole survivor', () => {
  it('once one player remains, the match is over and no further turns are possible', () => {
    const { om, pm, eh } = newGame(12, 12, 3);
    const [p1] = placeBothHqs(om, eh, pm);
    grant(p1);
    eh.aiBuildBuilding('Strange Device', emptyGrass(p1)[0]);
    deviceOf(om)!.setCountdown(1);
    eh.endTurn(); // P1 wins
    expect(pm.getPlayers()).toEqual([p1]);
    expect(eh.isGameOver()).toBe(true);
    // The game is locked: further turns and actions are no-ops — the winner cannot play on.
    const roundsAtWin = pm.getRoundsPlayed();
    eh.endTurn();
    eh.endTurn();
    expect(pm.getRoundsPlayed()).toBe(roundsAtWin);
    expect(pm.getPlayers()).toEqual([p1]);
  });
});

describe('Strange Device — destroyed when its tile is lost', () => {
  it('losing the Device tile destroys it, reopens the slot and restores the cap', () => {
    const { om, pm, eh } = newGame(12, 12, 7);
    const [p1, p2] = placeBothHqs(om, eh, pm);
    grant(p1);
    const tile = emptyGrass(p1)[0];
    eh.aiBuildBuilding('Strange Device', tile);
    expect(p1.getMaxSoldierAmount()).toBe(0); // max(0, 1−2) while owned (arc sd5)
    // Simulate the tile being taken (what conquest does: the tile changes owner).
    tile.setOwner(p2);
    eh.endTurn(); // the mismatch (tile owner != Device builder) destroys the Device
    expect(om.hasStrangeDevice()).toBe(false);
    expect(tile.getBuilding()).toBeNull();
    // Cap restored, and another tile may build a new Device again.
    expect(p1.getMaxSoldierAmount()).toBe(1);
    expect(emptyGrass(p1)[0].getBuildableBuildings()).toContain('Strange Device');
  });

  it('a Device on a tile cut off from its HQ (turned neutral) is destroyed too', () => {
    const { om, pm, eh } = newGame(12, 12, 5);
    const [p1] = placeBothHqs(om, eh, pm);
    grant(p1);
    const tile = emptyGrass(p1)[0];
    eh.aiBuildBuilding('Strange Device', tile);
    expect(om.hasStrangeDevice()).toBe(true);
    expect(p1.getMaxSoldierAmount()).toBe(0);
    // The HQ-connectivity cut turns a disconnected tile NEUTRAL (setOwner(null)); the
    // Device, no longer owned by its builder, must be destroyed on the next endTurn.
    tile.setOwner(null);
    eh.endTurn();
    expect(om.hasStrangeDevice()).toBe(false);
    expect(tile.getBuilding()).toBeNull();
    expect(p1.getMaxSoldierAmount()).toBe(1); // cap restored
  });

  it('end-to-end: a Device on a tile genuinely cut from the HQ is destroyed by the real endTurn cut', () => {
    const { om, pm, eh } = newGame(12, 12, 5);
    const [p1] = placeBothHqs(om, eh, pm);
    grant(p1);
    // Build the Device on an owned grassland, then ISOLATE that tile from P1's HQ by
    // making all of its P1-owned 4-neighbours neutral — so the HQ-connectivity flood can
    // no longer reach it. (P1 keeps its HQ, so the cut turns the orphan tile neutral.)
    const tile = emptyGrass(p1)[0];
    eh.aiBuildBuilding('Strange Device', tile);
    for (const n of tile.getNeighbourFourTiles()) {
      if (n.getOwner() === p1 && n.getBuilding()?.getType() !== 'Headquarters') n.setOwner(null);
    }
    eh.endTurn(); // P1's own turn — own tiles are never cut on your own turn; Device stands
    expect(om.hasStrangeDevice()).toBe(true);
    eh.endTurn(); // P2's turn — the connectivity cut runs on P1, orphan tile -> Device destroyed
    expect(om.hasStrangeDevice()).toBe(false);
    expect(tile.getBuilding()).toBeNull();
  });
});

describe('Strange Device — survives game-state save/restore (snapshot round-trip)', () => {
  it('persists the Device position, owner, and countdown through JSON save → load', () => {
    const W = 12, H = 12, SEED = 11;
    const a = newGame(W, H, SEED);
    const [p1] = placeBothHqs(a.om, a.eh, a.pm);
    grant(p1);
    const spot = emptyGrass(p1)[0];
    const dx = spot.getCoordinate().x();
    const dy = spot.getCoordinate().y();
    expect(a.eh.aiBuildBuilding('Strange Device', spot)).toBe(true);
    // Force a distinct mid-game clock so we prove the SAVED time is what comes back,
    // not the freshly-recomputed initial countdown.
    deviceOf(a.om)!.setCountdown(7);
    const ownerNum = deviceOf(a.om)!.getOwner()!.getPlayerNum();

    // Save exactly as localStorage does: snapshot → JSON string → parse back.
    const snap = buildSnapshot(a.om, a.pm, { width: W, height: H, seed: SEED });
    const roundTripped = JSON.parse(JSON.stringify(snap)) as GameSnapshot;

    // The serialised form actually carries the Device + its countdown.
    const devSnap = roundTripped.tiles.find((t) => t.b?.type === 'Strange Device');
    expect(devSnap).toBeDefined();
    expect(devSnap!.x).toBe(dx);
    expect(devSnap!.y).toBe(dy);
    expect(devSnap!.b!.countdown).toBe(7);

    // Restore into a FRESH engine from the same seed and confirm the Device is whole.
    const b = newGame(W, H, SEED);
    b.eh.restoreSnapshot(roundTripped);
    const restoredTile = b.om.findStrangeDeviceTile();
    expect(restoredTile).not.toBeNull();
    expect(restoredTile!.getCoordinate().x()).toBe(dx);
    expect(restoredTile!.getCoordinate().y()).toBe(dy);
    const restored = restoredTile!.getBuilding() as StrangeDevice;
    expect(restored.getCountdown()).toBe(7);
    expect(restored.getOwner()!.getPlayerNum()).toBe(ownerNum);
    expect(b.om.hasStrangeDevice()).toBe(true);
  });
});
