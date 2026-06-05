// Save/restore fidelity regression. The reported bug: after a browser reload the
// enemy's state restored incompletely (its units weren't placed on its tiles), so the
// enemy soon collapsed. Root cause: the unit placement ran a legality check tied to the
// *current* player, silently rejecting other players' units. These tests lock the fix.
import { describe, it, expect } from 'vitest';
import { GameSettingsManager } from '../src/managers/gamesettings';
import { ObjectManager } from '../src/managers/objectmanager';
import { PlayerManager } from '../src/managers/playermanager';
import { GameEventHandler } from '../src/managers/gameeventhandler';
import { WorldGenerator } from '../src/world/worldgenerator';
import { AiController } from '../src/managers/ai';
import { MouseHoverBorder } from '../src/model/overlays';
import { Coordinate } from '../src/core/coordinate';
import { TileBase } from '../src/model/tile';
import { PlayerConfig, PlayerBase } from '../src/model/player';
import { buildSnapshot } from '../src/managers/persistence';
// Shared headless harness (also used by the MCTS sandbox, src/ai/nn/sandbox.ts).
import { StubScene, CapturingMenu } from '../src/ai/nn/headless';

const menu = new CapturingMenu();
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
function playRounds(g: ReturnType<typeof setup>, n: number) {
  for (let r = 0; r < n && g.pm.getPlayers().length > 1; r++) {
    const played = new Set<string>(); const sl = g.pm.getPlayers().length; let s = 0;
    while (played.size < sl && g.pm.getPlayers().length > 1 && s++ < 6) {
      const cur = g.pm.getCurrentPlayer();
      if (played.has(cur.getName())) break;
      played.add(cur.getName());
      if (cur.isCpu()) { g.eh.setAiActive(true); g.ai.playTurn(cur); g.eh.setAiActive(false); }
      g.eh.endTurn();
    }
  }
}
const staffedFarms = (om: ObjectManager, p: PlayerBase) =>
  p.getObjects().filter((o): o is TileBase => o instanceof TileBase)
    .filter((t) => t.getBuilding()?.getType() === 'Farm' && t.getUnits().some((u) => u.getType() === 'BasicWorker')).length;

describe('save/restore fidelity', () => {
  const w = 18, h = 14;
  for (const seed of [7, 19, 33]) {
    it(`seed ${seed}: every player (incl. the enemy) restores fully and survives`, () => {
      const configs: PlayerConfig[] = [{ name: 'Aa', difficulty: 'medium' }, { name: 'Bb', difficulty: 'medium' }];
      const g1 = setup(w, h, seed, configs);
      g1.eh.setAiActive(true);
      for (let i = 0; i < configs.length; i++) g1.ai.placeHeadquarters(g1.pm.getCurrentPlayer());
      g1.eh.setAiActive(false);
      playRounds(g1, 24);

      const expected = g1.pm.getPlayers().map((p) => ({
        num: p.getPlayerNum(),
        res: [1, 2, 3, 4].map((r) => p.getResources().get(r)),
        workers: p.getCurrentBasicWorkerAmount(),
        staffedFarms: staffedFarms(g1.om, p),
        tiles: g1.om.getTileCountForPlayer(p),
      }));
      const snap = buildSnapshot(g1.om, g1.pm, { width: w, height: h, seed });

      // Rebuild from the snapshot (regen terrain + restore).
      const g2 = setup(w, h, seed, configs);
      g2.eh.restoreSnapshot(snap);

      // Restored state must match the original for BOTH players — crucially the
      // staffed-farm count (the enemy lost its staffing in the bug).
      for (const exp of expected) {
        const p = g2.pm.getPlayers().find((x) => x.getPlayerNum() === exp.num)!;
        expect([1, 2, 3, 4].map((r) => p.getResources().get(r))).toEqual(exp.res);
        expect(p.getCurrentBasicWorkerAmount()).toBe(exp.workers);
        expect(staffedFarms(g2.om, p)).toBe(exp.staffedFarms);
        expect(g2.om.getTileCountForPlayer(p)).toBe(exp.tiles);
      }

      // And the restored game must keep both players alive for a good while (the bug
      // made the under-restored enemy collapse within ~10 rounds).
      const before = g2.pm.getPlayers().map((p) => p.getName());
      playRounds(g2, 15);
      // No player should vanish purely from a bad restore (a real conquest is fine, but
      // with two matched medium CPUs neither should die this fast from self-collapse).
      const after = g2.pm.getPlayers().map((p) => p.getName());
      expect(after.length).toBe(before.length);
    });
  }
});
