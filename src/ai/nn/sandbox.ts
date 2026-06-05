// Headless, UI-detached engine for the in-browser MCTS search. Given a snapshot
// of the LIVE mid-turn game (built via persistence.buildSnapshot) it constructs a
// fresh ObjectManager + PlayerManager + GameEventHandler wired to a StubScene and
// a capturing stub menu, regenerates the deterministic terrain from the seed, and
// replays the snapshot on top. The restored sandbox's current player is the same
// seat as the live game's current player (snapshot.currentPlayerNum), so the root
// player IS the current player — exactly what the search needs.
//
// The sandbox is built ONCE per search.select() call and then cloned/branched by
// replaying edge actions (see search.ts), mirroring the Rust `search.rs` which
// clones one captured root Game rather than snapshotting per node.

import { GameSettingsManager } from '../../managers/gamesettings';
import { ObjectManager } from '../../managers/objectmanager';
import { PlayerManager } from '../../managers/playermanager';
import { GameEventHandler } from '../../managers/gameeventhandler';
import { WorldGenerator } from '../../world/worldgenerator';
import { MouseHoverBorder } from '../../model/overlays';
import { Coordinate } from '../../core/coordinate';
import { PlayerConfig } from '../../model/player';
import { GameSnapshot } from '../../managers/persistence';
import { StubScene, CapturingMenu } from './headless';

export interface Sandbox {
  gsm: GameSettingsManager;
  om: ObjectManager;
  pm: PlayerManager;
  eh: GameEventHandler;
  menu: CapturingMenu;
}

/**
 * Build a headless engine from a live snapshot. The `seed`/`width`/`height` come
 * from the snapshot's own `settings` block (which buildSnapshot fills from the
 * match settings), so the regenerated terrain is byte-identical to the live map.
 *
 * AI-active is left ON: candidate `execute()` primitives (aiBuildBuilding /
 * aiBuyAndPlaceUnit / aiMoveUnit) require it, and the search drives only AI
 * actions on the sandbox.
 */
export function createSandbox(snap: GameSnapshot): Sandbox {
  const { width, height, seed } = snap.settings;
  const gsm = GameSettingsManager.fromMapDimensions(width, height);
  const om = new ObjectManager();
  const configs: PlayerConfig[] = snap.settings.players.map((p) => ({
    name: p.name,
    difficulty: p.difficulty,
  }));
  const pm = new PlayerManager(configs, om);
  const menu = new CapturingMenu();
  const eh = new GameEventHandler(om, pm, menu, gsm);
  const scene = new StubScene();
  eh.setGameScene(scene);
  om.setGameScene(scene);
  om.addDALS(eh, menu, gsm);
  om.setHoverBorder(new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om));
  new WorldGenerator().generateMap(width, height, seed, {
    objectManager: om,
    eventHandler: eh,
    gameSettings: gsm,
    scene,
  });
  // Restore the live mid-turn state. This also sets the current player to
  // snapshot.currentPlayerNum (the live current player = the search root).
  eh.restoreSnapshot(snap);
  // The search executes only AI primitives; keep AI mode active throughout.
  eh.setAiActive(true);
  return { gsm, om, pm, eh, menu };
}
