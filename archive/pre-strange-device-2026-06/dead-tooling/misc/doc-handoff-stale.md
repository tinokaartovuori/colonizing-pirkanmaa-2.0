# Handoff — Colonizing Pirkanmaa (2026-05-31)

Continue the in-progress work from here. Read this top-to-bottom before touching code.

## 0. TL;DR status

| Area | State |
|------|-------|
| Build / typecheck (`npx tsc --noEmit`) | ✅ passes |
| `tests/economy.test.ts`, `tests/hqcrash.test.ts` | ✅ pass |
| `tests/gameplay.test.ts` | ✅ **pass** (CPU medium/hard solvency fixed) |
| `tests/aimeasure.test.ts`, `tests/aiduel.test.ts` | ✅ pass |
| AI (`src/managers/ai.ts`) | ✅ **FIXED & enhanced** — solvent, builds a real economy, uses units sensibly. See §3. |
| Everything else from earlier session | ✅ done & verified |

**Nothing is committed.** The whole tree is staged (`AM`) from the initial import; all edits sit on top in the working tree. No git commits were made, no pushes.

## 0.1 AI rewrite — DONE (2026-05-31, continued session)

The §3 blocker is resolved. `src/managers/ai.ts` was reworked into an income-aware,
unit-cap-driven economy bot and verified across many seeds (idle-opponent
`aimeasure` + live CPU-vs-CPU `aiduel`). It **never bankrupts** and grows a deep
economy (cap → 14+, multiple villages/mines/farms, ~34+ tiles by round 40 vs idle).

Key design (all in `ai.ts`):
- **The unit cap is the bottleneck** (HQ = 3 slots; Village +3, Mikontalo +2; soldier
  cap: HQ +1, Outpost +3). The plan keeps income running, raises the cap via Villages,
  then spends freed slots on territory. `maxFarms = cap − 2 − mines` always leaves a
  slot for a forest worker (wood) and a scout (expansion).
- **Never starve income**: farms are always staffed; the "saving-for-mine" pause only
  triggers with ≥2 staffed farms. `netMoneyPerRound()` + `canAffordUpkeep()` +
  `affords()` (reserve + 5×salary buffer) gate every salaried hire.
- **Bootstrap chain**: 1 farm + 1 forest worker → wood 250 → Mine (metal) → Village
  (needs 25 metal) → cap 6 → snowball. `secureWood` keeps a harvester without ever
  pulling a producer; `findSurplusProducerWorker` peels a *stacked* mine worker to
  scout when capped (this is what breaks the "48 tiles but no mine / no cap growth"
  churn — see git history of this file for the dead-ends).
- **Building variety** (per user request): `buildPowerPlants` (Hydro on straight
  rivers, Nuclear when wealthy), `buildOutposts` (soldier cap, only when an enemy is
  actually on the border — `hasReachableEnemy`), plus Farm/Mine/Village and captured
  Mikontalo.
- **Sensible unit use** (per user request): `attack` commits only a force that can
  take a tile *this turn* (no dribbling soldiers to their death), HQ-first then
  weakest target; `military` garrison scales with `enemyThreat()`; `expand` never
  parks a worker on a tile adjacent to an enemy soldier (`tileThreatened`).

Note: on the test maps the two CPUs usually don't share a border (neutral buffer), so
the military code stays (correctly) dormant — it is exercised by code review + the
`tileThreatened`/full-force guards, not by a forced battle. If you want to *see* combat,
force adjacent HQs or use a tiny map. `tests/aiduel.test.ts` is the permanent CPU-vs-CPU
solvency/crash regression guard; `tests/aimeasure.test.ts` remains the scratch tuning
harness (prints `[AI …]` metrics).

## 1. What the project is

A 1:1 TypeScript + Phaser 3 browser port of *Colonizing Pirkanmaa* (a TUNI C++/Qt
strategy game). Turn-based, 2–4 players, human or CPU. See `CLAUDE.md` for the
architecture (pure-logic `core`/`model`/`managers`/`world` layers with zero Phaser
deps; `scenes`/`ui` are the only Phaser/DOM layers; managers talk to the renderer
through interfaces declared in `src/model/base.ts`). Original C++ is in `reference/`
(source of truth for game logic).

### Run / test
```bash
npm run dev      # Vite dev server (prints the port, usually 5174 if 5173 is busy)
npm test         # full vitest suite
npm run build    # tsc --noEmit + vite build
npx vitest run tests/aimeasure.test.ts   # AI tuning harness (see §3)
```
Node pinned in `.nvmrc` (22). Dev handle in browser console: `window.__cp =
{ game, objectManager, playerManager, eventHandler, gsm, ai }` (DEV only).

## 2. Completed & verified earlier this session (keep — do not regress)

1. **Click-to-move units** — click a unit on the map to pick it up, click a
   neighbour to move it (no menu needed). `GameScene.unitAt` hit-test +
   `GameEventHandler.selectUnitForMove`, wired via `onUnitClick` in `main.ts`.
2. **CPU opponents, 3 difficulties** — `Difficulty`/`PlayerConfig` in
   `src/model/player.ts`; per-player Human/CPU-Easy/Medium/Hard selector in
   `src/ui/startdialog.ts`; `AiController` + a turn driver in `main.ts`.
3. **Help / descriptions polished** — `src/ui/help.ts` (gold headings, lists,
   Controls + Computer-players sections), typo fixes in `src/core/descriptions.ts`
   and the unit-shop text.
4. **Turn banner** — `src/ui/banner.ts` ("It's your turn, X!" / "X is playing…").
5. **Viewport fit** — tall maps no longer overflow the bottom. `main.ts` wraps the
   canvas + DOM menu in an inner `#cp-stage` that is CSS-scaled to fit the window
   (`fitStage`, re-runs on resize). Phaser input stays correct because
   `game.scale.refresh()` makes `displayScale` invert the CSS scale (verified:
   displayScale = gameW/renderedW). `index.html` `#game` is the full-viewport
   flex-centre container.
6. **Mikontalo (and all buildings) description fix** — tile-inspection now shows
   the *building's* description, not the terrain underneath. `src/ui/menu.ts`
   `setTileInspectionMenuView`: `building ? building.getBasicDescription() :
   tile.getBasicDescription()` (matches `reference/.../menuobjectmanager.cpp:721`).
7. **Crash on conquering an enemy HQ fixed** — root cause was the Phaser scene
   restart (New Game) leaving destroyed sprites in `GameScene.items`; the 450 ms
   animation tick then called `setTexture` on a destroyed sprite. Fix in
   `src/scenes/GameScene.ts`: `create()` clears `items`/borders/mouse state on
   every (re)start, plus a defensive guard in `applyTexture` (`if
   (!item.sprite.scene) return`). Regression test: `tests/hqcrash.test.ts`.

## 3. ⚠️ IN PROGRESS — AI rewrite (currently BROKEN)

Goal (user): the AI (esp. **Medium**) is too weak — make it much stronger, have it
reason about resources and **use more workers**. Also make CPU moves happen
**gradually** (not all at once), and fix the banner vs CPU.

### What's already changed and works
- **Step-by-step CPU moves (DONE).** `AiController.planTurn(player)` is now a
  **generator** that `yield`s once per action. The driver in `main.ts`
  (`runCpuTurn`) advances it one action per `CPU_ACTION_MS` (320 ms) so the human
  watches moves appear one at a time; `menu.setCpuTurnMenuView(player)` is
  re-rendered each step so the CPU's resources tick down. `playTurn()` (used by
  tests) drains the generator synchronously. **This part is good.**
- **Banner vs CPU (DONE).** `showTurnBanner(..., durationMs=0)` now means
  *persistent* — the "X is playing…" banner stays for the whole CPU turn and is
  replaced by "It's your turn, X!" (2400 ms) when control returns. Wired in
  `main.ts` `driveTurn`.
- **Enemy resources on HQ click (DONE).** `src/ui/menu.ts`: `resourceMenuFor()`
  helper; `setTileInspectionMenuView` shows a rival's money/wood/stone/metal panel
  when you inspect their Headquarters (`building==='Headquarters' && owner!==self`).

### 🔴 The blocker: the AI bankrupts itself
`tests/aimeasure.test.ts` (a scratch harness, prints `[AI medium]…` metrics) shows
medium & hard **go bankrupt at round ~16** (money -5, then lose all tiles). This
fails `tests/gameplay.test.ts` (CPU solvency).

**Diagnosis.** The latest change added a "saving for a mine" mode: when the CPU
owns an un-mined mountain and `wood < 250`, it *pauses farm building* and pours
workers onto forests (to reach the 250 wood a mine needs). But forests produce no
money, and with too few staffed farms the worker **salaries (-5/round each) drain
money faster than income**, so money slides negative. `affords()` only gates *new*
spending; it does not stop the ongoing salary bleed when income < expenses.

**Why the mine matters (the strategic chain we're trying to enable):**
mine on a mountain → **metal** → build **Villages** (need 25 metal) → unit cap +3 →
more workers → stack workers on mines (output scales per worker) + more farms +
soldiers → compounding growth + territory → 70% win. The unit cap (starts at 3, HQ
gives +3) is the real bottleneck; Mikontalo (+2, conquerable) and Villages are the
only ways to raise it, and Villages need metal, which needs a mine, which needs 250
wood. That dependency chain is the whole problem.

**Last-known-good intermediate (solvent but shallow):** before "saving for mine"
mode, the AI reached **45 tiles, cap 5 (took Mikontalo), 5 farms, money 3605,
solvent** — but **never built a mine** (stuck because all workers were on farms, so
wood never reached 250). That is already much stronger than the original Medium and
**stays solvent**.

### Recommended next step (pick one)
- **Option A (safer):** revert just the "saving for mine" logic so the AI is the
  solvent 45-tiles/cap-5/5-farms version, then add mine-building **income-aware**:
  only divert a worker to a forest when projected net money/round stays positive
  (compute `revenue - salaries`; keep ≥2 staffed farms; require a money cushion,
  e.g. `> 800`, before diverting). Re-measure each change with `aimeasure`.
- **Option B:** keep the mine ambition but gate it on solvency: track
  `eh.getCurrentNet()` money for the CPU (it exists), and never hire/keep a worker
  that pushes net money/round below ~0. Only enter saving-for-mine mode with a
  large money buffer and an *idle* worker to divert (never pull a farm worker).

Key AI methods to look at in `src/managers/ai.ts`: `planTurn` (priority order),
`secureWood` (the aggressive forest-staffing that causes the bleed), `buildFarms`
(now converts idle-worker grasslands into farms — good idea, keep), `buildMines`,
`ensureWorker`/`findIdleOnPlain`/`findSpareWorker` (relocation), `raiseUnitCap`
(village), `expand`, `military`, `attack`. `PARAMS` table tunes per difficulty.

**Acceptance for "AI fixed":** `tests/gameplay.test.ts` green again, and in
`aimeasure` medium/hard stay solvent for 40 rounds AND build ≥1 mine and ≥1
village (cap > 5) AND keep expanding. Then Playwright-verify a real game (gradual
moves visible, banner persists during CPU turn, enemy HQ shows resources).

## 3.2 UI rework + persistence — DONE (2026-05-31, continued session)

All of §4 below is now implemented and Playwright-verified.

- **Authentic pixel-art menus.** The original Qt build draws every menu container/
  button as a 9-slice from `multi_0..8.png` (8px pieces) tiled at 16px/cell — raised
  frames for buttons, inverse (180°-rotated) sunken frames for containers. The port
  was wrongly built on the *unused* single `container_2_2.png`/`button_1_2.png`. Fixed:
  a build step composed the 9 pieces into `public/assets/images/multi_frame.png`
  (raised) + `multi_frame_inv.png` (sunken, =frame rotated 180°); `src/ui/styles.ts`
  renders them via CSS `border-image … 8 fill repeat` at `border-width:16px` +
  `image-rendering:pixelated`, drawn by a `::before` layer so text/cells overlay the
  full box (mirrors how the original paints text over the rect). **Gotcha that bit me:**
  do NOT put `position:relative` on `.cp-container`/`.cp-btn` — it overrides
  `.cp-el{position:absolute}` (equal specificity, later wins) and drops every card into
  normal flow. The `::before` host just needs to be positioned, which `.cp-el` already
  provides; footer buttons (not `.cp-el`) get `position:relative` explicitly. Start
  dialog + help reuse the same frames. `CONTAINER_BORDER` in `menu.ts` is now 0.
- **Responsive + centred.** `main.ts fitStage` now scales the stage UP to fill the
  viewport (capped `MAX_STAGE_SCALE=2.5`, 24px margin); `#game` flex-centres it;
  `#cp-stage canvas { image-rendering: pixelated }` keeps it crisp when enlarged.
- **Always-on controls.** `MenuController` has a persistent footer (`cp-footer`) with
  MENU + END TURN, present in every view, floating over the panel bottom (views keep
  their 1:1 cell coords; the strip is empty in every layout so nothing is clipped).
  END TURN disables on CPU/first-tile/end-of-game turns; MENU opens a styled
  "New Game?" confirm before `quitToMenu`.
- **localStorage persistence.** `src/managers/persistence.ts` (snapshot type +
  build/save/load/clear). Terrain is regenerated from the seed; only mutable state is
  saved (tile owners, buildings + Farm growthPhase / HQ conquered, Forest harvest
  state, units incl. conquering, per-player resources, current player, rounds, lost
  players). `GameEventHandler.restoreSnapshot` re-applies it via cost-free
  `placeBuildingDirect`/`placeUnitDirect` (+ `makeHeadquarters` refactor); new setters:
  `PlayerBase.setResources`, `Forest.get/setHarvestState`, `PlayerManager.restore`.
  `main.ts` saves on every turn boundary, after each human tile action, and on
  `beforeunload` (the key hook for refresh); clears on game-over and on MENU→New Game.
  On boot, `showResumeDialog` offers Continue vs New Game. Verified: refresh → Continue
  restores the exact state and the game keeps playing.

## 3.3 Mass-simulation balance pass + bug fixes (2026-05-31, continued)

Built `scripts/sim.mts` (run `npx vite-node scripts/sim.mts [gamesPerConfig]`) — a headless
CPU-vs-CPU mass simulator across difficulties/player-counts/map-sizes/seeds that reports
exceptions, self- vs enemy-bankruptcies, win types (conquest vs 70%-territory vs stall) and
game length. Used it to find/fix bugs and tune balance. Latest 200-game run: **0
exceptions**, win split **≈87% conquest / 13% territory** (target 80/20), 2-player games
**0 self-bankruptcies**.

Bugs fixed (all had tests/sim/browser verification):
- **Cut-off enemy units stayed on the map** (the reported bug): `endTurn`'s HQ-connectivity
  cut and `neutralizePlayer` iterated the *live* `tile.getUnits()` while `deleteUnitFromTile`
  spliced it → iterator-skip left units behind. Fixed with `clearTileUnits` (iterates copies,
  handles conquering units too).
- **Phantom units / latent infinite loop**: `deleteUnitFromTile` never removed the unit from
  its owner's `objects_`, so killed units still counted toward the cap and salaries, and
  `eliminateExcessUnits` could spin forever. Both fixed (`gameeventhandler.ts`, `player.ts`).
- **Win crash**: no longer reproduces (both the 70%-territory and elimination/HQ-conquest win
  paths complete cleanly + render the win menu) — it was a downstream effect of the above
  iterator/lifecycle bugs.
- **AI self-bankruptcy** (hard rule: a CPU must only die to enemy attack): root causes were
  (a) Villages drain wood/round with no forest worker → wood death; (b) money buffer ignored
  Village/Outpost upkeep; (c) over-expansion/over-villaging eroding net to ~0. Fixes in
  `ai.ts`: `ensureWoodIncome` (guarantee harvesters for wood upkeep, runs first), `hasWoodBuffer`,
  `moneyDrainPerRound` in the cash buffer, strict `canAffordUpkeep` (net≥0), village count cap
  tied to harvesters, lenient `affordsIncomeBuild` for farms (fixes a fatal catch-22),
  net-money gates on villages.

AI improvements (`ai.ts` PARAMS + military): added `maxOutposts`/`strikeForce` per difficulty;
the AI now builds Outposts (soldier-cap → real armies), fields an offensive strike force when
an enemy is on its border, and `attack` gathers a full force (moving existing soldiers, not
just buying) to take enemy HQs — this is what makes games end by conquest instead of stalling.

Known remaining limitations (documented, not blocking): 3-/4-player games (esp. medium) stall
more often (~30–40%) and still have a residual self-bankruptcy rate in very long (150+ round)
capped games (wood-harvester-pulled + lumpy-farm-cycle edge cases). 2-player games are clean.

## 3.4 Restore-fidelity fix + stable dev port (2026-05-31, continued)

- **Enemy collapsed after a browser reload** (reported): restoring a saved game placed
  every player's units through `tile.addUnit` → `setLocationTile` → the unit's
  `canBePlacedOnTile`, which checks `getAvailableTiles()` **for the current player**. So
  only the current player's units passed; other players' units silently threw (caught) and
  never landed on their tiles — the enemy restored *unstaffed* and soon collapsed. Fix:
  `PlaceableGameObject.setLocationTileUnchecked` + `TileBase.addUnitRestored` (no legality
  check) used by `placeUnitDirect`. Locked by `tests/restore.test.ts` (asserts both players'
  resources/workers/staffed-farms/tiles match after restore and nobody self-collapses over
  15 post-restore rounds). Browser-verified end-to-end (play → reload → Continue → enemy
  thrives).
- **Dev server port**: `vite.config.ts` now sets `strictPort: true` for `server` and
  `preview` so it always runs on **5173** (and errors loudly on a stale instance) instead of
  silently hopping to 5174/5175.

## 4. NEW requests from the user (DONE — see §3.2)

1. **Persist the game to the browser.** On refresh, the current game should
   survive. Implement save/restore via `localStorage`. Notes:
   - Serialize enough to rebuild: start settings (`StartSettings`: width, height,
     seed, players incl. difficulty) + the *mutable* game state (tile owners,
     buildings + their state e.g. Farm growthPhase / Forest woodLeft / HQ
     conquered, units per tile with type/owner, each player's resources, current
     player index, roundsPlayed, lost players). The map *terrain* is deterministic
     from the seed (MSVCRT RNG, see `src/core/rng.ts`) so you can regenerate
     terrain from the seed and then re-apply ownership/buildings/units/resources.
   - Simplest robust approach: after every state change (end of `tileClicked`,
     `endTurn`, CPU actions) write a snapshot to `localStorage`; on load, if a
     snapshot exists, offer "Continue" vs "New Game". Hook points: `main.ts`
     (startMatch / driveTurn / quitToMenu) and `GameEventHandler`.
   - Watch the object graph: tiles ↔ players ↔ units have back-references; rebuild
     by id. `BaseObject` has an incrementing `ID` (see `src/model/base.ts`).
   - Don't break RNG fidelity: regenerating terrain must use the same seed and the
     same `WorldGenerator` call order. Re-applying state happens *after* generation.
2. **A menu to start a new game.** There's `showStartDialog` already, but the user
   wants an always-reachable way to start a new game (e.g. a persistent "Menu" /
   "New Game" button, or a top bar). `quitToMenu()` in `main.ts` already tears down
   and shows the start dialog — surface a button that calls it. Confirm before
   discarding an in-progress game.
3. **End Turn always reachable.** The `END TURN` button only appears in the default
   menu view (`setDefaultMenuView`). The user wants it visible from *every* menu
   view (or in a fixed spot) so it's easy to press. Options: render an END TURN
   button in a fixed position outside the swappable menu container (e.g. a small
   persistent bar on the stage), or add it to each view in `src/ui/menu.ts`. The
   handler is `this.eh.endTurn()`. Make sure it is disabled/ignored during CPU
   turns (the driver ignores human input via `onTileClick`/`onUnitClick`, but a DOM
   button bypasses that — guard with `playerManager.getCurrentPlayer().isCpu()`).

## 5. Other improvement ideas (user asked for suggestions)

- **Victory progress** indicator (% of map owned vs the 70% threshold) per player.
- **Highlight legal destination tiles** when a unit is picked up (overlays already
  exist: `addBlockTileOverlays` marks the *illegal* ones; consider the inverse).
- **Sound effects / music** (Phaser audio) — build, conquer, turn start.
- **Combat/conquest feedback** — a brief flash/animation when a tile is conquered.
- **"Skip animations / fast CPU" toggle** for impatient players (set
  `CPU_ACTION_MS`/`CPU_START_MS` low).
- **Difficulty/AI personalities** (aggressive vs economic).
- **Keyboard shortcuts** (Space = end turn, Esc = cancel move).
- **Tooltips** on resource icons and buttons.
- **Undo last action** within a turn.

## 6. Files touched this session (source)

- `src/managers/ai.ts` (NEW, generator AI — **broken, see §3**)
- `src/ui/banner.ts` (NEW)
- `src/main.ts` (stage scaling, turn driver, gradual CPU stepping, banner timing)
- `src/scenes/GameScene.ts` (unit hit-test; restart-clears-items + applyTexture guard)
- `src/ui/menu.ts` (CPU-turn view, enemy-resource panel, building-description fix)
- `src/ui/startdialog.ts` (per-player Human/CPU selector)
- `src/ui/styles.ts` (dialog selector, banner, help styles)
- `src/ui/help.ts`, `src/core/descriptions.ts` (text + typos)
- `src/model/player.ts` (Difficulty, PlayerConfig, isCpu)
- `src/managers/playermanager.ts` (config support, hasNoHumanLeft)
- `src/managers/menu-interface.ts` (setCpuTurnMenuView)
- `src/managers/gameeventhandler.ts` (onTurnChanged hook, setAiActive, AI action
  methods `aiBuildBuilding`/`aiBuyAndPlaceUnit`/`aiMoveUnit`, makeUnit/makeBuilding
  refactors, selectUnitForMove)
- `index.html` (unchanged structurally; #game is the viewport flex container)
- Tests: `tests/gameplay.test.ts` (CPU tests — currently failing due to AI),
  `tests/hqcrash.test.ts` (NEW, keep), `tests/aimeasure.test.ts` (NEW, **scratch
  tuning harness** — delete or keep as a tool; it has console.logs and a
  bankrupt-detector).

## 7. Verification gotchas (for Playwright/MCP)

- Driving the Phaser canvas with **synthetic** pointer events is flaky: the real
  Playwright cursor overwrites Phaser's pointer, and Phaser processes input on the
  next frame. Pattern that worked: dispatch `pointermove`+`pointerdown`+`pointerup`
  on the canvas with full coords, `await` 2–3 `requestAnimationFrame`s, and "park"
  the real cursor off-canvas first (click a temporary off-canvas DOM button).
  Driving via `window.__cp.eventHandler` directly is far more reliable for logic.
- DOM menu buttons (`.cp-btn`, `#cp-start`, selects) click reliably with normal
  Playwright element clicks.
- The start dialog / help / win menus are `position:fixed` on `document.body`
  (unaffected by the stage CSS scale). The banner is appended to `#cp-stage`
  (scales with the game).

## 8. Immediate first move next session

1. `npm test` — confirm the 2 gameplay CPU failures are the only reds.
2. Fix the AI bankruptcy (§3, Option A or B). Iterate with
   `npx vitest run tests/aimeasure.test.ts` until solvent + builds mines/villages.
3. Get `tests/gameplay.test.ts` green.
4. Playwright-verify gradual CPU moves + persistent banner + enemy-HQ resources.
5. Then start the new features (§4): localStorage persistence → new-game menu →
   always-on End Turn.
