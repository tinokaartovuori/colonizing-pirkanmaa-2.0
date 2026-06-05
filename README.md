# Colonizing Pirkanmaa

A 1:1 TypeScript + [Phaser 3](https://phaser.io/) port of **Colonizing Pirkanmaa**, a
turn-based strategy game originally written in C++/Qt for the Tampere University
*Programming 3* course (TIE-02402, 2019–2020) by Otto Ranta-Ojala & Tino Kaartovuori.

Two to four players colonise Pirkanmaa on a procedurally generated tile map:
place a headquarters, harvest resources (money, wood, stone, metal), construct
buildings, recruit workers/experts/soldiers, and conquer the map. Win by holding
70 % of the tiles or by being the last player standing.

Runs entirely in the browser — Windows, macOS, Linux, mobile — no install.

## Fidelity to the original

The game logic is a faithful re-implementation of the original C++ sources (kept
in `reference/` for comparison):

- Exact economy: every build cost, production value, salary and growth timer
  matches `resourcemaps.h`.
- The map generator reproduces the original Windows build's **MSVCRT `rand()`**
  bit-for-bit (`src/core/rng.ts`), so a given seed yields the **same map** the
  original produced. The RNG call order in `WorldGenerator` is preserved exactly.
- Turn flow, conquest, HQ-connectivity cutting, win/lose conditions, and all menu
  views/strings mirror the original. Original sprites and the PressStart2P font
  are used unchanged.

See `tests/` for regression tests that lock the economy, RNG sequence and a full
HQ-placement → farm-harvest loop.

## Development

```bash
npm install
npm run dev      # Vite dev server
npm test         # Vitest
npm run build    # production build to dist/
npm run preview  # serve the production build
```

Node version is pinned in `.nvmrc`.

## Architecture

```
src/
  core/       resources, coordinate, image/animation tables, RNG, descriptions
  model/      BaseObject → GameObject → PlaceableGameObject hierarchy,
              TileBase + tiles, BuildingBase + buildings, UnitBase + units, PlayerBase
  managers/   ObjectManager, PlayerManager, GameSettingsManager, GameEventHandler
  world/      WorldGenerator (seeded, MSVCRT-compatible)
  scenes/     Phaser BootScene (asset preload) + GameScene (map rendering, input)
  ui/         DOM menu panel, start dialog, help window (PressStart2P, original art)
public/assets images/ (108 original PNGs) + fonts/PressStart2P.ttf
reference/    the original C++/Qt sources, for comparison only (not built)
```

The pure-logic layers (`core`, `model`, `managers`, `world`) have no Phaser/DOM
dependency, which is what makes them unit-testable.

## Deployment

Pushing to `main` builds and publishes to GitHub Pages via
`.github/workflows/deploy.yml`. Vite's `base: './'` keeps it working from a
project sub-path. For native desktop binaries, wrap the build with
[Tauri](https://tauri.app/).

## Credits

Original game and all artwork: Otto Ranta-Ojala & Tino Kaartovuori (TUNI).
