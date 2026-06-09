// Port of startdialog.{ui,cpp} — the initial game configuration dialog, extended
// with a per-player Human/Computer selector.

import { Difficulty, PlayerConfig } from '../model/player';
import { AI_ROSTER, rosterCharacterFor } from '../managers/gamerecorder';

export interface StartSettings {
  width: number;
  height: number;
  seed: number;
  players: PlayerConfig[];
}

function checkCharacters(s: string): boolean {
  const lower = s.toLowerCase();
  for (const c of lower) {
    if (c < 'a' || c > 'z') return false;
  }
  return true;
}

const DEFAULT_NAMES = ['PlayerOne', 'PlayerTwo', 'PlayerThree', 'PlayerFour'];
// Opponent types offered in New Game: a human, or one of the three named AI
// CHARACTERS — Jorma (heuristic HARD bot), Kalevi (AlphaZero) and Gunnar
// (AlphaZero XL). Picking an AI LOCKS the seat's name to that character. The
// roster (names / difficulty strings / labels) lives in gamerecorder.ts.
const TYPE_OPTIONS: Array<[Difficulty, string]> = [
  ['human', 'Human'],
  ...AI_ROSTER.map((c) => [c.difficulty, c.label] as [Difficulty, string]),
];
// Player 2 starts as Kalevi (AlphaZero) so a single human gets a strong opponent
// straight away.
const DEFAULT_TYPES: Difficulty[] = ['human', 'model:kalevi', 'human', 'human'];

/** On load, when a saved game exists: let the player resume it or start fresh. */
export function showResumeDialog(onContinue: () => void, onNewGame: () => void): void {
  const overlay = document.createElement('div');
  overlay.className = 'cp-root cp-overlay';
  const dialog = document.createElement('div');
  dialog.className = 'cp-dialog';
  dialog.style.textAlign = 'center';
  dialog.innerHTML =
    `<h2>Colonizing Pirkanmaa</h2>` +
    `<p style="font-size:9px;line-height:1.7;margin:0 0 4px">A saved game was found.</p>` +
    `<div class="cp-actions"><button id="cp-continue" class="cp-primary">Continue</button><button id="cp-newgame">New Game</button></div>`;
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);
  dialog.querySelector('#cp-continue')!.addEventListener('click', () => {
    overlay.remove();
    onContinue();
  });
  dialog.querySelector('#cp-newgame')!.addEventListener('click', () => {
    overlay.remove();
    onNewGame();
  });
}

export function showStartDialog(onStart: (s: StartSettings) => void): void {
  const overlay = document.createElement('div');
  overlay.className = 'cp-root cp-overlay';
  const dialog = document.createElement('div');
  dialog.className = 'cp-dialog';

  const typeSelect = (i: number) =>
    `<select id="cp-t${i}" class="cp-type">` +
    TYPE_OPTIONS.map(([v, t]) => `<option value="${v}"${DEFAULT_TYPES[i] === v ? ' selected' : ''}>${t}</option>`).join('') +
    `</select>`;

  const playerRow = (i: number) =>
    `<div class="cp-row cp-prow" id="cp-prow${i}">` +
    `<input id="cp-p${i + 1}" type="text" value="${DEFAULT_NAMES[i]}" maxlength="15">` +
    typeSelect(i) +
    `</div>`;

  dialog.innerHTML = `
    <h2>New Game</h2>
    <div class="cp-row"><label>Map width (10-25)</label><input id="cp-w" type="number" min="10" max="25" value="14"></div>
    <div class="cp-row"><label>Map height (10-15)</label><input id="cp-h" type="number" min="10" max="15" value="12"></div>
    <div class="cp-row"><label>Map seed (1-200)</label><input id="cp-seed" type="number" min="1" max="200" value="1"><button id="cp-rand" class="cp-mini">Random</button></div>
    <div class="cp-row"><label>Players (2-4)</label><input id="cp-players" type="number" min="2" max="4" value="2"></div>
    <div class="cp-section">Name &amp; type for each player</div>
    ${playerRow(0)}${playerRow(1)}${playerRow(2)}${playerRow(3)}
    <div class="cp-error" id="cp-err"></div>
    <div class="cp-actions"><button id="cp-start" class="cp-primary">Start</button><button id="cp-exit">Exit</button></div>
  `;
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  const $ = <T extends HTMLElement>(id: string) => dialog.querySelector(`#${id}`) as T;
  const w = $<HTMLInputElement>('cp-w');
  const h = $<HTMLInputElement>('cp-h');
  const seed = $<HTMLInputElement>('cp-seed');
  const players = $<HTMLInputElement>('cp-players');
  const err = $<HTMLDivElement>('cp-err');
  const rows = [0, 1, 2, 3].map((i) => $<HTMLDivElement>(`cp-prow${i}`));

  const clamp = (input: HTMLInputElement, min: number, max: number) => {
    let v = parseInt(input.value, 10);
    if (isNaN(v)) v = min;
    v = Math.max(min, Math.min(max, v));
    input.value = String(v);
    return v;
  };

  const syncRows = () => {
    const n = clamp(players, 2, 4);
    rows.forEach((row, i) => {
      row.style.display = i < n ? 'flex' : 'none';
    });
  };
  players.addEventListener('input', syncRows);
  syncRows();

  // Per-seat name locking: when a seat's type is an AI character (Jorma / Kalevi /
  // Gunnar), the name input is forced to that character's name and made read-only;
  // switching back to Human restores the player's last-typed (editable) name. The
  // human name is stashed per seat so toggling type doesn't lose it.
  const humanNames = [...DEFAULT_NAMES];
  // Lock AI seats to their character name; when the SAME AI is chosen by more than one
  // active seat, suffix them (Jorma1, Jorma2, …) so they're distinguishable. Human seats
  // keep their editable, last-typed name. Recomputed across all seats on any change.
  const relabelSeats = () => {
    const n = clamp(players, 2, 4);
    const chars = [0, 1, 2, 3].map((i) => {
      const nameInput = $<HTMLInputElement>(`cp-p${i + 1}`);
      const character = rosterCharacterFor($<HTMLSelectElement>(`cp-t${i}`).value as Difficulty);
      if (character && !nameInput.readOnly) humanNames[i] = nameInput.value; // stash before locking
      return character ?? null;
    });
    // How many ACTIVE seats use each character (only active seats actually play).
    const total: Record<string, number> = {};
    for (let i = 0; i < n; i++) if (chars[i]) total[chars[i]!.name] = (total[chars[i]!.name] ?? 0) + 1;
    const seen: Record<string, number> = {};
    for (let i = 0; i < 4; i++) {
      const nameInput = $<HTMLInputElement>(`cp-p${i + 1}`);
      const character = chars[i];
      if (character) {
        let label = character.name;
        if (i < n && total[character.name] > 1) {
          seen[character.name] = (seen[character.name] ?? 0) + 1;
          label = `${character.name}${seen[character.name]}`;
        }
        nameInput.value = label;
        nameInput.readOnly = true;
        nameInput.classList.add('cp-locked');
      } else {
        if (nameInput.readOnly) nameInput.value = humanNames[i]; // restore on unlock
        nameInput.readOnly = false;
        nameInput.classList.remove('cp-locked');
      }
    }
  };
  [0, 1, 2, 3].forEach((i) => {
    $<HTMLSelectElement>(`cp-t${i}`).addEventListener('change', relabelSeats);
  });
  players.addEventListener('input', relabelSeats); // player-count change re-dedups active seats
  relabelSeats(); // apply initial lock state (e.g. seat 2 defaults to Kalevi)

  $<HTMLButtonElement>('cp-rand').addEventListener('click', () => {
    seed.value = String(Math.floor(Math.random() * 200) + 1);
  });

  $<HTMLButtonElement>('cp-exit').addEventListener('click', () => {
    overlay.remove();
  });

  $<HTMLButtonElement>('cp-start').addEventListener('click', () => {
    const width = clamp(w, 10, 25);
    const height = clamp(h, 10, 15);
    const seedVal = clamp(seed, 1, 200);
    const playerNum = clamp(players, 2, 4);

    const config: PlayerConfig[] = [];
    for (let i = 0; i < playerNum; i++) {
      const name = $<HTMLInputElement>(`cp-p${i + 1}`).value;
      const difficulty = $<HTMLSelectElement>(`cp-t${i}`).value as Difficulty;
      if (!checkCharacters(name)) {
        err.textContent = 'Names may only contain letters a-z.';
        return;
      }
      if (name.length === 0) {
        err.textContent = 'Enter a name for every player.';
        return;
      }
      config.push({ name, difficulty });
    }

    overlay.remove();
    onStart({ width, height, seed: seedVal, players: config });
  });
}
