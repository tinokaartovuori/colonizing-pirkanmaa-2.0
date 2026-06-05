// Transient banner shown over the map when a turn begins ("It's your turn, X" /
// "X is playing..."). One banner at a time; a new one replaces the old.

const COLOR_BALL = ['red', 'blue', 'purple', 'yellow'];

let current: HTMLElement | null = null;
let hideTimer: number | null = null;

/**
 * Show the banner. `durationMs <= 0` keeps it up until the next banner replaces
 * it or clearBanner() is called — used for the persistent "X is playing…" label
 * that stays for the whole CPU turn.
 */
export function showTurnBanner(parent: HTMLElement, text: string, playerNum?: number, durationMs = 1800): void {
  clearBanner();

  const banner = document.createElement('div');
  banner.className = 'cp-root cp-banner';

  if (playerNum && playerNum >= 1 && playerNum <= 4) {
    const dot = document.createElement('img');
    dot.className = 'cp-dot';
    dot.src = `assets/images/${COLOR_BALL[playerNum - 1]}.png`;
    banner.appendChild(dot);
  }
  const span = document.createElement('span');
  span.textContent = text;
  banner.appendChild(span);

  parent.appendChild(banner);
  current = banner;

  // Next frame: add the visible class so the opacity transition runs.
  requestAnimationFrame(() => banner.classList.add('cp-show'));

  if (durationMs > 0) {
    hideTimer = window.setTimeout(() => {
      banner.classList.remove('cp-show');
      window.setTimeout(() => {
        if (current === banner) clearBanner();
      }, 300);
    }, durationMs);
  }
}

export function clearBanner(): void {
  if (hideTimer !== null) {
    clearTimeout(hideTimer);
    hideTimer = null;
  }
  if (current) {
    current.remove();
    current = null;
  }
}
