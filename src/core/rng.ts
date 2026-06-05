// C-runtime rand()/srand() replica.
//
// The original game was built with MinGW on Windows, whose rand() is the
// classic MSVCRT linear congruential generator (RAND_MAX = 32767). Replicating
// it bit-for-bit means a given map seed reproduces the EXACT same map the
// original binary produced. WorldGenerator relies on the precise call order.

const RAND_MAX = 32767;

let state = 1 >>> 0;

/** srand(seed) */
export function srand(seed: number): void {
  state = seed >>> 0;
}

/** rand() — MSVCRT LCG, returns an int in [0, 32767]. */
export function rand(): number {
  // state = state * 214013 + 2531011 (mod 2^32)
  state = (Math.imul(state, 214013) + 2531011) >>> 0;
  return (state >>> 16) & RAND_MAX;
}

export { RAND_MAX };
