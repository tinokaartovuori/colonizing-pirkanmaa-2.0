//! C-runtime `rand()`/`srand()` replica — FIDELITY-CRITICAL.
//!
//! Faithful port of `src/core/rng.ts`. The original game was built with MinGW
//! on Windows, whose `rand()` is the classic MSVCRT linear congruential
//! generator (`RAND_MAX = 32767`). Replicating it bit-for-bit means a given map
//! seed reproduces the EXACT same map the original Windows binary produced.
//! `WorldGenerator` relies on the precise order of `rand()` calls, so this must
//! stay a verbatim transcription of the LCG.
//!
//! TS reference:
//! ```text
//! state = (Math.imul(state, 214013) + 2531011) >>> 0;   // mod 2^32, unsigned
//! return (state >>> 16) & 32767;                          // top bits, masked
//! ```
//! `Math.imul` is 32-bit signed multiply with wraparound; `>>> 0` reinterprets
//! as `u32`. We model the whole thing in `u32` (wrapping arithmetic), which is
//! bit-identical.

/// The MSVCRT `RAND_MAX`.
pub const RAND_MAX: u32 = 32767;

/// Deterministic MSVCRT-compatible LCG. Holds the 32-bit LCG state.
///
/// Field name maps to the TS module-level `state` variable. Unlike the TS
/// version (which uses a single global), this is an explicit value type so the
/// sim stays free of hidden global RNG state — construct one per generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rng {
    state: u32,
}

impl Rng {
    /// Equivalent to constructing then calling `srand(seed)`.
    ///
    /// Note: the TS module initialises `state = 1` at load time. A fresh `Rng`
    /// constructed here with `new(1)` matches that pre-seed default.
    pub fn new(seed: u32) -> Self {
        Rng { state: seed }
    }

    /// `srand(seed)` — reset the generator state.
    pub fn srand(&mut self, seed: u32) {
        self.state = seed;
    }

    /// `rand()` — MSVCRT LCG, returns an int in `[0, 32767]`.
    pub fn rand(&mut self) -> u32 {
        // state = state * 214013 + 2531011 (mod 2^32)
        self.state = self
            .state
            .wrapping_mul(214013)
            .wrapping_add(2531011);
        (self.state >> 16) & RAND_MAX
    }
}

impl Default for Rng {
    /// Matches the TS module's load-time default of `state = 1`.
    fn default() -> Self {
        Rng { state: 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected sequences were produced by running the actual TypeScript
    // implementation (src/core/rng.ts) under vite-node (node v26):
    //
    //   import { srand, rand } from 'src/core/rng.ts';
    //   srand(SEED); for (let i=0;i<10;i++) console.log(rand());
    //
    // captured 2026-06-01. These are the ground truth the Rust port must match
    // bit-for-bit.
    #[test]
    fn matches_ts_seed_12345() {
        let mut rng = Rng::new(12345);
        let got: Vec<u32> = (0..10).map(|_| rng.rand()).collect();
        assert_eq!(
            got,
            vec![7584, 19164, 25795, 22125, 5828, 23405, 27477, 5413, 29072, 23404]
        );
    }

    #[test]
    fn matches_ts_seed_1() {
        let mut rng = Rng::new(1);
        let got: Vec<u32> = (0..10).map(|_| rng.rand()).collect();
        assert_eq!(
            got,
            vec![41, 18467, 6334, 26500, 19169, 15724, 11478, 29358, 26962, 24464]
        );
    }

    #[test]
    fn matches_ts_seed_0() {
        let mut rng = Rng::new(0);
        let got: Vec<u32> = (0..10).map(|_| rng.rand()).collect();
        assert_eq!(
            got,
            vec![38, 7719, 21238, 2437, 8855, 11797, 8365, 32285, 10450, 30612]
        );
    }

    #[test]
    fn srand_resets_sequence() {
        let mut rng = Rng::new(999);
        rng.rand();
        rng.rand();
        rng.srand(1);
        assert_eq!(rng.rand(), 41);
    }

    #[test]
    fn output_in_range() {
        let mut rng = Rng::new(42);
        for _ in 0..10_000 {
            assert!(rng.rand() <= RAND_MAX);
        }
    }
}
