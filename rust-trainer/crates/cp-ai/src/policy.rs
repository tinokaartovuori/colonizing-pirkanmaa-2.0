//! Port of `src/ai/nn/policy.ts` — maps the network over candidates and selects.
//!
//! Network input per candidate is `[global(36) | intent one-hot(16) | local(16)]`
//! = 68 dims. `scoreCandidate` = `forward(...)[0]`. `select` = blunder check
//! (skipped when `blunder==0`), then argmax at temperature≤1e-6 (strict `>`,
//! ties → LOWEST index), else softmax sampling. Training uses temperature=0,
//! blunder=0 → deterministic argmax.

use crate::candidates::{Candidate, INTENT_COUNT, LOCAL_DIM};
use crate::features::GLOBAL_DIM;
use crate::mlp::{self, Genome};
use crate::tiers::TierConfig;

/// MLP input width for the action-scoring network (= 68 with the MarchSoldier
/// intent: 36 global + 16 intent one-hot + 16 local).
pub const POLICY_INPUT_DIM: usize = GLOBAL_DIM + INTENT_COUNT + LOCAL_DIM;

/// Default architecture: input → 24 → 16 → 1.
pub const DEFAULT_ARCH: [usize; 4] = [POLICY_INPUT_DIM, 24, 16, 1];

/// Build the 68-dim network input for one candidate.
pub fn policy_input(global_vec: &[f64], c: &Candidate) -> Vec<f64> {
    let mut input = Vec::with_capacity(POLICY_INPUT_DIM);
    input.extend_from_slice(global_vec);
    let intent = c.intent as usize;
    for i in 0..INTENT_COUNT {
        input.push(if i == intent { 1.0 } else { 0.0 });
    }
    for i in 0..LOCAL_DIM {
        input.push(c.local.get(i).copied().unwrap_or(0.0));
    }
    input
}

/// `scoreCandidate` — the scalar network head for a candidate.
pub fn score_candidate(genome: &Genome, global_vec: &[f64], c: &Candidate) -> f64 {
    mlp::score(genome, &policy_input(global_vec, c))
}

/// A reproducible RNG (xorshift) used only for blunders/softmax. Unused at
/// temperature=0/blunder=0. Trait so callers can plug their own.
pub trait Rng {
    fn next_f64(&mut self) -> f64;
}

/// `select` — returns the index of the chosen candidate (so the caller can
/// later remove it on a failed execute). Mirrors the TS selection exactly.
pub fn select_index<R: Rng>(
    genome: &Genome,
    global_vec: &[f64],
    candidates: &[Candidate],
    cfg: &TierConfig,
    rand: &mut R,
) -> usize {
    if candidates.len() == 1 {
        return 0;
    }

    // Deliberate blunder (weak tiers only).
    if cfg.blunder > 0.0 && rand.next_f64() < cfg.blunder {
        return (rand.next_f64() * candidates.len() as f64).floor() as usize;
    }

    let scores: Vec<f64> = candidates
        .iter()
        .map(|c| score_candidate(genome, global_vec, c))
        .collect();

    if cfg.temperature <= 1e-6 {
        let mut best = 0usize;
        for i in 1..scores.len() {
            if scores[i] > scores[best] {
                best = i;
            }
        }
        return best;
    }

    // Temperature softmax sampling.
    let t = cfg.temperature;
    let mut max = f64::NEG_INFINITY;
    for &s in &scores {
        if s > max {
            max = s;
        }
    }
    let mut sum = 0.0f64;
    let w: Vec<f64> = scores
        .iter()
        .map(|&s| {
            let e = ((s - max) / t).exp();
            sum += e;
            e
        })
        .collect();
    let mut r = rand.next_f64() * sum;
    for (i, &wi) in w.iter().enumerate() {
        r -= wi;
        if r <= 0.0 {
            return i;
        }
    }
    candidates.len() - 1
}

/// xorshift32 matching `training/harness.ts makeRng` (the trace RNG). Seeded with
/// the game seed; only consumed at temperature>0/blunder>0 (so it never affects
/// the deterministic training/parity runs, but is faithful for the weak tiers).
pub struct XorShift32 {
    state: u32,
}

impl XorShift32 {
    /// `makeRng(seed)`: `s = (seed * 2654435761) >>> 0`, or `0x9e3779b9` if zero.
    pub fn new(seed: u32) -> Self {
        // JS does the multiply in f64 then truncates to u32. Replicate that.
        let prod = (seed as f64) * 2654435761.0;
        let mut s = (prod.rem_euclid(4294967296.0)) as u32;
        if s == 0 {
            s = 0x9e3779b9;
        }
        XorShift32 { state: s }
    }
}

impl Rng for XorShift32 {
    fn next_f64(&mut self) -> f64 {
        let mut s = self.state;
        s ^= s << 13;
        s ^= s >> 17; // logical (unsigned) shift, matching JS on a u32 value
        s ^= s << 5;
        self.state = s;
        (s as f64) / 4294967296.0
    }
}
