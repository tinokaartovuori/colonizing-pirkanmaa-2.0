//! Port of `src/ai/nn/tiers.ts` + the `TierConfig` type from `candidates.ts`.
//!
//! A single trained network powers all three tiers; only these knobs differ.
//! Training (and nn-hard) use [`TRAINING_CONFIG`].

/// Tunable per-difficulty behaviour (the only thing that differs between tiers).
#[derive(Debug, Clone, Copy)]
pub struct TierConfig {
    /// Max discretionary actions per turn.
    pub budget: i64,
    /// Softmax temperature for intent choice (0 = greedy argmax).
    pub temperature: f64,
    /// Cash reserve kept before discretionary spend (higher = more cautious).
    pub reserve: i64,
    /// Probability of a deliberate blunder (pick a uniformly random legal intent).
    pub blunder: f64,
    /// Allow hiring experts (power plants / mine boosters).
    pub experts: bool,
    /// Allow building / fielding any military (outposts, soldiers, attacks).
    pub military: bool,
    /// Allow nuclear plants.
    pub nuclear: bool,
    /// Allow the Strange Device endgame (the decisive closing move).
    pub device: bool,
}

/// Full-strength config used during training and by nn-hard.
pub const TRAINING_CONFIG: TierConfig = TierConfig {
    budget: 40,
    temperature: 0.0,
    reserve: 120,
    blunder: 0.0,
    experts: true,
    military: true,
    nuclear: true,
    device: true,
};

/// `HARD_CONFIG` — identical to `TRAINING_CONFIG`.
pub const HARD_CONFIG: TierConfig = TRAINING_CONFIG;

/// A good human beats it, but only just.
pub const MEDIUM_CONFIG: TierConfig = TierConfig {
    budget: 14,
    temperature: 0.6,
    reserve: 160,
    blunder: 0.08,
    experts: true,
    military: true,
    nuclear: false,
    device: true,
};

/// Naturally easy to beat.
pub const EASY_CONFIG: TierConfig = TierConfig {
    budget: 6,
    temperature: 1.2,
    reserve: 220,
    blunder: 0.25,
    experts: false,
    military: false,
    nuclear: false,
    device: false,
};
