//! `cp-ai` — the neuroevolution AI for the *Colonizing Pirkanmaa* simulator.
//!
//! A 1:1 port of the TypeScript neural AI in `src/ai/nn/` (mlp, metrics,
//! features, candidates, policy, tiers, safety, controller). EXACT behavioural
//! parity with the TS is the goal; a later milestone (M5) asserts it against the
//! golden traces in `rust-trainer/golden/`.
//!
//! All feature / metric / MLP math is `f64`; game-state resources are `i64` in
//! `cp_sim` and are converted to `f64` exactly where the TS does.
//!
//! ## Entry points
//! - [`controller::NeuralAiController::place_headquarters`] / [`plan_turn`] — one
//!   AI turn for a player.
//! - [`run::run_game`] — a full deterministic self-play game between N genomes
//!   (what `cp-train` calls).
//!
//! [`plan_turn`]: controller::NeuralAiController::plan_turn

pub mod candidates;
pub mod cnn;
pub mod controller;
pub mod features;
pub mod hard_ai;
pub mod metrics;
pub mod mlp;
pub mod planes;
pub mod policy;
pub mod policy_spatial;
pub mod policy_train;
pub mod run;
pub mod safety;
pub mod search;
pub mod selfplay;
pub mod spatial;
pub mod spatial_net;
pub mod tiers;
pub mod value;

pub use controller::{DecisionCandidate, DecisionTrace, NeuralAiController};
pub use hard_ai::{HardAi, AiParams, HARD_PARAMS, MEDIUM_PARAMS, EASY_PARAMS, DEVICE_RUSH_PARAMS, ARMY_RUSH_PARAMS};
pub use mlp::{forward, param_count, score, Genome};
pub use policy::{Rng, XorShift32, DEFAULT_ARCH, POLICY_INPUT_DIM};
pub use run::{
    run_game, run_game_telemetry, GameReason, GameResult, GameTelemetry, PlayerResult,
    SeatTelemetry,
};
pub use search::{select as mcts_select, LeafEval, SearchConfig};
pub use tiers::{TierConfig, EASY_CONFIG, HARD_CONFIG, MEDIUM_CONFIG, TRAINING_CONFIG};
pub use value::{ValueExample, ValueNet, ValueTrainer, VALUE_ARCH};
