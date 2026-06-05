//! `cp-train` — the neuroevolution training harness and parity tooling.
//!
//! Holds the GA self-play loop, fitness, hall-of-fame, JSONL logging, and
//! champion export, plus the `train` and `parity` binaries. Depends on
//! [`cp_sim`] and [`cp_ai`].

pub mod reward;

pub use reward::{fitness_v2, RewardConfig};
