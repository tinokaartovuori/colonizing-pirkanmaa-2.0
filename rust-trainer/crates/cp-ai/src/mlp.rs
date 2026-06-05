//! Port of `src/ai/nn/mlp.ts` — a tiny dependency-free dense MLP.
//!
//! A genome is `{ arch:[57,24,16,1], params:[1809 f64] }`. Per-layer params are
//! the weights (`nin*nout`, row-major: for output `j` the weights live at
//! `offset + j*nin .. +nin`, and the bias at `offset + nin*nout + j`) followed by
//! the biases. Hidden layers use `tanh`, the output layer is linear. ALL math is
//! `f64` to reproduce the TS feature/score values bit-for-bit.

use serde::{Deserialize, Serialize};

/// A genome: the architecture plus the flat parameter vector that fills it.
/// Serde (de)serialises the exact `{arch, params}` JSON the TS uses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub arch: Vec<usize>,
    pub params: Vec<f64>,
}

impl Genome {
    /// Load a genome from the `{arch, params}` JSON used by the TS engine.
    pub fn from_json(s: &str) -> Result<Genome, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// Serialise to the `{arch, params}` JSON the TS engine reads.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("Genome serialises")
    }

    /// Load from a file path.
    pub fn from_file(path: &str) -> std::io::Result<Genome> {
        let s = std::fs::read_to_string(path)?;
        Genome::from_json(&s).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// A genome of all-zero weights (`forward` then returns zeros).
    pub fn zero(arch: &[usize]) -> Genome {
        Genome {
            arch: arch.to_vec(),
            params: vec![0.0; param_count(arch)],
        }
    }
}

/// Number of trainable parameters for a given architecture.
pub fn param_count(arch: &[usize]) -> usize {
    let mut n = 0usize;
    for l in 0..arch.len().saturating_sub(1) {
        let nin = arch[l];
        let nout = arch[l + 1];
        n += nin * nout + nout;
    }
    n
}

/// Forward pass. `input.len()` must equal `arch[0]`. Returns the output-layer
/// activations. Hidden layers: tanh; output: linear. Mirrors the TS loop and
/// indexing exactly.
pub fn forward(genome: &Genome, input: &[f64]) -> Vec<f64> {
    let arch = &genome.arch;
    let params = &genome.params;
    let mut act: Vec<f64> = input.to_vec();
    let mut offset = 0usize;
    for l in 0..arch.len() - 1 {
        let nin = arch[l];
        let nout = arch[l + 1];
        let mut out = vec![0.0f64; nout];
        let is_hidden = l < arch.len() - 2;
        for j in 0..nout {
            // Weights are laid down first, then biases (per layer).
            let mut sum = params[offset + nin * nout + j];
            let base = offset + j * nin;
            for i in 0..nin {
                sum += params[base + i] * act[i];
            }
            out[j] = if is_hidden { sum.tanh() } else { sum };
        }
        offset += nin * nout + nout;
        act = out;
    }
    act
}

/// Convenience: a network whose only output is a single scalar score.
pub fn score(genome: &Genome, input: &[f64]) -> f64 {
    forward(genome, input)[0]
}
