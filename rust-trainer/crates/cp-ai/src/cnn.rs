//! Minimal hand-rolled CNN primitives for the spatial AlphaZero representation.
//!
//! ADDITIVE and AZ-ONLY: nothing here is wired into the parity-locked feature /
//! policy / search path. This module just provides the layer math (forward +
//! backward) so a spatial value/policy head can be built and trained later. All
//! math is `f64`; the idiom mirrors `mlp.rs` (hand-rolled, serde-persistable
//! param structs, tanh hidden + linear out).
//!
//! Feature maps are flat `Vec<f64>` with shape (channels C, height H, width W)
//! and index layout `idx(c,y,x) = (c*H + y)*W + x`. H and W are passed at
//! forward time so every layer is board-size-agnostic.

use serde::{Deserialize, Serialize};

/// Flat index into a (C, H, W) feature map.
#[inline]
pub fn idx(c: usize, y: usize, x: usize, h: usize, w: usize) -> usize {
    (c * h + y) * w + x
}

/// Flat base offset of channel `c` in a (C, H, W) feature map (i.e. `idx(c,0,0)`),
/// given the per-channel area `hw = h*w`.
#[inline]
fn in_ch_out(c: usize, hw: usize) -> usize {
    c * hw
}

// ---------------------------------------------------------------------------
// Tiny seeded RNG for small random init (local xorshift, no external crates).
// Mirrors the spirit of `policy.rs`'s XorShift32 but is self-contained so this
// module has no cross-dependency on the parity-locked policy file.
// ---------------------------------------------------------------------------

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        // Avoid a zero state; splitmix-style scramble of the seed.
        let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
        if s == 0 {
            s = 0x1234_5678_9abc_def0;
        }
        Lcg { state: s }
    }
    /// Next f64 in [0,1).
    fn next_f64(&mut self) -> f64 {
        // xorshift64.
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        // Top 53 bits → [0,1).
        ((x >> 11) as f64) / ((1u64 << 53) as f64)
    }
    /// Next f64 in [-1,1).
    fn next_signed(&mut self) -> f64 {
        self.next_f64() * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Conv2d — zero-padded same-size 2D convolution (cross-correlation).
// weights layout: [oc][ic][ky][kx]; bias: [oc].
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv2d {
    pub in_ch: usize,
    pub out_ch: usize,
    /// Odd kernel size (e.g. 3).
    pub k: usize,
    pub pad: usize,
    /// `out_ch * in_ch * k * k`, layout `[oc][ic][ky][kx]`.
    pub weights: Vec<f64>,
    /// `out_ch`.
    pub bias: Vec<f64>,
}

impl Conv2d {
    /// Deterministic small-random init via a seeded xorshift. Weights uniform in
    /// `[-1,1) * sqrt(1/fan_in)` with `fan_in = in_ch*k*k`; biases zero.
    pub fn new_seeded(in_ch: usize, out_ch: usize, k: usize, pad: usize, seed: u64) -> Self {
        let mut rng = Lcg::new(seed);
        let fan_in = (in_ch * k * k).max(1);
        let scale = (1.0 / fan_in as f64).sqrt();
        let n = out_ch * in_ch * k * k;
        let weights = (0..n).map(|_| rng.next_signed() * scale).collect();
        let bias = vec![0.0; out_ch];
        Conv2d { in_ch, out_ch, k, pad, weights, bias }
    }

    #[inline]
    fn w_idx(&self, oc: usize, ic: usize, ky: usize, kx: usize) -> usize {
        ((oc * self.in_ch + ic) * self.k + ky) * self.k + kx
    }

    /// Forward pass. `input` is `in_ch*h*w`. Output is `out_ch*h*w` (same H,W when
    /// `pad == k/2`).
    pub fn forward(&self, input: &[f64], h: usize, w: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; self.out_ch * h * w];
        self.forward_into(input, h, w, &mut out);
        out
    }

    /// Allocation-free forward used by the inference hot path. Writes `out_ch*h*w`
    /// values into `out` (resized + fully overwritten). Mathematically identical
    /// to [`forward`](Self::forward) up to f64 summation-order rounding (this
    /// module is AZ-only and not bit-reproducibility-locked); the allocating
    /// version is retained for the gradient-tested code paths.
    ///
    /// For the common `k == 3, pad == 1` case (every conv in the spatial net) a
    /// fast path handles the zero-padded borders separately so the interior loop
    /// has NO per-tap bounds branch and walks three contiguous 3-wide input/weight
    /// slices per input channel. Any other kernel size falls back to the general
    /// bounds-checked path so tests at other sizes still pass.
    pub fn forward_into(&self, input: &[f64], h: usize, w: usize, out: &mut Vec<f64>) {
        out.clear();
        out.resize(self.out_ch * h * w, 0.0);
        if self.k == 3 && self.pad == 1 {
            self.forward_into_k3(input, h, w, out);
            return;
        }
        self.forward_into_general(input, h, w, out);
    }

    /// General zero-padded cross-correlation forward (any kernel size). Slow but
    /// always correct; the gradient-tested reference path.
    fn forward_into_general(&self, input: &[f64], h: usize, w: usize, out: &mut [f64]) {
        let k = self.k;
        let pad = self.pad as isize;
        for oc in 0..self.out_ch {
            let b = self.bias[oc];
            for oy in 0..h {
                for ox in 0..w {
                    let mut sum = b;
                    for ic in 0..self.in_ch {
                        for ky in 0..k {
                            let iy = oy as isize + ky as isize - pad;
                            if iy < 0 || iy >= h as isize {
                                continue;
                            }
                            for kx in 0..k {
                                let ix = ox as isize + kx as isize - pad;
                                if ix < 0 || ix >= w as isize {
                                    continue;
                                }
                                let inv = input[idx(ic, iy as usize, ix as usize, h, w)];
                                sum += self.weights[self.w_idx(oc, ic, ky, kx)] * inv;
                            }
                        }
                    }
                    out[idx(oc, oy, ox, h, w)] = sum;
                }
            }
        }
    }

    /// 3×3, pad=1 fast-path forward. The top/bottom rows and left/right columns
    /// (the only places a 3×3 tap can fall outside the board) are handled by the
    /// general path; the interior `(1..h-1, 1..w-1)` runs a branch-free dot over
    /// three contiguous 3-wide weight rows against three contiguous 3-wide input
    /// rows per input channel.
    #[inline]
    fn forward_into_k3(&self, input: &[f64], h: usize, w: usize, out: &mut [f64]) {
        let in_ch = self.in_ch;
        // Border cells need padding; the interior never does. If the board is too
        // small to have an interior, the general path covers everything.
        if h < 3 || w < 3 {
            self.forward_into_general(input, h, w, out);
            return;
        }
        let hw = h * w;
        for oc in 0..self.out_ch {
            let b = self.bias[oc];
            // Border ring: top row, bottom row, left col, right col.
            for ox in 0..w {
                out[idx(oc, 0, ox, h, w)] = self.conv_cell_k3(input, oc, 0, ox, h, w, b);
                out[idx(oc, h - 1, ox, h, w)] = self.conv_cell_k3(input, oc, h - 1, ox, h, w, b);
            }
            for oy in 1..h - 1 {
                out[idx(oc, oy, 0, h, w)] = self.conv_cell_k3(input, oc, oy, 0, h, w, b);
                out[idx(oc, oy, w - 1, h, w)] = self.conv_cell_k3(input, oc, oy, w - 1, h, w, b);
            }
            // Interior: no bounds checks. weight block for (oc,ic) is 9 contiguous
            // f64 laid out [ky*3 + kx]; input rows for the 3×3 window are contiguous.
            //
            // SUMMATION ORDER is preserved bit-for-bit vs the original
            // `sum += iw0*in0 + iw1*in1 + iw2*in2` (per row, then per ic): the
            // reduction stays serial in `sum`, so the optimized binary loads the
            // LIVE checkpoint and evaluates identically. The speedup comes purely
            // from removing per-iteration bounds checks / redundant index math via
            // `get_unchecked` — all indices are provably in range (interior cells
            // with h,w ≥ 3 and a contiguous [oc][ic][9] weight block).
            let w_oc_base = oc * in_ch * 9;
            let out_oc_base = in_ch_out(oc, hw);
            for oy in 1..h - 1 {
                let row_base = oy * w; // (0, oy, 0) within a channel
                let rm = row_base - w; // row above, col 0
                let rp = row_base + w; // row below, col 0
                for ox in 1..w - 1 {
                    let mut sum = b;
                    let c0 = ox - 1; // leftmost input column of the window
                    let mut wb = w_oc_base;
                    let mut in_ch_base = 0usize;
                    // SAFETY: c0 ∈ [0, w-3], rows rm/row_base/rp are the 3 valid
                    // neighbouring rows of an interior cell, and per ic the window
                    // (r{0,1,2}+{0,1,2}) plus the 9-wide weight block at `wb` are all
                    // within `input` / `self.weights`. `out` write is the cell itself.
                    unsafe {
                        for _ic in 0..in_ch {
                            let r0 = in_ch_base + rm + c0;
                            let r1 = in_ch_base + row_base + c0;
                            let r2 = in_ch_base + rp + c0;
                            let iw = self.weights.get_unchecked(wb..wb + 9);
                            sum += iw[0] * *input.get_unchecked(r0)
                                + iw[1] * *input.get_unchecked(r0 + 1)
                                + iw[2] * *input.get_unchecked(r0 + 2);
                            sum += iw[3] * *input.get_unchecked(r1)
                                + iw[4] * *input.get_unchecked(r1 + 1)
                                + iw[5] * *input.get_unchecked(r1 + 2);
                            sum += iw[6] * *input.get_unchecked(r2)
                                + iw[7] * *input.get_unchecked(r2 + 1)
                                + iw[8] * *input.get_unchecked(r2 + 2);
                            wb += 9;
                            in_ch_base += hw;
                        }
                        *out.get_unchecked_mut(out_oc_base + row_base + ox) = sum;
                    }
                }
            }
        }
    }

    /// One 3×3, pad=1 output cell with per-tap bounds checks (used only for the
    /// border ring of the k=3 fast path).
    #[inline]
    fn conv_cell_k3(&self, input: &[f64], oc: usize, oy: usize, ox: usize, h: usize, w: usize, b: f64) -> f64 {
        let mut sum = b;
        for ic in 0..self.in_ch {
            for ky in 0..3usize {
                let iy = oy as isize + ky as isize - 1;
                if iy < 0 || iy >= h as isize {
                    continue;
                }
                for kx in 0..3usize {
                    let ix = ox as isize + kx as isize - 1;
                    if ix < 0 || ix >= w as isize {
                        continue;
                    }
                    let inv = input[idx(ic, iy as usize, ix as usize, h, w)];
                    sum += self.weights[self.w_idx(oc, ic, ky, kx)] * inv;
                }
            }
        }
        sum
    }

    /// Backward pass. Returns `(grad_input [in_ch*h*w], grad_weights [same layout
    /// as weights], grad_bias [out_ch])`.
    pub fn backward(
        &self,
        input: &[f64],
        grad_out: &[f64],
        h: usize,
        w: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut grad_input = vec![0.0f64; self.in_ch * h * w];
        let mut grad_weights = vec![0.0f64; self.weights.len()];
        let mut grad_bias = vec![0.0f64; self.out_ch];
        if self.k == 3 && self.pad == 1 && h >= 3 && w >= 3 {
            self.backward_k3(input, grad_out, h, w, &mut grad_input, &mut grad_weights, &mut grad_bias);
        } else {
            self.backward_general(input, grad_out, h, w, &mut grad_input, &mut grad_weights, &mut grad_bias);
        }
        (grad_input, grad_weights, grad_bias)
    }

    /// General bounds-checked backward (any kernel size). Reference path.
    #[allow(clippy::too_many_arguments)]
    fn backward_general(
        &self,
        input: &[f64],
        grad_out: &[f64],
        h: usize,
        w: usize,
        grad_input: &mut [f64],
        grad_weights: &mut [f64],
        grad_bias: &mut [f64],
    ) {
        let k = self.k;
        let pad = self.pad as isize;
        for oc in 0..self.out_ch {
            for oy in 0..h {
                for ox in 0..w {
                    let go = grad_out[idx(oc, oy, ox, h, w)];
                    grad_bias[oc] += go;
                    for ic in 0..self.in_ch {
                        for ky in 0..k {
                            let iy = oy as isize + ky as isize - pad;
                            if iy < 0 || iy >= h as isize {
                                continue;
                            }
                            for kx in 0..k {
                                let ix = ox as isize + kx as isize - pad;
                                if ix < 0 || ix >= w as isize {
                                    continue;
                                }
                                let in_pos = idx(ic, iy as usize, ix as usize, h, w);
                                let wpos = self.w_idx(oc, ic, ky, kx);
                                grad_weights[wpos] += go * input[in_pos];
                                grad_input[in_pos] += go * self.weights[wpos];
                            }
                        }
                    }
                }
            }
        }
    }

    /// 3×3, pad=1 fast-path backward. Mirrors [`forward_into_k3`]: the border ring
    /// uses the bounds-checked per-cell helper; the interior accumulates both the
    /// weight-grad and the input-grad over contiguous 3-wide slices with no
    /// per-tap branch.
    #[allow(clippy::too_many_arguments)]
    fn backward_k3(
        &self,
        input: &[f64],
        grad_out: &[f64],
        h: usize,
        w: usize,
        grad_input: &mut [f64],
        grad_weights: &mut [f64],
        grad_bias: &mut [f64],
    ) {
        let in_ch = self.in_ch;
        let hw = h * w;
        for oc in 0..self.out_ch {
            // Border ring (bounds-checked, also accumulates bias for those cells).
            for ox in 0..w {
                self.backward_cell_k3(input, grad_out, oc, 0, ox, h, w, grad_input, grad_weights, grad_bias);
                self.backward_cell_k3(input, grad_out, oc, h - 1, ox, h, w, grad_input, grad_weights, grad_bias);
            }
            for oy in 1..h - 1 {
                self.backward_cell_k3(input, grad_out, oc, oy, 0, h, w, grad_input, grad_weights, grad_bias);
                self.backward_cell_k3(input, grad_out, oc, oy, w - 1, h, w, grad_input, grad_weights, grad_bias);
            }
            // Interior: branch-free. Accumulation order (and thus f64 rounding) is
            // preserved bit-for-bit vs the original; only the per-iteration bounds
            // checks / index math are removed via `get_unchecked`.
            let w_oc_base = oc * in_ch * 9;
            let out_oc_base = in_ch_out(oc, hw);
            for oy in 1..h - 1 {
                let row_base = oy * w;
                let rm = row_base - w;
                let rp = row_base + w;
                for ox in 1..w - 1 {
                    let go = grad_out[out_oc_base + row_base + ox];
                    grad_bias[oc] += go;
                    let c0 = ox - 1;
                    let mut wb = w_oc_base;
                    let mut in_ch_base = 0usize;
                    // SAFETY: identical in-range argument as the forward k3 interior;
                    // grad_weights/grad_input share `input`/`weights` shapes.
                    unsafe {
                        for _ic in 0..in_ch {
                            let r0 = in_ch_base + rm + c0;
                            let r1 = in_ch_base + row_base + c0;
                            let r2 = in_ch_base + rp + c0;
                            let i0 = *input.get_unchecked(r0);
                            let i1 = *input.get_unchecked(r0 + 1);
                            let i2 = *input.get_unchecked(r0 + 2);
                            let i3 = *input.get_unchecked(r1);
                            let i4 = *input.get_unchecked(r1 + 1);
                            let i5 = *input.get_unchecked(r1 + 2);
                            let i6 = *input.get_unchecked(r2);
                            let i7 = *input.get_unchecked(r2 + 1);
                            let i8 = *input.get_unchecked(r2 + 2);
                            let gw = grad_weights.get_unchecked_mut(wb..wb + 9);
                            gw[0] += go * i0;
                            gw[1] += go * i1;
                            gw[2] += go * i2;
                            gw[3] += go * i3;
                            gw[4] += go * i4;
                            gw[5] += go * i5;
                            gw[6] += go * i6;
                            gw[7] += go * i7;
                            gw[8] += go * i8;
                            let iw = self.weights.get_unchecked(wb..wb + 9);
                            *grad_input.get_unchecked_mut(r0) += go * iw[0];
                            *grad_input.get_unchecked_mut(r0 + 1) += go * iw[1];
                            *grad_input.get_unchecked_mut(r0 + 2) += go * iw[2];
                            *grad_input.get_unchecked_mut(r1) += go * iw[3];
                            *grad_input.get_unchecked_mut(r1 + 1) += go * iw[4];
                            *grad_input.get_unchecked_mut(r1 + 2) += go * iw[5];
                            *grad_input.get_unchecked_mut(r2) += go * iw[6];
                            *grad_input.get_unchecked_mut(r2 + 1) += go * iw[7];
                            *grad_input.get_unchecked_mut(r2 + 2) += go * iw[8];
                            wb += 9;
                            in_ch_base += hw;
                        }
                    }
                }
            }
        }
    }

    /// Backward of one 3×3, pad=1 output cell with per-tap bounds checks (border
    /// ring of the k=3 fast path). Accumulates bias, weight-grad and input-grad.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn backward_cell_k3(
        &self,
        input: &[f64],
        grad_out: &[f64],
        oc: usize,
        oy: usize,
        ox: usize,
        h: usize,
        w: usize,
        grad_input: &mut [f64],
        grad_weights: &mut [f64],
        grad_bias: &mut [f64],
    ) {
        let go = grad_out[idx(oc, oy, ox, h, w)];
        grad_bias[oc] += go;
        for ic in 0..self.in_ch {
            for ky in 0..3usize {
                let iy = oy as isize + ky as isize - 1;
                if iy < 0 || iy >= h as isize {
                    continue;
                }
                for kx in 0..3usize {
                    let ix = ox as isize + kx as isize - 1;
                    if ix < 0 || ix >= w as isize {
                        continue;
                    }
                    let in_pos = idx(ic, iy as usize, ix as usize, h, w);
                    let wpos = self.w_idx(oc, ic, ky, kx);
                    grad_weights[wpos] += go * input[in_pos];
                    grad_input[in_pos] += go * self.weights[wpos];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// tanh (free functions).
// ---------------------------------------------------------------------------

/// Element-wise tanh.
pub fn tanh_forward(input: &[f64]) -> Vec<f64> {
    input.iter().map(|&v| v.tanh()).collect()
}

/// Element-wise tanh into a reusable buffer (inference hot path). `out` is
/// resized to `input.len()` and fully overwritten. Numerically identical to
/// [`tanh_forward`].
pub fn tanh_forward_into(input: &[f64], out: &mut Vec<f64>) {
    out.clear();
    out.extend(input.iter().map(|&v| v.tanh()));
}

/// Backward through tanh: `grad_in = grad_out * (1 - out^2)`, where `out` is the
/// tanh forward output.
pub fn tanh_backward(out: &[f64], grad_out: &[f64]) -> Vec<f64> {
    out.iter()
        .zip(grad_out.iter())
        .map(|(&o, &g)| g * (1.0 - o * o))
        .collect()
}

// ---------------------------------------------------------------------------
// Global average pool: (C,H,W) -> (C,), mean over H*W.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct GlobalAvgPool;

impl GlobalAvgPool {
    /// Mean over the H*W spatial extent of each channel. Output len `c`.
    pub fn forward(&self, input: &[f64], c: usize, h: usize, w: usize) -> Vec<f64> {
        let area = (h * w) as f64;
        let mut out = vec![0.0f64; c];
        for ch in 0..c {
            let mut sum = 0.0;
            for y in 0..h {
                for x in 0..w {
                    sum += input[idx(ch, y, x, h, w)];
                }
            }
            out[ch] = sum / area;
        }
        out
    }

    /// Backward: each input cell of channel `ch` gets `grad_out[ch] / (H*W)`.
    pub fn backward(&self, grad_out: &[f64], c: usize, h: usize, w: usize) -> Vec<f64> {
        let area = (h * w) as f64;
        let mut grad_in = vec![0.0f64; c * h * w];
        for ch in 0..c {
            let g = grad_out[ch] / area;
            for y in 0..h {
                for x in 0..w {
                    grad_in[idx(ch, y, x, h, w)] = g;
                }
            }
        }
        grad_in
    }
}

// ---------------------------------------------------------------------------
// Dense (fully-connected). weights layout [o][i]; bias [o].
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dense {
    pub in_dim: usize,
    pub out_dim: usize,
    /// `out_dim * in_dim`, layout `[o][i]`.
    pub weights: Vec<f64>,
    /// `out_dim`.
    pub bias: Vec<f64>,
}

impl Dense {
    /// Deterministic small-random init; weights uniform in `[-1,1) * sqrt(1/in)`,
    /// biases zero.
    pub fn new_seeded(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let mut rng = Lcg::new(seed);
        let scale = (1.0 / in_dim.max(1) as f64).sqrt();
        let weights = (0..out_dim * in_dim)
            .map(|_| rng.next_signed() * scale)
            .collect();
        let bias = vec![0.0; out_dim];
        Dense { in_dim, out_dim, weights, bias }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0f64; self.out_dim];
        self.forward_into(input, &mut out);
        out
    }

    /// Allocation-free forward used by the inference hot path. Writes `out_dim`
    /// values into `out` (resized + fully overwritten). Numerically identical to
    /// [`forward`](Self::forward).
    pub fn forward_into(&self, input: &[f64], out: &mut Vec<f64>) {
        out.clear();
        out.resize(self.out_dim, 0.0);
        for o in 0..self.out_dim {
            let mut sum = self.bias[o];
            let base = o * self.in_dim;
            for i in 0..self.in_dim {
                sum += self.weights[base + i] * input[i];
            }
            out[o] = sum;
        }
    }

    /// Returns `(grad_input [in_dim], grad_weights [out*in], grad_bias [out])`.
    pub fn backward(&self, input: &[f64], grad_out: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut grad_input = vec![0.0f64; self.in_dim];
        let mut grad_weights = vec![0.0f64; self.weights.len()];
        let mut grad_bias = vec![0.0f64; self.out_dim];
        for o in 0..self.out_dim {
            let go = grad_out[o];
            grad_bias[o] = go;
            let base = o * self.in_dim;
            for i in 0..self.in_dim {
                grad_weights[base + i] = go * input[i];
                grad_input[i] += go * self.weights[base + i];
            }
        }
        (grad_input, grad_weights, grad_bias)
    }
}

// ---------------------------------------------------------------------------
// Tests: finite-difference gradient checks.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-5;
    const TOL: f64 = 1e-4;

    /// Deterministic pseudo-random fill in [-1,1) for test inputs/coeffs.
    fn fill(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Lcg::new(seed);
        (0..n).map(|_| rng.next_signed()).collect()
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    fn assert_close(a: f64, b: f64, what: &str) {
        assert!(
            (a - b).abs() < TOL,
            "{what}: analytic {a} vs numeric {b} (diff {})",
            (a - b).abs()
        );
    }

    // ---- Conv2d ----------------------------------------------------------

    #[test]
    fn conv_grad_input_weights_bias() {
        let (in_ch, out_ch, k, pad, h, w) = (2usize, 3usize, 3usize, 1usize, 4usize, 5usize);
        let mut conv = Conv2d::new_seeded(in_ch, out_ch, k, pad, 42);
        let input = fill(in_ch * h * w, 7);
        // Downstream scalar loss = sum(out * coeffs). So grad_out == coeffs.
        let coeffs = fill(out_ch * h * w, 99);
        let loss = |out: &[f64]| dot(out, &coeffs);

        let out = conv.forward(&input, h, w);
        let grad_out = coeffs.clone();
        let (gi, gw, gb) = conv.backward(&input, &grad_out, h, w);

        // grad w.r.t. input
        for j in 0..input.len() {
            let mut ip = input.clone();
            ip[j] += EPS;
            let lp = loss(&conv.forward(&ip, h, w));
            ip[j] -= 2.0 * EPS;
            let lm = loss(&conv.forward(&ip, h, w));
            assert_close(gi[j], (lp - lm) / (2.0 * EPS), "conv grad_input");
        }
        let _ = out;

        // grad w.r.t. weights
        for j in 0..conv.weights.len() {
            let save = conv.weights[j];
            conv.weights[j] = save + EPS;
            let lp = loss(&conv.forward(&input, h, w));
            conv.weights[j] = save - EPS;
            let lm = loss(&conv.forward(&input, h, w));
            conv.weights[j] = save;
            assert_close(gw[j], (lp - lm) / (2.0 * EPS), "conv grad_weights");
        }

        // grad w.r.t. bias
        for j in 0..conv.bias.len() {
            let save = conv.bias[j];
            conv.bias[j] = save + EPS;
            let lp = loss(&conv.forward(&input, h, w));
            conv.bias[j] = save - EPS;
            let lm = loss(&conv.forward(&input, h, w));
            conv.bias[j] = save;
            assert_close(gb[j], (lp - lm) / (2.0 * EPS), "conv grad_bias");
        }
    }

    // ---- Conv micro-benchmark (timing, ignored) --------------------------
    //
    //   cargo test -p cp-ai --release -- --ignored --nocapture conv_bench
    #[test]
    #[ignore]
    fn conv_bench() {
        use std::time::Instant;
        let (h, w) = (14usize, 12usize);
        let c1 = Conv2d::new_seeded(15, 16, 3, 1, 1);
        let c2 = Conv2d::new_seeded(16, 24, 3, 1, 2);
        let in1 = fill(15 * h * w, 7);
        let mid = fill(16 * h * w, 9);
        let n = 50_000usize;
        let mut out = Vec::new();
        let mut acc = 0.0;
        // forward
        let t = Instant::now();
        for _ in 0..n {
            c1.forward_into(&in1, h, w, &mut out);
            acc += out[0];
            c2.forward_into(&mid, h, w, &mut out);
            acc += out[0];
        }
        let fwd = t.elapsed();
        // backward (use grad_out = the conv output as a stand-in upstream grad)
        let go1 = c1.forward(&in1, h, w);
        let go2 = c2.forward(&mid, h, w);
        let t = Instant::now();
        for _ in 0..n {
            let (gi, _, _) = c1.backward(&in1, &go1, h, w);
            acc += gi[0];
            let (gi, _, _) = c2.backward(&mid, &go2, h, w);
            acc += gi[0];
        }
        let bwd = t.elapsed();
        eprintln!(
            "conv_bench N={n}: forward {fwd:?} ({:.0} ns/pair)  backward {bwd:?} ({:.0} ns/pair)  acc={acc:.1}",
            fwd.as_nanos() as f64 / n as f64,
            bwd.as_nanos() as f64 / n as f64
        );
    }

    // ---- Dense -----------------------------------------------------------

    #[test]
    fn dense_grad_input_weights_bias() {
        let (in_dim, out_dim) = (5usize, 4usize);
        let mut d = Dense::new_seeded(in_dim, out_dim, 11);
        let input = fill(in_dim, 3);
        let coeffs = fill(out_dim, 5);
        let loss = |out: &[f64]| dot(out, &coeffs);

        let grad_out = coeffs.clone();
        let (gi, gw, gb) = d.backward(&input, &grad_out);

        for j in 0..in_dim {
            let mut ip = input.clone();
            ip[j] += EPS;
            let lp = loss(&d.forward(&ip));
            ip[j] -= 2.0 * EPS;
            let lm = loss(&d.forward(&ip));
            assert_close(gi[j], (lp - lm) / (2.0 * EPS), "dense grad_input");
        }
        for j in 0..d.weights.len() {
            let save = d.weights[j];
            d.weights[j] = save + EPS;
            let lp = loss(&d.forward(&input));
            d.weights[j] = save - EPS;
            let lm = loss(&d.forward(&input));
            d.weights[j] = save;
            assert_close(gw[j], (lp - lm) / (2.0 * EPS), "dense grad_weights");
        }
        for j in 0..d.bias.len() {
            let save = d.bias[j];
            d.bias[j] = save + EPS;
            let lp = loss(&d.forward(&input));
            d.bias[j] = save - EPS;
            let lm = loss(&d.forward(&input));
            d.bias[j] = save;
            assert_close(gb[j], (lp - lm) / (2.0 * EPS), "dense grad_bias");
        }
    }

    // ---- GlobalAvgPool ---------------------------------------------------

    #[test]
    fn gap_grad_input() {
        let (c, h, w) = (3usize, 4usize, 5usize);
        let pool = GlobalAvgPool;
        let input = fill(c * h * w, 21);
        let coeffs = fill(c, 8);
        let loss = |out: &[f64]| dot(out, &coeffs);

        let grad_out = coeffs.clone();
        let gi = pool.backward(&grad_out, c, h, w);

        for j in 0..input.len() {
            let mut ip = input.clone();
            ip[j] += EPS;
            let lp = loss(&pool.forward(&ip, c, h, w));
            ip[j] -= 2.0 * EPS;
            let lm = loss(&pool.forward(&ip, c, h, w));
            assert_close(gi[j], (lp - lm) / (2.0 * EPS), "gap grad_input");
        }
    }

    // ---- Composite stack: Conv2d -> tanh -> GlobalAvgPool -> Dense(->1) ---

    #[test]
    fn composite_full_chain() {
        let (in_ch, out_ch, k, pad, h, w) = (2usize, 4usize, 3usize, 1usize, 4usize, 5usize);
        let mut conv = Conv2d::new_seeded(in_ch, out_ch, k, pad, 123);
        let pool = GlobalAvgPool;
        let mut dense = Dense::new_seeded(out_ch, 1, 456);
        let input = fill(in_ch * h * w, 77);

        // Forward + scalar loss = the single dense output.
        let fwd = |conv: &Conv2d, dense: &Dense, input: &[f64]| -> f64 {
            let a = conv.forward(input, h, w);
            let t = tanh_forward(&a);
            let p = pool.forward(&t, out_ch, h, w);
            dense.forward(&p)[0]
        };

        // Analytic backward chain.
        let a = conv.forward(&input, h, w);
        let t = tanh_forward(&a);
        let p = pool.forward(&t, out_ch, h, w);
        let _y = dense.forward(&p)[0];

        let grad_y = vec![1.0f64];
        let (grad_p, dense_gw, dense_gb) = dense.backward(&p, &grad_y);
        let grad_t = pool.backward(&grad_p, out_ch, h, w);
        let grad_a = tanh_backward(&t, &grad_t);
        let (grad_in, conv_gw, conv_gb) = conv.backward(&input, &grad_a, h, w);

        // Check grad w.r.t. conv input.
        for j in 0..input.len() {
            let mut ip = input.clone();
            ip[j] += EPS;
            let lp = fwd(&conv, &dense, &ip);
            ip[j] -= 2.0 * EPS;
            let lm = fwd(&conv, &dense, &ip);
            assert_close(grad_in[j], (lp - lm) / (2.0 * EPS), "composite grad_input");
        }
        // conv weights
        for j in 0..conv.weights.len() {
            let save = conv.weights[j];
            conv.weights[j] = save + EPS;
            let lp = fwd(&conv, &dense, &input);
            conv.weights[j] = save - EPS;
            let lm = fwd(&conv, &dense, &input);
            conv.weights[j] = save;
            assert_close(conv_gw[j], (lp - lm) / (2.0 * EPS), "composite conv_weights");
        }
        // conv bias
        for j in 0..conv.bias.len() {
            let save = conv.bias[j];
            conv.bias[j] = save + EPS;
            let lp = fwd(&conv, &dense, &input);
            conv.bias[j] = save - EPS;
            let lm = fwd(&conv, &dense, &input);
            conv.bias[j] = save;
            assert_close(conv_gb[j], (lp - lm) / (2.0 * EPS), "composite conv_bias");
        }
        // dense weights
        for j in 0..dense.weights.len() {
            let save = dense.weights[j];
            dense.weights[j] = save + EPS;
            let lp = fwd(&conv, &dense, &input);
            dense.weights[j] = save - EPS;
            let lm = fwd(&conv, &dense, &input);
            dense.weights[j] = save;
            assert_close(dense_gw[j], (lp - lm) / (2.0 * EPS), "composite dense_weights");
        }
        // dense bias
        for j in 0..dense.bias.len() {
            let save = dense.bias[j];
            dense.bias[j] = save + EPS;
            let lp = fwd(&conv, &dense, &input);
            dense.bias[j] = save - EPS;
            let lm = fwd(&conv, &dense, &input);
            dense.bias[j] = save;
            assert_close(dense_gb[j], (lp - lm) / (2.0 * EPS), "composite dense_bias");
        }
    }
}
