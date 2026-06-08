# PPO + GAE(λ) implementation spec (for cp-train AlphaZero trainer)

> The highest-EV lever to break the ~0.55-0.60 trueWin plateau. Root cause = LONG-HORIZON
> CREDIT ASSIGNMENT (delayed Mine→staff→Outpost→cap→army payoff that ~30-turn MCTS can't
> credit; reward-shaping is Ng-1999-bounded = dead end). PPO+GAE is parity-FREE (training-only;
> forward inference unchanged). Designed around the lever24 collapse lesson: CONSERVATIVE data
> mix + KL trust region or it collapses like the warm-start did (0.56→0.11).

## Files
- Trainer: `rust-trainer/crates/cp-train/src/bin/cnn_train.rs`
- Net: `rust-trainer/crates/cp-ai/src/spatial_net.rs` (SpatialNet is AZ-only/additive — NOT in the parity path; new training fns are parity-free)
- Candidates: `rust-trainer/crates/cp-ai/src/candidates.rs` (no edits; `enumerate`/`execute_action`/`Intent`)

## Reused API (unchanged forward path)
- `forward_board_scalars(planes,h,w,value_scalars)->BoardCache` (spatial_net.rs:286)
- `score_candidate_into(cache,tgt,local,intent,&mut PolicyScratch)->f64` (spatial_net.rs:388) — softmax of these over the enumerated candidates = π(a|s)
- `value_from(&cache)->f64` (spatial_net.rs:335)
- `SpatialGrad` (.add/.scale/.zeros_like), `apply_grad(&grad,lr,l2)` (spatial_net.rs:887)
- `CandFeat=(Option<(usize,usize)>,Vec<f64>,Vec<f64>)` (cnn_train.rs:92), `cand_feat(g,player,c)` (cnn_train.rs:374)
- `board_planes`, `value_scalars(g,seat)` (cnn_train.rs:118), `scaffold_ensure/staff/finalize`, `candidates::enumerate/execute_action`, `bench_vs_hard`/`league_bench`/`bench_net_greedy`, warm-start loader (cnn_train.rs:5224-5256), anchor loader (5302-5343), opponent-mix block (5403-5471), bench/checkpoint block (5765-6064).
- Model the new backward on `train_grad_cached_kl_inner` (spatial_net.rs:595) — it already shows custom-policy-upstream + standard value + standard trunk, AND the forward-KL term.

## 1. Buffer — `PpoStep` (cnn_train.rs near Example:231)
Fields: planes,h,w,value_scalars,cands:Vec<CandFeat>; chosen:usize; logp_old:f64; v_old:f64; reward:f64; seat; adv:f64; vtarg:f64; chosen_intent. `logp_old`/`v_old` captured from the FROZEN θ_old at collection (never recomputed in epochs).
- Termination: reuse play_one_game_explore loop (3289) + terminal_z (3514) incl tie_penalty.
- Per-step reward: terminal step = terminal_z(seat) ∈[-1,1]; non-terminal = 0.0. Optional Φ-difference shaping `--ppo-shape-weight` DEFAULT 0.0 (terminal-only; GAE propagates it).

## 2. GAE (cnn_train.rs near shaped_returns:3696), PER SEAT temporal order
delta_t = r_t + γ·V(s_{t+1}) − V(s_t)  (V at terminal = 0); A_t = delta_t + (γλ)·A_{t+1}; vtarg_t = A_t + V(s_t).
`fn compute_gae(rewards,values,gamma,lambda)->(adv,vtarg)` — pure, add a hand-computed unit test (mirror shaped_returns_match_hand_computation:9197).
Normalize advantages BATCH-WIDE once/iter: adv←(adv−mean)/(std+1e-8). Do NOT normalize vtarg.
γ=0.997 (per-decision steps; 0.999 if --turn-search/per-turn), λ=0.95 (value head weak → don't go below 0.92).

## 3. PPO loss + gradient
Collection: forward once, scores=score_candidate_into per cand, p=softmax(scores), logp_old=ln p[chosen], v_old=value_from.
Objective (per step, minimize): L_clip = −min(r·A, clip(r,1−ε,1+ε)·A) where r=exp(logp_new−logp_old);
  L_ent = −ent_coef·H(p_new), H=−Σ p_c ln p_c; L_val = val_coef·(V_new−vtarg)^2.
New fn `train_grad_ppo_cached(cache,cands,chosen,logp_old,adv,vtarg,v_old,clip_eps,ent_coef,val_coef,vclip)->(SpatialGrad,policy_loss,value_loss)` in spatial_net.rs (model on kl_inner:595; copy trunk backward 679-709 verbatim):
- Clip active (grad 0) iff (adv≥0 & r>1+ε) or (adv<0 & r<1−ε); else dL_clip/dlogp_new = −r·adv.
- ∂logp_new/∂s_c = [c==chosen] − p_c → g_c^policy = (dL_clip/dlogp_new)·([c==chosen]−p_c).
- Entropy: ∂L_ent/∂s_j = ent_coef·p_j·(ln p_j + H). Add to g_c.
- Per-candidate backward = the existing loop (spatial_net.rs:808-838): policy_d2.backward→tanh→policy_d1.backward→scatter into grad_board_embed + grad_global. EVERY candidate has nonzero upstream (the −p_c term).
- Value head: (V−vtarg)^2·val_coef; vclip DEFAULT 0 (off); d_value=2·val_coef·(V−vtarg)·(1−V^2).
- MANDATORY finite-diff grad-check test (both adv signs; r inside AND outside clip band — clipped branch must show ZERO policy grad).
Batch driver `train_batch_ppo` (near train_batch_lr:3788): par_iter map→train_grad_ppo_cached→reduce .add→.scale(1/n)→apply_grad(lr,l2). logp_new computed from CURRENT net inside. Return (surrogate_loss,value_loss,approx_kl).

## 4. Data collection — POLICY-HEAD SAMPLING, not MCTS
`play_one_game_ppo` (clone play_one_game_explore:3220, strip MCTS): each decision forward once, p=softmax(scores/temp), sample chosen∝p, record logp_old=ln(softmax(scores)[chosen]) (UN-tempered, τ=1), v_old. `--ppo-temp` 1.0 + entropy bonus for exploration. Keep MCTS strictly for deploy/bench.
Opponent mix: CONSERVATIVE — reuse run_train's block (5403-5471). Defaults `--vs-hard-frac 0.75 --script-opponents --script-frac 0.5 --pfsp` (do NOT strip vs-HARD / do NOT curriculum-first — that caused the collapse). Record seat-0 (learner) only.
BUFFER: FRESH on-policy each iter — collect --ppo-games → --ppo-epochs passes → DISCARD. Never carry across iters (stale logp_old).

## 5. Warm-start + trust region (collapse guards)
- Warm-start: reuse loader (5224-5256), --init sd3-az-003; hard-fail on dim mismatch (no cold-start).
- (1) PPO clip ε; (2) KL ANCHOR to a SECOND frozen net = warm-start (anchor loader 5302-5343): + kl_coef·KL(π_new‖π_anchor); the forward-KL grad already exists in kl_inner (the kl_grad_c term) — fold it in (anchor_pi per step like train_batch_lr_kl:3744). `--ppo-kl-anchor` 0.3. (3) KL EARLY-STOP per epoch: approx_kl≈mean((r−1)−ln r); if > `--ppo-target-kl` (0.02) break epoch loop.
- Anchor decay: after each bench, if true_win rose → kl_coef*=0.9 (floor 0.05); if dropped → restore coef (or revert to champion-best.json + halve lr). Keep target-KL on throughout.

## 6. Hyperparams (defaults)
clip 0.2 | ent_coef 0.01 (→0.02 if intents collapse to Pass) | val_coef 0.5 (≤1.0; trunk co-train risk) | vclip 0 | lr 3e-4 (much lower than AZ 0.01; SGD) | l2 1e-5 | batch 256 | ppo-epochs 4 | ppo-games 256 (~7-8k steps/iter, cheap no-MCTS) | γ 0.997 | λ 0.95 | target-kl 0.02 | kl-anchor 0.3 | temp 1.0 | shape-weight 0.0 | iters 200 | bench sims 64, bench-games 80, cap 300/150.

## 7. Integration — `--ppo` mode
New dispatch arm in main() parallel to --train (8216) → PpoCfg → run_ppo (clone run_train scaffolding 5217):
warm-start + anchor load → for iter: build seed/opp list (5403-5471) → par_iter play_one_game_ppo → per-seat compute_gae → pool + normalize adv → epoch loop (shuffle 5672, train_batch_ppo, approx-KL early-stop) → discard buffer → bench/checkpoint/log block (5765-6064, same JSON schema → dashboard works) + anchor decay + PFSP snapshot. Keep --train 100% intact. Bench via --validate-net --greedy (honest) + --sims 64 (deploy).

## 8. Risks/mitigations + GATE
- Ratio explosion → clip + target-KL stop + clamp(logp_new−logp_old,±20) before exp + entropy.
- Collapse (data shift) → conservative mix + KL anchor + lr 3e-4 + AUTO-REVERT (if a bench true_win drops >0.05 below best → reload champion-best.json + halve lr).
- Value co-train corrupting trunk → val_coef≤0.5; optional `--ppo-policy-only-warmup N` (value branch off first N iters).
- Adv normalization mandatory.
- PARITY: assert no cp-sim/*.ts/golden/controller.rs/inference touched; cargo test -p cp-ai + parity 8/8.
GATE (pass): on --validate-net --greedy AND --sims 64 (≥80 games vs HARD): trueWin>0.65 AND maxSoldiers≥1.5 AND outposts/game≥0.3 AND no Pass-collapse (Pass<60%, HireSoldier/Expand in top intents).
EARLY-ABORT (kill): trueWin<0.45 for 3 benches; OR policyEntropy→0 & Pass>80%; OR approx-KL slams target-KL at epoch 0 with no bench gain over 20 iters → THAT decisively reconfirms "economy unfundable" (not credit) → then sd3→sd4 economy rebalance is the lever.
