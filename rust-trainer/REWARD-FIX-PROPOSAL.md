# REWARD-FIX-PROPOSAL — diagnosing the Pass+wait equilibrium

_Authored 2026-06-05. Premise: every Φ-only fix attempted on top of `potential_step1`
(`--w-army`, `--cap-potential`, `--idle-flow-penalty` retune in i1) failed to move
`trueWinVsHard` off ~0.40. This memo argues the binding constraint is no longer the
Φ surface at all — it is the **terminal-signal distribution the net is fitting** —
and proposes the one minimal change that follows from that diagnosis._

---

## 1. Why Pass+wait wins 40%

The pure-Pass-after-r40 policy is sustained by **three coincident terminal-signal
sources**, all directly readable in `checkpoints-cnn-{b1,i1}/benchmark-history.jsonl`:

1. **Free wins by HARD bankruptcy.** Across b1/i1 last 5 benches, `champWins.bankruptcy ≈
   5–9/60` (15–30% of all wins). After the turn that triggers HARD's bankruptcy, the
   forward model awards a +1 terminal z to the still-living seat regardless of what it
   did. With `bench-games=60` and tie_penalty=0.4, an all-Pass trajectory's expected
   terminal z ≈ 0.15·(+1) + 0.55·(−1) + 0.30·(−0.4) ≈ −0.52. Not great, but **finite and
   non-suicidal** — and most importantly, the variance of an *active* trajectory under
   the current net's still-tiny army is WORSE: when the champ does field one soldier and
   commits it, the loss rate is ~0.48–0.53 because HARD's army-rush opponent and HARD's
   own Device line (`hardWins.device = 15–21/60`) crush a 1-soldier offence.
2. **Conquest "wins" that are bankruptcy-mirages in flight.** `champWins.conquest ≈
   18–20/60` looks healthy, but `maxSoldiersPerGame ≈ 0.6–0.78` means the *average* champ
   game peaks at <1 soldier. Reading those replays: the conquest event fires the turn
   HARD's last soldier is removed by attrition (HARD soldiers cost upkeep) or HARD's
   build-line stalls — the champ's 1 soldier walks onto a defender-zero HQ-adjacent tile
   per §3 conquest resolution. These wins are **structurally indistinguishable from
   bankruptcy wins** from the policy's standpoint: do-nothing-while-economy-ticks works
   in 30–40% of seeds. The honest skill rate is closer to `trueWinVsHard − P(HARD self-
   stalls) ≈ 0.40 − 0.25 = ~0.15`.
3. **Φ is locally Pass-neutral after r40.** Walking `potential_step1`
   (`cnn_train.rs:2629–2745`) on a champ at r40 with ~12 tiles, ~5 staffed farms, 1
   soldier, no Outpost: `inc≈1.0, staffed≈1.0, cap≈0.6, bank≈0.25` → econ core ≈ 0.91.
   Tile_lead saturates against a turtle; income_lead ≈ 0 once HARD plateaus; cap_potential
   stuck at 1/7 ≈ 0.14; w_army at 0.14. **Φ(s) for Pass-many-turns ≈ Φ(s) for build-then-
   Pass-many-turns within ±0.05** — and the SINGLE LARGEST term that makes Pass attractive
   is **not idle-flow** (refuted by i1) but the **terminal-side bankruptcy signal**, which
   shaping is *forbidden* to fight by Ng-1999 policy-invariance. Φ can shift the
   trajectory-integral by ε but cannot change which terminal is preferred. The terminal
   signal currently *literally rewards* Pass+wait at 15–30% rate. **No Φ term can fix
   that** — and that is why s3/i1/b1 all converged to the same trueWinVsHard.

The single largest term making Pass attractive is therefore not in Φ at all: it is the
**+1 terminal z paid out on HARD's self-bankruptcy** entering through the value-target
path at `cnn_train.rs:3152–3158`. Every passive trajectory that survives to round ~90
collects this with probability ~0.25 — a free coupon Φ cannot tax.

## 2. Mechanical implications the net should see but doesn't

- **§5 (cap):** "HQ only ⇒ soldier cap = 1." `cap_potential` *can* express this (the
  /7 saturator at `cnn_train.rs:2680`). But notice: `cap_potential·clamp(1/7) ≈ 0.04` and
  `cap_potential·clamp(4/7) ≈ 0.17` — the gap between "no Outpost" and "1 Outpost" is
  ~0.13 of one Φ term — easily drowned by tile_lead and income_lead noise.
- **§4 (threat):** *Pass under enemy frontier with enemy mobile-soldier-budget ≥ 2 = die.*
  But HARD's army-rush against a no-cap champ destroys tiles at ~1–2/turn → champ Φ falls
  via tile_lead. The shaping signal IS there. It just isn't strong enough to outweigh the
  bankruptcy coupon at terminal.
- **§6 (Device):** the Device tile holds zero defenders → 1 attacker cracks it. With
  `cap=1` the champ *has* exactly the soldier needed. But `--device-credit` only fires on
  the **building** seat's BuildStrangeDevice intent (`cnn_train.rs:3206`) — there is **no
  symmetric credit for staging a soldier on an enemy Device tile**, which is the highest-
  ROI single action in the whole game. The deviceDenialRate sits at 0.16–0.30; this is
  the only place a small action-credit is unambiguously justified by the mechanics.
- **§9 (turn loop):** `controller.rs:271–272` — **Intent::Pass breaks the loop.** So
  "100% Pass after r40" means the agent makes ZERO actions per turn after r40 (not "one
  thing then Pass"). `--turn-search-spend` exists but is policy-side, not target-side —
  the value head never gets a target that says "Pass-only-turn was −1 because you owned a
  standing-army opportunity." The policy is collapsing onto the *correct* argmax for the
  *broken* terminal-signal distribution.

## 3. The proposal — strip the bankruptcy coupon from the value target

**ONE change. The minimal Φ-side intervention that addresses §1's root cause directly:**
a terminal-z **down-weighting of opponent-bankruptcy wins**, applied in the value-target
pipeline at `cnn_train.rs:3151–3158` (the `terminal_z` closure inside
`play_one_game_explore`). This is parity-free (no game-rule mod, no plane change), no-op
at weight 0, and surgical: it changes only the value head's learning target, not Φ, not
MCTS, not the rules.

**Exact formula.** Add a new flag `--bankruptcy-discount d`, `d ∈ [0,1]`, default `0.0`
(EXACT no-op when not set). Modify `terminal_z` so:

```rust
// cnn_train.rs:3152-3158, in play_one_game_explore
let terminal_z = |seat: PlayerId| -> f64 {
    match winner_pid {
        Some(w) if w == seat => {
            // §3 — strip the bankruptcy coupon: if the OPPONENT lost by
            // self-bankruptcy and the winner did NOT field a real army or
            // engage in combat, discount the +z toward the tie line. This
            // teaches "free wins do not generalize" without altering the
            // game (Φ-shaping invariance unaffected; this is the terminal,
            // not a potential).
            let opp_bankrupt = matches!(g.last_win_cause(), Some(WinCause::Bankruptcy));
            let combat_engaged = examples.iter()
                .any(|e| e.seat == seat &&
                     (e.chosen_intent == candidates::Intent::Attack
                      || e.chosen_intent == candidates::Intent::HireSoldier
                      || e.chosen_intent == candidates::Intent::BuildOutpost));
            if opp_bankrupt && !combat_engaged {
                mag * (1.0 - tc.bankruptcy_discount)
            } else {
                mag
            }
        }
        Some(_) => -mag,
        None => -tc.tie_penalty,
    }
};
```

**File + function.** `crates/cp-train/src/bin/cnn_train.rs`, inside
`play_one_game_explore` at line 3152 (the existing `terminal_z` closure). The
`bankruptcy_discount` field is added to `TrainCfg` near `tie_penalty` (~line 839), CLI-
parsed alongside `--tie-penalty` (~line 4983).

**Flag.** `--bankruptcy-discount d` (`d` in [0,1]). Default `0.0` = bit-identical to
today. At `d=1.0` a passive-bankruptcy win pays z=0 (the tie line). Recommended starting
point: `d=0.7`.

**No-op test.** `bankruptcy_discount_zero_is_terminal_only_noop`: build the same fixture
as `shape_weight_zero_is_terminal_only_noop` (`cnn_train.rs:5381`), force a bankruptcy
end, assert `examples[i].z` is identical to the prior pipeline (mag/-mag/-tie_penalty)
when the flag is 0. Mirror at d=1.0: the winning seat's z is 0 when no combat-engaged
example exists; `mag` when one does.

**Gate (decides if it worked).** Over a 30-iter run with the same b1 launch + only
`--bankruptcy-discount 0.7` added:
- `champWins.bankruptcy` falls (the net stops *seeking* the path because the reward
  shrunk), AND `champWins.conquest` is sustained-by-army (concurrent: `maxSoldiersPerGame`
  > 1.5 trend, `intents.BuildOutpost` > 5/bench);
- `trueWinVsHard` is unchanged or up — falling here is the kill signal (we destroyed a
  real source of signal). If trueWin drops but `maxSoldiersPerGame` rises, the lever is
  on the right axis and the magnitude is too aggressive — re-run with d=0.4.

**Lower-priority alternatives.**

1. **Symmetric device-crack action credit** (REWARD-DESIGN P10 missing-mirror).
   `--device-crack-credit c`: in `cnn_train.rs:3194` after the existing device_credit
   loop, add credit on any seat's `Attack` decision whose target is an enemy-owned Device
   tile (free from §6 because the tile is defender-zero). Justified solely because §6
   says this single action averts a guaranteed loss — exactly the diffuse-payoff
   condition `--device-credit` is also justified by. Combines with primary at low weight.
2. **Pass-fraction terminal penalty.** Compute per-game `pass_share` of the winning
   seat's decisions; subtract `w_pass · pass_share` from that seat's terminal z (clamped).
   Cheaper to implement than (1) but blunter, and risks suppressing genuine "no good
   move" Passes mid-game. Use only if the primary fails its gate AND (1) doesn't move
   the needle.

## 4. Skeptic's check

**(a) Ng-1999 invariance — won't this just shrink without redirecting?** The proposal is
*not* a potential — it modifies the terminal z. Ng-1999 protects the policy's optimum
under γΦ(s')−Φ(s) shaping; it does NOT protect against changes to the *terminal-reward
distribution*, which is exactly what the bankruptcy coupon is. We are changing the
underlying MDP's reward function for one specific terminal class, deliberately. The new
optimum is provably different from the current one (the current optimum exploits the
coupon; the new one cannot). The risk this is wrong is the risk the bankruptcy coupon
was a *necessary scaffold* keeping the value head un-collapsed — possible (`az-pass-
collapse-fix` showed the value head collapses to 0 under symmetric draws), but the gate's
behavioural metrics (`maxSoldiersPerGame`, `intents.BuildOutpost`) will catch this
collapse before win-rate does, and we re-enable at d=0.

**(b) New local optimum: "lose by Device because I have no easy +1 to chase".** Possible.
If the net learns "bankruptcy-wins are now z≈0, so I should LOSE less" instead of "I
should WIN harder," it collapses harder into draw-seeking (the `az-draw-attractor`
failure). The `combat_engaged` qualifier guards against this: it discounts ONLY the wins
where the seat truly did nothing — a 1-soldier-army-but-HARD-bankrupted game keeps the
full +z, so the net is positively rewarded for the active line even if HARD self-
destructs anyway. Without that qualifier this objection sinks the proposal.

**(c) Collision with existing Φ terms.** `--cap-potential`, `--w-army`, `--idle-flow-
penalty`, `--device-credit` all push the policy toward "active." They were not biting
because the terminal coupon undermined them. Once the coupon is taxed, they finally have
something to grip — this is a **complementary** intervention, not a colliding one. The
only collision risk is with `--device-credit`: a game where the net builds a Device
and HARD goes bankrupt before countdown would now pay LESS than before (`device_decided=
false`, opp_bankrupt=true, combat_engaged unless an Outpost was built — usually true
because Device requires Outpost gate). This is *correct*: a Device-builder who never
needed the Device shouldn't be over-credited.

This is the best bet because every other lever was already tried at saturating weights
and the terminal-signal corruption is the residual that all of them collide against.
