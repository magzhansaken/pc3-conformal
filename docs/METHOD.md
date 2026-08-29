# Method

PC³ (Physics-Constrained Conformal prediction for Composites) produces
prediction intervals that are (a) calibrated and (b) physically admissible —
contained in a deterministic corridor `[L(x), U(x)]` derived from micromechanics
(e.g., Voigt–Reuss bounds), a normative envelope (e.g., a Eurocode decay curve) or
an empirical ceiling.

## Pipeline (paper Algorithm 1)

1. **Monotone quantile base models.** Lower/upper conditional quantiles
   (`q̂_lo`, `q̂_hi` at levels `α/2`, `1−α/2`) and a median are fit with
   monotonicity constraints on physically monotone features.
2. **Band-normalised CQR score.** On a held-out calibration set the CQR score is
   divided by the admissible-band width, so scores are comparable across inputs
   whose corridors differ in width.
3. **Projection-aware calibration (`robust=True`).** Calibration points whose
   response falls **outside** the corridor receive an **infinite** score. The
   conformal quantile `Q` is the ⌈(n+1)(1−α)⌉-th smallest score.
4. **Projection.** The conformalised interval is intersected with `[L(x), U(x)]`,
   so the output is admissible by construction (zero physical violations).

Step 3 is exactly nested conformal prediction (Gupta, Kuchibhotla & Ramdas, 2022)
applied to the *projected* family of intervals: `Y ∈ C̃_Q(X)` if and only if the
projection-aware score is `≤ Q` (paper Lemma 1). The device is therefore not new in
itself; what the paper analyses is what it guarantees when the bounds are wrong.

## Guarantees

Let `η = P(Y ∉ [L(X), U(X)])` be the fraction of responses outside the assumed
corridor (zero when the bounds are valid, positive when they are misspecified).

- **Finite-sample validity (Theorem 3(i)).** Whenever `Q < ∞`, the projected interval
  covers with probability `≥ 1 − α` under exchangeability alone — no assumption on
  the bounds is needed.
- **Admissibility ceiling (Proposition 3).** No interval contained in `[L, U]` can
  cover more than `1 − η`. Hence the nominal level `1 − α` is attainable inside the
  corridor if and only if `η ≤ α`; otherwise the best attainable coverage is `1 − η`,
  and the interval that attains it is the full corridor.
- **Attainment (Theorem 3(ii)).** Asymptotically `Q < ∞` iff `η < α`; the
  projection-aware interval then covers `1 − α`. For `η ≥ α`, `Q = +∞` and the
  method returns the full corridor, whose coverage is `1 − η`. In both regimes the
  realised coverage is `1 − max(α, η)`.
- **Calibrate-then-clip loses coverage (Theorem 2).** Calibrating first and clipping
  afterwards guarantees only `P(Y ∈ C̃) ≥ (1 − α) − η`; the loss is realised in all
  our experiments. Because replacing scores by `+∞` can only raise the quantile, the
  projection-aware interval always *contains* the calibrate-then-clip interval, so
  its coverage is higher by construction; the relevant question is therefore
  whether the extra width buys back the nominal level, which is what Theorem 3
  answers.

## What the diagnostics mean in deployment

After calibration the model exposes `eta_hat` (the calibration estimate of `η`) and
`feasible` (`Q < ∞`). A feasible calibration with a small `eta_hat` means the
bound is quantitatively loose but structurally sound. As `eta_hat` approaches `α`
the interval widens towards the corridor, and when `feasible` is `False` the
returned interval *is* the corridor: the bound is incompatible with the data at
the requested level and should be revisited rather than trusted.

When the bounds are valid (`η = 0`) the method reduces to standard
bound-constrained conformal prediction (Li et al., 2025) and recovers the target
`1 − α`; their Theorem 1 shows that, with valid bounds, clipping before or after
calibration gives the same interval — the misspecified case is precisely where
that equivalence breaks.
