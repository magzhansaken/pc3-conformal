# Usage

The method is the `PC3` class in [`pc3.py`](../pc3.py); the decision-support layer is
`PC3System` in [`ias.py`](../ias.py).

## API

```python
import pc3, ias

model = pc3.PC3(
    monotone,        # array of {-1, 0, +1} per feature: physical monotonicity direction
    bounds_fn,       # callable: X -> (L_array, U_array), the admissible corridor
    "cqr",           # base nonconformity score ("cqr" or "residual")
    True,            # use_monotone: monotone quantile base models
    True,            # project: intersect the interval with [L, U]
    True,            # physics_aware: band-normalised score (Eq. 4)
    alpha=0.1,       # miscoverage level (1 - alpha = target coverage)
    robust=True,     # projection-aware score (Eq. 6): +inf for calibration points outside [L, U]
)
model.fit(X_train, y_train).calibrate(X_cal, y_cal)
pred, lo, hi, L, U = model.predict(X_test)

model.eta_hat    # bound-violation rate on the calibration set
model.feasible   # True if Q < inf (target attainable inside the corridor)
```

`bounds_fn(X)` must return two arrays `(L, U)` with `L <= U`, one entry per row of
`X`. For a one-sided ceiling use `L = 0` (or `-inf`) and `U = ceiling`.

`robust=False` reproduces the naive calibrate-then-clip scheme used as a baseline in
the paper; `robust=True` is the recommended setting whenever the bounds may be
misspecified. `robust_cp.RobustPC3` is kept as an alias of `PC3(..., robust=True)`.

## Decision-support layer

```python
system = ias.PC3System(model, feature_names, background=X_train)   # SHAP optional
rec = system.query(x)          # dict: y_hat, lo, hi, L, U, point_clipped, interval_clipped,
                               #       width_ratio, feasible, eta_hat, n_cal, shap (top drivers)
df  = system.batch(X_new)      # DataFrame, one row per query (batch mode)
ias.render_card(rec, "card.png")   # the inference card of paper Figure 10
```

`python ias.py --demo concrete --ceiling-percentile 97.5` reproduces Figure 10 from real
data and writes `out/ias_record.json`, `out/ias_batch_demo.csv`, `out/figG_ias_card.png`.

## Defining the corridor

- **Rigorous bounds** (e.g., Voigt–Reuss for effective moduli): the corridor holds
  for all samples (`η = 0`); projection removes violations and shrinks intervals.
- **Approximate bounds** (rule-of-mixtures, normative envelopes, empirical
  ceilings): a real fraction `η` of responses may exceed the corridor;
  projection-aware calibration keeps coverage on the admissible frontier
  `1 − max(α, η)` (see [METHOD.md](METHOD.md)). Watch `eta_hat`: when it
  approaches `α` the interval widens towards the corridor, and when it exceeds
  `α` the corridor itself is returned.

## Reproducing the paper

```bash
pip install -r requirements.txt
python run_all.py
```

See the [README](../README.md) for the full script→figure mapping and a minimal
example in [`examples/demo.py`](../examples/demo.py).
