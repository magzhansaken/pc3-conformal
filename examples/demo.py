"""Minimal example: PC3 on synthetic composite data.

Fits the method, reports coverage and physical-violation rate, and saves a
small parity plot. Run from the repository root:

    python examples/demo.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pc3

# 1. synthetic data with rigorous Voigt-Reuss bounds
X, y, mono, bounds_fn, *_ = pc3.make_synthetic_composite(n=600, seed=0)
ntr, ncal = 300, 150
Xtr, ytr = X[:ntr], y[:ntr]
Xcal, ycal = X[ntr:ntr + ncal], y[ntr:ntr + ncal]
Xte, yte = X[ntr + ncal:], y[ntr + ncal:]

# 2. fit PC3: monotone quantiles -> physics-aware calibration -> projection
model = pc3.PC3(mono, bounds_fn, "cqr", True, True, True, alpha=0.1)
model.fit(Xtr, ytr).calibrate(Xcal, ycal)
pred, lo, hi, _, _ = model.predict(Xte)

# 3. report
cov = float(np.mean((yte >= lo) & (yte <= hi)))
L, U = bounds_fn(Xte)
viol = float(np.mean((lo < L - 1e-9) | (hi > U + 1e-9)))
print(f"target coverage : 90%")
print(f"empirical cover : {cov * 100:.1f}%")
print(f"physical viol.  : {viol * 100:.1f}%   (should be 0.0%)")
print(f"mean width      : {np.mean(hi - lo):.2f}")

# 4. small parity plot
order = np.argsort(yte)
plt.figure(figsize=(6, 4))
plt.fill_between(yte[order], lo[order], hi[order], alpha=0.3, label="90% interval")
plt.plot(yte[order], pred[order], ".", ms=4, label="prediction")
plt.plot(yte[order], yte[order], "k--", lw=1, label="ideal")
plt.xlabel("True value")
plt.ylabel("Prediction +/- interval")
plt.title(f"PC3 demo: {cov * 100:.0f}% coverage, {viol * 100:.0f}% violations")
plt.legend()
plt.tight_layout()
plt.savefig("demo_parity.png", dpi=120)
print("\nsaved: demo_parity.png")
