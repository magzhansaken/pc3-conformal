#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ias.py - decision-support (information-analytical) layer of PC3
================================================================
Paper Section 3.4 (system) and Figure 10 (inference card).

The layer wraps a fitted and calibrated ``pc3.PC3`` model and returns, for every
query, a structured record that a practitioner can act on:

  y_hat              projected point prediction (inside the corridor by construction)
  lo, hi             calibrated, physically admissible interval  C(x) ⊆ [L(x), U(x)]
  L, U               physical corridor at the query
  y_hat_raw          unprojected median prediction
  point_clipped      True if the raw prediction fell outside the corridor
  interval_clipped   True if the conformal interval had to be cut at L or U
  width              interval width;  corridor_width;  width_ratio = width / corridor_width
  branch             'conformal' if the calibrated threshold Q is finite, 'corridor' if Q = +inf
                     and the full corridor was returned (Lemma 2); ``feasible`` is kept as the
                     boolean alias (Q finite) -- it is NOT a certificate that eta <= alpha
  eta_hat            bound-violation rate on the calibration set, K/n
  eta_lo, eta_hi     one-sided (1-delta) Clopper-Pearson bounds on eta (Corollary 4)
  feasibility        'feasible' (eta_hi <= alpha), 'infeasible' (eta_lo > alpha) or
                     'indeterminate' -- each definite state holds with confidence 1-delta
  pi_inf_hat         probability that a calibration sample of this size falls back to the corridor
  coverage_floor_marginal   lower bound on the marginal coverage (Theorem 3(i)) at confidence 1-delta
  coverage_floor_realized   lower bound on the realized coverage of this calibrated interval, conf. 1-2*delta
  n_cal, K, alpha, delta    calibration size, out-of-corridor count, miscoverage level, confidence level
  shap               top feature contributions of the median model (optional)

Note that "the prediction lies inside the corridor" is always true after projection
(Corollary 1), so it is *not* reported as a flag: the informative outputs are the two
clipping indicators, the three-state feasibility diagnostic with its confidence interval, and the
coverage floors.

Modes
-----
  interactive : PC3System.query(x)  -> dict
  batch       : PC3System.batch(X)  -> pandas.DataFrame   (also .to_json / .to_csv)
  card        : render_card(record, path) draws the inference card from a real record

Demo (reproduces Figure 10 from real data):
  python ias.py --demo concrete --ceiling-percentile 97.5
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import pc3

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


class PC3System:
    """Decision-support wrapper around a calibrated ``pc3.PC3`` model."""

    def __init__(self, model, feature_names, background=None, explain=True, top_k=6):
        if getattr(model, "Qg", None) is None:
            raise ValueError("model must be fitted and calibrated before wrapping it in PC3System")
        self.model, self.feature_names, self.top_k = model, list(feature_names), top_k
        self.explainer = None
        if explain and background is not None:
            try:
                import shap
                bg = shap.utils.sample(np.asarray(background), min(80, len(background)), random_state=0)
                self.explainer = shap.Explainer(model.q_med.predict, bg)
            except Exception:                       # SHAP is optional: the record is complete without it
                self.explainer = None

    # ------------------------------------------------------------------ core
    def _records(self, X, with_shap):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        m = self.model
        point, lo, hi, L, U = m.predict(X)
        raw, lo_raw, hi_raw = m.predict_base(X)
        sv = None
        if with_shap and self.explainer is not None:
            try:
                sv = np.asarray(self.explainer(X).values)
            except Exception:
                sv = None
        recs = []
        for i in range(len(X)):
            width, cw = float(hi[i] - lo[i]), float(U[i] - L[i])
            rec = dict(
                y_hat=float(point[i]), lo=float(lo[i]), hi=float(hi[i]),
                L=float(L[i]), U=float(U[i]), y_hat_raw=float(raw[i]),
                point_clipped=bool(raw[i] < L[i] - 1e-9 or raw[i] > U[i] + 1e-9),
                interval_clipped=bool(lo_raw[i] < L[i] - 1e-9 or hi_raw[i] > U[i] + 1e-9),
                width=width, corridor_width=cw,
                width_ratio=float(width / cw) if cw > 0 else float("nan"),
                feasible=bool(m.feasible), eta_hat=float(m.eta_hat),
                n_cal=int(m.n_cal), alpha=float(m.alpha),
                **({k: m.diag[k] for k in ("branch", "eta_lo", "eta_hi", "feasibility", "pi_inf_hat",
                                           "coverage_floor_marginal", "coverage_floor_realized", "K", "delta")}
                   if getattr(m, "diag", None) else {}),
            )
            if sv is not None:
                order = np.argsort(-np.abs(sv[i]))[: self.top_k]
                rec["shap"] = [(self.feature_names[j], float(sv[i, j])) for j in order]
            rec["features"] = {f: float(v) for f, v in zip(self.feature_names, X[i])}
            recs.append(rec)
        return recs

    def query(self, x, with_shap=True):
        """Single query -> record (dict)."""
        return self._records(x, with_shap)[0]

    def batch(self, X, with_shap=False):
        """Batch mode -> DataFrame with one row per query (SHAP columns optional)."""
        recs = self._records(X, with_shap)
        rows = []
        for r in recs:
            row = {k: v for k, v in r.items() if k not in ("shap", "features")}
            if "shap" in r:
                row["top_driver"] = r["shap"][0][0]
                row["top_driver_shap"] = r["shap"][0][1]
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def to_json(record, path=None):
        s = json.dumps(record, indent=2)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s


# ---------------------------------------------------------------------- card
def render_card(rec, path, title="Inference card (PC3)", unit="MPa"):
    """Draw the deployment inference card (paper Figure 10) from a real record."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(13, 4.6))
    cov = int(round(100 * (1 - rec["alpha"])))
    fig.suptitle(title, fontsize=15, fontweight="bold", color="#1F4E78", x=0.02, y=0.985, ha="left")
    line1 = (f"Predicted strength: {rec['y_hat']:.1f} {unit}     {cov}% interval: "
             f"[{rec['lo']:.1f}, {rec['hi']:.1f}] {unit}     physical corridor: [{rec['L']:.1f}, {rec['U']:.1f}] {unit}")
    if "feasibility" in rec:
        ci = int(round(100 * (1 - 2 * rec["delta"]))); cf = int(round(100 * (1 - rec["delta"])))
        line2 = (f"Calibration (n = {rec['n_cal']}): bound-violation rate  \u03b7\u0302 = {100*rec['eta_hat']:.1f}%, "
                 f"{ci}% CI [{100*rec['eta_lo']:.1f}%, {100*rec['eta_hi']:.1f}%]     "
                 f"{cov}% level inside corridor: {rec['feasibility']} ({cf}% conf.)     "
                 f"returned: {'conformal interval' if rec['branch']=='conformal' else 'full corridor'}")
        line3 = (f"Coverage floors: marginal \u2265 {100*rec['coverage_floor_marginal']:.1f}% ({cf}% conf.), "
                 f"this calibration \u2265 {100*rec['coverage_floor_realized']:.1f}% ({ci}% conf.)     "
                 f"fallback probability at \u03b7\u0302: {rec['pi_inf_hat']:.2f}     "
                 f"point clipped: {'yes' if rec['point_clipped'] else 'no'}     "
                 f"interval clipped: {'yes' if rec['interval_clipped'] else 'no'}")
    else:
        feas = "Q finite" if rec["feasible"] else "Q = +inf (full corridor returned)"
        line2 = (f"Calibration: {feas}     \u03b7\u0302 = {100*rec['eta_hat']:.1f}% (n = {rec['n_cal']})     "
                 f"point clipped: {'yes' if rec['point_clipped'] else 'no'}     "
                 f"interval clipped: {'yes' if rec['interval_clipped'] else 'no'}")
        line3 = None
    fig.text(0.02, 0.885, line1, fontsize=10.5)
    fig.text(0.02, 0.835, line2, fontsize=9.2, color="#333333")
    if line3: fig.text(0.02, 0.79, line3, fontsize=9.2, color="#333333")
    axL = fig.add_axes([0.04, 0.14, 0.44, 0.54])
    L, U, lo, hi, p = rec["L"], rec["U"], rec["lo"], rec["hi"], rec["y_hat"]
    axL.axhspan(0, 1, color="#e7e7e7")
    axL.axvspan(lo, hi, color="#9CC3E0", alpha=0.9)
    axL.scatter([p], [0.5], s=160, color="#2E5496", zorder=5)
    if rec["point_clipped"]:
        axL.scatter([np.clip(rec["y_hat_raw"], L - 0.02*(U-L), U + 0.02*(U-L))], [0.5], s=90,
                    facecolors="none", edgecolors="#C0392B", zorder=6, label="raw (unprojected)")
    axL.axvline(lo, ls="--", color="#5B8FB9", lw=1); axL.axvline(hi, ls="--", color="#5B8FB9", lw=1)
    pad = 0.03 * (U - L)
    axL.set_xlim(L - pad, U + pad); axL.set_ylim(0, 1); axL.set_yticks([])
    axL.set_xlabel(f"Strength, {unit}"); axL.set_title("Prediction within the physical corridor")
    axL.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#2E5496", markersize=9, label="prediction"),
                        Patch(facecolor="#e7e7e7", label="physical corridor [L, U]"),
                        Patch(facecolor="#9CC3E0", label=f"{cov}% conformal interval")],
               loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8, frameon=False)
    axR = fig.add_axes([0.56, 0.14, 0.40, 0.54])
    if rec.get("shap"):
        feats = [f for f, _ in rec["shap"]][::-1]; vals = [v for _, v in rec["shap"]][::-1]
        cols = ["#C0392B" if v < 0 else "#3a8a4d" for v in vals]
        axR.barh(feats, vals, color=cols); axR.axvline(0, color="#888", lw=0.8)
        axR.set_xlabel(f"Contribution to the median prediction (SHAP), {unit}")
        axR.set_title("Why this prediction")
    else:
        axR.axis("off"); axR.text(0.5, 0.5, "SHAP explanation unavailable", ha="center", va="center")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=600, bbox_inches="tight"); plt.close()
    return path


# ---------------------------------------------------------------------- demo
def demo(dataset="concrete", ceiling_percentile=97.5, alpha=0.1, seed=0, row="auto", out=OUT):
    """Fit PC3 on a public dataset, wrap it in the system layer, query one test row,
    write the JSON record, a batch CSV and the inference card (Figure 10)."""
    if dataset != "concrete":
        raise ValueError("demo currently supports the UCI Concrete dataset")
    X, y, mono, _, feats, gf = pc3.load_concrete()
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=seed)
    Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed)
    cap = float(np.percentile(ytr, ceiling_percentile))          # empirical ceiling from TRAINING strengths
    bf = lambda Xq: (np.zeros(len(Xq)), np.full(len(Xq), cap))
    model = pc3.PC3(mono, bf, "cqr", True, True, True, alpha=alpha, robust=True)
    model.fit(Xtr, ytr).calibrate(Xcal, ycal)
    system = PC3System(model, feats, background=Xtr)
    os.makedirs(out, exist_ok=True)
    df = system.batch(Xte[:200]); df.to_csv(f"{out}/ias_batch_demo.csv", index=False)
    if row == "auto":                                          # first query whose interval is cut at the ceiling
        hits = np.where(df.interval_clipped.values)[0]; row = int(hits[0]) if len(hits) else 0
    rec = system.query(Xte[row])
    system.to_json(rec, f"{out}/ias_record.json")
    card = render_card(rec, f"{out}/figG_ias_card.png",
                       title=f"Inference card (PC3): UCI Concrete, empirical ceiling P{ceiling_percentile:g}")
    print(json.dumps({k: v for k, v in rec.items() if k != "features"}, indent=2))
    print(f"batch: {len(df)} queries, feasibility = {df['feasibility'].iloc[0] if 'feasibility' in df else 'n/a'}, "
          f"{100*df.point_clipped.mean():.1f}% points clipped, {100*df.interval_clipped.mean():.1f}% intervals clipped, "
          f"test coverage {100*np.mean((yte[:200] >= df.lo) & (yte[:200] <= df.hi)):.1f}%")
    print("wrote:", f"{out}/ias_record.json", f"{out}/ias_batch_demo.csv", card)
    return rec, df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="concrete")
    ap.add_argument("--ceiling-percentile", type=float, default=97.5)
    ap.add_argument("--row", default="auto", help="test-row index, or 'auto' for the first clipped interval")
    a = ap.parse_args()
    demo(a.demo, a.ceiling_percentile, row=a.row if a.row == "auto" else int(a.row))
