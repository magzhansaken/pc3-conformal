# PC³ — Physics-Constrained Conformal Prediction for Composites

Reference implementation and reproduction code for the paper:

> **An Uncertainty-Aware Decision-Support System for Composite-Property Prediction with Physically Admissible Conformal Intervals**
> M. Sarsenbay, S. Zhuzbayev, G. Baenova, G. Alkhanova, R. Niyazova, A. Bakytzhan. *Information* (MDPI), 2026, submitted. DOI: *to be assigned on acceptance*.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Reproducible](https://img.shields.io/badge/seeds-fixed-blueviolet)

This repository contains everything needed to reproduce the figures and tables in the paper. Random seeds are fixed, so results are deterministic.

> **Quick reproduction (one command):**
> ```bash
> pip install -r requirements.txt
> python run_all.py
> ```
> This runs every paper experiment in order and writes all figures to `./out/`.
> The composition-grouped validation of §4.4 runs via `python grouped_families.py`
> (UCI Concrete auto-downloads; place `uhpc.csv` and `scc_ht.csv` in `./data/` first — see `data/README.md`).
> (`composite_real_experiments.py` additionally needs the SFRC and textile-polymer files in `./data/`; see [`data/README.md`](data/README.md).)

---

## What is here

| File | Produces (numbering of the submitted manuscript) |
|------|--------------------------------------------------|
| `architecture_figure.py` | System architecture schematic → Fig. 1 |
| `pc3.py` | Core method (PC³) + synthetic/UCI-Concrete experiments → Fig. 2, 10; Table 3 (synthetic, concrete blocks), Table 5 |
| `frp_experiment.py` | FRP-modulus benchmark → Table 3 (FRP block); Fig. A1, A2 |
| `revision_experiments.py` | `base + clip` study → Table 4; Fig. A3, A4 |
| `robust_cp.py` | Robust bound-constrained CP recovery curve, coverage and width → Fig. 3; Tables 6, 8 (`robust_cp_fast.py`: same numbers, base fits shared across ρ) |
| `robust_concrete.py` | Real-concrete η-sweep (coverage, width) → Fig. 4; Tables 7, 8 |
| `elastic_experiment.py` | DFT elastic moduli with rigorous Voigt–Reuss bounds → Fig. 6 |
| `composite_real_experiments.py` | Real SFRC and textile-polymer composites → Fig. 7 (panels a, b); Table 9 |
| `grouped_families.py` | **Composition-grouped validation on five cementitious families (§4.4)** → Fig. 8; Table 10 |
| `ias.py` | **Decision-support layer** (per-query records, batch mode, JSON, inference card) → Fig. 9 |
| `make_orphan_figs.py` | Thin wrapper that regenerates Fig. 9 through `ias.py` |
| `omlt_comparison.py` | Calibration order under misspecified bounds (calibrate-then-clip vs. clip-then-calibrate) → Fig. 11; Table 11 |
| `finite_sample_v2.py` | **Finite-sample study of Theorem 3** (exact coverage law vs. calibration size; fixed models, i.i.d. calibration draws from the pool) → Fig. 5; Appendix A: Table A1, Fig. A5. `finite_sample.py` is the superseded earlier protocol |
| `verification/` | Numerical checks of Theorem 3, Lemma 2, Proposition 4 and Corollary 4 (`verify_*.py`) |
| `grouped_summary.py` | Prints the §4.4 summary numbers from `out/grouped_results.csv` |
| *(all figures are written at 600 dpi, the resolution MDPI requests)* | |
| `figures/` | The final rendered figures, for reference |

`pc3.py` defines the method as the `PC3` class: monotone quantile base models → physics-aware
conformal calibration → projection into the admissible corridor `[L(x), U(x)]`.

> **Paper's Algorithm 1** (`train → calibrate → predict`, including the robust ∞-score variant) is implemented in `pc3.py`: the band-normalized CQR score, the ⌈(n+1)(1−α)⌉ conformal quantile and the projection-aware ∞-score (`PC3(..., robust=True)`, out-of-corridor calibration points → +∞). After calibration the model exposes `eta_hat` (bound-violation rate on the calibration set) and `feasible` (`Q < ∞`). `robust_cp.RobustPC3` is kept as an alias.
>
> **Decision-support layer** (`ias.py`): `PC3System.query(x)` returns the projected prediction, the admissible interval, the corridor, clipping indicators, `feasible`, `eta_hat` and the top SHAP drivers; `PC3System.batch(X)` is the batch mode. Figure 9 (the inference card) is generated from a real query by `python ias.py`.

---

## Requirements

Python 3.10+. Install with any of:

```bash
pip install -r requirements.txt      # pip
# or
conda env create -f environment.yml  # conda
# or
pip install .                        # uses pyproject.toml
# or
docker build -t pc3 . && docker run --rm pc3   # containerised (runs the tests)
```

(All dependencies are needed for the full paper; `pymatgen`/`matminer` are used by `elastic_experiment.py` to build Figure 6 from the DFT dataset.)

---

## Data

- **UCI Concrete** — downloaded automatically on first run of `pc3.py` (no action needed).
- **DFT elastic moduli** (`elastic_tensor_2015`, de Jong et al., 2015) — loaded programmatically by `elastic_experiment.py` via `matminer` (no manual download).
- **Textile-polymer composite** (Malashin et al., 2024, `github.com/catauggie/TPCM`) — downloaded automatically by `composite_real_experiments.py` on first run.
- **Steel-fibre-reinforced concrete** (Shafighfard et al., 2022, Mendeley `doi:10.17632/hjrfgys29n.1`) — place `SFRC_Data_v1.xlsx` in `./data/` (see `data/README.md`).

All datasets used in the paper are publicly available; we redistribute none of them here.

---

## Reproducing the figures

From the repository root:

```bash
pip install -r requirements.txt

python architecture_figure.py         # Fig. 1  (schematic, no data needed)
python pc3.py                         # Fig. 2, 10; Table 3 (synthetic, concrete), Table 5
python frp_experiment.py              # Table 3 (FRP block); Fig. A1, A2
python revision_experiments.py        # Table 4;  Fig. A3, A4
python robust_cp.py                   # Fig. 3;   Tables 6, 8
python robust_concrete.py             # Fig. 4;   Tables 7, 8
python elastic_experiment.py          # Fig. 6            (loads data via matminer)
python composite_real_experiments.py  # Fig. 7 (a, b); Table 9  (SFRC needs ./data; polymer auto-downloads)
python grouped_families.py            # Fig. 8;  Table 10 (needs uhpc/scc_ht/mk_geopolymer/hybrid_aac csv in ./data)
python ias.py                         # Fig. 9 (decision-support layer; real query)
python omlt_comparison.py             # Fig. 11; Table 11
python finite_sample_v2.py            # Fig. 5;   Appendix A: Table A1, Fig. A5 (finite-sample study, Theorem 3)
```

Each script writes its PNGs to an `out/` folder and prints the corresponding tables to the console. The numbers reproduce the values reported in the paper exactly (fixed seeds).

> The paper's **Figures 1–11, Tables 1–11 and Appendix A (Figures A1–A5, Table A1)** are the ones listed in the table above (Figure 7 combines the SFRC and textile-polymer panels, `figures/figQR_real_composites.png`). Scripts additionally emit a few auxiliary diagnostic plots with letter IDs (e.g. `figB`, `figC`, `figI`, `figK`) that are **not** part of the paper; they are kept for completeness. Reference copies of all figures are in [`figures/`](figures/).

---

## Method, in brief

Given a regression target `Y` with deterministic physical bounds `L(x) ≤ Y ≤ U(x)` that may be
**misspecified** (a fraction `η = P(Y ∉ [L,U])` of responses fall outside the assumed corridor), a
*projection-aware* conformal calibration assigns an infinite nonconformity score to out-of-corridor
calibration points — nested conformal prediction applied to the projected family. No interval
contained in the corridor can cover more than `1 − η`, so the nominal `1 − α` is attainable inside the
corridor iff `η ≤ α`; the projection-aware calibration attains `1 − α` in that regime and returns the
full corridor (coverage `1 − η`) otherwise, i.e. `1 − max(α, η)` in both, at zero physical violations,
whereas calibrating first and clipping afterwards loses up to `η`. See `docs/METHOD.md` and the paper.

---

## Documentation and examples

- A minimal runnable example: [`examples/demo.py`](examples/demo.py) (`python examples/demo.py`).
- An annotated notebook: [`notebooks/demo.ipynb`](notebooks/demo.ipynb).
- Method and API docs: [`docs/METHOD.md`](docs/METHOD.md) and [`docs/USAGE.md`](docs/USAGE.md).

## Testing

A fast smoke test (no external data) checks that the method attains the target coverage and never leaves the admissible corridor:

```bash
pytest -q tests/
```

These tests also run automatically on every push via GitHub Actions (`.github/workflows/ci.yml`).

## Citation

If you use this code, please cite the paper (see `CITATION.cff`):

```bibtex
@article{sarsenbay2026pc3,
  title   = {An Uncertainty-Aware Decision-Support System for Composite-Property Prediction with Physically Admissible Conformal Intervals},
  author  = {Sarsenbay, M. and Zhuzbayev, S. and Baenova, G. and Alkhanova, G. and Niyazova, R. and Bakytzhan, A.},
  journal = {Information},
  year    = {2026},
  note    = {DOI to be assigned on acceptance}
}
```

## License

MIT License — see [`LICENSE`](LICENSE).
