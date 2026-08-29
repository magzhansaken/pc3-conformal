#!/usr/bin/env python3
"""Reproduce every figure and table in the paper, in order.

Usage:
    python run_all.py            # runs all paper experiments
    python run_all.py --quick    # skips the slowest native-baseline scripts

All outputs are written to ./out/. Random seeds are fixed, so results are
deterministic. composite_real_experiments.py needs the SFRC file in ./data/ (the
textile-polymer set auto-downloads); elastic_experiment.py loads its data via
matminer. See data/README.md.
"""
import subprocess, sys, time

QUICK = "--quick" in sys.argv

# (script, produces, needs_data)
PIPELINE = [
    ("architecture_figure.py",         "Fig. 1",          False),
    ("pc3.py",                         "Fig. 2, 10; Table 3, 5; S1, S4", False),
    ("frp_experiment.py",              "Supp. S2",        False),
    ("revision_experiments.py",        "Supp. S3",        False),
    ("robust_cp.py",                   "Fig. 3; Tables 6, 8", False),
    ("robust_concrete.py",             "Fig. 4, 5; Tables 7, 8", False),
    ("composite_real_experiments.py",  "Fig. 7; Table 9", True),
    ("grouped_families.py",            "Fig. 8, Table 10",True),
    ("omlt_comparison.py",             "Fig. 11, Table 11", False),
    ("finite_sample.py",               "Supplementary Table S1", False),
    ("elastic_experiment.py",          "Fig. 6",          False),
    ("ias.py",                         "Fig. 9 (decision-support layer)", False),
]

def main():
    failures = []
    for script, produces, needs_data in PIPELINE:
        print(f"\n{'='*72}\n▶ {script}  →  {produces}\n{'='*72}", flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, script])
        dt = time.time() - t0
        status = "ok" if rc == 0 else f"FAILED (exit {rc})"
        if rc != 0:
            failures.append(script)
        print(f"  [{status}, {dt:.0f}s]", flush=True)
    print(f"\n{'='*72}")
    if failures:
        print("Completed with failures:", ", ".join(failures))
        print("(composite_real_experiments.py needs ./data; see data/README.md)")
        sys.exit(1)
    print("All experiments completed. Figures are in ./out/.")

if __name__ == "__main__":
    main()
