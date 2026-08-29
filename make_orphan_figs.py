# -*- coding: utf-8 -*-
"""Figure 10 (deployment inference card).

The card is generated from a *real* query by the decision-support layer in ``ias.py``
(UCI Concrete, robust calibration, empirical ceiling at the 97.5th training percentile),
replacing the earlier hand-drawn illustration with representative values.
Writes out/figG_ias_card.png plus the JSON record and a batch CSV.
"""
from ias import demo

if __name__ == "__main__":
    demo("concrete", ceiling_percentile=97.5)
    print("Figure 10 written: out/figG_ias_card.png (from ias.py)")
