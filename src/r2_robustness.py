"""
r2_robustness.py - Robustness of the monthly R2 result: Driscoll-Kraay lags, HHI,
no controls, leave-one-sector-out. Writes output/v3/r2_robustness.csv.
"""
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from econometrics import SECTORS, driscoll_kraay, load  # noqa: E402
from inference_checks import OUT3, conc_controls, period_outcomes  # noqa: E402

CTRL = ["log_share_lag", "logvol"]


def main():
    rows = []
    for lab, fn in (("Constituent", "const_returns.csv"), ("ETF", "etf_returns.csv")):
        R = load(fn)[SECTORS]
        pan = period_outcomes(R, "M").merge(conc_controls("M"), on=["sector", "period"])
        pan = pan[pan["period"] >= pd.Period("2018-11", "M")].rename(columns={"period": "month"})
        specs = [(f"DK lag {l}", pan, ["top3_lag_pp"] + CTRL, l) for l in (3, 6, 12)]
        specs += [("HHI", pan, ["hhi_lag_pp"] + CTRL, 3), ("no controls", pan, ["top3_lag_pp"], 3)]
        specs += [(f"drop {s}", pan[pan["sector"] != s], ["top3_lag_pp"] + CTRL, 3) for s in SECTORS]
        for name, d, xs, lag in specs:
            tab, info = driscoll_kraay(d, "R2", xs, lag=lag)
            rows.append({"returns": lab, "spec": name, "mean_R2": d["R2"].mean(), **tab.loc[xs[0]].to_dict()})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(os.path.join(OUT3, "r2_robustness.csv"), index=False)
    print(out[~out["spec"].str.startswith("drop")].round(3).to_string(index=False))
    loo = out[out["spec"].str.startswith("drop")]
    print(loo.groupby("returns")[["coef", "t"]].agg(["min", "max"]).round(3))


if __name__ == "__main__":
    main()
