"""
mechanism.py - Granularity check: do concentrated sectors co-move less with the rest of the
market? Monthly R^2 of each sector's daily return on the equal-weighted return of the other ten
sectors, regressed on lagged concentration with two-way fixed effects (Driscoll-Kraay SEs).
Writes output/v2/mechanism.csv.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from econometrics import OUT, SECTORS, driscoll_kraay, load

def main():
    rows = []
    for label, fn in [("ETF", "etf_returns.csv"), ("Constituent", "const_returns.csv")]:
        R = load(fn)[SECTORS]
        recs = []
        for s in SECTORS:
            others = R.drop(columns=s).mean(axis=1)
            df = pd.DataFrame({"y": R[s], "x": others})
            for m, g in df.groupby(df.index.to_period("M")):
                if len(g) >= 15:
                    recs.append({"sector": s, "month": m, "R2": 100 * g["y"].corr(g["x"]) ** 2})
        M = pd.DataFrame(recs)
        base = pd.read_csv(os.path.join(OUT, "panel_monthly_ret.csv"))
        base["month"] = pd.PeriodIndex(base["month"], freq="M")
        pan = M.merge(base[["sector", "month", "top3_lag_pp", "hhi_lag_pp", "log_share_lag", "logvol"]],
                      on=["sector", "month"])
        for c in ("top3_lag_pp", "hhi_lag_pp"):
            tab, info = driscoll_kraay(pan, "R2", [c, "log_share_lag", "logvol"], lag=3, fe="twoway")
            rows.append({"returns": label, "conc": c, **tab.loc[c].to_dict(), **info,
                         "mean_R2": pan["R2"].mean()})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(os.path.join(OUT, "mechanism.csv"), index=False)
    print(out.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
