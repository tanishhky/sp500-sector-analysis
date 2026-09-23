"""
robust_panel.py - Robustness of the concentration -> connectedness panel result.

  (a) connectedness re-estimated on uncapped constituent-built sector returns, so the return
      series and the concentration measure share one weighting scheme (the SPDR ETFs cap
      single-name weights, which mutes concentration exactly where it is highest);
  (b) dropping Information Technology and Communication Services, the two most capped sectors;
  (c) between-sector means (the cross-section the fixed effects remove).
Run after econometrics.py. Writes output/v2/panel_robustness.csv and between_sector.csv.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yfinance as yf  # noqa: E402

from econometrics import (OUT, PANEL, SECTORS, driscoll_kraay, load,  # noqa: E402
                          tvp_var_connectedness)

NAMES = {"XLC": "Communication Services", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
         "XLE": "Energy", "XLF": "Financials", "XLV": "Health Care", "XLI": "Industrials",
         "XLK": "Information Technology", "XLB": "Materials", "XLRE": "Real Estate", "XLU": "Utilities"}


def monthly(tv, keys=("FROM", "TO", "NET")):
    mm = {}
    for key in keys:
        d = tv[key]
        mm[key] = d.groupby(d.index.to_period("M")).mean().stack().rename(key)
    M = pd.concat(mm, axis=1).reset_index()
    M.columns = ["month", "sector"] + list(keys)
    return M


def main():
    base = pd.read_csv(os.path.join(OUT, "panel_monthly_ret.csv"))
    base["month"] = pd.PeriodIndex(base["month"], freq="M")
    conc_cols = ["sector", "month", "hhi_lag_pp", "top3_lag_pp", "log_share_lag", "logvol"]

    raw = yf.download(list(NAMES), start="2018-06-20", end="2018-10-01", auto_adjust=True, progress=False)
    train = (100 * np.log(raw["Close"]).diff()).rename(columns=NAMES)[SECTORS].dropna()
    CR = load("const_returns.csv")[SECTORS]
    y = pd.concat([train, CR])
    y = y[~y.index.duplicated()]
    tv = tvp_var_connectedness(y, "2018-09-28")
    Mc = monthly(tv)
    pan_c = Mc.merge(base[conc_cols], on=["sector", "month"])

    rows = []
    specs = [("ETF (baseline)", base), ("Constituent-built, uncapped", pan_c),
             ("ETF, excl. IT and COM", base[~base["sector"].isin(["Information Technology", "Communication Services"])]),
             ("Constituent, excl. IT and COM", pan_c[~pan_c["sector"].isin(["Information Technology", "Communication Services"])])]
    for label, pan in specs:
        for dep in ("FROM", "TO", "NET"):
            for c in ("top3_lag_pp", "hhi_lag_pp"):
                tab, info = driscoll_kraay(pan, dep, [c, "log_share_lag", "logvol"], lag=3, fe="twoway")
                rows.append({"sample": label, "dep": dep, "conc": c, **tab.loc[c].to_dict(), **info})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(os.path.join(OUT, "panel_robustness.csv"), index=False)
    print(out[["sample", "dep", "conc", "coef", "se", "t", "p"]].round(3).to_string(index=False))

    conc = pd.read_csv(os.path.join(PANEL, "concentration.csv"), parse_dates=["date"])
    netr = pd.read_csv(os.path.join(OUT, "tvp_ret_NET.csv"), index_col=0, parse_dates=True)
    tor = pd.read_csv(os.path.join(OUT, "tvp_ret_TO.csv"), index_col=0, parse_dates=True)
    frr = pd.read_csv(os.path.join(OUT, "tvp_ret_FROM.csv"), index_col=0, parse_dates=True)
    btw = pd.DataFrame({"top3_%": 100 * conc.groupby("sector")["top3"].mean(),
                        "hhi_x100": 100 * conc.groupby("sector")["hhi"].mean(),
                        "n_firms": conc.groupby("sector")["n_firms"].mean(),
                        "top3_%_2018Q4": 100 * conc[conc["date"] < "2019-01-01"].groupby("sector")["top3"].mean(),
                        "top3_%_2026": 100 * conc[conc["date"] >= "2026-01-01"].groupby("sector")["top3"].mean(),
                        "NET": netr.mean(), "TO": tor.mean(), "FROM": frr.mean(),
                        "share_days_transmitter": (netr > 0).mean()})
    btw = btw.sort_values("top3_%")
    btw.round(2).to_csv(os.path.join(OUT, "between_sector.csv"))
    print(btw.round(2).to_string())
    print("between-sector corr(top3, NET) =", round(btw["top3_%"].corr(btw["NET"]), 3),
          " Spearman =", round(btw["top3_%"].corr(btw["NET"], method="spearman"), 3))


if __name__ == "__main__":
    main()
