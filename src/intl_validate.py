"""
intl_validate.py - Checks that each rebuilt market tracks its official index: daily correlation of
the cap-weighted universe return (sector returns weighted by previous-day sector shares) with the
index's daily log return. Writes output/intl/validation.csv.
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import yfinance as yf  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAN = os.path.join(BASE, "data", "intl", "panels")
OUT = os.path.join(BASE, "output", "intl")
INDEX = {"United States": "^GSPC", "Japan": "^N225", "China": "000001.SS", "United Kingdom": "^FTSE",
         "Canada": "^GSPTSE", "Korea": "^KS11", "Australia": "^AXJO", "India": "^NSEI",
         "Germany": "^GDAXI", "Hong Kong": "^HSI", "Brazil": "^BVSP"}


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for m, ix in INDEX.items():
        t = m.lower().replace(" ", "_")
        R = pd.read_csv(os.path.join(PAN, f"{t}_returns.csv"), index_col=0, parse_dates=True)
        C = pd.read_csv(os.path.join(PAN, f"{t}_concentration.csv"), parse_dates=["date"])
        w = C.pivot(index="date", columns="sector", values="sector_share").shift(1).reindex(R.index)
        mkt = (R * w).sum(axis=1, min_count=1) / w.where(R.notna()).sum(axis=1)
        px = yf.download(ix, start="2018-09-01", end="2026-09-01", auto_adjust=True, progress=False)["Close"].squeeze()
        ir = 100 * np.log(px).diff()
        j = pd.concat([mkt, ir], axis=1, keys=["universe", "index"]).dropna()
        top3 = 100 * C.groupby("sector")["top3"].mean()
        rows.append({"market": m, "index": ix, "days": len(j), "corr_daily": j.corr().iloc[0, 1],
                     "corr_monthly": j.resample("ME").sum().corr().iloc[0, 1],
                     "mean_top3_across_sectors": top3.mean()})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(os.path.join(OUT, "validation.csv"), index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
