"""
hhi_dynamics.py - Time-varying Herfindahl-Hirschman Index (market concentration)
per GICS sub-sector, 2019-2025. Quarterly is appropriate here: HHI is a
structural concentration measure, not a high-frequency return process.

For each sub-sector and quarter, HHI = sum of squared market-cap shares
(range 0..1; higher = more concentrated). We report the average level and the
linear trend (concentrating vs fragmenting) per sub-sector.

Output: output/tables/hhi_dynamics.csv
"""
from __future__ import annotations

import glob
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    rows = []
    series = {}
    for f in glob.glob(os.path.join(BASE, "data_mkt_cap", "*.csv")):
        sub = os.path.basename(f).replace("_mkt_cap_quarter_end.csv", "")
        df = pd.read_csv(f, parse_dates=["Date"])
        hhi_q = []
        for date, day in df.groupby("Date"):
            tot = day["MarketCap"].sum()
            if tot > 0 and len(day) >= 2:
                shares = day["MarketCap"] / tot
                hhi_q.append((date, float((shares ** 2).sum())))
        if len(hhi_q) < 8:
            continue
        s = pd.Series(dict(hhi_q)).sort_index()
        series[sub] = s
        t = np.arange(len(s))
        slope = np.polyfit(t, s.values, 1)[0]  # per quarter
        rows.append({
            "sub_sector": sub, "n_firms": df["Ticker"].nunique(),
            "hhi_mean": s.mean(), "hhi_start": s.iloc[0], "hhi_end": s.iloc[-1],
            "trend_per_qtr": slope, "trend_total": slope * (len(s) - 1),
        })
    out = pd.DataFrame(rows).sort_values("hhi_mean", ascending=False)
    os.makedirs(os.path.join(BASE, "output", "tables"), exist_ok=True)
    out.to_csv(os.path.join(BASE, "output", "tables", "hhi_dynamics.csv"), index=False)

    print(f"HHI dynamics for {len(out)} sub-sectors, {len(next(iter(series.values())))} quarters")
    print(f"mean HHI across sub-sectors: {out['hhi_mean'].mean():.3f}")
    print("\nMost concentrated (top 5 by mean HHI):")
    print(out[["sub_sector", "hhi_mean", "n_firms"]].head(5).to_string(index=False))
    print("\nFastest concentrating (top 5 by trend):")
    print(out.nlargest(5, "trend_total")[["sub_sector", "hhi_start", "hhi_end", "trend_total"]].to_string(index=False))
    print("\nFastest fragmenting (top 5):")
    print(out.nsmallest(5, "trend_total")[["sub_sector", "hhi_start", "hhi_end", "trend_total"]].to_string(index=False))
    conc = (out["trend_total"] > 0).sum()
    print(f"\nconcentrating: {conc}/{len(out)} sub-sectors; fragmenting: {len(out)-conc}/{len(out)}")


if __name__ == "__main__":
    main()
