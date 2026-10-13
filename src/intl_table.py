"""
intl_table.py - One table per market for the manuscript: universe, depth, concentration change,
the concentration effect on monthly R2 under current and fixed membership, the granular share,
and system co-movement at the start and end of the sample. Writes output/intl/paper_table.csv.
"""
import json
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "output", "intl")
FIX = os.path.join(BASE, "output", "intl_fixed")
UNIVERSE = {"United States": "S&P 500", "Japan": "TOPIX 500", "China": "CSI 300",
            "United Kingdom": "FTSE 350", "Canada": "S&P/TSX Comp.", "Korea": "KOSPI 200",
            "Australia": "S&P/ASX 200", "India": "NIFTY 100", "Germany": "DAX + MDAX",
            "Hong Kong": "Hang Seng", "Brazil": "Ibovespa"}


def main():
    res = pd.read_csv(os.path.join(OUT, "market_results.csv"))
    fx = pd.read_csv(os.path.join(FIX, "market_results.csv"))
    gran = pd.read_csv(os.path.join(OUT, "granular_by_market.csv")).set_index("market")
    het = pd.read_csv(os.path.join(OUT, "heterogeneity.csv")).set_index("market")
    val = pd.read_csv(os.path.join(OUT, "validation.csv")).set_index("market")
    build = {b["market"]: b for b in json.load(open(os.path.join(BASE, "data", "intl", "panels", "build_summary.json")))}
    rows = []
    for m in UNIVERSE:
        r = res[(res.market == m) & (res.outcome == "R2")].iloc[0]
        f = fx[(fx.market == m) & (fx.outcome == "R2")].iloc[0]
        rr = res[(res.market == m) & (res.outcome == "rho2")].iloc[0]
        g = gran.loc[m]
        rows.append({"market": m, "universe": UNIVERSE[m], "firms": build[m]["firms"],
                     "sectors": int(r["sectors"]), "firms_per_sector": het.loc[m, "firms_per_sector"],
                     "corr_index": val.loc[m, "corr_daily"],
                     "top3_first": g["top3_first12"], "top3_last": g["top3_last12"],
                     "coef_R2": r["coef"], "p_R2": r["p_perm"], "coef_rho2": rr["coef"], "p_rho2": rr["p_perm"],
                     "coef_R2_fixed": f["coef"], "p_R2_fixed": f["p_perm"],
                     "share_top_below_rest": g["share_top_below_rest"], "sector_years": g["sector_years"],
                     "rho2_first": g["avg_rho2_first12"], "rho2_last": g["avg_rho2_last12"],
                     "rho2_last_cf": g["avg_rho2_last12_cf"]})
    T = pd.DataFrame(rows)
    T.round(3).to_csv(os.path.join(OUT, "paper_table.csv"), index=False)
    pd.set_option("display.width", 250)
    print(T.round(2).to_string(index=False))
    intl = T[T.market != "United States"]
    print("pooled top<rest share (all 11):", round((T.share_top_below_rest * T.sector_years).sum() / T.sector_years.sum(), 3))
    print("markets with rho2 falling by >2 pts:", list(T[T.rho2_last < T.rho2_first - 2].market))
    print("max |actual - cf|:", round((T.rho2_last - T.rho2_last_cf).abs().max(), 2))
    print("negative coef R2 (current) among 10 intl:", int((intl.coef_R2 < 0).sum()))


if __name__ == "__main__":
    main()
