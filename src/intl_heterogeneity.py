"""
intl_heterogeneity.py - Market characteristics that might explain why the concentration effect is
strong in some markets and absent in others. For each market: firms per included sector, the
within-sector variation in the top-three share left after sector and month fixed effects, the level
and change of concentration, the gap between the R2 of the rest of the sector and of the top three,
and the estimated effect. Cross-market correlations are descriptive (eleven markets).
Run after intl_analysis.py (both variants). Writes output/intl/heterogeneity.csv.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "output", "intl")
FIX = os.path.join(BASE, "output", "intl_fixed")
EMERGING = {"China", "India", "Brazil"}


def within_sd(g, col):
    x = g[col] - g.groupby("sector")[col].transform("mean")
    x = x - x.groupby(g["month"]).transform("mean")
    return float(x.std())


def main():
    P = pd.read_csv(os.path.join(OUT, "pooled_panel.csv"))
    res = pd.read_csv(os.path.join(OUT, "market_results.csv"))
    gran = pd.read_csv(os.path.join(OUT, "granular_by_market.csv")).set_index("market")
    comp = pd.read_csv(os.path.join(OUT, "granular_components.csv"))
    fx = pd.read_csv(os.path.join(FIX, "market_results.csv")) if os.path.exists(os.path.join(FIX, "market_results.csv")) else None
    rows = []
    for m, g in P.groupby("market"):
        t = m.lower().replace(" ", "_")
        C = pd.read_csv(os.path.join(BASE, "data", "intl", "panels", f"{t}_concentration.csv"), parse_dates=["date"])
        C = C[C["sector"].isin(g["sector"].unique())]
        r = res[(res["market"] == m) & (res["outcome"] == "R2")].iloc[0]
        cm = comp[comp["market"] == m]
        row = {"market": m, "emerging": m in EMERGING, "sectors": g["sector"].nunique(),
               "firms_per_sector": float(C.groupby("sector")["n_firms"].median().mean()),
               "top3_mean": float(100 * C["top3"].mean()),
               "within_sd_top3": within_sd(g, "top3_lag_pp"),
               "d_top3_current": gran.loc[m, "top3_last12"] - gran.loc[m, "top3_first12"],
               "R2_gap_rest_minus_top": float((cm["R2_rest"] - cm["R2_top"]).mean()),
               "share_top_below_rest": gran.loc[m, "share_top_below_rest"],
               "coef_R2": r["coef"], "p_perm_R2": r["p_perm"]}
        if fx is not None:
            rf = fx[(fx["market"] == m) & (fx["outcome"] == "R2")].iloc[0]
            row.update({"coef_R2_fixed": rf["coef"], "p_perm_R2_fixed": rf["p_perm"]})
        rows.append(row)
    H = pd.DataFrame(rows).sort_values("coef_R2")
    H.round(3).to_csv(os.path.join(OUT, "heterogeneity.csv"), index=False)
    print(H.round(2).to_string(index=False))
    for c in ["firms_per_sector", "top3_mean", "within_sd_top3", "d_top3_current", "R2_gap_rest_minus_top"]:
        print(f"corr(coef_R2, {c}) = {H['coef_R2'].corr(H[c]):.2f}   Spearman {H['coef_R2'].corr(H[c], method='spearman'):.2f}")
    print("mean coef developed vs emerging:", H.groupby("emerging")["coef_R2"].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
