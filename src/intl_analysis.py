"""
intl_analysis.py - The concentration and co-movement tests of the paper, market by market and
pooled across markets, on the panels from intl_panel.py.

For each market, sectors with a median of at least five firms enter. Outcomes are measured afresh
each month from that month's daily returns: the R2 of a sector's return on the equal-weighted
return of the market's other sectors, and its average squared correlation with them. Regressors
are the previous month-end top-three share, log sector share of the market, and the log of the
sector's realized volatility in the month. Fixed effects are absorbed by alternating projections,
so unbalanced panels are handled exactly.

Inference: Driscoll-Kraay (three lags); wild cluster restricted bootstrap by sector with Webb
weights; and a permutation test that reassigns concentration paths across sectors within a market.
The pooled panel uses market-by-sector and market-by-month fixed effects, standard errors
clustered by market-sector, and the same within-market permutation.

The granular decomposition and the fixed-weight counterfactual are repeated in every market.
Writes output/intl/.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAN = os.path.join(BASE, "data", "intl", "panels")
FIXED = "--fixed" in sys.argv          # concentration among firms trading on the first sample day
CONC_COL = "top3_fixed" if FIXED else "top3"
OUT = os.path.join(BASE, "output", "intl_fixed" if FIXED else "intl")
RNG = np.random.default_rng(20261013)
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
CTRL = ["log_share_lag", "logvol"]
MIN_FIRMS = 5
N_PERM = 500 if FIXED else 1000


def slug(m):
    return m.lower().replace(" ", "_")


# ------------------------------------------------------------------ data
def load_market(market):
    t = slug(market)
    R = pd.read_csv(os.path.join(PAN, f"{t}_returns.csv"), index_col=0, parse_dates=True)
    T = pd.read_csv(os.path.join(PAN, f"{t}_top3.csv"), index_col=0, parse_dates=True)
    S = pd.read_csv(os.path.join(PAN, f"{t}_rest.csv"), index_col=0, parse_dates=True)
    C = pd.read_csv(os.path.join(PAN, f"{t}_concentration.csv"), parse_dates=["date"])
    keep = C.groupby("sector")["n_firms"].median()
    keep = sorted(keep[keep >= MIN_FIRMS].index)
    return R[keep], T[keep], S[keep], C[C["sector"].isin(keep)], keep


def monthly_panel(R, C, sectors):
    recs = []
    for p, g in R.groupby(R.index.to_period("M")):
        g = g.dropna(how="all")
        if len(g) < 15:
            continue
        for s in sectors:
            others = g.drop(columns=s).mean(axis=1)
            x = pd.concat([g[s], others], axis=1).dropna()
            if len(x) < 15:
                continue
            rho2 = [g[[s, o]].dropna().corr().iloc[0, 1] ** 2 for o in sectors if o != s]
            recs.append({"sector": s, "month": p, "R2": 100 * x.iloc[:, 0].corr(x.iloc[:, 1]) ** 2,
                         "rho2": 100 * np.nanmean(rho2),
                         "logvol": np.log(100 * np.sqrt(252) * x.iloc[:, 0].std())})
    M = pd.DataFrame(recs)
    C = C.copy()
    C["month"] = C["date"].dt.to_period("M")
    cm = C.sort_values("date").groupby(["sector", "month"]).last().reset_index().sort_values(["sector", "month"])
    cm["top3_lag_pp"] = 100 * cm.groupby("sector")[CONC_COL].shift(1)
    cm["log_share_lag"] = np.log(cm.groupby("sector")["sector_share"].shift(1))
    pan = M.merge(cm[["sector", "month", "top3_lag_pp", "log_share_lag"]], on=["sector", "month"])
    return pan.replace([np.inf, -np.inf], np.nan).dropna()


# ------------------------------------------------------------------ estimation
def demean(df, cols, g1, g2, tol=1e-9, maxit=1000):
    X = df[cols].astype(float).copy()
    for _ in range(maxit):
        X1 = X - X.groupby(df[g1].values).transform("mean")
        X2 = X1 - X1.groupby(df[g2].values).transform("mean")
        if (X2 - X).abs().max().max() < tol:
            return X2
        X = X2
    return X


def fit(df, y, xs, g1, g2):
    D = demean(df, [y] + xs, g1, g2)
    X, Y = D[xs].values, D[y].values
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ Y
    return b, Y - X @ b, X, XtX_inv


def se_dk(X, e, XtX_inv, time, lag=3):
    St = pd.DataFrame(X * e[:, None]).groupby(np.asarray(time)).sum().sort_index().values
    S = St.T @ St
    for l in range(1, lag + 1):
        G = St[l:].T @ St[:-l]
        S += (1 - l / (lag + 1)) * (G + G.T)
    return np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))


def se_cluster(X, e, XtX_inv, cl):
    cl = np.asarray(cl)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(cl):
        s = X[cl == g].T @ e[cl == g]
        meat += np.outer(s, s)
    G, (N, K) = len(np.unique(cl)), X.shape
    return np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv * G / (G - 1) * (N - 1) / (N - K)))


def wild_p(df, y, xs, g1, g2, cl, B=1999):
    b, e, X, XtX_inv = fit(df, y, xs, g1, g2)
    t_obs = b[0] / se_cluster(X, e, XtX_inv, df[cl])[0]
    D = demean(df, [y] + xs, g1, g2)
    Xr, Y = D[xs[1:]].values, D[y].values
    br = np.linalg.lstsq(Xr, Y, rcond=None)[0]
    fit_r, u_r = Xr @ br, Y - Xr @ br
    groups = df[cl].values
    ug = np.unique(groups)
    gi = np.searchsorted(ug, groups)
    Xf = D[xs].values
    tb = np.empty(B)
    for k in range(B):
        ys = fit_r + u_r * RNG.choice(WEBB, size=len(ug))[gi]
        bb = XtX_inv @ Xf.T @ ys
        tb[k] = bb[0] / se_cluster(Xf, ys - Xf @ bb, XtX_inv, groups)[0]
    return float(t_obs), float(np.mean(np.abs(tb) >= abs(t_obs)))


def permute_within(df, conc, unit, within, period):
    """Reassign each unit's concentration path to another unit of the same market."""
    out = df[conc].copy()
    for _, g in df.groupby(within):
        units = np.array(sorted(g[unit].unique()))
        if len(units) < 3:
            continue
        perm = RNG.permutation(units)
        while (perm == units).all():
            perm = RNG.permutation(units)
        mp = dict(zip(units, perm))
        wide = g.pivot(index=period, columns=unit, values=conc)
        out.loc[g.index] = [wide.at[p, mp[u]] if mp[u] in wide.columns and p in wide.index else np.nan
                            for p, u in zip(g[period], g[unit])]
    return out


# ------------------------------------------------------------------ per market
def market_tests(market, pan):
    rows = []
    pan = pan.copy()
    pan["market"] = market
    for y in ("R2", "rho2"):
        xs = ["top3_lag_pp"] + CTRL
        b, e, X, XtX_inv = fit(pan, y, xs, "sector", "month")
        t_dk = b[0] / se_dk(X, e, XtX_inv, pan["month"])[0]
        t_cl, p_wild = wild_p(pan, y, xs, "sector", "month", "sector")
        perm_t = []
        for _ in range(N_PERM):
            q = pan.copy()
            q["c"] = permute_within(q, "top3_lag_pp", "sector", "market", "month")
            q = q.dropna(subset=["c"])
            bb, ee, XX, XXi = fit(q, y, ["c"] + CTRL, "sector", "month")
            perm_t.append(bb[0] / se_dk(XX, ee, XXi, q["month"])[0])
        perm_t = np.array(perm_t)
        ac = pan.sort_values("month").groupby("sector")[y].apply(lambda s: s.autocorr(1)).mean()
        rows.append({"market": market, "outcome": y, "coef": b[0], "t_DK": t_dk, "t_cluster": t_cl,
                     "p_wild": p_wild, "p_perm": float(np.mean(np.abs(perm_t) >= abs(t_dk))),
                     "N": len(pan), "sectors": pan["sector"].nunique(), "months": pan["month"].nunique(),
                     "AR1": ac, "mean_y": pan[y].mean()})
    return rows


def granular(market, R, T, S, C):
    recs = []
    for s in R.columns:
        mkt = R.drop(columns=s).mean(axis=1)
        df = pd.DataFrame({"sector": R[s], "top": T[s], "rest": S[s], "mkt": mkt}).dropna()
        for yr, g in df.groupby(df.index.year):
            if len(g) < 60:
                continue
            recs.append({"market": market, "sector": s, "year": yr,
                         "R2_top": 100 * g["top"].corr(g["mkt"]) ** 2,
                         "R2_rest": 100 * g["rest"].corr(g["mkt"]) ** 2,
                         "vol_top": g["top"].std(), "vol_rest": g["rest"].std()})
    comp = pd.DataFrame(recs)
    W = C.pivot(index="date", columns="sector", values="W_top_lag").reindex(R.index)
    W0 = W.loc["2018-11-06":"2018-12-31"].mean()
    CF = T * W0 + S * (1 - W0)

    def avg_rho2(Y):
        Cm = Y.dropna().corr().values
        k = Cm.shape[0]
        return float(100 * ((Cm ** 2).sum() - k) / (k * (k - 1)))

    first, last = slice("2018-10-01", "2019-09-30"), slice("2025-09-01", "2026-08-31")
    top3 = C.pivot(index="date", columns="sector", values="top3")
    return comp, {"market": market,
                  "share_top_below_rest": float((comp["R2_top"] < comp["R2_rest"]).mean()),
                  "sector_years": int(len(comp)),
                  "share_top_more_volatile": float((comp["vol_top"] > comp["vol_rest"]).mean()),
                  "avg_rho2_first12": avg_rho2(R.loc[first]), "avg_rho2_last12": avg_rho2(R.loc[last]),
                  "avg_rho2_last12_cf": avg_rho2(CF.loc[last]),
                  "top3_first12": float(100 * top3.loc[first].mean().mean()),
                  "top3_last12": float(100 * top3.loc[last].mean().mean())}


# ------------------------------------------------------------------ main
def main():
    os.makedirs(OUT, exist_ok=True)
    summ = json.load(open(os.path.join(PAN, "build_summary.json")))
    markets = [s["market"] for s in summ]
    results, grans, comps, pans = [], [], [], []
    for m in markets:
        R, T, S, C, sectors = load_market(m)
        if len(sectors) < 5:
            print(m, "skipped: fewer than five sectors with five firms")
            continue
        pan = monthly_panel(R, C, sectors)
        pan["market"] = m
        pans.append(pan)
        results += market_tests(m, pan)
        comp, g = granular(m, R, T, S, C)
        comps.append(comp)
        grans.append(g)
        print(m, [(r["outcome"], round(r["coef"], 3), round(r["t_DK"], 2), round(r["p_perm"], 3)) for r in results[-2:]],
              round(g["share_top_below_rest"], 2), flush=True)
    res = pd.DataFrame(results)
    res.round(4).to_csv(os.path.join(OUT, "market_results.csv"), index=False)
    pd.DataFrame(grans).round(3).to_csv(os.path.join(OUT, "granular_by_market.csv"), index=False)
    pd.concat(comps).round(3).to_csv(os.path.join(OUT, "granular_components.csv"), index=False)

    # pooled: all non-U.S. markets, and all markets including the current-membership U.S. panel
    P = pd.concat(pans, ignore_index=True)
    P["unit"] = P["market"] + "|" + P["sector"]
    P["mm"] = P["market"] + "|" + P["month"].astype(str)
    P.to_csv(os.path.join(OUT, "pooled_panel.csv"), index=False)
    pooled = {}
    for label, Q in (("international", P[P["market"] != "United States"]), ("all", P)):
        Q = Q.reset_index(drop=True)
        for y in ("R2", "rho2"):
            xs = ["top3_lag_pp"] + CTRL
            b, e, X, XtX_inv = fit(Q, y, xs, "unit", "mm")
            t_cl = b[0] / se_cluster(X, e, XtX_inv, Q["unit"])[0]
            t_dk = b[0] / se_dk(X, e, XtX_inv, Q["month"].astype(str))[0]
            perm_t = []
            for _ in range(N_PERM):
                q = Q.copy()
                q["c"] = permute_within(q, "top3_lag_pp", "sector", "market", "month")
                q = q.dropna(subset=["c"]).reset_index(drop=True)
                bb, ee, XX, XXi = fit(q, y, ["c"] + CTRL, "unit", "mm")
                perm_t.append(bb[0] / se_cluster(XX, ee, XXi, q["unit"])[0])
            perm_t = np.array(perm_t)
            pooled[f"{label}_{y}"] = {"coef": float(b[0]), "t_cluster_unit": float(t_cl), "t_DK_month": float(t_dk),
                                      "p_perm_within_market": float(np.mean(np.abs(perm_t) >= abs(t_cl))),
                                      "N": int(len(Q)), "units": int(Q["unit"].nunique()),
                                      "markets": int(Q["market"].nunique())}
    json.dump(pooled, open(os.path.join(OUT, "pooled.json"), "w"), indent=2)
    print(res.round(3).to_string(index=False))
    print(json.dumps(pooled, indent=2))
    print(pd.DataFrame(grans).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
