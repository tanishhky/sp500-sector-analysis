"""
inference_checks.py - How much of the concentration result survives inference that allows for
slow-moving, trending regressors (v3 revision, October 2026).

Monthly averages of daily TVP-VAR measures inherit the filter's memory (forgetting factor 0.99,
an effective window of about 100 days), and within-sector concentration moves almost like a
sector-specific trend. Driscoll-Kraay errors with a few lags are not built for that. This script
  1. re-estimates the concentration effect on outcomes measured afresh in each period from that
     period's daily returns: monthly R2 against the rest of the market and quarterly average
     squared correlation with the other ten sectors;
  2. reports, for those outcomes and for the TVP measures, Driscoll-Kraay t, a sector-clustered
     wild bootstrap p (Webb weights, null imposed), and a permutation p that reassigns each
     sector's concentration path to another sector;
  3. compares long differences across the eleven sectors (last twelve months minus first twelve).
Writes output/v3/.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from econometrics import OUT, PANEL, SECTORS, driscoll_kraay, load, two_way_demean  # noqa: E402

OUT3 = os.path.join(os.path.dirname(OUT), "v3")
CONTROLS = ["log_share_lag", "logvol"]
RNG = np.random.default_rng(20261013)
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


# ------------------------------------------------------------------ outcomes
def period_outcomes(R: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Per sector and period: R2 on the equal-weighted return of the other ten sectors, and the
    average squared correlation with each of the other ten sectors."""
    recs = []
    per = R.index.to_period(freq)
    for p, g in R.groupby(per):
        if len(g) < 15:
            continue
        C = g.corr().values
        for i, s in enumerate(SECTORS):
            others = g.drop(columns=s).mean(axis=1)
            rho2 = np.delete(C[i] ** 2, i).mean()
            recs.append({"sector": s, "period": p, "R2": 100 * g[s].corr(others) ** 2,
                         "rho2": 100 * rho2})
    return pd.DataFrame(recs)


def conc_controls(freq: str) -> pd.DataFrame:
    conc = pd.read_csv(os.path.join(PANEL, "concentration.csv"), parse_dates=["date"])
    conc["period"] = conc["date"].dt.to_period(freq)
    cm = conc.sort_values("date").groupby(["sector", "period"]).last().reset_index()
    cm = cm.sort_values(["sector", "period"])
    for c in ("top3", "hhi", "sector_share"):
        cm[c + "_lag"] = cm.groupby("sector")[c].shift(1)
    cm["top3_lag_pp"] = 100 * cm["top3_lag"]
    cm["hhi_lag_pp"] = 100 * cm["hhi_lag"]
    cm["log_share_lag"] = np.log(cm["sector_share_lag"])
    LV = load("etf_logvol.csv")[SECTORS]
    lv = LV.groupby(LV.index.to_period(freq)).mean().stack().rename("logvol").reset_index()
    lv.columns = ["period", "sector", "logvol"]
    return cm[["sector", "period", "top3_lag_pp", "hhi_lag_pp", "log_share_lag"]].merge(lv, on=["period", "sector"])


# ------------------------------------------------------------------ inference
def demeaned(pan, y, xs):
    d = pan.dropna(subset=[y] + xs).copy()
    d["month"] = d["period"]
    d = two_way_demean(d, [y] + xs)
    return d


def ols_cluster(d, y, xs):
    X, Y = d[xs].values, d[y].values
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ Y
    e = Y - X @ b
    G = d["sector"].nunique()
    meat = np.zeros((len(xs), len(xs)))
    for _, ix in d.groupby("sector").indices.items():
        s = X[ix].T @ e[ix]
        meat += np.outer(s, s)
    N, K = X.shape
    V = XtX_inv @ meat @ XtX_inv * G / (G - 1) * (N - 1) / (N - K)
    return b, np.sqrt(np.diag(V))


def wild_cluster_p(d, y, xs, B=4999):
    """Wild cluster restricted bootstrap (Webb weights) for the first regressor."""
    b, se = ols_cluster(d, y, xs)
    t_obs = b[0] / se[0]
    Xr, Y = d[xs[1:]].values, d[y].values
    br = np.linalg.lstsq(Xr, Y, rcond=None)[0]
    fit_r, u_r = Xr @ br, Y - Xr @ br
    groups = d["sector"].values
    secs = np.unique(groups)
    gi = np.searchsorted(secs, groups)
    tb = np.empty(B)
    dd = d.copy()
    for r in range(B):
        v = RNG.choice(WEBB, size=len(secs))
        dd[y] = fit_r + u_r * v[gi]
        bb, ss = ols_cluster(dd, y, xs)
        tb[r] = bb[0] / ss[0]
    return float(t_obs), float(np.mean(np.abs(tb) >= abs(t_obs)))


def permutation_p(pan, y, conc, controls, n=2000):
    tab, _ = driscoll_kraay(pan.assign(month=pan["period"]), y, [conc] + controls, lag=3, fe="twoway")
    t_obs = float(tab.loc[conc, "t"])
    wide = pan.pivot(index="period", columns="sector", values=conc)
    draws = []
    secs = np.array(SECTORS)
    for _ in range(n):
        perm = RNG.permutation(secs)
        while (perm == secs).all():
            perm = RNG.permutation(secs)
        mp = dict(zip(secs, perm))
        p2 = pan.copy()
        p2["_c"] = [wide.at[p, mp[s]] for p, s in zip(p2["period"], p2["sector"])]
        tb, _ = driscoll_kraay(p2.assign(month=p2["period"]), y, ["_c"] + controls, lag=3, fe="twoway")
        draws.append(float(tb.loc["_c", "t"]))
    draws = np.array(draws)
    return t_obs, float(np.mean(np.abs(draws) >= abs(t_obs))), float(np.percentile(np.abs(draws), 95))


def run_all(pan, y, label, rows, conc="top3_lag_pp"):
    tab, info = driscoll_kraay(pan.assign(month=pan["period"]), y, [conc] + CONTROLS, lag=3, fe="twoway")
    d = demeaned(pan, y, [conc] + CONTROLS)
    t_cl, p_wild = wild_cluster_p(d, y, [conc] + CONTROLS)
    t_dk, p_perm, perm_abs95 = permutation_p(pan, y, conc, CONTROLS)
    # first-order autocorrelation of the within-sector outcome, as a persistence gauge
    ac = pan.sort_values("period").groupby("sector")[y].apply(lambda s: s.autocorr(1)).mean()
    rows.append({"outcome": label, "coef": float(tab.loc[conc, "coef"]), "t_DK3": t_dk,
                 "t_cluster": t_cl, "p_wild_cluster": p_wild, "p_permutation": p_perm,
                 "perm_|t|_95pct": perm_abs95, "mean_AR1_outcome": float(ac), "N": info["N"]})


def main():
    os.makedirs(OUT3, exist_ok=True)
    rows = []
    # 1. TVP measures (baseline and variants), monthly, from referee_checks.py output
    for label in ("ETF", "Constituent"):
        pv = pd.read_csv(os.path.join(OUT3, f"panel_variants_{label}.csv"))
        pv["period"] = pd.PeriodIndex(pv["month"], freq="M")
        for dep in ("row_NET", "row_TO", "maxrow_TO", "maxrow_FROM"):
            run_all(pv, dep, f"{label}: TVP {dep} (monthly)", rows)
    # 2. fresh monthly and quarterly outcomes
    for label, fn in (("ETF", "etf_returns.csv"), ("Constituent", "const_returns.csv")):
        R = load(fn)[SECTORS]
        for freq in ("M", "Q"):
            out = period_outcomes(R, freq)
            pan = out.merge(conc_controls(freq), on=["sector", "period"])
            start = pd.Period("2018-11", "M") if freq == "M" else pd.Period("2019Q1", "Q")
            pan = pan[pan["period"] >= start]
            for dep in ("R2", "rho2"):
                run_all(pan, dep, f"{label}: {dep} ({'monthly' if freq == 'M' else 'quarterly'})", rows)
    res = pd.DataFrame(rows)
    res.round(4).to_csv(os.path.join(OUT3, "inference_trending.csv"), index=False)
    print(res.round(3).to_string(index=False))

    # 3. long differences across sectors: last 12 months minus first 12 months
    conc = pd.read_csv(os.path.join(PANEL, "concentration.csv"), parse_dates=["date"])
    first = (conc["date"] >= "2018-10-01") & (conc["date"] < "2019-10-01")
    last = conc["date"] >= "2025-09-01"
    dtop3 = 100 * (conc[last].groupby("sector")["top3"].mean() - conc[first].groupby("sector")["top3"].mean())
    ld = {"d_top3": dtop3}
    for label, fn in (("ETF", "etf_returns.csv"), ("Constituent", "const_returns.csv")):
        R = load(fn)[SECTORS]
        o = period_outcomes(R, "M")
        o["ts"] = o["period"].dt.to_timestamp()
        f_ = o[(o["ts"] >= "2018-10-01") & (o["ts"] < "2019-10-01")].groupby("sector")[["R2", "rho2"]].mean()
        l_ = o[o["ts"] >= "2025-09-01"].groupby("sector")[["R2", "rho2"]].mean()
        for c in ("R2", "rho2"):
            ld[f"d_{c}_{label}"] = l_[c] - f_[c]
        pv = pd.read_csv(os.path.join(OUT3, f"panel_variants_{label}.csv"))
        pv["ts"] = pd.PeriodIndex(pv["month"], freq="M").to_timestamp()
        f_ = pv[(pv["ts"] >= "2018-11-01") & (pv["ts"] < "2019-11-01")].groupby("sector")[["row_NET", "maxrow_TO"]].mean()
        l_ = pv[pv["ts"] >= "2025-09-01"].groupby("sector")[["row_NET", "maxrow_TO"]].mean()
        for c in ("row_NET", "maxrow_TO"):
            ld[f"d_{c}_{label}"] = l_[c] - f_[c]
    ld = pd.DataFrame(ld)
    ld_dm = ld - ld.mean()                       # relative to the cross-sector average change
    summ = {}
    for c in ld.columns[1:]:
        x, y = ld_dm["d_top3"].values, ld_dm[c].values
        r = float(np.corrcoef(x, y)[0, 1])
        slope = float(np.polyfit(x, y, 1)[0])
        perm = np.array([np.corrcoef(x, RNG.permutation(y))[0, 1] for _ in range(20000)])
        summ[c] = {"corr": r, "slope": slope, "perm_p_two_sided": float(np.mean(np.abs(perm) >= abs(r)))}
    ld.round(2).to_csv(os.path.join(OUT3, "long_differences.csv"))
    print(ld.round(2).to_string())
    print(json.dumps(summ, indent=2))
    with open(os.path.join(OUT3, "long_differences.json"), "w") as f:
        json.dump(summ, f, indent=2)


if __name__ == "__main__":
    main()
