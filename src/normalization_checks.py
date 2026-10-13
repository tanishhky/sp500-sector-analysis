"""
normalization_checks.py - What the directional connectedness measures record, and how much the
concentration result depends on how the variance decomposition is normalized.

  1. How much of connectedness is contemporaneous correlation: static and TVP measures from the
     covariance matrix alone (H = 1, no lag dynamics) against the baseline VAR(1), H = 10.
  2. Row normalization: the TO and FROM responses the generalized decomposition implies when one
     sector's correlations with every other sector fall, and the panel re-estimated under the
     maximum row-sum (scalar) normalization of Caloia, Cipollini and Muzzioli (2019).
  3. Panel inference: sector-specific linear trends, Driscoll-Kraay lags 3/6/12, no controls,
     leave-one-sector-out, and a permutation test that reassigns concentration series across
     sectors.
Run after econometrics.py and robust_panel.py. Writes output/v3/.
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
import yfinance as yf  # noqa: E402

from econometrics import (H, OUT, SECTORS, driscoll_kraay, load, ma_coefs,  # noqa: E402
                          static)

OUT3 = os.path.join(os.path.dirname(OUT), "v3")
NAMES = {"XLC": "Communication Services", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
         "XLE": "Energy", "XLF": "Financials", "XLV": "Health Care", "XLI": "Industrials",
         "XLK": "Information Technology", "XLB": "Materials", "XLRE": "Real Estate", "XLU": "Utilities"}
CONTROLS = ["log_share_lag", "logvol"]


def gfevd_raw(phi, sigma):
    """Generalized FEVD before any normalization (rows sum to more than one)."""
    num = sum((P @ sigma) ** 2 for P in phi) / np.diag(sigma)[None, :]
    den = sum(np.diag(P @ sigma @ P.T) for P in phi)
    return num / den[:, None]


def dir_measures(theta):
    """FROM, TO, NET in percent from an already-normalized matrix."""
    off = 100 * theta.copy()
    np.fill_diagonal(off, 0.0)
    frm, to = off.sum(axis=1), off.sum(axis=0)
    return frm, to, to - frm


def normalizations(raw):
    row = raw / raw.sum(axis=1, keepdims=True)
    maxrow = raw / raw.sum(axis=1).max()
    return {"row": row, "maxrow": maxrow}


def tvp_variants(y: pd.DataFrame, burn_until: str, k1=0.99, k2=0.99, horizon=H, prior_var=0.1):
    """Same filter as econometrics.tvp_var_connectedness (p = 1). On each date it records
    FROM/TO/NET under (a) row normalization, H = 10 [baseline]; (b) maximum row-sum
    normalization, H = 10; (c) row normalization of the covariance matrix alone, H = 1."""
    Y = y.values
    T, k = Y.shape
    m = k * (k + 1)
    beta, P = np.zeros(m), prior_var * np.eye(m)
    Sig = np.cov(Y[:max(30, 5 * k)].T)
    burn = pd.Timestamp(burn_until)
    rec = {f"{v}_{x}": [] for v in ("row", "maxrow", "sigma") for x in ("FROM", "TO", "NET")}
    dates = []
    for t in range(1, T):
        z = np.concatenate([[1.0], Y[t - 1]])
        Z = np.kron(np.eye(k), z[None, :])
        P_pred = P / k1
        u = Y[t] - Z @ beta
        Sig = k2 * Sig + (1 - k2) * np.outer(u, u)
        S = Z @ P_pred @ Z.T + Sig
        K = P_pred @ Z.T @ np.linalg.solve(S, np.eye(k))
        beta = beta + K @ u
        P = P_pred - K @ Z @ P_pred
        P = 0.5 * (P + P.T)
        if y.index[t] <= burn:
            continue
        B = beta.reshape(k, k + 1)[:, 1:]
        raw = gfevd_raw(ma_coefs([B], horizon), Sig)
        nz = normalizations(raw)
        raw0 = gfevd_raw([np.eye(k)], Sig)
        for v, th in (("row", nz["row"]), ("maxrow", nz["maxrow"]),
                      ("sigma", raw0 / raw0.sum(axis=1, keepdims=True))):
            f, to, n = dir_measures(th)
            rec[f"{v}_FROM"].append(f); rec[f"{v}_TO"].append(to); rec[f"{v}_NET"].append(n)
        dates.append(y.index[t])
    ix = pd.DatetimeIndex(dates)
    return {key: pd.DataFrame(v, index=ix, columns=y.columns) for key, v in rec.items()}


def monthly(tv, keys):
    parts = {}
    for key in keys:
        d = tv[key]
        parts[key] = d.groupby(d.index.to_period("M")).mean().stack().rename(key)
    M = pd.concat(parts, axis=1).reset_index()
    M.columns = ["month", "sector"] + list(keys)
    return M


def dk(pan, dep, xs, lag=3):
    tab, info = driscoll_kraay(pan, dep, xs, lag=lag, fe="twoway")
    return tab, info


def correlation_response(y: pd.DataFrame, eps=0.01):
    """Shrink sector j's correlations with every other sector by a factor (1 - eps) and record
    the change in its TO and FROM under row normalization of the H = 1 decomposition."""
    C = np.corrcoef(y.values.T)
    k = C.shape[0]

    def tf(Cm):
        raw = Cm ** 2
        th = raw / raw.sum(axis=1, keepdims=True)
        return dir_measures(th)

    f0, t0, n0 = tf(C)
    rows = []
    for j in range(k):
        Cj = C.copy()
        Cj[j, :] *= (1 - eps)
        Cj[:, j] *= (1 - eps)
        Cj[j, j] = 1.0
        f1, t1, n1 = tf(Cj)
        rows.append({"sector": y.columns[j], "dTO": t1[j] - t0[j], "dFROM": f1[j] - f0[j],
                     "dNET": n1[j] - n0[j], "FROM": f0[j], "TO": t0[j]})
    out = pd.DataFrame(rows).set_index("sector")
    out["TO_over_FROM"] = out["dTO"] / out["dFROM"]
    return out


def main():
    os.makedirs(OUT3, exist_ok=True)
    R = load("etf_returns.csv")[SECTORS]
    CR = load("const_returns.csv")[SECTORS]
    summary = {}

    # ---- 1a. static: covariance-only vs VAR(1), H = 10
    st = {}
    for label, y in (("ETF", R), ("Constituent", CR)):
        _, mm = static(y, 1, H)
        raw0 = gfevd_raw([np.eye(len(SECTORS))], np.cov(y.values.T))
        f0, t0, n0 = dir_measures(raw0 / raw0.sum(axis=1, keepdims=True))
        st[label] = {"corr_NET": float(np.corrcoef(mm["NET"], n0)[0, 1]),
                     "max_abs_diff_NET": float(np.max(np.abs(mm["NET"] - n0))),
                     "TCI_var": float(mm["TCI"]), "TCI_sigma": float(f0.mean())}
        pd.DataFrame({"NET_VAR1_H10": mm["NET"], "NET_sigma_only": n0, "TO_VAR1_H10": mm["TO"],
                      "TO_sigma_only": t0, "FROM_VAR1_H10": mm["FROM"], "FROM_sigma_only": f0},
                     index=SECTORS).round(3).to_csv(os.path.join(OUT3, f"static_sigma_only_{label}.csv"))
    summary["static_sigma_only"] = st

    # ---- 2a. row-normalization arithmetic
    cr = correlation_response(R)
    cr.round(4).to_csv(os.path.join(OUT3, "correlation_response.csv"))
    summary["correlation_response"] = {"median_TO_over_FROM": float(cr["TO_over_FROM"].median()),
                                       "min": float(cr["TO_over_FROM"].min()),
                                       "max": float(cr["TO_over_FROM"].max()),
                                       "FROM_range": [float(cr["FROM"].min()), float(cr["FROM"].max())],
                                       "TO_range": [float(cr["TO"].min()), float(cr["TO"].max())]}

    # ---- TVP variants on both return series (training window June to September 2018 from the funds)
    raw = yf.download(list(NAMES), start="2018-06-20", end="2018-10-01", auto_adjust=True, progress=False)
    train = (100 * np.log(raw["Close"]).diff()).rename(columns=NAMES)[SECTORS].dropna()
    base = pd.read_csv(os.path.join(OUT, "panel_monthly_ret.csv"))
    base["month"] = pd.PeriodIndex(base["month"], freq="M")
    cols = ["sector", "month", "hhi_lag_pp", "top3_lag_pp", "log_share_lag", "logvol"]
    keys = [f"{v}_{x}" for v in ("row", "maxrow", "sigma") for x in ("FROM", "TO", "NET")]
    panels = {}
    for label, y in (("ETF", R), ("Constituent", CR)):
        yy = pd.concat([train, y])
        yy = yy[~yy.index.duplicated()]
        tv = tvp_variants(yy, "2018-09-28")
        M = monthly(tv, keys)
        pan = M.merge(base[cols], on=["sector", "month"])
        pan = pan[pan["month"] >= pd.Period("2018-11", "M")]
        panels[label] = pan
        pan.to_csv(os.path.join(OUT3, f"panel_variants_{label}.csv"), index=False)
        if label == "ETF":
            saved = pd.read_csv(os.path.join(OUT, "tvp_ret_NET.csv"), index_col=0, parse_dates=True)[SECTORS]
            common = saved.index.intersection(tv["row_NET"].index)
            summary["reproduction_corr_daily_NET"] = float(np.corrcoef(
                saved.loc[common].values.ravel(), tv["row_NET"].loc[common].values.ravel())[0, 1])
        a, b = tv["row_NET"].values.ravel(), tv["sigma_NET"].values.ravel()
        c = tv["maxrow_NET"].values.ravel()
        summary[f"daily_corr_{label}"] = {"row_vs_sigma_NET": float(np.corrcoef(a, b)[0, 1]),
                                          "row_vs_maxrow_NET": float(np.corrcoef(a, c)[0, 1])}

    # ---- 2b / 1b. panel under each variant
    rows = []
    for label, pan in panels.items():
        for v in ("row", "maxrow", "sigma"):
            for x in ("FROM", "TO", "NET"):
                for c in ("top3_lag_pp", "hhi_lag_pp"):
                    tab, info = dk(pan, f"{v}_{x}", [c] + CONTROLS)
                    rows.append({"returns": label, "variant": v, "dep": x, "conc": c,
                                 **tab.loc[c].to_dict(), **info})
    var_tab = pd.DataFrame(rows)
    var_tab.round(4).to_csv(os.path.join(OUT3, "panel_variants.csv"), index=False)

    # ---- 3. inference checks on the baseline (row, H = 10) NET and TO, top-three share
    checks = []
    for label, pan in panels.items():
        pan = pan.sort_values(["sector", "month"]).copy()
        months = sorted(pan["month"].unique())
        mi = {m: i for i, m in enumerate(months)}
        pan["t"] = pan["month"].map(mi).astype(float)
        trends = []
        for s in SECTORS[1:]:
            col = "trend_" + s.split()[0]
            pan[col] = pan["t"] * (pan["sector"] == s)
            trends.append(col)
        for dep in ("row_NET", "row_TO", "row_FROM"):
            for lag in (3, 6, 12):
                tab, info = dk(pan, dep, ["top3_lag_pp"] + CONTROLS, lag=lag)
                checks.append({"returns": label, "dep": dep, "spec": f"baseline, DK lag {lag}",
                               **tab.loc["top3_lag_pp"].to_dict()})
            tab, _ = dk(pan, dep, ["top3_lag_pp"])
            checks.append({"returns": label, "dep": dep, "spec": "no controls",
                           **tab.loc["top3_lag_pp"].to_dict()})
            tab, _ = dk(pan, dep, ["top3_lag_pp"] + CONTROLS + trends)
            checks.append({"returns": label, "dep": dep, "spec": "sector-specific linear trends",
                           **tab.loc["top3_lag_pp"].to_dict()})
            for s in SECTORS:
                tab, _ = dk(pan[pan["sector"] != s], dep, ["top3_lag_pp"] + CONTROLS)
                checks.append({"returns": label, "dep": dep, "spec": f"drop {s}",
                               **tab.loc["top3_lag_pp"].to_dict()})
    chk = pd.DataFrame(checks)
    chk.round(4).to_csv(os.path.join(OUT3, "inference_checks.csv"), index=False)

    # permutation test: reassign each sector's concentration path to another sector
    rng = np.random.default_rng(20261013)
    perm_out = {}
    for label, pan in panels.items():
        pan = pan.copy()
        obs = {}
        for dep in ("row_NET", "row_TO"):
            tab, _ = dk(pan, dep, ["top3_lag_pp"] + CONTROLS)
            obs[dep] = float(tab.loc["top3_lag_pp", "t"])
        conc = pan.pivot(index="month", columns="sector", values="top3_lag_pp")
        draws = {dep: [] for dep in obs}
        for _ in range(2000):
            perm = rng.permutation(SECTORS)
            while (perm == np.array(SECTORS)).all():
                perm = rng.permutation(SECTORS)
            mp = dict(zip(SECTORS, perm))
            p2 = pan.copy()
            p2["top3_perm"] = [conc.at[m, mp[s]] for m, s in zip(p2["month"], p2["sector"])]
            for dep in obs:
                tab, _ = dk(p2, dep, ["top3_perm"] + CONTROLS)
                draws[dep].append(float(tab.loc["top3_perm", "t"]))
        perm_out[label] = {dep: {"t_obs": obs[dep],
                                 "p_two_sided": float(np.mean(np.abs(draws[dep]) >= abs(obs[dep]))),
                                 "p_one_sided_neg": float(np.mean(np.array(draws[dep]) <= obs[dep])),
                                 "perm_t_5pct": float(np.percentile(draws[dep], 5)),
                                 "perm_t_95pct": float(np.percentile(draws[dep], 95))}
                           for dep in obs}
    summary["permutation"] = perm_out

    with open(os.path.join(OUT3, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(var_tab[var_tab["conc"] == "top3_lag_pp"][["returns", "variant", "dep", "coef", "se", "t"]]
          .round(3).to_string(index=False))
    print(chk[~chk["spec"].str.startswith("drop")][["returns", "dep", "spec", "coef", "se", "t"]]
          .round(3).to_string(index=False))
    d = chk[chk["spec"].str.startswith("drop") & (chk["dep"] == "row_NET")]
    print(d[["returns", "spec", "coef", "t"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
