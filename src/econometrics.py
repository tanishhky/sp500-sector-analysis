"""
econometrics.py - Connectedness, predictability, and the concentration link.

  1. Static Diebold-Yilmaz (2012) connectedness, returns and log range volatility,
     with a robustness grid (lag order, horizon, ETF vs constituent-built series).
  2. TVP-VAR connectedness (Antonakakis, Chatziantoniou and Gabauer 2020): Kalman filter
     with forgetting factors (Koop and Korobilis 2013). Filtered, so each date uses only
     data available on that date.
  3. Granger causality conditional on the full system, heteroskedasticity-robust Wald
     tests, Benjamini-Hochberg FDR; full sample and two halves.
  4. Out-of-sample predictability of next-day sector returns (Campbell-Thompson R2_OS,
     Clark-West test) from lagged cross-sector returns.
  5. Panel regressions of monthly connectedness on lagged within-sector concentration,
     two-way fixed effects, Driscoll-Kraay standard errors; and the index-level link.

Run: python src/econometrics.py      (writes output/v2/)
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import yfinance as yf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.api import VAR

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(BASE, "data", "panel")
OUT = os.path.join(BASE, "output", "v2")
H = 10
SECTORS = ["Communication Services", "Consumer Discretionary", "Consumer Staples", "Energy", "Financials",
           "Health Care", "Industrials", "Information Technology", "Materials", "Real Estate", "Utilities"]
SHORT = {"Communication Services": "COM", "Consumer Discretionary": "CD", "Consumer Staples": "CS",
         "Energy": "ENE", "Financials": "FIN", "Health Care": "HC", "Industrials": "IND",
         "Information Technology": "IT", "Materials": "MAT", "Real Estate": "RE", "Utilities": "UTL"}


def load(name):
    return pd.read_csv(os.path.join(PANEL, name), index_col=0, parse_dates=True)


# ---------------------------------------------------------------- connectedness core
def ma_coefs(B_list, horizon):
    """MA coefficients Phi_0..Phi_{H-1} from lag matrices B_1..B_p (each k x k)."""
    k, p = B_list[0].shape[0], len(B_list)
    phi = [np.eye(k)]
    for h in range(1, horizon):
        acc = np.zeros((k, k))
        for l in range(1, min(h, p) + 1):
            acc += B_list[l - 1] @ phi[h - l]
        phi.append(acc)
    return phi


def gfevd(phi, sigma):
    """Generalized FEVD (Pesaran-Shin), row-normalized: theta[i, j] = share of i's
    forecast-error variance due to shocks in j."""
    num = sum((P @ sigma) ** 2 for P in phi) / np.diag(sigma)[None, :]
    den = sum(np.diag(P @ sigma @ P.T) for P in phi)
    theta = num / den[:, None]
    return theta / theta.sum(axis=1, keepdims=True)


def measures(theta):
    k = theta.shape[0]
    off = theta * 100
    np.fill_diagonal(off, 0.0)
    frm, to = off.sum(axis=1), off.sum(axis=0)
    return {"TCI": off.sum() / k, "FROM": frm, "TO": to, "NET": to - frm, "OWN": 100 * np.diag(theta)}


def static(y, p, horizon=H):
    res = VAR(y.values).fit(p)
    B = [res.coefs[i] for i in range(p)]
    th = gfevd(ma_coefs(B, horizon), np.asarray(res.sigma_u))
    return th, measures(th)


# ---------------------------------------------------------------- TVP-VAR
def tvp_var_connectedness(y: pd.DataFrame, burn_until: str, p: int = 1, k1=0.99, k2=0.99,
                          horizon=H, prior_var=0.1):
    """Kalman filter with forgetting factors (Koop-Korobilis 2013 as used by
    Antonakakis-Chatziantoniou-Gabauer 2020). Returns dict of DataFrames indexed by date
    (after burn-in) and the per-date pairwise matrices."""
    Y = y.values
    T, k = Y.shape
    m = k * (k * p + 1)                    # intercept included: log volatility has a large mean
    beta = np.zeros(m)
    P = prior_var * np.eye(m)
    Sig = np.cov(Y[:max(30, 5 * k)].T)
    dates, tci, frm, to, net, own, pair = [], [], [], [], [], [], []
    burn = pd.Timestamp(burn_until)
    for t in range(p, T):
        z = np.concatenate([[1.0]] + [Y[t - l] for l in range(1, p + 1)])      # (1 + k p,)
        Z = np.kron(np.eye(k), z[None, :])                            # (k, m)
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
        Bm = beta.reshape(k, k * p + 1)[:, 1:]
        B = [Bm[:, l * k:(l + 1) * k] for l in range(p)]
        th = gfevd(ma_coefs(B, horizon), Sig)
        mm = measures(th)
        dates.append(y.index[t]); tci.append(mm["TCI"]); frm.append(mm["FROM"]); to.append(mm["TO"])
        net.append(mm["NET"]); own.append(mm["OWN"]); pair.append(th * 100)
    ix = pd.DatetimeIndex(dates)
    cols = list(y.columns)
    return {"TCI": pd.Series(tci, index=ix, name="TCI"),
            "FROM": pd.DataFrame(frm, index=ix, columns=cols),
            "TO": pd.DataFrame(to, index=ix, columns=cols),
            "NET": pd.DataFrame(net, index=ix, columns=cols),
            "OWN": pd.DataFrame(own, index=ix, columns=cols),
            "PAIR": np.array(pair)}


def rolling_tci(y, window=200, p=1, horizon=H):
    vals, ix = [], []
    for end in range(window, len(y) + 1):
        _, mm = static(y.iloc[end - window:end], p, horizon)
        vals.append(mm["TCI"]); ix.append(y.index[end - 1])
    return pd.Series(vals, index=ix, name="rolling_TCI")


# ---------------------------------------------------------------- Granger (system, HC3)
def system_granger(y: pd.DataFrame, p: int):
    """Wald test that lags of j do not enter equation i, conditional on all other lags.
    HC3 covariance. Returns DataFrame of p-values (row = cause j, column = effect i)."""
    Y = y.values
    T, k = Y.shape
    X = np.column_stack([np.ones(T - p)] + [Y[p - l:T - l] for l in range(1, p + 1)])  # (T-p, 1+kp)
    XtX_inv = np.linalg.inv(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    out = pd.DataFrame(np.nan, index=y.columns, columns=y.columns)
    for i in range(k):
        yi = Y[p:, i]
        b = XtX_inv @ X.T @ yi
        e = yi - X @ b
        w = (e / (1 - h)) ** 2
        V = XtX_inv @ (X.T * w) @ X @ XtX_inv
        for j in range(k):
            if i == j:
                continue
            idx = [1 + l * k + j for l in range(p)]
            R = np.zeros((p, X.shape[1]))
            for r, c in enumerate(idx):
                R[r, c] = 1.0
            rb = R @ b
            W = float(rb @ np.linalg.solve(R @ V @ R.T, rb))
            out.iloc[j, i] = 1 - stats.chi2.cdf(W, p)
    return out


def fdr_count(pmat: pd.DataFrame, alpha=0.05):
    pv = pmat.stack().values
    rej_fdr = multipletests(pv, alpha=alpha, method="fdr_bh")[0]
    rej_bon = multipletests(pv, alpha=alpha, method="bonferroni")[0]
    mask = pmat.stack()
    return int((pv < alpha).sum()), int(rej_fdr.sum()), int(rej_bon.sum()), pd.Series(rej_fdr, index=mask.index)


# ---------------------------------------------------------------- OOS predictability
def oos_predictability(y: pd.DataFrame, start_frac=0.25):
    """Expanding-window forecasts of each sector's next-day return.
    KS: OLS on all 11 lagged sector returns; COMB: equal-weight combination of the 11
    single-predictor forecasts (Rapach-Strauss-Zhou style); benchmark: prevailing mean."""
    Y = y.values
    T, k = Y.shape
    t0 = int(start_frac * T)
    rows = []
    for i in range(k):
        e_bench, e_ks, e_comb, f_ks_l, f_comb_l, f_b_l, act = [], [], [], [], [], [], []
        for t in range(t0, T - 1):
            X = np.column_stack([np.ones(t - 1), Y[:t - 1]])
            yy = Y[1:t, i]
            b = np.linalg.lstsq(X, yy, rcond=None)[0]
            f_ks = b[0] + Y[t] @ b[1:]
            fc = []
            for j in range(k):
                Xj = np.column_stack([np.ones(t - 1), Y[:t - 1, j]])
                bj = np.linalg.lstsq(Xj, yy, rcond=None)[0]
                fc.append(bj[0] + bj[1] * Y[t, j])
            f_comb = float(np.mean(fc))
            f_b = yy.mean()
            a = Y[t + 1, i]
            f_ks_l.append(f_ks); f_comb_l.append(f_comb); f_b_l.append(f_b); act.append(a)
        a, fb = np.array(act), np.array(f_b_l)
        res = {"sector": y.columns[i]}
        for name, f in [("KS", np.array(f_ks_l)), ("COMB", np.array(f_comb_l))]:
            r2 = 1 - np.sum((a - f) ** 2) / np.sum((a - fb) ** 2)
            adj = (a - fb) ** 2 - ((a - f) ** 2 - (fb - f) ** 2)     # Clark-West
            n = len(adj)
            lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
            d = adj - adj.mean()
            s = d @ d / n + 2 * sum((1 - l / (lag + 1)) * (d[l:] @ d[:-l]) / n for l in range(1, lag + 1))
            cw_t = adj.mean() / np.sqrt(s / n)
            res[f"R2OS_{name}"] = 100 * r2
            res[f"CW_p_{name}"] = 1 - stats.norm.cdf(cw_t)
        rows.append(res)
    return pd.DataFrame(rows).set_index("sector"), y.index[t0 + 1]


# ---------------------------------------------------------------- panel regressions
def two_way_demean(df, cols):
    out = df.copy()
    for c in cols:
        g_s = out.groupby("sector")[c].transform("mean")
        g_t = out.groupby("month")[c].transform("mean")
        out[c] = out[c] - g_s - g_t + out[c].mean()
    return out


def driscoll_kraay(df, y, xs, lag=3, fe="twoway"):
    d = df.dropna(subset=[y] + xs).copy()
    if fe == "twoway":
        d = two_way_demean(d, [y] + xs)
    elif fe == "sector":
        for c in [y] + xs:
            d[c] = d[c] - d.groupby("sector")[c].transform("mean")
    X = d[xs].values
    Yv = d[y].values
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ Yv
    e = Yv - X @ b
    d["_e"] = e
    St = pd.DataFrame(X * e[:, None], index=d["month"].values).groupby(level=0).sum().sort_index().values
    Tn = St.shape[0]
    S = St.T @ St
    for l in range(1, lag + 1):
        G = St[l:].T @ St[:-l]
        S += (1 - l / (lag + 1)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    tt = b / se
    pv = 2 * (1 - stats.t.cdf(np.abs(tt), df=Tn - 1))
    ss_res = e @ e
    ss_tot = ((Yv - Yv.mean()) ** 2).sum()
    return pd.DataFrame({"coef": b, "se": se, "t": tt, "p": pv}, index=xs), {"N": len(d), "T": Tn,
                                                                         "within_R2": 1 - ss_res / ss_tot}


def newey_west_ols(yv, X, lag=6):
    X = np.column_stack([np.ones(len(X)), X])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ yv
    e = yv - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, lag + 1):
        G = Xe[l:].T @ Xe[:-l]
        S += (1 - l / (lag + 1)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    r2 = 1 - (e @ e) / ((yv - yv.mean()) ** 2).sum()
    return b, se, b / se, r2


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    R = load("etf_returns.csv")[SECTORS]
    LV = load("etf_logvol.csv")[SECTORS]
    CR = load("const_returns.csv")[SECTORS]
    summary = {"sample": [str(R.index.min().date()), str(R.index.max().date())], "T": len(R)}

    # descriptive stats
    desc = pd.DataFrame({"mean_ann_%": R.mean() * 252, "vol_ann_%": R.std() * np.sqrt(252),
                         "skew": R.skew(), "exkurt": R.kurt(),
                         "mean_logvol": LV.mean(), "rho1_logvol": LV.apply(lambda s: s.autocorr(1))})
    desc.round(3).to_csv(os.path.join(OUT, "descriptives.csv"))

    # 1. static
    sel = {}
    for nm, y in [("ret", R), ("vol", LV)]:
        o = VAR(y.values).select_order(maxlags=10)
        sel[nm] = {"bic": int(o.bic), "aic": int(o.aic)}
    summary["lag_selection"] = sel
    p_ret, p_vol = max(1, sel["ret"]["bic"]), max(1, sel["vol"]["bic"])
    th_r, m_r = static(R, p_ret)
    th_v, m_v = static(LV, p_vol)
    for nm, th, mm in [("ret", th_r, m_r), ("vol", th_v, m_v)]:
        tbl = pd.DataFrame(th * 100, index=SECTORS, columns=SECTORS)
        tbl["FROM"] = mm["FROM"]
        tbl.loc["TO"] = list(mm["TO"]) + [np.nan]
        tbl.loc["NET"] = list(mm["NET"]) + [mm["TCI"]]
        tbl.round(2).to_csv(os.path.join(OUT, f"static_table_{nm}.csv"))
    summary["static_TCI"] = {"ret": m_r["TCI"], "vol": m_v["TCI"], "p_ret": p_ret, "p_vol": p_vol}

    grid = []
    for series, y in [("ETF returns", R), ("Constituent returns", CR), ("ETF log volatility", LV)]:
        for p in sorted({1, 2, 5, sel["ret" if "ret" in series.lower() else "vol"]["aic"]}):
            for hz in (5, 10, 20):
                _, mm = static(y, p, hz)
                grid.append({"series": series, "p": p, "H": hz, "TCI": mm["TCI"],
                             "top_transmitter": SECTORS[int(np.argmax(mm["NET"]))],
                             "top_receiver": SECTORS[int(np.argmin(mm["NET"]))]})
    pd.DataFrame(grid).round(2).to_csv(os.path.join(OUT, "static_robustness.csv"), index=False)

    # 2. TVP-VAR (burn-in on the June to September 2018 training window)
    raw = yf.download(["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"],
                      start="2018-06-20", end="2026-09-01", auto_adjust=True, progress=False)
    names = {"XLC": "Communication Services", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
             "XLE": "Energy", "XLF": "Financials", "XLV": "Health Care", "XLI": "Industrials",
             "XLK": "Information Technology", "XLB": "Materials", "XLRE": "Real Estate", "XLU": "Utilities"}
    Rf = (100 * np.log(raw["Close"]).diff()).rename(columns=names)[SECTORS].dropna()
    park = (np.log(raw["High"]) - np.log(raw["Low"])) ** 2 / (4 * np.log(2))
    LVf = np.log(100 * np.sqrt(252 * park.clip(lower=1e-10))).rename(columns=names)[SECTORS].dropna()
    burn = "2018-09-28"
    tvp = {}
    for nm, y in [("ret", Rf), ("vol", LVf)]:
        tv = tvp_var_connectedness(y, burn)
        tvp[nm] = tv
        for key in ("FROM", "TO", "NET", "OWN"):
            tv[key].to_csv(os.path.join(OUT, f"tvp_{nm}_{key}.csv"))
        tv["TCI"].to_csv(os.path.join(OUT, f"tvp_{nm}_TCI.csv"))
        np.save(os.path.join(OUT, f"tvp_{nm}_PAIR.npy"), tv["PAIR"])
    tvp_rob = {}
    for (a, b) in [(0.99, 0.96), (0.98, 0.99), (0.995, 0.99)]:
        tv = tvp_var_connectedness(Rf, burn, k1=a, k2=b)
        tvp_rob[f"k1={a},k2={b}"] = tv["TCI"]
    CRf = CR.copy()
    tv_c = tvp_var_connectedness(pd.concat([Rf.loc[:burn], CRf]).loc[~pd.concat([Rf.loc[:burn], CRf]).index.duplicated()], burn)
    tvp_rob["constituent returns"] = tv_c["TCI"]
    rob = pd.DataFrame(tvp_rob)
    rob["baseline"] = tvp["ret"]["TCI"]
    rob.to_csv(os.path.join(OUT, "tvp_robustness_TCI.csv"))
    summary["tvp_robustness_corr_with_baseline"] = rob.corr()["baseline"].round(3).to_dict()

    rt = rolling_tci(R, 200, p=p_ret)
    rt.to_csv(os.path.join(OUT, "rolling200_TCI.csv"))

    tr, tvv = tvp["ret"]["TCI"], tvp["vol"]["TCI"]
    summary["tvp_TCI_ret"] = {"mean": tr.mean(), "min": tr.min(), "min_date": str(tr.idxmin().date()),
                              "max": tr.max(), "max_date": str(tr.idxmax().date()),
                              "first": tr.iloc[0], "last": tr.iloc[-1]}
    summary["tvp_TCI_vol"] = {"mean": tvv.mean(), "min": tvv.min(), "min_date": str(tvv.idxmin().date()),
                              "max": tvv.max(), "max_date": str(tvv.idxmax().date()), "last": tvv.iloc[-1]}
    yearly = pd.DataFrame({"TCI_ret": tr, "TCI_vol": tvv}).resample("YE").mean()
    yearly.round(2).to_csv(os.path.join(OUT, "tvp_TCI_yearly.csv"))
    net_y = tvp["ret"]["NET"].resample("YE").mean()
    net_y.round(2).to_csv(os.path.join(OUT, "tvp_ret_NET_yearly.csv"))
    tvp["vol"]["NET"].resample("YE").mean().round(2).to_csv(os.path.join(OUT, "tvp_vol_NET_yearly.csv"))
    summary["share_days_transmitter_ret"] = (tvp["ret"]["NET"] > 0).mean().round(3).to_dict()

    # 3. Granger
    g = {}
    for label, y in [("full", R), ("first_half", R.loc[:"2022-08-31"]), ("second_half", R.loc["2022-09-01":])]:
        pm = system_granger(y, p_ret)
        raw_n, fdr_n, bon_n, rej = fdr_count(pm)
        g[label] = {"raw": raw_n, "fdr": fdr_n, "bonferroni": bon_n, "T": len(y)}
        pm.to_csv(os.path.join(OUT, f"granger_system_p_{label}.csv"))
        if label == "full":
            rej.to_csv(os.path.join(OUT, "granger_system_fdr_full.csv"))
    # bivariate HC3 for comparison with the pairwise literature
    biv = pd.DataFrame(np.nan, index=SECTORS, columns=SECTORS)
    for a in SECTORS:
        for b in SECTORS:
            if a != b:
                biv.loc[a, b] = system_granger(R[[a, b]], 5).loc[a, b]
    raw_n, fdr_n, bon_n, _ = fdr_count(biv)
    g["pairwise_lag5_HC3"] = {"raw": raw_n, "fdr": fdr_n, "bonferroni": bon_n}
    biv.to_csv(os.path.join(OUT, "granger_pairwise_hc3_p.csv"))
    summary["granger"] = g

    # 4. OOS predictability
    oos, oos_start = oos_predictability(R)
    oos.round(3).to_csv(os.path.join(OUT, "oos_predictability.csv"))
    summary["oos"] = {"start": str(oos_start.date()),
                      "n_R2pos_KS": int((oos["R2OS_KS"] > 0).sum()),
                      "n_R2pos_COMB": int((oos["R2OS_COMB"] > 0).sum()),
                      "n_CW_sig_KS": int((oos["CW_p_KS"] < 0.05).sum()),
                      "n_CW_sig_COMB": int((oos["CW_p_COMB"] < 0.05).sum()),
                      "fdr_CW_COMB": int(multipletests(oos["CW_p_COMB"], method="fdr_bh")[0].sum()),
                      "fdr_CW_KS": int(multipletests(oos["CW_p_KS"], method="fdr_bh")[0].sum())}

    # 5. concentration link (monthly)
    conc = pd.read_csv(os.path.join(PANEL, "concentration.csv"), parse_dates=["date"])
    conc["month"] = conc["date"].dt.to_period("M")
    cm = conc.sort_values("date").groupby(["sector", "month"]).last().reset_index()
    cm = cm[["sector", "month", "hhi", "top3", "top1", "n_firms", "sector_share"]]
    cm = cm.sort_values(["sector", "month"])
    for c in ["hhi", "top3", "sector_share"]:
        cm[c + "_lag"] = cm.groupby("sector")[c].shift(1)
    cm["log_share_lag"] = np.log(cm["sector_share_lag"])
    cm["hhi_lag_pp"] = 100 * cm["hhi_lag"]
    cm["top3_lag_pp"] = 100 * cm["top3_lag"]
    lvm = LV.groupby(LV.index.to_period("M")).mean().stack().rename("logvol").reset_index()
    lvm.columns = ["month", "sector", "logvol"]

    regs, specs = [], []
    for nm in ("ret", "vol"):
        mm = {}
        for key in ("FROM", "TO", "NET", "OWN"):
            d = tvp[nm][key]
            mm[key] = d.groupby(d.index.to_period("M")).mean().stack().rename(key)
        M = pd.concat(mm, axis=1).reset_index()
        M.columns = ["month", "sector"] + list(mm)
        pan = M.merge(cm, on=["sector", "month"]).merge(lvm, on=["month", "sector"])
        pan = pan[pan["month"] >= pd.Period("2018-11", "M")]
        pan.to_csv(os.path.join(OUT, f"panel_monthly_{nm}.csv"), index=False)
        for dep in ("FROM", "TO", "NET"):
            for cvar in ("hhi_lag_pp", "top3_lag_pp"):
                for fe in ("twoway", "sector"):
                    xs = [cvar, "log_share_lag", "logvol"]
                    tab, info = driscoll_kraay(pan, dep, xs, lag=3, fe=fe)
                    for v in xs:
                        regs.append({"system": nm, "dep": dep, "conc": cvar, "fe": fe, "var": v,
                                     **tab.loc[v].to_dict(), **info})
        # first differences, sector FE absorbed by differencing, time FE via demeaning by month
        pan = pan.sort_values(["sector", "month"])
        for c in ["FROM", "TO", "NET", "hhi_lag_pp", "log_share_lag", "logvol"]:
            pan["d_" + c] = pan.groupby("sector")[c].diff()
        for dep in ("d_FROM", "d_TO", "d_NET"):
            xs = ["d_hhi_lag_pp", "d_log_share_lag", "d_logvol"]
            dd = pan.dropna(subset=[dep] + xs).copy()
            for c in [dep] + xs:
                dd[c] = dd[c] - dd.groupby("month")[c].transform("mean")
            dd["sector"] = "all"
            tab, info = driscoll_kraay(dd, dep, xs, lag=3, fe="none")
            for v in xs:
                regs.append({"system": nm, "dep": dep, "conc": "d_hhi", "fe": "FD+time", "var": v,
                             **tab.loc[v].to_dict(), **info})
    pd.DataFrame(regs).round(4).to_csv(os.path.join(OUT, "panel_regressions.csv"), index=False)

    # index-level link: monthly system TCI vs top-10 share, controlling for log VIX
    idxc = load("index_conc.csv")
    vix = load("vix.csv").squeeze()
    ts = pd.DataFrame({"TCI_ret": tr, "TCI_vol": tvv}).resample("ME").mean()
    ts["top10"] = 100 * idxc["top10_share"].resample("ME").last()
    ts["logvix"] = np.log(vix.resample("ME").mean())
    ts = ts.dropna()
    ts.to_csv(os.path.join(OUT, "index_level_monthly.csv"))
    il = {}
    for dep in ("TCI_ret", "TCI_vol"):
        b, se, t, r2 = newey_west_ols(ts[dep].values, ts[["top10", "logvix"]].values, lag=6)
        il[f"levels_{dep}"] = {"b_top10": b[1], "t_top10": t[1], "b_logvix": b[2], "t_logvix": t[2], "R2": r2}
        dts = ts.diff().dropna()
        b, se, t, r2 = newey_west_ols(dts[dep].values, dts[["top10", "logvix"]].values, lag=6)
        il[f"diff_{dep}"] = {"b_top10": b[1], "t_top10": t[1], "b_logvix": b[2], "t_logvix": t[2], "R2": r2}
    summary["index_level"] = il
    summary["corr_TCIret_top10_levels"] = float(ts["TCI_ret"].corr(ts["top10"]))

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=lambda x: round(float(x), 4)))


if __name__ == "__main__":
    main()
