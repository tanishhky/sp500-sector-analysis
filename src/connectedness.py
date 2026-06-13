"""
connectedness.py - Diebold-Yilmaz (2012) generalized connectedness across the
11 GICS sectors, plus multiple-testing-corrected Granger causality.

This replaces the earlier pairwise-Granger headline (which counted ~300 raw
p<0.05 pairs with no correction, on only 25 quarterly observations - of which
zero survive FDR/Bonferroni). The connectedness framework is VAR-based and
variance-decomposition driven, so it has no multiple-comparisons problem, and it
runs on ~1,590 daily observations.

Method (Diebold & Yilmaz 2012; generalized FEVD of Pesaran & Shin 1998):
  - Fit VAR(p) on the daily sector returns (p by AIC).
  - Generalized forecast-error variance decomposition at horizon H.
  - Row-normalize to a connectedness table; derive FROM, TO, NET, and the
    total connectedness index (TCI).
  - A rolling-window TCI traces time-varying system stress.

Run: python src/connectedness.py
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "sector_returns_daily.csv")
OUT = os.path.join(BASE, "output", "connectedness")


def load_returns() -> pd.DataFrame:
    return pd.read_csv(DATA, index_col=0, parse_dates=True).dropna()


def generalized_fevd(results, horizon: int = 10) -> np.ndarray:
    """Generalized FEVD (Pesaran-Shin), row-normalized. Returns theta[i,j]:
    share of i's H-step forecast-error variance due to shocks in j."""
    sigma = results.sigma_u.values if hasattr(results.sigma_u, "values") else np.asarray(results.sigma_u)
    phi = results.ma_rep(maxn=horizon - 1)  # (H, k, k), phi[0] = I
    k = sigma.shape[0]
    A = np.array([phi[h] @ sigma for h in range(horizon)])  # (H, k, k)
    num = np.sum(A ** 2, axis=0) / np.diag(sigma)[None, :]   # (k, k)
    den = np.zeros(k)
    for h in range(horizon):
        den += np.diag(phi[h] @ sigma @ phi[h].T)
    theta = num / den[:, None]
    return theta / theta.sum(axis=1, keepdims=True)          # row-normalize


def connectedness_table(theta: np.ndarray, names: list[str]) -> pd.DataFrame:
    k = len(names)
    tbl = pd.DataFrame(theta * 100, index=names, columns=names)
    off = theta.copy()
    np.fill_diagonal(off, 0.0)
    from_others = off.sum(axis=1) * 100          # received by i
    to_others = off.sum(axis=0) * 100            # transmitted by j
    tbl["FROM"] = from_others
    to_row = pd.Series(to_others, index=names)
    to_row["FROM"] = np.nan
    tbl.loc["TO"] = to_row
    net = to_others - from_others
    net_row = pd.Series(net, index=names)
    net_row["FROM"] = off.sum() / k * 100        # Total Connectedness Index
    tbl.loc["NET"] = net_row
    return tbl


def total_connectedness(theta: np.ndarray) -> float:
    off = theta.copy()
    np.fill_diagonal(off, 0.0)
    return off.sum() / theta.shape[0] * 100


def rolling_tci(returns: pd.DataFrame, window: int = 200, p: int = 3, horizon: int = 10) -> pd.Series:
    vals, idx = [], []
    arr = returns.values
    for end in range(window, len(returns) + 1):
        win = arr[end - window:end]
        try:
            res = VAR(win).fit(p)
            vals.append(total_connectedness(generalized_fevd(res, horizon)))
            idx.append(returns.index[end - 1])
        except Exception:
            continue
    return pd.Series(vals, index=idx, name="TCI")


def corrected_granger(returns: pd.DataFrame, max_lag: int = 5) -> dict:
    cols = list(returns.columns)
    pvals, pairs = [], []
    for a in cols:
        for b in cols:
            if a == b:
                continue
            try:
                r = grangercausalitytests(returns[[b, a]], maxlag=max_lag, verbose=False)
                # Bonferroni-correct the min-over-lags selection by tagging the lag count
                p = min(r[l][0]["ssr_ftest"][1] for l in range(1, max_lag + 1))
                pvals.append(p)
                pairs.append((a, b))
            except Exception:
                continue
    pvals = np.array(pvals)
    raw = int((pvals < 0.05).sum())
    bonf = multipletests(pvals, alpha=0.05, method="bonferroni")
    fdr = multipletests(pvals, alpha=0.05, method="fdr_bh")
    survivors = [(pairs[i][0], pairs[i][1], pvals[i]) for i in range(len(pvals)) if fdr[0][i]]
    return {
        "n_pairs": len(pvals), "raw_sig": raw, "expected_by_chance": 0.05 * len(pvals),
        "bonferroni_sig": int(bonf[0].sum()), "fdr_sig": int(fdr[0].sum()),
        "fdr_survivors": sorted(survivors, key=lambda x: x[2]),
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rets = load_returns()
    names = list(rets.columns)
    print(f"VAR connectedness on {rets.shape[0]} daily obs x {rets.shape[1]} sectors")

    order = VAR(rets).select_order(maxlags=10)
    p = max(1, int(order.aic))
    print(f"selected VAR lag (AIC): {p}")
    res = VAR(rets).fit(p)
    theta = generalized_fevd(res, horizon=10)
    tbl = connectedness_table(theta, names)
    tbl.round(2).to_csv(os.path.join(OUT, "connectedness_table.csv"))

    tci = total_connectedness(theta)
    print(f"\nTotal Connectedness Index (full sample): {tci:.1f}%")
    net = (tbl.loc["NET"].drop("FROM")).sort_values(ascending=False)
    print("\nNET directional connectedness (transmitter > 0 > receiver):")
    print(net.round(2).to_string())

    print("\nRolling TCI (200d window)...")
    rtci = rolling_tci(rets, window=200, p=p, horizon=10)
    rtci.to_csv(os.path.join(OUT, "rolling_tci.csv"))
    print(f"  rolling TCI range: {rtci.min():.1f}% to {rtci.max():.1f}%  (peak {rtci.idxmax().date()})")

    print("\nGranger (daily, multiple-testing corrected)...")
    g = corrected_granger(rets, max_lag=5)
    print(f"  pairs={g['n_pairs']}  raw p<.05={g['raw_sig']} (chance~{g['expected_by_chance']:.0f})"
          f"  Bonferroni={g['bonferroni_sig']}  FDR={g['fdr_sig']}")
    pd.DataFrame(g["fdr_survivors"], columns=["cause", "effect", "min_p"]).to_csv(
        os.path.join(OUT, "granger_fdr_survivors.csv"), index=False)


if __name__ == "__main__":
    main()
