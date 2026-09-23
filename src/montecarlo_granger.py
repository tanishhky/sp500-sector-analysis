"""
montecarlo_granger.py - Size of daily Granger tests under the null of no lead-lag,
with GARCH volatility and a common factor calibrated to the sector ETF returns.

DGP: r_it = b_i f_t + e_it,  f_t GARCH(1,1)-t fitted to the equal-weighted sector return,
e_it GARCH(1,1)-t fitted to each sector's residual. No lagged cross-dependence by construction,
so every rejection is a false positive.

Reports per-test size at 5% for the classical F (fixed lag 5, and min over lags 1..5) and the
HC3 Wald (lag 1, lag 5), plus the number of the 110 directed pairs that survive BH-FDR under the
classical min-over-lags procedure and under HC3.
Run: python src/montecarlo_granger.py   (writes output/v2/montecarlo_granger.json)
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from arch import arch_model
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import grangercausalitytests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from econometrics import OUT, SECTORS, load, system_granger  # noqa: E402


def fit_garch(x):
    r = arch_model(x, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    return r.params


def sim_garch(params, T, rng, burn=500):
    w, a, b, nu = params["omega"], params["alpha[1]"], params["beta[1]"], params["nu"]
    z = rng.standard_t(nu, size=T + burn) * np.sqrt((nu - 2) / nu)
    h = np.empty(T + burn); x = np.empty(T + burn)
    h[0] = w / max(1 - a - b, 1e-3)
    for t in range(T + burn):
        if t > 0:
            h[t] = w + a * x[t - 1] ** 2 + b * h[t - 1]
        x[t] = np.sqrt(h[t]) * z[t]
    return x[burn:]


def classical_p(y2, L, use_min):
    r = grangercausalitytests(y2, maxlag=L, verbose=False)
    if use_min:
        return min(r[l][0]["ssr_ftest"][1] for l in range(1, L + 1))
    return r[L][0]["ssr_ftest"][1]


def main(reps_pair=1000, reps_system=100, seed=7):
    R = load("etf_returns.csv")[SECTORS]
    T, k = R.shape
    f = R.mean(axis=1)
    pf = fit_garch(f.values)
    betas, pe = {}, {}
    for s in SECTORS:
        b = np.polyfit(f.values, R[s].values, 1)[0]
        betas[s] = b
        pe[s] = fit_garch(R[s].values - b * f.values)
    rng = np.random.default_rng(seed)

    def draw():
        ff = sim_garch(pf, T, rng)
        return pd.DataFrame({s: betas[s] * ff + sim_garch(pe[s], T, rng) for s in SECTORS})

    # per-test size on one representative pair (Industrials -> Energy)
    rej = {"F_lag5": 0, "F_min1to5": 0, "HC3_lag1": 0, "HC3_lag5": 0}
    for _ in range(reps_pair):
        Y = draw()[["Industrials", "Energy"]]
        y2 = Y[["Energy", "Industrials"]]
        rej["F_lag5"] += classical_p(y2, 5, False) < 0.05
        rej["F_min1to5"] += classical_p(y2, 5, True) < 0.05
        rej["HC3_lag1"] += system_granger(Y, 1).loc["Industrials", "Energy"] < 0.05
        rej["HC3_lag5"] += system_granger(Y, 5).loc["Industrials", "Energy"] < 0.05
    size = {k_: v / reps_pair for k_, v in rej.items()}

    # full 110-pair FDR counts
    fdr_classic, fdr_hc3 = [], []
    for _ in range(reps_system):
        Y = draw()
        pv_c, pv_h = [], []
        for a in SECTORS:
            for b in SECTORS:
                if a == b:
                    continue
                pv_c.append(classical_p(Y[[b, a]], 5, True))
                pv_h.append(system_granger(Y[[a, b]], 5).loc[a, b])
        fdr_classic.append(int(multipletests(pv_c, method="fdr_bh")[0].sum()))
        fdr_hc3.append(int(multipletests(pv_h, method="fdr_bh")[0].sum()))
    out = {"T": T, "reps_pair": reps_pair, "reps_system": reps_system,
           "factor_garch": {k_: float(v) for k_, v in pf.items()},
           "size_5pct": size,
           "fdr_survivors_classical_min_lag": {"mean": float(np.mean(fdr_classic)),
                                               "median": float(np.median(fdr_classic)),
                                               "p90": float(np.percentile(fdr_classic, 90))},
           "fdr_survivors_hc3_lag5": {"mean": float(np.mean(fdr_hc3)), "median": float(np.median(fdr_hc3)),
                                      "p90": float(np.percentile(fdr_hc3, 90))}}
    with open(os.path.join(OUT, "montecarlo_granger.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
