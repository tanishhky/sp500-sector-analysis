"""
granular_decomposition.py - How much of the decline in sector connectedness is the weight shift
toward each sector's three largest firms? (v3 revision, October 2026)

Each day, the uncapped constituent sector return is split into the return on the sector's three
largest firms (chosen and weighted by the previous day's market capitalization) and the return on
the rest of the sector:
    r_s = W r_top + (1 - W) r_rest,   W = top-three share at t-1.
Accounting, not regression: nothing here relies on inference about two trending series.
  1. Validation: the top-three share from the candidate firms against the full panel.
  2. Co-movement of each component with the rest of the market, by sector and year.
  3. Counterfactual: hold W at its October to December 2018 level and recompute each sector's R2
     against the rest of the market and the connectedness of the eleven-sector system over the
     last twelve months of the sample. The exercise holds the component returns fixed, so it
     measures the mechanical contribution of the weight shift.
Needs data/firm/ from fetch_top_firms.py. Writes output/v3/.
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
from build_panel import GICS_2023, START  # noqa: E402
from econometrics import OUT, PANEL, SECTORS, load  # noqa: E402
from fetch_top_firms import CANDIDATES  # noqa: E402
from referee_checks import dir_measures  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRM = os.path.join(BASE, "data", "firm")
OUT3 = os.path.join(os.path.dirname(OUT), "v3")
PAY = {"V", "MA", "PYPL", "FIS", "GPN"}            # IT before 17 March 2023, Financials after
TO_STAPLES = {"TGT", "DG", "DLTR"}                 # Consumer Discretionary before, Staples after
ALIASES = {"META": ["FB", "META"]}
LAST12 = ("2025-09-01", "2026-08-31")
# base period starts after the Praxair-Linde merger (6 November 2018): before it, Praxair, which
# is not a candidate, was among the three largest Materials firms
BASE_Q = ("2018-11-06", "2018-12-31")


def membership(index: pd.DatetimeIndex, tickers) -> pd.DataFrame:
    ref = pd.read_csv(os.path.join(BASE, "data", "reference", "sp500_ticker_start_end.csv"),
                      parse_dates=["start_date", "end_date"])
    ref["ticker"] = ref["ticker"].str.replace(".", "-", regex=False)
    out = {}
    for tk in tickers:
        names = ALIASES.get(tk, [tk])
        rows = ref[ref["ticker"].isin(names)]
        m = np.zeros(len(index), dtype=bool)
        for _, r in rows.iterrows():
            end = r["end_date"] if pd.notna(r["end_date"]) else pd.Timestamp("2100-01-01")
            m |= (index >= r["start_date"]) & (index < end)
        out[tk] = m
    return pd.DataFrame(out, index=index)


def sector_of(index: pd.DatetimeIndex, tickers) -> pd.DataFrame:
    home = {t: s for s, v in CANDIDATES.items() for t in v}
    pre = index < GICS_2023
    out = {}
    for tk in tickers:
        s = np.array([home[tk]] * len(index), dtype=object)
        if tk in PAY:
            s[pre] = "Information Technology"
        if tk in TO_STAPLES:
            s[~pre] = "Consumer Staples"
        out[tk] = s
    return pd.DataFrame(out, index=index)


def main():
    os.makedirs(OUT3, exist_ok=True)
    px = pd.read_csv(os.path.join(FIRM, "prices.csv"), index_col=0, parse_dates=True)
    sh = pd.read_csv(os.path.join(FIRM, "shares.csv"), index_col=0, parse_dates=True)
    cols = [c for c in px.columns if c in sh.columns]
    mcap = (px[cols] * sh[cols])
    ret = 100 * np.log(px[cols]).diff()
    mcap, ret = mcap.loc[START:], ret.loc[START:]
    idx = mcap.index
    mem = membership(idx, cols)
    sec = sector_of(idx, cols)
    mcap = mcap.where(mem)

    conc = pd.read_csv(os.path.join(PANEL, "concentration.csv"), parse_dates=["date"])
    scap = conc.pivot(index="date", columns="sector", values="sector_cap").reindex(idx).ffill()
    top3_full = conc.pivot(index="date", columns="sector", values="top3").reindex(idx)

    CR = load("const_returns.csv")[SECTORS]
    days = CR.index
    W, Rtop, names_last = {}, {}, {}
    check = {}
    for s in SECTORS:
        in_s = (sec == s) & mcap.notna()
        w = mcap.where(in_s).div(scap[s], axis=0)                       # weight in the full sector
        top_w = w.apply(lambda r: r.nlargest(3), axis=1)                # same-day top three
        check[s] = (top_w.sum(axis=1) - top3_full[s]).abs()
        wl = w.shift(1)                                                 # yesterday's weights
        rank = wl.rank(axis=1, ascending=False, method="first")
        wtop = wl.where(rank <= 3)
        Wt = wtop.sum(axis=1)
        rt = (wtop * ret).sum(axis=1, min_count=1) / Wt
        W[s], Rtop[s] = Wt.reindex(days), rt.reindex(days)
        last = wl.iloc[-1].nlargest(3)
        names_last[s] = [f"{k} {100 * v:.1f}%" for k, v in last.items()]
    W, Rtop = pd.DataFrame(W)[SECTORS], pd.DataFrame(Rtop)[SECTORS]
    Rrest = (CR - W * Rtop) / (1 - W)
    chk = pd.DataFrame(check).reindex(days)
    post = chk.loc[BASE_Q[0]:]
    valid = {s: {"mean_abs_dev_pp": float(100 * chk[s].mean()), "p99_abs_dev_pp": float(100 * chk[s].quantile(0.99)),
                 "max_abs_dev_pp_after_base_start": float(100 * post[s].max())}
             for s in SECTORS}

    ok = W.notna().all(axis=1) & Rtop.notna().all(axis=1) & Rrest.notna().all(axis=1)
    CR, W, Rtop, Rrest = CR[ok], W[ok], Rtop[ok], Rrest[ok]

    # 2. component co-movement with the rest of the market, by sector and year
    recs = []
    for s in SECTORS:
        mkt = CR.drop(columns=s).mean(axis=1)
        df = pd.DataFrame({"sector": CR[s], "top": Rtop[s], "rest": Rrest[s], "mkt": mkt, "W": W[s]})
        for yr, g in df.groupby(df.index.year):
            recs.append({"sector": s, "year": yr, "W": 100 * g["W"].mean(),
                         "R2_sector": 100 * g["sector"].corr(g["mkt"]) ** 2,
                         "R2_top": 100 * g["top"].corr(g["mkt"]) ** 2,
                         "R2_rest": 100 * g["rest"].corr(g["mkt"]) ** 2,
                         "corr_top_rest": g["top"].corr(g["rest"]),
                         "vol_top": g["top"].std() * np.sqrt(252), "vol_rest": g["rest"].std() * np.sqrt(252)})
    comp = pd.DataFrame(recs)
    comp.round(3).to_csv(os.path.join(OUT3, "granular_components_by_year.csv"), index=False)
    share_top_lower = float((comp["R2_top"] < comp["R2_rest"]).mean())

    # 3. counterfactual: top-three weight frozen at its 2018Q4 level
    W0 = W.loc[BASE_Q[0]:BASE_Q[1]].mean()
    CF = Rtop * W0 + Rrest * (1 - W0)
    a, b = LAST12
    rows = []
    for s in SECTORS:
        mkt_act = CR.drop(columns=s).mean(axis=1)
        mkt_cf = CF.drop(columns=s).mean(axis=1)
        first = slice("2018-10-01", "2019-09-30")
        rows.append({"sector": s, "W_2018Q4": 100 * W0[s], "W_last12": 100 * W.loc[a:b, s].mean(),
                     "R2_first12": 100 * CR.loc[first, s].corr(mkt_act.loc[first]) ** 2,
                     "R2_last12_actual": 100 * CR.loc[a:b, s].corr(mkt_act.loc[a:b]) ** 2,
                     "R2_last12_cf": 100 * CF.loc[a:b, s].corr(mkt_cf.loc[a:b]) ** 2,
                     "top3_last": "; ".join(names_last[s])})
    cf = pd.DataFrame(rows).set_index("sector")
    cf["R2_change_actual"] = cf["R2_last12_actual"] - cf["R2_first12"]
    cf["R2_due_to_weights"] = cf["R2_last12_actual"] - cf["R2_last12_cf"]
    cf.round(2).to_csv(os.path.join(OUT3, "granular_counterfactual.csv"))

    def system(Y):
        C = np.corrcoef(Y.values.T)
        raw = C ** 2
        frm, to, net = dir_measures(raw / raw.sum(axis=1, keepdims=True))
        k = C.shape[0]
        avg_rho2 = (raw.sum() - k) / (k * (k - 1))
        return {"TCI": float(frm.mean()), "avg_rho2": float(100 * avg_rho2), "TO": to, "NET": net}

    s_first = system(CR.loc["2018-10-01":"2019-09-30"])
    s_act, s_cf = system(CR.loc[a:b]), system(CF.loc[a:b])
    s_cf_first = system(CF.loc["2018-10-01":"2019-09-30"])
    sys_tab = pd.DataFrame({"TO_actual_last12": s_act["TO"], "TO_cf_last12": s_cf["TO"],
                            "NET_actual_last12": s_act["NET"], "NET_cf_last12": s_cf["NET"]}, index=SECTORS)
    sys_tab.round(2).to_csv(os.path.join(OUT3, "granular_counterfactual_system.csv"))
    yearly = []
    for yr in sorted(set(CR.index.year)):
        ya, yc = system(CR[CR.index.year == yr]), system(CF[CF.index.year == yr])
        yearly.append({"year": yr, "avg_rho2_actual": ya["avg_rho2"], "avg_rho2_cf": yc["avg_rho2"],
                       "TCI_actual": ya["TCI"], "TCI_cf": yc["TCI"]})
    pd.DataFrame(yearly).round(3).to_csv(os.path.join(OUT3, "granular_system_by_year.csv"), index=False)

    # mega-cap block: correlation among the top-three components of COM, CD and IT
    blk = ["Communication Services", "Consumer Discretionary", "Information Technology"]
    other = [s for s in SECTORS if s not in blk]
    L = slice(a, b)
    ct, cr_ = Rtop.loc[L].corr(), Rrest.loc[L].corr()
    block = {"top_within_block": float(ct.loc[blk, blk].values[np.triu_indices(3, 1)].mean()),
             "rest_within_block": float(cr_.loc[blk, blk].values[np.triu_indices(3, 1)].mean()),
             "top_block_vs_other_sectors": float(pd.concat([Rtop.loc[L, blk], CR.loc[L, other]], axis=1)
                                                 .corr().loc[blk, other].values.mean()),
             "rest_block_vs_other_sectors": float(pd.concat([Rrest.loc[L, blk], CR.loc[L, other]], axis=1)
                                                  .corr().loc[blk, other].values.mean())}

    summary = {"validation_top3_vs_full_panel": valid, "days": int(ok.sum()),
               "share_sector_years_R2top_below_R2rest": share_top_lower,
               "system_first12": {k: v for k, v in s_first.items() if k in ("TCI", "avg_rho2")},
               "system_first12_cf": {k: v for k, v in s_cf_first.items() if k in ("TCI", "avg_rho2")},
               "system_last12_actual": {k: v for k, v in s_act.items() if k in ("TCI", "avg_rho2")},
               "system_last12_cf": {k: v for k, v in s_cf.items() if k in ("TCI", "avg_rho2")},
               "megacap_block_last12": block}
    with open(os.path.join(OUT3, "granular_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(cf.round(1).to_string())
    print(comp[comp["year"].isin([2019, 2022, 2024, 2026])].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
