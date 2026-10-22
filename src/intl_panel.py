"""
intl_panel.py - Daily sector returns, concentration, and top-three / rest-of-sector components for
every market in data/intl/universe.csv, built the same way as the U.S. constituent panel:
total market capitalization from split-adjusted closing prices and historical shares outstanding,
previous-day weights, top-three share and Herfindahl index within each sector.

Data checks specific to non-U.S. listings:
  - a firm's historical share series is replaced by its current share count when the two disagree
    by more than a factor of two at the end of the sample, and any stretch more than four times
    away from the current count is treated as a vendor unit error and filled from valid days;
  - daily log returns beyond +/-40% are treated as bad ticks and set to missing;
  - days on which more than 80% of firms show a zero return are dropped as stale holiday rows.
Raw vendor data go to data/intl/raw/ (gitignored); sector-level panels to data/intl/panels/.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yfinance as yf  # noqa: E402

from build_panel import FETCH_START, split_adjusted_shares  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTL = os.path.join(BASE, "data", "intl")
RAW = os.path.join(INTL, "raw")
PAN = os.path.join(INTL, "panels")
START, END = "2018-10-01", "2026-09-01"


def slug(m: str) -> str:
    return m.lower().replace(" ", "_")


def shares_for(args):
    tk, idx, now = args
    try:
        s = split_adjusted_shares(tk, idx)
    except Exception:
        s = None
    flag = "historical"
    if s is None or s.dropna().empty:
        s, flag = (pd.Series(float(now), index=idx), "constant") if now and now > 0 else (None, "missing")
    elif now and now > 0:
        ratio = float(s.dropna().iloc[-1]) / float(now)
        if not 0.5 <= ratio <= 2.0:
            s, flag = pd.Series(float(now), index=idx), "constant (unit mismatch)"
    return tk, s, flag


def clean_shares(sh: pd.DataFrame, now: dict, band: float = 4.0):
    """Vendor share series sometimes switch units for a stretch (Air Canada's early-2019 count is
    about 630 times too large). Values more than `band` times away from the firm's current share
    count are set to missing and filled from the nearest valid observation; a firm with more than
    half its history invalid falls back to its current count."""
    out, fixed = sh.copy(), 0
    for c in out.columns:
        ref = now.get(c)
        s = out[c]
        if not ref or ref <= 0 or s.dropna().empty:
            continue
        bad = (s / ref > band) | (s / ref < 1 / band)
        if bad.any():
            fixed += 1
            out[c] = s.mask(bad).ffill().bfill() if bad.mean() <= 0.5 else float(ref)
    return out, fixed


def build_market(market: str, u: pd.DataFrame) -> dict:
    tag = slug(market)
    u = u[(u["market"] == market) & u["sector"].notna() & (u["quote_type"].fillna("EQUITY") == "EQUITY")]
    tickers = sorted(u["ticker"].unique())
    pf = os.path.join(RAW, f"{tag}_prices.csv")
    if os.path.exists(pf):
        px = pd.read_csv(pf, index_col=0, parse_dates=True)
    else:
        px = yf.download(tickers, start=FETCH_START, end=END, auto_adjust=False, progress=False, threads=True)["Close"]
        px = px.dropna(axis=1, how="all")
        px.to_csv(pf)
    px = px[[c for c in px.columns if c in set(tickers)]]
    sf = os.path.join(RAW, f"{tag}_shares.csv")
    now = u.set_index("ticker")["shares_now"].to_dict()
    if os.path.exists(sf):
        sh = pd.read_csv(sf, index_col=0, parse_dates=True)
        flags = json.load(open(sf.replace(".csv", "_flags.json")))
    else:
        with ThreadPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(shares_for, [(tk, px.index, now.get(tk)) for tk in px.columns]))
        sh = pd.DataFrame({tk: s for tk, s, f in res if s is not None})
        flags = {tk: f for tk, s, f in res}
        sh.to_csv(sf)
        json.dump(flags, open(sf.replace(".csv", "_flags.json"), "w"), indent=1)

    cols = [c for c in px.columns if c in sh.columns]
    px, sh = px[cols], sh[cols]
    sh, n_fixed = clean_shares(sh, now)
    r = 100 * np.log(px).diff()
    r = r.where(r.abs() <= 40)
    mcap = px * sh
    r, mcap = r.loc[START:], mcap.loc[START:]
    zero_share = (r == 0).sum(axis=1) / r.notna().sum(axis=1).clip(lower=1)
    keep = (zero_share <= 0.8) & (r.notna().sum(axis=1) >= 0.5 * r.shape[1] * 0.5)
    r, mcap = r[keep], mcap[keep]

    sector = u.drop_duplicates("ticker").set_index("ticker").loc[cols, "sector"]
    total = mcap.sum(axis=1)
    out_r, out_top, out_rest, conc = {}, {}, {}, []
    wl_all = mcap.shift(1)
    for s in sorted(sector.unique()):
        members = sector.index[sector == s]
        cap = mcap[members]
        wl = wl_all[members]
        rr = r[members]
        valid = wl.notna() & rr.notna()
        w = wl.where(valid)
        W = w.div(w.sum(axis=1), axis=0)
        out_r[s] = (W * rr).sum(axis=1, min_count=1)
        rank = W.rank(axis=1, ascending=False, method="first")
        Wt = W.where(rank <= 3)
        wt = Wt.sum(axis=1)
        out_top[s] = (Wt * rr).sum(axis=1, min_count=1) / wt
        out_rest[s] = (out_r[s] - wt * out_top[s]) / (1 - wt)
        wd = cap.div(cap.sum(axis=1), axis=0)
        # fixed membership: firms already trading on the first sample day, so later listings
        # cannot move concentration
        fixed = cap.loc[:, cap.iloc[0].notna()]
        wf = fixed.div(fixed.sum(axis=1), axis=0)
        conc.append(pd.DataFrame({"date": cap.index, "sector": s,
                                  "top3": wd.apply(lambda x: x.nlargest(3).sum(), axis=1).values,
                                  "top3_fixed": wf.apply(lambda x: x.nlargest(3).sum(), axis=1).values,
                                  "n_fixed": fixed.notna().sum(axis=1).values,
                                  "hhi": (wd ** 2).sum(axis=1).values,
                                  "n_firms": cap.notna().sum(axis=1).values,
                                  "sector_share": (cap.sum(axis=1) / total).values,
                                  "W_top_lag": wt.values}))
    os.makedirs(PAN, exist_ok=True)
    pd.DataFrame(out_r).to_csv(os.path.join(PAN, f"{tag}_returns.csv"))
    pd.DataFrame(out_top).to_csv(os.path.join(PAN, f"{tag}_top3.csv"))
    pd.DataFrame(out_rest).to_csv(os.path.join(PAN, f"{tag}_rest.csv"))
    pd.concat(conc).to_csv(os.path.join(PAN, f"{tag}_concentration.csv"), index=False)
    fl = pd.Series(flags).reindex(cols).value_counts().to_dict()
    return {"market": market, "firms": len(cols), "days": int(len(r)),
            "first": str(r.index.min().date()), "last": str(r.index.max().date()),
            "share_flags": fl, "share_series_cleaned": n_fixed, "sectors": int(sector.nunique())}


def main():
    os.makedirs(RAW, exist_ok=True)
    u = pd.read_csv(os.path.join(INTL, "universe.csv"))
    markets = sys.argv[1:] or list(u["market"].unique())
    summary = []
    for m in markets:
        info = build_market(m, u)
        print(info, flush=True)
        summary.append(info)
    path = os.path.join(PAN, "build_summary.json")
    old = json.load(open(path)) if os.path.exists(path) else []
    old = [x for x in old if x["market"] not in markets] + summary
    json.dump(old, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
