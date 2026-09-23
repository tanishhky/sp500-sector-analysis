"""
build_panel.py - Daily panel for the concentration/connectedness study.

Outputs (data/panel/):
  etf_returns.csv       daily log returns (%) of the 11 Select Sector SPDR ETFs
  etf_logvol.csv        daily log Parkinson range volatility (annualized, %) of the same
  const_returns.csv     uncapped cap-weighted sector returns built from constituents (robustness)
  concentration.csv     long panel: date, sector, hhi, top1, top3, n_firms, sector_share
  index_conc.csv        S&P 500 top-10 weight share and HHI (all constituents)
  vix.csv               VIX close
Sample: 2018-10-01 (after the Sept-2018 Communication Services reorganization)
through 2026-08-31.
"""
from __future__ import annotations

import io
import os
import time
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
import yfinance as yf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "panel")
START, END = "2018-10-01", "2026-09-01"
FETCH_START = "2018-06-01"

ETF = {"XLC": "Communication Services", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
       "XLE": "Energy", "XLF": "Financials", "XLV": "Health Care", "XLI": "Industrials",
       "XLK": "Information Technology", "XLB": "Materials", "XLRE": "Real Estate", "XLU": "Utilities"}

# March 17, 2023 GICS changes: payment processors moved IT -> Financials; Target, Dollar General
# and Dollar Tree moved Consumer Discretionary -> Consumer Staples.
GICS_2023 = pd.Timestamp("2023-03-17")
PRE2023_CONS_DISC = {"TGT", "DG", "DLTR"}
# second share classes: the vendor reports firm-level shares on both lines, so keep one line per firm
SECOND_CLASS = {"GOOG", "FOX", "NWS", "UA", "DISCK"}


def etf_series():
    raw = yf.download(list(ETF), start=FETCH_START, end=END, auto_adjust=True, progress=False)
    close, high, low = raw["Close"], raw["High"], raw["Low"]
    ret = 100 * np.log(close).diff()
    park = (np.log(high) - np.log(low)) ** 2 / (4 * np.log(2))           # daily variance
    logvol = np.log(100 * np.sqrt(252 * park.clip(lower=1e-10)))          # log annualized vol, %
    ret, logvol = ret.rename(columns=ETF), logvol.rename(columns=ETF)
    sel = lambda d: d.loc[START:].dropna()
    return sel(ret)[sorted(ETF.values())], sel(logvol)[sorted(ETF.values())]


YAHOO_TO_GICS = {"Technology": "Information Technology", "Financial Services": "Financials",
                 "Healthcare": "Health Care", "Consumer Cyclical": "Consumer Discretionary",
                 "Consumer Defensive": "Consumer Staples", "Communication Services": "Communication Services",
                 "Industrials": "Industrials", "Energy": "Energy", "Basic Materials": "Materials",
                 "Real Estate": "Real Estate", "Utilities": "Utilities"}


def constituents():
    """Current members (Wikipedia, GICS, firm-level addition date) plus firms removed during
    the sample (point-in-time membership file), removed firms kept only if Yahoo has data."""
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers={"User-Agent": "Mozilla/5.0 academic research"}).text
    t = pd.read_html(io.StringIO(html))[0]
    t["Symbol"] = t["Symbol"].str.replace(".", "-", regex=False)
    t["start"] = pd.to_datetime(t["Date added"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
    t["end"] = pd.NaT
    t["source"] = "current"
    cur = t[["Symbol", "GICS Sector", "GICS Sub-Industry", "start", "end", "source"]]

    se = pd.read_csv(os.path.join(BASE, "data", "reference", "sp500_ticker_start_end.csv"),
                     parse_dates=["start_date", "end_date"])
    se["ticker"] = se["ticker"].str.replace(".", "-", regex=False)
    rem = se[se["end_date"].notna() & (se["end_date"] >= START) & (se["start_date"] <= END)
             & ~se["ticker"].isin(cur["Symbol"])]
    rem = rem.rename(columns={"ticker": "Symbol", "start_date": "start", "end_date": "end"})
    rem["GICS Sector"], rem["GICS Sub-Industry"], rem["source"] = None, None, "removed"
    return cur, rem[["Symbol", "GICS Sector", "GICS Sub-Industry", "start", "end", "source"]]


def split_adjusted_shares(tk: str, index: pd.DatetimeIndex) -> pd.Series | None:
    """Shares outstanding on the same split basis as yfinance's split-adjusted Close."""
    t = yf.Ticker(tk)
    s = t.get_shares_full(start=FETCH_START, end=END)
    if s is None or len(s) == 0:
        return None
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    splits = t.splits
    if splits is not None and len(splits):
        splits.index = pd.to_datetime(splits.index).tz_localize(None).normalize()
        factor = pd.Series(1.0, index=s.index)
        for d, ratio in splits.items():
            factor[s.index < d] *= ratio
        s = s * factor
    # vendor share series carry occasional one-day spikes; a rolling median removes them
    s = s.rolling(5, center=True, min_periods=1).median()
    return s.reindex(index, method="ffill").bfill()


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    print("ETF returns and range volatility...")
    ret, lv = etf_series()
    ret.to_csv(os.path.join(OUT, "etf_returns.csv"))
    lv.to_csv(os.path.join(OUT, "etf_logvol.csv"))
    print(f"  {ret.shape[0]} days {ret.index.min().date()} to {ret.index.max().date()}")

    vix = yf.download("^VIX", start=FETCH_START, end=END, progress=False, auto_adjust=False)["Close"]
    vix = vix.squeeze().loc[START:].rename("VIX")
    vix.to_csv(os.path.join(OUT, "vix.csv"))

    cur, rem = constituents()
    tickers = cur["Symbol"].tolist() + rem["Symbol"].tolist()
    print(f"constituent prices for {len(cur)} current + {len(rem)} removed tickers...")
    px = yf.download(tickers, start=FETCH_START, end=END, auto_adjust=False, progress=False)["Close"]
    px = px.dropna(axis=1, how="all")
    idx = px.index

    # a removed ticker whose price path duplicates a current one is a rename, not a separate firm
    rets_chk = px.pct_change()
    cur_cols = [c for c in cur["Symbol"] if c in px.columns]
    keep_rem, sectors_rem = [], {}
    for tk in rem["Symbol"]:
        if tk not in px.columns or px[tk].loc[START:].notna().sum() < 20:
            continue
        c = rets_chk[cur_cols].corrwith(rets_chk[tk]).max()
        if c > 0.995:
            continue
        try:
            sec = YAHOO_TO_GICS.get(yf.Ticker(tk).info.get("sector"))
        except Exception:
            sec = None
        if sec:
            keep_rem.append(tk)
            sectors_rem[tk] = sec
    rem = rem[rem["Symbol"].isin(keep_rem)].copy()
    rem["GICS Sector"] = rem["Symbol"].map(sectors_rem)
    rem.to_csv(os.path.join(OUT, "removed_firms_used.csv"), index=False)
    print(f"  removed firms recovered with data and sector: {len(rem)}")
    cons = pd.concat([cur, rem]).drop_duplicates("Symbol")
    px = px[[c for c in cons["Symbol"] if c in px.columns and c not in SECOND_CLASS]]

    print("historical shares outstanding...")
    shares, missing = {}, []
    t0 = time.time()
    for i, tk in enumerate(px.columns):
        try:
            s = split_adjusted_shares(tk, idx)
        except Exception:
            s = None
        if s is None:
            missing.append(tk)
        else:
            shares[tk] = s
        if i % 100 == 0:
            print(f"  {i}/{len(px.columns)}  {time.time() - t0:.0f}s")
    print(f"  shares found for {len(shares)}, missing {len(missing)}: {missing[:20]}")
    sh = pd.DataFrame(shares)
    mcap = (px[sh.columns] * sh).loc[START:]

    meta = cons.set_index("Symbol").loc[mcap.columns]
    # point-in-time index membership: [start, end) per firm
    member = pd.DataFrame({c: (mcap.index >= meta.loc[c, "start"]) &
                              ((mcap.index < meta.loc[c, "end"]) if pd.notna(meta.loc[c, "end"]) else True)
                           for c in mcap.columns}, index=mcap.index)
    mcap = mcap.where(member)

    sector = pd.DataFrame({c: meta.loc[c, "GICS Sector"] for c in mcap.columns}, index=mcap.index)
    pay = meta.index[meta["GICS Sub-Industry"] == "Transaction & Payment Processing Services"]
    pre = sector.index < GICS_2023
    for c in pay:
        sector.loc[pre, c] = "Information Technology"
    for c in PRE2023_CONS_DISC & set(mcap.columns):
        sector.loc[pre, c] = "Consumer Discretionary"

    print("concentration panel...")
    rows = []
    total = mcap.sum(axis=1)
    sec_names = sorted(ETF.values())
    stacked_sec = sector.stack()
    stacked_cap = mcap.stack()
    df = pd.DataFrame({"sector": stacked_sec, "mcap": stacked_cap}).dropna()
    df.index.names = ["date", "ticker"]
    df = df.reset_index()
    g = df.groupby(["date", "sector"])
    agg = g["mcap"].agg(["sum", "count"])
    df = df.join(agg, on=["date", "sector"])
    df["w"] = df["mcap"] / df["sum"]
    conc = df.groupby(["date", "sector"]).apply(
        lambda x: pd.Series({"hhi": float((x["w"] ** 2).sum()),
                             "top1": float(x["w"].max()),
                             "top3": float(x["w"].nlargest(3).sum()),
                             "n_firms": int(len(x)),
                             "sector_cap": float(x["mcap"].sum())})).reset_index()
    conc["sector_share"] = conc["sector_cap"] / conc["date"].map(total)
    conc = conc[conc["sector"].isin(sec_names)]
    conc.to_csv(os.path.join(OUT, "concentration.csv"), index=False)

    w_all = mcap.div(total, axis=0)
    idxc = pd.DataFrame({"top10_share": w_all.apply(lambda r: r.nlargest(10).sum(), axis=1),
                         "index_hhi": (w_all ** 2).sum(axis=1),
                         "n_members": mcap.notna().sum(axis=1)})
    idxc.to_csv(os.path.join(OUT, "index_conc.csv"))

    print("constituent-built (uncapped) sector returns...")
    r = np.log(px[mcap.columns]).diff().loc[START:] * 100
    wlag = mcap.shift(1)                                   # yesterday's cap: no same-day look-ahead
    cr = {}
    for s in sec_names:
        m = (sector == s) & wlag.notna() & r.notna()
        w = wlag.where(m)
        cr[s] = (r.where(m) * w).sum(axis=1) / w.sum(axis=1)
    cr = pd.DataFrame(cr).dropna()
    cr = cr.loc[cr.index.isin(ret.index)]
    cr.to_csv(os.path.join(OUT, "const_returns.csv"))

    corr = ret.loc[cr.index].corrwith(cr)
    print("ETF vs constituent-built daily return correlation:")
    print(corr.round(3).to_string())
    print(idxc.iloc[[0, -1]].round(3).to_string())


if __name__ == "__main__":
    main()
