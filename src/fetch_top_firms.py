"""
fetch_top_firms.py - Prices and split-adjusted shares for the firms that can hold a top-three
position in any GICS sector between October 2018 and August 2026.

The candidate list is broad on purpose: the three largest firms in each sector on every date
must be among the candidates, which granular_decomposition.py checks against the full
constituent panel (data/panel/concentration.csv). Same vendor and share treatment as
build_panel.py. Raw vendor data go to data/firm/ (gitignored).
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yfinance as yf  # noqa: E402

from build_panel import END, FETCH_START, split_adjusted_shares  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "firm")

CANDIDATES = {
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "TMUS", "VZ", "T", "CMCSA", "CHTR", "EA", "TTWO"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX", "TGT", "DG", "DLTR"],
    "Consumer Staples": ["WMT", "PG", "COST", "KO", "PEP", "PM", "MO", "MDLZ", "CL"],
    "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "PSX", "MPC", "KMI", "WMB"],
    "Financials": ["BRK-B", "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "AXP", "SPGI", "V", "MA", "PYPL", "FIS", "GPN"],
    "Health Care": ["LLY", "UNH", "JNJ", "PFE", "MRK", "ABBV", "TMO", "ABT", "AMGN", "ISRG", "DHR"],
    "Industrials": ["GE", "CAT", "RTX", "UNP", "HON", "BA", "UPS", "LMT", "MMM", "DE", "ETN", "UBER", "GEV"],
    "Information Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "INTC", "ADBE", "CRM", "AMD", "PLTR", "ACN", "TXN", "QCOM"],
    "Materials": ["LIN", "SHW", "APD", "ECL", "FCX", "DD", "DOW", "NEM", "CTVA", "NUE"],
    "Real Estate": ["PLD", "AMT", "EQIX", "WELL", "CCI", "SPG", "PSA", "O", "DLR"],
    "Utilities": ["NEE", "SO", "DUK", "CEG", "D", "AEP", "VST", "EXC", "SRE"],
}


def main():
    os.makedirs(OUT, exist_ok=True)
    tickers = sorted({t for v in CANDIDATES.values() for t in v})
    px = yf.download(tickers, start=FETCH_START, end=END, auto_adjust=False, progress=False)["Close"]
    px = px.dropna(axis=1, how="all")
    px.to_csv(os.path.join(OUT, "prices.csv"))
    shares, missing = {}, []
    t0 = time.time()
    for i, tk in enumerate(px.columns):
        try:
            s = split_adjusted_shares(tk, px.index)
        except Exception:
            s = None
        if s is None:
            missing.append(tk)
        else:
            shares[tk] = s
        if i % 20 == 0:
            print(f"{i}/{len(px.columns)} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(shares).to_csv(os.path.join(OUT, "shares.csv"))
    print("missing shares:", missing)
    print("missing prices:", sorted(set(tickers) - set(px.columns)))


if __name__ == "__main__":
    main()
