"""
build_sector_returns.py - Build daily GICS-sector return series for the
time-series (connectedness) analysis.

Maps the 69 GICS sub-sectors to the 11 GICS sectors, fetches daily adjusted
prices for all constituents (one batched yfinance call), and aggregates to
quarterly-rebalanced market-cap-weighted daily returns per sector. The 25
quarterly observations in data_mkt_cap/ are far too few for VAR / Granger /
connectedness; this produces ~1,500 daily observations.

Output: data/sector_returns_daily.csv  (Date index, 11 sector columns)
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import yfinance as yf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 69 GICS sub-sector  ->  11 GICS sector
GICS = {
    "Aerospace & Defense": "Industrials",
    "Air Freight & Logistics": "Industrials",
    "Building Products": "Industrials",
    "Construction Machinery & Heavy Transportation Equipment": "Industrials",
    "Diversified Support Services": "Industrials",
    "Electrical Components & Equipment": "Industrials",
    "Environmental & Facilities Services": "Industrials",
    "Human Resource & Employment Services": "Industrials",
    "Industrial Machinery & Supplies & Components": "Industrials",
    "Passenger Airlines": "Industrials",
    "Rail Transportation": "Industrials",
    "Application Software": "Information Technology",
    "Communications Equipment": "Information Technology",
    "Electronic Equipment & Instruments": "Information Technology",
    "IT Consulting & Other Services": "Information Technology",
    "Internet Services & Infrastructure": "Information Technology",
    "Semiconductor Materials & Equipment": "Information Technology",
    "Semiconductors": "Information Technology",
    "Systems Software": "Information Technology",
    "Technology Hardware, Storage & Peripherals": "Information Technology",
    "Asset Management & Custody Banks": "Financials",
    "Consumer Finance": "Financials",
    "Diversified Banks": "Financials",
    "Financial Exchanges & Data": "Financials",
    "Insurance Brokers": "Financials",
    "Investment Banking & Brokerage": "Financials",
    "Life & Health Insurance": "Financials",
    "Multi-line Insurance": "Financials",
    "Property & Casualty Insurance": "Financials",
    "Regional Banks": "Financials",
    "Transaction & Payment Processing Services": "Financials",
    "Biotechnology": "Health Care",
    "Health Care Distributors": "Health Care",
    "Health Care Equipment": "Health Care",
    "Health Care Services": "Health Care",
    "Health Care Supplies": "Health Care",
    "Life Sciences Tools & Services": "Health Care",
    "Managed Health Care": "Health Care",
    "Pharmaceuticals": "Health Care",
    "Apparel, Accessories & Luxury Goods": "Consumer Discretionary",
    "Automobile Manufacturers": "Consumer Discretionary",
    "Automotive Retail": "Consumer Discretionary",
    "Casinos & Gaming": "Consumer Discretionary",
    "Distributors": "Consumer Discretionary",
    "Homebuilding": "Consumer Discretionary",
    "Hotels, Resorts & Cruise Lines": "Consumer Discretionary",
    "Restaurants": "Consumer Discretionary",
    "Consumer Staples Merchandise Retail": "Consumer Staples",
    "Household Products": "Consumer Staples",
    "Packaged Foods & Meats": "Consumer Staples",
    "Personal Care Products": "Consumer Staples",
    "Soft Drinks & Non-alcoholic Beverages": "Consumer Staples",
    "Integrated Oil & Gas": "Energy",
    "Oil & Gas Equipment & Services": "Energy",
    "Oil & Gas Exploration & Production": "Energy",
    "Oil & Gas Refining & Marketing": "Energy",
    "Oil & Gas Storage & Transportation": "Energy",
    "Fertilizers & Agricultural Chemicals": "Materials",
    "Paper & Plastic Packaging Products & Materials": "Materials",
    "Specialty Chemicals": "Materials",
    "Health Care REITs": "Real Estate",
    "Multi-Family Residential REITs": "Real Estate",
    "Retail REITs": "Real Estate",
    "Telecom Tower REITs": "Real Estate",
    "Electric Utilities": "Utilities",
    "Multi-Utilities": "Utilities",
    "Broadcasting": "Communication Services",
    "Interactive Media & Services": "Communication Services",
    "Movies & Entertainment": "Communication Services",
}

START, END = "2019-09-01", "2025-12-31"


def _quarterly_caps() -> dict[str, pd.Series]:
    """ticker -> Series(index=quarter-end date, value=market cap)."""
    caps: dict[str, pd.Series] = {}
    mkt_dir = os.path.join(BASE, "data_mkt_cap")
    for fn in os.listdir(mkt_dir):
        if not fn.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(mkt_dir, fn), parse_dates=["Date"])
        for tkr, g in df.groupby("Ticker"):
            caps[tkr] = g.set_index("Date")["MarketCap"].sort_index()
    return caps


def main() -> None:
    constituents = json.load(open(os.path.join(BASE, "src", "sp500_constituents.json")))
    ticker_sector = {t: GICS[sub] for sub, tks in constituents.items() if sub in GICS for t in tks}
    tickers = sorted(ticker_sector)
    print(f"{len(tickers)} tickers across {len(set(ticker_sector.values()))} GICS sectors")

    print("Downloading daily prices (one batched call)...")
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.dropna(axis=1, how="all")
    rets = close.pct_change().iloc[1:]
    # drop tickers with too little data
    good = [t for t in rets.columns if rets[t].notna().sum() > 0.5 * len(rets)]
    rets = rets[good]
    print(f"usable tickers: {len(good)}  trading days: {len(rets)}")

    # Daily quarterly-rebalanced cap weights (most recent quarter cap, ffilled)
    caps = _quarterly_caps()
    wts = pd.DataFrame(index=rets.index, columns=good, dtype=float)
    for t in good:
        if t in caps:
            wts[t] = caps[t].reindex(rets.index, method="ffill").bfill()
        else:
            wts[t] = np.nan
    wts = wts.where(rets.notna())  # only weight days with a return

    # Cap-weighted daily return per GICS sector
    sectors = sorted(set(ticker_sector[t] for t in good))
    out = pd.DataFrame(index=rets.index)
    for sec in sectors:
        cols = [t for t in good if ticker_sector[t] == sec]
        w = wts[cols]
        r = rets[cols]
        out[sec] = (r * w).sum(axis=1) / w.sum(axis=1)
    out = out.dropna()

    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
    path = os.path.join(BASE, "data", "sector_returns_daily.csv")
    out.to_csv(path)
    print(f"\nSaved {out.shape[0]} days x {out.shape[1]} sectors -> {path}")
    print("annualized vol by sector:")
    print((out.std() * np.sqrt(252)).round(3).sort_values().to_string())


if __name__ == "__main__":
    main()
