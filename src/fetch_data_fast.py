"""
Fast batch data fetcher. Uses yf.download() for bulk price data.
All dates sourced from config.py -- change DATA_END_DATE there to update.
"""
import os, json, time
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from config import (
    QUARTER_ENDS, FETCH_START, FETCH_END,
    DATA_MKT_CAP, DATA_REVENUE, SECTOR_MAP_PATH
)

with open(SECTOR_MAP_PATH) as f:
    SECTOR_MAP = json.load(f)

ALL_TICKERS = sorted(set(t for tks in SECTOR_MAP.values() for t in tks))
print(f"Total tickers: {len(ALL_TICKERS)}")
print(f"Fetch window: {FETCH_START} to {FETCH_END}")
print(f"Quarter ends: {len(QUARTER_ENDS)} ({QUARTER_ENDS[0].date()} to {QUARTER_ENDS[-1].date()})")

# Step 1: Bulk download Close prices
print("\nDownloading bulk price data...")
prices = yf.download(ALL_TICKERS, start=FETCH_START, end=FETCH_END,
                      auto_adjust=False, progress=True, threads=True)
close = prices['Close'] if 'Close' in prices.columns.get_level_values(0) else prices
close.index = close.index.tz_localize(None) if close.index.tz else close.index
print(f"Price data: {close.shape}")

# Step 2: Get shares outstanding
print("\nFetching shares outstanding...")
shares_map = {}
for i, ticker in enumerate(ALL_TICKERS):
    if i % 50 == 0:
        print(f"  {i}/{len(ALL_TICKERS)}...")
    try:
        s = yf.Ticker(ticker).info.get('sharesOutstanding')
        if s and s > 0:
            shares_map[ticker] = s
    except:
        pass
    if i % 20 == 0:
        time.sleep(0.3)
print(f"Got shares for {len(shares_map)}/{len(ALL_TICKERS)} tickers")

# Step 3: Compute market caps at quarter ends
print("\nComputing quarterly market caps...")
records = []
for ticker in ALL_TICKERS:
    if ticker not in shares_map or ticker not in close.columns:
        continue
    series = close[ticker].dropna()
    shares = shares_map[ticker]
    for qe in QUARTER_ENDS:
        mask = series.index <= qe
        if mask.any():
            records.append({
                'Date': qe,
                'MarketCap': series[mask].iloc[-1] * shares,
                'Ticker': ticker
            })
mkt_df = pd.DataFrame(records)
print(f"Market cap records: {len(mkt_df)}")

# Step 4: Write sector mkt cap files
os.makedirs(DATA_MKT_CAP, exist_ok=True)
count = 0
for sector, tickers in SECTOR_MAP.items():
    sec = mkt_df[mkt_df['Ticker'].isin(tickers)].sort_values(['Ticker','Date'])
    if not sec.empty:
        sec.to_csv(DATA_MKT_CAP / f"{sector}_mkt_cap_quarter_end.csv", index=False)
        count += 1
print(f"Wrote {count} sector mkt cap files")

# Step 5: Fetch revenue
print("\nFetching revenue data...")
os.makedirs(DATA_REVENUE, exist_ok=True)
rev_count = 0
for si, (sector, tickers) in enumerate(sorted(SECTOR_MAP.items())):
    if si % 10 == 0:
        print(f"  Revenue: sector {si}/{len(SECTOR_MAP)}...")
    records = []
    for ticker in tickers:
        try:
            inc = yf.Ticker(ticker).quarterly_income_stmt
            if inc is None or inc.empty:
                continue
            for label in ['Total Revenue', 'Operating Revenue', 'Revenue']:
                if label in inc.index:
                    rev = inc.loc[label].sort_index()
                    for date, val in rev.items():
                        if pd.notna(val):
                            d = date.tz_localize(None) if hasattr(date, 'tz') and date.tz else date
                            records.append({'Date': d, 'Revenue': val, 'Ticker': ticker, 'company_name': ticker})
                    break
        except:
            continue
    if records:
        pd.DataFrame(records).sort_values(['Ticker','Date']).to_csv(
            DATA_REVENUE / f"{sector}_revenue.csv", index=False)
        rev_count += 1
    if si % 5 == 0:
        time.sleep(0.2)

print(f"Wrote {rev_count} sector revenue files")
print("DONE")
