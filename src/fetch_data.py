"""
Data Collection: Fetches quarterly market cap + annual revenue through Dec 2025.
Uses yfinance for market cap, FMP API (with .env keys) for revenue.
"""
import os, sys, time, json, requests
import numpy as np, pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_api_keys():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith('FMP_API_KEYS='):
                    return line.strip().split('=',1)[1].split(',')
    return []

API_KEYS = load_api_keys()
_ki = [0]
_cc = [0]*max(len(API_KEYS),1)

def get_key():
    if not API_KEYS: return None
    k = API_KEYS[_ki[0]]
    _cc[_ki[0]] += 1
    if _cc[_ki[0]] >= 195: _ki[0] = (_ki[0]+1)%len(API_KEYS)
    return k

QE = pd.date_range('2019-09-30','2025-12-31',freq='QE')

def fetch_mktcap(ticker):
    try:
        stk = yf.Ticker(ticker)
        h = stk.history(start='2019-07-01',end='2026-01-15',auto_adjust=False)
        if h.empty: return None
        h.index = h.index.tz_localize(None)
        shares = None
        try: shares = stk.info.get('sharesOutstanding')
        except: pass
        if not shares:
            try:
                bs = stk.quarterly_balance_sheet
                if not bs.empty:
                    for fld in ['Ordinary Shares Number','Share Issued']:
                        if fld in bs.index:
                            shares = bs.loc[fld].dropna().iloc[0]; break
            except: pass
        if not shares or shares==0: return None
        recs = []
        for qe in QE:
            m = h.index <= qe
            if m.any():
                c = h.index[m][-1]
                recs.append({'Date':qe,'MarketCap':float(h.loc[c,'Close'])*float(shares),'Ticker':ticker})
        return pd.DataFrame(recs) if recs else None
    except: return None

def fetch_rev_fmp(ticker, co=""):
    k = get_key()
    if not k: return None
    try:
        r = requests.get(f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=annual&apikey={k}",timeout=10)
        if r.status_code!=200: return None
        data = r.json()
        if not data or isinstance(data,dict): return None
        recs = []
        for it in data:
            d,rev = it.get('date',''),it.get('revenue',0)
            yr = int(d[:4]) if d else 0
            if 2018<=yr<=2025 and rev:
                recs.append({'date':d,'quarter':4,'year':yr,'revenue':rev,'ticker':ticker,'company_name':co})
        if recs:
            df = pd.DataFrame(recs).sort_values('year')
            df['revenue_yoy_growth'] = df['revenue'].pct_change()*100
            return df
    except: pass
    return None

def fetch_rev_yf(ticker, co=""):
    try:
        inc = yf.Ticker(ticker).income_stmt
        if inc is None or inc.empty: return None
        for fld in ['Total Revenue','Revenue']:
            if fld in inc.index:
                row = inc.loc[fld]; break
        else: return None
        recs = []
        for dt,v in row.items():
            if pd.notna(v):
                d = pd.to_datetime(dt)
                if 2018<=d.year<=2025:
                    recs.append({'date':d.strftime('%Y-%m-%d'),'quarter':4,'year':d.year,'revenue':float(v),'ticker':ticker,'company_name':co})
        if recs:
            df = pd.DataFrame(recs).sort_values('year')
            df['revenue_yoy_growth'] = df['revenue'].pct_change()*100
            return df
    except: pass
    return None

def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base,'src','sp500_constituents.json')) as f:
        sector_map = json.load(f)
    
    all_tickers = []
    for sec,tks in sector_map.items():
        for t in tks: all_tickers.append((t,sec))
    
    print(f"Fetching data for {len(all_tickers)} tickers across {len(sector_map)} sectors")
    
    # Market cap (threaded)
    print("\n--- Market Cap ---")
    mkt = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(fetch_mktcap,t):t for t,_ in all_tickers}
        dn = 0
        for f in as_completed(futs):
            dn += 1
            t = futs[f]
            try:
                df = f.result()
                if df is not None and not df.empty: mkt[t] = df
            except: pass
            if dn%50==0: print(f"  {dn}/{len(all_tickers)} ({len(mkt)} ok)")
    print(f"  Done: {len(mkt)}/{len(all_tickers)}")
    
    # Revenue
    print("\n--- Revenue ---")
    rev = {}
    dn = 0
    for t,_ in all_tickers:
        dn += 1
        df = fetch_rev_fmp(t)
        if df is None: df = fetch_rev_yf(t)
        if df is not None and not df.empty: rev[t] = df
        if dn%50==0: print(f"  {dn}/{len(all_tickers)} ({len(rev)} ok)")
        time.sleep(0.12)
    print(f"  Done: {len(rev)}/{len(all_tickers)}")
    
    # Save by sector
    mkt_dir = os.path.join(base,'data_mkt_cap')
    rev_dir = os.path.join(base,'data_revenue')
    for d in [mkt_dir,rev_dir]:
        os.makedirs(d,exist_ok=True)
        for f in os.listdir(d): os.remove(os.path.join(d,f))
    
    for sec,tks in sector_map.items():
        sdfs = [mkt[t] for t in tks if t in mkt]
        if sdfs:
            pd.concat(sdfs,ignore_index=True).to_csv(os.path.join(mkt_dir,f"{sec}_mkt_cap_quarter_end.csv"),index=False)
        rdfs = [rev[t] for t in tks if t in rev]
        if rdfs:
            pd.concat(rdfs,ignore_index=True).to_csv(os.path.join(rev_dir,f"{sec}_revenue.csv"),index=False)
    
    mc = len([f for f in os.listdir(mkt_dir) if f.endswith('.csv')])
    rc = len([f for f in os.listdir(rev_dir) if f.endswith('.csv')])
    print(f"\nSaved: {mc} mkt cap files, {rc} revenue files")

if __name__=='__main__':
    main()
