"""
intl_universe.py - Constituent universes for ten non-U.S. equity markets and a current-membership
U.S. comparison, with one sector classification for all of them.

Universes are current index members: TOPIX 500 (JPX weight file), CSI 300, FTSE 350 (FTSE 100 and
FTSE 250, closed-end investment companies excluded), S&P/TSX Composite, KOSPI 200, S&P/ASX 200,
NIFTY 50 and NIFTY Next 50, DAX and MDAX, Hang Seng, Ibovespa (B3 portfolio), and the S&P 500. Sectors are
Yahoo Finance sectors mapped to the eleven GICS sectors, so every market uses the same scheme.
Writes data/intl/universe.csv.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

warnings.filterwarnings("ignore")
import yfinance as yf  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "intl")
WIKI = "https://en.wikipedia.org/wiki/"
HDR = {"User-Agent": "Mozilla/5.0 academic research"}
YAHOO_TO_GICS = {"Technology": "Information Technology", "Financial Services": "Financials",
                 "Healthcare": "Health Care", "Consumer Cyclical": "Consumer Discretionary",
                 "Consumer Defensive": "Consumer Staples", "Communication Services": "Communication Services",
                 "Industrials": "Industrials", "Energy": "Energy", "Basic Materials": "Materials",
                 "Real Estate": "Real Estate", "Utilities": "Utilities"}


def wiki_table(page: str, must: list[str]) -> pd.DataFrame:
    html = requests.get(WIKI + page, headers=HDR, timeout=60).text
    for t in pd.read_html(io.StringIO(html)):
        cols = [str(c) for c in t.columns]
        if all(any(m.lower() in c.lower() for c in cols) for m in must) and len(t) >= 20:
            t.columns = cols
            return t
    raise ValueError(f"no table with {must} on {page}")


def digits(x) -> str:
    """Numeric part of a ticker such as 'SSE: 600519', 'SEHK: 5' or 5930.0."""
    s = str(x).split(".")[0] if isinstance(x, float) else str(x)
    return "".join(ch for ch in s if ch.isdigit())


def col(t: pd.DataFrame, key: str) -> str:
    return next(c for c in t.columns if key.lower() in c.lower())


def build() -> pd.DataFrame:
    rows = []

    def add(market, symbols, names=None):
        names = names if names is not None else [None] * len(symbols)
        rows.extend({"market": market, "ticker": s, "name": n} for s, n in zip(symbols, names))

    # United States (current members only, to test the current-membership design)
    t = wiki_table("List_of_S%26P_500_companies", ["Symbol", "GICS Sector"])
    add("United States", [s.replace(".", "-") for s in t["Symbol"]], t[col(t, "Security")])

    # Japan: TOPIX 500 = Core30 + Large70 + Mid400, from the JPX constituent weight file
    jp = pd.read_csv(io.BytesIO(requests.get(
        "https://www.jpx.co.jp/automation/english/markets/indices/topix/files/topixweight_e.csv",
        headers={"User-Agent": "Mozilla/5.0"}, timeout=60).content), encoding="cp932")
    jp = jp[jp["New Index Series Code"].isin(["TOPIX Core30", "TOPIX Large70", "TOPIX Mid400"])]
    add("Japan", [f"{c}.T" for c in jp["Code"].astype(str)], jp["Issue"])

    # China A-shares: CSI 300
    t = wiki_table("CSI_300_Index", ["Ticker", "Exchange"])
    codes = [f"{int(digits(x)):06d}" for x in t[col(t, "Ticker")]]
    add("China", [c + (".SS" if c.startswith("6") else ".SZ") for c in codes], t[col(t, "Company")])

    # United Kingdom: FTSE 100 + FTSE 250, closed-end investment companies excluded (REITs kept)
    funds = {"investment trusts", "collective investments", "equity investments",
             "equity investment instruments", "closed end investments", "closed-end investments"}
    for page in ("FTSE_100_Index", "FTSE_250_Index"):
        t = wiki_table(page, ["Company", "Ticker", "sector"])
        t = t[~t[col(t, "sector")].astype(str).str.strip().str.lower().isin(funds)]
        add("United Kingdom", [str(x).strip().rstrip(".").replace(".", "-") + ".L" for x in t["Ticker"]], t["Company"])

    # Canada: S&P/TSX Composite
    t = wiki_table("S%26P/TSX_Composite_Index", ["Ticker", "Company", "Sector"])
    add("Canada", [str(x).strip().replace(".", "-") + ".TO" for x in t["Ticker"]], t["Company"])

    # Korea: KOSPI 200
    t = wiki_table("KOSPI_200", ["Company", "Symbol"])
    add("Korea", [f"{int(digits(x)):06d}.KS" for x in t["Symbol"]], t["Company"])

    # Australia: S&P/ASX 200
    t = wiki_table("S%26P/ASX_200", ["Code", "Company"])
    add("Australia", [str(x).strip() + ".AX" for x in t["Code"]], t["Company"])

    # India: NIFTY 50 + NIFTY Next 50
    for page in ("NIFTY_50", "NIFTY_Next_50"):
        t = wiki_table(page, ["Symbol", "Sector"])
        add("India", [str(x).strip() + ".NS" for x in t[col(t, "Symbol")]], t[col(t, "ompany")])

    # Germany: DAX + MDAX
    t = wiki_table("DAX", ["Ticker", "Company"])
    add("Germany", [str(x).strip().split(".")[0] + ".DE" for x in t["Ticker"]], t["Company"])
    t = wiki_table("MDAX", ["Name", "Symbol"])
    add("Germany", [str(x).strip().split(".")[0] + ".DE" for x in t["Symbol"]], t["Name"])

    # Hong Kong: Hang Seng Index
    t = wiki_table("Hang_Seng_Index", ["Ticker", "Name"])
    add("Hong Kong", [f"{int(digits(x)):04d}.HK" for x in t["Ticker"]], t["Name"])

    # Brazil: current Ibovespa portfolio from B3, one share class per firm (the larger weight)
    q = base64.b64encode(json.dumps({"language": "en-us", "pageNumber": 1, "pageSize": 200,
                                     "index": "IBOV", "segment": "1"}).encode()).decode()
    ib = pd.DataFrame(requests.get("https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/"
                                   f"GetPortfolioDay/{q}", headers=HDR, timeout=60).json()["results"])
    ib["w"] = ib["part"].str.replace(",", "").astype(float)
    ib = ib.sort_values("w", ascending=False).drop_duplicates("asset")
    add("Brazil", [c.strip() + ".SA" for c in ib["cod"]], ib["asset"])

    u = pd.DataFrame(rows).drop_duplicates(["market", "ticker"])
    return u


META = ["yahoo_sector", "industry", "quote_type", "country", "long_name", "shares_now", "currency"]


def classify(tk: str, tries: int = 3) -> dict:
    for attempt in range(tries):
        try:
            info = yf.Ticker(tk).info
            if info.get("quoteType") is None and attempt < tries - 1:
                raise ValueError("empty info")
            return {"ticker": tk, "yahoo_sector": info.get("sector"), "industry": info.get("industry"),
                    "quote_type": info.get("quoteType"), "country": info.get("country"),
                    "long_name": info.get("longName"), "shares_now": info.get("sharesOutstanding"),
                    "currency": info.get("currency")}
        except Exception:
            time.sleep(3 * (attempt + 1))
    return {"ticker": tk}


def main():
    """Builds the universe; lookups already stored in universe.csv are reused, and only tickers
    without a successful lookup are queried again, one at a time, to stay under rate limits."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "universe.csv")
    u = build()
    print(u.groupby("market").size().to_string(), flush=True)
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["ticker"] + META)
    old = old[old["quote_type"].notna()].drop_duplicates("ticker").set_index("ticker")[META]
    todo = [t for t in u["ticker"].unique() if t not in old.index]
    print(f"lookups reused {len(old)}, to query {len(todo)}", flush=True)
    new = []
    for i, tk in enumerate(todo):
        new.append(classify(tk, tries=4))
        time.sleep(0.4)
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
    meta = pd.concat([old.reset_index(), pd.DataFrame(new)], ignore_index=True)
    u = u.merge(meta, on="ticker", how="left")
    u["sector"] = u["yahoo_sector"].map(YAHOO_TO_GICS)
    # U.K. closed-end funds that Wikipedia labels as ordinary financials
    fund_name = u["long_name"].fillna("").str.contains(r"\bTrust\b|\bFund\b|Investment Trust|Investments? plc", case=False)
    u = u[~((u["market"] == "United Kingdom") & (u["industry"] == "Asset Management") & fund_name)]
    u.to_csv(path, index=False)
    print(u.groupby("market")["sector"].apply(lambda s: f"{s.notna().sum()}/{len(s)} classified").to_string())


if __name__ == "__main__":
    main()
