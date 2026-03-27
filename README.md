# Sector-Level Analysis and Clustering of S&P 500 Companies Using Financial Metrics and Machine Learning

**Author:** Tanishk Yadav | NYU Tandon School of Engineering
**ORCID:** 0009-0006-2382-9411

## Abstract

Comprehensive analysis of 68 S&P 500 sub-sectors (data through Dec 2025) using five normalized financial metrics, clustering, and statistical testing. Introduces a squared-weight overperformance index validated against HHI (Pearson r = 0.537, p < 0.0001). Sector clustering (k=7) validated via leave-one-out cross-validation with Random Forest (89.7% accuracy). In-sample/out-of-sample testing shows 75% market structure stability but low growth dominance persistence (r = 0.181). Granger causality reveals 296 significant inter-sector lead-lag relationships.

## Key Results

| Metric | Value |
|---|---|
| Sectors analyzed | 68 |
| Companies covered | 400+ |
| Data range | Sep 2019 - Dec 2025 |
| Optimal clusters | 7 |
| LOO-CV Accuracy | 89.7% |
| Adjusted Rand Index | 0.840 |
| Fowlkes-Mallows Index | 0.878 |
| HHI Correlation | 0.537 |
| IS/OOS Structure Agreement | 75.0% |
| IS/OOS Growth Dominance r | 0.181 |
| Granger-causal pairs (p<0.05) | 296 |

## Project Structure

```
.
├── src/
│   ├── analysis.py              # Complete analysis pipeline
│   ├── fetch_data.py            # Data fetching (yfinance)
│   ├── fetch_data_fast.py       # Batch data fetcher
│   ├── config.py                # API key loader (reads .env)
│   └── sp500_constituents.json  # S&P 500 ticker list
├── data_mkt_cap/                # Quarterly market cap by sector (68 CSVs)
├── data_revenue/                # Revenue by sector
├── output/
│   ├── figures/                 # 9 publication-quality figures
│   └── tables/                  # All computed metrics as CSVs
├── paper/
│   ├── fig/                     # Figures for LaTeX compilation
│   ├── paper.tex                # IEEE-format LaTeX paper
│   └── paper.pdf                # Compiled PDF
├── .env                         # API keys (gitignored)
├── .gitignore
├── sector_ticker_map.json
└── README.md
```

## Running

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn yfinance
# Fetch fresh data (optional, data included)
python src/fetch_data_fast.py
# Run analysis
cd src && python -c "from analysis import run_full_analysis; run_full_analysis('..', '../output')"
```

## API Keys

FMP API keys are stored in `.env` (gitignored). Format:
```
FMP_API_KEY_1=your_key_here
```
Load in code via `from config import get_fmp_keys`.

## License

MIT
