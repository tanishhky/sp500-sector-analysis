"""
Central configuration for the S&P 500 Sector Analysis project.

CHANGE THE DATE HERE TO UPDATE THE ENTIRE PIPELINE:
"""
import os
from pathlib import Path
import pandas as pd

# ============================================================
# >>> CHANGE THIS ONE DATE TO UPDATE EVERYTHING <<<
# ============================================================
DATA_END_DATE = "2025-12-31"
# ============================================================

DATA_START_DATE = "2019-09-30"
FETCH_START = "2019-07-01"  # buffer before first quarter end

# IS/OOS split: 70% of quarters go to in-sample
_total_quarters = len(pd.date_range(DATA_START_DATE, DATA_END_DATE, freq='QE'))
_is_quarters = int(_total_quarters * 0.7)
_all_qe = pd.date_range(DATA_START_DATE, DATA_END_DATE, freq='QE')
IS_OOS_CUTOFF = str(_all_qe[_is_quarters - 1].date())  # auto-computed

# Derived dates
QUARTER_ENDS = pd.date_range(DATA_START_DATE, DATA_END_DATE, freq='QE')
FETCH_END = str((pd.Timestamp(DATA_END_DATE) + pd.Timedelta(days=15)).date())

# Minimum companies per sector
MIN_COMPANIES = 3

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_MKT_CAP = PROJECT_ROOT / "data_mkt_cap"
DATA_REVENUE = PROJECT_ROOT / "data_revenue"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIG_DIR = OUTPUT_DIR / "figures"
TBL_DIR = OUTPUT_DIR / "tables"
SECTOR_MAP_PATH = PROJECT_ROOT / "sector_ticker_map.json"

# API Keys
def load_env(env_path=None):
    if env_path is None:
        env_path = PROJECT_ROOT / '.env'
    if not env_path.exists():
        return {}
    config = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                config[key.strip()] = val.strip()
    return config

def get_fmp_keys():
    config = load_env()
    return [config[k] for k in sorted(config) if k.startswith('FMP_API_KEY')]


if __name__ == '__main__':
    print(f"Data range: {DATA_START_DATE} to {DATA_END_DATE}")
    print(f"Quarter ends: {len(QUARTER_ENDS)} quarters")
    print(f"IS/OOS cutoff: {IS_OOS_CUTOFF}")
    print(f"Fetch window: {FETCH_START} to {FETCH_END}")
    print(f"FMP keys loaded: {len(get_fmp_keys())}")
