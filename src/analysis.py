"""
Sector-Level Analysis and Clustering of S&P 500 Companies
Using Financial Metrics and Machine Learning

Complete analysis pipeline with proper statistical methodology.
Author: Tanishk Yadav | NYU Tandon School of Engineering
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, classification_report,
    confusion_matrix, adjusted_rand_score,
    normalized_mutual_info_score, v_measure_score,
    fowlkes_mallows_score
)
from sklearn.model_selection import LeaveOneOut
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

warnings.filterwarnings('ignore')
np.random.seed(42)


# ============================================================
# 1. DATA LOADING AND VALIDATION
# ============================================================

def load_sector_data(mkt_cap_dir, revenue_dir, min_companies=3):
    """
    Load and merge market cap and revenue data for all sectors.
    Only keeps sectors with at least `min_companies` constituents
    and consistent date ranges.
    """
    mkt_sectors = {
        f.replace('_mkt_cap_quarter_end.csv', ''): f
        for f in os.listdir(mkt_cap_dir) if f.endswith('.csv')
    }
    rev_sectors = {
        f.replace('_revenue.csv', ''): f
        for f in os.listdir(revenue_dir) if f.endswith('.csv')
    }

    common = set(mkt_sectors.keys()) & set(rev_sectors.keys())
    print(f"Sectors with both mkt cap and revenue data: {len(common)}")

    sector_data = {}
    skipped = []

    for sector in sorted(common):
        # Load market cap
        mkt = pd.read_csv(os.path.join(mkt_cap_dir, mkt_sectors[sector]))
        mkt['Date'] = pd.to_datetime(mkt['Date'], utc=True).dt.tz_localize(None)
        mkt['Date'] = mkt['Date'].dt.normalize()

        n_companies = mkt['Ticker'].nunique()
        if n_companies < min_companies:
            skipped.append((sector, n_companies))
            continue

        # Load revenue
        rev = pd.read_csv(os.path.join(revenue_dir, rev_sectors[sector]))
        if 'date' in rev.columns:
            rev = rev.rename(columns={'date': 'Date', 'ticker': 'Ticker',
                                       'revenue': 'Revenue'})
        rev['Date'] = pd.to_datetime(rev['Date']).dt.normalize()

        sector_data[sector] = {
            'mkt_cap': mkt,
            'revenue': rev,
            'n_companies': n_companies,
            'date_range': (mkt['Date'].min(), mkt['Date'].max())
        }

    print(f"Sectors passing min_companies={min_companies} filter: {len(sector_data)}")
    if skipped:
        print(f"Skipped {len(skipped)} sectors: {skipped}")

    return sector_data


# ============================================================
# 2. FINANCIAL METRIC COMPUTATION (CORRECTED)
# ============================================================

def compute_yoy_mktcap_growth(mkt_df):
    """
    Compute YoY market cap growth for each company.
    Uses pct_change(4) on quarterly data = 4-quarter lag = 1 year.
    """
    df = mkt_df.sort_values(['Ticker', 'Date']).copy()
    df['MktCap_YoY'] = df.groupby('Ticker')['MarketCap'].pct_change(periods=4) * 100
    return df.dropna(subset=['MktCap_YoY'])


def compute_growth_score(series, threshold_sigma=0.5):
    """
    Score growth relative to series distribution.

    CORRECTION: Uses 0.5 sigma threshold (not 0.07 which is essentially zero).
    At 0.5 sigma, ~38% of observations fall in neutral band, ~31% in each tail.
    This creates meaningful differentiation.

    Parameters
    ----------
    series : pd.Series of YoY growth values
    threshold_sigma : float, number of std deviations for threshold

    Returns
    -------
    float : mean score in [-1, 1]
    """
    mu = series.mean()
    sigma = series.std()

    if sigma == 0 or np.isnan(sigma):
        return 0.0

    high = mu + threshold_sigma * sigma
    low = mu - threshold_sigma * sigma

    scores = np.where(series > high, 1, np.where(series < low, -1, 0))
    return float(scores.mean())


def compute_weighted_simple_variance(mkt_df):
    """
    Compute the overperformance-weighted vs simple average variance.

    The weight w_i = c_i^2 / sum(c_j^2) where c_i is the count of times
    company i outperformed the sector mean.

    This metric captures market concentration: large values indicate
    monopolistic tendencies (few firms drive sector performance).
    """
    df = compute_yoy_mktcap_growth(mkt_df)

    if df.empty:
        return 0.0, {}, 0

    # Compute sector mean per quarter
    sector_avg = df.groupby('Date')['MktCap_YoY'].mean()

    # Count outperformance per company
    overperf_counts = {}
    for ticker in df['Ticker'].unique():
        company = df[df['Ticker'] == ticker].set_index('Date')
        aligned = company['MktCap_YoY'].reindex(sector_avg.index)
        count = (aligned > sector_avg).sum()
        overperf_counts[ticker] = count

    total_sq = sum(c ** 2 for c in overperf_counts.values())
    if total_sq == 0:
        return 0.0, overperf_counts, 0

    # Weighted average growth
    weights = {t: c ** 2 / total_sq for t, c in overperf_counts.items()}

    # Compute weighted vs simple avg per date
    dates = df['Date'].unique()
    diffs = []
    for date in dates:
        day_data = df[df['Date'] == date]
        simple_avg = day_data['MktCap_YoY'].mean()

        weighted_avg = 0
        for _, row in day_data.iterrows():
            w = weights.get(row['Ticker'], 0)
            weighted_avg += w * row['MktCap_YoY']

        diffs.append(abs(weighted_avg - simple_avg))

    variance = np.mean(diffs) if diffs else 0.0
    return variance, overperf_counts, total_sq


def compute_beta(mkt_df, period_quarters):
    """
    Compute sector beta using PRICE RETURNS (not market cap changes).

    CORRECTION: Beta should measure price sensitivity to market.
    Since we only have market cap data, we use market-cap-weighted
    sector returns vs equal-weighted market returns as the benchmark.

    Parameters
    ----------
    mkt_df : DataFrame with Date, MarketCap, Ticker
    period_quarters : int, number of quarters for lookback

    Returns
    -------
    float : sector beta coefficient
    """
    df = mkt_df.sort_values(['Ticker', 'Date']).copy()

    # Compute per-company returns
    df['Returns'] = df.groupby('Ticker')['MarketCap'].pct_change()
    df = df.dropna(subset=['Returns'])

    if len(df) < period_quarters:
        return np.nan

    # Use only last N quarters
    cutoff_dates = sorted(df['Date'].unique())
    if len(cutoff_dates) > period_quarters:
        cutoff_dates = cutoff_dates[-period_quarters:]
    df = df[df['Date'].isin(cutoff_dates)]

    # Market-cap-weighted sector return per date
    sector_returns = []
    market_returns = []

    for date in sorted(df['Date'].unique()):
        day = df[df['Date'] == date]
        if day.empty:
            continue

        # Sector return: market-cap-weighted
        total_mkt = day['MarketCap'].sum()
        if total_mkt > 0:
            w_ret = (day['MarketCap'] * day['Returns']).sum() / total_mkt
        else:
            w_ret = day['Returns'].mean()

        # Equal-weighted as simple benchmark
        eq_ret = day['Returns'].mean()

        sector_returns.append(w_ret)
        market_returns.append(eq_ret)

    if len(sector_returns) < 3:
        return np.nan

    sector_returns = np.array(sector_returns)
    market_returns = np.array(market_returns)

    cov = np.cov(sector_returns, market_returns)
    if cov[1, 1] == 0:
        return np.nan

    beta = cov[0, 1] / cov[1, 1]
    return beta


def compute_revenue_growth_score(rev_df, threshold_sigma=0.5):
    """Compute revenue YoY growth score for a sector."""
    df = rev_df.copy()

    if 'revenue_yoy_growth' in df.columns:
        growth = df['revenue_yoy_growth'].dropna()
    elif 'Revenue' in df.columns:
        df = df.sort_values(['Ticker', 'Date'])
        df['Rev_YoY'] = df.groupby('Ticker')['Revenue'].pct_change() * 100
        growth = df['Rev_YoY'].dropna()
    else:
        return 0.0

    if len(growth) < 2:
        return 0.0

    return compute_growth_score(growth, threshold_sigma)


def compute_all_metrics(sector_data):
    """
    Compute all 5 metrics for each sector.

    Returns DataFrame with columns:
    - Sector
    - MktCap_Growth_Score
    - Revenue_Growth_Score
    - Weighted_Simple_Variance
    - Beta_Short (2 quarters ~ 6 months)
    - Beta_Long (16 quarters ~ 4 years)
    - N_Companies
    - Date_Start, Date_End
    """
    records = []

    for sector, data in sector_data.items():
        mkt = data['mkt_cap']
        rev = data['revenue']

        # 1. Market Cap Growth Score
        growth_df = compute_yoy_mktcap_growth(mkt)
        mktcap_score = compute_growth_score(
            growth_df['MktCap_YoY'], threshold_sigma=0.5
        ) if not growth_df.empty else 0.0

        # 2. Revenue Growth Score
        rev_score = compute_revenue_growth_score(rev, threshold_sigma=0.5)

        # 3. Weighted-Simple Variance (overperformance concentration)
        ws_var, overperf_counts, _ = compute_weighted_simple_variance(mkt)

        # 4. Short-term beta (6 months = 2 quarters)
        beta_short = compute_beta(mkt, period_quarters=2)

        # 5. Long-term beta (4 years = 16 quarters)
        beta_long = compute_beta(mkt, period_quarters=16)

        records.append({
            'Sector': sector,
            'MktCap_Growth_Score': mktcap_score,
            'Revenue_Growth_Score': rev_score,
            'Weighted_Simple_Variance': ws_var,
            'Beta_Short': beta_short if not np.isnan(beta_short) else 0.0,
            'Beta_Long': beta_long if not np.isnan(beta_long) else 0.0,
            'N_Companies': data['n_companies'],
            'Date_Start': data['date_range'][0],
            'Date_End': data['date_range'][1],
        })

    df = pd.DataFrame(records)
    df = df.sort_values('Sector').reset_index(drop=True)
    return df


# ============================================================
# 3. OVERPERFORMANCE INDEX WITH STATISTICAL TESTING
# ============================================================

def compute_overperformance_index(sector_data):
    """
    Compute overperformance rankings with binomial significance tests.

    For each company, count how many times it beats the sector mean.
    Under H0 (no skill), P(beat) = 0.5 each period.
    Test using binomial test: p-value for observing >= count successes.

    Returns DataFrame with Sector, Ticker, Count, P_Value, Significant,
    and market structure classification.
    """
    all_records = []

    for sector, data in sector_data.items():
        mkt = data['mkt_cap']
        df = compute_yoy_mktcap_growth(mkt)

        if df.empty:
            continue

        sector_avg = df.groupby('Date')['MktCap_YoY'].mean()

        for ticker in df['Ticker'].unique():
            company = df[df['Ticker'] == ticker].set_index('Date')
            aligned = company['MktCap_YoY'].reindex(sector_avg.index).dropna()
            aligned_avg = sector_avg.reindex(aligned.index)

            n_obs = len(aligned)
            n_beats = int((aligned > aligned_avg).sum())

            # Binomial test: H0 is p=0.5 (random outperformance)
            if n_obs > 0:
                p_val = stats.binomtest(n_beats, n_obs, 0.5,
                                         alternative='greater').pvalue
            else:
                p_val = 1.0

            all_records.append({
                'Sector': sector,
                'Ticker': ticker,
                'Outperformance_Count': n_beats,
                'N_Observations': n_obs,
                'Outperformance_Rate': n_beats / n_obs if n_obs > 0 else 0,
                'P_Value': p_val,
                'Significant': p_val < 0.05
            })

    overperf_df = pd.DataFrame(all_records)

    # Classify market structure per sector
    structure_records = []
    for sector in overperf_df['Sector'].unique():
        sec = overperf_df[overperf_df['Sector'] == sector].copy()
        sec = sec.sort_values('Outperformance_Count', ascending=False)

        n_significant = sec['Significant'].sum()
        n_companies = len(sec)

        # Classification logic:
        # Monopolistic: 1 company has significantly more outperformance
        # Duopolistic: 2 companies dominate
        # Oligopolistic: no clear dominance
        if n_companies == 0:
            structure = 'Undefined'
        elif n_significant <= 1 and n_companies >= 3:
            # Check concentration: does top company have >40% of total count?
            total_count = sec['Outperformance_Count'].sum()
            top_share = sec.iloc[0]['Outperformance_Count'] / total_count if total_count > 0 else 0

            if top_share > 0.35:
                structure = 'Monopolistic'
            elif n_companies <= 3:
                structure = 'Duopolistic'
            else:
                structure = 'Oligopolistic'
        elif n_significant == 2:
            structure = 'Duopolistic'
        else:
            structure = 'Oligopolistic'

        # Also use HHI on outperformance counts as a cross-check
        total_count = sec['Outperformance_Count'].sum()
        if total_count > 0:
            shares = sec['Outperformance_Count'] / total_count
            hhi_overperf = (shares ** 2).sum()
        else:
            hhi_overperf = 0

        # HHI-based classification (standard DOJ thresholds adapted)
        if hhi_overperf > 0.25:
            hhi_structure = 'Monopolistic'
        elif hhi_overperf > 0.15:
            hhi_structure = 'Duopolistic'
        else:
            hhi_structure = 'Oligopolistic'

        structure_records.append({
            'Sector': sector,
            'Market_Structure_Binomial': structure,
            'Market_Structure_HHI': hhi_structure,
            'HHI_Overperformance': hhi_overperf,
            'N_Significant': n_significant,
            'N_Companies': n_companies,
            'Top_Company': sec.iloc[0]['Ticker'] if len(sec) > 0 else '',
            'Top_Count': sec.iloc[0]['Outperformance_Count'] if len(sec) > 0 else 0,
            'Ranking': ';'.join(
                f"{r['Ticker']}:{r['Outperformance_Count']}"
                for _, r in sec.iterrows()
            )
        })

    structure_df = pd.DataFrame(structure_records)
    return overperf_df, structure_df


# ============================================================
# 4. HHI (HERFINDAHL-HIRSCHMAN INDEX) FROM MARKET CAP
# ============================================================

def compute_hhi_from_mktcap(sector_data):
    """
    Compute traditional HHI from market cap shares.
    HHI = sum(s_i^2) where s_i = company_mktcap / sector_total_mktcap.

    Uses the LATEST available quarter for each sector.
    """
    records = []
    for sector, data in sector_data.items():
        mkt = data['mkt_cap']
        latest_date = mkt['Date'].max()
        latest = mkt[mkt['Date'] == latest_date]

        total = latest['MarketCap'].sum()
        if total > 0:
            shares = latest['MarketCap'] / total
            hhi = (shares ** 2).sum()
        else:
            hhi = 0

        records.append({
            'Sector': sector,
            'HHI_MktCap': hhi,
            'N_Companies': len(latest),
            'Top_Company': latest.loc[latest['MarketCap'].idxmax(), 'Ticker'] if len(latest) > 0 else '',
            'Top_Share': shares.max() if total > 0 else 0
        })

    return pd.DataFrame(records)


# ============================================================
# 5. CLUSTERING WITH PROPER VALIDATION
# ============================================================

def perform_clustering(metrics_df, max_k=10):
    """
    Cluster sectors using hierarchical + K-Means with proper validation.

    1. Standardize features
    2. Hierarchical clustering (Ward) for dendrogram
    3. Select optimal k using silhouette score
    4. K-Means with optimal k
    5. Validate with multiple metrics
    6. Leave-one-out Random Forest for classification accuracy

    Returns: cluster labels, validation metrics, optimal k, scaler, linkage matrix
    """
    features = ['MktCap_Growth_Score', 'Revenue_Growth_Score',
                'Weighted_Simple_Variance', 'Beta_Short', 'Beta_Long']

    X = metrics_df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hierarchical clustering
    Z = linkage(X_scaled, method='ward')

    # Find optimal k via silhouette
    silhouette_scores = {}
    for k in range(2, min(max_k + 1, len(X_scaled))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        silhouette_scores[k] = sil

    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"Optimal k by silhouette: {optimal_k} (score={silhouette_scores[optimal_k]:.3f})")
    print(f"All silhouette scores: {silhouette_scores}")

    # Final K-Means
    km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(X_scaled)

    # Validation metrics
    validation = {
        'Silhouette': silhouette_score(X_scaled, cluster_labels),
        'Calinski_Harabasz': calinski_harabasz_score(X_scaled, cluster_labels),
        'Davies_Bouldin': davies_bouldin_score(X_scaled, cluster_labels),
    }

    # Leave-One-Out cross-validation with Random Forest
    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    for train_idx, test_idx in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train = cluster_labels[train_idx]

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)

        y_true.append(cluster_labels[test_idx[0]])
        y_pred.append(pred[0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    loo_accuracy = (y_true == y_pred).mean()
    validation['LOO_Accuracy'] = loo_accuracy
    validation['ARI'] = adjusted_rand_score(y_true, y_pred)
    validation['NMI'] = normalized_mutual_info_score(y_true, y_pred)
    validation['V_Measure'] = v_measure_score(y_true, y_pred)
    validation['Fowlkes_Mallows'] = fowlkes_mallows_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    print(f"\nLeave-One-Out CV Accuracy: {loo_accuracy:.1%}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Validation metrics: {validation}")

    return {
        'labels': cluster_labels,
        'optimal_k': optimal_k,
        'silhouette_scores': silhouette_scores,
        'validation': validation,
        'confusion_matrix': cm,
        'linkage': Z,
        'scaler': scaler,
        'X_scaled': X_scaled,
        'y_true': y_true,
        'y_pred': y_pred,
    }


# ============================================================
# 6. TIME-VARYING ANALYSIS (ROLLING WINDOW)
# ============================================================

def compute_rolling_overperformance(sector_data, window_quarters=8):
    """
    Compute rolling-window overperformance concentration.

    For each sector, compute the HHI of outperformance counts
    in a rolling window to show how market structure evolves.
    """
    results = {}

    for sector, data in sector_data.items():
        mkt = data['mkt_cap']
        df = compute_yoy_mktcap_growth(mkt)
        if df.empty:
            continue

        dates = sorted(df['Date'].unique())
        if len(dates) < window_quarters:
            continue

        rolling_hhi = []
        rolling_dates = []

        for i in range(window_quarters, len(dates) + 1):
            window_dates = dates[i - window_quarters:i]
            window_df = df[df['Date'].isin(window_dates)]

            sector_avg = window_df.groupby('Date')['MktCap_YoY'].mean()

            counts = {}
            for ticker in window_df['Ticker'].unique():
                comp = window_df[window_df['Ticker'] == ticker].set_index('Date')
                aligned = comp['MktCap_YoY'].reindex(sector_avg.index).dropna()
                aligned_avg = sector_avg.reindex(aligned.index)
                counts[ticker] = int((aligned > aligned_avg).sum())

            total = sum(counts.values())
            if total > 0:
                shares = np.array(list(counts.values())) / total
                hhi = (shares ** 2).sum()
            else:
                hhi = 0

            rolling_hhi.append(hhi)
            rolling_dates.append(window_dates[-1])

        results[sector] = pd.DataFrame({
            'Date': rolling_dates,
            'Rolling_HHI': rolling_hhi
        })

    return results


# ============================================================
# 7. GRANGER CAUSALITY BETWEEN SECTORS
# ============================================================

def compute_sector_return_series(sector_data):
    """Compute quarterly market-cap-weighted return for each sector."""
    sector_returns = {}

    for sector, data in sector_data.items():
        mkt = data['mkt_cap'].copy()
        mkt = mkt.sort_values(['Ticker', 'Date'])
        mkt['Return'] = mkt.groupby('Ticker')['MarketCap'].pct_change()
        mkt = mkt.dropna(subset=['Return'])

        # Market-cap-weighted return per quarter
        quarterly = []
        for date in sorted(mkt['Date'].unique()):
            day = mkt[mkt['Date'] == date]
            total_mkt = day['MarketCap'].sum()
            if total_mkt > 0:
                wr = (day['MarketCap'] * day['Return']).sum() / total_mkt
            else:
                wr = 0
            quarterly.append({'Date': date, 'Return': wr})

        if quarterly:
            sector_returns[sector] = pd.DataFrame(quarterly).set_index('Date')['Return']

    return sector_returns


def granger_causality_matrix(sector_returns, max_lag=2, alpha=0.05):
    """
    Test pairwise Granger causality between sector returns.

    Returns matrix of p-values where entry (i,j) is the p-value
    for "sector j Granger-causes sector i".
    """
    sectors = sorted(sector_returns.keys())

    # Align all series to common dates
    all_df = pd.DataFrame(sector_returns)
    all_df = all_df.dropna()

    if len(all_df) < max_lag + 5:
        print("Warning: insufficient data for Granger causality tests")
        return pd.DataFrame(), []

    n = len(sectors)
    p_matrix = np.ones((n, n))
    significant_pairs = []

    for i, s1 in enumerate(sectors):
        for j, s2 in enumerate(sectors):
            if i == j:
                continue
            try:
                test_data = all_df[[s1, s2]].dropna()
                if len(test_data) < max_lag + 5:
                    continue
                result = grangercausalitytests(
                    test_data[[s1, s2]], maxlag=max_lag, verbose=False
                )
                # Get minimum p-value across lags
                min_p = min(
                    result[lag][0]['ssr_ftest'][1]
                    for lag in range(1, max_lag + 1)
                )
                p_matrix[i, j] = min_p
                if min_p < alpha:
                    significant_pairs.append((s2, s1, min_p))
            except Exception:
                pass

    p_df = pd.DataFrame(p_matrix, index=sectors, columns=sectors)
    significant_pairs.sort(key=lambda x: x[2])
    return p_df, significant_pairs


# ============================================================
# 8. VISUALIZATION
# ============================================================

def plot_dendrogram(Z, sectors, output_path):
    """Plot hierarchical clustering dendrogram."""
    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(
        Z,
        labels=sectors,
        leaf_rotation=90,
        leaf_font_size=7,
        color_threshold=0,
        ax=ax
    )
    ax.set_title('Hierarchical Clustering of S&P 500 Sub-Sectors', fontsize=14)
    ax.set_xlabel('Sector', fontsize=11)
    ax.set_ylabel('Ward Linkage Distance', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_silhouette_analysis(silhouette_scores, optimal_k, output_path):
    """Plot silhouette scores for different k values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = sorted(silhouette_scores.keys())
    scores = [silhouette_scores[k] for k in ks]
    ax.plot(ks, scores, 'o-', color='#2E75B6', linewidth=2, markersize=8)
    ax.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7,
               label=f'Optimal k={optimal_k}')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Analysis for Optimal k', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_cluster_pca(X_scaled, labels, sectors, output_path):
    """2D PCA projection of clustered sectors."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(12, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for k, color in zip(unique_labels, colors):
        mask = labels == k
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], label=f'Cluster {k}',
                   s=80, edgecolors='black', linewidth=0.5)
        for idx in np.where(mask)[0]:
            ax.annotate(sectors[idx], (X_2d[idx, 0], X_2d[idx, 1]),
                        fontsize=5.5, alpha=0.8,
                        xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)', fontsize=12)
    ax.set_title('PCA Projection of Sector Clusters', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_market_structure_distribution(structure_df, output_path):
    """Bar chart of market structure classification."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, col in enumerate(['Market_Structure_Binomial', 'Market_Structure_HHI']):
        counts = structure_df[col].value_counts()
        title = 'Binomial Test' if 'Binomial' in col else 'HHI-Based'
        colors = {'Monopolistic': '#E74C3C', 'Duopolistic': '#F39C12',
                  'Oligopolistic': '#27AE60', 'Undefined': '#95A5A6'}
        bar_colors = [colors.get(x, '#95A5A6') for x in counts.index]
        axes[idx].bar(counts.index, counts.values, color=bar_colors, edgecolor='black')
        axes[idx].set_title(f'Market Structure ({title})', fontsize=12)
        axes[idx].set_ylabel('Number of Sectors', fontsize=11)
        for i, v in enumerate(counts.values):
            axes[idx].text(i, v + 0.5, str(v), ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_hhi_comparison(structure_df, hhi_mktcap_df, output_path):
    """Scatter: HHI from overperformance index vs HHI from market cap."""
    merged = structure_df.merge(hhi_mktcap_df[['Sector', 'HHI_MktCap']], on='Sector')

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(merged['HHI_MktCap'], merged['HHI_Overperformance'],
               c='#2E75B6', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Label outliers
    for _, row in merged.iterrows():
        if row['HHI_MktCap'] > 0.2 or row['HHI_Overperformance'] > 0.25:
            ax.annotate(row['Sector'], (row['HHI_MktCap'], row['HHI_Overperformance']),
                        fontsize=6, alpha=0.8, xytext=(4, 4), textcoords='offset points')

    ax.plot([0, 0.8], [0, 0.8], 'r--', alpha=0.5, label='45-degree line')
    ax.set_xlabel('HHI from Market Cap Shares', fontsize=12)
    ax.set_ylabel('HHI from Overperformance Index', fontsize=12)
    ax.set_title('Market Concentration: Traditional HHI vs Overperformance HHI', fontsize=13)

    # Correlation
    r, p = stats.pearsonr(merged['HHI_MktCap'], merged['HHI_Overperformance'])
    ax.text(0.05, 0.95, f'Pearson r = {r:.3f} (p = {p:.4f})',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_rolling_hhi(rolling_results, sectors_to_plot, output_path):
    """Plot rolling HHI for selected sectors."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(sectors_to_plot)))

    for sector, color in zip(sectors_to_plot, colors):
        if sector in rolling_results:
            df = rolling_results[sector]
            ax.plot(df['Date'], df['Rolling_HHI'], '-o', label=sector,
                    color=color, markersize=3, linewidth=1.5)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rolling HHI (8-Quarter Window)', fontsize=12)
    ax.set_title('Time-Varying Market Concentration by Sector', fontsize=14)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.25, color='red', linestyle=':', alpha=0.5, label='DOJ threshold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_granger_heatmap(p_matrix, output_path, top_n=15):
    """Heatmap of Granger causality p-values for top sectors."""
    # Select sectors with most significant relationships
    sig_count = (p_matrix < 0.05).sum(axis=1) + (p_matrix < 0.05).sum(axis=0)
    top_sectors = sig_count.nlargest(top_n).index.tolist()

    if len(top_sectors) < 3:
        print("Not enough significant Granger relationships to plot.")
        return

    sub = p_matrix.loc[top_sectors, top_sectors]

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.eye(len(sub), dtype=bool)
    sns.heatmap(sub, mask=mask, cmap='RdYlGn_r', center=0.05,
                vmin=0, vmax=0.1, annot=True, fmt='.2f',
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'p-value'}, ax=ax)
    ax.set_title('Granger Causality p-values (Row caused by Column)', fontsize=13)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_confusion_matrix(cm, output_path):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[f'Pred {i}' for i in range(len(cm))],
                yticklabels=[f'True {i}' for i in range(len(cm))])
    ax.set_title('Leave-One-Out CV Confusion Matrix', fontsize=13)
    ax.set_xlabel('Predicted Cluster')
    ax.set_ylabel('True Cluster')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# 9. IN-SAMPLE / OUT-OF-SAMPLE VALIDATION
# ============================================================

def split_data_temporal(sector_data, cutoff='2023-12-31'):
    """
    Split sector data into in-sample (<=cutoff) and out-of-sample (>cutoff).
    """
    cutoff_dt = pd.Timestamp(cutoff)
    is_data = {}
    oos_data = {}

    for sector, data in sector_data.items():
        mkt = data['mkt_cap'].copy()
        mkt['Date'] = pd.to_datetime(mkt['Date'])

        mkt_is = mkt[mkt['Date'] <= cutoff_dt]
        mkt_oos = mkt[mkt['Date'] > cutoff_dt]

        rev = data['revenue'].copy()
        rev['Date'] = pd.to_datetime(rev['Date'])
        rev_is = rev[rev['Date'] <= cutoff_dt]
        rev_oos = rev[rev['Date'] > cutoff_dt]

        if not mkt_is.empty:
            is_data[sector] = {
                'mkt_cap': mkt_is, 'revenue': rev_is,
                'n_companies': mkt_is['Ticker'].nunique(),
                'date_range': (mkt_is['Date'].min(), mkt_is['Date'].max())
            }
        if not mkt_oos.empty:
            oos_data[sector] = {
                'mkt_cap': mkt_oos, 'revenue': rev_oos,
                'n_companies': mkt_oos['Ticker'].nunique(),
                'date_range': (mkt_oos['Date'].min(), mkt_oos['Date'].max())
            }

    return is_data, oos_data


def run_is_oos_validation(sector_data, output_dir):
    """
    Temporal in-sample / out-of-sample validation.

    In-sample: 2019-2023 (train clustering + compute rankings)
    Out-of-sample: 2024-2025 (test stability)

    Tests:
    1. Do cluster assignments remain stable?
    2. Do overperformance rankings persist?
    3. Do market structures hold?
    """
    fig_dir = os.path.join(output_dir, 'figures')
    tbl_dir = os.path.join(output_dir, 'tables')

    print("\n--- IS/OOS Split at Dec 2023 ---")
    try:
        from config import IS_OOS_CUTOFF, DATA_START_DATE, DATA_END_DATE
        cutoff = IS_OOS_CUTOFF
    except ImportError:
        cutoff = '2023-12-31'
        DATA_START_DATE, DATA_END_DATE = '2019-09-30', '2025-12-31'
    is_data, oos_data = split_data_temporal(sector_data, cutoff=cutoff)
    print(f"In-sample sectors: {len(is_data)}, Out-of-sample sectors: {len(oos_data)}")

    # IS metrics
    is_metrics = compute_all_metrics(is_data)
    oos_metrics = compute_all_metrics(oos_data)

    # IS clustering
    is_cluster = perform_clustering(is_metrics)
    is_metrics['Cluster_IS'] = is_cluster['labels']

    # OOS clustering (same k, fresh fit to see if structure holds)
    oos_cluster = perform_clustering(oos_metrics, max_k=is_cluster['optimal_k'] + 2)

    # Compare IS vs OOS metrics for same sectors
    common = set(is_metrics['Sector']) & set(oos_metrics['Sector'])
    is_sub = is_metrics[is_metrics['Sector'].isin(common)].set_index('Sector')
    oos_sub = oos_metrics[oos_metrics['Sector'].isin(common)].set_index('Sector')

    # Correlation of Weighted_Simple_Variance IS vs OOS
    merged = is_sub[['Weighted_Simple_Variance']].join(
        oos_sub[['Weighted_Simple_Variance']], lsuffix='_IS', rsuffix='_OOS'
    ).dropna()

    if len(merged) > 5:
        r, p = stats.pearsonr(merged['Weighted_Simple_Variance_IS'],
                               merged['Weighted_Simple_Variance_OOS'])
        print(f"\nOverperformance index stability (IS vs OOS):")
        print(f"  Pearson r = {r:.3f} (p = {p:.4f}), N = {len(merged)}")

        # Plot IS vs OOS
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(merged['Weighted_Simple_Variance_IS'],
                   merged['Weighted_Simple_Variance_OOS'],
                   c='#2E75B6', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        max_val = max(merged.values.max(), 1)
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        ax.set_xlabel(f'In-Sample W-S Variance ({DATA_START_DATE[:4]}-{cutoff[:4]})', fontsize=12)
        ax.set_ylabel(f'Out-of-Sample W-S Variance ({int(cutoff[:4])+1}-{DATA_END_DATE[:4]})', fontsize=12)
        ax.set_title(f'Overperformance Index Stability (r={r:.3f}, p={p:.4f})', fontsize=13)
        ax.grid(True, alpha=0.3)
        for idx, row in merged.iterrows():
            if row['Weighted_Simple_Variance_IS'] > merged['Weighted_Simple_Variance_IS'].quantile(0.9):
                ax.annotate(idx, (row['Weighted_Simple_Variance_IS'],
                                  row['Weighted_Simple_Variance_OOS']),
                            fontsize=6, alpha=0.8, xytext=(4, 4), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'is_oos_overperformance.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {os.path.join(fig_dir, 'is_oos_overperformance.png')}")

    # IS vs OOS overperformance rankings
    _, is_structure = compute_overperformance_index(is_data)
    _, oos_structure = compute_overperformance_index(oos_data)

    # Compare market structure classifications
    is_struct = is_structure.set_index('Sector')['Market_Structure_HHI']
    oos_struct = oos_structure.set_index('Sector')['Market_Structure_HHI']
    common_struct = is_struct.index.intersection(oos_struct.index)

    if len(common_struct) > 5:
        agree = (is_struct[common_struct] == oos_struct[common_struct]).mean()
        print(f"\nMarket structure stability (IS vs OOS):")
        print(f"  Agreement rate: {agree:.1%} ({(is_struct[common_struct] == oos_struct[common_struct]).sum()}/{len(common_struct)} sectors)")

    # Save IS/OOS comparison
    comparison = merged.copy()
    comparison.to_csv(os.path.join(tbl_dir, 'is_oos_comparison.csv'))

    return {
        'is_metrics': is_metrics,
        'oos_metrics': oos_metrics,
        'ws_var_correlation': (r, p) if len(merged) > 5 else (None, None),
        'structure_agreement': agree if len(common_struct) > 5 else None,
    }


# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def run_full_analysis(data_dir, output_dir):
    """Run the complete analysis pipeline."""
    mkt_cap_dir = os.path.join(data_dir, 'data_mkt_cap')
    revenue_dir = os.path.join(data_dir, 'data_revenue')
    fig_dir = os.path.join(output_dir, 'figures')
    tbl_dir = os.path.join(output_dir, 'tables')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    sector_data = load_sector_data(mkt_cap_dir, revenue_dir, min_companies=3)

    print("\n" + "=" * 60)
    print("STEP 2: Computing Financial Metrics")
    print("=" * 60)
    metrics_df = compute_all_metrics(sector_data)
    metrics_df.to_csv(os.path.join(tbl_dir, 'sector_metrics.csv'), index=False)
    print(f"Computed metrics for {len(metrics_df)} sectors")
    print(metrics_df[['Sector', 'Weighted_Simple_Variance']].nlargest(10, 'Weighted_Simple_Variance'))

    print("\n" + "=" * 60)
    print("STEP 3: Overperformance Index with Statistical Testing")
    print("=" * 60)
    overperf_df, structure_df = compute_overperformance_index(sector_data)
    overperf_df.to_csv(os.path.join(tbl_dir, 'overperformance_details.csv'), index=False)
    structure_df.to_csv(os.path.join(tbl_dir, 'market_structure.csv'), index=False)

    print("\nMarket Structure Distribution (Binomial Test):")
    print(structure_df['Market_Structure_Binomial'].value_counts())
    print("\nMarket Structure Distribution (HHI-Based):")
    print(structure_df['Market_Structure_HHI'].value_counts())

    print("\n" + "=" * 60)
    print("STEP 4: HHI from Market Capitalization")
    print("=" * 60)
    hhi_df = compute_hhi_from_mktcap(sector_data)
    hhi_df.to_csv(os.path.join(tbl_dir, 'hhi_market_cap.csv'), index=False)
    print(f"HHI range: [{hhi_df['HHI_MktCap'].min():.3f}, {hhi_df['HHI_MktCap'].max():.3f}]")

    # HHI correlation with overperformance index
    merged = structure_df.merge(hhi_df[['Sector', 'HHI_MktCap']], on='Sector')
    r, p = stats.pearsonr(merged['HHI_MktCap'], merged['HHI_Overperformance'])
    print(f"Correlation between HHI(MktCap) and HHI(Overperformance): r={r:.3f}, p={p:.4f}")

    print("\n" + "=" * 60)
    print("STEP 5: Clustering with Validation")
    print("=" * 60)
    cluster_results = perform_clustering(metrics_df)
    metrics_df['Cluster'] = cluster_results['labels']
    metrics_df.to_csv(os.path.join(tbl_dir, 'sector_metrics_clustered.csv'), index=False)

    print("\n" + "=" * 60)
    print("STEP 6: Time-Varying Analysis")
    print("=" * 60)
    rolling = compute_rolling_overperformance(sector_data, window_quarters=8)
    print(f"Computed rolling HHI for {len(rolling)} sectors")

    print("\n" + "=" * 60)
    print("STEP 7: Granger Causality Analysis")
    print("=" * 60)
    sector_returns = compute_sector_return_series(sector_data)
    granger_p, sig_pairs = granger_causality_matrix(sector_returns, max_lag=2)
    if len(sig_pairs) > 0:
        print(f"Found {len(sig_pairs)} significant Granger-causal pairs (alpha=0.05)")
        print("Top 10 pairs:")
        for cause, effect, p in sig_pairs[:10]:
            print(f"  {cause} -> {effect} (p={p:.4f})")
        granger_p.to_csv(os.path.join(tbl_dir, 'granger_pvalues.csv'))
        sig_df = pd.DataFrame(sig_pairs, columns=['Cause', 'Effect', 'P_Value'])
        sig_df.to_csv(os.path.join(tbl_dir, 'granger_significant_pairs.csv'), index=False)

    print("\n" + "=" * 60)
    print("STEP 8: Generating Figures")
    print("=" * 60)

    # Fig 1: Dendrogram
    plot_dendrogram(
        cluster_results['linkage'],
        metrics_df['Sector'].values,
        os.path.join(fig_dir, 'dendrogram.png')
    )

    # Fig 2: Silhouette analysis
    plot_silhouette_analysis(
        cluster_results['silhouette_scores'],
        cluster_results['optimal_k'],
        os.path.join(fig_dir, 'silhouette.png')
    )

    # Fig 3: PCA cluster projection
    plot_cluster_pca(
        cluster_results['X_scaled'],
        cluster_results['labels'],
        metrics_df['Sector'].values,
        os.path.join(fig_dir, 'pca_clusters.png')
    )

    # Fig 4: Market structure distribution
    plot_market_structure_distribution(
        structure_df,
        os.path.join(fig_dir, 'market_structure.png')
    )

    # Fig 5: HHI comparison
    plot_hhi_comparison(
        structure_df, hhi_df,
        os.path.join(fig_dir, 'hhi_comparison.png')
    )

    # Fig 6: Rolling HHI for interesting sectors
    top_variance = metrics_df.nlargest(6, 'Weighted_Simple_Variance')['Sector'].tolist()
    plot_rolling_hhi(
        rolling, top_variance,
        os.path.join(fig_dir, 'rolling_hhi.png')
    )

    # Fig 7: Confusion matrix
    plot_confusion_matrix(
        cluster_results['confusion_matrix'],
        os.path.join(fig_dir, 'confusion_matrix.png')
    )

    # Fig 8: Granger causality heatmap
    if not granger_p.empty:
        plot_granger_heatmap(
            granger_p,
            os.path.join(fig_dir, 'granger_heatmap.png')
        )

    print("\n" + "=" * 60)
    print("STEP 9: In-Sample / Out-of-Sample Validation")
    print("=" * 60)
    isoos = run_is_oos_validation(sector_data, output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return {
        'metrics': metrics_df,
        'structure': structure_df,
        'hhi': hhi_df,
        'cluster_results': cluster_results,
        'rolling': rolling,
        'granger_p': granger_p,
        'granger_sig': sig_pairs,
        'overperf': overperf_df,
        'isoos': isoos,
    }


if __name__ == '__main__':
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results = run_full_analysis(BASE, os.path.join(BASE, 'output'))
