"""
figures_intl.py - Manuscript figures for the ten other markets: the granular pattern and the path
of sector co-movement in each market, and the concentration effect by market against sector depth.
Run after intl_analysis.py (both variants) and intl_heterogeneity.py. Writes paper/v3/fig/.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures_v2 as f2  # noqa: E402  (shared style)
from intl_analysis import load_market  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "paper", "v3", "fig")
OUT = os.path.join(BASE, "output", "intl")
FIX = os.path.join(BASE, "output", "intl_fixed")
BLUE, ORANGE, INK2, MUTED, GRID = f2.BLUE, f2.ORANGE, f2.INK2, f2.MUTED, f2.GRID
SHORT = {"United States": "US", "Japan": "JP", "China": "CN", "United Kingdom": "UK", "Canada": "CA",
         "Korea": "KR", "Australia": "AU", "India": "IN", "Germany": "DE", "Hong Kong": "HK", "Brazil": "BR"}


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)


def yearly_rho2(market):
    R, _, _, _, _ = load_market(market)
    out = {}
    for yr, g in R.groupby(R.index.year):
        C = g.dropna().corr().values
        k = C.shape[0]
        out[yr] = 100 * ((C ** 2).sum() - k) / (k * (k - 1))
    return pd.Series(out)


def fig_granular_world():
    comp = pd.read_csv(os.path.join(OUT, "granular_components.csv"))
    gran = pd.read_csv(os.path.join(OUT, "granular_by_market.csv")).set_index("market")
    m = comp.groupby("market")[["R2_top", "R2_rest"]].mean()
    m = m.loc[(m["R2_rest"] - m["R2_top"]).sort_values().index]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    y = np.arange(len(m))
    ax.hlines(y, m["R2_top"], m["R2_rest"], color=MUTED, lw=1.0)
    ax.scatter(m["R2_top"], y, color=ORANGE, s=26, zorder=3, label="Three largest firms")
    ax.scatter(m["R2_rest"], y, color=BLUE, s=26, zorder=3, label="Rest of sector")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s} ({100 * gran.loc[s, 'share_top_below_rest']:.0f}%)" for s in m.index], fontsize=7)
    ax.set_xlabel("$R^2$ on the market's other sectors (%)")
    ax.set_title("(a) Components of sector returns", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=7)
    ax = axes[1]
    series = {mk: yearly_rho2(mk) for mk in SHORT}
    for mk, s in series.items():
        if mk in ("United States", "Korea"):
            continue
        ax.plot(s.index, s.values, color="#c9c8c3", lw=0.9)
    for mk, col in (("United States", BLUE), ("Korea", ORANGE)):
        s = series[mk]
        ax.plot(s.index, s.values, color=col, lw=1.8, marker="o", ms=3, label=mk)
    med = pd.DataFrame(series).median(axis=1)
    ax.plot(med.index, med.values, color=INK2, lw=1.4, ls="--", label="Median of eleven")
    ax.set_ylabel("Mean squared pairwise correlation (%)")
    ax.set_title("(b) Sector co-movement by year", loc="left")
    ax.set_xticks([2018, 2020, 2022, 2024, 2026])
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig6_granular_world")


def fig_effect_world():
    cur = pd.read_csv(os.path.join(OUT, "market_results.csv"))
    fx = pd.read_csv(os.path.join(FIX, "market_results.csv"))
    het = pd.read_csv(os.path.join(OUT, "heterogeneity.csv")).set_index("market")
    c = cur[cur.outcome == "R2"].set_index("market")
    f = fx[fx.outcome == "R2"].set_index("market")
    order = c["coef"].sort_values(ascending=False).index
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), gridspec_kw={"width_ratios": [1, 1.05]})
    ax = axes[0]
    y = np.arange(len(order))
    ax.axvline(0, color=MUTED, lw=0.7)
    for dy, d, col, lab in ((-0.15, c, BLUE, "Current members"), (0.15, f, ORANGE, "Firms listed in 2018")):
        sig = d.loc[order, "p_perm"] < 0.10
        ax.scatter(d.loc[order, "coef"][sig], (y + dy)[sig.values], color=col, s=28, zorder=3, label=lab)
        ax.scatter(d.loc[order, "coef"][~sig], (y + dy)[~sig.values], facecolor="white", edgecolor=col,
                   s=28, zorder=3, lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel("Effect of top-3 share on monthly $R^2$")
    ax.set_title("(a) By market", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, fontsize=7)
    ax = axes[1]
    x, yy = het["firms_per_sector"], het["coef_R2"]
    ax.axhline(0, color=MUTED, lw=0.7)
    sig = het["p_perm_R2"] < 0.10
    ax.scatter(x[sig], yy[sig], color=BLUE, s=40, edgecolor="white", lw=1.0, zorder=3)
    ax.scatter(x[~sig], yy[~sig], facecolor="white", edgecolor=BLUE, s=40, lw=1.1, zorder=3)
    for mk in het.index:
        off = {"CA": (-14, 5), "UK": (5, -9)}.get(SHORT[mk], (4, 3))
        ax.annotate(SHORT[mk], (x[mk], yy[mk]), xytext=off, textcoords="offset points", fontsize=7.5, color=INK2)
    b = np.polyfit(np.log(x), yy, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(b, np.log(xs)), color=ORANGE, lw=1.2)
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([8, 15, 25, 45])
    ax.set_xticklabels(["8", "15", "25", "45"])
    ax.set_xlabel("Firms per sector (median, log scale)")
    ax.set_ylabel("Effect on monthly $R^2$")
    ax.set_title("(b) Effect and sector depth", loc="left")
    fig.tight_layout()
    save(fig, "fig7_effect_world")


if __name__ == "__main__":
    fig_granular_world()
    fig_effect_world()
    print("figures ->", FIG)
