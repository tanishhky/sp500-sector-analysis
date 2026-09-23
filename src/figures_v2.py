"""
figures_v2.py - Figures for the concentration/connectedness manuscript.
Run after econometrics.py and robust_panel.py. Writes paper/v2/fig/*.pdf and .png.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from econometrics import OUT, PANEL, SECTORS, SHORT, two_way_demean  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "paper", "v2", "fig")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
EVENTS = [("2020-02-20", "2020-04-30", "COVID-19"), ("2022-01-03", "2022-10-12", "2022 rate shock"),
          ("2023-03-08", "2023-03-31", "SVB"), ("2025-04-02", "2025-04-30", "Tariff shock")]

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)


def shade(ax, label=True):
    for a, b, lab in EVENTS:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#efeeea", zorder=0, lw=0)
        if label:
            y = 1.07 if lab == "SVB" else 1.0
            ax.text(pd.Timestamp(a), y, lab, transform=ax.get_xaxis_transform(), fontsize=7,
                    color=MUTED, va="bottom", ha="left")


def fig_tci():
    tr = pd.read_csv(os.path.join(OUT, "tvp_ret_TCI.csv"), index_col=0, parse_dates=True).squeeze()
    tv = pd.read_csv(os.path.join(OUT, "tvp_vol_TCI.csv"), index_col=0, parse_dates=True).squeeze()
    rr = pd.read_csv(os.path.join(OUT, "rolling200_TCI.csv"), index_col=0, parse_dates=True).squeeze()
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.0), sharex=True)
    ax = axes[0]
    shade(ax)
    ax.plot(rr.index, rr.values, color=ORANGE, lw=1.4, label="200-day rolling window")
    ax.plot(tr.index, tr.values, color=BLUE, lw=1.4, label="TVP-VAR (filtered)")
    ax.set_ylabel("Total connectedness (%)")
    ax.set_title("(a) Return connectedness", loc="left", pad=22)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(40, 95)
    ax = axes[1]
    shade(ax, label=False)
    ax.plot(tv.index, tv.values, color=BLUE, lw=1.4)
    ax.set_ylabel("Total connectedness (%)")
    ax.set_title("(b) Volatility connectedness, TVP-VAR (filtered)", loc="left")
    ax.set_ylim(40, 95)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save(fig, "fig1_tci")


def fig_net_heatmap():
    net = pd.read_csv(os.path.join(OUT, "tvp_ret_NET.csv"), index_col=0, parse_dates=True)
    m = net.resample("ME").mean()
    order = m.mean().sort_values(ascending=False).index
    m = m[order]
    lim = np.nanpercentile(np.abs(m.values), 98)
    cmap = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ef8a7e", "#b8302f"])
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    im = ax.imshow(m.T.values, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7.5)
    years = [i for i, d in enumerate(m.index) if d.month == 1]
    ax.set_xticks(years)
    ax.set_xticklabels([m.index[i].year for i in years])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("NET (pp): transmitter > 0 > receiver", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("Net directional return connectedness, monthly mean of daily TVP-VAR estimates", loc="left")
    fig.tight_layout()
    save(fig, "fig2_net_heatmap")


def fig_concentration():
    conc = pd.read_csv(os.path.join(PANEL, "concentration.csv"), parse_dates=["date"])
    idx = pd.read_csv(os.path.join(PANEL, "index_conc.csv"), index_col=0, parse_dates=True)
    wide = conc.pivot(index="date", columns="sector", values="top3").resample("ME").last() * 100
    order = wide.mean().sort_values(ascending=False).index
    fig, axes = plt.subplots(3, 4, figsize=(7.0, 5.2), sharex=True)
    for ax, s in zip(axes.ravel(), list(order) + ["S&P 500 (top-10 share)"]):
        if s in wide:
            y = wide[s]
            ax.set_ylim(0, 100)
        else:
            y = idx["top10_share"].resample("ME").last() * 100
            ax.set_ylim(0, 50)
        ax.plot(y.index, y.values, color=BLUE, lw=1.3)
        ax.set_title(s, fontsize=7.8, loc="left")
        ax.text(y.index[-1], y.values[-1], f" {y.values[-1]:.0f}", fontsize=7, color=INK2, va="center")
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.supylabel("Top-3 firm share of sector market capitalization (%)", fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, "fig3_concentration")


def fig_link():
    btw = pd.read_csv(os.path.join(OUT, "between_sector.csv"), index_col=0)
    pan = pd.read_csv(os.path.join(OUT, "panel_monthly_ret.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    ax = axes[0]
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.scatter(btw["top3_%"], btw["NET"], s=46, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
    for s, r in btw.iterrows():
        off = {"HC": (-14, -11), "RE": (4, 4)}.get(SHORT[s], (4, 3))
        ax.annotate(SHORT[s], (r["top3_%"], r["NET"]), xytext=off, textcoords="offset points",
                    fontsize=7.5, color=INK2)
    b = np.polyfit(btw["top3_%"], btw["NET"], 1)
    xs = np.linspace(btw["top3_%"].min(), btw["top3_%"].max(), 10)
    ax.plot(xs, np.polyval(b, xs), color=ORANGE, lw=1.3)
    ax.set_xlabel("Mean top-3 share (%)")
    ax.set_ylabel("Mean NET connectedness (pp)")
    ax.set_title("(a) Between sectors", loc="left")
    ax = axes[1]
    d = pan.dropna(subset=["NET", "top3_lag_pp", "log_share_lag", "logvol"]).copy()
    d = two_way_demean(d, ["NET", "top3_lag_pp", "log_share_lag", "logvol"])
    # Frisch-Waugh: residualize both on the controls, then bin
    Xc = np.column_stack([d["log_share_lag"], d["logvol"]])
    res = lambda v: v - Xc @ np.linalg.lstsq(Xc, v, rcond=None)[0]
    x, y = res(d["top3_lag_pp"].values), res(d["NET"].values)
    q = pd.qcut(x, 20, labels=False)
    bx = pd.Series(x).groupby(q).mean()
    by = pd.Series(y).groupby(q).mean()
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.scatter(bx, by, s=26, color=BLUE, edgecolor="white", linewidth=1.0, zorder=3)
    bb = np.polyfit(x, y, 1)
    xs = np.linspace(bx.min(), bx.max(), 10)
    ax.plot(xs, np.polyval(bb, xs), color=ORANGE, lw=1.3)
    ax.set_xlabel("Top-3 share, lagged (pp, residualized)")
    ax.set_ylabel("NET (pp, residualized)")
    ax.set_title("(b) Within sector, two-way FE (20 bins)", loc="left")
    fig.tight_layout()
    save(fig, "fig4_concentration_link")


if __name__ == "__main__":
    fig_tci()
    fig_net_heatmap()
    fig_concentration()
    fig_link()
    print("figures ->", FIG)
