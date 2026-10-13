"""
figures_v3.py - New figures for the v3 manuscript (normalization, granular decomposition, and the
concentration link on co-movement measured afresh each month). Figures 1 and 2 are unchanged
from v2 and are copied. Run after referee_checks.py, inference_checks.py and
granular_decomposition.py. Writes paper/v3/fig/.
"""
from __future__ import annotations

import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures_v2 as f2  # noqa: E402
from econometrics import SECTORS, SHORT, load, two_way_demean  # noqa: E402
from inference_checks import conc_controls, period_outcomes  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "paper", "v3", "fig")
OUT3 = os.path.join(BASE, "output", "v3")
BLUE, ORANGE, INK2, MUTED = f2.BLUE, f2.ORANGE, f2.INK2, f2.MUTED


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)


def copy_v2():
    os.makedirs(FIG, exist_ok=True)
    src = os.path.join(BASE, "paper", "v2", "fig")
    for a, b in (("fig1_tci", "fig1_tci"), ("fig3_concentration", "fig2_concentration")):
        for ext in (".pdf", ".png"):
            shutil.copy(os.path.join(src, a + ext), os.path.join(FIG, b + ext))


def fig_normalization():
    pv = pd.read_csv(os.path.join(OUT3, "panel_variants_ETF.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax, col, title in ((axes[0], "sigma_NET", "(a) Covariance matrix alone, $H=1$"),
                           (axes[1], "maxrow_NET", "(b) Maximum row-sum normalization")):
        r = np.corrcoef(pv["row_NET"], pv[col])[0, 1]
        ax.scatter(pv["row_NET"], pv[col], s=5, color=BLUE, alpha=0.35, lw=0)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_xlabel("NET, row normalization, VAR(1), $H=10$ (pp)")
        ax.set_ylabel("NET (pp)")
        ax.set_title(title, loc="left")
        ax.text(0.03, 0.95, f"correlation {r:.2f}", transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    fig.tight_layout()
    save(fig, "fig3_normalization")


def fig_granular():
    comp = pd.read_csv(os.path.join(OUT3, "granular_components_by_year.csv"))
    sysy = pd.read_csv(os.path.join(OUT3, "granular_system_by_year.csv"))
    m = comp.groupby("sector")[["W", "R2_top", "R2_rest"]].mean().sort_values("W")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    y = np.arange(len(m))
    ax.hlines(y, m["R2_top"], m["R2_rest"], color=MUTED, lw=1.0)
    ax.scatter(m["R2_top"], y, color=ORANGE, s=26, zorder=3, label="Three largest firms")
    ax.scatter(m["R2_rest"], y, color=BLUE, s=26, zorder=3, label="Rest of sector")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s} ({w:.0f}%)" for s, w in zip(m.index, m["W"])], fontsize=7)
    ax.set_xlabel("$R^2$ on rest of market (%), mean of annual values")
    ax.set_title("(a) Components of each sector return", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=7)
    ax = axes[1]
    ax.plot(sysy["year"], sysy["avg_rho2_actual"], color=BLUE, marker="o", ms=3.5, lw=1.4, label="Actual weights")
    ax.plot(sysy["year"], sysy["avg_rho2_cf"], color=ORANGE, marker="o", ms=3.5, lw=1.4, ls="--",
            label="Top-three weight held at 2018")
    ax.set_ylabel("Mean squared pairwise correlation (%)")
    ax.set_title("(b) Eleven-sector system, by year", loc="left")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xticks(sysy["year"][::2])
    fig.tight_layout()
    save(fig, "fig4_granular")


def fig_link():
    ld = pd.read_csv(os.path.join(OUT3, "long_differences.csv"), index_col=0)
    R = load("const_returns.csv")[SECTORS]
    pan = period_outcomes(R, "M").merge(conc_controls("M"), on=["sector", "period"])
    pan = pan[pan["period"] >= pd.Period("2018-11", "M")].rename(columns={"period": "month"})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    ax = axes[0]
    x, yv = ld["d_top3"], ld["d_R2_Constituent"]
    ax.scatter(x, yv, s=46, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
    for s in ld.index:
        off = {"IND": (-20, 3), "ENE": (4, -10)}.get(SHORT[s], (4, 3))
        ax.annotate(SHORT[s], (x[s], yv[s]), xytext=off, textcoords="offset points", fontsize=7.5, color=INK2)
    b = np.polyfit(x, yv, 1)
    xs = np.linspace(x.min(), x.max(), 10)
    ax.plot(xs, np.polyval(b, xs), color=ORANGE, lw=1.3)
    ax.set_xlabel("Change in top-3 share (pp)")
    ax.set_ylabel("Change in $R^2$ on rest of market (pp)")
    ax.set_title("(a) Long differences, 2018-19 to 2025-26", loc="left")
    ax = axes[1]
    cols = ["R2", "top3_lag_pp", "log_share_lag", "logvol"]
    d = two_way_demean(pan.dropna(subset=cols).copy(), cols)
    Xc = np.column_stack([d["log_share_lag"], d["logvol"]])
    res = lambda v: v - Xc @ np.linalg.lstsq(Xc, v, rcond=None)[0]
    xr, yr = res(d["top3_lag_pp"].values), res(d["R2"].values)
    q = pd.qcut(xr, 20, labels=False)
    bx, by = pd.Series(xr).groupby(q).mean(), pd.Series(yr).groupby(q).mean()
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.scatter(bx, by, s=26, color=BLUE, edgecolor="white", linewidth=1.0, zorder=3)
    bb = np.polyfit(xr, yr, 1)
    xs = np.linspace(bx.min(), bx.max(), 10)
    ax.plot(xs, np.polyval(bb, xs), color=ORANGE, lw=1.3)
    ax.set_xlabel("Top-3 share, lagged (pp, residualized)")
    ax.set_ylabel("Monthly $R^2$ (pp, residualized)")
    ax.set_title("(b) Within sector, two-way FE (20 bins)", loc="left")
    fig.tight_layout()
    save(fig, "fig5_concentration_link")


if __name__ == "__main__":
    copy_v2()
    fig_normalization()
    fig_granular()
    fig_link()
    print("figures ->", FIG)
