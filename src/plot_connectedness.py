"""
plot_connectedness.py - Figures for the Diebold-Yilmaz connectedness analysis.

Generates:
  output/figures/rolling_tci.png        rolling total connectedness over time
  output/figures/spillover_network.png  directed sector spillover network
  output/figures/net_connectedness.png  net directional connectedness bars

Run: python src/plot_connectedness.py   (after src/connectedness.py)
"""
from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from statsmodels.tsa.api import VAR

from connectedness import load_returns, generalized_fevd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "output", "figures")
CONN = os.path.join(BASE, "output", "connectedness")

ABBR = {
    "Information Technology": "Info Tech", "Communication Services": "Comm Svcs",
    "Consumer Discretionary": "Cons Disc", "Consumer Staples": "Cons Stpl",
    "Health Care": "Health Care", "Real Estate": "Real Est",
}


def short(name: str) -> str:
    return ABBR.get(name, name)


def plot_rolling_tci() -> None:
    s = pd.read_csv(os.path.join(CONN, "rolling_tci.csv"), index_col=0, parse_dates=True).iloc[:, 0]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(s.index, s.values, color="#1f4e79", lw=1.6)
    ax.fill_between(s.index, s.values, s.min(), alpha=0.12, color="#1f4e79")
    peak = s.idxmax()
    ax.scatter([peak], [s.max()], color="#c0392b", zorder=5)
    ax.annotate(f"peak {s.max():.0f}%  ({peak.date()})", (peak, s.max()),
                textcoords="offset points", xytext=(10, -4), color="#c0392b", fontsize=10)
    ax.set_title("Rolling 200-day Total Connectedness Index (11 GICS sectors)")
    ax.set_ylabel("TCI (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "rolling_tci.png"), dpi=200)
    plt.close(fig)


def plot_network_and_net(theta: np.ndarray, names: list[str]) -> None:
    import networkx as nx
    off = theta.copy()
    np.fill_diagonal(off, 0.0)
    to = off.sum(axis=0) * 100
    frm = off.sum(axis=1) * 100
    net = to - frm

    # ---- spillover network ----
    G = nx.DiGraph()
    for n in names:
        G.add_node(n)
    thr = np.percentile(off[off > 0], 80)  # show strongest 20% of links
    for i, ti in enumerate(names):
        for j, tj in enumerate(names):
            if i != j and off[i, j] >= thr:
                G.add_edge(tj, ti, weight=off[i, j])  # j transmits to i

    pos = nx.circular_layout(G)
    fig, ax = plt.subplots(figsize=(9, 9))
    vmax = np.abs(net).max()
    node_colors = [net[names.index(n)] for n in G.nodes()]
    node_sizes = [400 + 60 * to[names.index(n)] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap="coolwarm",
                           vmin=-vmax, vmax=vmax, node_size=node_sizes, ax=ax)
    ew = [G[u][v]["weight"] for u, v in G.edges()]
    ew = np.array(ew)
    nx.draw_networkx_edges(G, pos, width=1.0 + 6 * (ew - ew.min()) / (ew.ptp() + 1e-9),
                           edge_color="#888", alpha=0.5, arrowsize=12,
                           connectionstyle="arc3,rad=0.08", ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: short(n) for n in G.nodes()}, font_size=9, ax=ax)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    fig.colorbar(sm, ax=ax, shrink=0.7, label="net connectedness (red = transmitter, blue = receiver)")
    ax.set_title("Sector Return Spillover Network (strongest 20% of links)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "spillover_network.png"), dpi=200)
    plt.close(fig)

    # ---- net directional bar ----
    order = np.argsort(net)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#c0392b" if net[k] > 0 else "#2c7fb8" for k in order]
    ax.bar([short(names[k]) for k in order], [net[k] for k in order], color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("net connectedness (TO minus FROM, %)")
    ax.set_title("Net Directional Connectedness by Sector")
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "net_connectedness.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    rets = load_returns()
    names = list(rets.columns)
    p = max(1, int(VAR(rets).select_order(maxlags=10).aic))
    theta = generalized_fevd(VAR(rets).fit(p), horizon=10)
    plot_rolling_tci()
    plot_network_and_net(theta, names)
    print("wrote rolling_tci.png, spillover_network.png, net_connectedness.png to output/figures/")


if __name__ == "__main__":
    main()
