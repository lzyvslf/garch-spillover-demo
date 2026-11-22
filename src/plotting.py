"""绘图：溢出热力图、条件波动率、滚动总溢出指数。无显示环境也能出图（Agg 后端）。"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
META = {"Software": "supubao garch-spillover-demo"}


def heatmap(tbl: pd.DataFrame, out: Path) -> None:
    mat = tbl.drop(columns=["FROM其他(%)", "TO其他(%)", "NET(%)"]).values
    labels = list(tbl.index)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                    color="black" if mat[i, j] < 60 else "white", fontsize=9)
    ax.set_title("Variance Decomposition (%)  row=affected, col=source")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, metadata=META)
    plt.close(fig)


def vol_lines(vol: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for c in vol.columns:
        ax.plot(vol.index, vol[c], label=c, linewidth=0.9)
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_title("GARCH(1,1) Conditional Volatility")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, metadata=META)
    plt.close(fig)


def toi_line(toi: pd.Series, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(toi.index, toi.values, linewidth=1.2, color="#1f77b4")
    ax.axhline(toi.mean(), color="red", linestyle="--", linewidth=0.8,
               label=f"mean={toi.mean():.1f}%")
    ax.set_ylabel("Total Spillover Index (%)")
    ax.set_title("Rolling Total Spillover Index (200-day window)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, metadata=META)
    plt.close(fig)
