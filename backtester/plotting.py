"""Graphiques standards : performance cumulée et drawdowns."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas d'écran dans un Codespace : on enregistre des images
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .metrics import drawdown

# Palette catégorielle à ordre fixe (même couleur = même stratégie sur tous les graphiques)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#3a3a38"
MUTED = "#8a8a85"
GRID = "#e6e5e0"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_performance(results, path: str | Path, title: str = "Performance cumulée") -> Path:
    """Deux panneaux partageant l'axe des dates : valeur du portefeuille (échelle log) et drawdown."""
    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )

    for i, res in enumerate(results):
        color = COLORS[i % len(COLORS)]
        equity = res.equity
        ax_eq.plot(equity.index, equity.values, color=color, linewidth=2, label=res.name)
        dd = drawdown(res.returns)
        ax_dd.plot(dd.index, dd.values, color=color, linewidth=1.5)
        # Étiquette directe en fin de courbe (lisible sans légende)
        ax_eq.annotate(
            f" {res.name}", xy=(equity.index[-1], equity.iloc[-1]),
            color=INK, fontsize=9, va="center",
        )

    ax_eq.set_yscale("log")
    ax_eq.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 7.0)))
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax_eq.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_eq.set_ylabel("Valeur (base 1, échelle log)", color=MUTED, fontsize=9)
    ax_eq.set_title(title, color=INK, fontsize=13, loc="left", pad=12)
    ax_eq.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")

    ax_dd.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax_dd.set_ylabel("Drawdown", color=MUTED, fontsize=9)
    ax_dd.axhline(0, color=MUTED, linewidth=0.8)

    for ax in (ax_eq, ax_dd):
        _style(ax)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
