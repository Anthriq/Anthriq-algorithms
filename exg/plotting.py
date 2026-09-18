# Copyright 2026 Nexstem India Private Limited (trading as Anthriq)
# Licensed under the Apache License, Version 2.0. See the LICENSE file.

"""
Shared figure styling, so the three analyses produce a consistent set of plots.

Nothing here is required to compute a result -- it only affects how figures
look. Matplotlib is imported lazily inside the functions so that the analyses
still run, and still print their numbers, in an environment where matplotlib is
not installed.

The palette is deliberately small and fixed:

    EMBER  #D9531E   the condition of interest (eyes closed, flicker on)
    TEAL   #1E7D8C   the comparison condition (eyes open, rest)
    MUTE   #6E655A   annotations, control channels, model curves
    INK    #2A2622   text and axes

Two colours for two conditions, one for everything else. Ember and teal stay
distinguishable in greyscale and for the common forms of colour blindness, since
they differ in lightness as well as hue.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["EMBER", "TEAL", "MUTE", "INK", "style_axes", "save_figure"]

EMBER = "#D9531E"
TEAL = "#1E7D8C"
MUTE = "#6E655A"
INK = "#2A2622"


def style_axes(
    ax,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> None:
    """Apply the house style to one set of axes.

    Removes the top and right spines, which carry no information and box the
    data in for no reason, and lightens the remaining two so the data is the
    most prominent thing in the frame.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTE)
        ax.spines[side].set_linewidth(0.8)

    ax.tick_params(colors=MUTE, labelsize=9, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK)

    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=10.5, pad=8)


def save_figure(figure, path: str | Path, *, dpi: int = 150) -> Path:
    """Save a figure and close it.

    Closing matters in a loop: matplotlib keeps every open figure in memory, and
    a script that saves a hundred without closing them will exhaust it.
    """
    import matplotlib.pyplot as plt

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path
