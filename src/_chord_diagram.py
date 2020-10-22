"""
https://github.com/fengwangPhysics/matplotlib-chord-diagram/blob/master/matplotlib-chord.py

With modifications
"""

from typing import List, Union, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd


Axis = Union[matplotlib.axes.Axes]
Array = Union[np.ndarray]
DataFrame = Union[pd.DataFrame]

LW = 0.3


def chord_diagram(
    X: Union[Array, DataFrame],
    ax: Axis = None,
    palette: str = "tab20",
    width: float = 0.1,
    pad: float = 2,
    chordwidth: float = 0.7,
    labels: List[str] = None,
) -> Array:
    """Plot a chord diagram
    Parameters
    ----------
    X :
        flux data, X[i, j] is the flux from i to j
    ax :
        matplotlib `axes` to show the plot
    colors : optional
        user defined colors in rgb format. Use function hex2rgb() to convert hex color to rgb color. Default: d3.js category10
    width : optional
        width/thickness of the ideogram arc
    pad : optional
        gap pad between two neighboring ideogram arcs, unit: degree, default: 2 degree
    chordwidth : optional
        position of the control points for the chords, controlling the shape of the chords
    """
    if isinstance(X, pd.DataFrame) and labels is None:
        labels = X.index
        X = X.values
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    # X[i, j]:  i -> j
    x = X.sum(axis=1)  # sum over rows

    if (x == 0).any():
        X = X + 1e-30
        x = X.sum(axis=1)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    colors = sns.color_palette(palette, len(x))

    # find position for each start and end
    y = x / np.sum(x).astype(float) * (360 - pad * len(x))

    pos = {}
    arc = []
    nodePos = []
    start = 0
    for i in range(len(x)):
        end = start + y[i]
        arc.append((start, end))
        angle = 0.5 * (start + end)
        # print(start, end, angle)
        if -30 <= angle <= 210:
            angle -= 90
        else:
            angle -= 270
        nodePos.append(
            tuple(polar2xy(1.1, 0.5 * (start + end) * np.pi / 180.0)) + (angle,)
        )
        z = (X[i, :] / x[i].astype(float)) * (end - start)
        ids = np.argsort(z)
        z0 = start
        for j in ids:
            pos[(i, j)] = (z0, z0 + z[j])
            z0 += z[j]
        start = end + pad

    for i in range(len(x)):
        start, end = arc[i]
        ideogram_arc(
            start=start,
            end=end,
            radius=1.0,
            ax=ax,
            color=colors[i],
            width=width,
        )
        start, end = pos[(i, i)]
        self_chord_arc(
            start,
            end,
            radius=1.0 - width,
            color=colors[i],
            chordwidth=chordwidth * 0.7,
            ax=ax,
        )
        for j in range(i):
            color = colors[i]
            if X[i, j] > X[j, i]:
                color = colors[j]
            start1, end1 = pos[(i, j)]
            start2, end2 = pos[(j, i)]
            chord_arc(
                start1,
                end1,
                start2,
                end2,
                radius=1.0 - width,
                color=colors[i],
                chordwidth=chordwidth,
                ax=ax,
            )

    ax.axis("off")
    if labels is not None:
        prop = dict(fontsize=16 * 0.8, ha="center", va="center")
        for i, label in enumerate(labels):
            ax.text(
                nodePos[i][0],
                nodePos[i][1],
                label,
                rotation=nodePos[i][2],
                **prop
            )

    if "fig" in locals():
        return fig


def polar2xy(r: float, theta: float) -> Array:
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def hex2rgb(c: str) -> Tuple[float, float, float]:
    return tuple(int(c[i : i + 2], 16) / 256.0 for i in (1, 3, 5))  # type: ignore[return-value]


def ideogram_arc(
    start=0, end=60, radius=1.0, width=0.2, ax=None, color=(1, 0, 0)
):
    # start, end should be in [0, 360)
    if start > end:
        start, end = end, start
    start *= np.pi / 180.0
    end *= np.pi / 180.0
    # optimal distance to the control points
    # https://stackoverflow.com/questions/1734745/how-to-create-circle-with-b%C3%A9zier-curves
    opt = 4.0 / 3.0 * np.tan((end - start) / 4.0) * radius
    inner = radius * (1 - width)
    verts = [
        polar2xy(radius, start),
        polar2xy(radius, start) + polar2xy(opt, start + 0.5 * np.pi),
        polar2xy(radius, end) + polar2xy(opt, end - 0.5 * np.pi),
        polar2xy(radius, end),
        polar2xy(inner, end),
        polar2xy(inner, end) + polar2xy(opt * (1 - width), end - 0.5 * np.pi),
        polar2xy(inner, start)
        + polar2xy(opt * (1 - width), start + 0.5 * np.pi),
        polar2xy(inner, start),
        polar2xy(radius, start),
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]

    if ax == None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, facecolor=color + (0.5,), edgecolor=color + (0.4,), lw=LW
        )
        ax.add_patch(patch)


def chord_arc(
    start1=0,
    end1=60,
    start2=180,
    end2=240,
    radius=1.0,
    chordwidth=0.7,
    ax=None,
    color=(1, 0, 0),
):
    # start, end should be in [0, 360)
    if start1 > end1:
        start1, end1 = end1, start1
    if start2 > end2:
        start2, end2 = end2, start2
    start1 *= np.pi / 180.0
    end1 *= np.pi / 180.0
    start2 *= np.pi / 180.0
    end2 *= np.pi / 180.0
    opt1 = 4.0 / 3.0 * np.tan((end1 - start1) / 4.0) * radius
    opt2 = 4.0 / 3.0 * np.tan((end2 - start2) / 4.0) * radius
    rchord = radius * (1 - chordwidth)
    verts = [
        polar2xy(radius, start1),
        polar2xy(radius, start1) + polar2xy(opt1, start1 + 0.5 * np.pi),
        polar2xy(radius, end1) + polar2xy(opt1, end1 - 0.5 * np.pi),
        polar2xy(radius, end1),
        polar2xy(rchord, end1),
        polar2xy(rchord, start2),
        polar2xy(radius, start2),
        polar2xy(radius, start2) + polar2xy(opt2, start2 + 0.5 * np.pi),
        polar2xy(radius, end2) + polar2xy(opt2, end2 - 0.5 * np.pi),
        polar2xy(radius, end2),
        polar2xy(rchord, end2),
        polar2xy(rchord, start1),
        polar2xy(radius, start1),
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
    ]

    if ax == None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, facecolor=color + (0.5,), edgecolor=color + (0.4,), lw=LW
        )
        ax.add_patch(patch)


def self_chord_arc(
    start=0, end=60, radius=1.0, chordwidth=0.7, ax=None, color=(1, 0, 0)
):
    # start, end should be in [0, 360)
    if start > end:
        start, end = end, start
    start *= np.pi / 180.0
    end *= np.pi / 180.0
    opt = 4.0 / 3.0 * np.tan((end - start) / 4.0) * radius
    rchord = radius * (1 - chordwidth)
    verts = [
        polar2xy(radius, start),
        polar2xy(radius, start) + polar2xy(opt, start + 0.5 * np.pi),
        polar2xy(radius, end) + polar2xy(opt, end - 0.5 * np.pi),
        polar2xy(radius, end),
        polar2xy(rchord, end),
        polar2xy(rchord, start),
        polar2xy(radius, start),
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
    ]

    if ax == None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, facecolor=color + (0.5,), edgecolor=color + (0.4,), lw=LW
        )
        ax.add_patch(patch)
