"""
Map annotation utilities: north arrow, scale bar, polygon overlay.

Pure matplotlib — no external dependencies beyond mpl_toolkits.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


def add_north_arrow(ax, x=0.95, y=0.95, size=0.07):
    """
    Add a north arrow to an axes at relative position (x, y).

    Args:
        ax: matplotlib Axes
        x, y: position in axes fraction (0-1)
        size: arrow size in axes fraction
    """
    ax.annotate('N',
                xy=(x, y), xycoords='axes fraction',
                xytext=(x, y - size), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=2.0, color='black'),
                fontsize=12, fontweight='bold', ha='center', va='bottom',
                color='black')


def add_scale_bar(ax, length_m=None, location='lower right'):
    """
    Add a scale bar to a map axes.

    Auto-computes a round length if not provided.

    Args:
        ax: matplotlib Axes (must have data coordinates in meters)
        length_m: bar length in meters (auto if None)
        location: 'lower right', 'lower left', 'upper right', 'upper left'
    """
    if length_m is None:
        xlim = ax.get_xlim()
        span = xlim[1] - xlim[0]
        length_m = _round_scale_length(span * 0.2)

    loc_map = {
        'lower right': 4,
        'lower left': 3,
        'upper right': 1,
        'upper left': 2,
    }
    loc_code = loc_map.get(location, 4)

    fontprops = fm.FontProperties(size=9)
    scalebar = AnchoredSizeBar(
        ax.transData,
        length_m,
        f'{length_m:.0f} m',
        loc=loc_code,
        pad=0.5,
        color='black',
        frameon=True,
        size_vertical=length_m * 0.02,
        fontproperties=fontprops,
        sep=5,
    )
    ax.add_artist(scalebar)


def plot_polygon(ax, polygon_xy, color='black', linestyle='--',
                 linewidth=1.5, label='Block Boundary', alpha=0.8):
    """
    Plot a closed polygon on the axes.

    Args:
        ax: matplotlib Axes
        polygon_xy: list of (x, y) tuples
        color, linestyle, linewidth, label, alpha: styling
    """
    if not polygon_xy:
        return
    xs = [p[0] for p in polygon_xy] + [polygon_xy[0][0]]
    ys = [p[1] for p in polygon_xy] + [polygon_xy[0][1]]
    ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=linewidth,
            label=label, alpha=alpha, zorder=8)


def _round_scale_length(raw_m):
    """Round a raw length to a 'nice' number for scale bar display."""
    nice = [100, 200, 500, 1000, 2000, 5000, 10000]
    for n in nice:
        if raw_m <= n * 1.5:
            return n
    return round(raw_m / 1000) * 1000
