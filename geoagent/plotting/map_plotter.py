"""
Well location map plotter with deviation paths and section lines.
Produces a 1:1 aspect ratio map showing:
  - All wells (gray background) with deviated well paths
  - Section lines with distinct colors, well markers, labels
  - TD markers at bottom-hole locations
"""
import matplotlib.pyplot as plt

from geoagent.well.deviation_utils import get_well_path_xy


def plot_location_map(sections_config, well_heads, output_path,
                      deviation=None, figure_dpi=200, title=None,
                      polygon=None, show_north_arrow=True,
                      show_scale_bar=True):
    """
    Plot a well location map showing all wells and section lines.

    Args:
        sections_config: dict of {section_name: {wells, color, direction, ...}}
        well_heads: DataFrame with well header data (Name/Well, Surface X, Surface Y)
        output_path: path to save PNG
        deviation: deviation dict (optional, for well path plotting)
        figure_dpi: output resolution (default 200)
        title: map title (optional; auto-generated if None)
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # Column detection
    name_col = 'Name' if 'Name' in well_heads.columns else 'Well'

    all_x = well_heads['Surface X'].values
    all_y = well_heads['Surface Y'].values
    all_names = well_heads[name_col].values

    # Collect all section wells for highlighting
    section_well_set = set()
    for sec_cfg in sections_config.values():
        section_well_set.update(sec_cfg['wells'])

    # Plot non-section wells as gray background
    for x, y, name in zip(all_x, all_y, all_names):
        if name not in section_well_set:
            ax.plot(x, y, 'o', color='lightgray', markersize=5,
                    markeredgecolor='gray', markeredgewidth=0.5, zorder=2)
            ax.annotate(name, (x, y), fontsize=4, xytext=(2, 2),
                       textcoords='offset points', color='gray')

    # Plot well paths for all deviated wells
    if deviation is not None:
        for well_name in all_names:
            path = get_well_path_xy(deviation, well_name)
            if path is not None:
                px, py = path
                ax.plot(px, py, '-', color='gray', linewidth=1.0, alpha=0.4, zorder=1)
                ax.plot(px[-1], py[-1], 's', color='gray', markersize=3,
                        markeredgecolor='darkgray', markeredgewidth=0.5, zorder=2, alpha=0.5)

    # Plot section lines with distinct colors
    for sec_name, sec_cfg in sections_config.items():
        wells = sec_cfg['wells']
        color = sec_cfg['color']
        direction = sec_cfg.get('direction', '')

        sec_x, sec_y, sec_names = [], [], []
        for w in wells:
            mask = well_heads[name_col] == w
            if mask.any():
                sec_x.append(float(well_heads.loc[mask, 'Surface X'].iloc[0]))
                sec_y.append(float(well_heads.loc[mask, 'Surface Y'].iloc[0]))
                sec_names.append(w)

        if len(sec_x) >= 2:
            # Section line
            ax.plot(sec_x, sec_y, color=color, linewidth=2.5, alpha=0.8, zorder=5,
                   label=f'{sec_name} ({direction})')

            # Well markers on section line
            ax.scatter(sec_x, sec_y, c=color, s=60, zorder=6,
                      edgecolors='black', linewidth=0.8)

            # Well name labels
            for x, y, name in zip(sec_x, sec_y, sec_names):
                ax.annotate(name, (x, y), fontsize=5.5, fontweight='bold',
                           xytext=(4, 4), textcoords='offset points', color=color,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                     alpha=0.7, edgecolor=color, linewidth=0.5))

            # Plot deviated well paths for section wells
            if deviation is not None:
                for w, sx, sy in zip(sec_names, sec_x, sec_y):
                    path = get_well_path_xy(deviation, w)
                    if path is not None:
                        px, py = path
                        ax.plot(px, py, '-', color=color, linewidth=1.5, alpha=0.6, zorder=4)
                        ax.plot(px[-1], py[-1], 's', color=color, markersize=5,
                                markeredgecolor='black', markeredgewidth=0.5, zorder=6)

            # Section name label at midpoint
            mid_idx = len(sec_x) // 2
            mx, my = sec_x[mid_idx], sec_y[mid_idx]
            offset = (0, 15) if direction == 'EW' else (-15, 0)
            ax.annotate(sec_name, (mx, my),
                       xytext=offset, textcoords='offset points',
                       fontsize=7, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.85),
                       zorder=7)

    ax.set_xlabel('Easting (m)', fontsize=10)
    ax.set_ylabel('Northing (m)', fontsize=10)
    if title is None:
        title = 'Well Location Map'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_aspect('equal')

    # Optional map annotations (backward-compatible)
    if polygon is not None or show_north_arrow or show_scale_bar:
        from tools.plotting.map_utils import add_north_arrow, add_scale_bar, plot_polygon
        if polygon is not None:
            plot_polygon(ax, polygon)
        if show_north_arrow:
            add_north_arrow(ax)
        if show_scale_bar:
            add_scale_bar(ax)

    ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)

    fig.savefig(output_path, dpi=figure_dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")
