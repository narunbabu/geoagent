"""
Single-well log panel display.

Produces a 5-track log panel for a single well:
  Depth (MD) | GR | Resistivity | NPHI+RHOB | DT

Follows the same track conventions as section_plotter but for
standalone single-well visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_well_panel(
    depth,
    logs,
    *,
    well_name='',
    depth_range=None,
    formation_tops=None,
    gr_range=(0, 150),
    lld_range=(0.2, 2000),
    nphi_range=(0.60, -0.15),
    rhob_range=(1.95, 2.95),
    dt_range=(140, 40),
    gr_cutoff=75,
    figsize=(12, 10),
    save_path=None,
    dpi=150,
):
    """
    Plot a 5-track log panel for a single well.

    Args:
        depth: 1D array of measured depth values
        logs: dict of log curves keyed by mnemonic. Expected keys:
              'GR', 'LLD' (or 'ILD'), 'NPHI', 'RHOB', 'DT' (or 'DTC')
              Missing logs are plotted as blank tracks.
        well_name: Well name for the title
        depth_range: (min_depth, max_depth) tuple. If None, uses full range.
        formation_tops: dict of {name: {'md': value, 'color': str}}
        gr_range: GR display range (API)
        lld_range: Deep resistivity range (ohm-m), log scale
        nphi_range: NPHI range (v/v), typically decreasing left-to-right
        rhob_range: RHOB range (g/cc)
        dt_range: DT range (us/ft), typically decreasing left-to-right
        gr_cutoff: GR sand/shale cutoff (API)
        figsize: Figure size
        save_path: Optional save path
        dpi: Resolution

    Returns:
        (fig, axes) tuple
    """
    depth = np.asarray(depth)

    if depth_range is None:
        depth_range = (depth.min(), depth.max())

    # Mask to depth range
    mask = (depth >= depth_range[0]) & (depth <= depth_range[1])
    d = depth[mask]

    if len(d) == 0:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, f'No data in range\n{depth_range}',
                ha='center', va='center', transform=ax.transAxes)
        return fig, [ax]

    # Track definitions
    track_names = ['Depth', 'GR', 'LLD', 'NPHI/RHOB', 'DT']
    widths = [0.6, 1.0, 1.0, 1.2, 1.0]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, len(track_names), width_ratios=widths, wspace=0.05)
    axes = []

    for i, (name, width) in enumerate(zip(track_names, widths)):
        ax = fig.add_subplot(gs[0, i])
        ax.set_ylim(depth_range[1], depth_range[0])  # Invert Y
        if i > 0:
            ax.set_yticklabels([])
        axes.append(ax)

    # --- Track 0: Depth ---
    ax_depth = axes[0]
    ax_depth.set_xlim(0, 1)
    ax_depth.set_xticks([])
    ax_depth.set_ylabel('MD (m)', fontsize=9)
    ax_depth.tick_params(axis='y', labelsize=8)
    # Gridlines
    for tick in np.arange(np.ceil(depth_range[0] / 10) * 10, depth_range[1] + 1, 10):
        ax_depth.axhline(tick, color='gray', linewidth=0.3, alpha=0.4)

    # --- Track 1: GR ---
    ax_gr = axes[1]
    gr = _get_log(logs, ['GR', 'GAMMA', 'GR_ARC_FILT', 'CGR', 'SGR'], mask)
    ax_gr.set_xlim(*gr_range)
    ax_gr.set_xlabel('GR (API)', fontsize=8)
    ax_gr.xaxis.set_label_position('top')
    ax_gr.tick_params(axis='x', labelsize=7, top=True, bottom=False,
                      labeltop=True, labelbottom=False)
    _add_grid(ax_gr, depth_range)

    if gr is not None:
        ax_gr.plot(gr, d, color='#2E7D32', linewidth=0.8)
        # Edge fill: sand (yellow, left of curve where GR < cutoff)
        ax_gr.fill_betweenx(d, gr_range[0], np.minimum(gr, gr_cutoff),
                            color='#FFF9C4', alpha=0.7)
        # Edge fill: shale (gray, right of curve where GR > cutoff)
        ax_gr.fill_betweenx(d, np.maximum(gr, gr_cutoff), gr_range[1],
                            color='#BDBDBD', alpha=0.5)
        # Cutoff line
        ax_gr.axvline(gr_cutoff, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    # --- Track 2: LLD (log scale) ---
    ax_lld = axes[2]
    lld = _get_log(logs, ['LLD', 'ILD', 'RESD', 'RT', 'DEEP_RES'], mask)
    ax_lld.set_xscale('log')
    ax_lld.set_xlim(*lld_range)
    ax_lld.set_xlabel('LLD (ohm-m)', fontsize=8)
    ax_lld.xaxis.set_label_position('top')
    ax_lld.tick_params(axis='x', labelsize=7, top=True, bottom=False,
                       labeltop=True, labelbottom=False)
    _add_grid(ax_lld, depth_range)

    if lld is not None:
        ax_lld.plot(lld, d, color='red', linewidth=0.8)

    # --- Track 3: NPHI + RHOB ---
    ax_nr = axes[3]
    nphi = _get_log(logs, ['NPHI', 'PHIN', 'NEUTRON', 'TNPH'], mask)
    rhob = _get_log(logs, ['RHOB', 'RHOZ', 'DENSITY', 'ZDEN'], mask)

    ax_nr.set_xlim(0, 1)
    ax_nr.set_xlabel('NPHI / RHOB', fontsize=8)
    ax_nr.xaxis.set_label_position('top')
    ax_nr.tick_params(axis='x', labelsize=7, top=True, bottom=False,
                      labeltop=True, labelbottom=False)
    _add_grid(ax_nr, depth_range)

    if nphi is not None:
        nphi_norm = (nphi - nphi_range[0]) / (nphi_range[1] - nphi_range[0])
        ax_nr.plot(nphi_norm, d, color='blue', linewidth=0.8, linestyle='--',
                   label=f'NPHI ({nphi_range[0]:.2f}-{nphi_range[1]:.2f})')

    if rhob is not None:
        rhob_norm = (rhob - rhob_range[0]) / (rhob_range[1] - rhob_range[0])
        ax_nr.plot(rhob_norm, d, color='red', linewidth=0.8,
                   label=f'RHOB ({rhob_range[0]:.2f}-{rhob_range[1]:.2f})')

    if nphi is not None and rhob is not None:
        ax_nr.fill_betweenx(
            d, rhob_norm, nphi_norm,
            where=(rhob_norm > nphi_norm),
            color='yellow', alpha=0.3,
        )

    if nphi is not None or rhob is not None:
        ax_nr.legend(loc='lower right', fontsize=6, framealpha=0.7)

    # --- Track 4: DT ---
    ax_dt = axes[4]
    dt = _get_log(logs, ['DT', 'DTC', 'DTCO', 'SONIC', 'AC'], mask)
    ax_dt.set_xlim(*dt_range)
    ax_dt.set_xlabel('DT (us/ft)', fontsize=8)
    ax_dt.xaxis.set_label_position('top')
    ax_dt.tick_params(axis='x', labelsize=7, top=True, bottom=False,
                      labeltop=True, labelbottom=False)
    _add_grid(ax_dt, depth_range)

    if dt is not None:
        ax_dt.plot(dt, d, color='purple', linewidth=0.8)

    # --- Formation tops ---
    if formation_tops:
        for name, info in formation_tops.items():
            md_val = info.get('md')
            color = info.get('color', 'black')
            if md_val is not None and depth_range[0] <= md_val <= depth_range[1]:
                for ax in axes:
                    ax.axhline(md_val, color=color, linewidth=1.2, alpha=0.8)
                # Label on last track
                axes[-1].text(
                    1.02, md_val, name,
                    transform=axes[-1].get_yaxis_transform(),
                    fontsize=8, color=color, fontweight='bold',
                    verticalalignment='center',
                )

    # Title
    fig.suptitle(well_name, fontsize=12, fontweight='bold', y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, axes


def _get_log(logs, mnemonics, mask):
    """Find the first available log curve by mnemonic list."""
    for m in mnemonics:
        if m in logs:
            data = np.asarray(logs[m])
            if len(data) == len(mask):
                return data[mask]
            return data
        # Case-insensitive fallback
        for key in logs:
            if key.upper() == m.upper():
                data = np.asarray(logs[key])
                if len(data) == len(mask):
                    return data[mask]
                return data
    return None


def _add_grid(ax, depth_range):
    """Add subtle horizontal gridlines to a log track."""
    for tick in np.arange(np.ceil(depth_range[0] / 10) * 10, depth_range[1] + 1, 10):
        ax.axhline(tick, color='gray', linewidth=0.2, alpha=0.3)
