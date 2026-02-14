"""
Matplotlib-based seismic section display.

Provides plot_seismic_section() for rendering inline/crossline seismic
data as variable-density (imshow) and/or wiggle-trace displays with
optional horizon overlays and well markers.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_seismic_section(
    traces,
    time_axis,
    *,
    trace_positions=None,
    display='variable_density',
    cmap='seismic',
    clip_percentile=99,
    wiggle_fill=True,
    wiggle_scale=1.0,
    horizons=None,
    well_markers=None,
    title='Seismic Section',
    xlabel='Trace',
    ylabel='TWT (ms)',
    figsize=None,
    ax=None,
    save_path=None,
    dpi=150,
):
    """
    Plot a seismic section.

    Args:
        traces: 2D array (n_samples, n_traces) of seismic amplitudes
        time_axis: 1D array of time values (ms) for the vertical axis
        trace_positions: 1D array of trace positions (inline/crossline numbers).
                        If None, uses 0-based indices.
        display: 'variable_density', 'wiggle', or 'both'
        cmap: Matplotlib colormap for variable density
        clip_percentile: Clip amplitudes at this percentile for color scaling
        wiggle_fill: If True, fill positive amplitudes on wiggle traces
        wiggle_scale: Scale factor for wiggle trace amplitudes
        horizons: dict of {name: (trace_positions, time_values, color)} for overlays
        well_markers: list of dicts {'position': x, 'name': str, 'color': str}
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size tuple. If None, auto-computed from trace count.
        ax: Existing matplotlib Axes. If None, creates new figure.
        save_path: If provided, saves figure to this path
        dpi: Resolution for saved figure

    Returns:
        (fig, ax) tuple
    """
    traces = np.asarray(traces)
    time_axis = np.asarray(time_axis)
    n_samples, n_traces = traces.shape

    if trace_positions is None:
        trace_positions = np.arange(n_traces)

    # Auto figsize
    if figsize is None:
        width = max(8, n_traces * 0.05)
        height = max(6, n_samples * 0.005)
        figsize = (min(width, 24), min(height, 14))

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Amplitude clipping
    vmax = np.percentile(np.abs(traces[np.isfinite(traces)]), clip_percentile)
    vmin = -vmax

    # Variable density display
    if display in ('variable_density', 'both'):
        extent = [
            trace_positions[0], trace_positions[-1],
            time_axis[-1], time_axis[0],
        ]
        ax.imshow(
            traces,
            aspect='auto',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            interpolation='bilinear',
        )

    # Wiggle trace display
    if display in ('wiggle', 'both'):
        trace_spacing = (trace_positions[-1] - trace_positions[0]) / max(n_traces - 1, 1)
        scale = trace_spacing * wiggle_scale * 0.5

        for i in range(n_traces):
            x_base = trace_positions[i]
            amplitudes = traces[:, i] / vmax * scale if vmax > 0 else traces[:, i] * 0
            x_vals = x_base + amplitudes

            ax.plot(x_vals, time_axis, color='black', linewidth=0.5)

            if wiggle_fill:
                ax.fill_betweenx(
                    time_axis, x_base, x_vals,
                    where=(amplitudes > 0),
                    color='black', alpha=0.6, linewidth=0,
                )

    # Horizon overlays
    if horizons:
        for name, (h_positions, h_times, color) in horizons.items():
            ax.plot(h_positions, h_times, color=color, linewidth=1.5, label=name)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

    # Well markers
    if well_markers:
        for wm in well_markers:
            pos = wm['position']
            name = wm.get('name', '')
            color = wm.get('color', 'red')
            ax.axvline(pos, color=color, linewidth=1.2, linestyle='--', alpha=0.7)
            ax.text(
                pos, time_axis[0], f' {name}',
                color=color, fontsize=8, fontweight='bold',
                verticalalignment='bottom',
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.invert_yaxis()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_trace_gather(
    traces,
    time_axis,
    *,
    well_name='',
    attribute_name='',
    figsize=(6, 8),
    ax=None,
    save_path=None,
    dpi=150,
):
    """
    Plot a small trace gather (e.g., ±5 traces around a well).

    Args:
        traces: 2D array (n_samples, n_traces) or 1D array (single trace)
        time_axis: 1D array of TWT values (ms)
        well_name: Well name for title
        attribute_name: Seismic attribute name
        figsize: Figure size
        ax: Existing Axes or None
        save_path: Optional save path
        dpi: Resolution

    Returns:
        (fig, ax)
    """
    traces = np.asarray(traces)
    if traces.ndim == 1:
        traces = traces.reshape(-1, 1)

    n_samples, n_traces = traces.shape

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    vmax = np.percentile(np.abs(traces[np.isfinite(traces)]), 98)

    for i in range(n_traces):
        x_base = i
        amplitudes = traces[:, i] / vmax * 0.4 if vmax > 0 else traces[:, i] * 0
        x_vals = x_base + amplitudes

        ax.plot(x_vals, time_axis, color='black', linewidth=0.5)
        ax.fill_betweenx(
            time_axis, x_base, x_vals,
            where=(amplitudes > 0),
            color='blue', alpha=0.4, linewidth=0,
        )

    # Center trace highlight
    center = n_traces // 2
    ax.axvline(center, color='red', linewidth=0.8, linestyle=':', alpha=0.5)

    ax.set_xlim(-0.5, n_traces - 0.5)
    ax.set_xlabel('Trace Offset')
    ax.set_ylabel('TWT (ms)')
    title_parts = [p for p in [well_name, attribute_name] if p]
    ax.set_title(' — '.join(title_parts) if title_parts else 'Trace Gather')
    ax.invert_yaxis()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax
