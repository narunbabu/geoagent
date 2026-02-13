"""
Core matplotlib plotting engine for well log correlation sections.
Produces professional cross-section panels with:
  - 6 sub-tracks per well: MD | TVDSS | GR | LLD | NPHI+RHOB | DT
  - Formation top connections between wells with labels
  - GR variable-color fill (copper_r colormap)
  - Log-scale resistivity, NPHI-RHOB crossover shading
  - Stratigraphic datum flattening

All display parameters come from a SectionPlotConfig instance — no module-level
config imports. Jobs pass their own config with project-specific overrides.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from geoagent.plotting.config import SectionPlotConfig


def _plot_md_track(ax, md, datum_md, cfg):
    """Plot the MD track as a blank axis with tick marks.
    Returns md_ticks array for post-draw fig.text() labeling."""
    y_top = datum_md - cfg.window_above
    y_bot = datum_md + cfg.window_below
    ax.set_xlim(0, 1)
    ax.set_ylim(y_bot, y_top)
    ax.set_xticks([])
    ax.set_yticks([])

    md_min = datum_md - cfg.window_above
    md_max = datum_md + cfg.window_below
    md_ticks = np.arange(np.ceil(md_min / 10) * 10, md_max + 1, 10)
    minor_ticks = np.arange(np.ceil(md_min / 2) * 2, md_max + 1, 2)

    for tick_md in md_ticks:
        ax.axhline(tick_md, color='gray', linewidth=0.3, alpha=0.4)
        ax.plot([0, 0.15], [tick_md, tick_md], color='black', linewidth=0.7,
                clip_on=True, zorder=3)

    for mt in minor_ticks:
        if mt not in md_ticks:
            ax.plot([0, 0.08], [mt, mt], color='black', linewidth=0.35,
                    clip_on=True, zorder=3)

    ax.set_title('MD', fontsize=6, fontweight='bold', pad=2)
    ax.set_facecolor('white')
    return md_ticks


def _plot_tvdss_track(ax, md, tvdss, datum_md, cfg):
    """Plot the TVDSS track as a blank axis with tick marks.
    Returns (md_ticks, tvdss_at_ticks) or None if no TVDSS data."""
    y_top = datum_md - cfg.window_above
    y_bot = datum_md + cfg.window_below
    ax.set_xlim(0, 1)
    ax.set_ylim(y_bot, y_top)
    ax.set_xticks([])
    ax.set_yticks([])

    md_min = datum_md - cfg.window_above
    md_max = datum_md + cfg.window_below
    md_ticks = np.arange(np.ceil(md_min / 10) * 10, md_max + 1, 10)
    minor_ticks = np.arange(np.ceil(md_min / 2) * 2, md_max + 1, 2)

    for tick_md in md_ticks:
        ax.axhline(tick_md, color='gray', linewidth=0.3, alpha=0.4)
        ax.plot([0.85, 1.0], [tick_md, tick_md], color='#1565C0', linewidth=0.7,
                clip_on=True, zorder=3)

    for mt in minor_ticks:
        if mt not in md_ticks:
            ax.plot([0.92, 1.0], [mt, mt], color='#1565C0', linewidth=0.35,
                    clip_on=True, zorder=3)

    ax.set_title('TVDSS', fontsize=6, fontweight='bold', color='#1565C0', pad=2)
    ax.set_facecolor('white')

    has_tvdss = tvdss is not None and len(tvdss) > 1
    if has_tvdss:
        tvdss_at_ticks = np.interp(md_ticks, md, tvdss)
        return md_ticks, tvdss_at_ticks
    return None


def _plot_gr_track(ax, md, gr, datum_md, cfg):
    """Plot GR track with variable-color fill (copper_r colormap)."""
    ax.set_xlim(*cfg.gr_range)
    ax.set_ylim(datum_md + cfg.window_below, datum_md - cfg.window_above)

    if gr is None:
        ax.text(0.5, 0.5, 'No GR', transform=ax.transAxes,
                ha='center', va='center', fontsize=6, color='gray')
        ax.set_title('GR', fontsize=6, fontweight='bold', pad=2)
        return

    gr_clipped = np.clip(gr, cfg.gr_range[0], cfg.gr_range[1])

    # Variable color fill: imshow background + white mask right of curve
    gr_img = np.tile(gr_clipped.reshape(-1, 1), (1, 2))
    extent = [cfg.gr_range[0], cfg.gr_range[1], md[-1], md[0]]
    ax.imshow(gr_img, aspect='auto', cmap='copper_r', origin='upper',
              extent=extent, vmin=cfg.gr_range[0], vmax=cfg.gr_range[1],
              interpolation='bilinear', zorder=1)

    # White mask: fill from curve to right edge
    ax.fill_betweenx(md, gr_clipped, cfg.gr_range[1],
                      color='white', linewidth=0, zorder=2)

    # GR line
    ax.plot(gr_clipped, md, color=cfg.gr_line_color, linewidth=0.8, zorder=5)

    # Cutoff line
    ax.axvline(cfg.gr_sand_cutoff, color='gray', linewidth=0.3, linestyle='--', alpha=0.5)

    # Scale ticks
    ax.set_xticks([0, 75, 150])
    ax.set_xticklabels(['0', '75', '150'], fontsize=5)
    ax.tick_params(axis='x', labelsize=4, length=2, pad=1)
    ax.set_title('GR\n0    API   150', fontsize=5, fontweight='bold', pad=2)
    ax.set_yticklabels([])


def _plot_lld_track(ax, md, lld, datum_md, cfg):
    """Plot LLD (deep resistivity) track on logarithmic scale."""
    ax.set_xlim(*cfg.lld_range)
    ax.set_xscale('log')
    ax.set_ylim(datum_md + cfg.window_below, datum_md - cfg.window_above)

    if lld is None:
        ax.text(0.5, 0.5, 'No LLD', transform=ax.transAxes,
                ha='center', va='center', fontsize=6, color='gray')
        ax.set_title('LLD', fontsize=6, fontweight='bold', pad=2)
        return

    lld_clipped = np.clip(lld, cfg.lld_range[0], cfg.lld_range[1])
    lld_clipped = np.where(lld_clipped > 0, lld_clipped, cfg.lld_range[0])

    ax.plot(lld_clipped, md, color=cfg.lld_line_color, linewidth=0.8)

    ax.set_xticks([0.2, 2, 20, 200, 2000])
    ax.set_xticklabels(['0.2', '2', '20', '200', '2K'], fontsize=5)
    ax.tick_params(axis='x', labelsize=4, length=2, pad=1)
    ax.set_title('LLD\n0.2  ohm-m  2K', fontsize=5, fontweight='bold', pad=2)
    ax.set_yticklabels([])
    ax.grid(True, which='major', axis='x', alpha=0.2, linewidth=0.3)


def _plot_nphi_rhob_track(ax, md, nphi, rhob, datum_md, cfg):
    """Plot NPHI and RHOB overlay with crossover shading."""
    ax.set_xlim(0, 1)
    ax.set_ylim(datum_md + cfg.window_below, datum_md - cfg.window_above)

    has_nphi = nphi is not None and np.any(np.isfinite(nphi))
    has_rhob = rhob is not None and np.any(np.isfinite(rhob))

    if not has_nphi and not has_rhob:
        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                ha='center', va='center', fontsize=6, color='gray')
        ax.set_title('NPHI+RHOB', fontsize=5, fontweight='bold', pad=2)
        return

    nphi_min, nphi_max = cfg.nphi_range
    rhob_min, rhob_max = cfg.rhob_range

    nphi_norm = None
    rhob_norm = None

    if has_nphi:
        nphi_clipped = np.clip(nphi, min(nphi_min, nphi_max), max(nphi_min, nphi_max))
        nphi_norm = (nphi_clipped - nphi_min) / (nphi_max - nphi_min)
        ax.plot(nphi_norm, md, color=cfg.nphi_line_color, linewidth=0.6,
                linestyle='--', label='NPHI')

    if has_rhob:
        rhob_clipped = np.clip(rhob, min(rhob_min, rhob_max), max(rhob_min, rhob_max))
        rhob_norm = (rhob_clipped - rhob_min) / (rhob_max - rhob_min)
        ax.plot(rhob_norm, md, color=cfg.rhob_line_color, linewidth=0.6, label='RHOB')

    # Crossover shading (gas effect)
    if has_nphi and has_rhob and nphi_norm is not None and rhob_norm is not None:
        ax.fill_betweenx(md, nphi_norm, rhob_norm,
                          where=(rhob_norm > nphi_norm),
                          color=cfg.crossover_gas_color, alpha=0.3)

    nphi_left = f'{nphi_min:.2f}'
    nphi_right = f'{nphi_max:.2f}'
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels([f'{nphi_left}\n1.95', '', f'{nphi_right}\n2.95'], fontsize=5)
    ax.tick_params(axis='x', labelsize=4, length=2, pad=1)
    ax.set_title(f'NPHI(--) RHOB(-)\n{nphi_left}  v/v  {nphi_right}\n1.95   g/cc   2.95',
                 fontsize=5, fontweight='bold', pad=2)
    ax.set_yticklabels([])


def _plot_dt_track(ax, md, dt, datum_md, cfg):
    """Plot DT (sonic) track."""
    ax.set_xlim(*cfg.dt_range)
    ax.set_ylim(datum_md + cfg.window_below, datum_md - cfg.window_above)

    if dt is None or not np.any(np.isfinite(dt)):
        ax.text(0.5, 0.5, 'No DT', transform=ax.transAxes,
                ha='center', va='center', fontsize=6, color='gray')
        ax.set_title('DT', fontsize=5, fontweight='bold', pad=2)
        ax.set_yticklabels([])
        return

    dt_clipped = np.clip(dt, min(cfg.dt_range), max(cfg.dt_range))
    ax.plot(dt_clipped, md, color=cfg.dt_line_color, linewidth=0.8)

    ax.set_xticks([140, 90, 40])
    ax.set_xticklabels(['140', '90', '40'], fontsize=5)
    ax.tick_params(axis='x', labelsize=4, length=2, pad=1)
    ax.set_title('DT\n140  us/ft  40', fontsize=5, fontweight='bold', pad=2)
    ax.set_yticklabels([])


def _draw_formation_tops_on_well(axes, well_data, datum_md, cfg):
    """Draw formation top lines across all tracks for one well."""
    for surface_name, top_style in cfg.formation_tops.items():
        if surface_name in well_data['tops']:
            top_md = well_data['tops'][surface_name]
            for ax in axes:
                ylim = ax.get_ylim()
                if min(ylim) <= top_md <= max(ylim):
                    ax.axhline(top_md, color=top_style.color,
                              linestyle=top_style.linestyle,
                              linewidth=top_style.linewidth * 1.5,
                              alpha=0.95, zorder=5)


def _draw_interval_fill(axes, well_data, cfg):
    """Fill the target interval with light color."""
    top_md = well_data['tops'].get(cfg.interval_top_surface)
    base_md = well_data['tops'].get(cfg.interval_base_surface)
    if top_md is not None and base_md is not None:
        for ax in axes:
            ax.axhspan(top_md, base_md, color=cfg.interval_fill_color, alpha=0.35, zorder=0)


def _draw_top_labels_on_well(axes, well_data, well_idx, n_wells, cfg):
    """Draw formation top name labels on the first and last well of a section."""
    if well_idx != 0 and well_idx != n_wells - 1:
        return

    ax = axes[0]
    side = 'left' if well_idx == 0 else 'right'

    for surface_name, top_style in cfg.formation_tops.items():
        if surface_name in well_data['tops']:
            top_md = well_data['tops'][surface_name]
            ylim = ax.get_ylim()
            if min(ylim) <= top_md <= max(ylim):
                label = top_style.label
                if side == 'left':
                    x_pos = ax.get_xlim()[0]
                    ax.annotate(
                        label, xy=(x_pos, top_md),
                        xytext=(-3, 0), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        color=top_style.color, ha='right', va='center',
                        annotation_clip=False
                    )
                else:
                    ax_last = axes[-1]
                    x_pos = ax_last.get_xlim()[1]
                    ax_last.annotate(
                        label, xy=(x_pos, top_md),
                        xytext=(3, 0), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        color=top_style.color, ha='left', va='center',
                        annotation_clip=False
                    )


def _connect_tops_between_wells(fig, well_axes_list, section_data, well_order, cfg):
    """Draw formation top connection lines between adjacent wells using figure transforms."""
    for i in range(len(well_order) - 1):
        w1 = well_order[i]
        w2 = well_order[i + 1]
        if w1 not in section_data or w2 not in section_data:
            continue

        ax_right = well_axes_list[i][-1]
        ax_left = well_axes_list[i + 1][0]

        for surface_name, top_style in cfg.formation_tops.items():
            md1 = section_data[w1]['tops'].get(surface_name)
            md2 = section_data[w2]['tops'].get(surface_name)
            if md1 is None or md2 is None:
                continue

            disp1 = ax_right.transData.transform((ax_right.get_xlim()[1], md1))
            fig1 = fig.transFigure.inverted().transform(disp1)

            disp2 = ax_left.transData.transform((ax_left.get_xlim()[0], md2))
            fig2 = fig.transFigure.inverted().transform(disp2)

            line = plt.Line2D([fig1[0], fig2[0]], [fig1[1], fig2[1]],
                             color=top_style.color,
                             linestyle=top_style.linestyle,
                             linewidth=top_style.linewidth * 1.2,
                             alpha=0.8,
                             transform=fig.transFigure,
                             zorder=10)
            fig.add_artist(line)


def plot_correlation_section(section_data, well_order, title, output_path,
                             config, distances=None):
    """
    Plot a complete well log correlation section.

    Args:
        section_data: dict of {well_name: well_data_dict} from prepare_section_data
        well_order: list of well names in display order (left to right)
        title: section title string
        output_path: path to save PNG
        config: SectionPlotConfig instance
        distances: list of distances between consecutive wells (optional)
    """
    cfg = config
    active_wells = [w for w in well_order if w in section_data]
    n_wells = len(active_wells)

    if n_wells == 0:
        print(f"  WARNING: No wells with data for {title}")
        return

    print(f"  Plotting {title}: {n_wells} wells")

    tracks_per_well = 6  # MD, TVDSS, GR, LLD, NPHI+RHOB, DT
    depth_w = cfg.track_widths['depth']
    track_total_width = (depth_w + cfg.track_widths['GR'] + cfg.track_widths['LLD'] +
                         cfg.track_widths['NPHI_RHOB'] + cfg.track_widths['DT'])
    well_width_inches = 3.5
    gap_inches = well_width_inches * cfg.well_gap_ratio
    fig_width = n_wells * well_width_inches + (n_wells - 1) * gap_inches + 1.5
    fig_height = 14

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Top-level GridSpec: one column per well, with gaps
    width_ratios = []
    for i in range(n_wells):
        width_ratios.append(track_total_width)
        if i < n_wells - 1:
            width_ratios.append(cfg.well_gap_ratio * track_total_width)

    n_cols = 2 * n_wells - 1
    outer_gs = gridspec.GridSpec(1, n_cols, figure=fig,
                                 width_ratios=width_ratios,
                                 left=0.06, right=0.96,
                                 top=0.90, bottom=0.06,
                                 wspace=0.0)

    well_axes_list = []
    md_label_jobs = []
    tvdss_label_jobs = []

    for well_idx, well_name in enumerate(active_wells):
        wd = section_data[well_name]
        datum_md = wd['datum_md']
        gs_col = well_idx * 2

        md_w = depth_w * 0.5
        tvdss_w = depth_w * 0.5
        track_widths = [md_w, tvdss_w, cfg.track_widths['GR'],
                        cfg.track_widths['LLD'], cfg.track_widths['NPHI_RHOB'],
                        cfg.track_widths['DT']]
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, tracks_per_well,
            subplot_spec=outer_gs[0, gs_col],
            width_ratios=track_widths,
            wspace=0.05
        )

        axes = []

        # Track 0: MD
        ax_md = fig.add_subplot(inner_gs[0, 0])
        md_ticks = _plot_md_track(ax_md, wd['md'], datum_md, cfg)
        md_label_jobs.append((ax_md, md_ticks))
        axes.append(ax_md)

        # Track 1: TVDSS
        ax_tvdss = fig.add_subplot(inner_gs[0, 1])
        tvdss_result = _plot_tvdss_track(ax_tvdss, wd['md'], wd['tvdss'], datum_md, cfg)
        if tvdss_result is not None:
            tvdss_label_jobs.append((ax_tvdss, tvdss_result[0], tvdss_result[1]))
        axes.append(ax_tvdss)

        # Track 2: GR
        ax_gr = fig.add_subplot(inner_gs[0, 2])
        _plot_gr_track(ax_gr, wd['md'], wd['gr'], datum_md, cfg)
        axes.append(ax_gr)

        # Track 3: LLD
        ax_lld = fig.add_subplot(inner_gs[0, 3])
        _plot_lld_track(ax_lld, wd['md'], wd['lld'], datum_md, cfg)
        axes.append(ax_lld)

        # Track 4: NPHI + RHOB
        ax_nphi_rhob = fig.add_subplot(inner_gs[0, 4])
        _plot_nphi_rhob_track(ax_nphi_rhob, wd['md'], wd['nphi'], wd['rhob'], datum_md, cfg)
        axes.append(ax_nphi_rhob)

        # Track 5: DT
        ax_dt = fig.add_subplot(inner_gs[0, 5])
        _plot_dt_track(ax_dt, wd['md'], wd['dt'], datum_md, cfg)
        axes.append(ax_dt)

        # Formation tops on all tracks
        _draw_formation_tops_on_well(axes, wd, datum_md, cfg)

        # Target interval fill
        _draw_interval_fill(axes, wd, cfg)

        # Formation top labels on first and last well
        _draw_top_labels_on_well(axes, wd, well_idx, n_wells, cfg)

        # Well name at top
        fig.text(
            (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2,
            0.92,
            well_name,
            ha='center', va='bottom',
            fontsize=8, fontweight='bold',
            transform=fig.transFigure
        )

        # Light grid lines + inward ticks on log tracks
        for i, ax in enumerate(axes):
            ax.grid(True, axis='y', alpha=0.15, linewidth=0.3)
            if i >= 2:
                ax.tick_params(axis='both', which='both', direction='in')

        well_axes_list.append(axes)

    # Render first to get accurate coordinate transforms
    fig.canvas.draw()

    # Add MD labels as figure-level text
    for ax_md, md_ticks in md_label_jobs:
        for tick_md in md_ticks:
            disp = ax_md.transData.transform((0.5, tick_md))
            fig_xy = fig.transFigure.inverted().transform(disp)
            fig.text(fig_xy[0], fig_xy[1], f'{int(tick_md)}',
                     ha='center', va='center', fontsize=5.5,
                     color='black', fontweight='bold',
                     transform=fig.transFigure)

    # Add TVDSS labels as figure-level text
    for ax_tv, md_ticks, tvdss_vals in tvdss_label_jobs:
        for tick_md, tv_val in zip(md_ticks, tvdss_vals):
            disp = ax_tv.transData.transform((0.5, tick_md))
            fig_xy = fig.transFigure.inverted().transform(disp)
            fig.text(fig_xy[0], fig_xy[1], f'{int(tv_val)}',
                     ha='center', va='center', fontsize=5.5,
                     color='#1565C0', fontweight='bold',
                     transform=fig.transFigure)

    # Connect formation tops between wells
    _connect_tops_between_wells(fig, well_axes_list, section_data, active_wells, cfg)

    # Distance annotations between wells at bottom
    if distances is not None:
        for i in range(min(len(distances), n_wells - 1)):
            if distances[i] is not None and i + 1 < len(well_axes_list):
                ax_r = well_axes_list[i][-1]
                ax_l = well_axes_list[i + 1][0]
                mid_x = (ax_r.get_position().x1 + ax_l.get_position().x0) / 2
                fig.text(mid_x, 0.02, f'{distances[i]:.0f}m',
                        ha='center', va='bottom', fontsize=6, color='gray',
                        transform=fig.transFigure)

    # Section title
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.96)

    # Datum line label
    fig.text(0.01, 0.5, f'Datum: {cfg.datum_surface}',
             rotation=90, va='center', ha='left', fontsize=7,
             color='red', alpha=0.7, transform=fig.transFigure)

    # Legend for formation tops
    legend_y = 0.94
    for surface_name, top_style in cfg.formation_tops.items():
        fig.text(0.99, legend_y, top_style.label,
                ha='right', va='top', fontsize=6,
                color=top_style.color, fontweight='bold',
                transform=fig.transFigure)
        legend_y -= 0.013

    fig.savefig(output_path, dpi=cfg.figure_dpi, facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")
