"""
Configuration dataclasses for well log correlation section plots.
Jobs instantiate SectionPlotConfig with project-specific overrides;
all display defaults match the proven Bakrol 3D reference implementation.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class FormationTop:
    """Display style for a single formation top line."""
    color: str
    linestyle: str = '-'
    linewidth: float = 1.5
    label: str = ''


@dataclass
class SectionPlotConfig:
    """
    Complete configuration for well log correlation section plots.

    All defaults match the reference implementation (Bakrol 3D, K-VIII-A-Lower).
    Jobs override only what differs — typically formation_tops, datum_surface,
    interval surfaces, and window sizes.
    """

    # --- Formation tops (REQUIRED per job) ---
    formation_tops: Dict[str, FormationTop] = field(default_factory=dict)
    datum_surface: str = ''

    # --- Interval fill (top → base highlighted band) ---
    interval_top_surface: str = ''
    interval_base_surface: str = ''

    # --- Track widths (relative proportions) ---
    track_widths: Dict[str, float] = field(default_factory=lambda: {
        'depth': 1.4,
        'GR': 1.0,
        'LLD': 1.0,
        'NPHI_RHOB': 1.2,
        'DT': 1.0,
    })
    well_gap_ratio: float = 0.3

    # --- Log display scales ---
    gr_range: Tuple[float, float] = (0, 150)
    lld_range: Tuple[float, float] = (0.2, 2000)
    nphi_range: Tuple[float, float] = (0.70, 0.05)
    rhob_range: Tuple[float, float] = (1.95, 2.95)
    dt_range: Tuple[float, float] = (140, 40)

    # --- GR cutoff ---
    gr_sand_cutoff: float = 75

    # --- Colors ---
    gr_line_color: str = '#2E7D32'
    gr_sand_fill: str = '#FFF9C4'
    gr_shale_fill: str = '#BDBDBD'
    lld_line_color: str = '#D32F2F'
    nphi_line_color: str = '#1565C0'
    rhob_line_color: str = '#D32F2F'
    dt_line_color: str = '#6A1B9A'
    interval_fill_color: str = '#FFE0B2'
    crossover_gas_color: str = '#FFEB3B'

    # --- Display window (meters above/below datum) ---
    window_above: float = 35
    window_below: float = 55

    # --- Output ---
    figure_dpi: int = 200
