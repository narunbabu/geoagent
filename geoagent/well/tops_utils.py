"""
Formation top lookups and KB (Kelly Bushing) elevation retrieval.
Works with SeisTrans pickle-loaded DataFrames (well_tops, well_heads).
"""
import pandas as pd


def get_formation_md(well_tops, well_name, surface_name):
    """
    Get the measured depth (MD) of a formation top for a well.

    Args:
        well_tops: DataFrame with columns 'Well', 'Surface', 'MD'
        well_name: well name string
        surface_name: formation top surface name

    Returns:
        float MD value, or None if not found
    """
    mask = (well_tops['Well'] == well_name) & (well_tops['Surface'] == surface_name)
    rows = well_tops[mask]
    if len(rows) > 0 and pd.notna(rows['MD'].iloc[0]):
        return float(rows['MD'].iloc[0])
    return None


def get_well_kb(well_heads, well_name):
    """
    Get Kelly Bushing elevation for a well.

    Tries multiple column name conventions for both well name and KB value.

    Args:
        well_heads: DataFrame with well header information
        well_name: well name string

    Returns:
        float KB elevation (m above sea level), or None
    """
    for name_col in ['Name', 'Well']:
        if name_col in well_heads.columns:
            row = well_heads[well_heads[name_col] == well_name]
            if len(row) > 0:
                for kb_col in ['KB', 'KB elevation', 'Elev KB', 'Well datum value']:
                    if kb_col in row.columns:
                        val = row[kb_col].iloc[0]
                        if pd.notna(val):
                            return float(val)
    return None


def get_well_coordinates(well_heads, well_name):
    """
    Get surface X, Y coordinates for a well.

    Args:
        well_heads: DataFrame with well header information
        well_name: well name string

    Returns:
        (x, y) tuple of floats, or (None, None) if not found
    """
    for name_col in ['Name', 'Well']:
        if name_col in well_heads.columns:
            row = well_heads[well_heads[name_col] == well_name]
            if len(row) > 0:
                x = float(row['Surface X'].iloc[0]) if 'Surface X' in row.columns else None
                y = float(row['Surface Y'].iloc[0]) if 'Surface Y' in row.columns else None
                return x, y
    return None, None
