"""
Log windowing — extract well log curves around a datum/formation top.
Ties together deviation_utils, tops_utils, and mnemonic_resolver to produce
the section_data dict consumed by the section plotter.
"""
import numpy as np

from geoagent.well.deviation_utils import compute_tvdss
from geoagent.well.tops_utils import get_formation_md, get_well_kb, get_well_coordinates
from geoagent.well.mnemonic_resolver import extract_curve


def prepare_section_data(data, well_list, *, datum_surface, formation_tops,
                         window_above=35, window_below=55, aliases=None):
    """
    Prepare windowed well log data for one correlation section.

    Args:
        data: dict with keys 'well_heads', 'well_tops', 'well_logs', 'deviation'
            (as returned by tools.io.project_loader.load_pickles)
        well_list: ordered list of well names for this section
        datum_surface: formation top name used as stratigraphic datum
        formation_tops: dict of {surface_name: ...} — keys are the surfaces to extract
        window_above: meters above datum to include
        window_below: meters below datum to include
        aliases: optional mnemonic alias dict (defaults to DEFAULT_ALIASES)

    Returns:
        dict of {well_name: well_data_dict} where well_data_dict contains:
            md, tvdss, gr, lld, rhob, nphi, dt, datum_md, tops, kb, x, y
    """
    well_heads = data['well_heads']
    well_tops = data['well_tops']
    well_logs = data['well_logs']
    deviation = data['deviation']

    section_data = {}
    skipped = []

    for well_name in well_list:
        # Get datum MD
        datum_md = get_formation_md(well_tops, well_name, datum_surface)
        if datum_md is None:
            skipped.append((well_name, 'no datum top'))
            continue

        # Get well logs
        if well_name not in well_logs:
            skipped.append((well_name, 'no logs'))
            continue

        logs = well_logs[well_name]

        # Get depth array
        depth_arr = extract_curve(logs, 'DEPTH', aliases=aliases)
        if depth_arr is None:
            skipped.append((well_name, 'no depth curve'))
            continue

        md_full = depth_arr

        # Window around datum
        md_min = datum_md - window_above
        md_max = datum_md + window_below
        mask = (md_full >= md_min) & (md_full <= md_max)

        if np.sum(mask) < 5:
            skipped.append((well_name, 'too few samples in window'))
            continue

        md = md_full[mask]

        # Extract log curves
        gr = extract_curve(logs, 'GR', mask=mask, aliases=aliases)
        lld = extract_curve(logs, 'LLD', mask=mask, aliases=aliases)
        rhob = extract_curve(logs, 'RHOB', mask=mask, aliases=aliases)
        nphi = extract_curve(logs, 'NPHI', mask=mask, aliases=aliases)
        dt = extract_curve(logs, 'DT', mask=mask, aliases=aliases)

        # Must have at least GR and LLD
        if gr is None or lld is None:
            skipped.append((well_name, 'missing GR or LLD'))
            continue

        # KB and TVDSS
        kb = get_well_kb(well_heads, well_name)
        tvdss = compute_tvdss(deviation, well_name, md, kb)

        # Fallback: assume vertical well
        if tvdss is None and kb is not None:
            tvdss = md - kb

        # Formation tops
        tops = {}
        for surface_name in formation_tops:
            top_md = get_formation_md(well_tops, well_name, surface_name)
            if top_md is not None:
                tops[surface_name] = top_md

        # Well coordinates
        x, y = get_well_coordinates(well_heads, well_name)

        section_data[well_name] = {
            'md': md,
            'tvdss': tvdss,
            'gr': gr,
            'lld': lld,
            'rhob': rhob,
            'nphi': nphi,
            'dt': dt,
            'datum_md': datum_md,
            'tops': tops,
            'kb': kb,
            'x': x,
            'y': y,
        }

    if skipped:
        print(f"  Skipped wells: {skipped}")

    return section_data


def compute_well_distances(section_data, well_order):
    """
    Compute distances between consecutive wells along the section line.

    Args:
        section_data: dict from prepare_section_data
        well_order: ordered list of well names

    Returns:
        list of distances (float or None) between consecutive wells
    """
    distances = []
    for i in range(len(well_order) - 1):
        w1 = well_order[i]
        w2 = well_order[i + 1]
        if w1 in section_data and w2 in section_data:
            x1, y1 = section_data[w1]['x'], section_data[w1]['y']
            x2, y2 = section_data[w2]['x'], section_data[w2]['y']
            if all(v is not None for v in [x1, y1, x2, y2]):
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distances.append(dist)
            else:
                distances.append(None)
        else:
            distances.append(None)
    return distances
