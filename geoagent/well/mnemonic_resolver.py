"""
Log curve mnemonic resolver — maps varied LAS/Petrel curve names to canonical names.
Handles GR/SGR/CGR → GR, RHOB/RHOZ/ZDEN → RHOB, etc.
"""
import numpy as np
from typing import Dict, List, Optional

# Standard mnemonic aliases for common well log curves.
# Keys are canonical names; values are lists of aliases in priority order.
DEFAULT_ALIASES: Dict[str, List[str]] = {
    'DEPTH': ['DEPTH', 'DEPT', 'MD'],
    'GR':    ['GR', 'SGR', 'CGR'],
    'LLD':   ['LLD', 'ILD', 'RLLD', 'RT', 'LLD_HRLT'],
    'RHOB':  ['RHOB', 'RHOZ', 'ZDEN', 'DEN'],
    'NPHI':  ['NPHI', 'TNPH', 'BPHI'],
    'DT':    ['DT', 'DTC', 'DTCO', 'AC', 'MDT', 'MDT_STC'],
    'LLS':   ['LLS', 'RLLS', 'MSFL', 'SFL'],
    'SP':    ['SP'],
    'CALI':  ['CALI', 'CAL', 'HCAL'],
    'PE':    ['PE', 'PEF'],
    'DTS':   ['DTS', 'DTSM'],
}


def resolve_curve_name(logs, canonical_name, aliases=None):
    """
    Find the actual curve name present in logs for a canonical name.

    Args:
        logs: dict-like mapping curve names to arrays
        canonical_name: standard name (e.g. 'GR', 'LLD')
        aliases: optional custom alias dict; defaults to DEFAULT_ALIASES

    Returns:
        The actual key found in logs, or None
    """
    if aliases is None:
        aliases = DEFAULT_ALIASES
    candidates = aliases.get(canonical_name, [canonical_name])
    for name in candidates:
        if name in logs:
            return name
    return None


def extract_curve(logs, canonical_name, mask=None, aliases=None):
    """
    Extract a log curve array by canonical name, applying optional depth mask.

    Args:
        logs: dict-like mapping curve names to arrays
        canonical_name: standard name (e.g. 'GR', 'RHOB')
        mask: boolean array to select depth window (optional)
        aliases: optional custom alias dict

    Returns:
        numpy array of curve values (masked if mask provided), or None
    """
    resolved = resolve_curve_name(logs, canonical_name, aliases)
    if resolved is None:
        return None
    arr = logs[resolved]
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if mask is not None:
        arr = arr[mask]
    return arr
