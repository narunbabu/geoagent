"""
Example 01 — Load a project and explore available data.

Usage:
    python examples/01_load_project.py /path/to/your/project
"""
import sys
from geoagent import CoreDataManager

# --- Load project ---
project_dir = sys.argv[1] if len(sys.argv) > 1 else input("Enter project directory: ")
dm = CoreDataManager(project_dir)
print(dm)

# --- Wells ---
wells = dm.get_available_wells()
print(f"\nWells ({len(wells)}):")
for w in wells[:10]:
    print(f"  {w}")
if len(wells) > 10:
    print(f"  ... and {len(wells) - 10} more")

# --- Surveys ---
surveys = dm.get_available_surveys()
print(f"\nSeismic Surveys ({len(surveys)}):")
for s in surveys:
    attrs = dm.seismic_handler.get_available_attributes(s)
    print(f"  {s}: {len(attrs)} attributes — {attrs[:5]}")

# --- Well heads sample ---
well_heads = dm.get_data('well_heads')
if well_heads is not None:
    print(f"\nWell Heads DataFrame: {well_heads.shape}")
    print(well_heads.head())

# --- Horizons ---
horizons = dm.get_data('horizons')
if horizons:
    print(f"\nHorizons ({len(horizons)}):")
    for name, data in horizons.items():
        z_shape = data['Z'].shape if hasattr(data['Z'], 'shape') else 'unknown'
        print(f"  {name}: Z shape {z_shape}")
