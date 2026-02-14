"""
Example 03 — Plot a well location map.

Usage:
    python examples/03_location_map.py /path/to/project
"""
import sys
import matplotlib
matplotlib.use('Agg')

from geoagent import CoreDataManager
from geoagent.plotting.map_plotter import plot_location_map

project_dir = sys.argv[1] if len(sys.argv) > 1 else input("Project dir: ")
dm = CoreDataManager(project_dir)

# Get well heads with coordinates
well_heads = dm.get_data('well_heads')
if well_heads is None:
    print("No well heads data found!")
    sys.exit(1)

print(f"Well heads: {len(well_heads)} wells")

# Optional: load block boundary polygon
polygon = None
try:
    from geoagent.io.polygon_utils import load_polygons
    polygons = load_polygons(project_dir)
    if polygons and 'block_boundary' in polygons:
        polygon = polygons['block_boundary']
        print("Block boundary polygon loaded")
except Exception:
    pass

# Plot location map
fig, ax = plot_location_map(
    dm,
    polygon=polygon,
    show_north_arrow=True,
    show_scale_bar=True,
    title='Well Location Map',
)

out_path = "location_map.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
