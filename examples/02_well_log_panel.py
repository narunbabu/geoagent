"""
Example 02 — Display a single-well log panel.

Usage:
    python examples/02_well_log_panel.py /path/to/project "WELL-1"
"""
import sys
import matplotlib
matplotlib.use('Agg')

from geoagent import CoreDataManager
from geoagent.plotting.well_panel import plot_well_panel

project_dir = sys.argv[1] if len(sys.argv) > 1 else input("Project dir: ")
well_name = sys.argv[2] if len(sys.argv) > 2 else input("Well name: ")

dm = CoreDataManager(project_dir)

# Get well logs
logs_df = dm.well_log_handler.get_well_logs(well_name)
if logs_df is None:
    print(f"No logs found for well '{well_name}'")
    sys.exit(1)

print(f"Loaded {len(logs_df)} samples, columns: {list(logs_df.columns)}")

# Convert DataFrame to dict for plot_well_panel
depth_col = next((c for c in logs_df.columns if c.upper() in ['DEPT', 'DEPTH', 'MD']), None)
if depth_col is None:
    print("No depth column found!")
    sys.exit(1)

depth = logs_df[depth_col].values
logs = {col: logs_df[col].values for col in logs_df.columns if col != depth_col}

# Get formation tops for this well
well_tops = dm.get_data('well_tops')
formation_tops = {}
if well_tops is not None and well_name in well_tops:
    for top_name, top_data in well_tops[well_name].items():
        md_val = top_data.get('MD')
        if md_val is not None:
            formation_tops[top_name] = {'md': md_val, 'color': 'black'}

# Plot
fig, axes = plot_well_panel(
    depth, logs,
    well_name=well_name,
    formation_tops=formation_tops,
)

out_path = f"{well_name}_log_panel.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
