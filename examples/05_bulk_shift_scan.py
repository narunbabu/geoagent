"""
Example 05 — Run a bulk-shift cross-correlation scan.

Scans a well's synthetic tie across a range of time shifts to find
the optimal bulk shift (best CC). Then plots the 8-panel synthetic
tie figure.

Usage:
    python examples/05_bulk_shift_scan.py /path/to/project "WELL-1" "SurveyName" "Amplitude"
"""
import sys
import matplotlib
matplotlib.use('Agg')

from geoagent import CoreDataManager
from geoagent.synthetic.bulk_shift import (
    audit_tdr_bulk_shifts,
    compute_bulk_shift_scan,
    plot_synthetic_tie,
)

project_dir = sys.argv[1] if len(sys.argv) > 1 else input("Project dir: ")
well_name = sys.argv[2] if len(sys.argv) > 2 else input("Well name: ")
survey = sys.argv[3] if len(sys.argv) > 3 else input("Survey name: ")
attribute = sys.argv[4] if len(sys.argv) > 4 else input("Attribute name: ")

dm = CoreDataManager(project_dir)
print(f"Loaded: {dm}")

# Step 1: Audit — see which wells are ready for scanning
print("\n--- TDR/Bulk Shift Audit ---")
report = audit_tdr_bulk_shifts(dm)
for row in report[:5]:
    print(f"  {row['well']}: has_bs={row['has_bs']}, "
          f"bs_amount={row.get('bs_amount_ms', 'N/A')}")
if len(report) > 5:
    print(f"  ... {len(report) - 5} more wells")

# Step 2: Compute bulk shift scan for target well
print(f"\n--- Computing bulk shift scan for {well_name} ---")
scan_result = compute_bulk_shift_scan(
    dm, well_name,
    survey=survey,
    attribute=attribute,
    shift_range_ms=(-30, 30),
    shift_step_ms=2,
)

if scan_result is None:
    print("Scan failed — check well data availability.")
    sys.exit(1)

print(f"Best shift: {scan_result['best_shift_ms']:.1f} ms")
print(f"Best CC: {scan_result['best_cc']:.4f}")
print(f"Zero-shift CC: {scan_result.get('zero_cc', 'N/A')}")

# Step 3: Plot 8-panel synthetic tie
print(f"\n--- Generating synthetic tie plot ---")
out_path = f"{well_name}_bulk_shift.png"
fig = plot_synthetic_tie(
    dm, well_name,
    survey=survey,
    attribute=attribute,
    bulk_shift_ms=scan_result['best_shift_ms'],
    save_path=out_path,
)

if fig:
    print(f"Saved: {out_path}")
else:
    print("Plot generation failed.")
