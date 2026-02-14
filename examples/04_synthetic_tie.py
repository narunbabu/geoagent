"""
Example 04 — Generate a synthetic seismogram for a well.

This example demonstrates the synthetic tie workflow:
1. Load project
2. Prepare well data (logs + TDR + seismic)
3. Generate synthetic via reflectivity convolution
4. Cross-correlate with real seismic trace

Usage:
    python examples/04_synthetic_tie.py /path/to/project "WELL-1" "SurveyName" "Amplitude"
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from geoagent import CoreDataManager
from geoagent.synthetic.functions import (
    prepare_data_w_tdr,
    create_reflectivity,
    create_synthetic_seismic_valid,
    calculate_correlation_with_shift,
)

project_dir = sys.argv[1] if len(sys.argv) > 1 else input("Project dir: ")
well_name = sys.argv[2] if len(sys.argv) > 2 else input("Well name: ")
survey = sys.argv[3] if len(sys.argv) > 3 else input("Survey name: ")
attribute = sys.argv[4] if len(sys.argv) > 4 else input("Attribute name: ")

dm = CoreDataManager(project_dir)
print(f"Loaded: {dm}")

# Get TDR for this well
well_tdr = dm.well_handler.get_preferred_tdr(well_name)
if well_tdr is None:
    print(f"No TDR available for well '{well_name}'")
    sys.exit(1)

# Prepare data (logs + seismic trace)
result = prepare_data_w_tdr(
    dm, well_tdr, well_name,
    use_upscaled=True,
    current_survey=survey,
    current_attribute=attribute,
)

if result is None:
    print("Failed to prepare data — check that upscaled logs and seismic data exist.")
    sys.exit(1)

# Unpack result
log_df, seismic_trace, time_array, impedance = result[:4]
print(f"Log samples: {len(log_df)}, Seismic samples: {len(seismic_trace)}")
print(f"Time range: {time_array[0]:.1f} - {time_array[-1]:.1f} ms")

# Generate synthetic
reflectivity = create_reflectivity(impedance)
print(f"Reflectivity: {len(reflectivity)} samples")

# Use a simple Ricker wavelet (30 Hz)
from scipy.signal import ricker
wavelet = ricker(51, 4)  # ~30 Hz at 2ms sampling

synthetic = create_synthetic_seismic_valid(reflectivity, wavelet)
print(f"Synthetic: {len(synthetic)} samples")

# Cross-correlation
best_shift, best_cc, cc_curve, shifts = calculate_correlation_with_shift(
    seismic_trace[:len(synthetic)], synthetic, max_shift=50
)
print(f"Best shift: {best_shift} samples, CC: {best_cc:.3f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(14, 8), sharey=True)

# Seismic trace
t = time_array[:len(seismic_trace)]
axes[0].plot(seismic_trace, t, 'b', linewidth=0.8)
axes[0].set_title(f'Seismic ({attribute})')
axes[0].set_xlabel('Amplitude')
axes[0].set_ylabel('TWT (ms)')
axes[0].invert_yaxis()

# Synthetic
t_syn = time_array[:len(synthetic)]
axes[1].plot(synthetic, t_syn, 'r', linewidth=0.8)
axes[1].set_title(f'Synthetic (shift={best_shift})')
axes[1].set_xlabel('Amplitude')

# Overlay
axes[2].plot(seismic_trace[:len(synthetic)], t_syn, 'b', linewidth=0.8, label='Seismic')
axes[2].plot(synthetic, t_syn, 'r', linewidth=0.8, label='Synthetic')
axes[2].set_title(f'Overlay (CC={best_cc:.3f})')
axes[2].set_xlabel('Amplitude')
axes[2].legend()

fig.suptitle(f'{well_name} — Synthetic Tie', fontsize=13, fontweight='bold')
fig.tight_layout()

out_path = f"{well_name}_synthetic_tie.png"
fig.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
