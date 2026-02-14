# Quickstart Guide

This guide walks through the main GeoAgent workflows using a real or synthetic project.

## 1. Loading a Project

GeoAgent loads projects from directories containing pickle files (`.pkl`), which store well heads, well logs, deviation surveys, checkshots, formation tops, and seismic metadata.

```python
from geoagent import CoreDataManager

dm = CoreDataManager("/path/to/project")
print(dm)  # Shows well count, survey count, project path

# List available wells
wells = dm.get_available_wells()
print(f"Wells: {wells}")

# Access data directly
well_heads = dm.get_data('well_heads')  # pandas DataFrame
well_tops = dm.get_data('well_tops')    # dict of dicts
```

Alternatively, use the low-level loader:

```python
from geoagent.io.project_loader import load_pickles
data = load_pickles("/path/to/project")
# data is a dict: {'well_heads': ..., 'well_logs': ..., ...}
```

## 2. Single-Well Log Panel

Display a 5-track well log panel (Depth, GR, LLD, NPHI+RHOB, DT):

```python
from geoagent.plotting.well_panel import plot_well_panel
import numpy as np

depth = np.linspace(1000, 1200, 300)
logs = {
    'GR': np.random.uniform(20, 120, 300),
    'LLD': 10 ** np.random.uniform(-0.5, 2.5, 300),
    'NPHI': np.random.uniform(0.05, 0.45, 300),
    'RHOB': np.random.uniform(2.0, 2.8, 300),
    'DT': np.random.uniform(50, 130, 300),
}

fig, axes = plot_well_panel(depth, logs, well_name='EXAMPLE-1',
                            save_path='well_panel.png')
```

Add formation top markers:

```python
tops = {
    'K-VIII': {'md': 1050, 'color': 'red'},
    'K-VIII-Base': {'md': 1150, 'color': 'blue'},
}
fig, axes = plot_well_panel(depth, logs, formation_tops=tops,
                            save_path='panel_with_tops.png')
```

## 3. Seismic Display

Plot a seismic section with variable-density or wiggle display:

```python
from geoagent.plotting.seismic_plotter import plot_seismic_section
import numpy as np

traces = np.random.randn(200, 50)  # 200 time samples, 50 traces
time_axis = np.linspace(1400, 1600, 200)

# Variable density
fig, ax = plot_seismic_section(traces, time_axis, display='variable_density',
                                save_path='seismic_vd.png')

# Wiggle traces
fig, ax = plot_seismic_section(traces, time_axis, display='wiggle',
                                save_path='seismic_wiggle.png')
```

Overlay horizons and well markers:

```python
positions = np.arange(50)
horizons = {
    'Top-A': (positions, np.full(50, 1480.0), 'red'),
    'Top-B': (positions, np.full(50, 1520.0), 'blue'),
}
markers = [{'position': 25, 'name': 'WELL-1', 'color': 'green'}]

fig, ax = plot_seismic_section(traces, time_axis,
                                trace_positions=positions,
                                horizons=horizons,
                                well_markers=markers)
```

## 4. Horizon Interpolation

Extract horizon Z values at well locations:

```python
from geoagent.utils.interpolation import (
    interpolate_horizon_at_wells,
    extract_horizon_along_line,
)
import numpy as np

# Horizon dict with X, Y (1D), Z (2D) arrays
horizon = dm.get_data('horizons')['my_horizon']

# At specific wells
wells = {'W-1': (270100, 2541100), 'W-2': (270300, 2541300)}
z_at_wells = interpolate_horizon_at_wells(horizon, wells)
print(z_at_wells)  # {'W-1': 1482.3, 'W-2': 1495.7}

# Along a line
x_line = np.linspace(270000, 271000, 100)
y_line = np.full(100, 2541500.0)
z_line = extract_horizon_along_line(horizon, x_line, y_line)
```

## 5. Well Log Correlation Section

Generate a multi-well correlation panel:

```python
from geoagent.io.project_loader import load_pickles
from geoagent.well.log_windower import prepare_section_data
from geoagent.plotting.section_plotter import plot_correlation_section
from geoagent.plotting.config import SectionPlotConfig, FormationTop

data = load_pickles("/path/to/project")
wells = ["WELL-1", "WELL-2", "WELL-3"]

section_data = prepare_section_data(
    data, wells,
    datum_surface="K-VIII",
    formation_tops={"K-VIII": None, "K-VIII-Base": None},
)

config = SectionPlotConfig(
    formation_tops={
        "K-VIII": FormationTop(color="red", label="K-VIII"),
        "K-VIII-Base": FormationTop(color="blue", label="K-VIII Base"),
    },
    datum_surface="K-VIII",
)

plot_correlation_section(section_data, wells, "Section A-A'",
                         "section_AA.png", config)
```

## 6. Synthetic Seismogram

Generate a synthetic seismogram tie:

```python
from geoagent.synthetic.functions import (
    create_reflectivity,
    create_synthetic_seismic_valid,
)
from geoagent.synthetic.wavelet_functions import ricker_wavelet
import numpy as np

# Acoustic impedance = velocity * density
velocity = np.random.uniform(2500, 4000, 500)
density = np.random.uniform(2.1, 2.7, 500)
impedance = velocity * density

# Reflectivity series
refl = create_reflectivity(impedance)

# Convolution with Ricker wavelet
wavelet = ricker_wavelet(f=30, length=51, dt=0.002)
times = np.linspace(1400, 1600, len(refl))
synthetic = create_synthetic_seismic_valid(refl, wavelet, times)
```

## 7. Bulk Shift Scan

Find optimal time shift for synthetic-to-seismic correlation:

```python
from geoagent.synthetic.bulk_shift import audit_tdr_bulk_shifts
from geoagent import CoreDataManager

dm = CoreDataManager("/path/to/project")
results = audit_tdr_bulk_shifts(dm, shift_range=(-20, 20), shift_step=1.0)

for r in results:
    print(f"{r['well']}: best_shift={r['best_shift']:.1f}ms, CC={r['best_cc']:.3f}")
```

## Next Steps

- See `examples/` for complete runnable scripts
- See [API Reference](api_reference.md) for detailed function signatures
- See [Data Formats](data_formats.md) for supported file formats
