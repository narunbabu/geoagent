# CLAUDE.md

## Project Overview

**GeoAgent** is the open-source core of the SeisTrans geoscience platform. Pure Python toolkit for well data loading, synthetic seismogram generation, bulk shift analysis, well log correlation sections, and location maps. No PyTorch, no PyQt6, no GUI — runs headless as a library.

**The Ultimate Goal**: If GeoAgent can programmatically build a project like test2 (the Bakrol 3D reference project), it passes as a proper Geophysicist. Every feature in GeoAgent must serve this end — creating SeisTrans-compatible interpretation projects from raw geoscience data.

## Running

```bash
# Install in dev mode
pip install -e .

# Run tests
pytest tests/ -v

# Verify clean imports (no torch/PyQt6)
python -c "import geoagent; import sys; assert 'torch' not in sys.modules; assert 'PyQt6' not in sys.modules; print('OK')"
```

## Architecture

```
geoagent/
├── core/       → Data handlers (seismic, well, well_log, horizons, wavelet)
│                 CoreDataManager is the Qt-free facade wrapping all handlers
├── io/         → File I/O (pickle project loader, polygon utils)
├── well/       → Well utilities (deviation, tops, log windowing, mnemonics)
├── synthetic/  → Synthetic tie pipeline, bulk shift CC scan, wavelet functions
├── plotting/   → Section plotter, map plotter, map utils, config dataclasses
└── utils/      → Shared utilities (safe_print, twt_utils, interpolation)
```

### Key Design Decisions

- **No Qt dependency**: CoreDataManager uses plain Python callbacks instead of pyqtSignal
- **No ML dependency**: Zero torch/sklearn imports anywhere in the package
- **Handlers copied full**: seismic_handler.py, well_handler.py etc. are complete copies from SeisTrans with only import path fixes — preserves all internal methods
- **Config via dataclasses**: SectionPlotConfig, FormationTop inject configuration without module-level globals

### Import Pattern

All internal imports use the `geoagent.` prefix:
```python
from geoagent.utils.safe_print import safe_print
from geoagent.well.deviation_utils import compute_tvdss
from geoagent.plotting.config import SectionPlotConfig
```

## Dependencies

**Required**: numpy, scipy, pandas, matplotlib, segyio, lasio, tqdm
**Optional**: zmapio (horizon grids), pytest (testing)
**Excluded**: torch, sklearn, PyQt6 (these belong in GeoAgent Pro)

## Relationship to SeisTrans

GeoAgent is extracted from `/mnt/c/ArunApps/SeisTransAgentic/`. Source mapping:

| GeoAgent | SeisTrans Source |
|----------|-----------------|
| `core/` | `datahandlers/` (minus DataManager Qt coupling) |
| `io/` | `tools/io/` |
| `well/` | `tools/well/` |
| `synthetic/` | `tools/synthetic/` + `utils/synthetic_functions.py` |
| `plotting/` | `tools/plotting/` |
| `utils/` | `utils/safe_print.py`, `utils/twt_utils.py` |

---

## SeisTrans Application Architecture

SeisTrans (aka SeisTransform 3.0) is the full PyQt6 desktop application at `/mnt/c/ArunApps/SeisTransAgentic/`.

### Entry Point & Main Window

- **`main.py`**: Creates `QApplication`, calls `create_modern_application()` from `ui/modern_integration.py`. Falls back to legacy `MainWindow` from `main_window.py`.
- **`main_window.py`**: `MainWindow(QMainWindow)` — initializes `DataManager`, UI managers (UIManager, DialogManager, ImportManager, ProjectManager, DockManager, WaveletManager, MenuHandler, SignalManager).
- **`ui/managers/project_manager.py`**: `ProjectManager` — handles New/Open/Save project workflows. Opens `.str` files, delegates data loading to `DataManager`.

### DataHandlers (the core data layer)

Located in `datahandlers/`:

| Handler | File | PKL Files Managed |
|---------|------|-------------------|
| `DataManager` | `DataManager.py` | `selection_indices.pkl`, `well_settings.pkl`, `synthetic_settings.pkl`, `import_settings.pkl`, `expanded_states.pkl`, `checked_states.pkl`, `parent_check_states.pkl`, `active_window_state.pkl`, `trace_location.pkl`, `dock_layout_state.pkl` |
| `SeismicHandler` | `seismic_handler.py` | `seismic_project_structure.pkl`, per-survey: `headerdata.pkl`, `endpoints.pkl`, `*_seis_params.pkl`, `*.sgy` files, `wavelets.pkl` |
| `WellHandler` | `well_handler.py` | `well_heads.pkl`, `well_tops.pkl`, `deviation.pkl`, `checkshot.pkl`, `checkshot_mapping.pkl`, `tdr_mappings.pkl`, `custom_trace_selections.pkl` |
| `WellLogHandler` | `well_log_handler.py` | `well_logs.pkl` |
| `HorizonsHandler` | `horizons_handler.py` | `horizons.pkl` |

### Project Open Flow

1. User selects `*.str` file via FileDialog
2. `ProjectManager.open_project_from_path(path)` → sets `base_folder` = dirname of .str file
3. `DataManager.load_project(file_path)` → sets `project_folder` = file_path without extension (e.g., `test2.str` → `test2/`)
4. Clears all handler caches, updates all handler project paths
5. Loads UI state PKLs (expanded_states, checked_states, active_window, trace_location, dock_layout)
6. Calls each handler's `load_project(project_folder)`:
   - `SeismicHandler.load_project()` → reads `seismic_project_structure.pkl`, iterates surveys, loads headerdata/endpoints/seis_params per volume
   - `WellHandler.load_project()` → reads well_heads, well_tops, deviation, checkshot, tdr_mappings PKLs
   - `WellLogHandler.load_project()` → reads well_logs.pkl
   - `HorizonsHandler.load_project()` → reads horizons.pkl
7. `ProjectManager.load_data()` → calls `DataManager.load_all_data()`, creates blank MapWidget + TracePlot windows
8. Restores navigation state, trace plot location, seismic attribute selection, dock layout

---

## PKL Format Requirements & Project File Structure

### The .str File

The `.str` file is a **plain text file** containing only `"Seismic Data Viewer Project"`. It serves as the project pointer — the actual project data lives in a folder with the same name (minus extension).

Example: `C:\Delete_ML_Projects\test2.str` points to `C:\Delete_ML_Projects\test2\`

### Project Folder Structure (Required for SeisTrans Compatibility)

```
project_name/                          # Same name as .str file (minus extension)
├── seismic_project_structure.pkl      # CRITICAL: Master index of all surveys and volumes
├── well_heads.pkl                     # Well header DataFrame
├── well_logs.pkl                      # Well log data dict
├── well_tops.pkl                      # Formation tops DataFrame
├── deviation.pkl                      # Well deviation surveys dict
├── checkshot.pkl                      # Checkshot/TDR data dict
├── checkshot_mapping.pkl              # Checkshot-to-well mapping DataFrame
├── tdr_mappings.pkl                   # Time-depth relationship mappings dict
├── horizons.pkl                       # Interpreted horizons dict
├── wavelets.pkl                       # Wavelet list
├── selection_indices.pkl              # UI state (optional)
├── well_settings.pkl                  # Per-well display settings (optional)
├── synthetic_settings.pkl             # Synthetic tie settings (optional)
├── import_settings.pkl                # Import filter settings (optional)
├── expanded_states.pkl                # Navigation tree state (optional)
├── checked_states.pkl                 # Checkbox states (optional)
├── active_window_state.pkl            # Active window: 'map' or 'trace' (optional)
├── trace_location.pkl                 # Last viewed trace location (optional)
├── dock_layout_state.pkl              # Qt dock layout (optional)
├── survey_subfolder/                  # One per seismic survey
│   ├── headerdata.pkl                 # Trace header lookup array
│   ├── endpoints.pkl                  # Inline/crossline endpoint coordinates
│   ├── kdtree_index.pkl               # Spatial index (optional, auto-rebuilt)
│   ├── AttributeName_AttributeName.sgy  # SEG-Y volume files
│   └── AttributeName_seis_params.pkl    # Per-volume seismic parameters
└── ...
```

### PKL Data Format Specifications

#### `seismic_project_structure.pkl` — List of survey dicts (MASTER INDEX)

```python
# Type: list[dict]
[
    {
        "survey_name": "Bakrol3D",           # Display name in UI
        "sub_folder_name": "bakrol3d",        # Subfolder within project dir
        "headerdata": "headerdata.pkl",       # Filename for trace headers
        "endpoints": "endpoints.pkl",         # Filename for line endpoints
        "survey_volumes": [                   # List of seismic volumes
            {
                "attribute_name": "Amplitude",                    # Volume display name
                "segy": "Amplitude_Amplitude.sgy",                # SEG-Y filename
                "seis_params": "Amplitude_seis_params.pkl"        # Params filename
            },
            # ... more volumes
        ]
    },
    # ... more surveys
]
```

#### `headerdata.pkl` — Trace header lookup (per survey)

```python
# Type: numpy.ndarray, shape=(num_traces, 5), dtype=float64
# Columns: [trace_index, inline, crossline, cdp_x, cdp_y]
# Example row: [0.0, 5.0, 24.0, 264685.0, 2539306.0]
```

#### `endpoints.pkl` — Inline/crossline line endpoints (per survey)

```python
# Type: dict with keys 'inlines' and 'crosslines'
{
    "inlines": {
        5: [[x_start, y_start], [x_end, y_end]],    # inline 5 endpoints
        6: [[x_start, y_start], [x_end, y_end]],    # inline 6 endpoints
        # ...
    },
    "crosslines": {
        24: [[x_start, y_start], [x_end, y_end]],
        # ...
    }
}
```

#### `*_seis_params.pkl` — Seismic volume parameters (per volume)

```python
# Type: dict
{
    "CDP_X": 181,              # Byte position for CDP X in SEG-Y trace header
    "CDP_Y": 185,              # Byte position for CDP Y
    "Inline": 189,             # Byte position for inline number
    "Crossline": 193,          # Byte position for crossline number
    "Sampling Rate": 2000.0,   # Microseconds (2000.0 = 2ms)
    "Number of Samples": 201,  # Samples per trace
    "Format": "IBM",           # Data format (IBM or IEEE)
    "Coord_Mult_Factor": 0.01, # Coordinate scalar multiplier
    "Start Time": 1200.0,      # ms — first sample time
    "End Time": 1600.0         # ms — last sample time
}
```

#### `well_heads.pkl` — Well header data

```python
# Type: pandas.DataFrame, shape=(num_wells, 20)
# Columns: ['Name', 'UWI', 'Well symbol', 'Surface X', 'Surface Y',
#           'Latitude', 'Latitude_dd', 'Longitude', 'Longitude_dd',
#           'Drilling structure', 'Well datum name', 'Well datum value',
#           'Well datum description', 'TD (MD)', 'Cost', 'Spud date',
#           'Operator', 'TWT auto', 'Bottom hole X', 'Bottom hole Y']
# Key fields:
#   Name: "BK-1", "BK-2", etc.
#   Surface X/Y: UTM coordinates (meters)
#   Well datum name: "KB" (Kelly bushing)
#   Well datum value: KB elevation in meters (e.g., 50.37)
#   TD (MD): Total depth measured depth in meters
```

#### `well_logs.pkl` — Well log curves

```python
# Type: dict[str, dict[str, numpy.ndarray]]
# Keys are well names, values are dicts of curve_name → 1D array
{
    "BK-8": {
        "DEPTH": ndarray(shape=(8073,)),   # Measured depth in meters
        "DT": ndarray(shape=(8073,)),      # Sonic (us/ft)
        "GR": ndarray(shape=(8073,)),      # Gamma Ray (API)
        "LLD": ndarray(shape=(8073,)),     # Deep Resistivity (ohm.m)
        "NPHI": ndarray(shape=(8073,)),    # Neutron Porosity (v/v)
        "RHOB": ndarray(shape=(8073,))     # Bulk Density (g/cc)
    },
    # ...
}
```

#### `deviation.pkl` — Well deviation surveys

```python
# Type: dict[str, dict]
{
    "AMBD-164": {
        "well_info": {
            "name": "AMBD-164",
            "x": 270179.76,      # Surface X (UTM)
            "y": 2539727.83,     # Surface Y (UTM)
            "kb": 53.5           # KB elevation (meters)
        },
        "dev_data": ndarray(shape=(n_stations, 11))  # Deviation survey data
    },
    # ...
}
```

#### `horizons.pkl` — Interpreted horizon grids

```python
# Type: dict[str, dict]
# Keys are horizon names, values have X/Y/Z grids
{
    "K_VIII_B_fromAmid_middle": {
        "X": ndarray(shape=(ny, nx)),   # X coordinates grid (UTM)
        "Y": ndarray(shape=(ny, nx)),   # Y coordinates grid (UTM)
        "Z": ndarray(shape=(ny, nx))    # TWT values grid (ms)
    },
    # ...
}
```

#### `wavelets.pkl` — Wavelet definitions

```python
# Type: list[dict]
[
    {
        "name": "Ricker_30",
        "data": ndarray(shape=(128,)),           # 1D wavelet amplitude array
        "metadata": {
            "type": "Ricker",
            "frequency": 30.0,
            "sample_rate": 0.002,                # seconds
            "length": 128                        # samples
        }
    },
    {
        "name": "BK-31",                         # Extracted wavelet from well
        "data": ndarray(shape=(257, 2)),          # 2-column: [time, amplitude]
        "metadata": {
            "WAVELET-TFS": -256.0,
            "SAMPLE-RATE": 2.0,
            "Filename": "",
            "Sample rate": "0.00200000 seconds",
            # ...
        }
    }
]
```

#### `tdr_mappings.pkl` — Time-depth relationship assignments

```python
# Type: dict[str, dict]
# Maps each well to its TDR source
{
    "BK-14": {
        "source_well": "BK-14",       # Well that owns the checkshot
        "checkshot_id": "CS_001"       # Checkshot identifier
    },
    # Wells can share TDRs from other wells
    "BK-15": {
        "source_well": "BK-14",       # Using BK-14's checkshot
        "checkshot_id": "CS_001"
    },
    # ...
}
```

---

## The test2 Reference Project (Quality Standard)

**Location**: `C:\Delete_ML_Projects\test2` (folder) + `C:\Delete_ML_Projects\test2.str` (pointer)

The Bakrol 3D project is the gold standard for what GeoAgent must be able to produce. It contains:

### Project Contents

- **50 wells** (BK-1 through BK-38, AMBD-*, BHAVDA-1, HRPR-*, HIRAPUR-4)
- **39 wells with log data** (DT, GR, LLD, NPHI, RHOB curves)
- **38 wells with deviation surveys**
- **14 interpreted horizons** (K-VIII, K-IX, K-X series)
- **2 seismic surveys**: Bakrol3D (15+ attribute volumes) and ML_Volumes (10 volumes)
- **2 wavelets**: Ricker_30 (synthetic) and BK-31 (extracted)
- **Bakrol3D attributes**: Amplitude, Azim, Chaos, D1, D2, Edge, Env, Freq, Phase, Pol, Q, RmsAmpl, Sweet, TraceAGC, TraceGrad, Zone Volume + ML-derived (GR_AET, RHOB_AET, etc.)
- **Seismic time range**: 1200-1600ms TWT, 2ms sample rate, 201 samples/trace
- **Coordinate system**: UTM (X ~264000-274000, Y ~2539000-2548000)
- **Well datum**: Kelly bushing (KB), values 48-58m elevation

### What Makes test2 Complete

1. **Well-to-seismic tie**: Wells have checkshots → TDR mappings → synthetic seismograms
2. **Spatial index**: headerdata.pkl has 466,710 traces indexed by (inline, crossline, cdp_x, cdp_y)
3. **Cross-referenced data**: Well locations map onto seismic grid, horizons cover survey extent
4. **Multiple seismic attributes**: Original + derived volumes in same survey

---

## Creating SeisTrans-Compatible Projects (e.g., from Volve Data)

To create a project that opens correctly in SeisTrans Application, GeoAgent must produce:

### Minimum Required Files

1. **`.str` file** — Text file with content `"Seismic Data Viewer Project"`
2. **`seismic_project_structure.pkl`** — Master index (list of survey dicts, see format above)
3. **Survey subfolder** with:
   - `headerdata.pkl` — numpy array shape=(num_traces, 5) with [trace_idx, inline, xline, cdp_x, cdp_y]
   - `endpoints.pkl` — dict with 'inlines' and 'crosslines' line endpoint coordinates
   - `AttributeName_seis_params.pkl` — seismic parameters dict per volume
   - `AttributeName_AttributeName.sgy` — actual SEG-Y file (or symlink/copy)
4. **`well_heads.pkl`** — DataFrame with at minimum: Name, Surface X, Surface Y, Well datum name, Well datum value, TD (MD)
5. **`well_logs.pkl`** — dict of well_name → {DEPTH: array, curve1: array, ...}
6. **`deviation.pkl`** — dict of well_name → {well_info: {name, x, y, kb}, dev_data: ndarray}
7. **`wavelets.pkl`** — list (can be empty `[]` initially)

### Optional But Recommended

- `horizons.pkl` — Interpreted horizons with X/Y/Z grids
- `checkshot.pkl` + `tdr_mappings.pkl` — For well-to-seismic tie
- `well_tops.pkl` — Formation tops

### SEG-Y File Naming Convention

SeisTrans expects: `AttributeName_AttributeName.sgy` (e.g., `Amplitude_Amplitude.sgy`)
The seis_params PKL follows: `AttributeName_seis_params.pkl`

### Building headerdata from SEG-Y

```python
import segyio
import numpy as np

with segyio.open(segy_path, ignore_geometry=True) as f:
    n_traces = f.tracecount
    headerdata = np.zeros((n_traces, 5))
    for i in range(n_traces):
        h = f.header[i]
        headerdata[i] = [
            i,                                           # trace index
            h[segyio.TraceField.INLINE_3D],             # inline
            h[segyio.TraceField.CROSSLINE_3D],          # crossline
            h[segyio.TraceField.CDP_X] * coord_scalar,  # cdp_x
            h[segyio.TraceField.CDP_Y] * coord_scalar   # cdp_y
        ]
```

### Building endpoints from headerdata

```python
# Group by inline, find min/max crossline coordinates
endpoints = {"inlines": {}, "crosslines": {}}
for inline_num in unique_inlines:
    mask = headerdata[:, 1] == inline_num
    traces = headerdata[mask]
    endpoints["inlines"][int(inline_num)] = [
        [traces[0, 3], traces[0, 4]],     # start [x, y]
        [traces[-1, 3], traces[-1, 4]]     # end [x, y]
    ]
# Similarly for crosslines
```

### Volve-Specific Notes

When building projects from Volve seismic data:
- Volve uses UTM Zone 31N (EPSG:23031 or similar)
- SEG-Y byte positions may differ from Bakrol — read the binary/text headers first
- Check `segyio.tools.metadata(f)` for sample rate, sample count, format
- The coordinate scalar in trace headers may be negative (meaning divide, not multiply)
- Volve wells have deviation surveys — load from .dev or LAS files
- Well logs come as LAS files — use `lasio` to read, extract DEPTH + curve arrays

### Validation Checklist

A properly built project should:
- [ ] Open in SeisTrans via File → Open Project → select .str file
- [ ] Show seismic data in trace viewer (inline/crossline navigation works)
- [ ] Show well locations on map view
- [ ] Display well log curves when a well is selected
- [ ] Allow horizon overlay on seismic sections
- [ ] Support synthetic seismogram generation (if checkshots provided)

---

## Testing

- Unit tests use in-memory data (no external files needed)
- Integration tests use real Bakrol 3D data at `C:\Delete_ML_Projects\test2` (manual, not in CI)
- Sample dataset will be added later (anonymized real data)

## License

Apache-2.0
