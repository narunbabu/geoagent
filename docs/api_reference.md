# API Reference

Complete public API for the GeoAgent package, organized by module.

## Core

### `geoagent.core.data_manager.CoreDataManager`

Central data facade — all data access goes through this class.

```python
class CoreDataManager:
    def __init__(self, project_folder=None)
    def load_project(self, project_folder)
    def save_project(self, project_folder=None)
    def get_data(self, data_type)
    def get_log_naming_settings(self)
    def set_log_naming_settings(self, log_settings)
    def find_log_column(self, log_type, well_name)
    def get_synthetic_settings_all(self)
    def get_synthetic_settings(self, well_name)
    def load_well_settings(self, well_name)
    def get_default_well_settings(self, well_name)
    def store_synthetic_result(self, well_name, result)
    def get_synthetic_result(self, well_name)
    def save_synthetic_results(self, file_path)
    def load_synthetic_results(self, file_path)
    def get_seismic_data(self, well_name, required_attributes, use_spatial_sampling=False,
                         well_trajectory_data=None, spatial_sampling_params=None, parent_widget=None)
    def find_survey_with_attributes(self, required_attributes)
    def get_well_tdr(self, well_name)
    def get_well_data(self, well_name)
    def get_available_wells(self)
    def get_available_surveys(self)
```

**Key methods:**

- `get_data(data_type)` — Multi-handler fallback: searches CoreDataManager → well_handler → well_log_handler → seismic_handler → horizons_handler
- `find_log_column(log_type, well_name)` — Resolves mnemonic aliases (e.g., `'sonic_log'` → `'DTC'`)
- `get_default_well_settings(well_name)` — Returns `{'bulk_shift': 0.0, 'sampling_interval': 2.0, ...}`

---

## Plotting

### `geoagent.plotting.seismic_plotter`

```python
def plot_seismic_section(traces, time_axis, *, trace_positions=None,
                         display='variable_density', cmap='seismic',
                         clip_percentile=99, wiggle_fill=True, wiggle_scale=1.0,
                         horizons=None, well_markers=None,
                         title='Seismic Section', xlabel='Trace', ylabel='TWT (ms)',
                         figsize=None, ax=None, save_path=None, dpi=150)
```
Variable-density or wiggle trace seismic display with optional horizon overlays and well markers.

- **traces**: 2D array (time_samples × num_traces)
- **horizons**: `{name: (positions, times, color)}` dict
- **well_markers**: `[{'position': int, 'name': str, 'color': str}]`
- **Returns**: `(fig, ax)` tuple

```python
def plot_trace_gather(traces, time_axis, *, well_name='', attribute_name='',
                      figsize=(6, 8), ax=None, save_path=None, dpi=150)
```
Small trace gather display for well ties.

### `geoagent.plotting.well_panel`

```python
def plot_well_panel(depth, logs, *, well_name='', depth_range=None,
                    formation_tops=None, gr_range=(0, 150), lld_range=(0.2, 2000),
                    nphi_range=(0.60, -0.15), rhob_range=(1.95, 2.95),
                    dt_range=(140, 40), gr_cutoff=75,
                    figsize=(12, 10), save_path=None, dpi=150)
```
5-track single-well log panel (Depth | GR | LLD | NPHI+RHOB | DT).

- **depth**: 1D numpy array
- **logs**: `{'GR': array, 'LLD': array, 'NPHI': array, 'RHOB': array, 'DT': array}`
- **formation_tops**: `{'name': {'md': float, 'color': str}}`
- **Returns**: `(fig, axes)` tuple

### `geoagent.plotting.section_plotter`

```python
def plot_correlation_section(section_data, well_order, title, output_path,
                             config, distances=None)
```
Multi-well log correlation section with datum flattening and formation top connections.

- **section_data**: Output of `prepare_section_data()`
- **config**: `SectionPlotConfig` dataclass

### `geoagent.plotting.map_plotter`

```python
def plot_location_map(sections_config, well_heads, output_path, deviation=None,
                      figure_dpi=200, title=None, polygon=None,
                      show_north_arrow=True, show_scale_bar=True)
```
Well location map with deviated paths, section lines, north arrow, and scale bar.

### `geoagent.plotting.config`

```python
@dataclass
class FormationTop:
    color: str
    linestyle: str = '-'
    linewidth: float = 1.5
    label: str = ''

@dataclass
class SectionPlotConfig:
    formation_tops: Dict[str, FormationTop]
    datum_surface: str = ''
    interval_top_surface: str = ''
    interval_base_surface: str = ''
    track_widths: Dict[str, float]   # default: depth=1.4, GR=1.0, LLD=1.0, NPHI_RHOB=1.2, DT=1.0
    gr_range: Tuple = (0, 150)
    lld_range: Tuple = (0.2, 2000)
    nphi_range: Tuple = (0.70, 0.05)
    rhob_range: Tuple = (1.95, 2.95)
    dt_range: Tuple = (140, 40)
    gr_sand_cutoff: float = 75
    window_above: float = 35
    window_below: float = 55
    figure_dpi: int = 200
    # ... plus color settings for each track
```

---

## Well Utilities

### `geoagent.well.deviation_utils`

```python
def get_dev_dataframe(deviation, well_name)
```
Extract DataFrame from deviation dict (handles both dict and DataFrame formats).

```python
def compute_tvdss(deviation, well_name, md_array, kb)
```
Convert MD array to TVDSS using deviation survey. Returns numpy array or None.

```python
def get_well_path_xy(deviation, well_name)
```
Get well path X, Y coordinates for map plotting. Returns `(x, y)` tuple or None.

### `geoagent.well.tops_utils`

```python
def get_formation_md(well_tops, well_name, surface_name)
```
Get formation top MD for a well. Returns float or None.

```python
def get_well_kb(well_heads, well_name)
```
Get Kelly Bushing elevation. Returns float or None.

```python
def get_well_coordinates(well_heads, well_name)
```
Get surface X, Y coordinates. Returns `(x, y)` tuple or None.

### `geoagent.well.log_windower`

```python
def prepare_section_data(data, well_list, *, datum_surface, formation_tops,
                         window_above=35, window_below=55, aliases=None)
```
Prepare windowed log data for correlation section plotting. Returns dict keyed by well name.

```python
def compute_well_distances(section_data, well_order)
```
Compute inter-well distances from surface coordinates.

### `geoagent.well.mnemonic_resolver`

```python
class LogNamingSettings:
    def __init__(self, log_names=None)
    def find_log_column(self, log_type, columns)
    def add_variant(self, log_type, variant)
    def validate_log_configuration(self, columns)
    def reset_to_defaults(self)
    def to_dict(self) / from_dict(cls, d)
```
Configurable mnemonic resolution with 3-tier matching (direct → case-insensitive → partial).

---

## Synthetic

### `geoagent.synthetic.functions`

```python
def create_reflectivity(impedance)
```
Compute reflection coefficient series from acoustic impedance. Returns array of length `N-1`.

```python
def create_synthetic_seismic_valid(reflectivity, wavelet, seismic_times_extracted,
                                    log_sampling_interval=None)
```
Convolve reflectivity with wavelet to produce synthetic seismogram.

```python
def calculate_correlation_with_shift(seismic_trace, seismic_times,
                                      synthetic_seismic, synthetic_times, shift=0.0)
```
Calculate correlation coefficient between seismic and synthetic with time shift. Returns float.

```python
def trigger_upscaling_workflow(data_manager, well_name, parent_widget=None)
```
Headless stub (logs warning, returns False). In SeisTrans Pro this opens a dialog.

### `geoagent.synthetic.bulk_shift`

```python
def audit_tdr_bulk_shifts(data_manager)
```
Audit all wells for TDR quality and existing bulk shifts. Returns list of dicts.

```python
def compute_bulk_shift_scan(data_manager, well_name, survey, attribute,
                            shift_range=(-50, 50), shift_step=2,
                            wavelet=None, extract_range=None)
```
Scan a range of time shifts to find optimal synthetic-to-seismic correlation.

### `geoagent.synthetic.wavelet_functions`

```python
def ricker_wavelet(f, length, dt)         # Mexican hat / Ricker
def ormsby_wavelet(f1, f2, f3, f4, length, dt)  # Bandpass trapezoidal
def klauder_wavelet(f1, f2, length, dt, phase=0)
def morlet_wavelet(f, length, dt)
def gabor_wavelet(f, sigma, length, dt)
def berlage_wavelet(f, n, alpha, length, dt)
def butterworth_wavelet(f, order, length, dt)
```

Wavelet extraction:
```python
def roy_white_method(seismic_trace, reflectivity, wavelet_length=64, epsilon=1e-6)
def constraints_inversion_method(seismic_trace, reflectivity, wavelet_length=64,
                                  lambda1=0.1, lambda2=0.1)
def extract_wavelet_deterministic(seismic_trace, reflectivity, wavelet_length=64, taper_alpha=0.1)
def frequency_domain_wavelet(trace_data, wavelet_length=64)
def extract_wavelet_autocorrelation(trace, wavelet_length)
def evaluate_wavelet(actual_seismic, reflectivity, wavelet)
```

---

## I/O

### `geoagent.io.project_loader`

```python
def load_pickles(project_dir, pickle_names=None)
```
Load pickle files from a project directory. Default names: `well_heads`, `well_tops`, `well_logs`, `checkshot`, `deviation`.

### `geoagent.io.polygon_utils`

```python
def load_polygon(project_dir, key='block_boundary')
```
Load polygon vertices from `polygons.pkl`.

---

## Utilities

### `geoagent.utils.interpolation`

```python
def interpolate_horizon_at_points(horizon_data, x_points, y_points, method='linear')
```
Interpolate horizon Z values at arbitrary (x, y) points using `RegularGridInterpolator`.

```python
def interpolate_horizon_at_wells(horizon_data, well_coords, method='linear')
```
Interpolate at well locations. Accepts dict `{name: (x, y)}` or list `[(x, y)]`.

```python
def extract_horizon_along_line(horizon_data, x_line, y_line, method='linear')
```
Extract horizon Z values along a line (e.g., seismic section path).

### `geoagent.utils.trace_spatial_indexer`

```python
class TraceSpatialIndexer:
    def __init__(self, coordinates)
    def find_nearest_trace(self, x, y, max_distance=None)
    def get_traces_within_radius(self, x, y, radius)
    def find_nearest_traces_batch(self, coordinates)
    def get_trace_coordinate(self, index)
    def get_coverage_info(self)
```
KD-tree based spatial index for fast trace coordinate lookups.
