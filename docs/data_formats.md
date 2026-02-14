# Data Formats

GeoAgent supports industry-standard geoscience file formats for seismic, well, and horizon data.

## Supported Formats

| Type | Format | Extension | Library |
|------|--------|-----------|---------|
| Seismic traces | SEG-Y | `.sgy`, `.segy` | segyio |
| Well logs | LAS 2.0 | `.las` | lasio |
| Well heads | Petrel ASCII | `.asc` | built-in parser |
| Deviation surveys | Petrel ASCII | `.dev` | built-in parser |
| Checkshot data | Petrel ASCII | `.asc` | built-in parser |
| Formation tops | Petrel ASCII | `.tops` | built-in parser |
| Horizons | ZMAP grid | `.zmap` | zmapio (optional) |
| Project persistence | Python pickle | `.pkl` | pickle |

## Project Directory Structure

GeoAgent expects a project directory containing pickle files:

```
project_folder/
├── well_heads.pkl        # pandas DataFrame (indexed by well name)
├── well_logs.pkl         # dict: {well_name: {curve_name: numpy_array}}
├── well_tops.pkl         # dict: {well_name: {top_name: {MD, TVD, Z, TWT Auto}}}
├── deviation.pkl         # dict: {well_name: {well_info: {...}, dev_data: DataFrame}}
├── check_shot_data.pkl   # dict: {well_name: DataFrame(MD, TWT picked)}
├── upscaled_logs.pkl     # dict: {well_name: DataFrame} (optional)
├── seismic/              # SEG-Y files (optional)
│   └── volume.sgy
└── horizons/             # Horizon grids (optional)
    └── surface.zmap
```

## Well Heads (well_heads.pkl)

Pandas DataFrame indexed by well name:

| Column | Type | Description |
|--------|------|-------------|
| Surface X | float | Easting (m) |
| Surface Y | float | Northing (m) |
| KB | float | Kelly Bushing elevation (m above sea level) |
| Latitude_dd | float | Decimal degree latitude (optional) |
| Longitude_dd | float | Decimal degree longitude (optional) |

## Well Logs (well_logs.pkl)

Dictionary keyed by well name. Each value is a dictionary of curve arrays:

```python
{
    "WELL-1": {
        "DEPT": np.array([...]),   # Depth curve (MD)
        "GR": np.array([...]),     # Gamma Ray
        "DTC": np.array([...]),    # Compressional sonic
        "RHOB": np.array([...]),   # Bulk density
        "NPHI": np.array([...]),   # Neutron porosity
        "LLD": np.array([...]),    # Deep resistivity
    }
}
```

GeoAgent resolves common mnemonic aliases automatically (e.g., DT/DTC/DTCO all map to sonic).

## Deviation Surveys (deviation.pkl)

Dictionary keyed by well name with nested structure:

```python
{
    "WELL-1": {
        "well_info": {"well_name": "WELL-1", "KB": 25.0},
        "dev_data": pd.DataFrame({
            "MD": [...],    # Measured depth
            "TVD": [...],   # True vertical depth
            "X": [...],     # Easting
            "Y": [...],     # Northing
        })
    }
}
```

## Formation Tops (well_tops.pkl)

Nested dictionary:

```python
{
    "WELL-1": {
        "K-VIII": {"MD": 950.5, "TVD": 948.2, "Z": 923.2, "TWT Auto": 1452.3},
        "K-VIII-Base": {"MD": 1020.0, "TVD": 1018.1, "Z": 993.1, "TWT Auto": 1478.5},
    }
}
```

Key fields:
- **MD**: Measured depth (meters)
- **TVD**: True vertical depth (meters)
- **Z**: TVDSS = TVD - KB (positive-down)
- **TWT Auto**: Two-way travel time (ms, positive-down)

## Checkshot Data (check_shot_data.pkl)

Dictionary keyed by well name, each value a DataFrame:

| Column | Type | Description |
|--------|------|-------------|
| MD | float | Measured depth (m) |
| TWT picked | float | Two-way time (ms, stored negative — use `abs()`) |

## Horizon Grids

Horizon data as dictionary:

```python
{
    "X": np.array([...]),       # 1D x-coordinates
    "Y": np.array([...]),       # 1D y-coordinates
    "Z": np.array([[...]]),     # 2D grid of Z values
}
```

Z can be TWT (ms) or depth (m), depending on the surface type.

## Sign Conventions

| Quantity | Convention |
|----------|-----------|
| TWT | Positive-down (increasing with depth) |
| TWT picked (checkshot) | Stored negative — always use `abs()` |
| Z (TVDSS) | Positive-down: Z = TVD - KB |
| Depth (MD, TVD) | Positive-down |

## Mnemonic Alias Resolution

GeoAgent's `LogNamingSettings` resolves common well log curve names:

| Log Type | Common Aliases |
|----------|---------------|
| sonic_log | DT, DTC, DTCO, DT4P, AC |
| density_log | RHOB, RHOZ, DEN, ZDEN |
| gr_log | GR, SGR, CGR |
| resistivity_log | LLD, ILD, RT, RILD |
| neutron_log | NPHI, TNPH, NPOR, NEU |

See `assets/mnemonics.txt` for the full alias table.
