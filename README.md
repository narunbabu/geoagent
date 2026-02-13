# GeoAgent — Open-Source Geoscience Data Toolkit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Load well data, generate synthetic seismograms, create professional well log correlation sections, and plot location maps — all in pure Python. No PyTorch, no GUI required.

## Install

```bash
git clone https://github.com/arunbharadwaj/geoagent.git
cd geoagent
pip install -e .

# Or directly from GitHub:
pip install git+https://github.com/arunbharadwaj/geoagent.git
```

**Dependencies** (all installed automatically): numpy, scipy, pandas, matplotlib, segyio, lasio, tqdm

## Quick Start

```python
from geoagent.io.project_loader import load_pickles
from geoagent.well.log_windower import prepare_section_data
from geoagent.plotting.section_plotter import plot_correlation_section
from geoagent.plotting.config import SectionPlotConfig, FormationTop

# Load a project
data = load_pickles("path/to/project")
print(f"Wells: {data['well_heads']['Name'].tolist()}")

# Generate a correlation section
section_data = prepare_section_data(
    data, ["WELL-1", "WELL-2", "WELL-3"],
    datum_surface="Target-Horizon",
    formation_tops={"Target-Horizon": None, "Base-Horizon": None}
)

config = SectionPlotConfig(
    formation_tops={
        "Target-Horizon": FormationTop(color="#1565C0", label="Target"),
        "Base-Horizon": FormationTop(color="#2E7D32", label="Base"),
    },
    datum_surface="Target-Horizon",
)

plot_correlation_section(section_data, ["WELL-1", "WELL-2", "WELL-3"],
                         "My Section", "output/section.png", config)
```

## Features

**Data Loading**
- SEGY seismic via `segyio` with KD-tree spatial indexing
- LAS well logs via `lasio` with mnemonic alias resolution
- Petrel ASCII format: well heads (.asc), deviation surveys (.dev), checkshots (.asc), formation tops (.tops)
- ZMAP horizon grids (optional `zmapio` dependency)
- Pickle-based project persistence

**Synthetic Seismograms**
- Full synthetic tie pipeline: logs → impedance → reflectivity → convolution
- Bulk shift cross-correlation scan with automatic best-shift detection
- Wavelet generation: Ricker, Ormsby (bandpass)
- Wavelet extraction: Roy-White, constrained inversion, deterministic optimization, frequency domain, autocorrelation
- Phase rotation via Hilbert transform
- Sampling interval auto-guard (0.25ms ML vs 2ms seismic)

**Visualization**
- Well log correlation sections: 6-track layout (MD, TVDSS, GR, LLD, NPHI+RHOB, DT), datum flattening, formation top connection lines
- Location maps: 1:1 aspect ratio, deviated well paths, north arrow, scale bar, block boundary polygon
- Synthetic tie 8-panel figures: TDR, sonic, density, impedance, reflectivity, seismic, synthetic, wavelet

**Well Data Utilities**
- Deviation survey processing: TVDSS computation, well path XY extraction
- Formation top management with property derivation chain (MD → TVD → Z → TWT)
- Log mnemonic resolution (GR, LLD, RHOB, NPHI, DT with industry aliases)
- Log windowing around datum surfaces for section plotting

## Project Structure

```
geoagent/
├── core/          # Data handlers (seismic, well, well_log, horizons, wavelet)
├── io/            # File I/O (project loader, polygon utils)
├── well/          # Well utilities (deviation, tops, log windowing, mnemonics)
├── synthetic/     # Synthetic tie pipeline, bulk shift, wavelet functions
├── plotting/      # Section plotter, map plotter, config dataclasses
└── utils/         # Shared utilities (safe_print, twt_utils, interpolation)
```

## Data Format Support

| Type | Format | Extension |
|------|--------|-----------|
| Seismic | SEG-Y | `.sgy` |
| Well Logs | LAS | `.las` |
| Well Heads | Petrel ASCII | `.asc` |
| Deviation | Petrel ASCII / Excel | `.dev`, `.xlsx` |
| Checkshots | Petrel ASCII | `.asc` |
| Formation Tops | Petrel ASCII | `.tops` |
| Horizons | ZMAP grid | `.zmap` |

## GeoAgent Pro & Enterprise

GeoAgent Core is the open-source foundation. For ML-powered reservoir characterization, autonomous agent workflows, and interactive desktop GUI, see **GeoAgent Pro** and **GeoAgent Enterprise**.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
