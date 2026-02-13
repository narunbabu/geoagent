# CLAUDE.md

## Project Overview

**GeoAgent** is the open-source core of the SeisTrans geoscience platform. Pure Python toolkit for well data loading, synthetic seismogram generation, bulk shift analysis, well log correlation sections, and location maps. No PyTorch, no PyQt6, no GUI — runs headless as a library.

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

## Testing

- Unit tests use in-memory data (no external files needed)
- Integration tests use real Bakrol 3D data at `C:\Delete_ML_Projects\test2` (manual, not in CI)
- Sample dataset will be added later (anonymized real data)

## License

Apache-2.0
