# Installation

## Requirements

- Python 3.10 or later
- Operating System: Windows, macOS, or Linux

## Install from Source

```bash
git clone https://github.com/arunbharadwaj/geoagent.git
cd geoagent
pip install -e .
```

## Install from GitHub

```bash
pip install git+https://github.com/arunbharadwaj/geoagent.git
```

## Dependencies

All dependencies are installed automatically:

| Package | Purpose |
|---------|---------|
| numpy | Array computation |
| scipy | Interpolation, signal processing |
| pandas | Tabular data handling |
| matplotlib | Plotting and visualization |
| segyio | SEG-Y seismic file I/O |
| lasio | LAS well log file I/O |
| tqdm | Progress bars |

### Optional Dependencies

```bash
# ZMAP horizon grid support
pip install zmapio

# Development (testing)
pip install -e ".[dev]"
```

## Verify Installation

```python
import geoagent
print(geoagent.__version__)  # 0.1.0

from geoagent import CoreDataManager
dm = CoreDataManager()
print(dm)  # CoreDataManager(wells=0, surveys=0, project='')
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All 65 tests should pass in under 10 seconds.
