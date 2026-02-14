# Changelog

## [0.1.0] - 2026-02-14

### Added
- **Core data handlers**: seismic (SEGY via segyio), well (Petrel ASCII), well logs (LAS via lasio), horizons (ZMAP)
- **CoreDataManager**: Qt-free central data facade with multi-handler fallback chain
- **LogNamingSettings**: 3-tier mnemonic resolution (direct → case-insensitive → partial)
- **Synthetic seismogram pipeline**: impedance → reflectivity → convolution, headless (no Qt)
- **Bulk shift engine**: cross-correlation scan with automatic best-shift detection
- **Wavelet generation**: Ricker, Ormsby, Klauder, Morlet, Gabor, Berlage, Butterworth
- **Wavelet extraction**: Roy-White, constrained inversion, deterministic, frequency domain, autocorrelation
- **Seismic section plotter**: variable-density and wiggle trace display with horizon overlays and well markers
- **Well log panel plotter**: 5-track single-well display (Depth, GR, LLD, NPHI+RHOB, DT) with formation tops
- **Well log correlation section plotter**: 6-track multi-well sections with datum flattening and top connections
- **Location map plotter**: 1:1 aspect, deviated well paths, north arrow, scale bar, block boundary
- **Horizon interpolation**: RegularGridInterpolator wrapper for point, well, and line extraction
- **TraceSpatialIndexer**: KD-tree spatial index for fast trace coordinate lookups
- **Well utilities**: deviation processing, formation tops, log windowing, mnemonic resolution
- **Project loader**: pickle-based project persistence
- **5 example scripts**: project loading, well panels, maps, synthetic ties, bulk shift scans
- **2 Jupyter notebooks**: quickstart walkthrough, well correlation workflow
- **65 unit tests** across 9 test modules, all passing
- **Documentation**: installation guide, quickstart, API reference, data format specification

### Technical Notes
- Pure Python — no PyQt6, no PyTorch, no scikit-learn required
- 7 dependencies: numpy, scipy, pandas, matplotlib, segyio, lasio, tqdm
- Python 3.10+ compatible
- Extracted from SeisTrans (~116K lines) with all Qt dependencies removed
