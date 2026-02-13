# Changelog

## [0.1.0] - 2026-02-13

### Added
- Initial open-source extraction from SeisTrans
- Core data handlers: seismic (SEGY), well (Petrel ASCII), well logs (LAS), horizons (ZMAP)
- Synthetic seismogram generation pipeline
- Bulk shift cross-correlation scan
- Wavelet generation (Ricker, Ormsby) and extraction (Roy-White, constrained inversion, deterministic, frequency domain, autocorrelation)
- Well log correlation section plotter (6-track, datum-flattened)
- Location map plotter with deviation paths, north arrow, scale bar
- Synthetic tie 8-panel figure generator
- Project loader for pickle-based persistence
- Well data utilities: deviation, tops, mnemonic resolution, log windowing
