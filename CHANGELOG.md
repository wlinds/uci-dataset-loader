## Version 0.0.2 (2024-04-09)

### Summary
Separate GUI from statistical functions, improve naming, and refactor logic.

### Added

- Linear regression plot
    - `render_linreg_scatter()` now supports displaying predictions

### Changed
- Separated GUI from statistical functions
    - All GUI is now in `gui_components.py`
    - `get_top_correlations()` move from `utils.py` to `analysis.py`
    - All Scipy logic moved from `gui_components` to `analysis.py` 

- T-Test:
    - Renamed var `null_hypothesis` top `popmean` 
    - Renamed fn `render_histogram_2` to `render_ttest_histogram`
    - Renamed fn `run_ttest_1samp` to `render_ttest`

- Z-Test:
    - Would benefit more testing

### Fixed
- Minor naming improvements