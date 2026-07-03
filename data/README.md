# Data Directory

## Structure
```
data/
├── raw/
│   ├── kepler/     # Raw Kepler FITS files (not tracked by Git)
│   └── tess/       # Raw TESS FITS files (not tracked by Git)
└── processed/
    ├── kepler/     # Preprocessed .npy arrays (not tracked by Git)
    └── tess/       # Preprocessed .tess arrays (not tracked by Git)
```

## How to Download Data

### Kepler
```python
import lightkurve as lk
result = lk.search_lightcurve("Kepler-17", mission="Kepler")
lc = result[0].download()
```

### TESS
```python
import lightkurve as lk
result = lk.search_lightcurve("TOI-270", mission="TESS")
lc = result[0].download()
```

## Dataset Stats (update as you download)
- Kepler confirmed planets: 0
- Kepler false positives: 0
- TESS candidates: 0
- Total: 0
