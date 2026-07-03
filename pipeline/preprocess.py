"""
LightSeek — Preprocessing Pipeline
Loads raw light curves, cleans, normalizes, and phase-folds them.
Author: Prajwal K / Team Integral X
"""

import numpy as np
import lightkurve as lk


def download_lightcurve(target: str, mission: str = "kepler", quarter: int = 1):
    """Download a light curve from NASA archives."""
    result = lk.search_lightcurve(target, mission=mission, quarter=quarter)
    if len(result) == 0:
        raise ValueError(f"No light curves found for {target}")
    lc = result[0].download()
    return lc


def clean(lc):
    """Remove NaNs and outliers, normalize flux."""
    lc = lc.remove_nans()
    lc = lc.remove_outliers(sigma=5.0)
    lc = lc.normalize()
    return lc


def to_array(lc, n_points: int = 1000):
    """Convert light curve to fixed-length numpy array."""
    time = lc.time.value
    flux = lc.flux.value
    # Interpolate to fixed length
    time_uniform = np.linspace(time.min(), time.max(), n_points)
    flux_interp = np.interp(time_uniform, time, flux)
    return time_uniform, flux_interp


def preprocess(target: str, mission: str = "kepler"):
    """Full preprocessing pipeline."""
    print(f"→ Downloading {target} from {mission}...")
    lc = download_lightcurve(target, mission)

    print(f"→ Cleaning light curve...")
    lc = clean(lc)

    print(f"→ Converting to array...")
    time, flux = to_array(lc)

    print(f"✓ Done. Shape: {flux.shape}")
    return time, flux


if __name__ == "__main__":
    # Quick test
    time, flux = preprocess("Kepler-17", mission="kepler")
    print(f"Time range: {time[0]:.2f} — {time[-1]:.2f} days")
    print(f"Flux range: {flux.min():.6f} — {flux.max():.6f}")
