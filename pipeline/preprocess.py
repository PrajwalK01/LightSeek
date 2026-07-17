"""
LightSeek — Preprocessing Pipeline (W2)
Author: Prajwal K / Team Integral X
"""

import numpy as np
import lightkurve as lk
import os


def download(target: str, mission: str = "kepler", quarter: int = 1):
    print(f"  Downloading {target} ({mission} Q{quarter})...")
    result = lk.search_lightcurve(target, mission=mission, quarter=quarter)
    if len(result) == 0:
        raise ValueError(f"No light curve found for {target}")
    lc = result[0].download()
    return lc


def clean(lc):
    lc = lc.remove_nans()
    lc = lc.remove_outliers(sigma=5.0)
    lc = lc.normalize()
    return lc


def to_array(lc, n_points: int = 1000):
    time = lc.time.value
    flux = lc.flux.value
    flux = np.nan_to_num(flux, nan=1.0)
    time_uniform = np.linspace(time.min(), time.max(), n_points)
    flux_interp = np.interp(time_uniform, time, flux)
    return time_uniform, flux_interp


def sigma_clip(flux: np.ndarray, sigma: float = 3.0):
    mean = np.mean(flux)
    std = np.std(flux)
    mask = np.abs(flux - mean) < sigma * std
    flux_clean = flux.copy()
    flux_clean[~mask] = mean
    return flux_clean


def preprocess_target(target: str, mission: str = "kepler", quarter: int = 1):
    lc = download(target, mission, quarter)
    lc = clean(lc)
    time, flux = to_array(lc)
    flux = sigma_clip(flux)
    return time, flux


def preprocess_and_save(target: str, label: int, save_dir: str,
                         mission: str = "kepler", quarter: int = 1):
    os.makedirs(save_dir, exist_ok=True)
    try:
        time, flux = preprocess_target(target, mission, quarter)
        safe_name = target.replace(" ", "_").replace("-", "_")
        np.save(os.path.join(save_dir, f"{safe_name}_flux.npy"), flux)
        np.save(os.path.join(save_dir, f"{safe_name}_time.npy"), time)
        print(f"  Saved {target} → label={label} | points={len(flux)}")
        return True
    except Exception as e:
        print(f"  FAILED {target}: {e}")
        return False


if __name__ == "__main__":
    print("Testing preprocessing pipeline...")
    time, flux = preprocess_target("Kepler-17", mission="kepler")
    print(f"Time range: {time[0]:.1f} to {time[-1]:.1f} days")
    print(f"Flux shape: {flux.shape}")
    print(f"Flux min/max: {flux.min():.4f} / {flux.max():.4f}")
    print("Preprocessing pipeline OK")