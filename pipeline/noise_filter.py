"""
LightSeek — Noise Filter (W2)
Author: Nagesh R / Team Integral X
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt


def sigma_clip(flux: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    median = np.median(flux)
    std = np.std(flux)
    mask = np.abs(flux - median) > sigma * std
    flux_clean = flux.copy()
    flux_clean[mask] = median
    return flux_clean


def median_filter(flux: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    return medfilt(flux, kernel_size=kernel_size)


def detrend(flux: np.ndarray, window: int = 101) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    trend = uniform_filter1d(flux, size=window)
    detrended = flux / (trend + 1e-8)
    return detrended


def normalize(flux: np.ndarray) -> np.ndarray:
    mean = np.mean(flux)
    std = np.std(flux)
    return (flux - mean) / (std + 1e-8)


def full_pipeline(flux: np.ndarray,
                   sigma: float = 3.0,
                   median_kernel: int = 5,
                   detrend_window: int = 101) -> np.ndarray:
    flux = sigma_clip(flux, sigma)
    flux = median_filter(flux, median_kernel)
    flux = detrend(flux, detrend_window)
    flux = normalize(flux)
    return flux


if __name__ == "__main__":
    print("Testing noise filter...")
    t = np.linspace(0, 30, 1000)
    flux_clean = 1.0 + 0.002 * np.sin(2 * np.pi * t / 10)
    noise = np.random.normal(0, 0.003, 1000)
    flux_noisy = flux_clean + noise
    flux_noisy[200] = 1.5

    flux_filtered = full_pipeline(flux_noisy)

    print(f"Input  — mean: {flux_noisy.mean():.4f} std: {flux_noisy.std():.4f}")
    print(f"Output — mean: {flux_filtered.mean():.4f} std: {flux_filtered.std():.4f}")
    print(f"Outlier removed: {abs(flux_filtered[200]) < 3.0}")
    print("Noise filter OK")