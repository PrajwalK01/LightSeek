"""
LightSeek — Wavelet Denoising (W2 upgrade)
Isolates transit-frequency components from stellar noise.
Author: Nagesh R / Team Integral X
"""

import numpy as np
import pywt


def wavelet_denoise(flux: np.ndarray,
                    wavelet: str = 'sym8',
                    level: int = 4,
                    threshold_mode: str = 'soft') -> np.ndarray:
    """
    Denoise light curve using wavelet thresholding.
    Removes high-frequency noise while preserving transit shape.
    """
    # Decompose
    coeffs = pywt.wavedec(flux, wavelet, level=level)

    # Universal threshold
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(flux)))

    # Threshold detail coefficients (keep approximation)
    coeffs_thresh = [coeffs[0]]
    for c in coeffs[1:]:
        coeffs_thresh.append(
            pywt.threshold(c, threshold, mode=threshold_mode)
        )

    # Reconstruct
    flux_denoised = pywt.waverec(coeffs_thresh, wavelet)

    # Match length
    return flux_denoised[:len(flux)]


def extract_transit_band(flux: np.ndarray,
                          wavelet: str = 'sym8',
                          level: int = 4,
                          transit_levels: list = None) -> np.ndarray:
    """
    Extract frequency band most likely to contain transit signals.
    Transit durations: 1-12 hours → mid-frequency detail coefficients.
    """
    if transit_levels is None:
        transit_levels = [2, 3]

    coeffs = pywt.wavedec(flux, wavelet, level=level)

    # Zero out non-transit bands
    coeffs_transit = [np.zeros_like(c) for c in coeffs]
    for lvl in transit_levels:
        if lvl < len(coeffs):
            coeffs_transit[lvl] = coeffs[lvl]

    return pywt.waverec(coeffs_transit, wavelet)[:len(flux)]


def full_wavelet_pipeline(flux: np.ndarray) -> np.ndarray:
    """Full wavelet processing: denoise + normalize."""
    flux = wavelet_denoise(flux)
    flux = (flux - flux.mean()) / (flux.std() + 1e-8)
    return flux


if __name__ == "__main__":
    print("Installing PyWavelets if needed...")
    import subprocess
    subprocess.run(["python", "-m", "pip", "install", "PyWavelets"])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Test on synthetic noisy transit
    t    = np.linspace(0, 30, 1000)
    flux = 1.0 + 0.003 * np.sin(2 * np.pi * t / 11.4)
    noise = np.random.normal(0, 0.004, 1000)
    flux += noise

    # Inject transit at day 5
    mask = np.abs(t - 5.0) < 0.1
    flux[mask] -= 0.012

    denoised = wavelet_denoise(flux)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))
    ax1.plot(t, flux,     color='gray', lw=0.5, label='Noisy')
    ax1.plot(t, denoised, color='blue', lw=1.0, label='Denoised')
    ax1.legend(); ax1.set_title('Wavelet Denoising')

    transit = extract_transit_band(flux)
    ax2.plot(t, transit, color='orange', lw=1.0)
    ax2.set_title('Transit Band Extracted')

    plt.tight_layout()
    plt.savefig('results/plots/wavelet_test.png', dpi=130)
    print("Saved results/plots/wavelet_test.png")
    print(f"Noise reduced: {flux.std():.4f} → {denoised.std():.4f}")
    print("Wavelet pipeline OK")