"""
LightSeek — DETECT Agent (W4)
Validates periodicity, SNR, and transit shape.
Author: Prajwal K / Team Integral X
"""

import numpy as np
from astropy.timeseries import BoxLeastSquares
import astropy.units as u


def run(time: np.ndarray, flux: np.ndarray) -> dict:
    """
    DETECT Agent — Run BLS periodogram + transit validation.
    Returns structured JSON report.
    """

    try:
        # BLS periodogram
        bls = BoxLeastSquares(time * u.day, flux)
        periods = np.linspace(0.5, 20, 2000)
        result  = bls.power(periods, 0.1)

        best_idx    = np.argmax(result.power)
        best_period = float(result.period[best_idx].value)
        best_power  = float(result.power[best_idx])

        # Phase fold at best period
        t0    = float(result.transit_time[best_idx].value)
        phase = ((time - t0) % best_period) / best_period
        phase[phase > 0.5] -= 1.0

        # Transit mask
        duration  = float(result.duration[best_idx].value)
        half_dur  = duration / (2 * best_period)
        in_transit = np.abs(phase) < half_dur

        if in_transit.sum() < 3:
            return _fail("Too few in-transit points")

        # Transit depth
        flux_in  = flux[in_transit].mean()
        flux_out = flux[~in_transit].mean()
        depth    = float(flux_out - flux_in)

        # SNR
        noise = flux[~in_transit].std()
        snr   = float(depth / (noise + 1e-10))

        # Count transits
        n_transits = max(1, int((time[-1] - time[0]) / best_period))

        # Shape check — flat-bottomed if std inside < std outside
        shape_ok = flux[in_transit].std() < flux[~in_transit].std()

        # Verdict
        verdict = "PASS" if (
            best_power > 0.005 and
            snr > 1.0 and
            depth > 0.001 and
            n_transits >= 2
        ) else "FAIL"

        concern = None
        if snr < 1.0:
            concern = f"Low SNR ({snr:.1f}) — may be noise"
        elif depth < 0.001:
            concern = "Very shallow depth — borderline detection"
        elif n_transits < 2:
            concern = "Only 1 transit found — cannot confirm periodicity"

        return {
            "verdict":        verdict,
            "period_days":    round(best_period, 4),
            "depth":          round(depth, 6),
            "duration_hours": round(duration * 24, 2),
            "snr":            round(snr, 2),
            "bls_power":      round(best_power, 4),
            "n_transits":     n_transits,
            "transit_shape":  "flat-bottomed" if shape_ok else "uncertain",
            "key_concern":    concern
        }

    except Exception as e:
        return _fail(f"BLS failed: {str(e)}")


def _fail(reason: str) -> dict:
    return {
        "verdict":        "FAIL",
        "period_days":    None,
        "depth":          None,
        "duration_hours": None,
        "snr":            None,
        "bls_power":      None,
        "n_transits":     None,
        "transit_shape":  "unknown",
        "key_concern":    reason
    }


if __name__ == "__main__":
    import lightkurve as lk
    print("Testing DETECT agent on Kepler-17...")

    lc = lk.search_lightcurve(
        "Kepler-17", mission="kepler", quarter=1
    )[0].download()
    lc   = lc.remove_nans().remove_outliers().normalize()
    time = lc.time.value
    flux = lc.flux.value

    report = run(time, flux)

    print(f"\nDETECT Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")