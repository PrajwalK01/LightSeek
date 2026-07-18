"""
LightSeek — ANALYZE Agent (W4)
Examines stellar activity: flares, rotation, scatter.
Author: Sinchana R / Team Integral X
"""

import numpy as np
from astropy.timeseries import LombScargle


def run(time: np.ndarray, flux: np.ndarray) -> dict:
    """
    ANALYZE Agent — Stellar activity analysis.
    """
    try:
        # Flare detection — sharp upward spikes > 3 sigma
        median   = np.median(flux)
        std      = np.std(flux)
        flares   = int(np.sum(flux > median + 3.5 * std))

        # Out-of-transit scatter (ppm)
        oot_std  = float(np.std(flux) * 1e6)

        # Stellar rotation via Lomb-Scargle
        ls = LombScargle(time, flux)
        freq, power = ls.autopower(
            minimum_frequency=1/30,
            maximum_frequency=1/0.5
        )
        best_freq   = float(freq[np.argmax(power)])
        rot_period  = float(1.0 / best_freq) if best_freq > 0 else None
        ls_power    = float(np.max(power))

        # Stellar quiet flag
        stellar_quiet = (
            flares <= 5 and
            oot_std < 15000 and
            ls_power < 0.6
        )

        concerns = []
        if flares > 2:
            concerns.append(
                f"{flares} flares detected — active star"
            )
        if oot_std > 15000:
            concerns.append(
                f"High OOT scatter ({oot_std:.0f} ppm)"
            )
        if ls_power > 0.6 and rot_period:
            concerns.append(
                f"Strong rotation signal P={rot_period:.1f}d"
            )

        verdict = "FAIL" if len(concerns) > 1 else "PASS"
        concern = " | ".join(concerns) if concerns else None

        return {
            "verdict":              verdict,
            "flares_detected":      flares,
            "rotation_period_days": round(rot_period, 2) if rot_period else None,
            "ls_power":             round(ls_power, 4),
            "oot_scatter_ppm":      round(oot_std, 1),
            "stellar_quiet":        stellar_quiet,
            "key_concern":          concern
        }

    except Exception as e:
        return {
            "verdict":              "UNCERTAIN",
            "flares_detected":      None,
            "rotation_period_days": None,
            "ls_power":             None,
            "oot_scatter_ppm":      None,
            "stellar_quiet":        None,
            "key_concern":          f"Analysis failed: {str(e)}"
        }


if __name__ == "__main__":
    import lightkurve as lk
    print("Testing ANALYZE agent on Kepler-17...")

    lc = lk.search_lightcurve(
        "Kepler-17", mission="kepler", quarter=1
    )[0].download()
    lc   = lc.remove_nans().remove_outliers().normalize()
    time = lc.time.value
    flux = lc.flux.value

    report = run(time, flux)
    print(f"\nANALYZE Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")