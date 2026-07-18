"""
LightSeek — CHALLENGE Agent (W4)
Hunts false positive signatures.
Author: Nagesh R / Team Integral X
"""

import numpy as np


def run(time: np.ndarray, flux: np.ndarray,
        detect_report: dict) -> dict:
    """
    CHALLENGE Agent — False positive analysis.
    Checks: secondary eclipse, even/odd depth, transit shape.
    """

    if detect_report.get("verdict") == "FAIL":
        return {
            "verdict": "SKIP",
            "secondary_eclipse_detected": None,
            "even_odd_depth_diff_pct": None,
            "centroid_shift_px": 0.0,
            "transit_shape": "unknown",
            "key_concern": "DETECT failed — no transit to challenge"
        }

    try:
        period = detect_report.get("period_days", 3.0)
        depth  = detect_report.get("depth", 0.01)

        if period is None:
            period = 3.0

        # Phase fold
        phase = (time % period) / period
        phase[phase > 0.5] -= 1.0

        half_dur   = 0.05
        in_transit = np.abs(phase) < half_dur

        # Check 1: Secondary eclipse at phase 0.5
        phase_sec = ((time % period) / period)
        phase_sec = phase_sec - 0.5
        phase_sec[phase_sec > 0.5] -= 1.0
        in_secondary = np.abs(phase_sec) < half_dur

        secondary_depth = 0.0
        if in_secondary.sum() > 2 and in_transit.sum() > 2:
            flux_out       = flux[~in_transit & ~in_secondary].mean()
            flux_sec       = flux[in_secondary].mean()
            secondary_depth = float(flux_out - flux_sec)

        secondary_detected = (
            secondary_depth > 0.5 * depth
            if depth and depth > 0 else False
        )

        # Check 2: Even vs odd transit depth
        transit_times = []
        t = time[0] + period / 2
        while t < time[-1]:
            transit_times.append(t)
            t += period

        even_depths, odd_depths = [], []
        for i, tc in enumerate(transit_times):
            mask = np.abs(time - tc) < (period * half_dur)
            if mask.sum() < 2:
                continue
            d = flux[~in_transit].mean() - flux[mask].mean()
            if i % 2 == 0:
                even_depths.append(d)
            else:
                odd_depths.append(d)

        even_mean = np.mean(even_depths) if even_depths else 0.0
        odd_mean  = np.mean(odd_depths)  if odd_depths  else 0.0
        diff_pct  = 0.0
        if even_mean > 0:
            diff_pct = abs(even_mean - odd_mean) / (even_mean + 1e-10) * 100

        # Check 3: Transit shape — V vs flat
        if in_transit.sum() > 3:
            flux_in   = flux[in_transit]
            center    = len(flux_in) // 2
            half      = max(1, len(flux_in) // 4)
            center_f  = flux_in[center-half:center+half].mean()
            edge_f    = np.concatenate([
                flux_in[:half], flux_in[-half:]
            ]).mean()
            shape = "flat-bottomed" if center_f < edge_f else "v-shaped"
        else:
            shape = "unknown"

        # Verdict
        concerns = []
        if secondary_detected:
            concerns.append("Secondary eclipse detected — likely eclipsing binary")
        if diff_pct > 20:
            concerns.append(f"Even/odd depth differ by {diff_pct:.1f}% — binary signature")
        if shape == "v-shaped":
            concerns.append("V-shaped transit — possible binary or grazing eclipse")

        verdict = "FAIL" if concerns else "PASS"
        concern = " | ".join(concerns) if concerns else None

        return {
            "verdict":                   verdict,
            "secondary_eclipse_detected": secondary_detected,
            "secondary_depth":           round(secondary_depth, 6),
            "even_odd_depth_diff_pct":   round(diff_pct, 2),
            "centroid_shift_px":         0.0,
            "transit_shape":             shape,
            "key_concern":               concern
        }

    except Exception as e:
        return {
            "verdict":                   "UNCERTAIN",
            "secondary_eclipse_detected": None,
            "even_odd_depth_diff_pct":   None,
            "centroid_shift_px":         0.0,
            "transit_shape":             "unknown",
            "key_concern":               f"Challenge failed: {str(e)}"
        }


if __name__ == "__main__":
    import lightkurve as lk
    print("Testing CHALLENGE agent on Kepler-17...")

    lc = lk.search_lightcurve(
        "Kepler-17", mission="kepler", quarter=1
    )[0].download()
    lc   = lc.remove_nans().remove_outliers().normalize()
    time = lc.time.value
    flux = lc.flux.value

    detect_mock = {
        "verdict": "PASS", "period_days": 1.486,
        "depth": 0.014, "snr": 28.4
    }
    report = run(time, flux, detect_mock)
    print(f"\nCHALLENGE Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")