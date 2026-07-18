"""
LightSeek — Full Pipeline Runner (W4)
DETECT → CHALLENGE → ANALYZE → JUDGE
Author: Team Integral X
"""

import sys
import os
import json
import numpy as np
import lightkurve as lk

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from pipeline import detect, challenge, analyze, judge


def run_pipeline(target: str,
                  mission: str = "kepler",
                  quarter: int = 1) -> dict:

    print(f"\n{'='*50}")
    print(f"LightSeek — Analyzing {target}")
    print(f"{'='*50}")

    # Download + preprocess
    print(f"Downloading light curve...")
    result = lk.search_lightcurve(
        target, mission=mission, quarter=quarter
    )
    if len(result) == 0:
        return {"error": f"No light curve found for {target}"}

    lc   = result[0].download()
    lc   = lc.remove_nans().remove_outliers().normalize()
    time = lc.time.value
    flux = lc.flux.value
    flux = np.nan_to_num(flux, nan=1.0)

    # DETECT
    print(f"DETECT agent running...")
    detect_report = detect.run(time, flux)
    print(f"  → {detect_report['verdict']} "
          f"(P={detect_report.get('period_days')}d, "
          f"SNR={detect_report.get('snr')})")

    # CHALLENGE
    print(f"CHALLENGE agent running...")
    challenge_report = challenge.run(time, flux, detect_report)
    print(f"  → {challenge_report['verdict']}")

    # ANALYZE
    print(f"ANALYZE agent running...")
    analyze_report = analyze.run(time, flux)
    print(f"  → {analyze_report['verdict']} "
          f"(flares={analyze_report.get('flares_detected')})")

    # JUDGE
    print(f"JUDGE deliberating...")
    verdict = judge.run(detect_report, challenge_report, analyze_report)

    result_data = {
        "target":  target,
        "mission": mission,
        "agents": {
            "detect":    detect_report,
            "challenge": challenge_report,
            "analyze":   analyze_report
        },
        "verdict": verdict
    }

    print(f"\n{'='*50}")
    print(f"VERDICT:    {verdict.get('verdict')}")
    print(f"CONFIDENCE: {verdict.get('confidence', 0)*100:.0f}%")
    print(f"REASONING:  {verdict.get('reasoning')}")
    print(f"FOLLOWUP:   {verdict.get('recommended_followup')}")
    print(f"{'='*50}\n")

    return result_data


if __name__ == "__main__":
    # Test on confirmed planet
    print("TEST 1 — Confirmed planet (Kepler-17b)")
    r1 = run_pipeline("Kepler-17", "kepler", 1)

    # Save report
    os.makedirs("results/sample_verdicts", exist_ok=True)
    with open("results/sample_verdicts/kepler17_report.json", "w") as f:
        json.dump(r1, f, indent=2)
    print("Report saved to results/sample_verdicts/kepler17_report.json")