"""
LightSeek — Main Pipeline
Runs full detection + vetting for a given star.
Usage: python pipeline/run.py --target "Kepler-17" --mission kepler
Author: Team Integral X
"""

import json
import argparse
from pipeline import preprocess, detect, challenge, analyze, judge


def run_pipeline(target: str, mission: str = "kepler") -> dict:
    print(f"\n🔭 LightSeek — Analyzing {target}\n{'─'*40}")

    # Step 1: Preprocess
    time, flux = preprocess.preprocess(target, mission)

    # Step 2: DETECT
    print("🔍 DETECT agent running...")
    detect_report = detect.run(time, flux)
    print(f"   → {detect_report['verdict']}")

    # Step 3: CHALLENGE
    print("⚔️  CHALLENGE agent running...")
    challenge_report = challenge.run(time, flux, detect_report)
    print(f"   → {challenge_report['verdict']}")

    # Step 4: ANALYZE
    print("🔬 ANALYZE agent running...")
    analyze_report = analyze.run(time, flux)
    print(f"   → {analyze_report['verdict']}")

    # Step 5: JUDGE
    print("⚖️  JUDGE deliberating...")
    verdict = judge.run(detect_report, challenge_report, analyze_report)

    result = {
        "target": target,
        "mission": mission,
        "agents": {
            "detect": detect_report,
            "challenge": challenge_report,
            "analyze": analyze_report
        },
        "verdict": verdict
    }

    print(f"\n{'═'*40}")
    print(f"🌟 FINAL VERDICT: {verdict.get('verdict', 'PENDING')}")
    print(f"   Confidence: {verdict.get('confidence', 0)*100:.1f}%")
    print(f"   Reasoning: {verdict.get('reasoning', 'N/A')}")
    print(f"{'═'*40}\n")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="Kepler-17")
    parser.add_argument("--mission", default="kepler")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = run_pipeline(args.target, args.mission)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✓ Report saved to {args.output}")
