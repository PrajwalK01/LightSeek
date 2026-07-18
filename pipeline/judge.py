"""
LightSeek — JUDGE Agent (W4)
Consensus verdict from all three agents.
Author: Prajwal K / Team Integral X
"""

import os
import json
import anthropic


def build_prompt(detect: dict, challenge: dict, analyze: dict) -> str:
    return f"""You are a senior exoplanet astronomer vetting a planet candidate.

Three specialist agents analyzed this transit signal:

DETECT AGENT:
- Verdict: {detect.get('verdict')}
- Period: {detect.get('period_days')} days
- Transit depth: {detect.get('depth')}
- SNR: {detect.get('snr')}
- Transits found: {detect.get('n_transits')}
- Transit shape: {detect.get('transit_shape')}
- Concern: {detect.get('key_concern')}

CHALLENGE AGENT:
- Verdict: {challenge.get('verdict')}
- Secondary eclipse: {challenge.get('secondary_eclipse_detected')}
- Even/odd depth diff: {challenge.get('even_odd_depth_diff_pct')}%
- Transit shape: {challenge.get('transit_shape')}
- Concern: {challenge.get('key_concern')}

ANALYZE AGENT:
- Verdict: {analyze.get('verdict')}
- Flares detected: {analyze.get('flares_detected')}
- Rotation period: {analyze.get('rotation_period_days')} days
- OOT scatter: {analyze.get('oot_scatter_ppm')} ppm
- Stellar quiet: {analyze.get('stellar_quiet')}
- Concern: {analyze.get('key_concern')}

Based on all three agents, provide your consensus verdict.
Respond ONLY in this exact JSON format with no extra text:
{{
  "verdict": "PLANET CANDIDATE" or "FALSE POSITIVE" or "UNCERTAIN",
  "confidence": 0.0 to 1.0,
  "reasoning": "2-3 sentence scientific explanation",
  "dominant_evidence": "which agent finding was most decisive",
  "recommended_followup": "what observation would confirm this"
}}"""


def run(detect: dict, challenge: dict, analyze: dict) -> dict:
    """JUDGE Agent — LLM consensus verdict."""

    api_key = os.getenv("ANTHROPIC_API_KEY")

    # If no API key — use rule-based fallback
    if not api_key:
        return _rule_based_judge(detect, challenge, analyze)

    try:
        client  = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": build_prompt(detect, challenge, analyze)
            }]
        )
        raw = message.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except Exception as e:
        print(f"  LLM call failed: {e} — using rule-based fallback")
        return _rule_based_judge(detect, challenge, analyze)


def _rule_based_judge(detect: dict,
                       challenge: dict,
                       analyze: dict) -> dict:
    """Fallback when no API key available."""

    d_pass = detect.get("verdict")   == "PASS"
    c_pass = challenge.get("verdict") in ("PASS", "SKIP")
    a_pass = analyze.get("verdict")  in ("PASS", "UNCERTAIN")

    passes = sum([d_pass, c_pass, a_pass])

    if passes == 3:
        verdict    = "PLANET CANDIDATE"
        confidence = 0.82
        reasoning  = (
            "All three agents passed. "
            "Periodic signal confirmed, no false positive "
            "signatures, stellar activity within normal range."
        )
    elif passes == 2:
        verdict    = "UNCERTAIN"
        confidence = 0.52
        reasoning  = (
            "Two of three agents passed. "
            "Signal shows some promise but requires "
            "further validation before confirmation."
        )
    else:
        verdict    = "FALSE POSITIVE"
        confidence = 0.75
        reasoning  = (
            "Multiple agents flagged issues. "
            "Signal is likely caused by instrumental "
            "artifact or eclipsing binary."
        )

    concerns = [
        a.get("key_concern")
        for a in [detect, challenge, analyze]
        if a.get("key_concern")
    ]

    return {
        "verdict":             verdict,
        "confidence":          confidence,
        "reasoning":           reasoning,
        "dominant_evidence":   concerns[0] if concerns else "All agents agree",
        "recommended_followup": "Radial velocity follow-up recommended"
    }


if __name__ == "__main__":
    print("Testing JUDGE agent...")

    detect_mock = {
        "verdict": "PASS", "period_days": 1.486,
        "depth": 0.014, "snr": 28.4, "n_transits": 8,
        "transit_shape": "flat-bottomed", "key_concern": None
    }
    challenge_mock = {
        "verdict": "PASS",
        "secondary_eclipse_detected": False,
        "even_odd_depth_diff_pct": 1.2,
        "transit_shape": "flat-bottomed",
        "key_concern": None
    }
    analyze_mock = {
        "verdict": "PASS",
        "flares_detected": 0,
        "rotation_period_days": 11.9,
        "oot_scatter_ppm": 145.0,
        "stellar_quiet": True,
        "key_concern": None
    }

    verdict = run(detect_mock, challenge_mock, analyze_mock)
    print(f"\nJUDGE Verdict:")
    for k, v in verdict.items():
        print(f"  {k}: {v}")