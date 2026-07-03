"""
LightSeek — JUDGE Agent
Reads all three agent reports, weighs evidence,
and produces a final explainable verdict.
Author: Prajwal K / Team Integral X
"""

import os
import anthropic


def build_prompt(detect: dict, challenge: dict, analyze: dict) -> str:
    return f"""You are a senior exoplanet astronomer vetting a planet candidate.

Three specialist agents have analyzed this transit signal:

DETECT AGENT:
- Verdict: {detect['verdict']}
- Period: {detect['period_days']} days
- Transit depth: {detect['depth']}
- SNR: {detect['snr']}
- Number of transits: {detect['n_transits']}
- Concern: {detect['key_concern']}

CHALLENGE AGENT:
- Verdict: {challenge['verdict']}
- Secondary eclipse: {challenge['secondary_eclipse_detected']}
- Even/odd depth diff: {challenge['even_odd_depth_diff_pct']}%
- Centroid shift: {challenge['centroid_shift_px']} px
- Transit shape: {challenge['transit_shape']}
- Concern: {challenge['key_concern']}

ANALYZE AGENT:
- Verdict: {analyze['verdict']}
- Flares detected: {analyze['flares_detected']}
- Stellar rotation period: {analyze['rotation_period_days']} days
- Out-of-transit scatter: {analyze['oot_scatter_ppm']} ppm
- Stellar quiet: {analyze['stellar_quiet']}
- Concern: {analyze['key_concern']}

Based on all three agents, provide your consensus verdict.
Respond ONLY in this JSON format:
{{
  "verdict": "PLANET CANDIDATE" or "FALSE POSITIVE" or "UNCERTAIN",
  "confidence": 0.0 to 1.0,
  "reasoning": "2-3 sentence scientific explanation",
  "dominant_evidence": "which agent's finding was most decisive",
  "recommended_followup": "what observation would resolve remaining uncertainty"
}}"""


def run(detect: dict, challenge: dict, analyze: dict) -> dict:
    """
    JUDGE Agent — Consensus verdict from all three agents.

    Returns:
        dict with verdict, confidence, reasoning, followup
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = build_prompt(detect, challenge, analyze)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    import json
    raw = message.content[0].text
    # Strip markdown fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)
