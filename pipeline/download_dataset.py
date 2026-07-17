"""
LightSeek — Batch Dataset Downloader (W2)
Author: Thanushree B S / Team Integral X
"""

import os
import pandas as pd
import sys
sys.path.append(os.path.dirname(__file__))
from preprocess import preprocess_and_save

CONFIRMED = [
    "Kepler-17",  "Kepler-22",  "Kepler-62",  "Kepler-69",
    "Kepler-186", "Kepler-442", "Kepler-452", "Kepler-296",
    "Kepler-438", "Kepler-440", "Kepler-61",  "Kepler-174",
    "Kepler-283", "Kepler-298", "Kepler-314", "Kepler-379",
    "Kepler-395", "Kepler-407", "Kepler-446", "Kepler-503",
]

FALSE_POSITIVES = [
    "KIC 3544595", "KIC 4544670", "KIC 5090937", "KIC 5376836",
    "KIC 5383248", "KIC 5459778", "KIC 5513490", "KIC 5652983",
    "KIC 5735762", "KIC 6021275", "KIC 6278762", "KIC 6442340",
    "KIC 6521045", "KIC 6603043", "KIC 6669809", "KIC 6851425",
    "KIC 7021681", "KIC 7199397", "KIC 7272437", "KIC 7440746",
]

CONFIRMED_DIR = "data/processed/kepler/confirmed"
FP_DIR        = "data/processed/kepler/false_positive"
MANIFEST_PATH = "data/processed/manifest.csv"


def download_all():
    records = []
    total = len(CONFIRMED) + len(FALSE_POSITIVES)
    done = 0

    print(f"\n{'='*50}")
    print(f"LightSeek Dataset Downloader — {total} targets")
    print(f"{'='*50}\n")

    print(f"[1/2] Confirmed planets ({len(CONFIRMED)})...")
    for target in CONFIRMED:
        success = preprocess_and_save(target, 1, CONFIRMED_DIR)
        if success:
            safe = target.replace(" ", "_").replace("-", "_")
            records.append({
                "target": target, "label": 1,
                "label_name": "CONFIRMED",
                "flux_path": f"{CONFIRMED_DIR}/{safe}_flux.npy",
                "time_path": f"{CONFIRMED_DIR}/{safe}_time.npy"
            })
        done += 1
        print(f"  Progress: {done}/{total}")

    print(f"\n[2/2] False positives ({len(FALSE_POSITIVES)})...")
    for target in FALSE_POSITIVES:
        success = preprocess_and_save(target, 0, FP_DIR)
        if success:
            safe = target.replace(" ", "_").replace("-", "_")
            records.append({
                "target": target, "label": 0,
                "label_name": "FALSE POSITIVE",
                "flux_path": f"{FP_DIR}/{safe}_flux.npy",
                "time_path": f"{FP_DIR}/{safe}_time.npy"
            })
        done += 1
        print(f"  Progress: {done}/{total}")

    os.makedirs("data/processed", exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(MANIFEST_PATH, index=False)

    print(f"\n{'='*50}")
    print(f"Confirmed saved:      {len(df[df.label==1])}")
    print(f"False positive saved: {len(df[df.label==0])}")
    print(f"Total:                {len(df)}")
    print(f"Manifest:             {MANIFEST_PATH}")
    print(f"{'='*50}\n")
    return df


if __name__ == "__main__":
    download_all()