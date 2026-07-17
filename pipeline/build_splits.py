"""
LightSeek — Train/Val/Test Split Builder (W2)
Stratified 70/15/15 split.
Author: Sinchana R / Team Integral X
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

MANIFEST  = "data/processed/manifest.csv"
SPLIT_DIR = "data/processed/splits"


def build_splits(random_seed: int = 42):
    df = pd.read_csv(MANIFEST)
    print(f"\nTotal samples: {len(df)}")
    print(f"Class distribution:\n{df['label_name'].value_counts()}\n")

    train_df, temp_df = train_test_split(
        df, test_size=0.30,
        stratify=df['label'], random_state=random_seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50,
        stratify=temp_df['label'], random_state=random_seed
    )

    os.makedirs(SPLIT_DIR, exist_ok=True)
    train_df.to_csv(f"{SPLIT_DIR}/train.csv", index=False)
    val_df.to_csv(f"{SPLIT_DIR}/val.csv",     index=False)
    test_df.to_csv(f"{SPLIT_DIR}/test.csv",   index=False)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Splits saved to {SPLIT_DIR}/")
    return train_df, val_df, test_df


if __name__ == "__main__":
    build_splits()