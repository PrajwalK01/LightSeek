"""
LightSeek — PyTorch Dataset (W2)
Author: Prajwal K / Team Integral X
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class LightCurveDataset(Dataset):
    def __init__(self, manifest_csv: str, n_points: int = 1000):
        self.df = pd.read_csv(manifest_csv)
        self.n_points = n_points
        print(f"Loaded: {len(self.df)} samples "
              f"({self.df['label'].sum()} confirmed / "
              f"{(self.df['label']==0).sum()} FP)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        flux = np.load(row['flux_path'])

        if len(flux) != self.n_points:
            flux = np.interp(
                np.linspace(0, 1, self.n_points),
                np.linspace(0, 1, len(flux)),
                flux
            )

        flux = (flux - flux.mean()) / (flux.std() + 1e-8)
        x = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(row['label'], dtype=torch.float32)
        return x, y


def get_dataloaders(split_dir: str = "data/processed/splits",
                     batch_size: int = 32):
    train_ds = LightCurveDataset(f"{split_dir}/train.csv")
    val_ds   = LightCurveDataset(f"{split_dir}/val.csv")
    test_ds  = LightCurveDataset(f"{split_dir}/test.csv")

    n_pos = train_ds.df['label'].sum()
    n_neg = (train_ds.df['label'] == 0).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    print(f"Class weight: {pos_weight.item():.2f}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, pos_weight


if __name__ == "__main__":
    try:
        train, val, test, pw = get_dataloaders()
        x, y = next(iter(train))
        print(f"Batch shape: {x.shape}")
        print(f"Labels: {y.tolist()[:8]}")
        print("DataLoader OK")
    except FileNotFoundError:
        print("Run download_dataset.py and build_splits.py first")