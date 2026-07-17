"""
LightSeek — Training Loop (W3)
Author: Prajwal K / Team Integral X
"""

import torch
import torch.nn as nn
import os
import sys
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from models.backbone.model import LightSeekBackbone
from pipeline.dataset import get_dataloaders


def train(epochs: int = 30,
          lr: float = 1e-3,
          batch_size: int = 16,
          run_name: str = None):

    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M")

    print(f"\n{'='*55}")
    print(f"LightSeek Training — {run_name}")
    print(f"Epochs: {epochs} | LR: {lr} | Batch: {batch_size}")
    print(f"{'='*55}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    train_loader, val_loader, _, pos_weight = get_dataloaders(
        batch_size=batch_size
    )

    model = LightSeekBackbone().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_val_f1 = 0.0
    os.makedirs("models/backbone", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    log_path = f"experiments/{run_name}.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','train_loss','val_loss',
                         'precision','recall','f1'])

    for epoch in range(1, epochs + 1):

        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x).squeeze(1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).squeeze(1)
                loss = criterion(pred, y)
                val_loss += loss.item()
                binary = (pred >= 0.5).float()
                all_preds.extend(binary.cpu().tolist())
                all_labels.extend(y.cpu().tolist())

        val_loss /= len(val_loader)

        tp = sum(p==1 and l==1 for p,l in zip(all_preds, all_labels))
        fp = sum(p==1 and l==0 for p,l in zip(all_preds, all_labels))
        fn = sum(p==0 and l==1 for p,l in zip(all_preds, all_labels))

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        scheduler.step(f1)

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}",
                             f"{val_loss:.4f}", f"{precision:.3f}",
                             f"{recall:.3f}", f"{f1:.3f}"])

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"P: {precision:.3f} R: {recall:.3f} F1: {f1:.3f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(),
                       "models/backbone/best_model.pth")
            print(f"  ✓ Saved best model (F1={f1:.3f})")

    print(f"\nTraining complete!")
    print(f"Best F1: {best_val_f1:.3f}")
    print(f"Model:   models/backbone/best_model.pth")
    print(f"Log:     {log_path}")
    return model, best_val_f1


if __name__ == "__main__":
    train(epochs=50, lr=3e-4, batch_size=8)