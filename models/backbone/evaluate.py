"""
LightSeek — Evaluation (W3)
Author: Sinchana R / Team Integral X
"""

import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.backbone.model import LightSeekBackbone
from pipeline.dataset import get_dataloaders


def evaluate(model_path: str = "models/backbone/best_model.pth",
             split: str = "test"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = LightSeekBackbone().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")

    # Data
    train_loader, val_loader, test_loader, _ = get_dataloaders(batch_size=16)
    loader = {"train": train_loader,
               "val":   val_loader,
               "test":  test_loader}[split]

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            prob = model(x).squeeze(1)
            pred = (prob >= 0.5).float()
            all_probs.extend(prob.cpu().tolist())
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(y.tolist())

    # Metrics
    tp = sum(p==1 and l==1 for p,l in zip(all_preds, all_labels))
    fp = sum(p==1 and l==0 for p,l in zip(all_preds, all_labels))
    tn = sum(p==0 and l==0 for p,l in zip(all_preds, all_labels))
    fn = sum(p==0 and l==1 for p,l in zip(all_preds, all_labels))

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (tp + tn) / (tp + fp + tn + fn + 1e-8)

    print(f"\n{'='*40}")
    print(f"Evaluation on {split} set")
    print(f"{'='*40}")
    print(f"Samples:   {len(all_labels)}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP={tp}  FP={fp}")
    print(f"  FN={fn}  TN={tn}")
    print(f"{'='*40}\n")

    return {"accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1}


if __name__ == "__main__":
    evaluate(split="val")