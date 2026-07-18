"""
LightSeek — Denoising Autoencoder (W2 upgrade)
Learns to reconstruct clean light curves from noisy input.
Author: Prajwal K / Team Integral X
"""

import torch
import torch.nn as nn
import numpy as np
import os


class LightCurveAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for light curve denoising.
    Encoder compresses → Decoder reconstructs clean signal.
    Input/Output: (batch, 1, 1000)
    """

    def __init__(self):
        super().__init__()

        # Encoder: 1000 → 125
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),                    # 500

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),                    # 250

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),                    # 125
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Decoder: 125 → 1000
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),        # 250

            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),        # 500

            nn.Conv1d(32, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),        # 1000

            nn.Conv1d(16, 1, kernel_size=7, padding=3),
        )

    def forward(self, x):
        encoded   = self.encoder(x)
        bottlenck = self.bottleneck(encoded)
        decoded   = self.decoder(bottlenck)
        return decoded[:, :, :x.shape[2]]


def add_noise(flux: torch.Tensor,
               noise_level: float = 0.01) -> torch.Tensor:
    """Add Gaussian noise for training."""
    return flux + torch.randn_like(flux) * noise_level


def train_autoencoder(data_dir: str = "data/processed",
                       epochs: int = 50,
                       lr: float = 1e-3):
    """Train autoencoder on existing processed data."""

    import pandas as pd
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Training autoencoder on {device}...")

    # Load all processed flux files
    manifest = pd.read_csv(f"{data_dir}/manifest.csv")
    fluxes   = []

    for _, row in manifest.iterrows():
        try:
            flux = np.load(row['flux_path'])
            flux = (flux - flux.mean()) / (flux.std() + 1e-8)
            if len(flux) == 1000:
                fluxes.append(flux)
        except Exception:
            continue

    if len(fluxes) < 5:
        print("Not enough data — need at least 5 samples")
        return None

    data    = torch.tensor(np.array(fluxes),
                            dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(data)
    loader  = DataLoader(dataset, batch_size=8, shuffle=True)

    model     = LightCurveAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    os.makedirs("models/autoencoder", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for (batch,) in loader:
            batch      = batch.to(device)
            noisy      = add_noise(batch, noise_level=0.015)
            recon      = model(noisy)
            loss       = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(),
                       "models/autoencoder/best_autoencoder.pth")

    print(f"\nAutoencoder trained. Best loss: {best_loss:.6f}")
    print("Saved: models/autoencoder/best_autoencoder.pth")
    return model


def denoise_with_autoencoder(flux: np.ndarray,
                              model_path: str = "models/autoencoder/best_autoencoder.pth"
                              ) -> np.ndarray:
    """
    Denoise a light curve using trained autoencoder.
    Falls back to input if model not found.
    """
    if not os.path.exists(model_path):
        print("  Autoencoder not trained yet — using raw flux")
        return flux

    device = torch.device("cpu")
    model  = LightCurveAutoencoder().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()

    # Normalize input
    mean = flux.mean()
    std  = flux.std() + 1e-8
    flux_norm = (flux - mean) / std

    # Resize to 1000 if needed
    if len(flux_norm) != 1000:
        flux_norm = np.interp(
            np.linspace(0, 1, 1000),
            np.linspace(0, 1, len(flux_norm)),
            flux_norm
        )

    x = torch.tensor(flux_norm, dtype=torch.float32
                     ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        denoised = model(x).squeeze().numpy()

    # Denormalize
    return denoised * std + mean


if __name__ == "__main__":
    print("Training denoising autoencoder...")
    model = train_autoencoder(epochs=50)

    if model:
        # Test denoising
        flux_noisy = np.random.normal(1.0, 0.005, 1000)
        flux_noisy[400:420] -= 0.012  # inject transit

        flux_clean = denoise_with_autoencoder(flux_noisy)
        noise_in   = np.std(flux_noisy)
        noise_out  = np.std(flux_clean)
        print(f"\nNoise before: {noise_in:.5f}")
        print(f"Noise after:  {noise_out:.5f}")
        print(f"Reduction:    {(1-noise_out/noise_in)*100:.1f}%")
        print("Autoencoder OK")