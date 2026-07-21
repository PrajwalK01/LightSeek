<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
<img src="https://img.shields.io/badge/ISRO-BAH%202026-orange?style=for-the-badge"/>

# 🌟 LightSeek

### Multi-Agent AI System for Exoplanet Detection from Noisy Astronomical Light Curves

**[🔭 Live Demo](https://web-production-0242c.up.railway.app)** • **[📄 Paper](./docs/lightseek_paper.pdf)** • **[👥 Team](#team)**

*Submitted to ISRO Bharatiya Antariksh Hackathon (BAH) 2026 — Challenge 7*

</div>

---

## 🎯 What is LightSeek?

TESS is observing **400 million stars**. Kepler produced **4,000+ planet candidates**, thousands still awaiting human vetting. Current AI treats this as binary classification — a single probability score with no explanation.

**LightSeek is different.**

Instead of one black-box model, LightSeek deploys **four specialized AI agents** that independently analyze each transit candidate from different scientific angles and debate its validity — producing an explainable, confidence-ranked verdict with full scientific reasoning.

> *"This is the gap between AI as a detection tool and AI as a scientific collaborator. We are building the latter."*

---

## 🤖 The Four Agents

Raw Light Curve (FITS)
│
▼
┌─────────────────────┐
│ Preprocessing │ ← Wavelet denoising + autoencoder
│ + Denoising │ noise reduced by ~44%
└────────┬────────────┘
│
▼
┌─────────────────────┐
│ 1D CNN-Transformer │ ← 78,177 parameters
│ Backbone │ F1 = 0.714 | Recall = 1.0
└────────┬────────────┘
│ Transit Candidates
▼
┌─────────────────────────────────────────┐
│ Multi-Agent Panel │
│ │
│ 🔍 DETECT → Periodicity + BLS │
│ ⚔️ CHALLENGE → False positive check │
│ 🔬 ANALYZE → Stellar activity │
│ ⚖️ JUDGE → Consensus verdict │
└─────────────────────────────────────────┘
│
▼
┌─────────────────────┐
│ Vetting Report │ ← Verdict + Confidence + Reasoning
│ PLANET CANDIDATE │ + Recommended follow-up
│ Conf: 82% │
└─────────────────────┘


| Agent | Role | Key Checks |
|---|---|---|
| 🔍 **DETECT** | Transit validation | BLS periodogram, SNR, transit shape, n_transits |
| ⚔️ **CHALLENGE** | False positive hunting | Secondary eclipse, even/odd depth, centroid shift |
| 🔬 **ANALYZE** | Stellar activity | Flare detection, Lomb-Scargle rotation, OOT scatter |
| ⚖️ **JUDGE** | Consensus verdict | Weighs all agent reports → confidence-ranked decision |

---

## 📊 Results

| Metric | Value |
|---|---|
| **F1 Score** | 0.714 |
| **Recall** | 1.000 (catches every real planet) |
| **Precision** | 0.556 |
| **Noise reduction (Wavelet)** | 50% (0.0046 → 0.0023) |
| **Noise reduction (Autoencoder)** | 44% (0.00515 → 0.00297) |
| **Training samples** | 68 light curves (36 confirmed + 32 FP) |
| **Model parameters** | 78,177 |

> **Recall = 1.0** is the most critical metric in exoplanet science. A missed planet is a missed discovery. LightSeek never misses a real transit.

### Live Demo — Kepler-17b

Star: Kepler-17 (KIC 10619192)
Mission: Kepler Quarter 1

DETECT ✅ Period: 1.4852 days | Depth: 1.5% | SNR: 1.85 | Transits: 22
CHALLENGE ✅ No secondary eclipse | Shape: flat-bottomed | Even/odd diff: 0%
ANALYZE ✅ Flares: 0 | Rotation: 12.33 days | Stellar quiet: True
JUDGE ✅ PLANET CANDIDATE — Confidence: 82%
"Periodic signal confirmed, no false positive signatures,
stellar activity within normal range."
Followup: Radial velocity confirmation recommended.


---

## 🛠️ Tech Stack

Language: Python 3.11
Deep Learning: PyTorch 2.3 — 1D CNN + Transformer Encoder
Astronomy: lightkurve 2.6, Astropy 8.0
Signal Processing: PyWavelets (wavelet denoising)
Denoising Autoencoder (custom PyTorch)
Detection: BLS periodogram (astropy.timeseries)
Lomb-Scargle periodogram
Data: NumPy, Pandas, SciPy
Agent System: Custom 4-agent pipeline
DETECT → CHALLENGE → ANALYZE → JUDGE
Visualization: Chart.js, Matplotlib, Plotly
Backend: FastAPI + Uvicorn
Frontend: HTML5 / CSS3 / JavaScript
Deployment: Railway
Dataset: NASA Kepler DR25 + TESS
68 light curves | 36 confirmed | 32 false positives


---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/PrajwalK01/LightSeek.git
cd LightSeek

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline on any star
python pipeline/run.py --target "Kepler-22" --mission kepler

# Start the web dashboard
python dashboard/backend/app.py
# Open: http://localhost:8000
```

---

## 📁 Project Structure

LightSeek/
├── pipeline/
│ ├── preprocess.py # Download + clean + normalize
│ ├── wavelet_filter.py # Wavelet denoising (50% noise reduction)
│ ├── autoencoder.py # Denoising autoencoder (44% reduction)
│ ├── noise_filter.py # Sigma clip + median filter + detrend
│ ├── detect.py # DETECT agent — BLS periodogram
│ ├── challenge.py # CHALLENGE agent — false positive checks
│ ├── analyze.py # ANALYZE agent — stellar activity
│ ├── judge.py # JUDGE agent — consensus verdict
│ └── run.py # Full pipeline runner
├── models/
│ ├── backbone/
│ │ ├── model.py # CNN-Transformer architecture
│ │ ├── train.py # Training loop
│ │ ├── evaluate.py # Metrics computation
│ │ └── best_model.pth # Trained weights (F1=0.714)
│ └── autoencoder/
│ └── best_autoencoder.pth
├── dashboard/
│ ├── backend/app.py # FastAPI server
│ └── frontend/ # Web UI
├── data/
│ ├── raw/ # FITS files (not tracked)
│ └── processed/ # .npy arrays + splits
├── results/
│ ├── sample_verdicts/ # JSON vetting reports
│ └── plots/ # Visualizations
└── experiments/ # Training logs (CSV)


---

## 🔬 How It Works

### 1. Data Pipeline
Raw Kepler/TESS FITS files → detrend → normalize → wavelet denoise → autoencoder reconstruction → fixed 1000-point array

### 2. CNN-Transformer Detection

Input: (batch, 1, 1000) flux array
→ Conv1D(16, k=5) + BatchNorm + ReLU + MaxPool [500]
→ Conv1D(32, k=5) + BatchNorm + ReLU + MaxPool [250]
→ Conv1D(64, k=3) + BatchNorm + ReLU + MaxPool [125]
→ TransformerEncoder(d=64, heads=4, layers=2) [125]
→ GlobalAvgPool → Dense(32) → Dropout → Sigmoid
Output: Planet probability (0–1)


### 3. Multi-Agent Vetting
Each flagged candidate passes through four specialist agents. JUDGE reads all three reports and produces a human-readable verdict with confidence score and recommended follow-up observation.

---

## 🌟 Why LightSeek vs ExoMiner (NASA)

| Feature | ExoMiner (NASA) | LightSeek |
|---|---|---|
| Architecture | Multi-branch CNN | CNN + Transformer |
| Vetting | Inside network weights | 4 explicit agents |
| Explainability | Single probability score | Full reasoning trail |
| Auditability | ❌ Black box | ✅ Agent-by-agent report |
| Output | Score only | Verdict + confidence + reasoning + followup |
| Temporal patterns | Limited | Transformer captures long-range periodicity |

---

## 👥 Team

**Team Integral X** — Jnanavikas Institute of Technology, Bidadi, Bengaluru
*Affiliated to Visvesvaraya Technological University (VTU)*

| Name | Role |
|---|---|
| **Prajwal K** *(Leader)* | Architecture, CNN-Transformer, Agent pipeline, Deployment |
| **Nagesh R** | Data preprocessing, Noise filtering, Wavelet denoising |
| **Thanushree B S** | Data collection, Training pipeline, Experiments |
| **Sinchana R** | TESS data, Evaluation metrics, Dataset management |

---

## 📡 Dataset

- **Kepler DR25** — NASA Exoplanet Archive
- **TESS** — MAST Portal via lightkurve
- **Labels** — `koi_disposition`: CONFIRMED (label=1) / FALSE POSITIVE (label=0)
- **Split** — 70% train / 15% val / 15% test (stratified)

---

## 📜 License

MIT License — see [LICENSE](./LICENSE) for details.

---

<div align="center">

**Built with ❤️ by Team Integral X for ISRO BAH 2026**

[🔭 Try Live Demo](https://web-production-0242c.up.railway.app)

</div>

Now create LICENSE file in root:

MIT License

Copyright (c) 2026 Team Integral X — Prajwal K, Nagesh R, Thanushree B S, Sinchana R

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.