# 🌟 LightSeek
### AI-Enabled Exoplanet Detection from Noisy Astronomical Light Curves

> **ISRO Bharatiya Antariksh Hackathon (BAH) 2026**
> Challenge 7 — AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves
> Team: **Integral X** | Jnanavikas Institute of Technology, Bengaluru

---

## 🚀 What is LightSeek?

LightSeek is a **multi-agent AI vetting system** for exoplanet detection from noisy light curves produced by space telescopes like Kepler and TESS.

Unlike single-model classifiers that output a black-box probability score, LightSeek deploys **four specialized AI agents** that independently analyze each transit candidate and debate its validity — producing an explainable, confidence-ranked verdict with full scientific reasoning.

---

## 🤖 The Four Agents

| Agent | Role |
|---|---|
| **DETECT** | Validates periodicity — BLS periodogram, SNR, transit shape |
| **CHALLENGE** | Hunts false positives — eclipsing binaries, centroid shifts, secondary eclipses |
| **ANALYZE** | Examines stellar activity — flares, rotation, out-of-transit variability |
| **JUDGE** | Weighs all arguments → confidence-ranked verdict + reasoning trail |

---

## 🏗️ Architecture

```
Raw Light Curve (FITS)
        │
        ▼
 ┌─────────────────┐
 │  Preprocessing  │  ← Detrend, normalize, sigma-clip
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │ 1D CNN-Transformer│  ← Backbone detector
 │    Backbone     │
 └────────┬────────┘
          │ Transit Candidates
          ▼
 ┌────────────────────────────────────┐
 │         Multi-Agent Panel          │
 │  DETECT → CHALLENGE → ANALYZE      │
 └────────────────┬───────────────────┘
                  │ Agent Reports (JSON)
                  ▼
          ┌──────────────┐
          │    JUDGE     │  ← LLM-powered consensus
          └──────┬───────┘
                 │
                 ▼
        Vetting Report
   (verdict + confidence + reasoning)
```

---

## 👥 Team Integral X

| Name | Role |
|---|---|
| **Prajwal K** *(Leader)* | Architecture, CNN-Transformer backbone, Agent pipeline |
| **Nagesh R** | Data preprocessing, noise filtering, phase folding |
| **Thanushree B S** | Kepler data collection, training pipeline, experiments |
| **Sinchana R** | TESS data collection, evaluation metrics, dataset splits |

---

## 📁 Project Structure

```
LightSeek/
├── data/
│   ├── raw/
│   │   ├── kepler/          # Raw Kepler FITS files
│   │   └── tess/            # Raw TESS FITS files
│   └── processed/
│       ├── kepler/          # Preprocessed .npy arrays
│       └── tess/            # Preprocessed .npy arrays
├── models/
│   ├── backbone/            # CNN-Transformer model files
│   └── agents/              # Individual agent modules
├── pipeline/
│   ├── preprocess.py        # Full preprocessing pipeline
│   ├── detect.py            # DETECT agent
│   ├── challenge.py         # CHALLENGE agent
│   ├── analyze.py           # ANALYZE agent
│   └── judge.py             # JUDGE agent
├── dashboard/
│   ├── backend/             # FastAPI server
│   └── frontend/            # HTML/CSS/JS UI
├── notebooks/               # Jupyter experiments
├── experiments/             # Training run logs
├── results/
│   ├── sample_verdicts/     # Sample vetting reports
│   └── plots/               # Charts and visualizations
├── schemas/                 # JSON schemas
├── tests/                   # Unit tests
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/PrajwalK01/LightSeek.git
cd LightSeek

# Install dependencies
pip install -r requirements.txt

# Download a sample light curve and run
python pipeline/preprocess.py --target "Kepler-17" --mission kepler
```

---

## 📊 Current Status

| Component | Status |
|---|---|
| Folder structure | ✅ Done |
| Data pipeline | 🔄 In progress |
| CNN-Transformer backbone | 🔜 Week 3 |
| Multi-agent pipeline | 🔜 Week 5 |
| Web dashboard | 🔜 Week 6 |

---

## 🛠️ Tech Stack

```
Language:       Python 3.11
ML Framework:   PyTorch
Astronomy:      lightkurve, Astropy
Data:           NumPy, Pandas, SciPy
Agents:         LangChain + Claude API
Visualization:  Plotly, Matplotlib
Backend:        FastAPI
Frontend:       HTML / CSS / JavaScript
Deployment:     Railway
```

---

## 📡 Data Sources

- **Kepler DR25** — NASA Exoplanet Archive
- **TESS** — MAST Portal (via lightkurve)
- **Ground truth labels** — koi_disposition (CONFIRMED / FALSE POSITIVE)

---

*Built with ❤️ by Team Integral X for ISRO BAH 2026*
