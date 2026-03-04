# 🫀 CardioSense — Real-Time AFib Detection

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat-square&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> **Research tool only — not a certified medical device. Always consult a physician for diagnosis.**

CardioSense is an ECG analysis dashboard that detects **Atrial Fibrillation (AFib)** using a hybrid **1D CNN + BiLSTM** deep learning model and a **Random Forest** fallback — both trained on the PhysioNet AF Database.

---

## ✨ Features

- 📡 **ECG Analysis** — Upload a CSV or run a demo signal (Normal or AFib)
- 🔴 **Live ECG Simulation** — Scrolling hospital-style monitor with real-time classification
- 📊 **Model Metrics** — ROC curves, confusion matrices, feature importance charts
- 🗄️ **Dataset Info** — Class distribution and imbalance correction strategy
- ⬇️ **Report Export** — Download analysis as JSON

---

## 🧠 Models

| Model         | Architecture                            | AUC                   | Sensitivity           |
| ------------- | --------------------------------------- | --------------------- | --------------------- |
| CNN+BiLSTM    | 1D Multi-scale CNN → BiLSTM → Attention | See `models/results/` | See `models/results/` |
| Random Forest | SMOTE + HRV features (15 features)      | See `models/results/` | See `models/results/` |

**Imbalance strategy (3-pronged):**

1. SMOTE oversampling on HRV feature space
2. Class-weighted loss
3. Focal Loss (γ=2, α=0.75)

---

## 🗂️ Project Structure

```
cardiosense/
├── app.py                  # Streamlit dashboard
├── serial_bridge.py        # AD8232 live ECG bridge
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Theme config
├── src/
│   ├── preprocess.py       # PhysioNet AFDB download + HRV extraction
│   ├── train.py            # CNN+BiLSTM and Random Forest training
│   ├── evaluate.py         # Full evaluation report
│   └── predict.py          # Inference module used by app.py
├── models/
│   ├── saved/              # Trained model weights
│   └── results/            # ROC curves, confusion matrices, JSON metrics
└── data/
    └── processed/          # HRV features CSV, signals.npy, labels.npy
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/cardiosense.git
cd cardiosense
pip install -r requirements.txt
```

### 2. Run the App (pre-trained models included)

```bash
streamlit run app.py
```

The demo modes work immediately without any data download.

---

## 🏋️ Train Your Own Models

### Step 1 — Download & Preprocess PhysioNet AFDB

```bash
python src/preprocess.py
```

This will automatically download the PhysioNet AF Database (~500MB) and extract HRV features + 30-second ECG segments.

### Step 2 — Train

```bash
python src/train.py              # Train both CNN and Random Forest
python src/train.py --model rf   # Random Forest only (faster, no GPU needed)
python src/train.py --model cnn  # CNN+BiLSTM only
```

### Step 3 — Evaluate

```bash
python src/evaluate.py
```

Results saved to `models/results/`.

---

## 🔌 Live AD8232 Hardware (Optional)

Connect an AD8232 ECG sensor to your computer via Arduino and run:

```bash
python serial_bridge.py --port /dev/ttyUSB0 --baud 115200   # Linux/macOS
python serial_bridge.py --port COM3 --baud 115200            # Windows
python serial_bridge.py --list                               # List available ports
```

Then open the app and select **Live Serial** in the sidebar.

**Arduino sketch** — output one ADC value per line:

```cpp
void loop() {
  Serial.println(analogRead(A0));
  delay(4);  // ~250 Hz
}
```

---

## 📊 Dataset

**PhysioNet AF Database 1.0.0**

- 25 long-term ambulatory ECG recordings
- ~10 hours each, 250 Hz, 12-bit ADC
- Classes: Normal, AFib, AFL, Junctional rhythm
- Patient-wise train/val/test split to prevent data leakage

---

## ⚠️ Disclaimer

CardioSense is a **research and educational tool**. It is **not** a certified medical device and should **not** be used for clinical diagnosis. Always consult a qualified physician for medical advice.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
