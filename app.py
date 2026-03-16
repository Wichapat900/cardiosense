"""
app.py — CardioSense Streamlit Application
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import json
from pathlib import Path
import sys

# SHAP for explainability — graceful fallback if not installed
try:
    import shap as _shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent / "src"))

st.set_page_config(
    page_title="CardioSense",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "bg":            "#050b12",
    "panel":         "#080f18",
    "panel2":        "#0c1620",
    "border":        "#1a2d3d",
    "border_light":  "#243d55",
    "text":          "#c8dde8",
    "text_mid":      "#7a9bb8",
    "text_dim":      "#3a5a78",
    "accent":        "#2ab5b5",
    "accent2":       "#1e6fa8",
    "white":         "#ffffff",
    "success":       "#1fcc7a",
    "danger":        "#f04060",
    "warn":          "#f4a124",
    "ecg_bg":        "#fff8f0",
    "ecg_grid_maj":  "rgba(210,50,50,0.30)",
    "ecg_grid_min":  "rgba(210,50,50,0.10)",
    "ecg_normal":    "#1a5fa8",
    "ecg_afib":      "#d03030",
}

SAMPLE_RATE = 250

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Sora:wght@600;700&display=swap');
  * { box-sizing: border-box; }
  .stApp { background: #050b12; font-family: 'Inter', sans-serif; color: #c8dde8; }
  .main .block-container { padding: 0 !important; max-width: 100% !important; }

  [data-testid="stSidebar"] { background: #080f18 !important; border-right: 1px solid #1a2d3d !important; }
  [data-testid="stSidebar"] * { color: #c8dde8 !important; }
  [data-testid="stSidebar"] hr { border-color: #1a2d3d !important; }
  [data-testid="stSidebar"] .stRadio label span { font-size: 0.83rem !important; color: #7a9bb8 !important; }
  [data-testid="stSidebar"] .stCheckbox label span { font-size: 0.82rem !important; color: #7a9bb8 !important; }
  [data-testid="stSidebar"] .stSelectbox label { font-size: 0.75rem !important; color: #3a5a78 !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }
  [data-testid="stSidebar"] [data-baseweb="select"] { background: #0c1620 !important; border-color: #1a2d3d !important; }
  [data-testid="stSidebar"] [data-baseweb="select"] * { background: #0c1620 !important; color: #c8dde8 !important; }

  [data-baseweb="slider"] [role="slider"] { background: #2ab5b5 !important; border-color: #2ab5b5 !important; box-shadow: 0 0 0 4px rgba(42,181,181,0.2) !important; }
  .stSlider [data-baseweb="slider"] > div > div > div:first-child { background: #1a2d3d !important; }
  .stSlider [data-baseweb="slider"] > div > div > div:nth-child(2) { background: #2ab5b5 !important; }

  .stTabs [data-baseweb="tab-list"] { background: #080f18; border-bottom: 1px solid #1a2d3d; padding: 0 1.5rem; gap: 0; }
  .stTabs [data-baseweb="tab"] { color: #7a9bb8 !important; font-family: 'Inter', sans-serif !important; font-size: 0.78rem !important; font-weight: 500 !important; letter-spacing: 0.07em !important; text-transform: uppercase !important; padding: 0.9rem 1.4rem !important; border-bottom: 2px solid transparent !important; margin-bottom: -1px !important; background: transparent !important; }
  .stTabs [aria-selected="true"] { color: #2ab5b5 !important; border-bottom: 2px solid #2ab5b5 !important; }
  .stTabs [data-baseweb="tab-panel"] { padding: 1.5rem 2rem !important; background: #050b12; }

  [data-testid="metric-container"] { background: #080f18; border: 1px solid #1a2d3d; border-radius: 10px; padding: 1rem !important; }
  [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-size: 1.55rem !important; color: #ffffff !important; font-weight: 500 !important; }
  [data-testid="stMetricLabel"] { font-size: 0.65rem !important; font-weight: 600 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; color: #3a5a78 !important; }

  .cs-alert { border-radius: 10px; padding: 1rem 1.3rem; margin: 0.6rem 0; display: flex; align-items: flex-start; gap: 0.8rem; }
  .cs-alert-afib { background: rgba(240,64,96,0.1); border: 1px solid rgba(240,64,96,0.4); border-left: 4px solid #f04060; }
  .cs-alert-normal { background: rgba(31,204,122,0.08); border: 1px solid rgba(31,204,122,0.3); border-left: 4px solid #1fcc7a; }
  .cs-alert-borderline { background: rgba(244,161,36,0.08); border: 1px solid rgba(244,161,36,0.3); border-left: 4px solid #f4a124; }

  .cs-label { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #3a5a78; margin-bottom: 0.5rem; padding-bottom: 0.3rem; border-bottom: 1px solid #1a2d3d; }
  .cs-card { background: #080f18; border: 1px solid #1a2d3d; border-radius: 12px; padding: 1.4rem; margin-bottom: 0.8rem; }
  .cs-badge { display: inline-flex; align-items: center; gap: 5px; background: #0c1620; border: 1px solid #1a2d3d; border-radius: 16px; padding: 3px 10px; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: #7a9bb8; margin: 2px 0; }

  .stDownloadButton > button { background: linear-gradient(135deg, #1e6fa8, #2ab5b5) !important; color: white !important; border: none !important; border-radius: 8px !important; font-family: 'Inter', sans-serif !important; font-weight: 600 !important; font-size: 0.8rem !important; }
  [data-testid="stFileUploader"] { background: #080f18; border: 1.5px dashed #243d55; border-radius: 10px; }
  .streamlit-expanderHeader { background: #080f18 !important; border: 1px solid #1a2d3d !important; border-radius: 8px !important; color: #c8dde8 !important; font-family: 'Inter', sans-serif !important; font-size: 0.82rem !important; }
  .streamlit-expanderContent { background: #080f18 !important; border: 1px solid #1a2d3d !important; border-top: none !important; border-radius: 0 0 8px 8px !important; }
  .stDataFrame { border: 1px solid #1a2d3d !important; border-radius: 8px !important; overflow: hidden; }
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: #050b12; }
  ::-webkit-scrollbar-thumb { background: #243d55; border-radius: 3px; }
  code { background: #0c1620 !important; color: #2ab5b5 !important; border: 1px solid #1a2d3d !important; border-radius: 4px !important; padding: 1px 5px !important; }
  pre { background: #0c1620 !important; border: 1px solid #1a2d3d !important; border-radius: 8px !important; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ─── Loaders ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    models = {}
    try:
        import joblib
        rf_path = Path("models/saved/rf_pipeline.pkl")
        if rf_path.exists():
            models["rf"] = joblib.load(rf_path)
            models["rf_loaded"] = True
        else:
            models["rf_loaded"] = False
    except Exception:
        models["rf_loaded"] = False
    return models


@st.cache_data
def load_results():
    results = {}
    for tag in ["rf", "cnn"]:
        p = Path(f"models/results/report_{tag}.json")
        if p.exists():
            with open(p) as f:
                results[tag] = json.load(f)
    return results


# ─── History Helpers ──────────────────────────────────────────────────────────

HISTORY_FILE = Path("data/history/sessions.json")

def load_history():
    """Load all past sessions from JSON file."""
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_session(record: dict):
    """Append a new session record to history."""
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        history = load_history()
        history.append(record)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
        return True
    except Exception:
        return False

def delete_history():
    """Wipe all history."""
    try:
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        return True
    except Exception:
        return False


# ─── Demo Signals ─────────────────────────────────────────────────────────────

def load_real_demo(mode="normal"):
    try:
        signals = np.load(Path("data/processed/signals.npy"))
        labels  = np.load(Path("data/processed/labels.npy"))
        target  = 1 if mode == "afib" else 0
        indices = np.where(labels == target)[0]
        if len(indices) == 0:
            return None
        rng = np.random.default_rng(0 if mode == "normal" else 99)
        idx = rng.choice(indices)
        return signals[idx].astype(np.float32)
    except Exception:
        return None


def generate_demo_signal(duration_sec=30, mode="normal"):
    t = np.linspace(0, duration_sec, duration_sec * SAMPLE_RATE)
    signal = np.zeros(len(t))
    if mode == "normal":
        rr_base = 0.833
        for i, ti in enumerate(t):
            phase = (ti % rr_base) / rr_base
            signal[i] += 0.15 * np.exp(-((phase - 0.20) * 12) ** 2)
            signal[i] -= 0.10 * np.exp(-((phase - 0.48) * 40) ** 2)
            signal[i] += 1.20 * np.exp(-((phase - 0.50) * 55) ** 2)
            signal[i] -= 0.25 * np.exp(-((phase - 0.52) * 50) ** 2)
            signal[i] += 0.30 * np.exp(-((phase - 0.72) *  7) ** 2)
    else:
        # AFib: irregular RR intervals — min 450ms (133bpm max) for clean display
        # CV ~0.25, RMSSD ~130ms — clearly above AFib thresholds
        rng2 = np.random.default_rng(3)
        pos  = 0.0
        beats = []
        while pos < duration_sec:
            # Irregular but not too fast: 450ms–1050ms range
            rr = 0.45 + rng2.random() * 0.60
            beats.append((pos, rr))
            pos += rr
        for beat_pos, rr in beats:
            bt   = t - beat_pos
            mask = (bt >= 0) & (bt < rr)
            ph   = bt[mask] / rr
            signal[mask] += 1.00 * np.exp(-((ph - 0.50) * 65) ** 2)
            signal[mask] -= 0.22 * np.exp(-((ph - 0.52) * 55) ** 2)
            signal[mask] += 0.20 * np.exp(-((ph - 0.72) *  7) ** 2)
            # f-wave fibrillatory baseline
            signal[mask] += 0.10 * np.sin(bt[mask] * 32) * np.sin(bt[mask] * 47)
    return (signal + np.random.normal(0, 0.030, len(t))).astype(np.float32)


# ─── Charts ───────────────────────────────────────────────────────────────────

def plot_ecg_clinical(signal, fs=SAMPLE_RATE, r_peaks=None, title="ECG Lead I", is_afib=False):
    orig_signal = np.array(signal)
    orig_fs     = fs

    # Downsample DISPLAY signal only — keep original for R-peak amplitude lookup
    max_pts = 1500
    step = max(1, len(orig_signal) // max_pts)
    disp_signal = orig_signal[::step]
    disp_t      = np.arange(len(disp_signal)) * step / orig_fs  # real time axis

    trace_color = COLORS["ecg_afib"] if is_afib else COLORS["ecg_normal"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=disp_t, y=disp_signal, mode="lines",
        line=dict(color=trace_color, width=1.4),
        name="ECG",
        hovertemplate="%{x:.3f}s  %{y:.3f}mV<extra></extra>",
    ))
    # R-peaks: convert original sample indices to time, use original amplitude
    if r_peaks and len(r_peaks) > 0:
        arr   = np.array(r_peaks, dtype=int)
        valid = arr[(arr >= 0) & (arr < len(orig_signal))]
        fig.add_trace(go.Scatter(
            x=valid / orig_fs,
            y=orig_signal[valid],
            mode="markers",
            marker=dict(color=COLORS["danger"], size=8, symbol="circle",
                        line=dict(color="white", width=1.5)),
            name="R peaks",
            hovertemplate="R peak @ %{x:.3f}s<extra></extra>",
        ))

    # Use Plotly's built-in grid instead of hundreds of vline/hline calls
    fig.update_layout(
        title=dict(text=title, font=dict(family="Inter", size=12, color=COLORS["text_mid"]), x=0.01),
        xaxis=dict(
            title="Time (s)", color=COLORS["text_mid"],
            gridcolor="rgba(210,50,50,0.20)", gridwidth=1,
            dtick=1.0, showgrid=True,
            minor=dict(dtick=0.2, gridcolor="rgba(210,50,50,0.08)", showgrid=True),
            tickfont=dict(family="JetBrains Mono", size=10, color=COLORS["text_mid"]),
        ),
        yaxis=dict(
            title="mV", color=COLORS["text_mid"],
            gridcolor="rgba(210,50,50,0.20)", gridwidth=1,
            dtick=0.5, showgrid=True,
            minor=dict(dtick=0.1, gridcolor="rgba(210,50,50,0.08)", showgrid=True),
            tickfont=dict(family="JetBrains Mono", size=10, color=COLORS["text_mid"]),
        ),
        plot_bgcolor=COLORS["ecg_bg"],
        paper_bgcolor=COLORS["panel"],
        legend=dict(bgcolor="rgba(15,31,53,0.8)", bordercolor=COLORS["border"],
                    borderwidth=1, font=dict(family="Inter", size=11, color=COLORS["text"])),
        height=300, margin=dict(l=55, r=15, t=40, b=45),
    )
    return fig


def plot_rr_series(rr):
    m = np.mean(rr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=rr, mode="lines+markers",
        line=dict(color=COLORS["accent"], width=1.8),
        marker=dict(color=COLORS["accent"], size=4),
        hovertemplate="Beat %{x}<br>RR: %{y:.0f}ms<extra></extra>",
    ))
    fig.add_hline(y=m, line_dash="dash", line_color=COLORS["warn"], opacity=0.6,
                  annotation_text=f"Mean: {m:.0f}ms",
                  annotation_font=dict(color=COLORS["warn"], size=10))
    fig.update_layout(
        title=dict(text="RR Interval Series", font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
        xaxis=dict(title="Beat #", color=COLORS["text_mid"], gridcolor=COLORS["border"],
                   tickfont=dict(family="JetBrains Mono", size=10)),
        yaxis=dict(title="RR (ms)", color=COLORS["text_mid"], gridcolor=COLORS["border"],
                   tickfont=dict(family="JetBrains Mono", size=10)),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        height=240, margin=dict(l=55, r=15, t=40, b=45),
    )
    return fig


def plot_poincare(rr, is_afib=False):
    if len(rr) < 4:
        return go.Figure()
    arr = np.array(rr)
    color = COLORS["danger"] if is_afib else COLORS["accent"]
    lim = [max(300, arr.min() - 50), min(2000, arr.max() + 50)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=arr[:-1], y=arr[1:], mode="markers",
        marker=dict(color=color, size=5, opacity=0.65,
                    line=dict(color="rgba(255,255,255,0.1)", width=0.5)),
        name="RRn+1 vs RRn",
        hovertemplate="RRn: %{x:.0f}ms<br>RRn+1: %{y:.0f}ms<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=lim, y=lim, mode="lines",
        line=dict(color=COLORS["border_light"], dash="dash", width=1), showlegend=False,
    ))
    fig.update_layout(
        title=dict(text="Poincaré Plot", font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
        xaxis=dict(title="RRn (ms)", color=COLORS["text_mid"], gridcolor=COLORS["border"],
                   range=lim, tickfont=dict(family="JetBrains Mono", size=10)),
        yaxis=dict(title="RRn+1 (ms)", color=COLORS["text_mid"], gridcolor=COLORS["border"],
                   range=lim, tickfont=dict(family="JetBrains Mono", size=10)),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        height=240, margin=dict(l=55, r=15, t=40, b=45),
    )
    return fig


def plot_gauge(prob):
    color = COLORS["success"] if prob < 0.35 else COLORS["warn"] if prob < 0.65 else COLORS["danger"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number=dict(suffix="%", font=dict(color=color, size=34, family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=COLORS["text_dim"],
                      tickfont=dict(color=COLORS["text_dim"], family="JetBrains Mono", size=9)),
            bar=dict(color=color, thickness=0.28),
            bgcolor=COLORS["panel2"],
            borderwidth=1, bordercolor=COLORS["border"],
            steps=[
                dict(range=[0,   35], color="rgba(31,204,122,0.07)"),
                dict(range=[35,  65], color="rgba(244,161,36,0.07)"),
                dict(range=[65, 100], color="rgba(240,64,96,0.07)"),
            ],
            threshold=dict(line=dict(color=COLORS["danger"], width=2), value=65),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=COLORS["panel"], height=190,
        margin=dict(l=15, r=15, t=15, b=10),
        font=dict(color=COLORS["text"]),
    )
    return fig


def plot_shap(shap_values: dict, base_value: float, prob: float):
    """Waterfall chart showing each feature's contribution to AFib probability."""
    # Sort by absolute SHAP value
    items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
    names  = [i[0].replace("_", " ").title() for i in items]
    values = [i[1] for i in items]

    colors = [COLORS["danger"] if v > 0 else COLORS["success"] for v in values]

    # Build cumulative waterfall
    cumulative = [base_value]
    for v in values:
        cumulative.append(cumulative[-1] + v)

    fig = go.Figure()

    # Base value line
    fig.add_hline(y=base_value, line_dash="dot",
                  line_color=COLORS["text_dim"], opacity=0.6,
                  annotation_text=f"Base: {base_value:.2f}",
                  annotation_font=dict(color=COLORS["text_dim"], size=9),
                  annotation_position="right")

    # Bars
    fig.add_trace(go.Bar(
        x=names,
        y=values,
        marker_color=colors,
        marker_line=dict(color=COLORS["panel"], width=0.5),
        text=[f"{'+' if v>0 else ''}{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=9, color=COLORS["text_mid"]),
        hovertemplate="%{x}<br>SHAP: %{y:.4f}<extra></extra>",
    ))

    # Final probability line
    fig.add_hline(y=prob, line_dash="dash",
                  line_color=COLORS["danger"] if prob >= 0.65 else COLORS["success"],
                  opacity=0.8,
                  annotation_text=f"Final: {prob:.2f}",
                  annotation_font=dict(
                      color=COLORS["danger"] if prob >= 0.65 else COLORS["success"],
                      size=10),
                  annotation_position="right")

    fig.update_layout(
        title=dict(text="SHAP Feature Contributions  (red = pushes toward AFib, green = away)",
                   font=dict(family="Inter", size=11, color=COLORS["text_mid"])),
        xaxis=dict(color=COLORS["text_mid"], tickangle=-35,
                   tickfont=dict(family="Inter", size=10)),
        yaxis=dict(title="SHAP Value", color=COLORS["text_mid"],
                   gridcolor=COLORS["border"],
                   tickfont=dict(family="JetBrains Mono", size=9)),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        height=320, margin=dict(l=55, r=80, t=50, b=90),
        showlegend=False,
    )
    return fig


def plot_roc(report):
    sens = report.get("sensitivity", 0)
    spec = report.get("specificity", 0)
    fpr  = 1 - spec
    auc  = report.get("roc_auc", 0)
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color=COLORS["border_light"], dash="dash", width=1.5))
    fig.add_trace(go.Scatter(
        x=[0, fpr, 1], y=[0, sens, 1],
        mode="lines+markers",
        line=dict(color=COLORS["accent"], width=2.5),
        marker=dict(size=8, color=["rgba(0,0,0,0)", COLORS["danger"], "rgba(0,0,0,0)"],
                    line=dict(color=COLORS["accent"], width=2)),
        name=f"AUC = {auc:.3f}",
        fill="tozeroy", fillcolor="rgba(42,181,181,0.07)",
    ))
    fig.update_layout(
        title=dict(text=f"ROC Curve  (AUC = {auc:.3f})",
                   font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
        xaxis=dict(title="False Positive Rate", color=COLORS["text_mid"],
                   gridcolor=COLORS["border"], range=[0, 1],
                   tickfont=dict(family="JetBrains Mono", size=10)),
        yaxis=dict(title="True Positive Rate", color=COLORS["text_mid"],
                   gridcolor=COLORS["border"], range=[0, 1],
                   tickfont=dict(family="JetBrains Mono", size=10)),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        legend=dict(font=dict(family="Inter", size=11, color=COLORS["text"])),
        height=280, margin=dict(l=55, r=15, t=45, b=45),
    )
    return fig


# ─── Inference ────────────────────────────────────────────────────────────────

def run_inference(signal, fs=SAMPLE_RATE):
    try:
        from predict import predict
        return predict(signal, fs)
    except Exception as e:
        st.error(f"⚠️ predict.py failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        from scipy.signal import find_peaks, butter, filtfilt
        nyq = fs / 2
        b, a = butter(4, [0.5 / nyq, 40.0 / nyq], btype="band")
        sig = filtfilt(b, a, signal)
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
        if np.abs(np.min(sig)) > np.abs(np.max(sig)):
            sig = -sig
        sig_max = float(np.max(sig))
        # AFib can have very short RR intervals (down to ~300ms = 200bpm)
        # Use 320ms min distance and relative prominence
        thr = max(0.35, sig_max * 0.35)
        peaks, _ = find_peaks(
            sig,
            height=thr,
            distance=int(0.32 * fs),    # min 320ms (~188 bpm) — handles fast AFib
            prominence=sig_max * 0.28,  # 28% of max — catches weaker beats
            wlen=int(0.6 * fs),         # 600ms window
        )
        # Fallback if too few
        if len(peaks) < 3:
            thr2 = max(0.25, sig_max * 0.25)
            peaks, _ = find_peaks(sig, height=thr2, distance=int(0.32 * fs),
                                  prominence=sig_max * 0.20)
        rr = np.diff(peaks) / fs * 1000 if len(peaks) > 1 else np.array([])
        rr = rr[(rr > 300) & (rr < 2000)] if len(rr) > 0 else np.array([])
        mean_rr = float(np.mean(rr)) if len(rr) > 0 else 833.0
        sdnn    = float(np.std(rr))   if len(rr) > 1 else 0.0
        rmssd   = float(np.sqrt(np.mean(np.diff(rr)**2))) if len(rr) > 2 else 0.0
        cv      = sdnn / mean_rr if mean_rr > 0 else 0.0
        hr      = 60000 / mean_rr if mean_rr > 0 else 0.0

        # Heuristic thresholds (from literature):
        # Normal sinus: CV ~0.02-0.05, RMSSD ~15-40ms
        # AFib:         CV ~0.18-0.40, RMSSD ~100-250ms
        cv_score    = min(1.0, cv / 0.18)          # saturates at CV=0.18 (clear AFib)
        rmssd_score = min(1.0, rmssd / 80.0)       # saturates at RMSSD=80ms (clear AFib)
        # Tachycardia boost — AFib often presents with fast rate (HR>100)
        tachy_boost = 0.15 if hr > 100 else 0.0
        prob = min(0.97, max(0.03, cv_score * 0.55 + rmssd_score * 0.35 + tachy_boost))

        cls = "Normal" if prob < 0.35 else ("Borderline" if prob < 0.65 else "AFib")
        return {
            "afib_probability": round(prob, 4), "classification": cls,
            "confidence": "Low", "heart_rate": round(hr, 1),
            "hrv_features": {"mean_rr": mean_rr, "sdnn": sdnn, "rmssd": rmssd,
                             "cv": cv, "pnn50": 0.0, "mean_hr": hr,
                             "sd1": 0.0, "sd2": 0.0, "n_beats": float(len(peaks))},
            "rr_intervals": rr.tolist(), "r_peaks": peaks.tolist(),
            "n_beats": len(peaks), "signal_quality": "Unknown",
            "signal": sig.tolist(),
        }


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style='padding:1rem 0 0.8rem;'>
          <div style='font-size:1.8rem; margin-bottom:6px;'>🫀</div>
          <div style='font-family:"Sora",sans-serif; font-size:1.3rem; color:white; font-weight:700; line-height:1;'>
            CardioSense
          </div>
          <div style='font-family:"JetBrains Mono",monospace; font-size:0.55rem; color:{COLORS["text_dim"]}; letter-spacing:0.12em; margin-top:4px;'>
            AFIB DETECTION v1.0
          </div>
          <div style='font-size:0.7rem; color:{COLORS["text_mid"]}; margin-top:6px; line-height:1.5;'>
            Real-Time AFib Detection<br>Live ECG Monitoring
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="cs-label">Input Mode</div>', unsafe_allow_html=True)

        # FIX 1: Added non-empty label + label_visibility="collapsed"
        mode = st.radio(
            "Input Mode",
            ["Upload ECG File", "Demo — Normal", "Demo — AFib", "Live Serial"],
            label_visibility="collapsed"
        )

        st.divider()
        st.markdown('<div class="cs-label">Settings</div>', unsafe_allow_html=True)
        fs_opt    = st.selectbox("Sample Rate (Hz)", [250, 360, 500], index=0)
        threshold = st.slider("AFib Threshold", 0.30, 0.80, 0.65, 0.05,
                              help="Probability above which AFib is flagged.")
        show_peaks    = st.checkbox("Show R-peaks",       value=True)
        show_rr       = st.checkbox("Show RR series",     value=True)
        show_poincare = st.checkbox("Show Poincaré plot", value=True)

        st.divider()
        st.markdown('<div class="cs-label">Model Status</div>', unsafe_allow_html=True)
        rf_ok = Path("models/saved/rf_pipeline.pkl").exists()
        st.markdown(
            f'<div class="cs-badge">{"🟢" if rf_ok else "🔴"} Random Forest (Explainable)</div>',
            unsafe_allow_html=True
        )
        if not rf_ok:
            st.warning("No model found.\n\n`python src/train.py`")

        st.divider()
        st.markdown(f"""
        <div style='font-size:0.6rem; color:{COLORS["text_dim"]}; line-height:1.8;'>
          ⚠️ Research tool only.<br>Not a certified medical device.<br>Consult a physician for diagnosis.
        </div>""", unsafe_allow_html=True)

        return mode, fs_opt, threshold, show_peaks, show_rr, show_poincare


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    mode, fs, threshold, show_peaks, show_rr, show_poincare = sidebar()

    st.markdown(f"""
    <div style='background:{COLORS["panel"]}; border-bottom:1px solid {COLORS["border"]};
                padding:0.85rem 2rem; display:flex; align-items:center; justify-content:space-between;'>
      <div style='display:flex; align-items:center; gap:12px;'>
        <span style='font-size:1.6rem;'>🫀</span>
        <div>
          <span style='font-family:"Sora",sans-serif; font-size:1.25rem; color:white; font-weight:700;'>
            ECG Monitor
          </span>
          <span style='font-family:"Inter",sans-serif; font-size:0.72rem; color:{COLORS["text_dim"]};
                       margin-left:10px; letter-spacing:0.08em; text-transform:uppercase;'>
            AFib Detection Dashboard
          </span>
        </div>
      </div>
      <div style='font-family:"JetBrains Mono",monospace; font-size:0.7rem; color:{COLORS["text_dim"]};'>
        {time.strftime("%d %b %Y  %H:%M:%S")}
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab_live, tab2, tab3, tab_hist = st.tabs(["📡  Analysis", "🔴  Live Demo", "📊  Model Metrics", "🗄️  Dataset", "📈  History"])

    # ══ TAB 1 — ANALYSIS ══════════════════════════════════════════════════════
    with tab1:
        signal = None

        if mode == "Upload ECG File":
            uploaded = st.file_uploader("Upload ECG CSV (one column, 250 Hz)", type=["csv", "txt"])
            if uploaded:
                try:
                    df = pd.read_csv(uploaded, header=None)
                    col = st.selectbox("Select ECG column", df.columns,
                                       format_func=lambda x: f"Column {x}")
                    signal = df[col].dropna().values.astype(np.float32)
                    st.success(f"✓ {len(signal):,} samples — {len(signal)/fs:.1f}s @ {fs} Hz")
                except Exception as e:
                    st.error(f"Failed to parse: {e}")

        elif mode in ("Demo — Normal", "Demo — AFib"):
            demo_mode = "normal" if "Normal" in mode else "afib"
            st.session_state["demo_mode"] = demo_mode
            with st.spinner("Loading PhysioNet ECG segment..."):
                signal = load_real_demo(demo_mode)
                if signal is None:
                    signal = generate_demo_signal(30, demo_mode)
                    st.info(f"Using synthetic {demo_mode.upper()} ECG (30s @ {fs} Hz)")
                else:
                    st.info(f"Using real PhysioNet **{demo_mode.upper()}** ECG segment (30s @ {fs} Hz)")

        elif mode == "Live Serial":
            st.markdown(f"""
            <div class='cs-card'>
              <div style='font-weight:600; color:{COLORS["text"]}; margin-bottom:6px;'>Connect your AD8232</div>
              <div style='font-size:0.82rem; color:{COLORS["text_mid"]};'>Run the serial bridge in a separate terminal, then refresh.</div>
            </div>""", unsafe_allow_html=True)
            st.code("python serial_bridge.py --port /dev/ttyUSB0 --baud 115200")

        if signal is not None and len(signal) > fs * 5:
            result   = run_inference(signal, fs)


            prob     = result["afib_probability"]
            cls      = result["classification"]
            is_afib  = prob >= threshold
            hrv      = result["hrv_features"]
            rr       = result["rr_intervals"]
            peaks    = result["r_peaks"]
            disp_sig = np.array(result.get("signal", signal))

            # Alert banner — also flag tachycardia (HR>100) as a warning
            is_tachy = result["heart_rate"] > 100
            if is_afib or cls == "AFib" or (is_tachy and cls in ("AFib", "Borderline")):
                st.markdown(f"""
                <div class='cs-alert cs-alert-afib'>
                  <span style='font-size:1.5rem; flex-shrink:0;'>⚠️</span>
                  <div>
                    <div style='font-weight:700; font-size:0.95rem; color:{COLORS["danger"]}; font-family:"Sora",sans-serif;'>
                      Atrial Fibrillation Detected
                    </div>
                    <div style='font-size:0.78rem; color:{COLORS["text_mid"]}; margin-top:3px;'>
                      AFib probability: <strong>{prob*100:.1f}%</strong> — above threshold ({threshold*100:.0f}%). Consult a physician immediately.
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
            elif cls == "Borderline":
                st.markdown(f"""
                <div class='cs-alert cs-alert-borderline'>
                  <span style='font-size:1.5rem; flex-shrink:0;'>⚡</span>
                  <div>
                    <div style='font-weight:700; font-size:0.95rem; color:{COLORS["warn"]}; font-family:"Sora",sans-serif;'>
                      Borderline — Continue Monitoring
                    </div>
                    <div style='font-size:0.78rem; color:{COLORS["text_mid"]}; margin-top:3px;'>
                      AFib probability: <strong>{prob*100:.1f}%</strong> — Consult a physician if symptoms develop.
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='cs-alert cs-alert-normal'>
                  <span style='font-size:1.5rem; flex-shrink:0;'>✅</span>
                  <div>
                    <div style='font-weight:700; font-size:0.95rem; color:{COLORS["success"]}; font-family:"Sora",sans-serif;'>
                      Normal Sinus Rhythm
                    </div>
                    <div style='font-size:0.78rem; color:{COLORS["text_mid"]}; margin-top:3px;'>
                      AFib probability: <strong>{prob*100:.1f}%</strong> — No atrial fibrillation detected.
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

            # Metrics
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("AFib Probability", f"{prob*100:.1f}%",
                      delta="HIGH ⚠" if is_afib else "Normal",
                      delta_color="inverse" if is_afib else "normal")
            _hr = result["heart_rate"]
            _is_afib_result = cls in ("AFib", "Borderline")
            if _hr > 100:
                _hr_label = "⚠ Tachycardia"
                _hr_color = "inverse"
            elif _hr < 60:
                _hr_label = "⚠ Bradycardia"
                _hr_color = "inverse"
            elif _is_afib_result:
                _hr_label = "⚠ Controlled"  # AFib but rate is normal — still warrants attention
                _hr_color = "inverse"
            else:
                _hr_label = "Normal"
                _hr_color = "normal"
            m2.metric("Heart Rate", f"{_hr:.0f} bpm",
                      delta=_hr_label, delta_color=_hr_color)
            m3.metric("RMSSD", f"{hrv.get('rmssd', 0):.1f} ms",
                      help="Root Mean Square Successive Differences — elevated in AFib")
            m4.metric("SDNN",  f"{hrv.get('sdnn', 0):.1f} ms")
            m5.metric("Signal Quality", result.get("signal_quality", "—"))

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            # ECG + gauge
            ecg_col, gauge_col = st.columns([3, 1])
            with ecg_col:
                fig_ecg = plot_ecg_clinical(
                    disp_sig[:fs*15], fs,
                    peaks if show_peaks else None,
                    title="ECG Lead I  ·  First 15 seconds",
                    is_afib=is_afib,
                )
                # FIX 2: replaced use_container_width with width
                st.plotly_chart(fig_ecg, width="stretch")
            with gauge_col:
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                st.plotly_chart(plot_gauge(prob), width="stretch")
                lbl_c = COLORS["danger"] if is_afib else (COLORS["warn"] if cls == "Borderline" else COLORS["success"])
                st.markdown(
                    f"<div style='text-align:center; font-family:Inter; font-size:0.72rem;"
                    f" color:{lbl_c}; font-weight:700; margin-top:-10px;'>{cls.upper()}</div>",
                    unsafe_allow_html=True
                )

            # HRV charts
            if len(rr) > 3:
                cols = st.columns(2 if show_poincare else 1)
                if show_rr:
                    with cols[0]:
                        st.plotly_chart(plot_rr_series(rr), width="stretch")
                if show_poincare and len(cols) > 1:
                    with cols[1]:
                        st.plotly_chart(plot_poincare(rr, is_afib), width="stretch")

            # ── SHAP Explainability ───────────────────────────────────────────
            shap_vals = result.get("shap_values", {})
            shap_base = result.get("shap_base_value", 0.5)
            if shap_vals and any(v != 0 for v in shap_vals.values()):
                with st.expander("🔍  Why did the AI decide this?  (SHAP Explanation)", expanded=True):
                    st.markdown(f"""
                    <div style='font-size:0.8rem; color:{COLORS["text_mid"]}; margin-bottom:0.8rem; line-height:1.6;'>
                      Each bar shows how much a feature <strong style='color:{COLORS["danger"]}'>pushed toward AFib</strong>
                      or <strong style='color:{COLORS["success"]}'>pushed away from AFib</strong>.
                      The final probability is the base value plus all contributions.
                    </div>""", unsafe_allow_html=True)
                    st.plotly_chart(plot_shap(shap_vals, shap_base, prob), width="stretch")

                    # Top 3 reasons in plain English
                    top3 = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                    st.markdown(f'<div class="cs-label">Top Reasons for This Decision</div>',
                                unsafe_allow_html=True)
                    for feat, val in top3:
                        feat_val = result["hrv_features"].get(feat, 0)
                        direction = "elevated" if val > 0 else "low"
                        impact    = "increases" if val > 0 else "decreases"
                        color     = COLORS["danger"] if val > 0 else COLORS["success"]
                        st.markdown(
                            f"<div style='font-size:0.82rem; color:{COLORS['text_mid']};"
                            f" padding:6px 0; border-bottom:1px solid {COLORS['border']};'>"
                            f"  <span style='color:{color}; font-weight:700;'>"
                            f"{feat.replace('_',' ').title()}</span>"
                            f"  = <span style='font-family:JetBrains Mono; color:{COLORS['text']};'>"
                            f"{feat_val:.2f}</span>"
                            f"  — {direction} value {impact} AFib probability by "
                            f"  <span style='color:{color}; font-family:JetBrains Mono;'>"
                            f"{abs(val):.3f}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

            # HRV table
            with st.expander("📋  Full HRV Feature Report", expanded=False):
                desc_map = {
                    "mean_rr":             "Mean RR interval (ms)",
                    "sdnn":                "Standard deviation of RR intervals",
                    "rmssd":               "Root Mean Square Successive Differences — key AFib indicator",
                    "pnn50":               "Proportion of successive differences > 50ms",
                    "cv":                  "Coefficient of variation of RR",
                    "mean_hr":             "Mean heart rate (bpm)",
                    "sd1":                 "Poincare SD1 — short-term HRV",
                    "sd2":                 "Poincare SD2 — long-term HRV",
                    "sd1_sd2_ratio":       "SD1/SD2 ratio",
                    "sample_entropy":      "Sample entropy — unpredictability",
                    "approx_entropy":      "Approximate entropy",
                    "turning_point_ratio": "Fraction of turning points",
                    "dominant_freq":       "Dominant frequency in RR spectrum",
                    "spectral_entropy":    "Spectral entropy — flatness of spectrum",
                    "p_wave_absence":      "P-wave absence score — key AFib morphology marker",
                    "qrs_width":           "QRS complex width (ms)",
                    "t_wave_ratio":        "T-wave energy relative to QRS",
                    "rr_skewness":         "Skewness of RR distribution",
                    "rr_kurtosis":         "Kurtosis of RR distribution",
                    "rr_range":            "Range of RR intervals (max - min)",
                }
                fd = {
                    "Feature":     list(hrv.keys()),
                    "Value":       [f"{v:.4f}" for v in hrv.values()],
                    "Description": [desc_map.get(k, "") for k in hrv.keys()],
                }
                st.dataframe(pd.DataFrame(fd), use_container_width=True, hide_index=True)

            # Download
            rep = {
                "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
                "classification":   cls,
                "afib_probability": prob,
                "threshold_used":   threshold,
                "heart_rate":       result["heart_rate"],
                "n_beats":          result["n_beats"],
                "signal_quality":   result["signal_quality"],
                "hrv_features":     hrv,
                "source":           mode,
            }

            # Auto-save to history
            _saved = save_session(rep)
            if _saved:
                st.caption("✓ Session saved to History")

            st.download_button(
                "⬇  Download Analysis Report (JSON)",
                data=json.dumps(rep, indent=2),
                file_name=f"cardiosense_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

        elif signal is not None:
            st.warning(f"Signal too short ({len(signal)/fs:.1f}s). Need ≥5 seconds.")
        else:
            st.markdown(f"""
            <div style='padding:80px 20px; text-align:center;'>
              <div style='font-size:3rem; margin-bottom:12px;'>🫀</div>
              <div style='font-family:"Inter",sans-serif; font-size:0.9rem;
                          color:{COLORS["text_dim"]}; letter-spacing:0.06em;'>
                Select an input mode in the sidebar to begin
              </div>
            </div>""", unsafe_allow_html=True)


    # ══ TAB LIVE — LIVE ECG DEMO ═══════════════════════════════════════════════
    with tab_live:
        st.markdown("""
        <div style='margin-bottom:0.8rem;'>
          <span style='font-family:"Sora",sans-serif; font-size:1.1rem; color:white; font-weight:700;'>
            🔴 Live ECG Simulation
          </span>
          <span style='font-size:0.78rem; color:#7a9bb8; margin-left:12px;'>
            Smooth scrolling — hospital monitor style
          </span>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 1, 1])
        live_mode  = c1.selectbox("Rhythm",     ["Normal Sinus", "AFib"], key="lmode")
        speed      = c2.selectbox("Speed",      ["0.5x", "1x", "2x"],    key="lspeed", index=1)
        window_sec = c3.selectbox("Window (s)", [5, 8, 10],               key="lwin",   index=1)

        for _k, _v in [("lrun", False), ("lbuf", []), ("lresult", None), ("lctr", 0)]:
            if _k not in st.session_state:
                st.session_state[_k] = _v

        bc1, bc2, bc3, _ = st.columns([1, 1, 1, 3])
        if bc1.button("▶  Start", key="lstart", use_container_width=True):
            st.session_state.lrun = True
        if bc2.button("⏹  Stop",  key="lstop",  use_container_width=True):
            st.session_state.lrun = False
        if bc3.button("↺  Reset", key="lreset", use_container_width=True):
            st.session_state.lrun    = False
            st.session_state.lbuf   = []
            st.session_state.lresult = None
            st.session_state.lctr    = 0
            st.rerun()

        FS      = SAMPLE_RATE
        WIN_N   = window_sec * FS           # samples in display window
        spd_map = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}
        spd     = spd_map[speed]
        # Chunk = samples added per frame (~30ms real time at 1x)
        CHUNK   = max(4, int(FS * 0.030 * spd))

        # ── Continuous signal generator (sample-accurate) ─────────────────────
        def _sample(idx, mode):
            t = idx / FS
            if mode == "Normal Sinus":
                rr  = 0.833
                ph  = (t % rr) / rr
                v   =  1.20 * np.exp(-((ph - 0.50) * 55) ** 2)
                v  += -0.25 * np.exp(-((ph - 0.52) * 50) ** 2)
                v  +=  0.15 * np.exp(-((ph - 0.20) * 12) ** 2)
                v  += -0.10 * np.exp(-((ph - 0.48) * 40) ** 2)
                v  +=  0.30 * np.exp(-((ph - 0.72) *  7) ** 2)
            else:
                rr  = max(0.45, 0.67 + 0.22 * np.sin(t * 1.3) + 0.18 * np.sin(t * 2.9))
                ph  = (t % rr) / rr
                v   =  1.20 * np.exp(-((ph - 0.50) * 55) ** 2)
                v  += -0.25 * np.exp(-((ph - 0.52) * 50) ** 2)
                v  +=  0.30 * np.exp(-((ph - 0.72) *  7) ** 2)
                v  +=  0.09 * np.sin(t * 37) * np.sin(t * 53)
            return float(v)

        # Extend buffer with new chunk
        buf = st.session_state.lbuf
        if st.session_state.lrun or len(buf) == 0:
            start_idx = len(buf)
            new_samples = [_sample(start_idx + i, live_mode) for i in range(CHUNK)]
            buf.extend(new_samples)
            # Keep buffer to 3× window max — discard oldest
            if len(buf) > WIN_N * 3:
                buf = buf[-WIN_N * 3:]
            st.session_state.lbuf = buf

        # Sliding window — always show last WIN_N samples
        win = np.array(buf[-WIN_N:] if len(buf) >= WIN_N else buf, dtype=np.float32)
        win_norm = (win - np.mean(win)) / (np.std(win) + 1e-8)
        if len(win_norm) > 0 and np.abs(win_norm.min()) > np.abs(win_norm.max()):
            win_norm = -win_norm

        # Time axis always 0 → window_sec (fixed window, signal slides left)
        t_disp = np.linspace(0, window_sec, len(win_norm))

        # R-peak detection on current window
        from scipy.signal import find_peaks as _fp
        live_peaks = np.array([], dtype=int)
        if len(win_norm) > FS:
            sig_max = float(np.max(win_norm))
            thr_pk  = max(0.35, sig_max * 0.35)
            live_peaks, _ = _fp(win_norm, height=thr_pk,
                                distance=int(0.32 * FS),
                                prominence=sig_max * 0.28,
                                wlen=int(0.6 * FS))

        # Re-classify every 5s worth of new samples
        st.session_state.lctr += CHUNK
        if st.session_state.lctr >= FS * 5 or st.session_state.lresult is None:
            if len(win_norm) >= FS * 3:
                st.session_state.lresult = run_inference(win_norm, FS)
                st.session_state.lctr    = 0

        result  = st.session_state.lresult
        prob    = result["afib_probability"] if result else 0.0
        cls     = result["classification"]   if result else "Collecting…"
        is_afib = prob >= 0.65

        # Status banner
        if not result:
            st.markdown("<div class='cs-alert' style='background:rgba(42,181,181,0.07);border:1px solid rgba(42,181,181,0.3);border-left:4px solid #2ab5b5;margin-bottom:0.5rem;'><span>⏳</span>&nbsp; Collecting data…</div>", unsafe_allow_html=True)
        elif is_afib:
            st.markdown(f"<div class='cs-alert cs-alert-afib' style='margin-bottom:0.5rem;'><span style='font-size:1.3rem;'>⚠️</span>&nbsp;<strong style='color:#f04060;'>AFib Detected — {prob*100:.1f}%</strong>&nbsp; re-classifies every 5s</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='cs-alert cs-alert-normal' style='margin-bottom:0.5rem;'><span style='font-size:1.3rem;'>✅</span>&nbsp;<strong style='color:#1fcc7a;'>Normal Sinus — {prob*100:.1f}%</strong>&nbsp; re-classifies every 5s</div>", unsafe_allow_html=True)

        # Build chart — use st.empty() placeholder so it updates in-place
        if "lecg_placeholder" not in st.session_state:
            st.session_state.lecg_placeholder = None

        tc     = "#d03030" if is_afib else "#1a5fa8"
        sig_lo = -2.8
        sig_hi =  2.8

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t_disp, y=win_norm, mode="lines",
            line=dict(color=tc, width=1.5),
            name="ECG", hoverinfo="skip",
        ))
        if len(live_peaks) > 0:
            fig.add_trace(go.Scatter(
                x=t_disp[live_peaks], y=win_norm[live_peaks], mode="markers",
                marker=dict(color="#f04060", size=7, symbol="circle",
                            line=dict(color="white", width=1.5)),
                name="R peaks", hoverinfo="skip",
            ))
        fig.update_layout(
            title=dict(
                text=f"ECG Monitor  ·  {live_mode}  ·  HR≈{result['heart_rate']:.0f}bpm  ·  {cls}" if result else f"ECG Monitor  ·  {live_mode}",
                font=dict(family="Inter", size=12, color="#7a9bb8"), x=0.01,
            ),
            xaxis=dict(
                title="Time (s)", color="#7a9bb8",
                gridcolor="rgba(210,50,50,0.20)", gridwidth=1,
                dtick=1.0, showgrid=True,
                minor=dict(dtick=0.2, gridcolor="rgba(210,50,50,0.08)", showgrid=True),
                range=[0, window_sec], fixedrange=True,
                tickfont=dict(family="JetBrains Mono", size=10, color="#7a9bb8"),
            ),
            yaxis=dict(
                title="mV", color="#7a9bb8",
                gridcolor="rgba(210,50,50,0.20)", gridwidth=1,
                dtick=1.0, showgrid=True,
                range=[sig_lo, sig_hi], fixedrange=True,
                tickfont=dict(family="JetBrains Mono", size=10, color="#7a9bb8"),
            ),
            plot_bgcolor="#fff8f0",
            paper_bgcolor="#080f18",
            legend=dict(bgcolor="rgba(15,31,53,0.85)", bordercolor="#1a2d3d",
                        borderwidth=1, font=dict(family="Inter", size=11, color="#c8dde8")),
            height=320, margin=dict(l=55, r=15, t=40, b=45),
            uirevision="ecg-live",   # keeps zoom/pan state between reruns
        )

        ecg_slot = st.empty()
        ecg_slot.plotly_chart(fig, width="stretch")

        hrv = result.get("hrv_features", {}) if result else {}
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("AFib Probability", f"{prob*100:.1f}%",
                  delta="HIGH ⚠" if is_afib else "Normal",
                  delta_color="inverse" if is_afib else "normal")
        m2.metric("Heart Rate",     f"{result['heart_rate']:.0f} bpm" if result else "—")
        m3.metric("RMSSD",          f"{hrv.get('rmssd',0):.0f} ms"    if hrv    else "—")
        m4.metric("Beats Detected", str(len(live_peaks)))
        m5.metric("Classification", cls)

        if st.session_state.lrun:
            time.sleep(max(0.04, 0.08 / spd))   # ~12-25fps target
            st.rerun()
        elif len(buf) == 0:
            st.markdown("<div style='text-align:center;padding:10px 0 0;font-size:0.78rem;color:#3a5a78;'>Press ▶ Start · switch Rhythm anytime while running</div>", unsafe_allow_html=True)


    # ══ TAB 2 — MODEL METRICS ═════════════════════════════════════════════════
    with tab2:
        results = load_results()
        if not results:
            st.info("No results found. Run `python src/evaluate.py`")
        else:
            for tag, r in results.items():
                model_name = r.get("model", tag.upper())
                st.markdown(f"### {model_name}")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("AUC-ROC",       f"{r.get('roc_auc', 0):.4f}")
                c2.metric("Sensitivity",   f"{r.get('sensitivity', 0):.4f}",
                          help="AFib recall — critical for clinical safety")
                c3.metric("Specificity",   f"{r.get('specificity', 0):.4f}")
                c4.metric("F1 Score",      f"{r.get('f1', 0):.4f}")
                c5.metric("Avg Precision", f"{r.get('avg_precision', 0):.4f}")

                col_roc, col_cm = st.columns(2)
                with col_roc:
                    st.plotly_chart(plot_roc(r), width="stretch")
                with col_cm:
                    tp = r.get("tp", 0); fp = r.get("fp", 0)
                    fn = r.get("fn", 0); tn = r.get("tn", 0)
                    cmf = go.Figure(go.Heatmap(
                        z=[[tn, fp], [fn, tp]],
                        x=["Pred Normal", "Pred AFib"],
                        y=["Actual Normal", "Actual AFib"],
                        colorscale=[[0, COLORS["panel2"]], [1, COLORS["accent2"]]],
                        showscale=False,
                    ))
                    cmf.update_layout(
                        annotations=[
                            dict(x="Pred Normal", y="Actual Normal", text=str(tn),
                                 font=dict(size=16, color=COLORS["white"],  family="JetBrains Mono"), showarrow=False),
                            dict(x="Pred AFib",   y="Actual Normal", text=str(fp),
                                 font=dict(size=16, color=COLORS["danger"], family="JetBrains Mono"), showarrow=False),
                            dict(x="Pred Normal", y="Actual AFib",   text=str(fn),
                                 font=dict(size=16, color=COLORS["danger"], family="JetBrains Mono"), showarrow=False),
                            dict(x="Pred AFib",   y="Actual AFib",   text=str(tp),
                                 font=dict(size=16, color=COLORS["white"],  family="JetBrains Mono"), showarrow=False),
                        ],
                        title=dict(text="Confusion Matrix",
                                   font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
                        paper_bgcolor=COLORS["panel"], plot_bgcolor=COLORS["panel"],
                        font=dict(color=COLORS["text"]),
                        height=280, margin=dict(l=10, r=10, t=45, b=10),
                        xaxis=dict(tickfont=dict(color=COLORS["text_mid"])),
                        yaxis=dict(tickfont=dict(color=COLORS["text_mid"])),
                    )
                    st.plotly_chart(cmf, width="stretch")

                fi_path = Path("models/results/feature_importance.csv")
                if tag == "rf" and fi_path.exists():
                    fi_df = pd.read_csv(fi_path).head(10)
                    fi_fig = go.Figure(go.Bar(
                        x=fi_df["importance"][::-1],
                        y=fi_df["feature"][::-1],
                        orientation="h",
                        marker_color=[COLORS["accent"] if i < 5 else COLORS["border_light"]
                                      for i in range(len(fi_df)-1, -1, -1)],
                        text=[f"{v:.3f}" for v in fi_df["importance"][::-1]],
                        textposition="outside",
                        textfont=dict(family="JetBrains Mono", size=10, color=COLORS["text_mid"]),
                    ))
                    fi_fig.update_layout(
                        title=dict(text="Top HRV Feature Importances",
                                   font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
                        paper_bgcolor=COLORS["panel"], plot_bgcolor=COLORS["panel"],
                        xaxis=dict(color=COLORS["text_mid"], gridcolor=COLORS["border"],
                                   tickfont=dict(family="JetBrains Mono", size=10)),
                        yaxis=dict(color=COLORS["text"], tickfont=dict(family="Inter", size=11)),
                        height=320, margin=dict(l=130, r=60, t=45, b=40),
                    )
                    st.plotly_chart(fi_fig, width="stretch")

                st.divider()

    # ══ TAB 3 — DATASET ═══════════════════════════════════════════════════════
    with tab3:
        st.markdown("### PhysioNet AF Database")
        col_info, col_strategy = st.columns(2)

        with col_info:
            st.markdown(f"""
            <div class='cs-card'>
              <div class='cs-label'>Dataset Overview</div>
              <table style='width:100%; font-family:"Inter",sans-serif; font-size:0.82rem; border-collapse:collapse;'>
                <tr><td style='padding:5px 0; color:{COLORS["text_mid"]};'>Source</td>
                    <td style='color:{COLORS["text"]}; font-weight:500;'>PhysioNet AFDB 1.0.0</td></tr>
                <tr><td style='padding:5px 0; color:{COLORS["text_mid"]};'>Records</td>
                    <td style='color:{COLORS["text"]}; font-weight:500;'>25 long-term ambulatory ECG</td></tr>
                <tr><td style='padding:5px 0; color:{COLORS["text_mid"]};'>Duration</td>
                    <td style='color:{COLORS["text"]}; font-weight:500;'>~10 hours each</td></tr>
                <tr><td style='padding:5px 0; color:{COLORS["text_mid"]};'>Sample Rate</td>
                    <td style='color:{COLORS["text"]}; font-weight:500;'>250 Hz, 12-bit ADC</td></tr>
                <tr><td style='padding:5px 0; color:{COLORS["text_mid"]};'>Classes</td>
                    <td style='color:{COLORS["text"]}; font-weight:500;'>Normal, AFib, AFL, Junctional</td></tr>
              </table>
            </div>""", unsafe_allow_html=True)

            imb = go.Figure(go.Bar(
                x=["Normal Sinus", "AFib", "AFL", "Other"],
                y=[68, 22, 6, 4],
                marker_color=[COLORS["accent2"], COLORS["danger"], COLORS["warn"], COLORS["border_light"]],
                text=["~68%", "~22%", "~6%", "~4%"],
                textposition="outside",
                textfont=dict(family="Inter", size=11, color=COLORS["text_mid"]),
            ))
            imb.update_layout(
                title=dict(text="Class Distribution",
                           font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
                paper_bgcolor=COLORS["panel"], plot_bgcolor=COLORS["panel"],
                yaxis=dict(title="Approx %", color=COLORS["text_mid"], gridcolor=COLORS["border"],
                           tickfont=dict(family="JetBrains Mono", size=10)),
                xaxis=dict(color=COLORS["text_mid"], tickfont=dict(family="Inter", size=11)),
                height=280, margin=dict(l=50, r=20, t=45, b=40), showlegend=False,
            )
            st.plotly_chart(imb, width="stretch")

        with col_strategy:
            st.markdown(f"""
            <div class='cs-card'>
              <div class='cs-label'>Three-Pronged Imbalance Correction</div>

              <div style='margin-bottom:1.1rem;'>
                <div style='font-weight:600; color:{COLORS["accent"]}; font-size:0.82rem; margin-bottom:4px;'>1 · SMOTE Oversampling</div>
                <div style='font-size:0.78rem; color:{COLORS["text_mid"]}; line-height:1.6;'>
                  Generates synthetic AFib samples by interpolating between real AFib HRV feature vectors. Target: AFib = 80% of Normal count.
                </div>
              </div>

              <div style='margin-bottom:1.1rem;'>
                <div style='font-weight:600; color:{COLORS["accent"]}; font-size:0.82rem; margin-bottom:4px;'>2 · Class-Weighted Loss</div>
                <div style='font-size:0.78rem; color:{COLORS["text_mid"]}; line-height:1.6;'>
                  Penalizes missed AFib cases more heavily, pushing toward higher sensitivity.
                </div>
              </div>

              <div style='margin-bottom:1.1rem;'>
                <div style='font-weight:600; color:{COLORS["accent"]}; font-size:0.82rem; margin-bottom:4px;'>3 · Focal Loss (γ=2, α=0.75)</div>
                <div style='font-size:0.78rem; color:{COLORS["text_mid"]}; line-height:1.6;'>
                  Down-weights easy normal examples, forcing focus on hard borderline AFib cases.
                </div>
              </div>

              <div style='padding:0.8rem; background:{COLORS["panel2"]}; border-radius:8px; border-left:3px solid {COLORS["danger"]};'>
                <div style='font-size:0.75rem; color:{COLORS["text_mid"]}; line-height:1.5;'>
                  <strong style='color:{COLORS["text"]};'>Why this matters:</strong>
                  A naive model gets ~94% accuracy by always predicting Normal — but with 0% AFib sensitivity. Our approach targets &gt;95% sensitivity.
                </div>
              </div>
            </div>""", unsafe_allow_html=True)


    # ══ TAB HISTORY ═══════════════════════════════════════════════════════════
    with tab_hist:
        history = load_history()

        st.markdown(f"""
        <div style='display:flex; align-items:center; justify-content:space-between; margin-bottom:1rem;'>
          <div>
            <span style='font-family:"Sora",sans-serif; font-size:1.1rem; color:white; font-weight:700;'>
              📈 Session History
            </span>
            <span style='font-size:0.78rem; color:{COLORS["text_mid"]}; margin-left:10px;'>
              {len(history)} sessions recorded
            </span>
          </div>
        </div>""", unsafe_allow_html=True)

        if not history:
            st.markdown(f"""
            <div style='padding:60px 20px; text-align:center;'>
              <div style='font-size:3rem; margin-bottom:12px;'>📋</div>
              <div style='font-size:0.9rem; color:{COLORS["text_dim"]};'>
                No sessions yet — run an analysis in the Analysis tab to start tracking
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            # ── Build dataframe ────────────────────────────────────────────────
            rows = []
            for s in history:
                hrv_f = s.get("hrv_features", {})
                rows.append({
                    "timestamp":   s.get("timestamp", ""),
                    "cls":         s.get("classification", ""),
                    "prob":        s.get("afib_probability", 0),
                    "hr":          s.get("heart_rate", 0),
                    "rmssd":       hrv_f.get("rmssd", 0),
                    "sdnn":        hrv_f.get("sdnn", 0),
                    "signal_q":    s.get("signal_quality", ""),
                    "source":      s.get("source", ""),
                })
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            # ── Summary KPI row ────────────────────────────────────────────────
            total     = len(df)
            n_afib    = (df["cls"] == "AFib").sum()
            n_normal  = (df["cls"] == "Normal").sum()
            avg_hr    = df["hr"].mean()
            avg_rmssd = df["rmssd"].mean()
            last_cls  = df["cls"].iloc[-1]
            streak    = 0
            for c in reversed(df["cls"].tolist()):
                if c == "Normal":
                    streak += 1
                else:
                    break

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Total Sessions", total)
            k2.metric("AFib Episodes",  n_afib,
                      delta="High" if n_afib > total * 0.3 else "Low",
                      delta_color="inverse" if n_afib > total * 0.3 else "normal")
            k3.metric("Normal Sessions", n_normal)
            k4.metric("Avg Heart Rate",  f"{avg_hr:.0f} bpm")
            k5.metric("Normal Streak",   f"{streak} sessions",
                      delta="Good" if streak >= 3 else None)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # ── Trend charts ──────────────────────────────────────────────────
            col_a, col_b = st.columns(2)

            with col_a:
                # AFib probability over time
                colors_prob = [COLORS["danger"] if p >= 0.65
                               else COLORS["warn"] if p >= 0.35
                               else COLORS["success"] for p in df["prob"]]
                fig_prob = go.Figure()
                fig_prob.add_hrect(y0=0,    y1=0.35, fillcolor="rgba(31,204,122,0.05)",  line_width=0)
                fig_prob.add_hrect(y0=0.35, y1=0.65, fillcolor="rgba(244,161,36,0.05)",  line_width=0)
                fig_prob.add_hrect(y0=0.65, y1=1.0,  fillcolor="rgba(240,64,96,0.05)",   line_width=0)
                fig_prob.add_hline(y=0.65, line_dash="dash",
                                   line_color=COLORS["danger"], opacity=0.5,
                                   annotation_text="AFib threshold",
                                   annotation_font=dict(color=COLORS["danger"], size=9))
                fig_prob.add_trace(go.Scatter(
                    x=df["timestamp"], y=df["prob"],
                    mode="lines+markers",
                    line=dict(color=COLORS["accent"], width=2),
                    marker=dict(color=colors_prob, size=8,
                                line=dict(color="white", width=1.5)),
                    name="AFib Probability",
                    hovertemplate="%{x|%d %b %H:%M}<br>Prob: %{y:.1%}<extra></extra>",
                ))
                fig_prob.update_layout(
                    title=dict(text="AFib Probability Over Time",
                               font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
                    xaxis=dict(color=COLORS["text_mid"], gridcolor=COLORS["border"],
                               tickfont=dict(family="JetBrains Mono", size=9)),
                    yaxis=dict(title="%", color=COLORS["text_mid"], gridcolor=COLORS["border"],
                               range=[0, 1], tickformat=".0%",
                               tickfont=dict(family="JetBrains Mono", size=9)),
                    plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
                    height=260, margin=dict(l=55, r=15, t=40, b=45),
                    showlegend=False,
                )
                st.plotly_chart(fig_prob, width="stretch")

            with col_b:
                # Heart Rate trend
                fig_hr = go.Figure()
                fig_hr.add_hrect(y0=60,  y1=100, fillcolor="rgba(31,204,122,0.05)", line_width=0)
                fig_hr.add_hline(y=100, line_dash="dash", line_color=COLORS["warn"],
                                 opacity=0.5, annotation_text="Tachycardia",
                                 annotation_font=dict(color=COLORS["warn"], size=9))
                fig_hr.add_hline(y=60,  line_dash="dash", line_color=COLORS["accent2"],
                                 opacity=0.5, annotation_text="Bradycardia",
                                 annotation_font=dict(color=COLORS["accent2"], size=9))
                fig_hr.add_trace(go.Scatter(
                    x=df["timestamp"], y=df["hr"],
                    mode="lines+markers",
                    line=dict(color=COLORS["accent2"], width=2),
                    marker=dict(color=COLORS["accent2"], size=7,
                                line=dict(color="white", width=1.5)),
                    name="Heart Rate",
                    hovertemplate="%{x|%d %b %H:%M}<br>HR: %{y:.0f} bpm<extra></extra>",
                ))
                fig_hr.update_layout(
                    title=dict(text="Heart Rate Over Time",
                               font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
                    xaxis=dict(color=COLORS["text_mid"], gridcolor=COLORS["border"],
                               tickfont=dict(family="JetBrains Mono", size=9)),
                    yaxis=dict(title="bpm", color=COLORS["text_mid"], gridcolor=COLORS["border"],
                               tickfont=dict(family="JetBrains Mono", size=9)),
                    plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
                    height=260, margin=dict(l=55, r=15, t=40, b=45),
                    showlegend=False,
                )
                st.plotly_chart(fig_hr, width="stretch")

            col_c, col_d = st.columns(2)

            with col_c:
                # RMSSD trend — higher = more variable = more AFib risk
                fig_rmssd = go.Figure()
                fig_rmssd.add_hline(y=80, line_dash="dash", line_color=COLORS["warn"],
                                    opacity=0.5, annotation_text="AFib risk zone",
                                    annotation_font=dict(color=COLORS["warn"], size=9))
                fig_rmssd.add_trace(go.Scatter(
                    x=df["timestamp"], y=df["rmssd"],
                    mode="lines+markers",
                    line=dict(color=COLORS["warn"], width=2),
                    marker=dict(color=COLORS["warn"], size=7,
                                line=dict(color="white", width=1.5)),
                    fill="tozeroy", fillcolor="rgba(244,161,36,0.05)",
                    hovertemplate="%{x|%d %b %H:%M}<br>RMSSD: %{y:.1f}ms<extra></extra>",
                ))
                fig_rmssd.update_layout(
                    title=dict(text="RMSSD Trend  (elevated = irregular rhythm)",
                               font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
                    xaxis=dict(color=COLORS["text_mid"], gridcolor=COLORS["border"],
                               tickfont=dict(family="JetBrains Mono", size=9)),
                    yaxis=dict(title="ms", color=COLORS["text_mid"], gridcolor=COLORS["border"],
                               tickfont=dict(family="JetBrains Mono", size=9)),
                    plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
                    height=260, margin=dict(l=55, r=15, t=40, b=45),
                    showlegend=False,
                )
                st.plotly_chart(fig_rmssd, width="stretch")

            with col_d:
                # Classification distribution donut
                cls_counts = df["cls"].value_counts()
                fig_donut = go.Figure(go.Pie(
                    labels=cls_counts.index.tolist(),
                    values=cls_counts.values.tolist(),
                    hole=0.55,
                    marker=dict(colors=[
                        COLORS["danger"]  if l == "AFib"       else
                        COLORS["warn"]    if l == "Borderline" else
                        COLORS["success"] for l in cls_counts.index
                    ],
                    line=dict(color=COLORS["panel"], width=2)),
                    textfont=dict(family="Inter", size=11, color="white"),
                    hovertemplate="%{label}: %{value} sessions (%{percent})<extra></extra>",
                ))
                fig_donut.update_layout(
                    title=dict(text="Classification Breakdown",
                               font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
                    paper_bgcolor=COLORS["panel"],
                    legend=dict(font=dict(family="Inter", size=11, color=COLORS["text"]),
                                bgcolor="rgba(0,0,0,0)"),
                    height=260, margin=dict(l=15, r=15, t=40, b=15),
                    annotations=[dict(
                        text=f"{n_afib/total*100:.0f}%<br>AFib",
                        x=0.5, y=0.5, font=dict(size=16, color=COLORS["danger"],
                                                  family="JetBrains Mono"),
                        showarrow=False
                    )]
                )
                st.plotly_chart(fig_donut, width="stretch")

            # ── Session log table ──────────────────────────────────────────────
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            st.markdown(f'<div class="cs-label">Session Log</div>', unsafe_allow_html=True)

            display_df = df[["timestamp","cls","prob","hr","rmssd","sdnn","signal_q"]].copy()
            display_df.columns = ["Time","Classification","AFib Prob","Heart Rate","RMSSD","SDNN","Signal Quality"]
            display_df["Time"]      = display_df["Time"].dt.strftime("%d %b %Y  %H:%M")
            display_df["AFib Prob"] = display_df["AFib Prob"].apply(lambda x: f"{x*100:.1f}%")
            display_df["Heart Rate"]= display_df["Heart Rate"].apply(lambda x: f"{x:.0f} bpm")
            display_df["RMSSD"]     = display_df["RMSSD"].apply(lambda x: f"{x:.1f} ms")
            display_df["SDNN"]      = display_df["SDNN"].apply(lambda x: f"{x:.1f} ms")
            st.dataframe(display_df.iloc[::-1].reset_index(drop=True),
                         use_container_width=True, hide_index=True)

            # ── Export + Clear ─────────────────────────────────────────────────
            ex1, ex2, _ = st.columns([1, 1, 4])
            with ex1:
                st.download_button(
                    "⬇  Export Full History (JSON)",
                    data=json.dumps(history, indent=2),
                    file_name=f"cardiosense_history_{time.strftime('%Y%m%d')}.json",
                    mime="application/json",
                )
            with ex2:
                if st.button("🗑  Clear History", key="clear_hist"):
                    delete_history()
                    st.rerun()


if __name__ == "__main__":
    main()