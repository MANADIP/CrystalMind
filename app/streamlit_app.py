"""
app/streamlit_app.py
CrystalMind — Interactive XRD Crystal Classification Demo

Run:
    streamlit run app/streamlit_app.py

Features:
  - Upload a raw XRD pattern (.txt, .xy, .csv  —  two-column: 2θ, intensity)
  - OR select a random sample from the test set
  - Predict crystal system, space group, band gap, formation energy
  - Show Grad-CAM saliency overlay
  - Show MC Dropout confidence intervals
"""

import sys
import io
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.multitask       import CrystalMind
from src.explainability.gradcam  import GradCAM1D
from src.data.xrd_simulator      import simulate_xrd
from src.utils.io                import load_config, load_checkpoint

# ─── Constants ────────────────────────────────────────────────────────────────

CRYSTAL_SYSTEM_NAMES = [
    "Cubic", "Hexagonal", "Trigonal", "Tetragonal",
    "Orthorhombic", "Monoclinic", "Triclinic",
]
ORDERING_NAMES = ["Non-magnetic", "Ferromagnetic", "Antiferromagnetic"]

# ─── App setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CrystalMind",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 CrystalMind")
st.caption("Advanced XRD Crystal Structure Classification · Materials Project · PyTorch")

# ─── Load model (cached) ──────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    cfg   = load_config("configs/default.yaml")
    model = CrystalMind(cfg)
    ckpt  = cfg["paths"]["checkpoint"]
    if not Path(ckpt).exists():
        return None, None, cfg
    load_checkpoint(model, ckpt, device="cpu")
    model.eval()
    return model, GradCAM1D(model, task="crystal_system"), cfg


model, gradcam, cfg = load_model()

if model is None:
    st.error(
        "No trained checkpoint found at `outputs/checkpoints/best_model.pt`.\n\n"
        "Run the training pipeline first:\n"
        "```\n"
        "python scripts/01_fetch_data.py\n"
        "python scripts/02_build_dataset.py\n"
        "python scripts/03_train.py\n"
        "```"
    )
    st.stop()

# ─── Sidebar — input options ──────────────────────────────────────────────────

st.sidebar.header("Input")
input_mode = st.sidebar.radio(
    "XRD source",
    ["Upload .txt / .xy / .csv", "Random from test set", "Example (Silicon)"],
)
n_mc_passes = st.sidebar.slider("MC Dropout passes", 5, 50, 20,
                                 help="More passes → better uncertainty estimate")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def interpolate_to_fixed(two_theta_raw, intensity_raw,
                          n_points: int = 2000,
                          tmin: float = 5.0, tmax: float = 90.0) -> np.ndarray:
    """Resample any uploaded XRD pattern onto the model's fixed grid."""
    tt_fixed = np.linspace(tmin, tmax, n_points)
    sig = np.interp(tt_fixed, two_theta_raw, intensity_raw)
    sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-8)
    return sig.astype(np.float32)


def silicon_example(cfg: dict) -> np.ndarray:
    """Simulate Silicon (mp-149) as a built-in demo."""
    try:
        from mp_api.client import MPRester
        import os
        key = os.environ.get("MP_API_KEY", "")
        if not key:
            raise RuntimeError("No API key")
        with MPRester(key) as mpr:
            struct = mpr.get_structure_by_material_id("mp-149")
        return simulate_xrd(
            struct,
            wavelength=cfg["data"]["xrd_wavelength"],
            n_points=cfg["data"]["xrd_n_points"],
        )
    except Exception:
        # Fallback: synthetic Si-like pattern
        tt = np.linspace(5, 90, cfg["data"]["xrd_n_points"])
        sig = np.zeros_like(tt)
        for peak_deg in [28.4, 47.3, 56.1, 69.1, 76.4, 88.0]:
            sig += np.exp(-0.5 * ((tt - peak_deg) / 0.15) ** 2)
        return (sig / sig.max()).astype(np.float32)


def load_uploaded(file) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse a two-column text file: [2θ, intensity]."""
    try:
        content = file.read().decode("utf-8")
        rows = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                rows.append((float(parts[0]), float(parts[1])))
        if not rows:
            return None
        arr = np.array(rows)
        return arr[:, 0], arr[:, 1]
    except Exception:
        return None


# ─── Get pattern ──────────────────────────────────────────────────────────────

pattern_1d = None
label_str  = ""

if input_mode == "Upload .txt / .xy / .csv":
    uploaded = st.sidebar.file_uploader(
        "Drop your XRD file here",
        type=["txt", "xy", "csv", "dat"],
    )
    if uploaded:
        result = load_uploaded(uploaded)
        if result is None:
            st.error("Could not parse file. Expected two columns: 2θ, intensity.")
        else:
            tt_raw, int_raw = result
            pattern_1d = interpolate_to_fixed(
                tt_raw, int_raw,
                n_points=cfg["data"]["xrd_n_points"],
                tmin=cfg["data"]["two_theta_min"],
                tmax=cfg["data"]["two_theta_max"],
            )
            label_str = f"Uploaded: {uploaded.name}"

elif input_mode == "Random from test set":
    npz_path = cfg["paths"]["processed"]
    if not Path(npz_path).exists():
        st.warning("Dataset not found — run scripts/02_build_dataset.py first.")
    else:
        ds = np.load(npz_path, allow_pickle=True)
        N  = len(ds["xrd"])
        idx = st.sidebar.number_input("Sample index", 0, N - 1,
                                       int(np.random.randint(0, N)))
        pattern_1d = ds["xrd"][idx]
        true_cs    = CRYSTAL_SYSTEM_NAMES[ds["crystal_system"][idx]]
        mat_id     = str(ds["material_ids"][idx]) if "material_ids" in ds else "?"
        label_str  = f"Sample {idx} | {mat_id} | True: {true_cs}"

else:  # Silicon example
    with st.spinner("Generating Silicon XRD …"):
        pattern_1d = silicon_example(cfg)
    label_str = "Silicon (mp-149) — Cu Kα"

# ─── Predict ──────────────────────────────────────────────────────────────────

if pattern_1d is not None:
    x = torch.tensor(pattern_1d).unsqueeze(0).unsqueeze(0)   # (1, 1, 2000)

    # Deterministic prediction
    with torch.no_grad():
        preds = model(x)

    pred_cs  = preds["crystal_system"].argmax(1).item()
    pred_sg  = preds["space_group"].argmax(1).item() + 1   # 1-indexed
    pred_bg  = preds["band_gap"].item()
    pred_fe  = preds["formation_energy"].item()
    pred_mag = preds["magnetic_ordering"].argmax(1).item()

    cs_probs = preds["crystal_system"].softmax(1).squeeze().numpy()

    # MC Dropout uncertainty
    with st.spinner(f"Running {n_mc_passes} MC Dropout passes …"):
        unc = model.predict_with_uncertainty(x, n_passes=n_mc_passes)
    cs_std = unc["std_crystal_system"].squeeze().numpy()

    # Grad-CAM
    cam, _ = gradcam.compute(x, class_idx=pred_cs)

    two_theta = np.linspace(
        cfg["data"]["two_theta_min"],
        cfg["data"]["two_theta_max"],
        cfg["data"]["xrd_n_points"],
    )

    # ─── Layout ───────────────────────────────────────────────────────
    st.subheader(label_str)

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Crystal system", CRYSTAL_SYSTEM_NAMES[pred_cs])
    col_b.metric("Space group",    f"#{pred_sg}")
    col_c.metric("Band gap",       f"{max(pred_bg, 0):.3f} eV")
    col_d.metric("Formation E",    f"{pred_fe:.3f} eV/atom")
    col_e.metric("Magnetic order", ORDERING_NAMES[pred_mag])

    # ─── XRD + Grad-CAM plot ──────────────────────────────────────────
    fig = go.Figure()

    # Colour XRD by saliency
    for i in range(len(two_theta) - 1):
        alpha = float(cam[i])
        r = int(232 + (255 - 232) * alpha)
        g = int(89  * (1 - alpha))
        b = int(60  * (1 - alpha))
        fig.add_trace(go.Scatter(
            x=two_theta[i:i+2], y=pattern_1d[i:i+2],
            mode="lines",
            line=dict(color=f"rgb({r},{g},{b})", width=2),
            showlegend=False, hoverinfo="skip",
        ))

    # Grad-CAM overlay
    fig.add_trace(go.Scatter(
        x=two_theta, y=cam * pattern_1d.max(),
        mode="lines", name="Grad-CAM saliency",
        line=dict(color="orange", width=1, dash="dot"),
        opacity=0.7,
    ))

    fig.update_layout(
        title="XRD pattern coloured by Grad-CAM saliency",
        xaxis_title="2θ (°)", yaxis_title="Intensity (norm.)",
        height=380, legend=dict(x=0.75, y=0.98),
        margin=dict(l=50, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── Crystal system probability + uncertainty ─────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Crystal system probabilities")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=CRYSTAL_SYSTEM_NAMES, y=cs_probs,
            error_y=dict(type="data", array=cs_std, visible=True),
            marker_color=["#185FA5" if i == pred_cs else "#B5D4F4"
                          for i in range(7)],
        ))
        fig2.update_layout(
            yaxis_title="Probability", height=280,
            margin=dict(l=20, r=20, t=10, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Prediction confidence")
        confidence = float(cs_probs[pred_cs])
        uncertainty = float(cs_std[pred_cs])
        st.metric("Confidence",   f"{confidence:.1%}")
        st.metric("Uncertainty",  f"±{uncertainty:.3f}  (MC Dropout std)")
        st.progress(min(confidence, 1.0))

        st.caption(
            "Uncertainty is estimated via Monte Carlo Dropout: "
            f"{n_mc_passes} stochastic forward passes with dropout active. "
            "Lower std → higher reliability."
        )

# ─── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "CrystalMind · Data: [Materials Project](https://materialsproject.org) · "
    "Model: 1D ResNet + SE blocks + multi-task heads · "
    "Explainability: Grad-CAM"
)
