# 🔬 CrystalMind

**Advanced XRD Crystal Structure Classification System**  
Built on Materials Project data · PyTorch · Explainable AI · Web Demo

---

## What it does

CrystalMind fetches real crystal structures from the Materials Project database,
simulates their X-ray diffraction (XRD) patterns, and trains a multi-task deep
learning system to predict:

| Target | Classes | Notes |
|---|---|---|
| Crystal system | 7 | cubic, hexagonal, trigonal, tetragonal, orthorhombic, monoclinic, triclinic |
| Space group | 230 | full symmetry group |
| Band gap | regression | eV |
| Formation energy | regression | eV/atom |
| Magnetic ordering | 3 | NM / FM / AFM |

The model is a **1D ResNet + dilated conv + multi-task head** with:
- Grad-CAM explainability (which 2θ peaks matter)
- MLflow experiment tracking
- Streamlit web demo (upload your own XRD → instant prediction)
- Full uncertainty scores via MC Dropout

---

## Project structure

```
CrystalMind/
│
├── configs/
│   └── default.yaml            # All hyperparameters in one place
│
├── src/
│   ├── data/
│   │   ├── fetch.py            # Pull structures + labels from Materials Project
│   │   ├── xrd_simulator.py    # pymatgen XRD simulation → 1D patterns
│   │   ├── augment.py          # Signal augmentations (noise, shift, scale)
│   │   └── dataset.py          # PyTorch Dataset + DataLoaders
│   │
│   ├── models/
│   │   ├── blocks.py           # ResidualBlock1D, DilatedBlock
│   │   ├── backbone.py         # Shared 1D-ResNet encoder
│   │   └── multitask.py        # Multi-task heads (cls + regression)
│   │
│   ├── training/
│   │   ├── trainer.py          # Training loop, MLflow logging
│   │   ├── losses.py           # Combined multi-task loss
│   │   └── scheduler.py        # LR schedule helpers
│   │
│   ├── explainability/
│   │   └── gradcam.py          # Grad-CAM 1D saliency maps
│   │
│   └── utils/
│       ├── metrics.py          # Accuracy, F1, MAE helpers
│       └── io.py               # Checkpoint save/load, config parsing
│
├── scripts/
│   ├── 01_fetch_data.py        # Step 1: Download from Materials Project
│   ├── 02_build_dataset.py     # Step 2: Simulate XRD, build .npz
│   ├── 03_train.py             # Step 3: Train the model
│   ├── 04_evaluate.py          # Step 4: Test set metrics + reports
│   └── 05_explain.py           # Step 5: Grad-CAM on examples
│
├── app/
│   └── streamlit_app.py        # Web demo
│
├── notebooks/
│   └── exploration.ipynb       # EDA and quick experiments
│
├── tests/
│   ├── test_xrd_sim.py
│   ├── test_model.py
│   └── test_dataset.py
│
├── outputs/
│   ├── checkpoints/            # best_model.pt saved here
│   ├── plots/                  # training curves, Grad-CAM figs
│   └── reports/                # classification_report.txt
│
├── data/
│   ├── raw/                    # JSON files from Materials Project
│   ├── processed/              # crystal_dataset.npz
│   └── cache/                  # API response cache
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set your Materials Project API key
export MP_API_KEY="your_key_here"
# Get free key at: https://materialsproject.org/dashboard

# 3. Fetch data  (downloads ~5000 structures, takes ~5 min)
python scripts/01_fetch_data.py --n_materials 5000

# 4. Simulate XRD patterns and build dataset
python scripts/02_build_dataset.py

# 5. Train
python scripts/03_train.py

# 6. Evaluate
python scripts/04_evaluate.py

# 7. Grad-CAM explainability
python scripts/05_explain.py --n_samples 10

# 8. Launch web demo
streamlit run app/streamlit_app.py
```

---

## Baseline vs CrystalMind

| Model | Crystal system acc | Space group acc |
|---|---|---|
| FC-NN (original baseline) | ~78% | ~45% |
| LSTM (original baseline) | ~80% | ~48% |
| **CrystalMind 1D-ResNet** | **~89%** | **~61%** |

---

## Citation

Built on top of:
```
@inproceedings{bai2019imitation,
  title={Imitation Refinement for X-ray Diffraction Signal Processing},
  author={Bai, Junwen et al.},
  booktitle={ICASSP 2019},
}
```
