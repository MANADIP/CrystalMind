"""
src/data/dataset.py
PyTorch Dataset for XRD crystal classification (multi-task).

Loads the pre-built .npz file and returns a dict of tensors for each task.
Handles both the original format (with regression targets) and the
simplified format (crystal_system + space_group only).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from src.data.augment import XRDAugmentor


class XRDDataset(Dataset):
    """
    Multi-task XRD dataset.

    Each sample is:
        x  : FloatTensor (1, 2000)  — normalised XRD pattern
        y  : dict of tensors for each task

    Parameters
    ----------
    patterns          : np.ndarray  (N, 2000)
    labels            : dict  task_name -> np.ndarray
    augmentor         : XRDAugmentor | None   (None = no augmentation)
    """

    def __init__(self,
                 patterns: np.ndarray,
                 labels: dict,
                 augmentor: XRDAugmentor = None):
        self.patterns  = patterns.astype(np.float32)
        self.labels    = labels
        self.augmentor = augmentor

        # Pre-normalise each pattern to [0, 1] if not already
        maxes = self.patterns.max(axis=1, keepdims=True)
        maxes[maxes == 0] = 1.0
        self.patterns = self.patterns / maxes

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        x = self.patterns[idx]

        if self.augmentor is not None:
            x = self.augmentor(x)

        x = torch.tensor(x).unsqueeze(0)   # (1, 2000)

        y = {}
        for task, arr in self.labels.items():
            val = arr[idx]
            if arr.dtype in (np.int32, np.int64, int):
                y[task] = torch.tensor(int(val), dtype=torch.long)
            else:
                y[task] = torch.tensor(float(val), dtype=torch.float)

        return x, y


def build_dataloaders(
    npz_path: str,
    cfg: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load .npz, split into train/val/test, return three DataLoaders.

    The training loader uses WeightedRandomSampler on crystal_system
    to handle class imbalance.

    Supports both npz formats:
      - Original: keys = xrd, crystal_system, space_group, band_gap, ...
      - Simplified: keys = xrd, crystal_system, space_group, mp
    """
    data_cfg  = cfg["data"]
    task_cfg  = cfg["tasks"]
    val_frac  = data_cfg["val_fraction"]
    test_frac = data_cfg["test_fraction"]
    seed      = data_cfg["random_seed"]
    batch     = cfg["training"]["batch_size"]

    # ── Load ──────────────────────────────────────────────────────────
    ds = np.load(npz_path, allow_pickle=True)
    X  = ds["xrd"]                                       # (N, 2000)

    labels = {}
    for task in task_cfg:
        if task in ds:
            labels[task] = ds[task]

    if not labels:
        raise RuntimeError(
            f"No matching task keys found in {npz_path}. "
            f"NPZ keys: {list(ds.keys())}, "
            f"Config tasks: {list(task_cfg.keys())}"
        )

    # ── Split ─────────────────────────────────────────────────────────
    N     = len(X)
    idx   = np.arange(N)
    strat = ds["crystal_system"] if "crystal_system" in ds else None

    idx_tmp, idx_test = train_test_split(
        idx, test_size=test_frac, random_state=seed, stratify=strat)
    strat_tmp = strat[idx_tmp] if strat is not None else None
    idx_train, idx_val = train_test_split(
        idx_tmp, test_size=val_frac / (1 - test_frac),
        random_state=seed, stratify=strat_tmp)

    def subset(idxs):
        return X[idxs], {t: v[idxs] for t, v in labels.items()}

    X_train, y_train = subset(idx_train)
    X_val,   y_val   = subset(idx_val)
    X_test,  y_test  = subset(idx_test)

    # ── Augmentation (train only) ─────────────────────────────────────
    augmentor = XRDAugmentor(data_cfg) if data_cfg.get("augment", {}).get("enabled") else None

    train_ds = XRDDataset(X_train, y_train, augmentor=augmentor)
    val_ds   = XRDDataset(X_val,   y_val)
    test_ds  = XRDDataset(X_test,  y_test)

    # ── Weighted sampler for class balance ────────────────────────────
    if "crystal_system" in y_train:
        cs_labels  = y_train["crystal_system"].astype(int)
        class_freq = np.bincount(cs_labels, minlength=7)
        # Guard against zero-count classes
        class_freq = np.maximum(class_freq, 1)
        weights    = 1.0 / class_freq[cs_labels]
        sampler    = WeightedRandomSampler(weights, num_samples=len(train_ds),
                                           replacement=True)
    else:
        sampler = None

    train_loader = DataLoader(train_ds, batch_size=batch,
                              sampler=sampler,
                              shuffle=(sampler is None),
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"Dataset  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    return train_loader, val_loader, test_loader
