"""
src/explainability/gradcam.py
Grad-CAM saliency for 1D XRD signals.

Highlights which 2θ regions the model focuses on when predicting
a crystal property. Produces interpretable peak-attribution plots.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


CRYSTAL_SYSTEM_NAMES = {
    0: "Cubic",
    1: "Hexagonal",
    2: "Trigonal",
    3: "Tetragonal",
    4: "Orthorhombic",
    5: "Monoclinic",
    6: "Triclinic",
}


class GradCAM1D:
    """
    Computes Grad-CAM saliency over the 2θ axis for a CrystalMind model.

    Parameters
    ----------
    model      : CrystalMind  (or any model with forward_with_features())
    task       : str  which head to backprop through ('crystal_system', etc.)
    """

    def __init__(self, model, task: str = "crystal_system"):
        self.model      = model
        self.task       = task
        self.gradients  = None
        self.activations = None

        # Hook the last conv layer of the backbone (layer4's last block)
        target = list(model.backbone.layer4.children())[-1].conv2

        target.register_forward_hook(self._fwd_hook)
        target.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def compute(self,
                x: torch.Tensor,
                class_idx: int = None) -> tuple[np.ndarray, int]:
        """
        Parameters
        ----------
        x         : (1, 1, 2000) tensor on the correct device
        class_idx : target class (None → predicted class)

        Returns
        -------
        cam       : np.ndarray (2000,)  saliency in [0, 1]
        class_idx : int  the class whose score was backpropagated
        """
        self.model.eval()
        # Enable gradient computation even in eval mode
        x = x.requires_grad_(True)

        preds, _ = self.model.forward_with_features(x)
        logits   = preds[self.task]

        if class_idx is None:
            if logits.dim() > 1:
                class_idx = logits.argmax(dim=1).item()
            else:
                class_idx = 0   # regression: single output

        score = logits[0, class_idx] if logits.dim() > 1 else logits[0]
        self.model.zero_grad()
        score.backward()

        # Pool gradients across the channel dim → (1, L)
        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)      # (1, L)
        cam = F.relu(cam).squeeze(0)                        # (L,)

        # Interpolate to input length
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=2000, mode="linear", align_corners=False
        ).squeeze().cpu().detach().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

    def plot(self,
             xrd_pattern: np.ndarray,
             cam: np.ndarray,
             class_idx: int,
             material_id: str = "",
             save_path: str = None,
             two_theta_range: tuple = (5, 90)):
        """
        Two-panel plot:
          Top    — XRD pattern coloured by Grad-CAM saliency
          Bottom — Saliency curve with filled area
        """
        two_theta = np.linspace(*two_theta_range, len(xrd_pattern))
        class_name = CRYSTAL_SYSTEM_NAMES.get(class_idx, str(class_idx))
        title_str  = f"Grad-CAM  |  {class_name}"
        if material_id:
            title_str += f"  ({material_id})"

        fig, axes = plt.subplots(
            2, 1, figsize=(13, 5), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]}
        )
        fig.suptitle(title_str, fontsize=13, y=1.01)

        # ── Top panel: pattern coloured by saliency ──────────────────
        cmap   = cm.get_cmap("hot_r")
        colors = cmap(cam)
        for i in range(len(two_theta) - 1):
            axes[0].fill_between(
                two_theta[i:i+2], xrd_pattern[i:i+2],
                color=colors[i], alpha=0.85, linewidth=0
            )
        axes[0].plot(two_theta, xrd_pattern, color="0.3", linewidth=0.5, alpha=0.6)
        axes[0].set_ylabel("Intensity (norm.)")
        axes[0].set_ylim(0, 1.05)

        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        fig.colorbar(sm, ax=axes[0], label="Saliency", pad=0.01)

        # ── Bottom panel: saliency curve ─────────────────────────────
        axes[1].plot(two_theta, cam, color="#E8593C", linewidth=1.2)
        axes[1].fill_between(two_theta, cam, alpha=0.25, color="#E8593C")
        axes[1].set_ylabel("Grad-CAM")
        axes[1].set_xlabel("2θ (°)")
        axes[1].set_ylim(0, 1.05)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Grad-CAM] saved → {save_path}")

        plt.show()
        plt.close()

    def run_batch(self,
                  patterns: np.ndarray,
                  material_ids: list = None,
                  out_dir: str = "outputs/plots",
                  device: str = "cpu"):
        """
        Run Grad-CAM on a batch of patterns and save all plots.

        Parameters
        ----------
        patterns     : (N, 2000)
        material_ids : list of str (optional)
        out_dir      : where to save PNG files
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for i, pattern in enumerate(patterns):
            mid  = material_ids[i] if material_ids else f"sample_{i:04d}"
            x    = torch.tensor(pattern).unsqueeze(0).unsqueeze(0).float().to(device)
            cam, cls_idx = self.compute(x)
            self.plot(
                pattern, cam, cls_idx,
                material_id=mid,
                save_path=f"{out_dir}/gradcam_{mid}.png",
            )
