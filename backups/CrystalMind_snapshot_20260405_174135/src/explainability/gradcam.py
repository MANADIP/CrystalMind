"""
src/explainability/gradcam.py
Grad-CAM saliency for 1D XRD signals.
"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Compute Grad-CAM saliency over the 2theta axis for a CrystalMind model.

    Parameters
    ----------
    model : nn.Module
        Any model with forward_with_features().
    task : str
        Which prediction head to backprop through.
    target_layer : str | None
        Optional dotted module path. Defaults to backbone.layer4 when present.
    """

    def __init__(self, model, task: str = "crystal_system", target_layer: str = None):
        self.model = model
        self.task = task
        self.gradients = None
        self.activations = None

        self.target, self.feature_layout = self._resolve_target_layer(target_layer)
        self._fwd_handle = self.target.register_forward_hook(self._fwd_hook)
        self._bwd_handle = self.target.register_full_backward_hook(self._bwd_hook)

    def _lookup_module(self, dotted_path: str):
        module = self.model
        path = dotted_path.replace("model.", "")
        for part in path.split("."):
            if not hasattr(module, part):
                return None
            module = getattr(module, part)
        return module

    def _resolve_target_layer(self, target_layer: str):
        candidates = []
        if target_layer is not None:
            candidates.append(target_layer)
            if "." not in target_layer and hasattr(self.model, "backbone"):
                candidates.append(f"backbone.{target_layer}")
        else:
            if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "layer4"):
                candidates.append("backbone.layer4")
            candidates.extend(["features", "encoder"])

        for candidate in candidates:
            module = self._lookup_module(candidate)
            if module is None:
                continue
            layout = "bld" if isinstance(module, nn.TransformerEncoder) else "bcl"
            return module, layout

        raise ValueError(f"Could not resolve Grad-CAM target layer '{target_layer}'.")

    def _fwd_hook(self, module, inputs, output):
        activations = output[0] if isinstance(output, (tuple, list)) else output
        if activations.dim() == 3 and self.feature_layout == "bld":
            activations = activations.transpose(1, 2)
        self.activations = activations.detach()

    def _bwd_hook(self, module, grad_input, grad_output):
        gradients = grad_output[0]
        if gradients.dim() == 3 and self.feature_layout == "bld":
            gradients = gradients.transpose(1, 2)
        self.gradients = gradients.detach()

    def compute(self, x: torch.Tensor, class_idx: int = None) -> tuple[np.ndarray, int]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (1, 1, signal_length) on the correct device.
        class_idx : int | None
            Target class. Uses the predicted class when omitted.

        Returns
        -------
        cam : np.ndarray
            Saliency normalized to [0, 1] on the input grid.
        class_idx : int
            The class whose score was backpropagated.
        """
        self.model.eval()
        self.gradients = None
        self.activations = None
        x = x.requires_grad_(True)

        preds, _ = self.model.forward_with_features(x)
        logits = preds[self.task]

        if class_idx is None:
            if logits.dim() > 1:
                class_idx = logits.argmax(dim=1).item()
            else:
                class_idx = 0

        score = logits[0, class_idx] if logits.dim() > 1 else logits[0]
        self.model.zero_grad()
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=x.shape[-1],
            mode="linear",
            align_corners=False,
        ).squeeze().detach().cpu().numpy()

        cam = np.maximum(cam, 0.0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

    def plot(
        self,
        xrd_pattern: np.ndarray,
        cam: np.ndarray,
        class_idx: int,
        material_id: str = "",
        save_path: str = None,
        two_theta_range: tuple = (5, 90),
    ):
        """Plot the XRD pattern colored by Grad-CAM saliency."""
        two_theta = np.linspace(*two_theta_range, len(xrd_pattern))
        class_name = CRYSTAL_SYSTEM_NAMES.get(class_idx, str(class_idx))
        title = f"Grad-CAM | {class_name}"
        if material_id:
            title += f" ({material_id})"

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(13, 5),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
        fig.suptitle(title, fontsize=13, y=1.01)

        cmap = cm.get_cmap("hot_r")
        colors = cmap(cam)
        for idx in range(len(two_theta) - 1):
            axes[0].fill_between(
                two_theta[idx : idx + 2],
                xrd_pattern[idx : idx + 2],
                color=colors[idx],
                alpha=0.85,
                linewidth=0,
            )
        axes[0].plot(two_theta, xrd_pattern, color="0.3", linewidth=0.5, alpha=0.6)
        axes[0].set_ylabel("Intensity (norm.)")
        axes[0].set_ylim(0, 1.05)

        scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        scalar_map.set_array([])
        fig.colorbar(scalar_map, ax=axes[0], label="Saliency", pad=0.01)

        axes[1].plot(two_theta, cam, color="#E8593C", linewidth=1.2)
        axes[1].fill_between(two_theta, cam, alpha=0.25, color="#E8593C")
        axes[1].set_ylabel("Grad-CAM")
        axes[1].set_xlabel("2theta (deg)")
        axes[1].set_ylim(0, 1.05)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Grad-CAM] saved -> {save_path}")

        plt.show()
        plt.close()

    def run_batch(
        self,
        patterns: np.ndarray,
        material_ids: list = None,
        out_dir: str = "outputs/plots",
        device: str = "cpu",
    ):
        """Run Grad-CAM on a batch of patterns and save all plots."""
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for idx, pattern in enumerate(patterns):
            material_id = material_ids[idx] if material_ids else f"sample_{idx:04d}"
            x = torch.tensor(pattern).unsqueeze(0).unsqueeze(0).float().to(device)
            cam, cls_idx = self.compute(x)
            self.plot(
                pattern,
                cam,
                cls_idx,
                material_id=material_id,
                save_path=f"{out_dir}/gradcam_{material_id}.png",
            )

    def close(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()
