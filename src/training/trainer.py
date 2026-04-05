"""
src/training/trainer.py
Main training engine for CrystalMind.

Features
--------
- Multi-task loss (classification + regression heads)
- Grad-norm clipping
- Linear warmup + cosine LR schedule
- MLflow experiment tracking (metrics, params, artifacts)
- Early stopping on val crystal_system accuracy
- Best-checkpoint saving
"""

import time
import torch
import torch.nn.functional as F
import mlflow

from src.training.losses    import MultiTaskLoss
from src.training.scheduler import build_scheduler, LinearWarmup
from src.utils.io            import save_checkpoint
from src.utils.metrics       import cls_metrics, reg_metrics


# ── Single epoch ──────────────────────────────────────────────────────

def _run_epoch(model, loader, criterion, optimiser, device, train: bool):
    model.train() if train else model.eval()

    total_loss   = 0.0
    task_losses  = {}
    all_preds    = {}
    all_targets  = {}
    n_samples    = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb = xb.to(device)
            yb = {k: v.to(device) for k, v in yb.items()}

            preds = model(xb)
            loss, per_task = criterion(preds, yb)

            if train:
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            n_samples  += bs

            for t, l in per_task.items():
                task_losses[t] = task_losses.get(t, 0.0) + l * bs

            # Collect predictions for metrics
            for t, pred in preds.items():
                if t not in all_preds:
                    all_preds[t]   = []
                    all_targets[t] = []
                if pred.dim() > 1:          # classification
                    all_preds[t].append(pred.argmax(1).cpu())
                else:                        # regression
                    all_preds[t].append(pred.detach().cpu())
                all_targets[t].append(yb[t].cpu())

    # Concatenate
    for t in all_preds:
        all_preds[t]   = torch.cat(all_preds[t]).numpy()
        all_targets[t] = torch.cat(all_targets[t]).numpy()

    # Metrics
    metrics = {"loss": total_loss / n_samples}
    for t in task_losses:
        metrics[f"loss_{t}"] = task_losses[t] / n_samples

    task_cfg = criterion.task_cfg
    for t, tcfg in task_cfg.items():
        if t not in all_preds:
            continue
        if tcfg["type"] == "classification":
            m = cls_metrics(all_targets[t], all_preds[t])
            metrics[f"acc_{t}"] = m["accuracy"]
            metrics[f"f1_{t}"]  = m["f1"]
        else:
            m = reg_metrics(all_targets[t], all_preds[t])
            metrics[f"mae_{t}"] = m["mae"]
            metrics[f"r2_{t}"]  = m["r2"]

    return metrics


# ── Main trainer ──────────────────────────────────────────────────────

class Trainer:

    def __init__(self, model, cfg: dict, device: torch.device):
        self.model  = model.to(device)
        self.cfg    = cfg
        self.device = device

        train_cfg = cfg["training"]
        self.epochs   = train_cfg["epochs"]
        self.patience = train_cfg.get("early_stopping_patience", 15)
        self.ckpt_path = cfg["paths"]["checkpoint"]

        self.criterion = MultiTaskLoss(
            cfg["tasks"],
            label_smoothing=train_cfg.get("label_smoothing", 0.05),
        ).to(device)

        self.optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        self.scheduler, warmup_ep = build_scheduler(self.optimiser, cfg)
        self.warmup = LinearWarmup(
            self.optimiser,
            base_lr=float(train_cfg["learning_rate"]),
            warmup_epochs=warmup_ep,
        )

    def fit(self, train_loader, val_loader):
        best_val_acc  = 0.0
        patience_ctr  = 0

        mlflow.set_experiment(self.cfg["mlflow"]["experiment_name"])
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                "epochs":       self.epochs,
                "lr":           self.cfg["training"]["learning_rate"],
                "batch_size":   self.cfg["training"]["batch_size"],
                "scheduler":    self.cfg["training"]["scheduler"],
                "n_params":     self.model.count_parameters(),
            })

            for epoch in range(1, self.epochs + 1):
                t0 = time.time()

                # Warmup
                self.warmup.step(epoch - 1)

                train_m = _run_epoch(self.model, train_loader,
                                     self.criterion, self.optimiser,
                                     self.device, train=True)
                val_m   = _run_epoch(self.model, val_loader,
                                     self.criterion, None,
                                     self.device, train=False)

                # Scheduler step (plateau uses val loss)
                if hasattr(self.scheduler, "step"):
                    if isinstance(self.scheduler,
                                  torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_m["loss"])
                    elif epoch > self.warmup.warmup_epochs:
                        self.scheduler.step()

                # MLflow logging
                log_dict = {}
                for k, v in train_m.items():
                    log_dict[f"train_{k}"] = v
                for k, v in val_m.items():
                    log_dict[f"val_{k}"] = v
                log_dict["lr"] = self.optimiser.param_groups[0]["lr"]
                mlflow.log_metrics(log_dict, step=epoch)

                # Console
                elapsed = time.time() - t0
                val_acc = val_m.get("acc_crystal_system", 0.0)
                print(
                    f"Ep {epoch:3d}/{self.epochs}  "
                    f"loss={train_m['loss']:.4f}  "
                    f"val_loss={val_m['loss']:.4f}  "
                    f"val_acc(sys)={val_acc:.4f}  "
                    f"({elapsed:.1f}s)"
                )

                # Checkpoint + early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_ctr = 0
                    save_checkpoint(
                        self.model, self.optimiser, epoch,
                        val_m, self.ckpt_path
                    )
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.patience:
                        print(f"Early stopping at epoch {epoch}.")
                        break

            mlflow.log_metric("best_val_acc_crystal_system", best_val_acc)
            mlflow.log_artifact(self.ckpt_path)
            print(f"\nTraining complete. Best val acc (crystal_system): {best_val_acc:.4f}")

        return best_val_acc
