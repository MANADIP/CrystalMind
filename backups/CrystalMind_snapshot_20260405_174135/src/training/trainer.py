"""
src/training/trainer.py
Main training engine for CrystalMind.
"""

import time

import mlflow
import torch

from src.training.losses import MultiTaskLoss
from src.training.scheduler import LinearWarmup, build_scheduler
from src.utils.io import save_checkpoint
from src.utils.metrics import cls_metrics, reg_metrics


def _run_epoch(model, loader, criterion, optimiser, device, train: bool, grad_clip: float = 1.0):
    model.train() if train else model.eval()

    total_loss = 0.0
    task_losses = {}
    all_preds = {}
    all_targets = {}
    n_samples = 0

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
                params_to_clip = list(model.parameters()) + [
                    p for p in criterion.parameters() if p.requires_grad
                ]
                torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip)
                optimiser.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

            for task_name, task_loss in per_task.items():
                task_losses[task_name] = task_losses.get(task_name, 0.0) + task_loss * bs

            for task_name, pred in preds.items():
                if task_name not in all_preds:
                    all_preds[task_name] = []
                    all_targets[task_name] = []
                if pred.dim() > 1:
                    all_preds[task_name].append(pred.argmax(1).cpu())
                else:
                    all_preds[task_name].append(pred.detach().cpu())
                all_targets[task_name].append(yb[task_name].cpu())

    for task_name in all_preds:
        all_preds[task_name] = torch.cat(all_preds[task_name]).numpy()
        all_targets[task_name] = torch.cat(all_targets[task_name]).numpy()

    metrics = {"loss": total_loss / n_samples}
    for task_name, task_loss in task_losses.items():
        metrics[f"loss_{task_name}"] = task_loss / n_samples

    for task_name, task_cfg in criterion.task_cfg.items():
        if task_name not in all_preds:
            continue
        if task_cfg["type"] == "classification":
            result = cls_metrics(all_targets[task_name], all_preds[task_name])
            metrics[f"acc_{task_name}"] = result["accuracy"]
            metrics[f"f1_{task_name}"] = result["f1"]
        else:
            result = reg_metrics(all_targets[task_name], all_preds[task_name])
            metrics[f"mae_{task_name}"] = result["mae"]
            metrics[f"r2_{task_name}"] = result["r2"]

    if hasattr(criterion, "get_weight_metrics"):
        metrics.update(criterion.get_weight_metrics())

    return metrics


class Trainer:
    def __init__(self, model, cfg: dict, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        train_cfg = cfg["training"]
        self.epochs = train_cfg["epochs"]
        self.patience = train_cfg.get("early_stopping_patience", 15)
        self.ckpt_path = cfg["paths"]["checkpoint"]
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))

        self.criterion = MultiTaskLoss(
            cfg["tasks"],
            label_smoothing=train_cfg.get("label_smoothing", 0.05),
            multi_task_weighting=train_cfg.get("multi_task_weighting", "static"),
            uncertainty_init_log_var=float(train_cfg.get("uncertainty_init_log_var", 0.0)),
        ).to(device)

        optimiser_params = [{"params": self.model.parameters()}]
        criterion_params = [p for p in self.criterion.parameters() if p.requires_grad]
        if criterion_params:
            optimiser_params.append({"params": criterion_params, "weight_decay": 0.0})

        self.optimiser = torch.optim.AdamW(
            optimiser_params,
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
        best_val_acc = float("-inf")
        patience_ctr = 0

        mlflow.set_experiment(self.cfg["mlflow"]["experiment_name"])
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "epochs": self.epochs,
                    "lr": self.cfg["training"]["learning_rate"],
                    "batch_size": self.cfg["training"]["batch_size"],
                    "scheduler": self.cfg["training"]["scheduler"],
                    "n_params": self.model.count_parameters(),
                }
            )

            for epoch in range(1, self.epochs + 1):
                t0 = time.time()

                self.warmup.step(epoch - 1)

                train_m = _run_epoch(
                    self.model,
                    train_loader,
                    self.criterion,
                    self.optimiser,
                    self.device,
                    train=True,
                    grad_clip=self.grad_clip,
                )
                val_m = _run_epoch(
                    self.model,
                    val_loader,
                    self.criterion,
                    None,
                    self.device,
                    train=False,
                    grad_clip=self.grad_clip,
                )

                if hasattr(self.scheduler, "step"):
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_m["loss"])
                    elif epoch > self.warmup.warmup_epochs:
                        self.scheduler.step()

                log_dict = {}
                for key, value in train_m.items():
                    log_dict[f"train_{key}"] = value
                for key, value in val_m.items():
                    log_dict[f"val_{key}"] = value
                log_dict["lr"] = self.optimiser.param_groups[0]["lr"]
                mlflow.log_metrics(log_dict, step=epoch)

                elapsed = time.time() - t0
                val_acc = val_m.get("acc_crystal_system", 0.0)
                print(
                    f"Ep {epoch:3d}/{self.epochs}  "
                    f"loss={train_m['loss']:.4f}  "
                    f"val_loss={val_m['loss']:.4f}  "
                    f"val_acc(sys)={val_acc:.4f}  "
                    f"({elapsed:.1f}s)"
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_ctr = 0
                    save_checkpoint(
                        self.model,
                        self.optimiser,
                        epoch,
                        val_m,
                        self.ckpt_path,
                        criterion=self.criterion,
                    )
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.patience:
                        print(f"Early stopping at epoch {epoch}.")
                        break

            mlflow.log_metric("best_val_acc_crystal_system", best_val_acc)
            mlflow.log_artifact(self.ckpt_path)
            print(
                "\nTraining complete. "
                f"Best val acc (crystal_system): {best_val_acc:.4f}"
            )

        return best_val_acc
