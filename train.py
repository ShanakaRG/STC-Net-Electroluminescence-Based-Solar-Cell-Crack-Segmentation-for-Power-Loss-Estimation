import argparse
from typing import Dict, List

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import make_datasets
from losses import CombinedSolarLoss
from metrics import compute_metrics
from models.build import build_model
from utils.common import (
    CSVLogger,
    count_parameters,
    ensure_dir,
    flexible_load_model,
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_config,
    seed_everything,
    state_dict_for_saving,
)


def make_loader(dataset, cfg: dict, train: bool):
    loader_cfg = cfg["loader"]
    return DataLoader(
        dataset,
        batch_size=int(loader_cfg["batch_size"]),
        shuffle=train,
        num_workers=int(loader_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out




def _threshold_candidates(cfg: dict):
    eval_cfg = cfg.get("eval", {})
    vals = eval_cfg.get("threshold_candidates", None)
    if vals is None:
        return [float(eval_cfg.get("threshold", 0.5))]
    return [float(x) for x in vals]


def _best_threshold_on_loader(model, loader, device, num_classes, candidates):
    if num_classes > 1:
        return float(candidates[0])
    scores = {float(t): 0.0 for t in candidates}
    n_batches = max(1, len(loader))
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["image"])
            logits = outputs["logits"]
            for t in candidates:
                stats = compute_metrics(logits, batch["mask"], num_classes=num_classes, threshold=float(t))
                scores[float(t)] += stats["iou"]
    best_t = sorted(scores.items(), key=lambda kv: kv[1] / n_batches, reverse=True)[0][0]
    return float(best_t)

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, cfg, epoch: int):
    model.train()
    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(epoch)
    num_classes = int(cfg["model"].get("num_classes", 1))
    # threshold = float(threshold_override if threshold_override is not None else cfg["eval"].get("threshold", 0.5))
    threshold = float(cfg["eval"].get("threshold", 0.5))
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"

    running = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    pbar = tqdm(loader, desc="Train {}".format(epoch), leave=False)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            outputs = model(batch["image"])
            loss, details = criterion(outputs, batch)

        scaler.scale(loss).backward()
        grad_clip = cfg["train"].get("grad_clip", None)
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        scaler.step(optimizer)
        scaler.update()

        stats = compute_metrics(outputs["logits"].detach(), batch["mask"], num_classes=num_classes, threshold=threshold)
        running["loss"] += float(loss.detach().item())
        for k in ["dice", "iou", "precision", "recall"]:
            running[k] += stats[k]
        pbar.set_postfix(loss="{:.4f}".format(loss.item()), dice="{:.4f}".format(stats["dice"]), iou="{:.4f}".format(stats["iou"]))

    n_batches = max(1, len(loader))
    return {k: v / n_batches for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, cfg, epoch: int, split_name: str = "Val", threshold_override=None):
    model.eval()
    num_classes = int(cfg["model"].get("num_classes", 1))
    threshold = float(threshold_override if threshold_override is not None else cfg["eval"].get("threshold", 0.5))

    running = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    pbar = tqdm(loader, desc="{} {}".format(split_name, epoch), leave=False)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["image"])
        loss, details = criterion(outputs, batch)
        stats = compute_metrics(outputs["logits"], batch["mask"], num_classes=num_classes, threshold=threshold)
        running["loss"] += float(loss.item())
        for k in ["dice", "iou", "precision", "recall"]:
            running[k] += stats[k]
        pbar.set_postfix(loss="{:.4f}".format(loss.item()), dice="{:.4f}".format(stats["dice"]), iou="{:.4f}".format(stats["iou"]))

    n_batches = max(1, len(loader))
    return {k: v / n_batches for k, v in running.items()}


def main():
    parser = argparse.ArgumentParser(description="Train solar-cell defect segmentation network")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["train"].get("seed", 42)))

    out_dir = ensure_dir(cfg["train"]["output_dir"])
    ensure_dir(out_dir / "checkpoints")
    save_config(cfg, out_dir / "resolved_config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device: {}".format(device))
    print("[INFO] Visible GPU count: {}".format(torch.cuda.device_count() if torch.cuda.is_available() else 0))

    train_ds, val_ds, test_ds = make_datasets(cfg)
    train_loader = make_loader(train_ds, cfg, train=True)
    val_loader = make_loader(val_ds, cfg, train=False)
    test_loader = make_loader(test_ds, cfg, train=False) if test_ds is not None else None

    model = build_model(cfg).to(device)
    use_dp = bool(cfg["train"].get("data_parallel", True)) and device.type == "cuda" and torch.cuda.device_count() > 1
    if use_dp:
        model = torch.nn.DataParallel(model)
        print("[INFO] Using DataParallel on {} GPUs".format(torch.cuda.device_count()))

    print("[INFO] Trainable params: {:.2f} M".format(count_parameters(model) / 1e6))

    criterion = CombinedSolarLoss(cfg)
    current_threshold = float(cfg["eval"].get("threshold", 0.5))
    auto_threshold = bool(cfg.get("eval", {}).get("auto_threshold", True)) and int(cfg["model"].get("num_classes", 1)) <= 1
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"].get("weight_decay", 1e-4)),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg["train"]["epochs"]), eta_min=float(cfg["optim"].get("min_lr", 1e-6)))
    scaler = GradScaler(enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")

    start_epoch = 1
    best_metric = -1.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device.type)
        flexible_load_model(model, ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt.get("best_metric", -1.0)
        print("[INFO] Resumed from epoch {} with best metric {:.4f}".format(ckpt["epoch"], best_metric))

    log_fields = [
        "epoch", "lr", "threshold",
        "train_loss", "train_dice", "train_iou", "train_precision", "train_recall",
        "val_loss", "val_dice", "val_iou", "val_precision", "val_recall",
    ]
    if test_loader is not None:
        log_fields += ["test_loss", "test_dice", "test_iou", "test_precision", "test_recall"]
    csv_logger = CSVLogger(out_dir / "metrics.csv", log_fields)

    patience = int(cfg["train"].get("early_stopping_patience", 0))
    stale_epochs = 0

    for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, cfg, epoch)
        if auto_threshold:
            current_threshold = _best_threshold_on_loader(model, val_loader, device, int(cfg["model"].get("num_classes", 1)), _threshold_candidates(cfg))
        val_metrics = validate(model, val_loader, criterion, device, cfg, epoch, split_name="Val", threshold_override=current_threshold)
        test_metrics = None
        if test_loader is not None:
            test_metrics = validate(model, test_loader, criterion, device, cfg, epoch, split_name="Test", threshold_override=current_threshold)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        score_name = cfg["eval"].get("selection_metric", "iou")
        score = val_metrics[score_name]
        is_best = score > best_metric
        state = {
            "epoch": epoch,
            "model": state_dict_for_saving(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_metric": best_metric if not is_best else score,
            "config": cfg,
            "best_threshold": current_threshold,
        }
        if is_best:
            best_metric = score
            stale_epochs = 0
            save_checkpoint(state, out_dir / "checkpoints" / "best.pt")
            print("[INFO] Saved new best model at epoch {} with {}={:.4f}".format(epoch, score_name, score))
        else:
            stale_epochs += 1

        state["best_metric"] = best_metric
        save_checkpoint(state, out_dir / "checkpoints" / "last.pt")

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "threshold": current_threshold,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        if test_metrics is not None:
            row.update(
                {
                    "test_loss": test_metrics["loss"],
                    "test_dice": test_metrics["dice"],
                    "test_iou": test_metrics["iou"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                }
            )
        csv_logger.log(row)

        msg = (
            "Epoch [{}/{}] | Train Loss: {:.4f} | Train IoU: {:.4f} | Val Loss: {:.4f} | Val Dice: {:.4f} | Val IoU: {:.4f}".format(
                epoch,
                cfg["train"]["epochs"],
                train_metrics["loss"],
                train_metrics["iou"],
                val_metrics["loss"],
                val_metrics["dice"],
                val_metrics["iou"],
            )
        )
        if test_metrics is not None:
            msg += " | Test IoU: {:.4f}".format(test_metrics["iou"])
        print(msg)

        if patience > 0 and stale_epochs >= patience:
            print("[INFO] Early stopping triggered after {} stale epochs.".format(stale_epochs))
            break

    print("[INFO] Training complete. Best validation {}: {:.4f}".format(cfg["eval"].get("selection_metric", "iou"), best_metric))


if __name__ == "__main__":
    main()
