import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.factory import create_model
from src.utils.device import get_device
from src.utils.checkpoint import load_checkpoint
from src.metrics.evaluation import (
    compute_metrics,
    compute_confusion_matrix,
    aggregate_fold_metrics,
)
from src.visualization.plots import plot_confusion_matrix
from src.constants.labels import CLASS_NAMES, NUM_CLASSES


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.device = get_device()

    def _load_model(self, fold):
        model = create_model(self.config.model_name, self.config.num_classes, pretrained=False)
        ckpt_path = os.path.join(
            "outputs", "models", self.config.save_name,
            f"{self.config.save_name}_fold_{fold}.pth",
        )
        load_checkpoint(model, ckpt_path, device=self.device)
        model = model.to(self.device)
        model.eval()
        return model

    def evaluate_loader(self, model, loader):
        all_preds = []
        all_labels = []
        per_image_times = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
                images = images.to(self.device)
                batch_size = images.size(0)

                # Warmup: skip first batch timing (GPU kernel init)
                if batch_idx == 0:
                    with torch.amp.autocast("cuda", enabled=self.config.mixed_precision and self.device.type == "cuda"):
                        outputs = model(images)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    outputs = outputs.float()
                else:
                    start = time.perf_counter()
                    with torch.amp.autocast("cuda", enabled=self.config.mixed_precision and self.device.type == "cuda"):
                        outputs = model(images)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    outputs = outputs.float()

                    # Per-image time
                    per_image_times.extend([elapsed / batch_size] * batch_size)

                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        metrics = compute_metrics(all_labels, all_preds)
        cm = compute_confusion_matrix(all_labels, all_preds, NUM_CLASSES)
        return metrics, cm, per_image_times

    def evaluate_fold(self, fold, test_loader, train_loader=None, val_loader=None):
        print(f"\n--- Evaluating Fold {fold} ---")
        model = self._load_model(fold)

        results = {}

        if train_loader is not None:
            metrics, _, _ = self.evaluate_loader(model, train_loader)
            results["train"] = metrics
            print(f"  Train — Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

        if val_loader is not None:
            metrics, _, _ = self.evaluate_loader(model, val_loader)
            results["val"] = metrics
            print(f"  Val   — Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

        metrics, cm, inf_times = self.evaluate_loader(model, test_loader)
        results["test"] = metrics
        results["confusion_matrix"] = cm
        results["inference_times"] = inf_times
        print(f"  Test  — Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
        if inf_times:
            avg_t = np.mean(inf_times) * 1000
            std_t = np.std(inf_times, ddof=1) * 1000
            print(f"  Test  — Inference time (per image): {avg_t:.4f} ± {std_t:.4f} ms")

        # Save confusion matrix
        save_dir = os.path.join("outputs", "models", self.config.save_name, "confusion-matrix")
        cm_path = os.path.join(save_dir, f"{self.config.save_name}_fold_{fold}.png")
        plot_confusion_matrix(
            cm, CLASS_NAMES,
            title=f"{self.config.save_name} Fold #{fold}",
            save_path=cm_path,
        )
        print(f"  Confusion matrix saved to {cm_path}")

        # Cleanup
        del model
        torch.cuda.empty_cache()

        return results

    def evaluate_all_folds(self, folds, test_loader, fold_loaders=None):
        all_fold_metrics = []
        all_inference_times = []

        for fold in folds:
            train_loader = fold_loaders[fold]["train"] if fold_loaders else None
            val_loader = fold_loaders[fold]["val"] if fold_loaders else None
            results = self.evaluate_fold(fold, test_loader, train_loader, val_loader)
            all_fold_metrics.append(results["test"])
            all_inference_times.extend(results["inference_times"])

        aggregated = aggregate_fold_metrics(all_fold_metrics)

        print(f"\n{'='*60}")
        print(f"Aggregated Test Results ({len(folds)} folds)")
        print(f"{'='*60}")
        for metric, vals in aggregated.items():
            print(f"  {metric}: {vals['mean']:.4f} ± {vals['std']:.4f}")

        avg_inf = np.mean(all_inference_times)
        std_inf = np.std(all_inference_times, ddof=1)
        print(f"  Inference time (per image): {avg_inf*1000:.4f} ± {std_inf*1000:.4f} ms")

        return aggregated
