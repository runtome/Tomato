import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.factory import create_model
from src.utils.device import get_device
from src.utils.checkpoint import save_checkpoint
from src.utils.early_stopping import EarlyStopping
from src.utils.timer import Timer
from src.visualization.plots import plot_loss_curves


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        self.model = create_model(
            config.model_name, config.num_classes, pretrained=True
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        if config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
            )
        elif config.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
            )
        elif config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        self.scheduler = self._build_scheduler()
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and self.device.type == "cuda")

    def _build_scheduler(self):
        cfg = self.config
        params = dict(cfg.scheduler_params or {})
        warmup_epochs = params.pop("warmup_epochs", 0)

        if cfg.scheduler == "CosineAnnealingLR":
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, **params
            )
            if warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.1, total_iters=warmup_epochs
                )
                return torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
                )
            return cosine
        elif cfg.scheduler == "StepLR":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, **params
            )
        elif cfg.scheduler == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **params
            )
        else:
            return None

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        total = 0

        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.config.mixed_precision and self.device.type == "cuda"):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)
            total += images.size(0)

        return running_loss / total

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                with torch.amp.autocast("cuda", enabled=self.config.mixed_precision and self.device.type == "cuda"):
                    outputs = self.model(images)

                # Cast to float32 before loss to prevent float16 overflow
                outputs = outputs.float()
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                total += images.size(0)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return running_loss / total, all_preds, all_labels

    def fit(self, train_loader, val_loader, fold):
        cfg = self.config
        early_stopping = EarlyStopping(patience=cfg.early_stopping_patience)

        save_dir = os.path.join("outputs", "models", cfg.save_name)
        ckpt_path = os.path.join(save_dir, f"{cfg.save_name}_fold_{fold}.pth")
        graph_dir = os.path.join(save_dir, "train_graph")
        graph_path = os.path.join(graph_dir, f"{cfg.save_name}_fold_{fold}.png")

        train_losses = []
        val_losses = []
        epoch_times = []

        print(f"\n{'='*60}")
        print(f"Training {cfg.model_name} — Fold {fold}")
        print(f"{'='*60}")

        best_val_loss = float("inf")

        for epoch in range(1, cfg.epochs + 1):
            with Timer() as t:
                train_loss = self.train_one_epoch(train_loader)
                val_loss, val_preds, val_labels = self.validate(val_loader)

            epoch_times.append(t.elapsed)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(
                f"Epoch {epoch}/{cfg.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Time: {t.elapsed:.1f}s"
            )

            if math.isnan(val_loss) or math.isinf(val_loss):
                print(f"  WARNING: val_loss is {val_loss}, skipping checkpoint & scheduler")
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, ckpt_path)
                print(f"  -> Saved best model (val_loss={val_loss:.4f})")

            if self.scheduler is not None and not (math.isnan(val_loss) or math.isinf(val_loss)):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if early_stopping(val_loss):
                print(f"  Early stopping triggered at epoch {epoch}")
                break

        plot_loss_curves(
            train_losses, val_losses,
            title=f"{cfg.save_name} Fold #{fold}",
            save_path=graph_path,
        )
        print(f"Loss graph saved to {graph_path}")

        total_time = sum(epoch_times)
        avg_epoch_time = total_time / len(epoch_times)
        print(f"Total training time: {total_time:.1f}s | Avg per epoch: {avg_epoch_time:.1f}s")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epoch_times": epoch_times,
            "best_val_loss": best_val_loss,
        }

    def cleanup(self):
        del self.model
        del self.optimizer
        if self.scheduler is not None:
            del self.scheduler
        torch.cuda.empty_cache()
