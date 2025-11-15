"""

 - torch, numpy, pandas, scikit-learn, matplotlib, seaborn, joblib, pyarrow
"""

from pathlib import Path
from datetime import datetime
import json
import math
import time
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ----------------------------- Loss / Aug -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FocalLossLS(nn.Module):
    """
    Focal Loss + Label Smoothing
    """
    def __init__(self, alpha=None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        ce = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            weight=self.alpha,
            label_smoothing=self.label_smoothing if self.label_smoothing is not None else 0.0,
        )
        p_t = torch.exp(-ce)
        loss = ((1 - p_t) ** self.gamma) * ce
        return loss.mean()

class WarmupCosineScheduler:
    def __init__(self, optimizer, base_lr, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_epochs = int(warmup_epochs)
        self.total_epochs = int(total_epochs)
        self.min_lr = float(min_lr)

    def step(self, epoch: int):
        if epoch < self.warmup_epochs and self.warmup_epochs > 0:
            lr = self.base_lr * float(epoch + 1) / float(self.warmup_epochs)
        else:
            t = (epoch - self.warmup_epochs) / max(1, (self.total_epochs - self.warmup_epochs))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * min(1.0, t)))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

# ----------------------------- Data -----------------------------
class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None):
        self.X = torch.from_numpy(X).float()
        self.y = None if y is None else torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]

# ----------------------------- Model core -----------------------------
class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight + self.bias

class FTTransformer(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 3,
                 d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)  # (B, F, D)
        B = tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        enc = self.encoder(seq)
        cls_out = self.norm(enc[:, 0, :])
        logits = self.head(cls_out)
        return logits

# ----------------------------- Trainer -----------------------------
class FTTransformerCOClassifier:
    def __init__(self,
                 horizon: int = 24,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 batch_size: int = 512,
                 accumulation_steps: int = 4,
                 lr: float = 5e-4,
                 weight_decay: float = 5e-3,
                 max_epochs: int = 120,
                 warmup_epochs: int = 8,
                 patience: int = 20,
                 mixup_alpha: float = 0.4,
                 mixup_prob: float = 0.5,
                 focal_gamma: float = 2.0,
                 label_smoothing: float | None = None,
                 seed: int = 42,
                 device: str | None = None,
                 num_workers: int = 0,
                 checkpoint_dir: str | Path | None = None,
                 ):
        self.horizon = int(horizon)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.accumulation_steps = int(accumulation_steps)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.patience = int(patience)
        self.mixup_alpha = float(mixup_alpha)
        self.mixup_prob = float(mixup_prob)
        self.focal_gamma = float(focal_gamma)
        self.label_smoothing = label_smoothing
        self.seed = int(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_workers = int(num_workers)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = None
        self.model = None
        self.results = {}
        self.feature_names = None
        self.history = None
        self.label_mapping = {"low": 0, "mid": 1, "high": 2}
        self.rev_mapping = {v: k for k, v in self.label_mapping.items()}

        set_seed(self.seed)

    def _discretize_naive(self, s: pd.Series) -> pd.Series:
        return pd.cut(s, bins=[-np.inf, 1.5, 2.5, np.inf], labels=["low", "mid", "high"])

    def load_data(self, data_path: str | Path):
        base_path = Path(data_path) / f"h{self.horizon}"
        print(f"\n{'='*64}\nLoading data for horizon h+{self.horizon}\n{'='*64}")
        t0 = time.time()

        train = pd.read_parquet(base_path / "train.parquet")
        valid = pd.read_parquet(base_path / "valid.parquet")
        test = pd.read_parquet(base_path / "test.parquet")

        target_col = f"co_level_t+{self.horizon}"
        drop_cols = [f"y_t+{self.horizon}", f"naive_yhat_t+{self.horizon}", target_col]

        X_train = train.drop(columns=drop_cols, errors="ignore")
        X_valid = valid.drop(columns=drop_cols, errors="ignore")
        X_test = test.drop(columns=drop_cols, errors="ignore")
        y_train = train[target_col]
        y_valid = valid[target_col]
        y_test = test[target_col]

        self.feature_names = X_train.columns.tolist()
        print(f"Features: {len(self.feature_names)}")

        self.scaler = StandardScaler()
        X_train_np = self.scaler.fit_transform(X_train.values.astype(np.float32))
        X_valid_np = self.scaler.transform(X_valid.values.astype(np.float32))
        X_test_np = self.scaler.transform(X_test.values.astype(np.float32))

        y_train_enc = y_train.map(self.label_mapping).values.astype(np.int64)
        y_valid_enc = y_valid.map(self.label_mapping).values.astype(np.int64)
        y_test_enc = y_test.map(self.label_mapping).values.astype(np.int64)

        self.train_ds = TabularDataset(X_train_np, y_train_enc)
        self.valid_ds = TabularDataset(X_valid_np, y_valid_enc)
        self.test_ds = TabularDataset(X_test_np, y_test_enc)

        if self.device.type == "cuda":
            pin_memory = True
            torch.backends.cudnn.benchmark = True
        else:
            pin_memory = False
            try:
                torch.set_num_threads(1)
            except Exception:
                pass

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
            drop_last=False,
        )
        self.valid_loader = DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            drop_last=False,
        )
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            drop_last=False,
        )

        naive_col = f"naive_yhat_t+{self.horizon}"
        if naive_col in train.columns:
            self.naive_train = self._discretize_naive(train[naive_col])
            self.naive_valid = self._discretize_naive(valid[naive_col])
            self.naive_test = self._discretize_naive(test[naive_col])
        else:
            self.naive_train = None
            self.naive_valid = None
            self.naive_test = None

        print(f"✓ Data loaded: Train={X_train_np.shape}, Valid={X_valid_np.shape}, Test={X_test_np.shape} | Time: {time.time() - t0:.2f}s")
        print(f"Device: {self.device} | num_workers=0 | pin_memory={pin_memory}")

        print("\nClass distribution (train):")
        cls_dist = pd.Series(y_train_enc).value_counts(normalize=True).sort_index()
        for cls, prop in cls_dist.items():
            print(f"  {self.rev_mapping[int(cls)]}: {prop:.3f}")

        print("\n[Building FT-Transformer ...]")
        self.model = FTTransformer(
            n_features=X_train_np.shape[1],
            n_classes=3,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Model params: {total_params:,}")
        print(f"  Effective batch size: {self.batch_size * self.accumulation_steps}")
        print(f"  Device: {self.device}")

        cls_counts = pd.Series(y_train_enc).value_counts().sort_index()
        weights = cls_counts.sum() / (len(cls_counts) * cls_counts)
        self.class_weights = torch.tensor(weights.values, dtype=torch.float32, device=self.device)

        return self

    def train(self):
        print(f"\n{'='*64}\nTraining FT-Transformer h+{self.horizon}\n{'='*64}")
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            base_lr=self.lr,
            warmup_epochs=self.warmup_epochs,
            total_epochs=self.max_epochs,
            min_lr=1e-6,
        )
        criterion = FocalLossLS(
            alpha=self.class_weights,
            gamma=self.focal_gamma,
            label_smoothing=self.label_smoothing if self.label_smoothing is not None else 0.0,
        )

        scaler = GradScaler() if self.device.type == "cuda" else None

        best_f1 = -math.inf
        best_state = None
        patience_left = self.patience
        history = {"train_loss": [], "val_f1": [], "lr": []}
        last_val_f1 = 0.0

        start_time = time.time()
        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()
            lr_now = scheduler.step(epoch - 1)
            history['lr'].append(lr_now)

            self.model.train()
            running_loss = 0.0
            optimizer.zero_grad()
            n_samples = 0

            for step, (xb, yb) in enumerate(self.train_loader):
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                use_mix = (self.mixup_alpha > 0 and np.random.rand() < self.mixup_prob)
                if use_mix:
                    if self.mixup_alpha > 0:
                        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    else:
                        lam = 1.0
                    batch_size = xb.size(0)
                    index = torch.randperm(batch_size, device=xb.device)
                    xb_m = lam * xb + (1 - lam) * xb[index]
                    y_a, y_b = yb, yb[index]
                    xb_input = xb_m
                else:
                    xb_input = xb

                if scaler is not None:
                    with autocast():
                        logits = self.model(xb_input)
                        if use_mix:
                            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                        else:
                            loss = criterion(logits, yb)
                        loss = loss / self.accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    logits = self.model(xb_input)
                    if use_mix:
                        loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                    else:
                        loss = criterion(logits, yb)
                    loss = loss / self.accumulation_steps
                    loss.backward()

                running_loss += float(loss.item()) * xb.size(0) * self.accumulation_steps
                n_samples += xb.size(0)

                if (step + 1) % self.accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad()

            avg_loss = running_loss / max(1, n_samples)
            history['train_loss'].append(avg_loss)

            should_validate = (epoch <= 10) or (epoch % 5 == 0) or (epoch == self.max_epochs)

            if should_validate:
                val_metrics = self._eval_loader(self.valid_loader)
                val_f1 = val_metrics['f1_macro']
                last_val_f1 = val_f1
            else:
                val_f1 = last_val_f1

            history['val_f1'].append(val_f1)

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch:3d}/{self.max_epochs} | Loss: {avg_loss:.6f} | Val F1: {val_f1:.4f} | LR: {lr_now:.2e} | Time: {epoch_time:.1f}s", end="")

            if should_validate and val_f1 > best_f1 + 1e-6:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_left = self.patience
                ckpt = {"state_dict": best_state, "epoch": epoch, "val_f1": float(best_f1)}
                try:
                    torch.save(ckpt, self.checkpoint_dir / f"ftt_best_h{self.horizon}.pt")
                except Exception:
                    pass
                print(" ✓ NEW BEST")
            else:
                if should_validate:
                    patience_left -= 1
                print(f" (patience {patience_left}/{self.patience})")
                if patience_left <= 0:
                    print(f"\n⚠ Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.training_time_seconds = time.time() - start_time
        self.history = history
        print(f"\n✓ Training finished in {self.training_time_seconds/60:.2f} min | Best Val F1: {best_f1:.4f}")
        return self

    @torch.no_grad()
    def _eval_loader(self, loader: DataLoader):
        self.model.eval()
        all_logits = []
        all_y = []
        for batch in loader:
            xb, yb = batch
            xb = xb.to(self.device, non_blocking=(self.device.type == "cuda"))
            logits = self.model(xb)
            all_logits.append(logits.cpu())
            all_y.append(yb)
        all_logits = torch.cat(all_logits, dim=0)
        all_y = torch.cat(all_y, dim=0).numpy()
        probs = torch.softmax(all_logits, dim=1).numpy()
        preds = probs.argmax(axis=1)
        acc = accuracy_score(all_y, preds)
        f1m = f1_score(all_y, preds, average="macro")
        f1w = f1_score(all_y, preds, average="weighted")
        precision, recall, f1_per, support = precision_recall_fscore_support(all_y, preds, average=None, labels=[0,1,2])
        return {
            "accuracy": acc,
            "f1_macro": f1m,
            "f1_weighted": f1w,
            "precision": precision,
            "recall": recall,
            "f1_per_class": f1_per,
            "support": support,
            "pred": preds,
            "prob": probs,
        }

    def evaluate_all(self):
        print(f"\n{'='*64}\nEVALUATION (final)\n{'='*64}")
        train_res = self._eval_loader(self.train_loader)
        valid_res = self._eval_loader(self.valid_loader)
        test_res = self._eval_loader(self.test_loader)
        self.results = {"train": train_res, "valid": valid_res, "test": test_res}

        def decode(arr):
            return pd.Series(arr).map(self.rev_mapping)

        def report(name, res, naive_series, y_true_enc):
            y_true = decode(y_true_enc)
            y_pred = decode(res['pred'])
            print(f"\n{name.upper()}: Acc={res['accuracy']:.4f}, F1_macro={res['f1_macro']:.4f}, F1_wtd={res['f1_weighted']:.4f}")
            if naive_series is not None:
                naive_acc = accuracy_score(y_true, naive_series)
                naive_f1 = f1_score(y_true, naive_series, average="macro")
                print(f"  Naive Acc {naive_acc:.4f} (Δ {res['accuracy']-naive_acc:+.4f}), Naive F1 {naive_f1:.4f} (Δ {res['f1_macro']-naive_f1:+.4f})")
            print("  Per-class:")
            for i, lab in enumerate(['low','mid','high']):
                print(f"    {lab:>4}: P={res['precision'][i]:.3f} R={res['recall'][i]:.3f} F1={res['f1_per_class'][i]:.3f} (n={int(res['support'][i])})")

        report("train", train_res, self.naive_train, self.train_ds.y.numpy())
        report("valid", valid_res, self.naive_valid, self.valid_ds.y.numpy())
        report("test", test_res, self.naive_test, self.test_ds.y.numpy())
        return self

    def plot_results(self, save_path=None):
        """综合图：loss / val_f1 / lr + 混淆矩阵"""
        if getattr(self, "history", None) is None:
            print("[WARN] no history to plot in plot_results.")
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 4)

        ax_loss = fig.add_subplot(gs[0, 0])
        if self.history is not None:
            ax_loss.plot(self.history['train_loss'], label='train loss')
        ax_loss.set_title('Train Loss')
        ax_loss.set_xlabel('epoch')

        ax_f1 = fig.add_subplot(gs[0, 1])
        if self.history is not None:
            ax_f1.plot(self.history['val_f1'], label='val f1', color='orange')
        ax_f1.set_title('Val F1')

        ax_lr = fig.add_subplot(gs[0, 2])
        if self.history is not None:
            ax_lr.plot(self.history['lr'])
        ax_lr.set_title('LR schedule')
        ax_lr.set_yscale('log')

        names = ['train','valid','test']
        positions = [(1,0),(1,1),(1,2)]
        for (name, pos) in zip(names, positions):
            ax = fig.add_subplot(gs[pos[0], pos[1]])
            cm = confusion_matrix(
                getattr(self, f"{name}_ds").y.numpy(),
                self.results[name]['pred']
            )
            sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                        xticklabels=['low','mid','high'], yticklabels=['low','mid','high'])
            ax.set_title(f"{name} Acc {self.results[name]['accuracy']:.3f} F1 {self.results[name]['f1_macro']:.3f}")

        plt.suptitle(f"FT-Transformer Effect-First (h+{self.horizon})")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved results to {save_path}")
        plt.show()

    def plot_training_history(self, save_path=None):
        """单独绘制训练曲线：train loss / val f1 / lr"""
        if getattr(self, "history", None) is None:
            print("[WARN] No training history available to plot.")
            return
        hist = self.history
        epochs = list(range(1, len(hist['train_loss']) + 1))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(epochs, hist['train_loss'], marker='o')
        axes[0].set_title('Train Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, hist['val_f1'], marker='o')
        axes[1].set_title('Validation F1 (macro)')
        axes[1].set_xlabel('Epoch')
        axes[1].grid(alpha=0.3)

        axes[2].plot(epochs, hist['lr'], marker='o')
        axes[2].set_title('Learning Rate')
        axes[2].set_xlabel('Epoch')
        axes[2].set_yscale('log')
        axes[2].grid(alpha=0.3)

        plt.suptitle(f"FT-Transformer Training History (h+{self.horizon})", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        plt.show()

    def plot_confusion_matrices(self, save_path=None):
        """单独画 train/valid/test 混淆矩阵，方便和 DeepGBM 对齐"""
        if not self.results:
            print("[WARN] No results to plot confusion matrices.")
            return
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        names = ["train", "valid", "test"]
        ds_map = {"train": self.train_ds, "valid": self.valid_ds, "test": self.test_ds}
        for idx, name in enumerate(names):
            y_true = ds_map[name].y.numpy()
            y_pred = self.results[name]["pred"]
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["low", "mid", "high"],
                yticklabels=["low", "mid", "high"],
                ax=axes[idx],
            )
            axes[idx].set_title(
                f"{name.capitalize()} Set\nAcc: {self.results[name]['accuracy']:.3f}, F1: {self.results[name]['f1_macro']:.3f}"
            )
            axes[idx].set_ylabel("True Label")
            axes[idx].set_xlabel("Predicted Label")
        plt.suptitle(f"FT-Transformer - Confusion Matrices (h+{self.horizon})", fontsize=16)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrices saved to: {save_path}")
        plt.show()

    def save_model(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            "state_dict": self.model.state_dict(),
            "params": {
                "n_features": len(self.feature_names),
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout,
                "horizon": self.horizon,
            },
            "history": getattr(self, "history", None)
        }, output_dir / f"ftt_effect_h{self.horizon}.pt")

        joblib.dump(self.scaler, output_dir / f"ftt_scaler_h{self.horizon}.joblib")

        def _pack_results(res_dict):
            out = {}
            for ds, res in res_dict.items():
                out[ds] = {
                    "accuracy": float(res["accuracy"]),
                    "f1_macro": float(res["f1_macro"]),
                    "f1_weighted": float(res["f1_weighted"]),
                    "per_class_f1": {
                        lab: float(f1) for lab, f1 in zip(["low", "mid", "high"], res["f1_per_class"])
                    }
                }
            return out

        summary = {
            "horizon": self.horizon,
            "model_type": "ft_transformer",
            "timestamp": datetime.now().isoformat(),
            "training_time_seconds": float(getattr(self, "training_time_seconds", 0.0)),
            "params": {
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout,
                "batch_size": self.batch_size,
                "accumulation_steps": self.accumulation_steps,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "max_epochs": self.max_epochs,
                "warmup_epochs": self.warmup_epochs,
                "patience": self.patience,
                "mixup_alpha": self.mixup_alpha,
                "mixup_prob": self.mixup_prob,
                "focal_gamma": self.focal_gamma,
                "label_smoothing": self.label_smoothing,
                "seed": self.seed,
            },
            "results": _pack_results(self.results),
            "history": self.history if getattr(self, "history", None) is not None else None,
        }

        with open(output_dir / f"ftt_effect_results_h{self.horizon}.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Model & results saved to: {output_dir}")
        return self

# ----------------------------- main -----------------------------
def main():
    DATA_PATH = Path("/app/data_artifacts/splits")
    OUTPUT_DIR = Path("/app/classification-analysis/ft_transformer_effect_first_unified")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    horizons = [1, 6, 12, 24]

    for h in horizons:
        print("\n" + "#" * 80)
        print(f"# HORIZON: {h} HOUR(S)")
        print("#" * 80)

        
        if h == 1:
            clf = FTTransformerCOClassifier(
                horizon=h,
                d_model=256,
                nhead=8,
                num_layers=4,
                dim_feedforward=1024,
                dropout=0.15,            
                batch_size=128,          
                accumulation_steps=1,
                lr=7e-4,
                weight_decay=5e-4,       
                max_epochs=120,
                warmup_epochs=8,
                patience=25,
                mixup_alpha=0.15,       
                mixup_prob=0.3,
                focal_gamma=1.2,         
                label_smoothing=0.03,    
                seed=42,
                device=None,
                num_workers=0,
                checkpoint_dir=OUTPUT_DIR / "checkpoints",
            )
        elif h == 6:
            clf = FTTransformerCOClassifier(
                horizon=h,
                d_model=256,
                nhead=8,
                num_layers=4,
                dim_feedforward=1024,
                dropout=0.18,
                batch_size=128,
                accumulation_steps=1,
                lr=4e-4,
                weight_decay=6e-4,
                max_epochs=140,
                warmup_epochs=10,
                patience=22,
                mixup_alpha=0.15,
                mixup_prob=0.3,
                focal_gamma=1.2,
                label_smoothing=0.03,
                seed=42,
                device=None,
                num_workers=0,
                checkpoint_dir=OUTPUT_DIR / "checkpoints",
            )
        elif h == 12:
            clf = FTTransformerCOClassifier(
                horizon=h,
                d_model=320,
                nhead=8,
                num_layers=6,
                dim_feedforward=1536,
                dropout=0.18,
                batch_size=96,           
                accumulation_steps=1,
                lr=4e-4,
                weight_decay=6e-4,
                max_epochs=160,
                warmup_epochs=12,
                patience=25,
                mixup_alpha=0.12,
                mixup_prob=0.25,
                focal_gamma=1.1,
                label_smoothing=0.025,
                seed=42,
                device=None,
                num_workers=0,
                checkpoint_dir=OUTPUT_DIR / "checkpoints",
            )
        else:  # h == 24
            clf = FTTransformerCOClassifier(
                horizon=h,
                d_model=288,
                nhead=8,
                num_layers=5,
                dim_feedforward=1408,
                dropout=0.2,
                batch_size=96,
                accumulation_steps=1,
                lr=4e-4,
                weight_decay=7e-4,
                max_epochs=160,
                warmup_epochs=10,
                patience=25,
                mixup_alpha=0.12,
                mixup_prob=0.3,
                focal_gamma=1.2,
                label_smoothing=0.03,
                seed=42,
                device=None,
                num_workers=0,
                checkpoint_dir=OUTPUT_DIR / "checkpoints",
            )

        clf.load_data(DATA_PATH)
        clf.train()
        clf.evaluate_all()

        out_dir = OUTPUT_DIR / f"h{h}"
        out_dir.mkdir(parents=True, exist_ok=True)
        clf.plot_results(save_path=out_dir / f"results_h{h}.png")
        clf.plot_training_history(save_path=out_dir / f"training_history_h{h}.png")
        clf.plot_confusion_matrices(save_path=out_dir / f"confusion_matrices_h{h}.png")
        clf.save_model(out_dir)

        print("\n" + "=" * 80)
        print(f"✓ Completed horizon h+{h}")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    main()