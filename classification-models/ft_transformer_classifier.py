"""
FT-Transformer Classifier for CO Pollution Level Prediction
Predicts CO concentration levels (low, mid, high) at different time horizons.

This implementation treats each numerical feature as a token (scalar -> d_model via per-feature linear mapping),
adds a learnable CLS token, passes through Transformer encoder layers, and classifies via CLS representation.

Dependencies: torch, numpy, pandas, scikit-learn, matplotlib, seaborn
"""

from pathlib import Path
from datetime import datetime
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except Exception:  # fallback if tqdm not installed
    tqdm = None

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


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


class FeatureTokenizer(nn.Module):
    """Numeric feature tokenizer: each feature is mapped to a d_model token via affine transform."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) -> tokens: (B, F, D)
        # broadcast multiply per feature: x[..., None] * weight + bias
        return x.unsqueeze(-1) * self.weight + self.bias


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,F)
        tok = self.tokenizer(x)  # (B,F,D)
        B = tok.size(0)
        cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        seq = torch.cat([cls, tok], dim=1)  # (B,1+F,D)
        enc = self.encoder(seq)  # (B,1+F,D)
        cls_out = self.norm(enc[:, 0, :])
        logits = self.head(cls_out)
        return logits


class FTTransformerCOClassifier:
    def __init__(
        self,
        horizon: int = 24,
        d_model: int = 96,  # 减小模型维度 (原 128) - CPU 友好
        nhead: int = 6,  # 减少注意力头数 (原 8) - 必须能整除 d_model
        num_layers: int = 3,  # 减少层数 (原 4) - 更快训练
        dim_feedforward: int = 192,  # 减小前馈层 (原 256) - 通常是 d_model 的 2x
        dropout: float = 0.15,  # 稍微增加 dropout (原 0.1) - 防止小 batch 过拟合
        batch_size: int = 32,  # CPU 推荐 batch size (原 512)
        lr: float = 5e-4,  # 降低学习率 (原 1e-3) - 适应小 batch
        weight_decay: float = 1e-3,  # 增加正则化 (原 1e-4) - 防止过拟合
        max_epochs: int = 60,  # 增加 epochs (原 50) - 补偿小 batch 的慢收敛
        patience: int = 10,  # 增加 patience (原 8) - 给更多时间收敛
        seed: int = 42,
        device: str | None = None,
    ):
        self.horizon = horizon
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = None
        self.model = None
        self.results = {}
        self.feature_names = None

        self.label_mapping = {"low": 0, "mid": 1, "high": 2}
        self.rev_mapping = {v: k for k, v in self.label_mapping.items()}

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    # ---------- Data ----------
    def _discretize_naive(self, s: pd.Series) -> pd.Series:
        return pd.cut(s, bins=[-np.inf, 1.5, 2.5, np.inf], labels=["low", "mid", "high"])

    def load_data(self, data_path: str | Path):
        base_path = Path(data_path) / f"h{self.horizon}"
        print(f"\n{'='*60}\nLoading data for horizon h+{self.horizon}\n{'='*60}")
        
        t_start = time.time()
        print("[1/6] Reading parquet files...")
        train = pd.read_parquet(base_path / "train.parquet")
        valid = pd.read_parquet(base_path / "valid.parquet")
        test = pd.read_parquet(base_path / "test.parquet")
        print(f"      ✓ Loaded in {time.time() - t_start:.2f}s")

        target_col = f"co_level_t+{self.horizon}"
        drop_cols = [f"y_t+{self.horizon}", f"naive_yhat_t+{self.horizon}", target_col]

        print("[2/6] Preparing features and targets...")
        X_train = train.drop(columns=drop_cols)
        y_train = train[target_col]
        X_valid = valid.drop(columns=drop_cols)
        y_valid = valid[target_col]
        X_test = test.drop(columns=drop_cols)
        y_test = test[target_col]

        self.feature_names = X_train.columns.tolist()
        print(f"      ✓ Features: {len(self.feature_names)}")

        # scaler
        print("[3/6] Scaling features with StandardScaler...")
        t_scale = time.time()
        self.scaler = StandardScaler()
        X_train_np = self.scaler.fit_transform(X_train.values.astype(np.float32))
        X_valid_np = self.scaler.transform(X_valid.values.astype(np.float32))
        X_test_np = self.scaler.transform(X_test.values.astype(np.float32))
        print(f"      ✓ Scaled in {time.time() - t_scale:.2f}s")

        print("[4/6] Encoding labels...")
        # Convert to numpy array to avoid Categorical type issue
        y_train_enc = y_train.map(self.label_mapping).values.astype(np.int64)
        y_valid_enc = y_valid.map(self.label_mapping).values.astype(np.int64)
        y_test_enc = y_test.map(self.label_mapping).values.astype(np.int64)

        # Store for later use
        self.train_y = y_train_enc
        self.valid_y = y_valid_enc
        self.test_y = y_test_enc

        # naive baseline
        print("[5/6] Preparing naive baseline...")
        naive_col = f"naive_yhat_t+{self.horizon}"
        if naive_col in train.columns and naive_col in valid.columns and naive_col in test.columns:
            self.naive_train = self._discretize_naive(train[naive_col])
            self.naive_valid = self._discretize_naive(valid[naive_col])
            self.naive_test = self._discretize_naive(test[naive_col])
        else:
            print("      [WARN] naive_yhat column not found; falling back to shifted labels as naive baseline.")
            self.naive_train = y_train.shift(self.horizon).bfill()
            self.naive_valid = y_valid.shift(self.horizon).bfill()
            self.naive_test = y_test.shift(self.horizon).bfill()

        print("[6/6] Creating PyTorch DataLoaders...")
        self.train_ds = TabularDataset(X_train_np, y_train_enc)
        self.valid_ds = TabularDataset(X_valid_np, y_valid_enc)
        self.test_ds = TabularDataset(X_test_np, y_test_enc)

        self.train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.valid_loader = DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        print(f"\n✓ Data loaded: Train={X_train_np.shape}, Valid={X_valid_np.shape}, Test={X_test_np.shape}")
        print(f"  Total time: {time.time() - t_start:.2f}s")
        print("\nClass distribution (train):")
        print(pd.Series(y_train_enc).value_counts(normalize=True).sort_index())

        # build model
        print("\n[Building FT-Transformer model...]")
        t_model = time.time()
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
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✓ Model built in {time.time() - t_model:.2f}s")
        print(f"  Total parameters: {total_params:,} | Trainable: {trainable_params:,}")
        print(f"  Device: {self.device}")

        # class weights
        cls_counts = pd.Series(y_train_enc).value_counts().sort_index()
        weights = cls_counts.sum() / (len(cls_counts) * cls_counts)
        self.class_weights = torch.tensor(weights.values, dtype=torch.float32, device=self.device)

        return self

    # ---------- Training ----------
    def train(self):
        print(f"\n{'='*60}\nTraining FT-Transformer (h+{self.horizon})\n{'='*60}")
        print(f"Hyperparameters:")
        print(f"  d_model={self.d_model}, nhead={self.nhead}, layers={self.num_layers}")
        print(f"  feedforward={self.dim_feedforward}, dropout={self.dropout}")
        print(f"  batch_size={self.batch_size}, lr={self.lr}, weight_decay={self.weight_decay}")
        print(f"  max_epochs={self.max_epochs}, patience={self.patience}")
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        best_f1 = -math.inf
        best_state = None
        patience_left = self.patience

        start_time = time.time()
        print(f"\nStarting training...")
        epoch_iter = range(1, self.max_epochs + 1)
        if tqdm is not None:
            epoch_iter = tqdm(epoch_iter, desc=f"FTT h+{self.horizon} Training", total=self.max_epochs, unit="epoch")

        for epoch in epoch_iter:
            epoch_start = time.time()
            self.model.train()
            total_loss = 0.0
            inner = self.train_loader
            if tqdm is not None:
                inner = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.max_epochs}", leave=False, unit="batch")
            for xb, yb in inner:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

            # validation
            val_metrics = self._eval_loader(self.valid_loader)
            val_f1 = val_metrics["f1_macro"]
            epoch_time = time.time() - epoch_start
            
            msg = (f"Epoch {epoch:02d}/{self.max_epochs} | loss={total_loss/len(self.train_ds):.4f} | "
                  f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_f1:.4f} | {epoch_time:.1f}s")
            
            if tqdm is not None and hasattr(epoch_iter, 'write'):
                epoch_iter.write(msg)
            else:
                print(msg)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_left = self.patience
                improvement_msg = f"  ✓ New best F1: {best_f1:.4f}"
                if tqdm is not None and hasattr(epoch_iter, 'write'):
                    epoch_iter.write(improvement_msg)
                else:
                    print(improvement_msg)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    early_stop_msg = f"\n⚠ Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)"
                    if tqdm is not None and hasattr(epoch_iter, 'write'):
                        epoch_iter.write(early_stop_msg)
                    else:
                        print(early_stop_msg)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        self.training_time_seconds = float(time.time() - start_time)
        final_msg = f"\n✓ Training completed in {self.training_time_seconds:.2f}s ({self.training_time_seconds/60:.2f} min) | Best val F1: {best_f1:.4f}"
        if tqdm is not None and hasattr(epoch_iter, 'write'):
            epoch_iter.write(final_msg)
        else:
            print(final_msg)
        return self

    @torch.no_grad()
    def _eval_loader(self, loader: DataLoader):
        self.model.eval()
        all_logits, all_y = [], []
        for xb, yb in loader:
            xb = xb.to(self.device)
            logits = self.model(xb)
            all_logits.append(logits.cpu())
            all_y.append(yb)
        y_true = torch.cat(all_y).numpy()
        y_prob = torch.softmax(torch.cat(all_logits), dim=1).numpy()
        y_pred = y_prob.argmax(axis=1)

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        precision, recall, f1_per, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1, 2]
        )
        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision": precision,
            "recall": recall,
            "f1_per_class": f1_per,
            "support": support,
            "pred": y_pred,
            "prob": y_prob,
        }

    def evaluate_all(self):
        print(f"\n{'='*60}\nEVALUATION - FT-Transformer (h+{self.horizon})\n{'='*60}")
        train_res = self._eval_loader(self.train_loader)
        valid_res = self._eval_loader(self.valid_loader)
        test_res = self._eval_loader(self.test_loader)

        # convert encoded predictions to labels for reporting and naive comparison
        def decode(arr):
            return pd.Series(arr).map(self.rev_mapping)

        def per_dataset(name, res, naive_series):
            y_true = decode(self.valid_ds.y.numpy() if name == "valid" else self.train_ds.y.numpy() if name == "train" else self.test_ds.y.numpy())
            y_pred = decode(res["pred"])
            print(f"\n{name.capitalize()} Results:\n" + "─" * 60)
            print(f"Accuracy:         {res['accuracy']:.4f}")
            print(f"F1-Score (Macro): {res['f1_macro']:.4f}")
            print(f"F1-Score (Wtd):   {res['f1_weighted']:.4f}")
            if naive_series is not None:
                naive_acc = accuracy_score(y_true, naive_series)
                naive_f1 = f1_score(y_true, naive_series, average="macro")
                print("\nNaive Baseline:")
                print(f"  Accuracy:  {naive_acc:.4f} (Δ: {res['accuracy'] - naive_acc:+.4f})")
                print(f"  F1-Macro:  {naive_f1:.4f} (Δ: {res['f1_macro'] - naive_f1:+.4f})")
            print("\nPer-Class Metrics:")
            for i, label in enumerate(["low", "mid", "high"]):
                print(
                    f"  {label:>4}: P={res['precision'][i]:.3f}, R={res['recall'][i]:.3f}, "
                    f"F1={res['f1_per_class'][i]:.3f}, Support={int(res['support'][i])}"
                )
            print("\nDetailed Classification Report:")
            print(classification_report(y_true, y_pred, digits=4))

        self.results = {"train": train_res, "valid": valid_res, "test": test_res}
        per_dataset("train", train_res, self.naive_train)
        per_dataset("valid", valid_res, self.naive_valid)
        per_dataset("test", test_res, self.naive_test)
        return self

    def plot_confusion_matrices(self, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        names = ["train", "valid", "test"]
        y_trues = [self.train_ds.y.numpy(), self.valid_ds.y.numpy(), self.test_ds.y.numpy()]
        for idx, name in enumerate(names):
            cm = confusion_matrix(y_trues[idx], self.results[name]["pred"])
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Purples",
                xticklabels=["low", "mid", "high"],
                yticklabels=["low", "mid", "high"],
                ax=axes[idx],
            )
            axes[idx].set_title(
                f"{name.capitalize()} Set\nAcc: {self.results[name]['accuracy']:.3f}, F1: {self.results[name]['f1_macro']:.3f}"
            )
            axes[idx].set_ylabel("True Label")
            axes[idx].set_xlabel("Predicted Label")
        plt.suptitle(f"FT-Transformer - Confusion Matrices (h+{self.horizon})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\nConfusion matrices saved to: {save_path}")
        plt.show()

    def save_model(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Torch model
        model_path = output_dir / f"ftt_classifier_h{self.horizon}.pt"
        torch.save({"state_dict": self.model.state_dict(), "params": {
            "n_features": len(self.feature_names),
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
        }}, model_path)
        joblib.dump(self.scaler, output_dir / f"ftt_scaler_h{self.horizon}.joblib")
        print(f"\nModel saved to: {model_path}")

        # Save results summary
        results_summary = {
            "horizon": self.horizon,
            "model": "FT-Transformer",
            "timestamp": datetime.now().isoformat(),
            "training_time_seconds": getattr(self, "training_time_seconds", None),
            "parameters": {
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
            },
            "results": {
                ds: {
                    "accuracy": float(res["accuracy"]),
                    "f1_macro": float(res["f1_macro"]),
                    "f1_weighted": float(res["f1_weighted"]),
                    "per_class_f1": {lab: float(f1) for lab, f1 in zip(["low", "mid", "high"], res["f1_per_class"])},
                }
                for ds, res in self.results.items()
            },
        }
        with open(output_dir / f"ftt_results_h{self.horizon}.json", "w") as f:
            json.dump(results_summary, f, indent=2)
        return self


def main():
    DATA_PATH = Path("../data_artifacts/splits")
    OUTPUT_DIR = Path("../classification-analysis/ft_transformer")
    horizons = [1, 6, 12, 24]

    for h in horizons:
        print("\n" + "#" * 70)
        print(f"# HORIZON: {h} HOUR(S)")
        print("#" * 70)

        clf = FTTransformerCOClassifier(
            horizon=h,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_size=16,
            lr=1e-3,
            weight_decay=1e-4,
            max_epochs=50,
            patience=8,
            seed=42,
        )

        clf.load_data(DATA_PATH)
        clf.train()
        clf.evaluate_all()

        out_dir = OUTPUT_DIR / f"h{h}"
        out_dir.mkdir(parents=True, exist_ok=True)
        clf.plot_confusion_matrices(save_path=out_dir / f"confusion_matrices_h{h}.png")
        clf.save_model(out_dir)

        print("\n" + "=" * 70)
        print(f"Completed horizon h+{h}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
