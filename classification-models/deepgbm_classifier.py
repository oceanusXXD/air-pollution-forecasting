"""
DeepGBM-like Classifier for CO Pollution Level Prediction
Predicts CO concentration levels (low, mid, high) at different time horizons.

This implementation approximates DeepGBM by combining:
- A GBDT component (XGBoost) to model complex interactions and produce leaf indices;
- A Deep component (PyTorch MLP) that embeds hashed (tree,leaf) ids and fuses with scaled raw features.

Dependencies: xgboost, torch, numpy, pandas, scikit-learn, matplotlib, seaborn
"""

from pathlib import Path
from datetime import datetime
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except Exception:
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


class LeafHasher:
    def __init__(self, num_buckets: int = 200000):
        self.num_buckets = num_buckets

    def transform(self, leaf_matrix: np.ndarray) -> np.ndarray:
        # leaf_matrix: (N, n_trees) int leaf ids
        n_trees = leaf_matrix.shape[1]
        ids = (np.arange(n_trees, dtype=np.int64)[None, :] * 10_000_000 + leaf_matrix.astype(np.int64))
        # Python's hash can vary; use a fixed hashing by modulo a large prime
        hashed = (ids % (2_147_483_647)) % self.num_buckets
        return hashed.astype(np.int64)


class DeepComponent(nn.Module):
    def __init__(self, n_features: int, num_buckets: int = 200000, emb_dim: int = 128, hidden: tuple[int, ...] = (256, 128), dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(num_buckets, emb_dim)
        self.dropout = nn.Dropout(dropout)
        layers: list[nn.Module] = []
        in_dim = n_features + emb_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 3)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, leaf_ids: torch.Tensor) -> torch.Tensor:
        # x_num: (B,F), leaf_ids: (B,T)
        emb = self.embedding(leaf_ids)  # (B,T,E)
        emb = emb.sum(dim=1)  # pooled embedding (B,E)
        feat = torch.cat([x_num, emb], dim=1)
        logits = self.mlp(feat)
        return logits


class DeepGBMCOClassifier:
    def __init__(
        self,
        horizon: int = 24,
        xgb_params: dict | None = None,
        num_buckets: int = 200000,
        emb_dim: int = 96,  # 减小嵌入维度 (原 128) - CPU 友好
        hidden: tuple[int, ...] = (192, 96),  # 减小隐藏层 (原 256,128) - 更快训练
        dropout: float = 0.25,  # 增加 dropout (原 0.2) - 防止小 batch 过拟合
        batch_size: int = 32,  # CPU 推荐 batch size (原 512)
        lr: float = 5e-4,  # 降低学习率 (原 1e-3) - 适应小 batch
        weight_decay: float = 1e-3,  # 增加正则化 (原 1e-4) - 防止过拟合
        max_epochs: int = 50,  # 增加 epochs (原 40) - 补偿小 batch 的慢收敛
        patience: int = 10,  # 增加 patience (原 8) - 给更多时间收敛
        seed: int = 42,
        device: str | None = None,
    ):
        self.horizon = horizon
        self.xgb_params = xgb_params or dict(
            n_estimators=400,  # 减少树数量 (原 600) - CPU 训练更快
            max_depth=6,  # 减小深度 (原 8) - 防止过拟合且更快
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3.0,  # 增加 (原 2.0) - 更保守,防止过拟合
            gamma=0.1,  # 增加 (原 0.0) - 更多正则化
            reg_lambda=1.5,  # 增加 (原 1.0) - 更多 L2 正则化
            reg_alpha=0.1,  # 增加 (原 0.0) - 添加 L1 正则化
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42,
        )
        self.num_buckets = num_buckets
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = None
        self.xgb_model = None
        self.deep_model = None
        self.leaf_hasher = LeafHasher(num_buckets=self.num_buckets)
        self.results = {}
        self.feature_names = None

        self.label_mapping = {"low": 0, "mid": 1, "high": 2}
        self.rev_mapping = {v: k for k, v in self.label_mapping.items()}

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _discretize_naive(self, s: pd.Series) -> pd.Series:
        return pd.cut(s, bins=[-np.inf, 1.5, 2.5, np.inf], labels=["low", "mid", "high"])

    def load_data(self, data_path: str | Path):
        base_path = Path(data_path) / f"h{self.horizon}"
        print(f"\n{'='*60}\nLoading data for horizon h+{self.horizon}\n{'='*60}")
        
        t_start = time.time()
        print("[1/7] Reading parquet files...")
        train = pd.read_parquet(base_path / "train.parquet")
        valid = pd.read_parquet(base_path / "valid.parquet")
        test = pd.read_parquet(base_path / "test.parquet")
        print(f"      ✓ Loaded in {time.time() - t_start:.2f}s")

        target_col = f"co_level_t+{self.horizon}"
        drop_cols = [f"y_t+{self.horizon}", f"naive_yhat_t+{self.horizon}", target_col]

        print("[2/7] Preparing features and targets...")
        X_train = train.drop(columns=drop_cols)
        y_train = train[target_col]
        X_valid = valid.drop(columns=drop_cols)
        y_valid = valid[target_col]
        X_test = test.drop(columns=drop_cols)
        y_test = test[target_col]
        self.feature_names = X_train.columns.tolist()
        print(f"      ✓ Features: {len(self.feature_names)}")

        print("[3/7] Scaling features with StandardScaler...")
        t_scale = time.time()
        self.scaler = StandardScaler()
        X_train_np = self.scaler.fit_transform(X_train.values.astype(np.float32))
        X_valid_np = self.scaler.transform(X_valid.values.astype(np.float32))
        X_test_np = self.scaler.transform(X_test.values.astype(np.float32))
        print(f"      ✓ Scaled in {time.time() - t_scale:.2f}s")

        print("[4/7] Encoding labels...")
        y_train_enc = y_train.map(self.label_mapping).values
        y_valid_enc = y_valid.map(self.label_mapping).values
        y_test_enc = y_test.map(self.label_mapping).values

        # class weights for imbalance -> sample weights for XGB
        print("[5/7] Computing class weights for imbalance handling...")
        cls_counts = pd.Series(y_train_enc).value_counts().sort_index()
        class_weights = (cls_counts.sum() / (len(cls_counts) * cls_counts)).to_dict()
        sw_train = pd.Series(y_train_enc).map(class_weights).values
        sw_valid = pd.Series(y_valid_enc).map(class_weights).values

        # Train XGBoost base learner
        print("[6/7] Training XGBoost base learner...")
        print(f"      XGB params: n_est={self.xgb_params.get('n_estimators', 600)}, "
              f"depth={self.xgb_params.get('max_depth', 8)}, lr={self.xgb_params.get('learning_rate', 0.05)}")
        t_xgb = time.time()
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        
        # Create callback for progress display
        class XGBProgressCallback(xgb.callback.TrainingCallback):
            def __init__(self, n_estimators):
                self.n_estimators = n_estimators
                self.pbar = None
                
            def before_training(self, model):
                if tqdm is not None:
                    self.pbar = tqdm(total=self.n_estimators, desc="      XGBoost Training", unit="tree")
                return model
                
            def after_iteration(self, model, epoch, evals_log):
                if self.pbar is not None:
                    self.pbar.update(1)
                return False
                
            def after_training(self, model):
                if self.pbar is not None:
                    self.pbar.close()
                return model
        
        callbacks = [XGBProgressCallback(self.xgb_params.get('n_estimators', 600))]
        
        self.xgb_model.fit(
            X_train_np,
            y_train_enc,
            eval_set=[(X_valid_np, y_valid_enc)],
            sample_weight=sw_train,
            sample_weight_eval_set=[sw_valid],
            verbose=False,
            callbacks=callbacks,
        )
        xgb_time = time.time() - t_xgb
        print(f"      ✓ XGBoost trained in {xgb_time:.2f}s (best iteration: {self.xgb_model.best_iteration})")

        # Leaf indices for all splits
        print("[7/7] Extracting leaf indices and hashing...")
        t_leaf = time.time()
        leaves_train = self.xgb_model.predict(X_train_np, pred_leaf=True)
        leaves_valid = self.xgb_model.predict(X_valid_np, pred_leaf=True)
        leaves_test = self.xgb_model.predict(X_test_np, pred_leaf=True)

        hashed_train = self.leaf_hasher.transform(leaves_train)
        hashed_valid = self.leaf_hasher.transform(leaves_valid)
        hashed_test = self.leaf_hasher.transform(leaves_test)
        print(f"      ✓ Leaves extracted & hashed in {time.time() - t_leaf:.2f}s")

        self.train_X = X_train_np
        self.valid_X = X_valid_np
        self.test_X = X_test_np
        self.train_y = y_train_enc
        self.valid_y = y_valid_enc
        self.test_y = y_test_enc
        self.train_leaf = hashed_train
        self.valid_leaf = hashed_valid
        self.test_leaf = hashed_test

        # naive baseline
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

        print(f"\n✓ Data loaded: Train={self.train_X.shape}, Valid={self.valid_X.shape}, Test={self.test_X.shape}")
        print(f"  Total time: {time.time() - t_start:.2f}s")
        print("\nClass distribution (train):")
        print(pd.Series(self.train_y).value_counts(normalize=True).sort_index())
        return self

    def _build_deep(self):
        print("\n[Building Deep component (MLP with leaf embeddings)...]")
        t_build = time.time()
        self.deep_model = DeepComponent(
            n_features=self.train_X.shape[1],
            num_buckets=self.num_buckets,
            emb_dim=self.emb_dim,
            hidden=self.hidden,
            dropout=self.dropout,
        ).to(self.device)
        total_params = sum(p.numel() for p in self.deep_model.parameters())
        trainable_params = sum(p.numel() for p in self.deep_model.parameters() if p.requires_grad)
        print(f"✓ Deep model built in {time.time() - t_build:.2f}s")
        print(f"  Total parameters: {total_params:,} | Trainable: {trainable_params:,}")
        print(f"  Device: {self.device}")

    def train(self):
        print(f"\n{'='*60}\nTraining DeepGBM Deep Component (h+{self.horizon})\n{'='*60}")
        self._build_deep()

        # class weights for CE
        print("\nPreparing Deep training...")
        cls_counts = pd.Series(self.train_y).value_counts().sort_index()
        weights = cls_counts.sum() / (len(cls_counts) * cls_counts)
        class_weights = torch.tensor(weights.values, dtype=torch.float32, device=self.device)
        print(f"  Class weights: {dict(zip(['low', 'mid', 'high'], weights.tolist()))}")

        train_ds = _DeepDS(self.train_X, self.train_leaf, self.train_y)
        valid_ds = _DeepDS(self.valid_X, self.valid_leaf, self.valid_y)
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.valid_loader = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        print(f"  Batch size: {self.batch_size}, LR: {self.lr}, Weight decay: {self.weight_decay}")
        optimizer = torch.optim.AdamW(self.deep_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_f1 = -np.inf
        best_state = None
        patience_left = self.patience

        start_time = time.time()
        print(f"\nStarting Deep training (max {self.max_epochs} epochs, patience={self.patience})...")
        epoch_iter = range(1, self.max_epochs + 1)
        if tqdm is not None:
            epoch_iter = tqdm(epoch_iter, desc=f"DeepGBM h+{self.horizon} Training", total=self.max_epochs, unit="epoch")

        for epoch in epoch_iter:
            epoch_start = time.time()
            self.deep_model.train()
            loss_sum = 0.0
            n_sum = 0
            inner = self.train_loader
            if tqdm is not None:
                inner = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.max_epochs}", leave=False, unit="batch")
            for xb, lb, yb in inner:
                xb = xb.to(self.device)
                lb = lb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.deep_model(xb, lb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.deep_model.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item() * xb.size(0)
                n_sum += xb.size(0)

            val_metrics = self._eval_split(self.valid_X, self.valid_leaf, self.valid_y)
            epoch_time = time.time() - epoch_start
            
            msg = (f"Epoch {epoch:02d}/{self.max_epochs} | loss={loss_sum/max(n_sum,1):.4f} | "
                  f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['f1_macro']:.4f} | {epoch_time:.1f}s")
            
            if tqdm is not None and hasattr(epoch_iter, 'write'):
                epoch_iter.write(msg)
            else:
                print(msg)

            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                best_state = {k: v.cpu().clone() for k, v in self.deep_model.state_dict().items()}
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
            self.deep_model.load_state_dict(best_state)
        
        self.training_time_seconds = float(time.time() - start_time)
        final_msg = f"\n✓ Deep training completed in {self.training_time_seconds:.2f}s ({self.training_time_seconds/60:.2f} min) | Best val F1: {best_f1:.4f}"
        if tqdm is not None and hasattr(epoch_iter, 'write'):
            epoch_iter.write(final_msg)
        else:
            print(final_msg)
        return self

    @torch.no_grad()
    def _eval_split(self, X_np: np.ndarray, leaf_np: np.ndarray, y_np: np.ndarray):
        self.deep_model.eval()
        bs = self.batch_size
        prob_list = []
        pred_list = []
        for i in range(0, len(X_np), bs):
            xb = torch.from_numpy(X_np[i : i + bs]).float().to(self.device)
            lb = torch.from_numpy(leaf_np[i : i + bs]).long().to(self.device)
            logits = self.deep_model(xb, lb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            pred = prob.argmax(axis=1)
            prob_list.append(prob)
            pred_list.append(pred)
        prob_all = np.concatenate(prob_list, axis=0)
        pred_all = np.concatenate(pred_list, axis=0)

        acc = accuracy_score(y_np, pred_all)
        f1_macro = f1_score(y_np, pred_all, average="macro")
        f1_weighted = f1_score(y_np, pred_all, average="weighted")
        precision, recall, f1_per, support = precision_recall_fscore_support(y_np, pred_all, average=None, labels=[0, 1, 2])
        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision": precision,
            "recall": recall,
            "f1_per_class": f1_per,
            "support": support,
            "pred": pred_all,
            "prob": prob_all,
        }

    def evaluate_all(self):
        print(f"\n{'='*60}\nEVALUATION - DeepGBM (h+{self.horizon})\n{'='*60}")
        self.results = {
            "train": self._eval_split(self.train_X, self.train_leaf, self.train_y),
            "valid": self._eval_split(self.valid_X, self.valid_leaf, self.valid_y),
            "test": self._eval_split(self.test_X, self.test_leaf, self.test_y),
        }

        def decode(arr):
            return pd.Series(arr).map(self.rev_mapping)

        def report(name: str, res, y_true_enc: np.ndarray, naive_series: pd.Series):
            y_true = decode(y_true_enc)
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

        report("train", self.results["train"], self.train_y, self.naive_train)
        report("valid", self.results["valid"], self.valid_y, self.naive_valid)
        report("test", self.results["test"], self.test_y, self.naive_test)
        return self

    def plot_confusion_matrices(self, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        names = ["train", "valid", "test"]
        y_trues = [self.train_y, self.valid_y, self.test_y]
        for idx, name in enumerate(names):
            cm = confusion_matrix(y_trues[idx], self.results[name]["pred"])
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Oranges",
                xticklabels=["low", "mid", "high"],
                yticklabels=["low", "mid", "high"],
                ax=axes[idx],
            )
            axes[idx].set_title(
                f"{name.capitalize()} Set\nAcc: {self.results[name]['accuracy']:.3f}, F1: {self.results[name]['f1_macro']:.3f}"
            )
            axes[idx].set_ylabel("True Label")
            axes[idx].set_xlabel("Predicted Label")
        plt.suptitle(f"DeepGBM - Confusion Matrices (h+{self.horizon})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\nConfusion matrices saved to: {save_path}")
        plt.show()

    def save_model(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save xgb
        joblib.dump(self.xgb_model, output_dir / f"deepgbm_xgb_h{self.horizon}.joblib")
        # Save deep model
        torch.save({"state_dict": self.deep_model.state_dict(), "params": {
            "n_features": len(self.feature_names),
            "num_buckets": self.num_buckets,
            "emb_dim": self.emb_dim,
            "hidden": self.hidden,
            "dropout": self.dropout,
        }}, output_dir / f"deepgbm_deep_h{self.horizon}.pt")
        joblib.dump(self.scaler, output_dir / f"deepgbm_scaler_h{self.horizon}.joblib")

        results_summary = {
            "horizon": self.horizon,
            "model": "DeepGBM",
            "timestamp": datetime.now().isoformat(),
            "training_time_seconds": getattr(self, "training_time_seconds", None),
            "xgb_params": self.xgb_params,
            "deep_params": {
                "num_buckets": self.num_buckets,
                "emb_dim": self.emb_dim,
                "hidden": self.hidden,
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
        with open(output_dir / f"deepgbm_results_h{self.horizon}.json", "w") as f:
            json.dump(results_summary, f, indent=2)
        print(f"Models & results saved to: {output_dir}")
        return self


class _DeepDS(Dataset):
    def __init__(self, X: np.ndarray, leaf: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.leaf = torch.from_numpy(leaf).long()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.leaf[idx], self.y[idx]


def main():
    DATA_PATH = Path("../data_artifacts/splits")
    OUTPUT_DIR = Path("../classification-analysis/deepgbm")
    horizons = [1, 6, 12, 24]

    for h in horizons:
        print("\n" + "#" * 70)
        print(f"# HORIZON: {h} HOUR(S)")
        print("#" * 70)

        clf = DeepGBMCOClassifier(
            horizon=h,
            # xgb params already set to a strong baseline; can be tuned further
            num_buckets=200000,
            emb_dim=128,
            hidden=(256, 128),
            dropout=0.2,
            batch_size=16,
            lr=1e-3,
            weight_decay=1e-4,
            max_epochs=40,
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
