"""
依赖:
    - xgboost
    - torch
    - numpy
    - pandas
    - scikit-learn
    - matplotlib
    - seaborn
    - joblib
    - pyarrow (for parquet)
"""

from pathlib import Path
from datetime import datetime
import json
import time
import os
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
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
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ============================= Helpers =============================

class LeafHasher:
    """Hash tree leaf indices into a fixed number of buckets."""
    def __init__(self, num_buckets: int = 100_000):
        self.num_buckets = int(num_buckets)
        self._prime = 2_147_483_647

    def transform(self, leaf_matrix: np.ndarray) -> np.ndarray:
        leaf_matrix = np.asarray(leaf_matrix, dtype=np.int64)
        if leaf_matrix.ndim != 2:
            leaf_matrix = leaf_matrix.reshape(-1, leaf_matrix.shape[-1])
        n_trees = leaf_matrix.shape[1]
        ids = (np.arange(n_trees, dtype=np.int64)[None, :] * 10_000_000 + leaf_matrix)
        hashed = (ids % self._prime) % self.num_buckets
        return hashed.astype(np.int64)


# ============================= Deep component =============================

class DeepComponent(nn.Module):
    """
    小容量 Deep 模块：数值特征 + 叶子 embedding -> MLP -> 3 类 logits
    """
    def __init__(
        self,
        n_features: int,
        num_buckets: int = 100_000,
        emb_dim: int = 32,
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.35,
    ):
        super().__init__()
        self.num_buckets = int(num_buckets)
        self.emb_dim = int(emb_dim)
        self.embedding = nn.EmbeddingBag(self.num_buckets, self.emb_dim, mode="sum", sparse=False)

        in_dim = n_features + self.emb_dim
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, leaf_ids: torch.Tensor) -> torch.Tensor:
        """
        x_num: (B, F)
        leaf_ids: (B, T) -> hashed indices
        """
        B, T = leaf_ids.shape
        flat = leaf_ids.reshape(-1)
        offsets = torch.arange(0, B * T, T, dtype=torch.long, device=leaf_ids.device)
        emb = self.embedding(flat, offsets)  # (B, D)
        feat = torch.cat([x_num, emb], dim=1)
        logits = self.mlp(feat)
        return logits


# ============================= Dataset =============================

class _DeepDS(Dataset):
    def __init__(self, X: np.ndarray, leaf: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.leaf = np.asarray(leaf, dtype=np.int64)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]).float(),
            torch.from_numpy(self.leaf[idx]).long(),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# ============================= Main Model =============================

class DeepGBMCOClassifier:
    def __init__(
        self,
        horizon: int = 24,
        xgb_params: dict | None = None,
        num_buckets: int = 100_000,
        emb_dim: int = 32,
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.35,
        batch_size: int = 128,
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
        max_epochs: int = 120,
        patience: int = 16,
        min_delta: float = 0.001,
        es_warmup_epochs: int = 5,
        gradient_accumulation_steps: int = 1,
        # xgb_weight 不再固定，用验证集搜索最优
        seed: int = 42,
        device: str | None = None,
        checkpoint_dir: str | Path | None = None,
    ):
        self.horizon = int(horizon)
        self.num_buckets = int(num_buckets)
        self.emb_dim = int(emb_dim)
        self.hidden = tuple(hidden)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.es_warmup_epochs = int(es_warmup_epochs)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)

        self.seed = int(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # XGB 参数
        self.xgb_params = xgb_params or dict(
            n_estimators=800,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3.0,
            gamma=0.1,
            reg_lambda=2.0,
            reg_alpha=0.2,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=self.seed,
        )

        self.scaler: StandardScaler | None = None
        self.xgb_model: Any = None
        self.deep_model: nn.Module | None = None
        self.leaf_hasher = LeafHasher(num_buckets=self.num_buckets)

        self.results: Dict[str, Any] = {}
        self.feature_names: List[str] | None = None
        self.history: Dict[str, list] | None = None

        self.label_mapping = {"low": 0, "mid": 1, "high": 2}
        self.rev_mapping = {v: k for k, v in self.label_mapping.items()}

        self.best_xgb_weight: float = 0.8  # 会在验证集上自动调整

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    # ---------- utils ----------

    def _discretize_naive(self, s: pd.Series) -> pd.Series:
        return pd.cut(s, bins=[-np.inf, 1.5, 2.5, np.inf], labels=["low", "mid", "high"])

    def _prepare_xgb_params_for_train(self) -> dict:
        params = {}
        params["objective"] = "multi:softprob"
        params["num_class"] = 3
        params["max_depth"] = int(self.xgb_params.get("max_depth", 6))
        params["eta"] = float(self.xgb_params.get("learning_rate", 0.05))
        params["subsample"] = float(self.xgb_params.get("subsample", 1.0))
        params["colsample_bytree"] = float(self.xgb_params.get("colsample_bytree", 1.0))
        params["min_child_weight"] = float(self.xgb_params.get("min_child_weight", 1.0))
        params["gamma"] = float(self.xgb_params.get("gamma", 0.0))
        params["lambda"] = float(self.xgb_params.get("reg_lambda", 1.0))
        params["alpha"] = float(self.xgb_params.get("reg_alpha", 0.0))
        if torch.cuda.is_available():
            params["tree_method"] = "gpu_hist"
        else:
            params["tree_method"] = self.xgb_params.get("tree_method", "hist")
        params["eval_metric"] = self.xgb_params.get("eval_metric", "mlogloss")
        params["verbosity"] = 1
        return params

    # ---------- 加载 & 训练 XGB + 叶子 ----------

    def load_data(self, data_path: str | Path):
        base_path = Path(data_path) / f"h{self.horizon}"
        print(f"\n{'='*60}\nLoading data for horizon h+{self.horizon}\n{'='*60}")
        t_start = time.time()

        train = pd.read_parquet(base_path / "train.parquet")
        valid = pd.read_parquet(base_path / "valid.parquet")
        test = pd.read_parquet(base_path / "test.parquet")

        target_col = f"co_level_t+{self.horizon}"
        drop_cols = [f"y_t+{self.horizon}", f"naive_yhat_t+{self.horizon}", target_col]

        X_train = train.drop(columns=drop_cols, errors="ignore")
        y_train = train[target_col]
        X_valid = valid.drop(columns=drop_cols, errors="ignore")
        y_valid = valid[target_col]
        X_test = test.drop(columns=drop_cols, errors="ignore")
        y_test = test[target_col]

        self.feature_names = X_train.columns.tolist()
        print(f"Features: {len(self.feature_names)}")

        self.scaler = StandardScaler()
        X_train_np = self.scaler.fit_transform(X_train.values.astype(np.float32))
        X_valid_np = self.scaler.transform(X_valid.values.astype(np.float32))
        X_test_np = self.scaler.transform(X_test.values.astype(np.float32))

        y_train_enc = y_train.map(self.label_mapping).astype("int64").to_numpy()
        y_valid_enc = y_valid.map(self.label_mapping).astype("int64").to_numpy()
        y_test_enc = y_test.map(self.label_mapping).astype("int64").to_numpy()

        # class weights
        cls_counts = pd.Series(y_train_enc).value_counts().sort_index()
        class_weights = (cls_counts.sum() / (len(cls_counts) * cls_counts)).to_dict()
        sw_train = pd.Series(y_train_enc).map(class_weights).values
        sw_valid = pd.Series(y_valid_enc).map(class_weights).values

        # train XGB
        print("\n[Train XGBoost]")
        dtrain = xgb.DMatrix(X_train_np, label=y_train_enc, weight=sw_train)
        dvalid = xgb.DMatrix(X_valid_np, label=y_valid_enc, weight=sw_valid)

        train_params = self._prepare_xgb_params_for_train()
        num_boost_round = int(self.xgb_params.get("n_estimators", 800))
        bst = xgb.train(
            params=train_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=50,
            verbose_eval=True,
        )
        self.xgb_model = bst

        print("[Extract leaf indices & hash]")
        dtest = xgb.DMatrix(X_test_np)
        leaves_train = bst.predict(dtrain, pred_leaf=True)
        leaves_valid = bst.predict(dvalid, pred_leaf=True)
        leaves_test = bst.predict(dtest, pred_leaf=True)

        hashed_train = self.leaf_hasher.transform(np.asarray(leaves_train))
        hashed_valid = self.leaf_hasher.transform(np.asarray(leaves_valid))
        hashed_test = self.leaf_hasher.transform(np.asarray(leaves_test))

        self.train_X = X_train_np
        self.valid_X = X_valid_np
        self.test_X = X_test_np
        self.train_y = y_train_enc
        self.valid_y = y_valid_enc
        self.test_y = y_test_enc
        self.train_leaf = hashed_train
        self.valid_leaf = hashed_valid
        self.test_leaf = hashed_test

        naive_col = f"naive_yhat_t+{self.horizon}"
        if naive_col in train.columns and naive_col in valid.columns and naive_col in test.columns:
            self.naive_train = self._discretize_naive(train[naive_col])
            self.naive_valid = self._discretize_naive(valid[naive_col])
            self.naive_test = self._discretize_naive(test[naive_col])
        else:
            self.naive_train = None
            self.naive_valid = None
            self.naive_test = None

        print(f"\n✓ Data loaded. Train={self.train_X.shape}, Valid={self.valid_X.shape}, Test={self.test_X.shape}")
        print("Class distribution (train):")
        print(pd.Series(self.train_y).value_counts(normalize=True).sort_index())
        return self

    # ---------- Deep 部分 ----------

    def _build_deep(self):
        print("\n[Building Deep component...]")
        self.deep_model = DeepComponent(
            n_features=self.train_X.shape[1],
            num_buckets=self.num_buckets,
            emb_dim=self.emb_dim,
            hidden=self.hidden,
            dropout=self.dropout,
        ).to(self.device)
        total_params = sum(p.numel() for p in self.deep_model.parameters())
        print(f"Deep params: {total_params:,} | Device: {self.device}")

    def train(self):
        print(f"\n{'='*60}\nTrain DeepGBM Deep Component (h+{self.horizon})\n{'='*60}")
        self._build_deep()

        cls_counts = pd.Series(self.train_y).value_counts().sort_index()
        weights = cls_counts.sum() / (len(cls_counts) * cls_counts)
        class_weights = torch.tensor(weights.values, dtype=torch.float32, device=self.device)
        print("Class weights:", dict(zip(["low", "mid", "high"], weights.tolist())))

        train_ds = _DeepDS(self.train_X, self.train_leaf, self.train_y)
        valid_ds = _DeepDS(self.valid_X, self.valid_leaf, self.valid_y)

        pin_memory = self.device.type == "cuda"
        if self.device.type != "cuda":
            try:
                torch.set_num_threads(1)
            except Exception:
                pass

        self.train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory
        )
        self.valid_loader = DataLoader(
            valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory
        )

        optimizer = torch.optim.AdamW(self.deep_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=4,
            threshold=self.min_delta,
            min_lr=1e-6,
        )

        scaler = GradScaler() if self.device.type == "cuda" else None

        best_f1 = -np.inf
        best_state = None
        patience_left = self.patience
        last_val_f1 = 0.0
        self.history = {"train_loss": [], "val_f1": [], "lr": []}

        start_t = time.time()

        for epoch in range(1, self.max_epochs + 1):
            ep_start = time.time()
            self.deep_model.train()
            loss_sum = 0.0
            n_sum = 0
            optimizer.zero_grad()
            acc_steps = max(1, self.gradient_accumulation_steps)

            for step, (xb, lb, yb) in enumerate(self.train_loader):
                xb = xb.to(self.device, non_blocking=True)
                lb = lb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                if scaler is not None:
                    with autocast():
                        logits = self.deep_model(xb, lb)
                        loss = criterion(logits, yb) / acc_steps
                    scaler.scale(loss).backward()
                else:
                    logits = self.deep_model(xb, lb)
                    loss = criterion(logits, yb) / acc_steps
                    loss.backward()

                loss_sum += float(loss.item()) * xb.size(0) * acc_steps
                n_sum += xb.size(0)

                if (step + 1) % acc_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.deep_model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.deep_model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad()

            avg_loss = loss_sum / max(1, n_sum)
            self.history["train_loss"].append(float(avg_loss))

            # 验证策略：前10个epoch每轮验证，之后每5轮验证一次
            should_val = (epoch <= 10) or (epoch % 5 == 0) or (epoch == self.max_epochs)
            if should_val:
                val_metrics = self._eval_split(self.valid_X, self.valid_leaf, self.valid_y)
                val_f1 = float(val_metrics["f1_macro"])
                last_val_f1 = val_f1
                scheduler.step(val_f1)
            else:
                val_f1 = last_val_f1

            self.history["val_f1"].append(val_f1)
            cur_lr = optimizer.param_groups[0]["lr"]
            self.history["lr"].append(cur_lr)

            ep_time = time.time() - ep_start
            print(
                f"Epoch {epoch:3d}/{self.max_epochs} | Loss {avg_loss:.4f} | Val F1 {val_f1:.4f} | LR {cur_lr:.2e} | {ep_time:.1f}s",
                end=""
            )

            if epoch <= self.es_warmup_epochs:
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_state = {k: v.cpu().clone() for k, v in self.deep_model.state_dict().items()}
                    print(" ✓ warmup improve")
                else:
                    print()
                continue

            if should_val and val_f1 > best_f1 + self.min_delta:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in self.deep_model.state_dict().items()}
                patience_left = self.patience
                print(" ✓ NEW BEST")
                ckpt = {"state_dict": best_state, "epoch": epoch, "val_f1": best_f1}
                try:
                    torch.save(ckpt, self.checkpoint_dir / f"deep_best_h{self.horizon}.pt")
                except Exception:
                    pass
            else:
                if should_val:
                    patience_left -= 1
                print(f" (patience {patience_left}/{self.patience})")
                if patience_left <= 0:
                    print(f"\n⚠ Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.deep_model.load_state_dict(best_state)
        self.training_time_seconds = float(time.time() - start_t)
        print(f"\n✓ Deep training done in {self.training_time_seconds/60:.2f} min | Best val F1={best_f1:.4f}")
        return self

    # ---------- 预测与评估 ----------

    def _xgb_predict_proba(self, X_np: np.ndarray) -> np.ndarray | None:
        try:
            if isinstance(self.xgb_model, xgb.core.Booster):
                dmat = xgb.DMatrix(X_np)
                return self.xgb_model.predict(dmat)
            else:
                return self.xgb_model.predict_proba(X_np)
        except Exception:
            return None

    @torch.no_grad()
    def _deep_predict_proba(self, X_np: np.ndarray, leaf_np: np.ndarray) -> np.ndarray:
        self.deep_model.eval()
        bs = self.batch_size
        probs = []
        for i in range(0, len(X_np), bs):
            xb = torch.from_numpy(X_np[i : i + bs]).float().to(self.device)
            lb = torch.from_numpy(leaf_np[i : i + bs]).long().to(self.device)
            logits = self.deep_model(xb, lb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(prob)
        return np.concatenate(probs, axis=0)

    def _eval_split(self, X_np: np.ndarray, leaf_np: np.ndarray, y_np: np.ndarray) -> Dict[str, Any]:
        prob = self._deep_predict_proba(X_np, leaf_np)
        pred = prob.argmax(axis=1)
        acc = accuracy_score(y_np, pred)
        f1m = f1_score(y_np, pred, average="macro")
        f1w = f1_score(y_np, pred, average="weighted")
        precision, recall, f1_per, support = precision_recall_fscore_support(
            y_np, pred, average=None, labels=[0, 1, 2]
        )
        return {
            "accuracy": acc,
            "f1_macro": f1m,
            "f1_weighted": f1w,
            "precision": precision,
            "recall": recall,
            "f1_per_class": f1_per,
            "support": support,
            "pred": pred,
            "prob": prob,
        }

    def _xgb_only_metrics(self, X_np: np.ndarray, y_np: np.ndarray) -> Tuple[float, float, float] | None:
        proba = self._xgb_predict_proba(X_np)
        if proba is None:
            return None
        pred = proba.argmax(axis=1)
        acc = accuracy_score(y_np, pred)
        f1m = f1_score(y_np, pred, average="macro")
        f1w = f1_score(y_np, pred, average="weighted")
        return acc, f1m, f1w

    def _fuse_probs(self, deep_prob: np.ndarray, xgb_prob: np.ndarray | None, w: float) -> np.ndarray:
        if xgb_prob is None:
            return deep_prob
        w = float(w)
        return (1 - w) * deep_prob + w * xgb_prob

    def _search_best_xgb_weight(self) -> float:
        """
        在验证集上用网格搜索找到 best xgb_weight，使 Deep+XGB 的 F1_macro 最大。
        """
        xgb_proba_valid = self._xgb_predict_proba(self.valid_X)
        deep_proba_valid = self.results["valid"]["prob"]
        y_valid = self.valid_y

        if xgb_proba_valid is None:
            print("[WARN] XGB proba not available, skip weight search.")
            return 0.0  # pure Deep

        candidates = np.linspace(0.0, 1.0, num=11)  # 0.0, 0.1, ..., 1.0
        best_w = 0.0
        best_f1 = -np.inf
        for w in candidates:
            fused = self._fuse_probs(deep_proba_valid, xgb_proba_valid, w)
            pred = fused.argmax(axis=1)
            f1m = f1_score(y_valid, pred, average="macro")
            if f1m > best_f1 + 1e-5:
                best_f1 = f1m
                best_w = w
        print(f"\n[Search xgb_weight] best_w={best_w:.2f} on valid (F1_macro={best_f1:.4f})")
        self.best_xgb_weight = best_w
        return best_w

    def evaluate_all(self):
        print(f"\n{'='*60}\nEVALUATION - DeepGBM (h+{self.horizon})\n{'='*60}")
        # Deep-only 结果
        self.results = {
            "train": self._eval_split(self.train_X, self.train_leaf, self.train_y),
            "valid": self._eval_split(self.valid_X, self.valid_leaf, self.valid_y),
            "test":  self._eval_split(self.test_X,  self.test_leaf,  self.test_y),
        }

        # XGB-only 结果
        xgb_metrics_train = self._xgb_only_metrics(self.train_X, self.train_y)
        xgb_metrics_valid = self._xgb_only_metrics(self.valid_X, self.valid_y)
        xgb_metrics_test = self._xgb_only_metrics(self.test_X,  self.test_y)

        # 搜索最优 xgb_weight
        self._search_best_xgb_weight()

        def decode(arr):
            return pd.Series(arr).map(self.rev_mapping)

        def report(split_name: str, res: Dict[str, Any], y_enc: np.ndarray, naive_series: pd.Series | None,
                   xgb_metrics: Tuple[float, float, float] | None, X_np: np.ndarray):
            y_true = decode(y_enc)
            # Deep-only
            deep_pred = res["pred"]
            y_pred_deep = decode(deep_pred)
            deep_acc = accuracy_score(y_true, y_pred_deep)
            deep_f1m = f1_score(y_true, y_pred_deep, average="macro")

            # XGB-only
            if xgb_metrics is not None:
                xgb_acc, xgb_f1m, _ = xgb_metrics
            else:
                xgb_acc = xgb_f1m = None

            # Deep+XGB with best weight
            xgb_prob = self._xgb_predict_proba(X_np)
            fused_prob = self._fuse_probs(res["prob"], xgb_prob, self.best_xgb_weight)
            fused_pred = fused_prob.argmax(axis=1)
            y_pred_fused = decode(fused_pred)
            fused_acc = accuracy_score(y_true, y_pred_fused)
            fused_f1m = f1_score(y_true, y_pred_fused, average="macro")

            print(f"\n{split_name.capitalize()} Results (h+{self.horizon}):")
            print("─" * 60)
            print(f"Deep-only:           Acc={deep_acc:.4f}, F1_macro={deep_f1m:.4f}")
            if xgb_acc is not None:
                print(f"XGB-only:            Acc={xgb_acc:.4f}, F1_macro={xgb_f1m:.4f}")
            print(f"Deep+XGB (w={self.best_xgb_weight:.2f}): Acc={fused_acc:.4f}, F1_macro={fused_f1m:.4f}")

            if naive_series is not None:
                naive_acc = accuracy_score(y_true, naive_series)
                naive_f1 = f1_score(y_true, naive_series, average="macro")
                print("\nNaive baseline:")
                print(f"  Acc:     {naive_acc:.4f}")
                print(f"  F1Macro: {naive_f1:.4f}")

            print("\nClassification report (Deep+XGB):")
            print(classification_report(y_true, y_pred_fused, digits=4))

        report(
            "train",
            self.results["train"],
            self.train_y,
            self.naive_train,
            xgb_metrics_train,
            self.train_X,
        )
        report(
            "valid",
            self.results["valid"],
            self.valid_y,
            self.naive_valid,
            xgb_metrics_valid,
            self.valid_X,
        )
        report(
            "test",
            self.results["test"],
            self.test_y,
            self.naive_test,
            xgb_metrics_test,
            self.test_X,
        )
        return self

    # ---------- Plot & Save ----------

    def plot_training_history(self, save_path=None):
        if self.history is None:
            print("[WARN] No training history to plot.")
            return
        hist = self.history
        epochs = list(range(1, len(hist["train_loss"]) + 1))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(epochs, hist["train_loss"], marker="o")
        axes[0].set_title("Train Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, hist["val_f1"], marker="o", color="orange")
        axes[1].set_title("Validation F1 (macro)")
        axes[1].set_xlabel("Epoch")
        axes[1].grid(alpha=0.3)

        axes[2].plot(epochs, hist["lr"], marker="o", color="green")
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_yscale("log")
        axes[2].grid(alpha=0.3)

        plt.suptitle(f"DeepGBM Training History (h+{self.horizon})", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Training history saved to: {save_path}")
        plt.show()

    def plot_confusion_matrices(self, save_path=None):
        if not self.results:
            print("[WARN] No results to plot confusion matrices.")
            return
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
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
        plt.suptitle(f"DeepGBM - Confusion Matrices (h+{self.horizon})", fontsize=16, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrices saved to: {save_path}")
        plt.show()

    def save_model(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存 XGB
        try:
            if isinstance(self.xgb_model, xgb.core.Booster):
                self.xgb_model.save_model(str(output_dir / f"deepgbm_xgb_h{self.horizon}.model"))
            else:
                joblib.dump(self.xgb_model, output_dir / f"deepgbm_xgb_h{self.horizon}.joblib")
        except Exception:
            joblib.dump(self.xgb_model, output_dir / f"deepgbm_xgb_h{self.horizon}.joblib")

        # Deep & scaler
        torch.save(
            {
                "state_dict": self.deep_model.state_dict(),
                "params": {
                    "n_features": len(self.feature_names),
                    "num_buckets": self.num_buckets,
                    "emb_dim": self.emb_dim,
                    "hidden": self.hidden,
                    "dropout": self.dropout,
                    "horizon": self.horizon,
                    "best_xgb_weight": self.best_xgb_weight,
                },
                "history": self.history,
            },
            output_dir / f"deepgbm_deep_h{self.horizon}.pt",
        )
        joblib.dump(self.scaler, output_dir / f"deepgbm_scaler_h{self.horizon}.joblib")

        def _pack_results(res_dict):
            out = {}
            for ds, res in res_dict.items():
                out[ds] = {
                    "accuracy": float(res["accuracy"]),
                    "f1_macro": float(res["f1_macro"]),
                    "f1_weighted": float(res["f1_weighted"]),
                    "per_class_f1": {
                        lab: float(f1) for lab, f1 in zip(["low", "mid", "high"], res["f1_per_class"])
                    },
                }
            return out

        results_summary = {
            "horizon": self.horizon,
            "model_type": "deepgbm",
            "timestamp": datetime.now().isoformat(),
            "training_time_seconds": getattr(self, "training_time_seconds", None),
            "best_xgb_weight": self.best_xgb_weight,
            "params": {
                "num_buckets": self.num_buckets,
                "emb_dim": self.emb_dim,
                "hidden": self.hidden,
                "dropout": self.dropout,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "min_delta": self.min_delta,
                "es_warmup_epochs": self.es_warmup_epochs,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "seed": self.seed,
                "xgb_params": self.xgb_params,
            },
            "results_deep_only": _pack_results(self.results),
            # 提示：XGB-only 与融合指标需要在 evaluate_all 时另行记录，如有需要可扩展
        }

        with open(output_dir / f"deepgbm_results_h{self.horizon}.json", "w") as f:
            json.dump(results_summary, f, indent=2)
        print(f"Models & results saved to: {output_dir}")
        return self


# ============================= main =============================

def main():
    DATA_PATH = Path("/app/data_artifacts/splits")
    OUTPUT_DIR = Path("/app/classification-analysis/deepgbm_unified_v2")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    horizons = [1, 6, 12, 24]

    for h in horizons:
        print("\n" + "#" * 80)
        print(f"# HORIZON: {h} HOUR(S)")
        print("#" * 80)

        # 相对保守的 Deep 配置：小容量 + 强一点正则
        if h == 1:
            cfg = dict(lr=5e-4, patience=20, min_delta=0.0005, es_warmup_epochs=4, max_epochs=80)
        elif h == 6:
            cfg = dict(lr=4e-4, patience=20, min_delta=0.0007, es_warmup_epochs=5, max_epochs=100)
        else:  # 12 / 24
            cfg = dict(lr=3e-4, patience=22, min_delta=0.0007, es_warmup_epochs=6, max_epochs=120)

        clf = DeepGBMCOClassifier(
            horizon=h,
            num_buckets=100_000,
            emb_dim=32,
            hidden=(128, 64),
            dropout=0.35,
            batch_size=128,
            lr=cfg["lr"],
            weight_decay=1e-3,
            max_epochs=cfg["max_epochs"],
            patience=cfg["patience"],
            min_delta=cfg["min_delta"],
            es_warmup_epochs=cfg["es_warmup_epochs"],
            gradient_accumulation_steps=1,
            seed=42,
            device=None,
            checkpoint_dir=OUTPUT_DIR / "checkpoints",
        )

        clf.load_data(DATA_PATH)
        clf.train()
        clf.evaluate_all()

        out_dir = OUTPUT_DIR / f"h{h}"
        out_dir.mkdir(parents=True, exist_ok=True)
        clf.plot_training_history(save_path=out_dir / f"training_history_h{h}.png")
        clf.plot_confusion_matrices(save_path=out_dir / f"confusion_matrices_h{h}.png")
        clf.save_model(out_dir)

        print("\n" + "=" * 80)
        print(f"Completed horizon h+{h}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()