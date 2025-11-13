"""
DeepGBM-like Classifier for CO Pollution Level Prediction (效果优先版本)

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

# ----------------------------- Helpers -----------------------------
class LeafHasher:
    """
    将 (n_samples, n_trees) 的 leaf id 矩阵映射到 [0, num_buckets)
    使用固定模大素数以保证稳定性
    """
    def __init__(self, num_buckets: int = 300000):
        self.num_buckets = int(num_buckets)
        self._prime = 2_147_483_647

    def transform(self, leaf_matrix: np.ndarray) -> np.ndarray:
        leaf_matrix = np.asarray(leaf_matrix, dtype=np.int64)
        n_trees = leaf_matrix.shape[1]
        ids = (np.arange(n_trees, dtype=np.int64)[None, :] * 10_000_000 + leaf_matrix)
        hashed = (ids % self._prime) % self.num_buckets
        return hashed.astype(np.int64)


# ----------------------------- Deep component -----------------------------
class DeepComponent(nn.Module):
    """
    EmbeddingBag + 深层 MLP (LayerNorm + Dropout)
    """
    def __init__(
        self,
        n_features: int,
        num_buckets: int = 300000,
        emb_dim: int = 512,
        hidden: tuple[int, ...] = (1024, 512, 256),
        dropout: float = 0.25,
    ):
        super().__init__()
        self.num_buckets = int(num_buckets)
        self.emb_dim = int(emb_dim)
        self.embedding = nn.EmbeddingBag(self.num_buckets, self.emb_dim, mode="sum", sparse=False)
        self.dropout = nn.Dropout(dropout)

        in_dim = n_features + self.emb_dim
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 3))  # 3 classes
        self.mlp = nn.Sequential(*layers)

        self._in_dim = in_dim
        self._out_dim = prev

    def forward(self, x_num: torch.Tensor, leaf_ids: torch.Tensor) -> torch.Tensor:
        B, T = leaf_ids.shape
        flat = leaf_ids.reshape(-1)
        offsets = torch.arange(0, B * T, T, dtype=torch.long, device=leaf_ids.device)
        emb = self.embedding(flat, offsets)  # (B, emb_dim)
        feat = torch.cat([x_num, emb], dim=1)
        logits = self.mlp(feat)
        return logits


# ----------------------------- Dataset -----------------------------
class _DeepDS(Dataset):
    def __init__(self, X: np.ndarray, leaf: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.leaf = np.asarray(leaf, dtype=np.int64)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]).float(),
            torch.from_numpy(self.leaf[idx]).long(),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# ----------------------------- Main Model -----------------------------
class DeepGBMCOClassifier:
    def __init__(
        self,
        horizon: int = 24,
        xgb_params: dict | None = None,
        num_buckets: int = 500_000,
        emb_dim: int = 512,
        hidden: tuple[int, ...] = (1024, 512, 256),
        dropout: float = 0.25,
        batch_size: int = 256,
        lr: float = 2e-4,
        weight_decay: float = 1e-3,
        max_epochs: int = 120,
        patience: int = 16,
        min_delta: float = 0.001,
        es_warmup_epochs: int = 5,
        gradient_accumulation_steps: int = 1,
        xgb_weight: float = 0.35,
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
        self.xgb_weight = float(xgb_weight)
        self.seed = int(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Default xgb params (will be mapped to xgb.train params)
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

        self.scaler = None
        self.xgb_model = None  # will hold xgboost.Booster
        self.deep_model = None
        self.leaf_hasher = LeafHasher(num_buckets=self.num_buckets)
        self.results = {}
        self.feature_names = None

        self.label_mapping = {"low": 0, "mid": 1, "high": 2}
        self.rev_mapping = {v: k for k, v in self.label_mapping.items()}

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _discretize_naive(self, s: pd.Series) -> pd.Series:
        return pd.cut(s, bins=[-np.inf, 1.5, 2.5, np.inf], labels=["low", "mid", "high"])

    def _prepare_xgb_params_for_train(self):
        # Convert sklearn wrapper style params to xgb.train params
        params = {}
        # objective and num_class for multiclass
        params["objective"] = "multi:softprob"
        params["num_class"] = 3
        # map common keys
        params["max_depth"] = int(self.xgb_params.get("max_depth", 6))
        params["eta"] = float(self.xgb_params.get("learning_rate", 0.05))
        params["subsample"] = float(self.xgb_params.get("subsample", 1.0))
        params["colsample_bytree"] = float(self.xgb_params.get("colsample_bytree", 1.0))
        params["min_child_weight"] = float(self.xgb_params.get("min_child_weight", 1.0))
        params["gamma"] = float(self.xgb_params.get("gamma", 0.0))
        params["lambda"] = float(self.xgb_params.get("reg_lambda", 1.0))
        params["alpha"] = float(self.xgb_params.get("reg_alpha", 0.0))
        # tree method - choose gpu_hist when CUDA available
        if torch.cuda.is_available():
            params["tree_method"] = "gpu_hist"
        else:
            params["tree_method"] = self.xgb_params.get("tree_method", "hist")
        # eval metric
        params["eval_metric"] = self.xgb_params.get("eval_metric", "mlogloss")
        # nthread / verbosity
        params["verbosity"] = 1
        return params

    def load_data(self, data_path: str | Path):
        """
        读取 data_path/h{horizon}/(train|valid|test).parquet
        并对特征做 StandardScaler
        """
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
        X_train = train.drop(columns=drop_cols, errors="ignore")
        y_train = train[target_col]
        X_valid = valid.drop(columns=drop_cols, errors="ignore")
        y_valid = valid[target_col]
        X_test = test.drop(columns=drop_cols, errors="ignore")
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
        y_train_enc = y_train.map(self.label_mapping).astype("int64").to_numpy()
        y_valid_enc = y_valid.map(self.label_mapping).astype("int64").to_numpy()
        y_test_enc = y_test.map(self.label_mapping).astype("int64").to_numpy()

        print("[5/7] Computing class weights for imbalance handling...")
        cls_counts = pd.Series(y_train_enc).value_counts().sort_index()
        class_weights = (cls_counts.sum() / (len(cls_counts) * cls_counts)).to_dict()
        sw_train = pd.Series(y_train_enc).map(class_weights).values
        sw_valid = pd.Series(y_valid_enc).map(class_weights).values

        # ---------------- XGBoost training using xgb.train ----------------
        print("[6/7] Training XGBoost base learner (xgb.train with DMatrix)...")
        print(f"      XGB params: n_estimators={self.xgb_params.get('n_estimators')}, depth={self.xgb_params.get('max_depth')}, lr={self.xgb_params.get('learning_rate')}")
        t_xgb = time.time()

        dtrain = xgb.DMatrix(X_train_np, label=y_train_enc, weight=sw_train)
        dvalid = xgb.DMatrix(X_valid_np, label=y_valid_enc, weight=sw_valid)

        train_params = self._prepare_xgb_params_for_train()
        num_boost_round = int(self.xgb_params.get("n_estimators", 800))
        early_stopping_rounds = 50

        evals = [(dtrain, "train"), (dvalid, "valid")]

        # train with early stopping via xgb.train
        bst = xgb.train(
            params=train_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=True,
        )
        self.xgb_model = bst  # store Booster
        xgb_time = time.time() - t_xgb

        best_iter = getattr(bst, "best_iteration", None)
        if best_iter is None:
            print(f"      ✓ XGBoost trained in {xgb_time:.2f}s (early stopping: N/A)")
        else:
            print(f"      ✓ XGBoost trained in {xgb_time:.2f}s (best iteration: {best_iter})")

        # ---------------- Extract leaf indices and hashing ----------------
        print("[7/7] Extracting leaf indices and hashing...")
        t_leaf = time.time()
        try:
            leaves_train = bst.predict(dtrain, pred_leaf=True)
            leaves_valid = bst.predict(dvalid, pred_leaf=True)
            dtest = xgb.DMatrix(X_test_np)
            leaves_test = bst.predict(dtest, pred_leaf=True)
        except Exception:
            # fallback: try using sklearn wrapper apply (unlikely here)
            try:
                sk_wrapper = xgb.XGBClassifier(**self.xgb_params)
                sk_wrapper.fit(X_train_np, y_train_enc)
                leaves_train = sk_wrapper.apply(X_train_np)
                leaves_valid = sk_wrapper.apply(X_valid_np)
                leaves_test = sk_wrapper.apply(X_test_np)
                # override stored model with wrapper for prediction compatibility
                self.xgb_model = sk_wrapper
            except Exception as e:
                raise RuntimeError("Failed to extract leaf indices from XGBoost model.") from e

        hashed_train = self.leaf_hasher.transform(np.asarray(leaves_train))
        hashed_valid = self.leaf_hasher.transform(np.asarray(leaves_valid))
        hashed_test = self.leaf_hasher.transform(np.asarray(leaves_test))
        print(f"      ✓ Leaves extracted & hashed in {time.time() - t_leaf:.2f}s")

        # assign
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

        print(f"\n✓ Data loaded: Train={self.train_X.shape}, Valid={self.valid_X.shape}, Test={self.test_X.shape}")
        print(f"  Total time: {time.time() - t_start:.2f}s")
        print("\nClass distribution (train):")
        print(pd.Series(self.train_y).value_counts(normalize=True).sort_index())
        return self

    # ---------------- build deep ----------------
    def _build_deep(self):
        print("\n[Building Deep component (large embedding + deep MLP)...]")
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

    # ---------------- train deep ----------------
    def train(self):
        print(f"\n{'='*60}\nTraining DeepGBM Deep Component (h+{self.horizon})\n{'='*60}")
        self._build_deep()

        # class weights for CE
        cls_counts = pd.Series(self.train_y).value_counts().sort_index()
        weights = cls_counts.sum() / (len(cls_counts) * cls_counts)
        class_weights = torch.tensor(weights.values, dtype=torch.float32, device=self.device)
        print(f"  Class weights: {dict(zip(['low', 'mid', 'high'], weights.tolist()))}")

        train_ds = _DeepDS(self.train_X, self.train_leaf, self.train_y)
        valid_ds = _DeepDS(self.valid_X, self.valid_leaf, self.valid_y)
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.valid_loader = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        print(f"  Batch size: {self.batch_size}, LR: {self.lr}, Weight decay: {self.weight_decay}")
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

        start_time = time.time()
        print(f"\nStarting Deep training (max {self.max_epochs} epochs, patience={self.patience}, warmup={self.es_warmup_epochs})...")
        epoch_iter = range(1, self.max_epochs + 1)
        if tqdm is not None:
            epoch_iter = tqdm(epoch_iter, desc=f"DeepGBM h+{self.horizon} Training", total=self.max_epochs, unit="epoch")

        accumulation = max(1, self.gradient_accumulation_steps)
        for epoch in epoch_iter:
            epoch_start = time.time()
            self.deep_model.train()
            loss_sum = 0.0
            n_sum = 0
            optimizer.zero_grad()
            inner = self.train_loader
            if tqdm is not None:
                inner = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.max_epochs}", leave=False, unit="batch")
            for step, (xb, lb, yb) in enumerate(inner):
                xb = xb.to(self.device, non_blocking=True)
                lb = lb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                if scaler is not None:
                    with autocast():
                        logits = self.deep_model(xb, lb)
                        loss = criterion(logits, yb) / accumulation
                    scaler.scale(loss).backward()
                else:
                    logits = self.deep_model(xb, lb)
                    loss = criterion(logits, yb) / accumulation
                    loss.backward()

                loss_sum += float(loss.item()) * xb.size(0) * accumulation
                n_sum += xb.size(0)

                if (step + 1) % accumulation == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.deep_model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.deep_model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad()

            # Validate
            val_metrics = self._eval_split(self.valid_X, self.valid_leaf, self.valid_y)
            scheduler.step(val_metrics["f1_macro"])
            epoch_time = time.time() - epoch_start

            msg = (f"Epoch {epoch:02d}/{self.max_epochs} | loss={loss_sum/max(1, n_sum):.6f} | "
                  f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['f1_macro']:.4f} | {epoch_time:.1f}s")
            if tqdm is not None and hasattr(epoch_iter, 'write'):
                epoch_iter.write(msg)
            else:
                print(msg)

            current_lr = optimizer.param_groups[0]['lr']
            if tqdm is not None and hasattr(epoch_iter, 'write'):
                epoch_iter.write(f"  Current LR: {current_lr:.6f}")
            else:
                print(f"  Current LR: {current_lr:.6f}")

            # Warmup handling
            if epoch <= self.es_warmup_epochs:
                if val_metrics["f1_macro"] > best_f1:
                    best_f1 = val_metrics["f1_macro"]
                    best_state = {k: v.cpu().clone() for k, v in self.deep_model.state_dict().items()}
                    print(f"  ✓ Warmup improve: best F1 -> {best_f1:.4f}")
                continue

            if val_metrics["f1_macro"] > best_f1 + self.min_delta:
                best_f1 = val_metrics["f1_macro"]
                best_state = {k: v.cpu().clone() for k, v in self.deep_model.state_dict().items()}
                patience_left = self.patience
                print(f"  ✓ New best F1: {best_f1:.4f} (patience reset)")
                # save checkpoint
                ckpt = {"state_dict": best_state, "epoch": epoch, "val_f1": float(best_f1)}
                try:
                    torch.save(ckpt, self.checkpoint_dir / f"deep_best_h{self.horizon}.pt")
                except Exception:
                    pass
            else:
                patience_left -= 1
                print(f"  No significant improvement (Δ<{self.min_delta:.4f}), patience {patience_left}/{self.patience}")
                if patience_left <= 0:
                    print(f"\n⚠ Early stopping at epoch {epoch} (no improvement >= {self.min_delta} for {self.patience} epochs)")
                    break

        if best_state is not None:
            self.deep_model.load_state_dict(best_state)
        self.training_time_seconds = float(time.time() - start_time)
        print(f"\n✓ Deep training completed in {self.training_time_seconds:.2f}s ({self.training_time_seconds/60:.2f} min) | Best val F1: {best_f1:.4f}")
        return self

    # ---------------- evaluation / inference ----------------
    def _xgb_predict_proba(self, X_np: np.ndarray):
        """
        Return XGBoost predicted probabilities for input numpy array.
        Works for Booster (xgb.train result) or sklearn wrapper.
        """
        try:
            if isinstance(self.xgb_model, xgb.core.Booster):
                dmat = xgb.DMatrix(X_np)
                proba = self.xgb_model.predict(dmat)
                return proba
            else:
                # sklearn wrapper
                return self.xgb_model.predict_proba(X_np)
        except Exception:
            return None

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

        # XGBoost probabilities（用于融合）
        xgb_proba = self._xgb_predict_proba(X_np)

        acc = accuracy_score(y_np, pred_all)
        f1_macro = f1_score(y_np, pred_all, average="macro")
        f1_weighted = f1_score(y_np, pred_all, average="weighted")
        precision, recall, f1_per, support = precision_recall_fscore_support(y_np, pred_all, average=None, labels=[0,1,2])
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
            "xgb_proba": xgb_proba,
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

        def fuse_probs(deep_prob, xgb_prob, weight_xgb=self.xgb_weight):
            if xgb_prob is None:
                return deep_prob
            w = float(weight_xgb)
            return (1 - w) * deep_prob + w * xgb_prob

        def report(name: str, res, y_true_enc: np.ndarray, naive_series):
            # 尝试融合
            fused_pred = res["pred"]
            if res.get("xgb_proba") is not None:
                fused_prob = fuse_probs(res["prob"], res["xgb_proba"], weight_xgb=self.xgb_weight)
                fused_pred = fused_prob.argmax(axis=1)

            y_true = decode(y_true_enc)
            y_pred = decode(fused_pred)
            print(f"\n{name.capitalize()} Results (融合 XGB weight={self.xgb_weight}):\n" + "─" * 60)
            acc = accuracy_score(y_true, y_pred)
            f1m = f1_score(y_true, y_pred, average="macro")
            f1w = f1_score(y_true, y_pred, average="weighted")
            print(f"Accuracy:         {acc:.4f}")
            print(f"F1-Score (Macro): {f1m:.4f}")
            print(f"F1-Score (Wtd):   {f1w:.4f}")
            if naive_series is not None:
                naive_acc = accuracy_score(y_true, naive_series)
                naive_f1 = f1_score(y_true, naive_series, average="macro")
                print("\nNaive Baseline:")
                print(f"  Accuracy:  {naive_acc:.4f} (Δ: {acc - naive_acc:+.4f})")
                print(f"  F1-Macro:  {naive_f1:.4f} (Δ: {f1m - naive_f1:+.4f})")
            print("\nDetailed Classification Report:")
            print(classification_report(y_true, y_pred, digits=4))

        report("train", self.results["train"], self.train_y, self.naive_train)
        report("valid", self.results["valid"], self.valid_y, self.naive_valid)
        report("test", self.results["test"], self.test_y, self.naive_test)
        return self

    def plot_confusion_matrices(self, save_path=None):
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
            print(f"\nConfusion matrices saved to: {save_path}")
        plt.show()

    def save_model(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save xgboost booster to native model file
        try:
            if isinstance(self.xgb_model, xgb.core.Booster):
                self.xgb_model.save_model(str(output_dir / f"deepgbm_xgb_h{self.horizon}.model"))
            else:
                joblib.dump(self.xgb_model, output_dir / f"deepgbm_xgb_h{self.horizon}.joblib")
        except Exception:
            # fallback to joblib
            joblib.dump(self.xgb_model, output_dir / f"deepgbm_xgb_h{self.horizon}.joblib")

        torch.save({
            "state_dict": self.deep_model.state_dict(),
            "params": {
                "n_features": len(self.feature_names),
                "num_buckets": self.num_buckets,
                "emb_dim": self.emb_dim,
                "hidden": self.hidden,
                "dropout": self.dropout,
            }
        }, output_dir / f"deepgbm_deep_h{self.horizon}.pt")
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
                    "per_class_f1": {lab: float(f1) for lab, f1 in zip(["low", "mid", "high"], res["f1_per_class"])}
                } for ds, res in self.results.items()
            }
        }
        with open(output_dir / f"deepgbm_results_h{self.horizon}.json", "w") as f:
            json.dump(results_summary, f, indent=2)
        print(f"Models & results saved to: {output_dir}")
        return self


# ----------------------------- main -----------------------------
def main():
    DATA_PATH = Path("/app/data_artifacts/splits")  # 修改为你的数据路径
    OUTPUT_DIR = Path("/app/classification-analysis/deepgbm")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    horizons = [1, 6, 12, 24]

    for h in horizons:
        print("\n" + "#" * 80)
        print(f"# HORIZON: {h} HOUR(S)")
        print("#" * 80)

        # 针对不同 horizon 的微调（可按需修改）
        if h == 1:
            cfg = dict(lr=1e-3, patience=16, min_delta=0.001, es_warmup_epochs=3, max_epochs=80)
        elif h == 6:
            cfg = dict(lr=5e-4, patience=18, min_delta=0.0015, es_warmup_epochs=4, max_epochs=100)
        else:  # h == 12 or 24
            cfg = dict(lr=3e-4, patience=20, min_delta=0.002, es_warmup_epochs=5, max_epochs=120)

        clf = DeepGBMCOClassifier(
            horizon=h,
            num_buckets=500_000,
            emb_dim=512,
            hidden=(1024, 512, 256),
            dropout=0.25,
            batch_size=8,  # For CPU low-memory run, keep small; on GPU increase
            lr=cfg['lr'],
            weight_decay=1e-3,
            max_epochs=cfg['max_epochs'],
            patience=cfg['patience'],
            min_delta=cfg['min_delta'],
            es_warmup_epochs=cfg['es_warmup_epochs'],
            gradient_accumulation_steps=1,
            xgb_weight=0.35,
            seed=42,
            device=None,  # 自动选择 CUDA（若可用）
            checkpoint_dir=OUTPUT_DIR / "checkpoints",
        )

        # 加载数据并训练
        clf.load_data(DATA_PATH)
        clf.train()
        clf.evaluate_all()

        out_dir = OUTPUT_DIR / f"h{h}"
        out_dir.mkdir(parents=True, exist_ok=True)
        clf.plot_confusion_matrices(save_path=out_dir / f"confusion_matrices_h{h}.png")
        clf.save_model(out_dir)

        print("\n" + "=" * 80)
        print(f"Completed horizon h+{h}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
