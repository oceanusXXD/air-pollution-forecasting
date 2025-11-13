"""
XGBoost Classifier for CO Pollution Level Prediction (GPU-optimized)
- Uses xgb.train + DMatrix for robust compatibility across xgboost versions
- Tries to use GPU (gpu_hist) and falls back to CPU (hist) if not available
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
from datetime import datetime
import time
import os
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


class XGBoostCOClassifier:
    """XGBoost classifier for CO level prediction across multiple horizons (GPU-optimized)"""

    def __init__(
        self,
        horizon=24,
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3.0,
        gamma=0.1,
        reg_lambda=2.0,
        reg_alpha=0.5,
        early_stopping_rounds=100,
        random_state=42,
        use_gpu: bool | None = None,
    ):
        self.horizon = horizon
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.min_child_weight = float(min_child_weight)
        self.gamma = float(gamma)
        self.reg_lambda = float(reg_lambda)
        self.reg_alpha = float(reg_alpha)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.random_state = int(random_state)
        # If None: try GPU if xgboost supports it and CUDA visible
        self.use_gpu = use_gpu
        if self.use_gpu is None:
            # heuristic: if CUDA_VISIBLE_DEVICES set or NVIDIA_VISIBLE_DEVICES set, assume GPU
            self.use_gpu = bool(os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("NVIDIA_VISIBLE_DEVICES"))
        self.bst = None
        self.evals_result = None
        self.feature_importance = None
        self.results = {}

    def load_data(self, data_path):
        base_path = Path(data_path) / f"h{self.horizon}"

        print(f"\n{'='*60}")
        print(f"Loading data for horizon h+{self.horizon}")
        print(f"{'='*60}")

        train = pd.read_parquet(base_path / "train.parquet")
        valid = pd.read_parquet(base_path / "valid.parquet")
        test = pd.read_parquet(base_path / "test.parquet")

        target_col = f"co_level_t+{self.horizon}"
        drop_cols = [f"y_t+{self.horizon}", f"naive_yhat_t+{self.horizon}", target_col]

        self.X_train = train.drop(columns=drop_cols, errors="ignore")
        self.y_train = train[target_col]

        self.X_valid = valid.drop(columns=drop_cols, errors="ignore")
        self.y_valid = valid[target_col]

        self.X_test = test.drop(columns=drop_cols, errors="ignore")
        self.y_test = test[target_col]

        # Label mapping
        self.label_mapping = {"low": 0, "mid": 1, "high": 2}
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}

        self.y_train_encoded = self.y_train.map(self.label_mapping).astype(int).values
        self.y_valid_encoded = self.y_valid.map(self.label_mapping).astype(int).values
        self.y_test_encoded = self.y_test.map(self.label_mapping).astype(int).values

        # naive baseline
        def _discretize_to_level(s: pd.Series) -> pd.Series:
            return pd.cut(s, bins=[-np.inf, 1.5, 2.5, np.inf], labels=["low", "mid", "high"])

        naive_col = f"naive_yhat_t+{self.horizon}"
        if naive_col in train.columns and naive_col in valid.columns and naive_col in test.columns:
            self.naive_train = _discretize_to_level(train[naive_col])
            self.naive_valid = _discretize_to_level(valid[naive_col])
            self.naive_test = _discretize_to_level(test[naive_col])
        else:
            print("[WARN] naive_yhat column not found; falling back to shifted labels as naive baseline.")
            self.naive_train = self.y_train.shift(self.horizon).bfill()
            self.naive_valid = self.y_valid.shift(self.horizon).bfill()
            self.naive_test = self.y_test.shift(self.horizon).bfill()

        print(f"Train: {self.X_train.shape}, Valid: {self.X_valid.shape}, Test: {self.X_test.shape}")
        print("\nClass distribution (train):")
        print(self.y_train.value_counts(normalize=True).sort_index())

        return self

    def _build_params(self, use_gpu_flag: bool):
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eta": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "lambda": self.reg_lambda,
            "alpha": self.reg_alpha,
            "seed": self.random_state,
            "verbosity": 1,
            "eval_metric": "mlogloss",
        }
        if use_gpu_flag:
            params["tree_method"] = "gpu_hist"
            # predictor will be auto chosen for gpu_hist; but set gpu_predictor for older versions
            params["predictor"] = "gpu_predictor"
        else:
            params["tree_method"] = "hist"
            params["predictor"] = "auto"
        return params

    def train(self):
        print(f"\n{'='*60}")
        print(f"Training XGBoost Classifier (h+{self.horizon}) - GPU attempt: {self.use_gpu}")
        print(f"{'='*60}")
        print(f"Parameters: n_estimators={self.n_estimators}, max_depth={self.max_depth}, lr={self.learning_rate}, "
              f"min_child_weight={self.min_child_weight}, gamma={self.gamma}, subsample={self.subsample}, "
              f"colsample_bytree={self.colsample_bytree}, reg_lambda={self.reg_lambda}, reg_alpha={self.reg_alpha}")

        # sample weights to mitigate imbalance
        class_counts = pd.Series(self.y_train_encoded).value_counts().sort_index()
        class_weights = (class_counts.sum() / (len(class_counts) * class_counts)).to_dict()
        weight_train = pd.Series(self.y_train.map(self.label_mapping)).map(class_weights).values if hasattr(self, "y_train") else None
        weight_valid = pd.Series(self.y_valid.map(self.label_mapping)).map(class_weights).values if hasattr(self, "y_valid") else None

        # create DMatrix with feature names for correct importance mapping
        dtrain = xgb.DMatrix(self.X_train.values, label=self.y_train_encoded, weight=weight_train, feature_names=list(self.X_train.columns))
        dvalid = xgb.DMatrix(self.X_valid.values, label=self.y_valid_encoded, weight=weight_valid, feature_names=list(self.X_valid.columns))
        dtest = xgb.DMatrix(self.X_test.values, label=self.y_test_encoded, feature_names=list(self.X_test.columns))

        watchlist = [(dtrain, "train"), (dvalid, "valid")]

        # try GPU first (if flagged), fallback to CPU on exception
        tried_gpu = False
        use_gpu_flag = bool(self.use_gpu)
        params = self._build_params(use_gpu_flag)
        evals_result = {}
        start_time = time.time()
        try:
            if use_gpu_flag:
                tried_gpu = True
                print("Attempting GPU training with params:", {k: params[k] for k in ["tree_method", "predictor"] if k in params})
            self.bst = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=watchlist,
                early_stopping_rounds=self.early_stopping_rounds,
                evals_result=evals_result,
                verbose_eval=True,
            )
        except Exception as e:
            print("GPU training failed or not supported. Falling back to CPU hist. Exception:")
            print(e)
            # fallback CPU
            params = self._build_params(False)
            evals_result = {}
            self.bst = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=watchlist,
                early_stopping_rounds=self.early_stopping_rounds,
                evals_result=evals_result,
                verbose_eval=True,
            )

        self.evals_result = evals_result
        self.training_time_seconds = float(time.time() - start_time)
        best_it = getattr(self.bst, "best_iteration", None)
        print(f"\nTraining finished in {self.training_time_seconds:.2f}s | best_iteration: {best_it}")

        # feature importance (gain)
        importance_dict = self.bst.get_score(importance_type="gain")
        fi = pd.DataFrame({"feature": list(importance_dict.keys()), "importance": list(importance_dict.values())})
        # map 'f{i}' -> actual column names if necessary
        feature_map = {f"f{i}": name for i, name in enumerate(self.X_train.columns)}
        fi["feature"] = fi["feature"].map(lambda x: feature_map.get(x, x))
        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
        self.feature_importance = fi

        print("\nTop 10 Features:")
        print(self.feature_importance.head(10))

        return self

    def evaluate(self, X_df, y_encoded, y_original, dataset_name="Dataset", naive_baseline=None):
        dmat = xgb.DMatrix(X_df.values, feature_names=list(X_df.columns))
        proba = self.bst.predict(dmat, iteration_range=(0, getattr(self.bst, "best_iteration", self.n_estimators)))
        # proba shape (N, num_class)
        if proba.ndim == 1:
            # binary? unlikely; fallback
            preds_encoded = (proba > 0.5).astype(int)
            proba_full = np.vstack([1-proba, proba]).T
        else:
            preds_encoded = np.argmax(proba, axis=1)
            proba_full = proba

        preds = pd.Series(preds_encoded).map(self.reverse_mapping)

        acc = accuracy_score(y_original, preds)
        f1_macro = f1_score(y_original, preds, average="macro")
        f1_weighted = f1_score(y_original, preds, average="weighted")
        precision, recall, f1, support = precision_recall_fscore_support(y_original, preds, average=None, labels=["low", "mid", "high"])

        print(f"\n{dataset_name} Results:")
        print("─" * 60)
        print(f"Accuracy:         {acc:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"F1-Score (Wtd):   {f1_weighted:.4f}")

        if naive_baseline is not None:
            naive_acc = accuracy_score(y_original, naive_baseline)
            naive_f1 = f1_score(y_original, naive_baseline, average="macro")
            print("\nNaive Baseline:")
            print(f"  Accuracy:  {naive_acc:.4f} (Δ: {acc - naive_acc:+.4f})")
            print(f"  F1-Macro:  {naive_f1:.4f} (Δ: {f1_macro - naive_f1:+.4f})")

        print("\nPer-Class Metrics:")
        for i, label in enumerate(["low", "mid", "high"]):
            print(f"  {label:>4}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}, Support={int(support[i])}")

        print("\nDetailed Classification Report:")
        print(classification_report(y_original, preds, digits=4))

        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision": precision,
            "recall": recall,
            "f1_per_class": f1,
            "support": support,
            "confusion_matrix": confusion_matrix(y_original, preds),
            "predictions": preds,
            "probabilities": proba_full,
        }

    def evaluate_all(self):
        print(f"\n{'='*60}")
        print(f"EVALUATION - XGBoost (h+{self.horizon})")
        print(f"{'='*60}")

        self.results["train"] = self.evaluate(self.X_train, self.y_train_encoded, self.y_train, "Training Set", self.naive_train)
        self.results["valid"] = self.evaluate(self.X_valid, self.y_valid_encoded, self.y_valid, "Validation Set", self.naive_valid)
        self.results["test"] = self.evaluate(self.X_test, self.y_test_encoded, self.y_test, "Test Set", self.naive_test)
        return self

    def plot_confusion_matrices(self, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for idx, (dataset, results) in enumerate(self.results.items()):
            cm = results["confusion_matrix"]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                        xticklabels=["low", "mid", "high"],
                        yticklabels=["low", "mid", "high"],
                        ax=axes[idx])
            axes[idx].set_title(f"{dataset.capitalize()} Set\nAcc: {results['accuracy']:.3f}, F1: {results['f1_macro']:.3f}")
            axes[idx].set_ylabel("True Label")
            axes[idx].set_xlabel("Predicted Label")
        plt.suptitle(f"XGBoost - Confusion Matrices (h+{self.horizon})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\nConfusion matrices saved to: {save_path}")
        plt.show()

    def plot_feature_importance(self, top_n=20, save_path=None):
        if self.feature_importance is None:
            print("No feature importance available")
            return
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        plt.barh(range(len(top_features)), top_features["importance"])
        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.xlabel("Feature Importance (Gain)")
        plt.title(f"Top {top_n} Feature Importances - XGBoost (h+{self.horizon})")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\nFeature importance plot saved to: {save_path}")
        plt.show()

    def plot_training_history(self, save_path=None):
        if not self.evals_result:
            print("No training history available")
            return
        # validation logloss
        val_loss = self.evals_result.get("valid", {}).get("mlogloss", [])
        epochs = range(1, len(val_loss) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, val_loss, "b-", label="Validation mlogloss", linewidth=2)
        best_it = getattr(self.bst, "best_iteration", None)
        if best_it is not None:
            plt.axvline(x=best_it, color="r", linestyle="--", label=f"Best Iter ({best_it})")
        plt.xlabel("Iteration")
        plt.ylabel("Log Loss")
        plt.title(f"XGBoost Training History (h+{self.horizon})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\nTraining history plot saved to: {save_path}")
        plt.show()

    def save_model(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # save booster
        model_path = output_dir / f"xgb_booster_h{self.horizon}.model"
        self.bst.save_model(str(model_path))
        print(f"\nBooster saved to: {model_path}")

        # save feature importance
        if self.feature_importance is not None:
            fi_path = output_dir / f"xgb_feature_importance_h{self.horizon}.csv"
            self.feature_importance.to_csv(fi_path, index=False)
        # save evals_result
        if self.evals_result is not None:
            with open(output_dir / f"xgb_evals_h{self.horizon}.json", "w") as f:
                json.dump(self.evals_result, f, indent=2)

        # save summary
        results_summary = {
            "horizon": self.horizon,
            "model": "XGBoost",
            "timestamp": datetime.now().isoformat(),
            "training_time_seconds": getattr(self, "training_time_seconds", None),
            "parameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "min_child_weight": self.min_child_weight,
                "gamma": self.gamma,
                "reg_lambda": self.reg_lambda,
                "reg_alpha": self.reg_alpha,
                "early_stopping_rounds": self.early_stopping_rounds,
                "use_gpu": self.use_gpu,
            },
            "results_available": list(self.results.keys()),
        }
        with open(output_dir / f"xgb_results_h{self.horizon}.json", "w") as f:
            json.dump(results_summary, f, indent=2)

        print(f"Results & artifacts saved to: {output_dir}")
        return self


def main():
    DATA_PATH = Path("/app/data_artifacts/splits")
    OUTPUT_DIR = Path("/app/classification-analysis/xgboost_gpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # GPU-friendly stronger configs (for H100)
    horizon_configs = {
        1: {
            "n_estimators": 1200,
            "max_depth": 10,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 2.0,
            "gamma": 0.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "early_stopping_rounds": 80,
        },
        6: {
            "n_estimators": 1200,
            "max_depth": 8,
            "learning_rate": 0.025,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "min_child_weight": 3.0,
            "gamma": 0.1,
            "reg_lambda": 2.0,
            "reg_alpha": 0.5,
            "early_stopping_rounds": 100,
        },
        12: {
            "n_estimators": 1500,
            "max_depth": 7,
            "learning_rate": 0.02,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "min_child_weight": 4.0,
            "gamma": 0.2,
            "reg_lambda": 3.0,
            "reg_alpha": 1.0,
            "early_stopping_rounds": 120,
        },
        24: {
            "n_estimators": 1800,
            "max_depth": 6,
            "learning_rate": 0.015,
            "subsample": 0.65,
            "colsample_bytree": 0.65,
            "min_child_weight": 5.0,
            "gamma": 0.3,
            "reg_lambda": 4.0,
            "reg_alpha": 1.5,
            "early_stopping_rounds": 150,
        },
    }

    horizons = [1, 6, 12, 24]
    for h in horizons:
        print("\n" + "#" * 70)
        print(f"# HORIZON: {h} HOUR(S)")
        print("#" * 70)
        cfg = horizon_configs[h]
        clf = XGBoostCOClassifier(
            horizon=h,
            use_gpu=True,  # 尝试用 GPU（如果容器/驱动/库支持则使用）
            random_state=42,
            **cfg,
        )
        clf.load_data(DATA_PATH)
        clf.train()
        clf.evaluate_all()

        out_dir = OUTPUT_DIR / f"h{h}"
        out_dir.mkdir(parents=True, exist_ok=True)
        clf.plot_confusion_matrices(save_path=out_dir / f"confusion_matrices_h{h}.png")
        clf.plot_feature_importance(top_n=30, save_path=out_dir / f"feature_importance_h{h}.png")
        clf.plot_training_history(save_path=out_dir / f"training_history_h{h}.png")
        clf.save_model(out_dir)

        print("\n" + "=" * 70)
        print(f"Completed horizon h+{h}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
