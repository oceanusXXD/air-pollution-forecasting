"""
XGBoost Classifier for CO Pollution Level Prediction
Predicts CO concentration levels (low, mid, high) at different time horizons
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
from datetime import datetime
import time
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


class XGBoostCOClassifier:
    """XGBoost classifier for CO level prediction across multiple horizons"""
    
    def __init__(self, horizon=24, n_estimators=100, max_depth=6, 
                 learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                 min_child_weight=1.0, gamma=0.0, reg_lambda=1.0, reg_alpha=0.0,
                 early_stopping_rounds=30, random_state=42):
        """
        Initialize XGBoost Classifier
        
        Args:
            horizon: Prediction horizon (1, 6, 12, or 24 hours)
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            min_child_weight: Minimum sum of instance weight(hessian) needed in a child
            gamma: Minimum loss reduction required to make a further partition
            reg_lambda: L2 regularization term on weights
            reg_alpha: L1 regularization term on weights
            early_stopping_rounds: Early stopping rounds on validation set
            random_state: Random state for reproducibility
        """
        self.horizon = horizon
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.early_stopping_rounds = early_stopping_rounds
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            random_state=random_state,
            tree_method='hist',
            enable_categorical=False,
            eval_metric='mlogloss',
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=-1
        )
        
        self.feature_importance = None
        self.results = {}
        self.training_history = None
        
    def load_data(self, data_path):
        """Load train, validation, and test data"""
        base_path = Path(data_path) / f"h{self.horizon}"
        
        print(f"\n{'='*60}")
        print(f"Loading data for horizon h+{self.horizon}")
        print(f"{'='*60}")
        
        train = pd.read_parquet(base_path / "train.parquet")
        valid = pd.read_parquet(base_path / "valid.parquet")
        test = pd.read_parquet(base_path / "test.parquet")
        
        # Prepare features and target
        target_col = f"co_level_t+{self.horizon}"
        drop_cols = [f"y_t+{self.horizon}", f"naive_yhat_t+{self.horizon}", target_col]
        
        self.X_train = train.drop(columns=drop_cols)
        self.y_train = train[target_col]
        
        self.X_valid = valid.drop(columns=drop_cols)
        self.y_valid = valid[target_col]
        
        self.X_test = test.drop(columns=drop_cols)
        self.y_test = test[target_col]
        
        # Encode labels to integers for XGBoost
        self.label_mapping = {'low': 0, 'mid': 1, 'high': 2}
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        
        self.y_train_encoded = self.y_train.map(self.label_mapping)
        self.y_valid_encoded = self.y_valid.map(self.label_mapping)
        self.y_test_encoded = self.y_test.map(self.label_mapping)
        
        # Store naive baseline for comparison
        # Prefer discretizing provided naive numeric forecast to labels; fall back to shift if missing
        def _discretize_to_level(s: pd.Series) -> pd.Series:
            return pd.cut(
                s,
                bins=[-np.inf, 1.5, 2.5, np.inf],
                labels=['low', 'mid', 'high']
            )

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
        print(f"\nClass distribution (train):")
        print(self.y_train.value_counts(normalize=True).sort_index())
        
        return self
    
    def train(self):
        """Train the XGBoost model with early stopping"""
        print(f"\n{'='*60}")
        print(f"Training XGBoost Classifier (h+{self.horizon})")
        print(f"{'='*60}")
        print(f"Parameters: n_estimators={self.n_estimators}, max_depth={self.max_depth}, "
              f"lr={self.learning_rate}, min_child_weight={self.min_child_weight}, "
              f"gamma={self.gamma}, subsample={self.subsample}, colsample_bytree={self.colsample_bytree}, "
              f"reg_lambda={self.reg_lambda}, reg_alpha={self.reg_alpha}")
        
        # Compute class weights -> sample weights (to mitigate imbalance)
        class_counts = self.y_train_encoded.value_counts().sort_index()
        class_weights = (class_counts.sum() / (len(class_counts) * class_counts)).to_dict()
        sample_weight_train = self.y_train_encoded.map(class_weights).values
        sample_weight_valid = self.y_valid_encoded.map(class_weights).values

        start_time = time.time()
        # Train with early stopping and sample weights
        # Note: callbacks parameter removed for XGBoost 2.x compatibility
        self.model.fit(
            self.X_train,
            self.y_train_encoded,
            eval_set=[(self.X_valid, self.y_valid_encoded)],
            sample_weight=sample_weight_train,
            sample_weight_eval_set=[sample_weight_valid],
            verbose=True  # Enable verbose to see training progress
        )
        self.training_time_seconds = float(time.time() - start_time)
        
        # Get best iteration
        best_iteration = self.model.best_iteration
        print(f"\nBest iteration: {best_iteration}")
        
        # Get feature importance
        importance_dict = self.model.get_booster().get_score(importance_type='gain')
        self.feature_importance = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        }).sort_values('importance', ascending=False)
        
        # Map feature names back
        feature_map = {f'f{i}': name for i, name in enumerate(self.X_train.columns)}
        self.feature_importance['feature'] = self.feature_importance['feature'].map(
            lambda x: feature_map.get(x, x)
        )
        
        print("\nTop 10 Most Important Features:")
        print(self.feature_importance.head(10))
        
        # Store training history
        self.training_history = self.model.evals_result()
        
        return self
    
    def evaluate(self, X, y_encoded, y_original, dataset_name="Dataset", naive_baseline=None):
        """Evaluate model on given dataset"""
        y_pred_encoded = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        
        # Convert predictions back to original labels
        y_pred = pd.Series(y_pred_encoded).map(self.reverse_mapping)
        
        # Calculate metrics
        acc = accuracy_score(y_original, y_pred)
        f1_macro = f1_score(y_original, y_pred, average='macro')
        f1_weighted = f1_score(y_original, y_pred, average='weighted')
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_original, y_pred, average=None, labels=['low', 'mid', 'high']
        )
        
        print(f"\n{dataset_name} Results:")
        print(f"{'─'*60}")
        print(f"Accuracy:         {acc:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"F1-Score (Wtd):   {f1_weighted:.4f}")
        
        # Naive baseline comparison
        if naive_baseline is not None:
            naive_acc = accuracy_score(y_original, naive_baseline)
            naive_f1 = f1_score(y_original, naive_baseline, average='macro')
            print(f"\nNaive Baseline:")
            print(f"  Accuracy:  {naive_acc:.4f} (Δ: {acc - naive_acc:+.4f})")
            print(f"  F1-Macro:  {naive_f1:.4f} (Δ: {f1_macro - naive_f1:+.4f})")
        
        print(f"\nPer-Class Metrics:")
        for i, label in enumerate(['low', 'mid', 'high']):
            print(f"  {label:>4}: P={precision[i]:.3f}, R={recall[i]:.3f}, "
                  f"F1={f1[i]:.3f}, Support={int(support[i])}")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_original, y_pred, digits=4))
        
        return {
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'f1_per_class': f1,
            'support': support,
            'confusion_matrix': confusion_matrix(y_original, y_pred),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def evaluate_all(self):
        """Evaluate on train, validation, and test sets"""
        print(f"\n{'='*60}")
        print(f"EVALUATION - XGBoost (h+{self.horizon})")
        print(f"{'='*60}")
        
        self.results['train'] = self.evaluate(
            self.X_train, self.y_train_encoded, self.y_train, 
            "Training Set", self.naive_train
        )
        self.results['valid'] = self.evaluate(
            self.X_valid, self.y_valid_encoded, self.y_valid,
            "Validation Set", self.naive_valid
        )
        self.results['test'] = self.evaluate(
            self.X_test, self.y_test_encoded, self.y_test,
            "Test Set", self.naive_test
        )
        
        return self
    
    def plot_confusion_matrices(self, save_path=None):
        """Plot confusion matrices for all datasets"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (dataset, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                       xticklabels=['low', 'mid', 'high'],
                       yticklabels=['low', 'mid', 'high'],
                       ax=axes[idx])
            axes[idx].set_title(f'{dataset.capitalize()} Set\n'
                               f'Acc: {results["accuracy"]:.3f}, '
                               f'F1: {results["f1_macro"]:.3f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.suptitle(f'XGBoost - Confusion Matrices (h+{self.horizon})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nConfusion matrices saved to: {save_path}")
        
        plt.show()
        
    def plot_feature_importance(self, top_n=20, save_path=None):
        """Plot top N most important features"""
        plt.figure(figsize=(10, 8))
        
        top_features = self.feature_importance.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance (Gain)')
        plt.title(f'Top {top_n} Feature Importances - XGBoost (h+{self.horizon})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFeature importance plot saved to: {save_path}")
        
        plt.show()
    
    def plot_training_history(self, save_path=None):
        """Plot training history (loss curve)"""
        if self.training_history is None:
            print("No training history available")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Get validation loss
        val_loss = self.training_history['validation_0']['mlogloss']
        epochs = range(1, len(val_loss) + 1)
        
        plt.plot(epochs, val_loss, 'b-', label='Validation Loss', linewidth=2)
        plt.axvline(x=self.model.best_iteration, color='r', linestyle='--', 
                   label=f'Best Iteration ({self.model.best_iteration})')
        
        plt.xlabel('Iteration')
        plt.ylabel('Log Loss')
        plt.title(f'XGBoost Training History (h+{self.horizon})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nTraining history plot saved to: {save_path}")
        
        plt.show()
        
    def save_model(self, output_dir):
        """Save model and results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / f"xgb_classifier_h{self.horizon}.joblib"
        joblib.dump(self.model, model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save feature importance
        importance_path = output_dir / f"xgb_feature_importance_h{self.horizon}.csv"
        self.feature_importance.to_csv(importance_path, index=False)
        
        # Save results summary
        results_summary = {
            'horizon': self.horizon,
            'model': 'XGBoost',
            'timestamp': datetime.now().isoformat(),
            'training_time_seconds': getattr(self, 'training_time_seconds', None),
            'parameters': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'min_child_weight': self.min_child_weight,
                'gamma': self.gamma,
                'reg_lambda': self.reg_lambda,
                'reg_alpha': self.reg_alpha,
                'early_stopping_rounds': self.early_stopping_rounds,
                'random_state': self.random_state,
                'best_iteration': int(self.model.best_iteration)
            },
            'results': {
                dataset: {
                    'accuracy': float(res['accuracy']),
                    'f1_macro': float(res['f1_macro']),
                    'f1_weighted': float(res['f1_weighted']),
                    'per_class_f1': {
                        label: float(f1) 
                        for label, f1 in zip(['low', 'mid', 'high'], res['f1_per_class'])
                    }
                }
                for dataset, res in self.results.items()
            }
        }
        
        results_path = output_dir / f"xgb_results_h{self.horizon}.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"Results saved to: {results_path}")
        
        return self


def main():
    """Main execution function"""
    # Configuration
    DATA_PATH = Path("/app/data_artifacts/splits")
    OUTPUT_DIR = Path("/app/classification-analysis/xgboost")
    
    # Train models for all horizons
    horizons = [1, 6, 12, 24]
    
    # Horizon-specific hyperparameters (stronger regularization for longer horizons)
    horizon_configs = {
        1: {  # Short-term: more aggressive learning
            'n_estimators': 600,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 2.0,
            'gamma': 0.0,
            'reg_lambda': 1.0,
            'reg_alpha': 0.0,
            'early_stopping_rounds': 50
        },
        6: {  # Medium-term: moderate regularization
            'n_estimators': 800,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 3.0,
            'gamma': 0.1,
            'reg_lambda': 2.0,
            'reg_alpha': 0.5,
            'early_stopping_rounds': 100  # 增加早停轮数（原60）
        },
        12: {  # Long-term: strong regularization
            'n_estimators': 1000,
            'max_depth': 5,
            'learning_rate': 0.02,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'min_child_weight': 5.0,
            'gamma': 0.2,
            'reg_lambda': 3.0,
            'reg_alpha': 1.0,
            'early_stopping_rounds': 120  # 增加早停轮数（原80）
        },
        24: {  # Very long-term: strongest regularization
            'n_estimators': 1200,  # 增加树数量（原1000）
            'max_depth': 4,
            'learning_rate': 0.015,  # 降低学习率（原0.02）- 更细致的学习
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'min_child_weight': 5.0,
            'gamma': 0.3,
            'reg_lambda': 4.0,
            'reg_alpha': 1.5,
            'early_stopping_rounds': 150  # 增加早停轮数（原80）
        }
    }
    
    for horizon in horizons:
        print(f"\n{'#'*70}")
        print(f"# HORIZON: {horizon} HOUR(S)")
        print(f"{'#'*70}")
        
        # Get horizon-specific configuration
        config = horizon_configs[horizon]
        
        # Initialize and train model
        clf = XGBoostCOClassifier(
            horizon=horizon,
            **config,
            random_state=42
        )
        
        # Load data and train
        clf.load_data(DATA_PATH)
        clf.train()
        clf.evaluate_all()
        
        # Create visualizations
        output_dir = OUTPUT_DIR / f"h{horizon}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        clf.plot_confusion_matrices(
            save_path=output_dir / f"confusion_matrices_h{horizon}.png"
        )
        clf.plot_feature_importance(
            top_n=20,
            save_path=output_dir / f"feature_importance_h{horizon}.png"
        )
        clf.plot_training_history(
            save_path=output_dir / f"training_history_h{horizon}.png"
        )
        
        # Save model and results
        clf.save_model(OUTPUT_DIR / f"h{horizon}")
        
        print(f"\n{'='*70}")
        print(f"Completed horizon h+{horizon}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
