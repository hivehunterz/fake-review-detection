"""
Fine-tune Fake Review Detection Model with Anomaly Data

This script fine-tunes an existing model using only the anomaly-labeled data
from the metadata analysis pipeline. This helps improve detection of suspicious patterns.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils.class_weight import compute_class_weight

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnomalyFineTuner:
    def __init__(self, anomaly_data_path=None, model_path=None):
        """Initialize the AnomalyFineTuner."""
        self.anomaly_data_path = Path(anomaly_data_path) if anomaly_data_path else None
        self.model_path = Path(model_path) if model_path else Path("models/anomaly_model.joblib")
        self.model = None
        self.feature_columns = None
        
        # Ensure model directory exists
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing model
        if self.model_path.exists():
            self.load_model()
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
    
    def load_anomaly_data(self):
        """Load and prepare anomaly data for fine-tuning."""
        logger.info(f"Loading anomaly data from {self.anomaly_data_path}")
        df = pd.read_csv(self.anomaly_data_path)
        
        # Pivot to wide form (review_id × features)
        features = df.pivot_table(
            index='review_id',
            columns='feature',
            values='score',
            aggfunc='first'
        ).fillna(0)
        features = features.add_prefix('anomaly_')
        
        # Labels
        if 'label' in df.columns:
            labels = df.groupby('review_id')['label'].agg(lambda x: x.mode()[0])
            labels = labels.astype(int)  # already 0/1
        else:
            logger.warning("No labels found in anomaly data — defaulting to 0")
            labels = pd.Series(0, index=features.index, name='label')
        
        result_df = pd.DataFrame({'label': labels}).join(features, how='inner')
        self.feature_columns = [c for c in result_df.columns if c.startswith('anomaly_')]
        
        if not self.feature_columns:
            raise ValueError("No anomaly features found in data")
        
        logger.info(f"Loaded {len(result_df)} reviews with {len(self.feature_columns)} features")
        logger.info(f"Label distribution:\n{result_df['label'].value_counts()}")
        return result_df
    
    def train(self):
        """Train and evaluate the anomaly detection model."""
        data = self.load_anomaly_data()
        X, y = data[self.feature_columns], data['label']
        
        if len(np.unique(y)) < 2:
            logger.error("Training aborted: only one class present in data.")
            return
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {cls: w for cls, w in zip(np.unique(y_train), class_weights)}
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_metrics = self.evaluate_model(X_train, y_train, "Training")
        test_metrics = self.evaluate_model(X_test, y_test, "Test")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='f1', n_jobs=-1)
        logger.info(f"Cross-validation F1 scores: {cv_scores}")
        
        self.save_model()
        return {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "cv_scores": cv_scores.tolist(),
            "mean_cv_score": float(cv_scores.mean())
        }
    
    def evaluate_model(self, X, y_true, set_name="Test"):
        """Evaluate the model on a dataset."""
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
        
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        
        logger.info(f"\n{set_name} Set Evaluation")
        logger.info(classification_report(y_true, y_pred, zero_division=0))
        
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        metrics = {
            "accuracy": (y_pred == y_true).mean(),
            "confusion_matrix": cm.tolist()
        }
        if len(set(y_true)) > 1:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            metrics["avg_precision"] = average_precision_score(y_true, y_proba)
        
        return metrics
    
    def save_model(self):
        """Save the trained model."""
        joblib.dump({
            "model": self.model,
            "feature_columns": self.feature_columns,
            "model_type": "RandomForestClassifier"
        }, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self, path=None):
        """Load a trained model from disk."""
        load_path = Path(path) if path else self.model_path
        model_data = joblib.load(load_path)
        self.model = model_data["model"]
        self.feature_columns = model_data.get("feature_columns")
        logger.info(f"Model loaded from {load_path}")
        return self.model


def main():
    base_dir = Path(__file__).parent
    anomaly_data_path = base_dir / "output" / "anomaly_training_data.csv"
    model_path = base_dir / "models" / "anomaly_model.joblib"
    
    fine_tuner = AnomalyFineTuner(anomaly_data_path=anomaly_data_path, model_path=model_path)
    fine_tuner.train()
    logger.info("✅ Anomaly model training completed successfully.")


if __name__ == "__main__":
    sys.exit(main())
