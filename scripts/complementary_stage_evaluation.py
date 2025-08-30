"""
🔬 STAGE COMPLEMENTARY EVALUATION
Evaluates how well each stage complements others in the multi-stage pipeline.
Focuses on coverage, complementary value, and business impact rather than just accuracy.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
CORE_PATH = PROJECT_ROOT / "core"
DATA_PATH = PROJECT_ROOT / "data"
MODELS_PATH = PROJECT_ROOT / "models"
OUTPUT_PATH = PROJECT_ROOT / "output"

# Add core modules to path
sys.path.append(str(CORE_PATH / "stage1_bart"))
sys.path.append(str(CORE_PATH / "stage2_metadata"))
sys.path.append(str(CORE_PATH / "fusion"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComplementaryStageEvaluator:
    """Evaluates complementary value of multi-stage system"""
    
    def __init__(self, test_data_path: str):
        self.test_data_path = test_data_path
        self.load_test_data()
        
        # Store predictions from each stage
        self.stage1_predictions = None
        self.stage2_predictions = None
        self.stage1_confidences = None
        self.stage2_scores = None
        
    def load_test_data(self):
        """Load and prepare test dataset"""
        logger.info("📊 Loading test dataset...")
        self.df = pd.read_csv(self.test_data_path)
        logger.info(f"✅ Loaded {len(self.df)} test samples")
        
        # Create ground truth binary labels (problematic vs normal)
        self.true_binary = self._create_binary_labels(self.df['llm_classification'].tolist())
        logger.info(f"📈 Ground truth: {sum(self.true_binary)} problematic, {len(self.true_binary)-sum(self.true_binary)} normal")
    
    def _create_binary_labels(self, labels: List[str]) -> List[int]:
        """Convert multi-class labels to binary (0=normal, 1=problematic)"""
        problematic_classes = ['spam', 'advertisement', 'fake_rant', 'inappropriate', 'irrelevant']
        return [1 if label in problematic_classes else 0 for label in labels]
    
    def get_stage1_predictions(self) -> Tuple[List[int], List[float]]:
        """Get Stage 1 (BART) predictions"""
        logger.info("🤖 Getting Stage 1 (BART) predictions...")
        
        try:
            from enhanced_bart_review_classifier import BARTReviewClassifier
            
            bart_model_path = MODELS_PATH / "bart_classifier"
            classifier = BARTReviewClassifier(model_path=str(bart_model_path))
            
            texts = self.df['text'].fillna('').astype(str).tolist()
            predictions = []
            confidences = []
            
            for i, text in enumerate(texts):
                if i % 50 == 0:
                    logger.info(f"   Processing sample {i+1}/{len(texts)}")
                
                try:
                    result = classifier.predict(text)
                    if isinstance(result, list) and len(result) > 0:
                        pred_result = result[0]
                        # Convert to binary (problematic=1, normal=0)
                        pred_class = pred_result['prediction']
                        is_problematic = 1 if pred_class in ['spam', 'advertisement', 'fake_rant', 'inappropriate', 'irrelevant'] else 0
                        predictions.append(is_problematic)
                        confidences.append(pred_result['confidence'])
                    else:
                        predictions.append(0)  # Default to normal
                        confidences.append(0.5)
                except Exception as e:
                    predictions.append(0)
                    confidences.append(0.5)
            
            logger.info(f"✅ Stage 1: {sum(predictions)} flagged as problematic")
            return predictions, confidences
            
        except Exception as e:
            logger.error(f"❌ Stage 1 evaluation failed: {e}")
            return [0] * len(self.df), [0.5] * len(self.df)
    
    def get_stage2_predictions(self) -> Tuple[List[int], List[float]]:
        """Get Stage 2 (Metadata) predictions"""
        logger.info("📊 Getting Stage 2 (Metadata) predictions...")
        
        try:
            # Load metadata analyzer
            analyzer_path = MODELS_PATH / "metadata_analyzer.pkl"
            model_data = joblib.load(analyzer_path)
            model = model_data['model']
            scaler = model_data['scaler']
            feature_columns = model_data['feature_columns']
            
            # Extract features (simplified version)
            df_features = self.df.copy()
            
            # Basic feature engineering
            if 'timestamp' in df_features.columns:
                df_features['timestamp'] = pd.to_datetime(df_features['timestamp'], errors='coerce')
                df_features['hour_of_day'] = df_features['timestamp'].dt.hour
                df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
                df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
            
            if 'text' in df_features.columns:
                df_features['text_length'] = df_features['text'].fillna('').astype(str).apply(len)
                df_features['word_count'] = df_features['text'].fillna('').astype(str).apply(lambda x: len(x.split()))
            
            # Handle missing features
            for feat in feature_columns:
                if feat not in df_features.columns:
                    df_features[feat] = 0
            
            # Extract and scale features
            X = df_features[feature_columns].fillna(0)
            X_scaled = scaler.transform(X)
            
            # Get predictions
            anomaly_scores = model.decision_function(X_scaled)
            predictions = model.predict(X_scaled)
            
            # Convert to binary (anomaly=-1 becomes 1, normal=1 becomes 0)
            binary_predictions = [(1 if pred == -1 else 0) for pred in predictions]
            
            # Convert scores to confidence-like values (higher score = less anomalous)
            confidence_scores = [(score + 1) / 2 for score in anomaly_scores]  # Normalize to 0-1
            
            logger.info(f"✅ Stage 2: {sum(binary_predictions)} flagged as anomalous")
            return binary_predictions, confidence_scores
            
        except Exception as e:
            logger.error(f"❌ Stage 2 evaluation failed: {e}")
            return [0] * len(self.df), [0.5] * len(self.df)
    
    def calculate_complementary_metrics(self) -> Dict:
        """Calculate complementary value metrics"""
        logger.info("\n🔍 CALCULATING COMPLEMENTARY METRICS")
        logger.info("=" * 60)
        
        # Get predictions from both stages
        self.stage1_predictions, self.stage1_confidences = self.get_stage1_predictions()
        self.stage2_predictions, self.stage2_scores = self.get_stage2_predictions()
        
        metrics = {}
        
        # Convert to numpy arrays for easier calculation
        s1_pred = np.array(self.stage1_predictions)
        s2_pred = np.array(self.stage2_predictions)
        true_labels = np.array(self.true_binary)
        
        # 1. COVERAGE METRICS
        # How many problematic samples does each stage catch?
        s1_catches = np.sum((s1_pred == 1) & (true_labels == 1))
        s2_catches = np.sum((s2_pred == 1) & (true_labels == 1))
        total_problematic = np.sum(true_labels == 1)
        
        metrics['stage1_coverage'] = s1_catches / total_problematic if total_problematic > 0 else 0
        metrics['stage2_coverage'] = s2_catches / total_problematic if total_problematic > 0 else 0
        
        # 2. COMPLEMENTARY VALUE
        # What does Stage 2 catch that Stage 1 misses?
        s1_misses = (s1_pred == 0) & (true_labels == 1)  # Stage 1 missed these problematic samples
        s2_catches_s1_misses = np.sum(s2_pred[s1_misses] == 1)  # Stage 2 catches some of these
        s1_total_misses = np.sum(s1_misses)
        
        metrics['stage2_recovery_rate'] = s2_catches_s1_misses / s1_total_misses if s1_total_misses > 0 else 0
        metrics['stage2_unique_catches'] = s2_catches_s1_misses
        
        # What does Stage 1 catch that Stage 2 misses?
        s2_misses = (s2_pred == 0) & (true_labels == 1)
        s1_catches_s2_misses = np.sum(s1_pred[s2_misses] == 1)
        s2_total_misses = np.sum(s2_misses)
        
        metrics['stage1_recovery_rate'] = s1_catches_s2_misses / s2_total_misses if s2_total_misses > 0 else 0
        metrics['stage1_unique_catches'] = s1_catches_s2_misses
        
        # 3. COMBINED SYSTEM PERFORMANCE
        # Union: Either stage flags it
        combined_or = (s1_pred == 1) | (s2_pred == 1)
        combined_or_catches = np.sum(combined_or & (true_labels == 1))
        
        # Intersection: Both stages agree it's problematic
        combined_and = (s1_pred == 1) & (s2_pred == 1)
        combined_and_catches = np.sum(combined_and & (true_labels == 1))
        
        metrics['combined_or_coverage'] = combined_or_catches / total_problematic if total_problematic > 0 else 0
        metrics['combined_and_precision'] = combined_and_catches / np.sum(combined_and) if np.sum(combined_and) > 0 else 0
        
        # 4. AGREEMENT PATTERNS
        # How often do the stages agree?
        agreement = (s1_pred == s2_pred)
        metrics['overall_agreement'] = np.mean(agreement)
        
        # Agreement on problematic samples
        problematic_agreement = np.mean(agreement[true_labels == 1]) if total_problematic > 0 else 0
        metrics['problematic_agreement'] = problematic_agreement
        
        # 5. PRECISION METRICS
        # When Stage 2 flags something Stage 1 doesn't, how often is it right?
        s2_only = (s2_pred == 1) & (s1_pred == 0)
        s2_only_correct = np.sum(s2_only & (true_labels == 1))
        metrics['stage2_unique_precision'] = s2_only_correct / np.sum(s2_only) if np.sum(s2_only) > 0 else 0
        
        # When Stage 1 flags something Stage 2 doesn't, how often is it right?
        s1_only = (s1_pred == 1) & (s2_pred == 0)
        s1_only_correct = np.sum(s1_only & (true_labels == 1))
        metrics['stage1_unique_precision'] = s1_only_correct / np.sum(s1_only) if np.sum(s1_only) > 0 else 0
        
        # 6. BUSINESS VALUE METRICS
        # Reduction in false negatives
        s1_false_negatives = np.sum((s1_pred == 0) & (true_labels == 1))
        combined_false_negatives = np.sum((combined_or == 0) & (true_labels == 1))
        metrics['false_negative_reduction'] = (s1_false_negatives - combined_false_negatives) / s1_false_negatives if s1_false_negatives > 0 else 0
        
        # Trade-off: increase in false positives
        s1_false_positives = np.sum((s1_pred == 1) & (true_labels == 0))
        combined_false_positives = np.sum((combined_or == 1) & (true_labels == 0))
        metrics['false_positive_increase'] = (combined_false_positives - s1_false_positives) / len(true_labels)
        
        # Store counts for reporting
        metrics['total_samples'] = len(true_labels)
        metrics['total_problematic'] = int(total_problematic)
        metrics['stage1_flags'] = int(np.sum(s1_pred == 1))
        metrics['stage2_flags'] = int(np.sum(s2_pred == 1))
        metrics['combined_flags'] = int(np.sum(combined_or))
        
        return metrics
    
    def generate_complementary_report(self) -> str:
        """Generate comprehensive complementary evaluation report"""
        metrics = self.calculate_complementary_metrics()
        
        # Create detailed breakdown
        breakdown = self._create_detailed_breakdown()
        
        # Save to CSV
        report_data = {
            'Metric': [],
            'Value': [],
            'Description': []
        }
        
        # Coverage metrics
        report_data['Metric'].extend([
            'Stage 1 Coverage', 'Stage 2 Coverage', 'Combined OR Coverage'
        ])
        report_data['Value'].extend([
            f"{metrics['stage1_coverage']:.3f}",
            f"{metrics['stage2_coverage']:.3f}",
            f"{metrics['combined_or_coverage']:.3f}"
        ])
        report_data['Description'].extend([
            'Fraction of problematic samples caught by Stage 1',
            'Fraction of problematic samples caught by Stage 2',
            'Fraction caught by either stage (union)'
        ])
        
        # Complementary value
        report_data['Metric'].extend([
            'Stage 2 Recovery Rate', 'Stage 2 Unique Catches', 'Stage 2 Unique Precision'
        ])
        report_data['Value'].extend([
            f"{metrics['stage2_recovery_rate']:.3f}",
            f"{metrics['stage2_unique_catches']}",
            f"{metrics['stage2_unique_precision']:.3f}"
        ])
        report_data['Description'].extend([
            'Fraction of Stage 1 misses that Stage 2 recovers',
            'Number of problematic samples only Stage 2 caught',
            'Precision when Stage 2 flags but Stage 1 does not'
        ])
        
        # Business value
        report_data['Metric'].extend([
            'False Negative Reduction', 'False Positive Increase', 'Agreement Rate'
        ])
        report_data['Value'].extend([
            f"{metrics['false_negative_reduction']:.3f}",
            f"{metrics['false_positive_increase']:.3f}",
            f"{metrics['overall_agreement']:.3f}"
        ])
        report_data['Description'].extend([
            'Reduction in missed problematic samples',
            'Increase in false alarms (per sample)',
            'Overall agreement between stages'
        ])
        
        # Save complementary metrics
        OUTPUT_PATH.mkdir(exist_ok=True)
        csv_path = OUTPUT_PATH / "complementary_evaluation.csv"
        pd.DataFrame(report_data).to_csv(csv_path, index=False)
        
        # Save detailed breakdown
        breakdown_path = OUTPUT_PATH / "stage_breakdown.csv"
        breakdown.to_csv(breakdown_path, index=False)
        
        # Print summary
        logger.info("\n📊 COMPLEMENTARY EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"📈 Stage 1 Coverage: {metrics['stage1_coverage']:.1%}")
        logger.info(f"📈 Stage 2 Coverage: {metrics['stage2_coverage']:.1%}")
        logger.info(f"🔄 Stage 2 Recovery Rate: {metrics['stage2_recovery_rate']:.1%}")
        logger.info(f"🎯 Combined Coverage: {metrics['combined_or_coverage']:.1%}")
        logger.info(f"🤝 Agreement Rate: {metrics['overall_agreement']:.1%}")
        logger.info(f"📉 False Negative Reduction: {metrics['false_negative_reduction']:.1%}")
        
        logger.info(f"\n📄 Reports saved:")
        logger.info(f"   Metrics: {csv_path}")
        logger.info(f"   Breakdown: {breakdown_path}")
        
        return str(csv_path)
    
    def _create_detailed_breakdown(self) -> pd.DataFrame:
        """Create detailed sample-by-sample breakdown"""
        breakdown_data = {
            'sample_id': range(len(self.df)),
            'true_label': self.df['llm_classification'].tolist(),
            'true_binary': self.true_binary,
            'stage1_prediction': self.stage1_predictions,
            'stage1_confidence': self.stage1_confidences,
            'stage2_prediction': self.stage2_predictions,
            'stage2_score': self.stage2_scores,
            'agreement': [1 if s1 == s2 else 0 for s1, s2 in zip(self.stage1_predictions, self.stage2_predictions)],
            'combined_or': [1 if s1 == 1 or s2 == 1 else 0 for s1, s2 in zip(self.stage1_predictions, self.stage2_predictions)],
            'stage1_correct': [1 if s1 == true else 0 for s1, true in zip(self.stage1_predictions, self.true_binary)],
            'stage2_correct': [1 if s2 == true else 0 for s2, true in zip(self.stage2_predictions, self.true_binary)],
            'combined_correct': [1 if (s1 == 1 or s2 == 1) == true else 0 for s1, s2, true in zip(self.stage1_predictions, self.stage2_predictions, self.true_binary)]
        }
        
        return pd.DataFrame(breakdown_data)

def main():
    """Main evaluation function"""
    test_data_path = DATA_PATH / "data_all_test.csv"
    
    if not test_data_path.exists():
        logger.error(f"❌ Test data not found: {test_data_path}")
        return
    
    evaluator = ComplementaryStageEvaluator(str(test_data_path))
    csv_path = evaluator.generate_complementary_report()
    
    logger.info(f"\n✅ Complementary evaluation completed!")

if __name__ == "__main__":
    main()
