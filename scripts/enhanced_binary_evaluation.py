"""
🔬 ENHANCED BINARY EVALUATION
Evaluates Stage 1 binary classification: Genuine vs Non-Genuine
Generates comprehensive metrics, CSV files, and visualizations.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc,
    average_precision_score, roc_auc_score
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

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class EnhancedBinaryEvaluator:
    """Enhanced evaluator focused on binary classification performance"""
    
    def __init__(self, test_data_path: str):
        self.test_data_path = test_data_path
        self.load_test_data()
        
        # Store predictions and probabilities
        self.stage1_predictions = None
        self.stage1_probabilities = None
        self.stage1_confidences = None
        
    def load_test_data(self):
        """Load and prepare test dataset"""
        logger.info("📊 Loading test dataset...")
        self.df = pd.read_csv(self.test_data_path)
        logger.info(f"✅ Loaded {len(self.df)} test samples")
        
        # Create binary labels: 0=genuine, 1=non-genuine
        self.true_binary = self._create_binary_labels(self.df['llm_classification'].tolist())
        self.genuine_count = sum([1 for x in self.true_binary if x == 0])
        self.non_genuine_count = sum([1 for x in self.true_binary if x == 1])
        
        logger.info(f"📈 Ground truth: {self.genuine_count} genuine, {self.non_genuine_count} non-genuine")
    
    def _create_binary_labels(self, labels: List[str]) -> List[int]:
        """Convert multi-class labels to binary (0=genuine, 1=non-genuine)"""
        # Non-genuine classes
        non_genuine_classes = ['spam', 'advertisement', 'fake_rant', 'inappropriate', 'irrelevant']
        return [1 if label in non_genuine_classes else 0 for label in labels]
    
    def get_stage1_predictions_and_probabilities(self) -> Tuple[List[int], List[float], List[float]]:
        """Get Stage 1 predictions with probabilities for binary classification"""
        logger.info("🤖 Getting Stage 1 (BART) predictions with probabilities...")
        
        try:
            from enhanced_bart_review_classifier import BARTReviewClassifier
            
            bart_model_path = MODELS_PATH / "bart_classifier"
            classifier = BARTReviewClassifier(model_path=str(bart_model_path))
            
            texts = self.df['text'].fillna('').astype(str).tolist()
            predictions = []
            probabilities = []  # Probability of being non-genuine
            confidences = []
            
            for i, text in enumerate(texts):
                if i % 50 == 0:
                    logger.info(f"   Processing sample {i+1}/{len(texts)}")
                
                try:
                    result = classifier.predict(text)
                    if isinstance(result, list) and len(result) > 0:
                        pred_result = result[0]
                        pred_class = pred_result['prediction']
                        confidence = pred_result['confidence']
                        
                        # Convert to binary (non-genuine=1, genuine=0)
                        is_non_genuine = 1 if pred_class in ['spam', 'advertisement', 'fake_rant', 'inappropriate', 'irrelevant'] else 0
                        predictions.append(is_non_genuine)
                        
                        # Probability of being non-genuine
                        if is_non_genuine == 1:
                            prob_non_genuine = confidence
                        else:
                            prob_non_genuine = 1.0 - confidence
                        
                        probabilities.append(prob_non_genuine)
                        confidences.append(confidence)
                    else:
                        predictions.append(0)  # Default to genuine
                        probabilities.append(0.5)
                        confidences.append(0.5)
                except Exception as e:
                    predictions.append(0)
                    probabilities.append(0.5)
                    confidences.append(0.5)
            
            logger.info(f"✅ Stage 1: {sum(predictions)} flagged as non-genuine")
            return predictions, probabilities, confidences
            
        except Exception as e:
            logger.error(f"❌ Stage 1 evaluation failed: {e}")
            return [0] * len(self.df), [0.5] * len(self.df), [0.5] * len(self.df)
    
    def calculate_binary_metrics(self) -> Dict:
        """Calculate comprehensive binary classification metrics"""
        logger.info("\n🔍 CALCULATING BINARY CLASSIFICATION METRICS")
        logger.info("=" * 60)
        
        # Get predictions and probabilities
        self.stage1_predictions, self.stage1_probabilities, self.stage1_confidences = self.get_stage1_predictions_and_probabilities()
        
        y_true = np.array(self.true_binary)
        y_pred = np.array(self.stage1_predictions)
        y_prob = np.array(self.stage1_probabilities)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 for both classes
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        # Class 0 (Genuine)
        metrics['genuine_precision'] = precision[0]
        metrics['genuine_recall'] = recall[0]
        metrics['genuine_f1'] = f1[0]
        metrics['genuine_support'] = support[0]
        
        # Class 1 (Non-genuine)
        metrics['non_genuine_precision'] = precision[1]
        metrics['non_genuine_recall'] = recall[1]
        metrics['non_genuine_f1'] = f1[1]
        metrics['non_genuine_support'] = support[1]
        
        # Weighted and macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        metrics['macro_precision'] = precision_macro
        metrics['macro_recall'] = recall_macro
        metrics['macro_f1'] = f1_macro
        metrics['weighted_precision'] = precision_weighted
        metrics['weighted_recall'] = recall_weighted
        metrics['weighted_f1'] = f1_weighted
        
        # ROC AUC and PR AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        except:
            metrics['roc_auc'] = 0.5
            metrics['pr_auc'] = 0.5
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        # Additional derived metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        return metrics
    
    def generate_visualizations(self, metrics: Dict):
        """Generate comprehensive visualizations"""
        logger.info("📊 Generating visualizations...")
        
        OUTPUT_PATH.mkdir(exist_ok=True)
        
        y_true = np.array(self.true_binary)
        y_pred = np.array(self.stage1_predictions)
        y_prob = np.array(self.stage1_probabilities)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Genuine', 'Non-Genuine'], 
                   yticklabels=['Genuine', 'Non-Genuine'])
        plt.title('Confusion Matrix\nStage 1 Binary Classification', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 2. ROC Curve
        ax2 = plt.subplot(2, 3, 2)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve\nStage 1 Binary Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        ax3 = plt.subplot(2, 3, 3)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
        plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve\nStage 1 Binary Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        # 4. Metrics Bar Chart
        ax4 = plt.subplot(2, 3, 4)
        metric_names = ['Accuracy', 'Precision\n(Non-Genuine)', 'Recall\n(Non-Genuine)', 'F1-Score\n(Non-Genuine)', 'ROC AUC', 'PR AUC']
        metric_values = [
            metrics['accuracy'],
            metrics['non_genuine_precision'],
            metrics['non_genuine_recall'],
            metrics['non_genuine_f1'],
            metrics['roc_auc'],
            metrics['pr_auc']
        ]
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'orange']
        bars = plt.bar(metric_names, metric_values, color=colors)
        plt.ylim(0, 1)
        plt.title('Key Performance Metrics\nStage 1 Binary Classification', fontsize=14, fontweight='bold')
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Class Distribution
        ax5 = plt.subplot(2, 3, 5)
        true_counts = [self.genuine_count, self.non_genuine_count]
        pred_counts = [len(y_pred) - sum(y_pred), sum(y_pred)]
        
        x = np.arange(2)
        width = 0.35
        
        plt.bar(x - width/2, true_counts, width, label='True Distribution', color='lightblue')
        plt.bar(x + width/2, pred_counts, width, label='Predicted Distribution', color='lightcoral')
        
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution\nTrue vs Predicted', fontsize=14, fontweight='bold')
        plt.xticks(x, ['Genuine', 'Non-Genuine'])
        plt.legend()
        
        # Add value labels
        for i, (true_val, pred_val) in enumerate(zip(true_counts, pred_counts)):
            plt.text(i - width/2, true_val + 2, str(true_val), ha='center', va='bottom', fontweight='bold')
            plt.text(i + width/2, pred_val + 2, str(pred_val), ha='center', va='bottom', fontweight='bold')
        
        # 6. Confidence Distribution
        ax6 = plt.subplot(2, 3, 6)
        genuine_confidences = [conf for i, conf in enumerate(self.stage1_confidences) if y_true[i] == 0]
        non_genuine_confidences = [conf for i, conf in enumerate(self.stage1_confidences) if y_true[i] == 1]
        
        plt.hist(genuine_confidences, bins=20, alpha=0.7, label='Genuine', color='lightblue', density=True)
        plt.hist(non_genuine_confidences, bins=20, alpha=0.7, label='Non-Genuine', color='lightcoral', density=True)
        plt.xlabel('Confidence Score')
        plt.ylabel('Density')
        plt.title('Confidence Distribution by True Class\nStage 1 Predictions', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        plot_path = OUTPUT_PATH / "stage1_binary_classification_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Comprehensive analysis saved: {plot_path}")
        
        # Save individual high-quality plots
        self._save_individual_plots(y_true, y_pred, y_prob, metrics)
        
        plt.close()
    
    def _save_individual_plots(self, y_true, y_pred, y_prob, metrics):
        """Save individual high-quality plots"""
        
        # High-quality ROC curve
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Stage 1 Binary Classification\nGenuine vs Non-Genuine Reviews', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = OUTPUT_PATH / "stage1_roc_curve.png"
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 ROC curve saved: {roc_path}")
        
        # High-quality PR curve
        plt.figure(figsize=(10, 8))
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
        plt.plot(recall_curve, precision_curve, color='blue', lw=3, label=f'PR curve (AUC = {pr_auc:.3f})')
        # Add baseline (random classifier performance)
        baseline = sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='red', linestyle='--', lw=2, label=f'Random Classifier (AP = {baseline:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve - Stage 1 Binary Classification\nGenuine vs Non-Genuine Reviews', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pr_path = OUTPUT_PATH / "stage1_pr_curve.png"
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 PR curve saved: {pr_path}")
    
    def save_detailed_results(self, metrics: Dict):
        """Save detailed results to CSV files"""
        logger.info("💾 Saving detailed results...")
        
        OUTPUT_PATH.mkdir(exist_ok=True)
        
        # 1. Summary metrics
        summary_data = {
            'Metric': [
                'Accuracy',
                'Genuine Precision', 'Genuine Recall', 'Genuine F1-Score',
                'Non-Genuine Precision', 'Non-Genuine Recall', 'Non-Genuine F1-Score',
                'Macro Precision', 'Macro Recall', 'Macro F1-Score',
                'Weighted Precision', 'Weighted Recall', 'Weighted F1-Score',
                'ROC AUC', 'PR AUC',
                'Specificity', 'Sensitivity',
                'Positive Predictive Value', 'Negative Predictive Value'
            ],
            'Value': [
                metrics['accuracy'],
                metrics['genuine_precision'], metrics['genuine_recall'], metrics['genuine_f1'],
                metrics['non_genuine_precision'], metrics['non_genuine_recall'], metrics['non_genuine_f1'],
                metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'],
                metrics['weighted_precision'], metrics['weighted_recall'], metrics['weighted_f1'],
                metrics['roc_auc'], metrics['pr_auc'],
                metrics['specificity'], metrics['sensitivity'],
                metrics['positive_predictive_value'], metrics['negative_predictive_value']
            ],
            'Description': [
                'Overall classification accuracy',
                'Precision for genuine reviews', 'Recall for genuine reviews', 'F1-score for genuine reviews',
                'Precision for non-genuine reviews', 'Recall for non-genuine reviews', 'F1-score for non-genuine reviews',
                'Macro-averaged precision', 'Macro-averaged recall', 'Macro-averaged F1-score',
                'Weighted-averaged precision', 'Weighted-averaged recall', 'Weighted-averaged F1-score',
                'Area under ROC curve', 'Area under PR curve',
                'True negative rate', 'True positive rate',
                'Precision for non-genuine class', 'Precision for genuine class'
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = OUTPUT_PATH / "stage1_binary_metrics_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"📄 Summary metrics saved: {summary_path}")
        
        # 2. Confusion matrix breakdown
        confusion_data = {
            'Predicted': ['Genuine', 'Genuine', 'Non-Genuine', 'Non-Genuine'],
            'Actual': ['Genuine', 'Non-Genuine', 'Genuine', 'Non-Genuine'],
            'Count': [metrics['true_negatives'], metrics['false_negatives'], 
                     metrics['false_positives'], metrics['true_positives']],
            'Type': ['True Negative', 'False Negative', 'False Positive', 'True Positive']
        }
        
        confusion_df = pd.DataFrame(confusion_data)
        confusion_path = OUTPUT_PATH / "stage1_confusion_matrix.csv"
        confusion_df.to_csv(confusion_path, index=False)
        logger.info(f"📄 Confusion matrix saved: {confusion_path}")
        
        # 3. Sample-by-sample results
        sample_data = {
            'sample_id': range(len(self.df)),
            'true_label': self.df['llm_classification'].tolist(),
            'true_binary': self.true_binary,
            'predicted_binary': self.stage1_predictions,
            'confidence': self.stage1_confidences,
            'probability_non_genuine': self.stage1_probabilities,
            'correct_prediction': [1 if pred == true else 0 for pred, true in zip(self.stage1_predictions, self.true_binary)],
            'text_length': self.df['text'].fillna('').astype(str).apply(len).tolist()
        }
        
        if 'text' in self.df.columns:
            sample_data['text_preview'] = [text[:100] + "..." if len(text) > 100 else text 
                                         for text in self.df['text'].fillna('').astype(str)]
        
        sample_df = pd.DataFrame(sample_data)
        sample_path = OUTPUT_PATH / "stage1_sample_predictions.csv"
        sample_df.to_csv(sample_path, index=False)
        logger.info(f"📄 Sample predictions saved: {sample_path}")
        
        return summary_path, confusion_path, sample_path
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive binary classification report"""
        logger.info("\n🔬 GENERATING COMPREHENSIVE BINARY CLASSIFICATION REPORT")
        logger.info("=" * 70)
        
        # Calculate metrics
        metrics = self.calculate_binary_metrics()
        
        # Generate visualizations
        self.generate_visualizations(metrics)
        
        # Save detailed results
        summary_path, confusion_path, sample_path = self.save_detailed_results(metrics)
        
        # Print comprehensive summary
        logger.info("\n📊 STAGE 1 BINARY CLASSIFICATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"🎯 Overall Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"📈 ROC AUC: {metrics['roc_auc']:.3f}")
        logger.info(f"📈 PR AUC: {metrics['pr_auc']:.3f}")
        logger.info("")
        logger.info("📊 GENUINE REVIEWS:")
        logger.info(f"   Precision: {metrics['genuine_precision']:.3f}")
        logger.info(f"   Recall: {metrics['genuine_recall']:.3f}")
        logger.info(f"   F1-Score: {metrics['genuine_f1']:.3f}")
        logger.info("")
        logger.info("📊 NON-GENUINE REVIEWS:")
        logger.info(f"   Precision: {metrics['non_genuine_precision']:.3f}")
        logger.info(f"   Recall: {metrics['non_genuine_recall']:.3f}")
        logger.info(f"   F1-Score: {metrics['non_genuine_f1']:.3f}")
        logger.info("")
        logger.info("📊 MACRO AVERAGES:")
        logger.info(f"   Precision: {metrics['macro_precision']:.3f}")
        logger.info(f"   Recall: {metrics['macro_recall']:.3f}")
        logger.info(f"   F1-Score: {metrics['macro_f1']:.3f}")
        logger.info("")
        logger.info("📊 CONFUSION MATRIX:")
        logger.info(f"   True Negatives (Genuine→Genuine): {metrics['true_negatives']}")
        logger.info(f"   False Positives (Genuine→Non-Genuine): {metrics['false_positives']}")
        logger.info(f"   False Negatives (Non-Genuine→Genuine): {metrics['false_negatives']}")
        logger.info(f"   True Positives (Non-Genuine→Non-Genuine): {metrics['true_positives']}")
        
        logger.info(f"\n📄 Files generated:")
        logger.info(f"   📊 Comprehensive analysis: {OUTPUT_PATH}/stage1_binary_classification_analysis.png")
        logger.info(f"   📈 ROC curve: {OUTPUT_PATH}/stage1_roc_curve.png")
        logger.info(f"   📈 PR curve: {OUTPUT_PATH}/stage1_pr_curve.png")
        logger.info(f"   📄 Summary metrics: {summary_path}")
        logger.info(f"   📄 Confusion matrix: {confusion_path}")
        logger.info(f"   📄 Sample predictions: {sample_path}")
        
        return str(summary_path)

def main():
    """Main evaluation function"""
    test_data_path = DATA_PATH / "data_all_test.csv"
    
    if not test_data_path.exists():
        logger.error(f"❌ Test data not found: {test_data_path}")
        return
    
    evaluator = EnhancedBinaryEvaluator(str(test_data_path))
    summary_path = evaluator.generate_comprehensive_report()
    
    logger.info(f"\n✅ Enhanced binary evaluation completed!")

if __name__ == "__main__":
    main()
