"""
🔬 SIMPLE BINARY EVALUATION
Evaluates Stage 1 binary classification: Genuine vs Non-Genuine
Uses existing BART classifier directly for reliable evaluation.
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
sys.path.append(str(CORE_PATH))
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

class SimpleBinaryEvaluator:
    """Simple evaluator for binary classification performance"""
    
    def __init__(self, test_data_path: str):
        self.test_data_path = test_data_path
        self.load_test_data()
        
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
    
    def get_stage1_predictions(self) -> Tuple[List[int], List[float]]:
        """Get Stage 1 predictions using existing infrastructure"""
        logger.info("🤖 Getting Stage 1 (BART) predictions...")
        
        try:
            # Import the existing classifier
            from enhanced_bart_review_classifier import BARTReviewClassifier
            
            bart_model_path = MODELS_PATH / "bart_classifier"
            classifier = BARTReviewClassifier(model_path=str(bart_model_path))
            
            texts = self.df['text'].fillna('').astype(str).tolist()
            predictions = []
            probabilities = []
            
            batch_size = 32  # Process in batches for efficiency
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                
                logger.info(f"   Processing batch {batch_idx + 1}/{total_batches} (samples {start_idx + 1}-{end_idx})")
                
                for text in batch_texts:
                    try:
                        result = classifier.predict(text)
                        if isinstance(result, list) and len(result) > 0:
                            pred_result = result[0]
                            pred_class = pred_result['prediction']
                            confidence = pred_result['confidence']
                            
                            # Convert to binary (non-genuine=1, genuine=0)
                            is_non_genuine = 1 if pred_class in ['spam', 'advertisement', 'fake_rant', 'inappropriate', 'irrelevant'] else 0
                            predictions.append(is_non_genuine)
                            
                            # Calculate probability of being non-genuine
                            if is_non_genuine == 1:
                                prob_non_genuine = confidence
                            else:
                                prob_non_genuine = 1.0 - confidence
                            
                            probabilities.append(prob_non_genuine)
                        else:
                            predictions.append(0)  # Default to genuine
                            probabilities.append(0.5)
                    except Exception as e:
                        logger.warning(f"Error processing text: {e}")
                        predictions.append(0)
                        probabilities.append(0.5)
            
            logger.info(f"✅ Stage 1: {sum(predictions)} flagged as non-genuine")
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"❌ Stage 1 evaluation failed: {e}")
            # Return default predictions
            return [0] * len(self.df), [0.5] * len(self.df)
    
    def calculate_metrics(self, y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict:
        """Calculate comprehensive binary classification metrics"""
        
        metrics = {}
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 for both classes
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        
        # Handle case where one class might be missing
        if len(precision) >= 2:
            metrics['genuine_precision'] = precision[0]
            metrics['genuine_recall'] = recall[0]
            metrics['genuine_f1'] = f1[0]
            metrics['genuine_support'] = support[0]
            
            metrics['non_genuine_precision'] = precision[1]
            metrics['non_genuine_recall'] = recall[1]
            metrics['non_genuine_f1'] = f1[1]
            metrics['non_genuine_support'] = support[1]
        else:
            # Handle edge case
            metrics['genuine_precision'] = precision[0] if len(precision) > 0 else 0
            metrics['genuine_recall'] = recall[0] if len(recall) > 0 else 0
            metrics['genuine_f1'] = f1[0] if len(f1) > 0 else 0
            metrics['genuine_support'] = support[0] if len(support) > 0 else 0
            
            metrics['non_genuine_precision'] = 0
            metrics['non_genuine_recall'] = 0
            metrics['non_genuine_f1'] = 0
            metrics['non_genuine_support'] = 0
        
        # Weighted and macro averages
        try:
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
            
            metrics['macro_precision'] = precision_macro
            metrics['macro_recall'] = recall_macro
            metrics['macro_f1'] = f1_macro
            metrics['weighted_precision'] = precision_weighted
            metrics['weighted_recall'] = recall_weighted
            metrics['weighted_f1'] = f1_weighted
        except:
            metrics['macro_precision'] = 0
            metrics['macro_recall'] = 0
            metrics['macro_f1'] = 0
            metrics['weighted_precision'] = 0
            metrics['weighted_recall'] = 0
            metrics['weighted_f1'] = 0
        
        # ROC AUC and PR AUC
        try:
            if len(np.unique(y_true)) > 1:  # Need both classes for AUC
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                metrics['pr_auc'] = average_precision_score(y_true, y_prob)
            else:
                metrics['roc_auc'] = 0.5
                metrics['pr_auc'] = 0.5
        except:
            metrics['roc_auc'] = 0.5
            metrics['pr_auc'] = 0.5
        
        # Confusion matrix components
        try:
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                # Handle edge cases
                tn = fp = fn = tp = 0
                if cm.shape == (1, 1):
                    if np.unique(y_true)[0] == 0:  # Only genuine samples
                        tn = cm[0, 0]
                    else:  # Only non-genuine samples
                        tp = cm[0, 0]
            
            metrics['true_negatives'] = tn
            metrics['false_positives'] = fp
            metrics['false_negatives'] = fn
            metrics['true_positives'] = tp
            
            # Additional derived metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        except:
            metrics['true_negatives'] = 0
            metrics['false_positives'] = 0
            metrics['false_negatives'] = 0
            metrics['true_positives'] = 0
            metrics['specificity'] = 0
            metrics['sensitivity'] = 0
            metrics['positive_predictive_value'] = 0
            metrics['negative_predictive_value'] = 0
        
        return metrics
    
    def create_visualizations(self, y_true: List[int], y_pred: List[int], y_prob: List[float], metrics: Dict):
        """Create and save visualizations"""
        logger.info("📊 Creating visualizations...")
        
        OUTPUT_PATH.mkdir(exist_ok=True)
        
        # Create comprehensive figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Stage 1 Binary Classification Analysis\nGenuine vs Non-Genuine Reviews', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        ax = axes[0, 0]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Genuine', 'Non-Genuine'], 
                   yticklabels=['Genuine', 'Non-Genuine'])
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # 2. Metrics Bar Chart
        ax = axes[0, 1]
        metric_names = ['Accuracy', 'Precision\n(Non-Genuine)', 'Recall\n(Non-Genuine)', 'F1-Score\n(Non-Genuine)']
        metric_values = [
            metrics['accuracy'],
            metrics['non_genuine_precision'],
            metrics['non_genuine_recall'],
            metrics['non_genuine_f1']
        ]
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        bars = ax.bar(metric_names, metric_values, color=colors)
        ax.set_ylim(0, 1)
        ax.set_title('Key Performance Metrics')
        ax.set_ylabel('Score')
        
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Class Distribution
        ax = axes[0, 2]
        true_counts = [self.genuine_count, self.non_genuine_count]
        pred_counts = [len(y_pred) - sum(y_pred), sum(y_pred)]
        
        x = np.arange(2)
        width = 0.35
        
        ax.bar(x - width/2, true_counts, width, label='True Distribution', color='lightblue')
        ax.bar(x + width/2, pred_counts, width, label='Predicted Distribution', color='lightcoral')
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(['Genuine', 'Non-Genuine'])
        ax.legend()
        
        # 4. ROC Curve
        ax = axes[1, 0]
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'ROC curve not available\n(only one class present)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('ROC Curve (N/A)')
        
        # 5. Precision-Recall Curve
        ax = axes[1, 1]
        if len(np.unique(y_true)) > 1:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall_curve, precision_curve)
            ax.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend(loc="lower left")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'PR curve not available\n(only one class present)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('PR Curve (N/A)')
        
        # 6. Performance Summary
        ax = axes[1, 2]
        summary_text = f"""Performance Summary:

Accuracy: {metrics['accuracy']:.3f}
ROC AUC: {metrics['roc_auc']:.3f}
PR AUC: {metrics['pr_auc']:.3f}

Non-Genuine Detection:
• Precision: {metrics['non_genuine_precision']:.3f}
• Recall: {metrics['non_genuine_recall']:.3f}
• F1-Score: {metrics['non_genuine_f1']:.3f}

Genuine Detection:
• Precision: {metrics['genuine_precision']:.3f}
• Recall: {metrics['genuine_recall']:.3f}
• F1-Score: {metrics['genuine_f1']:.3f}"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Performance Summary')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = OUTPUT_PATH / "stage1_binary_classification_comprehensive.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Comprehensive plot saved: {plot_path}")
        plt.close()
    
    def save_results_to_csv(self, y_true: List[int], y_pred: List[int], y_prob: List[float], metrics: Dict):
        """Save results to CSV files"""
        logger.info("💾 Saving results to CSV...")
        
        OUTPUT_PATH.mkdir(exist_ok=True)
        
        # 1. Summary metrics
        summary_data = {
            'Metric': [
                'Accuracy', 'ROC_AUC', 'PR_AUC',
                'Genuine_Precision', 'Genuine_Recall', 'Genuine_F1',
                'Non_Genuine_Precision', 'Non_Genuine_Recall', 'Non_Genuine_F1',
                'Macro_Precision', 'Macro_Recall', 'Macro_F1',
                'Weighted_Precision', 'Weighted_Recall', 'Weighted_F1'
            ],
            'Value': [
                metrics['accuracy'], metrics['roc_auc'], metrics['pr_auc'],
                metrics['genuine_precision'], metrics['genuine_recall'], metrics['genuine_f1'],
                metrics['non_genuine_precision'], metrics['non_genuine_recall'], metrics['non_genuine_f1'],
                metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'],
                metrics['weighted_precision'], metrics['weighted_recall'], metrics['weighted_f1']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = OUTPUT_PATH / "stage1_binary_metrics.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"📄 Summary metrics saved: {summary_path}")
        
        # 2. Sample-by-sample results
        sample_data = {
            'sample_id': range(len(self.df)),
            'true_label_text': self.df['llm_classification'].tolist(),
            'true_binary': y_true,
            'predicted_binary': y_pred,
            'probability_non_genuine': y_prob,
            'correct_prediction': [1 if pred == true else 0 for pred, true in zip(y_pred, y_true)]
        }
        
        sample_df = pd.DataFrame(sample_data)
        sample_path = OUTPUT_PATH / "stage1_sample_predictions.csv"
        sample_df.to_csv(sample_path, index=False)
        logger.info(f"📄 Sample predictions saved: {sample_path}")
        
        return summary_path, sample_path
    
    def run_evaluation(self):
        """Run complete binary evaluation"""
        logger.info("\n🔬 RUNNING STAGE 1 BINARY CLASSIFICATION EVALUATION")
        logger.info("=" * 60)
        
        # Get predictions
        y_pred, y_prob = self.get_stage1_predictions()
        y_true = self.true_binary
        
        # Calculate metrics
        logger.info("📊 Calculating metrics...")
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        
        # Create visualizations
        self.create_visualizations(y_true, y_pred, y_prob, metrics)
        
        # Save results
        summary_path, sample_path = self.save_results_to_csv(y_true, y_pred, y_prob, metrics)
        
        # Print results
        logger.info("\n📊 EVALUATION RESULTS")
        logger.info("=" * 40)
        logger.info(f"🎯 Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"📈 ROC AUC: {metrics['roc_auc']:.3f}")
        logger.info(f"📈 PR AUC: {metrics['pr_auc']:.3f}")
        logger.info(f"📊 Non-Genuine F1: {metrics['non_genuine_f1']:.3f}")
        logger.info(f"📊 Genuine F1: {metrics['genuine_f1']:.3f}")
        logger.info(f"📊 Macro F1: {metrics['macro_f1']:.3f}")
        logger.info("")
        logger.info(f"📄 Files generated:")
        logger.info(f"   📊 Visualization: {OUTPUT_PATH}/stage1_binary_classification_comprehensive.png")
        logger.info(f"   📄 Metrics: {summary_path}")
        logger.info(f"   📄 Predictions: {sample_path}")
        
        return metrics

def main():
    """Main evaluation function"""
    test_data_path = DATA_PATH / "data_all_test.csv"
    
    if not test_data_path.exists():
        logger.error(f"❌ Test data not found: {test_data_path}")
        return
    
    evaluator = SimpleBinaryEvaluator(str(test_data_path))
    metrics = evaluator.run_evaluation()
    
    logger.info(f"\n✅ Binary evaluation completed!")

if __name__ == "__main__":
    main()
