#!/usr/bin/env python3
"""
Comprehensive Model Evaluation: Fine-tuned vs Zero-shot BART
Compare fine-tuned model against base zero-shot model performance
Evaluate against LLM ground truth classifications
"""

import pandas as pd
import numpy as np
import torch
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline
)
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveModelEvaluator:
    """
    Comprehensive evaluation comparing fine-tuned vs zero-shot BART models
    """
    
    def __init__(self, fine_tuned_model_path: str, test_data_path: str):
        self.fine_tuned_model_path = fine_tuned_model_path
        self.test_data_path = test_data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Label mappings
        self.labels = [
            'genuine_positive', 'genuine_negative', 'spam', 
            'advertisement', 'irrelevant', 'fake_rant', 'inappropriate'
        ]
        
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        self.id_to_label = {i: label for i, label in enumerate(self.labels)}
        
        # Models
        self.fine_tuned_model = None
        self.fine_tuned_tokenizer = None
        self.zero_shot_classifier = None
        
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def load_models(self):
        """Load fine-tuned and zero-shot models"""
        logger.info("Loading models...")
        
        # Load fine-tuned model
        try:
            self.fine_tuned_tokenizer = AutoTokenizer.from_pretrained(self.fine_tuned_model_path)
            self.fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(
                self.fine_tuned_model_path
            ).to(self.device)
            logger.info("✅ Fine-tuned model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load fine-tuned model: {e}")
            raise
        
        # Load zero-shot classifier
        try:
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("✅ Zero-shot model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load zero-shot model: {e}")
            raise
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test dataset"""
        logger.info(f"Loading test data from {self.test_data_path}")
        
        df = pd.read_csv(self.test_data_path)
        df = df.dropna(subset=['text', 'llm_classification'])
        df = df[df['llm_classification'].isin(self.labels)]
        
        logger.info(f"Loaded {len(df)} test samples")
        logger.info("Test set label distribution:")
        for label in self.labels:
            count = len(df[df['llm_classification'] == label])
            percentage = (count / len(df)) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        
        return df
    
    def predict_fine_tuned(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """Get predictions from fine-tuned model"""
        logger.info("Getting fine-tuned model predictions...")
        
        predictions = []
        confidences = []
        
        self.fine_tuned_model.eval()
        with torch.no_grad():
            for i, text in enumerate(texts):
                if i % 50 == 0:
                    logger.info(f"Processing batch {i//50 + 1}/{(len(texts)-1)//50 + 1}")
                
                # Tokenize
                inputs = self.fine_tuned_tokenizer(
                    text, 
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(self.device)
                
                # Predict
                outputs = self.fine_tuned_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Get prediction and confidence
                pred_id = torch.argmax(probs, dim=-1).item()
                confidence = torch.max(probs).item()
                
                predictions.append(self.id_to_label[pred_id])
                confidences.append(confidence)
        
        return predictions, confidences
    
    def predict_zero_shot(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """Get predictions from zero-shot model"""
        logger.info("Getting zero-shot model predictions...")
        
        predictions = []
        confidences = []
        
        candidate_labels = self.labels
        
        for i, text in enumerate(texts):
            if i % 50 == 0:
                logger.info(f"Processing batch {i//50 + 1}/{(len(texts)-1)//50 + 1}")
            
            try:
                result = self.zero_shot_classifier(text, candidate_labels)
                predictions.append(result['labels'][0])
                confidences.append(result['scores'][0])
            except Exception as e:
                logger.warning(f"Zero-shot prediction failed for sample {i}: {e}")
                predictions.append('genuine_positive')  # Default fallback
                confidences.append(0.1)
        
        return predictions, confidences
    
    def calculate_metrics(self, y_true: List[str], y_pred: List[str], 
                         confidences: List[float] = None) -> Dict[str, Any]:
        """Calculate comprehensive metrics with both exact and binary evaluation"""
        
        metrics = {}
        
        # 1. EXACT CLASSIFICATION METRICS (7-class)
        metrics['exact_accuracy'] = accuracy_score(y_true, y_pred)
        metrics['exact_f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['exact_f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        metrics['exact_f1_micro'] = f1_score(y_true, y_pred, average='micro')
        
        # 2. BINARY DETECTION METRICS (Genuine vs Non-genuine)
        # This is the main metric for flagging fake reviews
        y_true_binary = ['genuine' if label in ['genuine_positive', 'genuine_negative'] else 'non_genuine' for label in y_true]
        y_pred_binary = ['genuine' if label in ['genuine_positive', 'genuine_negative'] else 'non_genuine' for label in y_pred]
        
        metrics['binary_accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
        metrics['binary_f1'] = f1_score(y_true_binary, y_pred_binary, pos_label='genuine')
        metrics['binary_f1_non_genuine'] = f1_score(y_true_binary, y_pred_binary, pos_label='non_genuine')
        metrics['binary_f1_macro'] = f1_score(y_true_binary, y_pred_binary, average='macro')
        metrics['binary_f1_weighted'] = f1_score(y_true_binary, y_pred_binary, average='weighted')
        
        # Binary classification report
        binary_report = classification_report(y_true_binary, y_pred_binary, output_dict=True, zero_division=0)
        metrics['binary_classification_report'] = binary_report
        
        # 3. DETAILED CLASSIFICATION REPORTS
        # Exact classification report
        exact_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['exact_classification_report'] = exact_report
        
        # Per-class F1 scores for exact classification
        metrics['per_class_f1'] = {}
        for label in self.labels:
            if label in exact_report:
                metrics['per_class_f1'][label] = exact_report[label]['f1-score']
            else:
                metrics['per_class_f1'][label] = 0.0
        
        # 4. PRIMARY METRIC (for model selection)
        # Use binary accuracy as the main metric since detecting fake reviews is the primary goal
        metrics['accuracy'] = metrics['binary_accuracy']
        
        # PR-AUC if confidences provided
        if confidences is not None:
            try:
                # Binarize labels for PR-AUC calculation
                lb = LabelBinarizer()
                y_true_bin = lb.fit_transform(y_true)
                
                # Create confidence matrix (assuming highest confidence for predicted class)
                y_score = np.zeros((len(y_pred), len(self.labels)))
                for i, (pred, conf) in enumerate(zip(y_pred, confidences)):
                    if pred in self.label_to_id:
                        y_score[i, self.label_to_id[pred]] = conf
                
                # Calculate PR-AUC for each class
                pr_auc_scores = {}
                for i, label in enumerate(self.labels):
                    if i < y_true_bin.shape[1] and i < y_score.shape[1]:
                        try:
                            pr_auc = average_precision_score(y_true_bin[:, i], y_score[:, i])
                            pr_auc_scores[label] = pr_auc
                        except:
                            pr_auc_scores[label] = 0.0
                    else:
                        pr_auc_scores[label] = 0.0
                
                metrics['pr_auc_per_class'] = pr_auc_scores
                metrics['pr_auc_macro'] = np.mean(list(pr_auc_scores.values()))
                
            except Exception as e:
                logger.warning(f"PR-AUC calculation failed: {e}")
                metrics['pr_auc_per_class'] = {label: 0.0 for label in self.labels}
                metrics['pr_auc_macro'] = 0.0
        
        return metrics
    
    def create_confusion_matrix_plot(self, y_true: List[str], y_pred: List[str], 
                                   title: str, save_path: str):
        """Create and save confusion matrix plot"""
        
        cm = confusion_matrix(y_true, y_pred, labels=self.labels)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.labels, yticklabels=self.labels)
        plt.title(f'Confusion Matrix - {title}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved: {save_path}")
    
    def create_binary_confusion_matrix_plot(self, y_true: List[str], y_pred: List[str], 
                                          title: str, save_path: str):
        """Create and save binary confusion matrix plot"""
        
        binary_labels = ['Genuine', 'Non-Genuine']
        cm = confusion_matrix(y_true, y_pred, labels=binary_labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=binary_labels, yticklabels=binary_labels)
        plt.title(f'Binary Confusion Matrix - {title}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Binary confusion matrix saved: {save_path}")
    
    def create_comparison_plot(self, ft_metrics: Dict, zs_metrics: Dict, save_path: str):
        """Create comparison visualization"""
        
        # Prepare data for plotting - focus on key metrics
        metrics_names = ['Exact Accuracy', 'Exact F1-Macro', 'Binary Accuracy', 'Binary F1-Macro', 'PR-AUC-Macro']
        ft_values = [
            ft_metrics['exact_accuracy'],
            ft_metrics['exact_f1_macro'], 
            ft_metrics['binary_accuracy'],
            ft_metrics.get('binary_f1_macro', 0),
            ft_metrics.get('pr_auc_macro', 0)
        ]
        zs_values = [
            zs_metrics['exact_accuracy'],
            zs_metrics['exact_f1_macro'],
            zs_metrics['binary_accuracy'],
            zs_metrics.get('binary_f1_macro', 0),
            zs_metrics.get('pr_auc_macro', 0)
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bars1 = ax.bar(x - width/2, ft_values, width, label='Fine-tuned BART', color='skyblue')
        bars2 = ax.bar(x + width/2, zs_values, width, label='Zero-shot BART', color='orange')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison: Fine-tuned vs Zero-shot BART')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, rotation=15, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        autolabel(bars1)
        autolabel(bars2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plot saved: {save_path}")
    
    def save_detailed_results(self, results: Dict, save_path: str):
        """Save detailed results to JSON"""
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        results_clean = convert_types(results)
        
        with open(save_path, 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        logger.info(f"Detailed results saved: {save_path}")
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        logger.info("🎯 STARTING COMPREHENSIVE MODEL EVALUATION")
        logger.info("=" * 60)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"evaluation_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load models and data
        self.load_models()
        test_df = self.load_test_data()
        
        texts = test_df['text'].tolist()
        true_labels = test_df['llm_classification'].tolist()
        
        logger.info(f"Evaluating on {len(texts)} test samples")
        
        # Get predictions
        ft_predictions, ft_confidences = self.predict_fine_tuned(texts)
        zs_predictions, zs_confidences = self.predict_zero_shot(texts)
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        ft_metrics = self.calculate_metrics(true_labels, ft_predictions, ft_confidences)
        zs_metrics = self.calculate_metrics(true_labels, zs_predictions, zs_confidences)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        self.create_confusion_matrix_plot(
            true_labels, ft_predictions, 
            "Fine-tuned BART", 
            f"{output_dir}/confusion_matrix_fine_tuned.png"
        )
        
        self.create_confusion_matrix_plot(
            true_labels, zs_predictions,
            "Zero-shot BART",
            f"{output_dir}/confusion_matrix_zero_shot.png"
        )
        
        # Create binary confusion matrices
        true_binary = ['Genuine' if label in ['genuine_positive', 'genuine_negative'] else 'Non-Genuine' for label in true_labels]
        ft_pred_binary = ['Genuine' if label in ['genuine_positive', 'genuine_negative'] else 'Non-Genuine' for label in ft_predictions]
        zs_pred_binary = ['Genuine' if label in ['genuine_positive', 'genuine_negative'] else 'Non-Genuine' for label in zs_predictions]
        
        self.create_binary_confusion_matrix_plot(
            true_binary, ft_pred_binary,
            "Fine-tuned BART (Binary)",
            f"{output_dir}/binary_confusion_matrix_fine_tuned.png"
        )
        
        self.create_binary_confusion_matrix_plot(
            true_binary, zs_pred_binary,
            "Zero-shot BART (Binary)",
            f"{output_dir}/binary_confusion_matrix_zero_shot.png"
        )
        
        self.create_comparison_plot(
            ft_metrics, zs_metrics,
            f"{output_dir}/model_comparison.png"
        )
        
        # Compile results
        results = {
            'evaluation_timestamp': timestamp,
            'test_set_size': len(texts),
            'fine_tuned_model_path': self.fine_tuned_model_path,
            'fine_tuned_metrics': ft_metrics,
            'zero_shot_metrics': zs_metrics,
            'detailed_predictions': {
                'true_labels': true_labels,
                'fine_tuned_predictions': ft_predictions,
                'fine_tuned_confidences': ft_confidences,
                'zero_shot_predictions': zs_predictions,
                'zero_shot_confidences': zs_confidences
            }
        }
        
        # Save detailed results
        self.save_detailed_results(results, f"{output_dir}/detailed_results.json")
        
        # Print summary
        self.print_evaluation_summary(ft_metrics, zs_metrics, output_dir)
        
        return results
    
    def print_evaluation_summary(self, ft_metrics: Dict, zs_metrics: Dict, output_dir: str):
        """Print comprehensive evaluation summary"""
        
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE MODEL EVALUATION RESULTS")
        print("="*80)
        
        print(f"\n📊 OVERALL PERFORMANCE COMPARISON:")
        print(f"{'Metric':<25} {'Fine-tuned BART':<18} {'Zero-shot BART':<18} {'Improvement':<12}")
        print("-" * 75)
        
        # Primary metrics comparison
        primary_metrics = [
            ('Binary Accuracy (Main)', 'binary_accuracy'),
            ('Binary F1-Macro', 'binary_f1_macro'),
            ('Exact Accuracy', 'exact_accuracy'),
            ('Exact F1-Macro', 'exact_f1_macro')
        ]
        
        for name, key in primary_metrics:
            ft_val = ft_metrics.get(key, 0)
            zs_val = zs_metrics.get(key, 0)
            improvement = ((ft_val - zs_val) / zs_val * 100) if zs_val > 0 else 0
            
            print(f"{name:<25} {ft_val:<18.4f} {zs_val:<18.4f} {improvement:+6.1f}%")
        
        print(f"\n🎯 BINARY DETECTION (Genuine vs Non-Genuine) - MAIN TASK:")
        print(f"{'Metric':<25} {'Fine-tuned BART':<18} {'Zero-shot BART':<18} {'Improvement':<12}")
        print("-" * 75)
        
        binary_metrics = [
            ('Detection Accuracy', 'binary_accuracy'),
            ('F1-Genuine Detection', 'binary_f1'),
            ('F1-Fake Detection', 'binary_f1_non_genuine'),
            ('Precision-Genuine', 'binary_precision_genuine'),
            ('Recall-Genuine', 'binary_recall_genuine')
        ]
        
        for name, key in binary_metrics:
            if key.startswith('binary_precision') or key.startswith('binary_recall'):
                # Extract from classification report
                if key == 'binary_precision_genuine':
                    ft_val = ft_metrics.get('binary_classification_report', {}).get('genuine', {}).get('precision', 0)
                    zs_val = zs_metrics.get('binary_classification_report', {}).get('genuine', {}).get('precision', 0)
                elif key == 'binary_recall_genuine':
                    ft_val = ft_metrics.get('binary_classification_report', {}).get('genuine', {}).get('recall', 0)
                    zs_val = zs_metrics.get('binary_classification_report', {}).get('genuine', {}).get('recall', 0)
                else:
                    continue
            else:
                ft_val = ft_metrics.get(key, 0)
                zs_val = zs_metrics.get(key, 0)
            
            improvement = ((ft_val - zs_val) / zs_val * 100) if zs_val > 0 else 0
            print(f"{name:<25} {ft_val:<18.4f} {zs_val:<18.4f} {improvement:+6.1f}%")
        
        print(f"\n📝 EXACT CLASSIFICATION (7-class) - DETAILED ANALYSIS:")
        print(f"{'Metric':<25} {'Fine-tuned BART':<18} {'Zero-shot BART':<18} {'Improvement':<12}")
        print("-" * 75)
        
        exact_metrics = [
            ('Exact Accuracy', 'exact_accuracy'),
            ('Exact F1-Macro', 'exact_f1_macro'),
            ('Exact F1-Weighted', 'exact_f1_weighted')
        ]
        
        for name, key in exact_metrics:
            ft_val = ft_metrics.get(key, 0)
            zs_val = zs_metrics.get(key, 0)
            improvement = ((ft_val - zs_val) / zs_val * 100) if zs_val > 0 else 0
            
            print(f"{name:<25} {ft_val:<18.4f} {zs_val:<18.4f} {improvement:+6.1f}%")
        
        print(f"\n🎯 PER-CLASS F1-SCORE COMPARISON:")
        print(f"{'Class':<18} {'Fine-tuned':<12} {'Zero-shot':<12} {'Improvement':<12}")
        print("-" * 56)
        
        for label in self.labels:
            ft_f1 = ft_metrics['per_class_f1'].get(label, 0)
            zs_f1 = zs_metrics['per_class_f1'].get(label, 0)
            improvement = ((ft_f1 - zs_f1) / zs_f1 * 100) if zs_f1 > 0 else 0
            
            print(f"{label:<18} {ft_f1:<12.4f} {zs_f1:<12.4f} {improvement:+6.1f}%")
        
        print(f"\n📁 RESULTS SAVED TO: {output_dir}/")
        print(f"   ✅ Confusion matrices: confusion_matrix_*.png")
        print(f"   ✅ Binary confusion matrices: binary_confusion_matrix_*.png")
        print(f"   ✅ Comparison chart: model_comparison.png") 
        print(f"   ✅ Detailed results: detailed_results.json")
        
        print(f"\n🏆 SUMMARY:")
        ft_acc = ft_metrics['accuracy']
        zs_acc = zs_metrics['accuracy']
        ft_binary_f1 = ft_metrics.get('binary_f1', 0)
        zs_binary_f1 = zs_metrics.get('binary_f1', 0)
        winner = "Fine-tuned" if ft_acc > zs_acc else "Zero-shot"
        print(f"   Best performing model: {winner} BART")
        print(f"   Overall accuracy improvement: {((ft_acc - zs_acc) / zs_acc * 100):+.1f}%")
        print(f"   Genuine detection F1 improvement: {((ft_binary_f1 - zs_binary_f1) / zs_binary_f1 * 100):+.1f}%")


def main():
    """Main evaluation function"""
    print("🎯 COMPREHENSIVE MODEL EVALUATION")
    print("Comparing Fine-tuned vs Zero-shot BART Performance")
    print("=" * 65)
    
    # Configuration
    fine_tuned_model_path = "enhanced_bart_review_classifier_20250830_173055"
    test_data_path = "../data/data_all_test.csv"
    
    print(f"\n📋 EVALUATION CONFIGURATION:")
    print(f"  Fine-tuned model: {fine_tuned_model_path}")
    print(f"  Test dataset: {test_data_path}")
    print(f"  Baseline: facebook/bart-large-mnli (zero-shot)")
    print(f"  Metrics: Accuracy, F1-scores, PR-AUC")
    
    try:
        # Initialize evaluator
        evaluator = ComprehensiveModelEvaluator(
            fine_tuned_model_path=fine_tuned_model_path,
            test_data_path=test_data_path
        )
        
        # Run evaluation
        results = evaluator.run_evaluation()
        
        print(f"\n✅ Comprehensive evaluation completed successfully!")
        print(f"🎯 Check the evaluation_results_* folder for detailed analysis!")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
