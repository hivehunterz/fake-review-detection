#!/usr/bin/env python3
"""
Fine-tuned BART Evaluator for Google Reviews
Compare fine-tuned BART performance with LLM labels and zero-shot BART
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from sklearn.metrics import classification_report, accuracy_score
import json
from datetime import datetime

class FineTunedBARTEvaluator:
    """
    Evaluate fine-tuned BART model performance on Google Reviews
    """
    
    def __init__(self, model_path: str):
        """
        Initialize with fine-tuned BART model
        
        Args:
            model_path: Path to fine-tuned BART model
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.zero_shot_classifier = None
        
        # Load fine-tuned model
        self._load_fine_tuned_model()
        
        # Load zero-shot model for comparison
        self._load_zero_shot_model()
        
    def _load_fine_tuned_model(self):
        """Load the fine-tuned BART model"""
        print(f"Loading fine-tuned BART model from {self.model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        
        # Load label mappings
        with open(f"{self.model_path}/label_mappings.json", 'r') as f:
            mappings = json.load(f)
            self.id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}
            self.label_to_id = mappings['label_to_id']
            self.labels = mappings['labels']
        
        print(f"✅ Fine-tuned model loaded with {len(self.labels)} labels")
        
    def _load_zero_shot_model(self):
        """Load zero-shot BART model for comparison"""
        print("Loading zero-shot BART model for comparison...")
        
        self.zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        
        print("✅ Zero-shot model loaded")
    
    def predict_fine_tuned(self, texts: list) -> list:
        """
        Make predictions using fine-tuned BART model with GPU batching
        
        Args:
            texts: List of review texts
            
        Returns:
            List of predicted labels
        """
        predictions = []
        self.model.eval()
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Process in batches for GPU efficiency
        batch_size = 16
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                
                # Move inputs to GPU
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Predict batch
                outputs = self.model(**inputs)
                predicted_ids = torch.argmax(outputs.logits, dim=-1)
                
                # Convert to labels
                for pred_id in predicted_ids:
                    predicted_label = self.id_to_label[pred_id.item()]
                    predictions.append(predicted_label)
        
        return predictions
    
    def predict_zero_shot(self, texts: list) -> list:
        """
        Make predictions using zero-shot BART model with GPU batching
        
        Args:
            texts: List of review texts
            
        Returns:
            List of predicted labels
        """
        # Zero-shot categories (matching original evaluator)
        categories = [
            "spam", "advertisement", "irrelevant content", "fake rant",
            "inappropriate content", "genuine positive review", "genuine negative review"
        ]
        
        # Label mapping
        label_map = {
            'genuine positive review': 'genuine_positive',
            'genuine negative review': 'genuine_negative',
            'spam': 'spam', 
            'advertisement': 'advertisement',
            'irrelevant content': 'irrelevant', 
            'fake rant': 'fake_rant',
            'inappropriate content': 'inappropriate'
        }
        
        predictions = []
        
        # Process in batches for GPU efficiency
        batch_size = 8  # Smaller batch for zero-shot as it's more memory intensive
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Process batch
            for text in batch_texts:
                result = self.zero_shot_classifier(text, categories)
                top_label = result['labels'][0]
                mapped_label = label_map.get(top_label, 'spam')
                predictions.append(mapped_label)
        
        return predictions
    
    def evaluate_model(self, csv_file: str, sample_size: int = None, 
                      text_column: str = 'text', 
                      label_column: str = 'llm_classification') -> dict:
        """
        Comprehensive evaluation of fine-tuned vs zero-shot BART
        
        Args:
            csv_file: Path to labeled CSV file
            sample_size: Number of samples to evaluate (None for all)
            text_column: Column name for review text
            label_column: Column name for true labels
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\\n🔬 COMPREHENSIVE BART EVALUATION")
        print("=" * 80)
        
        # Load data
        df = pd.read_csv(csv_file)
        df = df[df[text_column].notna()]
        df = df[df[label_column].isin(self.labels)]
        
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        texts = df[text_column].astype(str).tolist()
        true_labels = df[label_column].tolist()
        
        print(f"Evaluating on {len(texts)} reviews...")
        print(f"Label distribution:")
        for label, count in df[label_column].value_counts().items():
            print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
        
        # Predictions
        print("\\n🎯 Making Fine-tuned BART predictions...")
        print(f"Processing {len(texts)} reviews in batches of 16 on GPU...")
        fine_tuned_preds = self.predict_fine_tuned(texts)
        
        print("📊 Making Zero-shot BART predictions...")
        print(f"Processing {len(texts)} reviews in batches of 8 on GPU...")
        zero_shot_preds = self.predict_zero_shot(texts)
        
        # Calculate metrics
        fine_tuned_accuracy = accuracy_score(true_labels, fine_tuned_preds)
        zero_shot_accuracy = accuracy_score(true_labels, zero_shot_preds)
        
        # Generate reports
        fine_tuned_report = classification_report(
            true_labels, fine_tuned_preds, 
            target_names=self.labels, 
            output_dict=True, 
            zero_division=0
        )
        
        zero_shot_report = classification_report(
            true_labels, zero_shot_preds, 
            target_names=self.labels, 
            output_dict=True, 
            zero_division=0
        )
        
        # Results summary
        improvement = fine_tuned_accuracy - zero_shot_accuracy
        
        print("\\n" + "="*80)
        print("🎯 BART MODEL COMPARISON RESULTS")
        print("="*80)
        
        print(f"\\n📊 OVERALL ACCURACY:")
        print(f"  Zero-shot BART:   {zero_shot_accuracy:.3f} ({zero_shot_accuracy*100:.1f}%)")
        print(f"  Fine-tuned BART:  {fine_tuned_accuracy:.3f} ({fine_tuned_accuracy*100:.1f}%)")
        print(f"  Improvement:      {improvement:+.3f} ({improvement*100:+.1f}%)")
        
        print(f"\\n📈 ZERO-SHOT BART DETAILED RESULTS:")
        print(classification_report(true_labels, zero_shot_preds, target_names=self.labels, digits=3))
        
        print(f"\\n🎯 FINE-TUNED BART DETAILED RESULTS:")
        print(classification_report(true_labels, fine_tuned_preds, target_names=self.labels, digits=3))
        
        # Agreement analysis
        agreement = sum([1 for ft, zs in zip(fine_tuned_preds, zero_shot_preds) if ft == zs]) / len(fine_tuned_preds)
        llm_ft_agreement = sum([1 for true, ft in zip(true_labels, fine_tuned_preds) if true == ft]) / len(true_labels)
        llm_zs_agreement = sum([1 for true, zs in zip(true_labels, zero_shot_preds) if true == zs]) / len(true_labels)
        
        print(f"\\n🤝 MODEL AGREEMENT:")
        print(f"  LLM vs Fine-tuned BART:  {llm_ft_agreement:.3f} ({llm_ft_agreement*100:.1f}%)")
        print(f"  LLM vs Zero-shot BART:   {llm_zs_agreement:.3f} ({llm_zs_agreement*100:.1f}%)")
        print(f"  Fine-tuned vs Zero-shot: {agreement:.3f} ({agreement*100:.1f}%)")
        
        # Save detailed results
        results_df = pd.DataFrame({
            'text': texts,
            'true_label': true_labels,
            'fine_tuned_pred': fine_tuned_preds,
            'zero_shot_pred': zero_shot_preds,
            'fine_tuned_correct': [t == p for t, p in zip(true_labels, fine_tuned_preds)],
            'zero_shot_correct': [t == p for t, p in zip(true_labels, zero_shot_preds)],
            'models_agree': [ft == zs for ft, zs in zip(fine_tuned_preds, zero_shot_preds)]
        })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"fine_tuned_bart_evaluation_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        print(f"\\n💾 Detailed results saved to: {results_file}")
        
        return {
            'fine_tuned_accuracy': fine_tuned_accuracy,
            'zero_shot_accuracy': zero_shot_accuracy,
            'improvement': improvement,
            'fine_tuned_report': fine_tuned_report,
            'zero_shot_report': zero_shot_report,
            'agreement_scores': {
                'llm_fine_tuned': llm_ft_agreement,
                'llm_zero_shot': llm_zs_agreement,
                'fine_tuned_zero_shot': agreement
            },
            'results_file': results_file
        }
    
    def predict_single_text(self, text: str) -> dict:
        """
        Predict classification for a single text using both models
        
        Args:
            text: Review text to classify
            
        Returns:
            Dictionary with predictions from both models
        """
        print(f"\\n🔍 ANALYZING TEXT:")
        print(f"'{text[:100]}{'...' if len(text) > 100 else ''}'")
        print("-" * 60)
        
        # Fine-tuned prediction
        fine_tuned_pred = self.predict_fine_tuned([text])[0]
        
        # Zero-shot prediction  
        zero_shot_pred = self.predict_zero_shot([text])[0]
        
        # Get confidence scores for fine-tuned model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            
            # Get probabilities
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            confidence = torch.max(probs).item()
            
            # Get all class probabilities
            class_probs = {}
            for i, prob in enumerate(probs):
                label = self.id_to_label[i]
                class_probs[label] = prob.item()
        
        results = {
            'text': text,
            'fine_tuned_prediction': fine_tuned_pred,
            'zero_shot_prediction': zero_shot_pred,
            'confidence': confidence,
            'class_probabilities': class_probs,
            'models_agree': fine_tuned_pred == zero_shot_pred
        }
        
        # Display results
        print(f"🎯 FINE-TUNED BART: {fine_tuned_pred} (confidence: {confidence:.3f})")
        print(f"📊 ZERO-SHOT BART:  {zero_shot_pred}")
        print(f"🤝 MODELS AGREE:    {results['models_agree']}")
        
        print(f"\\n📈 CLASS PROBABILITIES:")
        sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_probs:
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  {label:>15}: {prob:.3f} |{bar}|")
        
        return results
    
    def interactive_prediction(self):
        """
        Interactive mode for testing individual texts
        """
        print(f"\\n🚀 INTERACTIVE PREDICTION MODE")
        print("=" * 60)
        print("Enter review texts to classify (type 'quit' to exit)")
        print("Examples:")
        print("  - 'Great food and service!'")
        print("  - 'This place is terrible, worst experience ever'")
        print("  - 'Click here to win $1000!'")
        print("=" * 60)
        
        while True:
            try:
                text = input("\\n📝 Enter review text: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not text:
                    print("❌ Please enter some text")
                    continue
                
                # Make prediction
                result = self.predict_single_text(text)
                
                # Ask for another prediction
                continue_choice = input("\\n🔄 Analyze another text? (y/N): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    print("👋 Goodbye!")
                    break
                    
            except KeyboardInterrupt:
                print("\\n\\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue

def main():
    """Main evaluation workflow"""
    
    print("🎯 FINE-TUNED BART EVALUATION")
    print("Comparing fine-tuned vs zero-shot BART performance")
    print("=" * 70)
    
    # Configuration
    model_path = "enhanced_bart_review_classifier_20250829_185505"
    csv_file = "google_reviews_labeled_combined_with_json.csv"
    sample_size = 300  # Reduced for faster GPU evaluation
    
    print(f"\\n📋 OPTIONS:")
    print("1. Run full evaluation on dataset")
    print("2. Interactive text prediction")
    print("3. Both")
    
    try:
        choice = input("\\nChoose option (1/2/3): ").strip()
        
        # Initialize evaluator
        evaluator = FineTunedBARTEvaluator(model_path)
        
        if choice in ['1', '3']:
            print(f"\\n📋 EVALUATION CONFIGURATION:")
            print(f"  Fine-tuned model: {model_path}")
            print(f"  Dataset: {csv_file}")
            print(f"  Sample size: {sample_size} reviews")
            
            # Run evaluation
            results = evaluator.evaluate_model(csv_file, sample_size=sample_size)
            
            print(f"\\n🎉 EVALUATION COMPLETED!")
            print(f"Fine-tuned BART achieved {results['improvement']*100:+.1f}% improvement!")
            print(f"Final accuracy: {results['fine_tuned_accuracy']*100:.1f}%")
        
        if choice in ['2', '3']:
            # Interactive prediction mode
            evaluator.interactive_prediction()
        
        if choice not in ['1', '2', '3']:
            print("❌ Invalid choice. Running full evaluation...")
            results = evaluator.evaluate_model(csv_file, sample_size=sample_size)
            print(f"Final accuracy: {results['fine_tuned_accuracy']*100:.1f}%")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find file - {str(e)}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
