#!/usr/bin/env python3
"""
Google Reviews Evaluation System
Evaluates LLM-labeled reviews using local Hugging Face BART classifier
Compares LLM predictions vs local model predictions
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List
from transformers import pipeline
import torch
from sklearn.metrics import classification_report, confusion_matrix

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GoogleReviewsEvaluator:
    """
    Evaluates LLM-labeled Google Reviews using local BART classifier
    Provides comparison metrics and analysis
    """
    
    def __init__(self):
        """Initialize with local Hugging Face BART classifier"""
        print("Loading local BART classifier for evaluation...")
        
        # Initialize zero-shot classification pipeline
        self.bart_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Define evaluation categories (matching LLM categories exactly)
        self.evaluation_categories = [
            "spam",
            "advertisement", 
            "irrelevant content",
            "fake rant",
            "inappropriate content",
            "genuine positive review",
            "genuine negative review"
        ]
        
        print(f"BART classifier loaded. Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
    def classify_with_bart(self, reviews_batch: list) -> list:
        """Classify reviews using local BART model"""
        
        results = []
        
        for text in reviews_batch:
            try:
                # Run zero-shot classification
                result = self.bart_classifier(text, self.evaluation_categories)
                
                # Get top prediction
                top_label = result['labels'][0]
                confidence = result['scores'][0]
                
                # Map BART labels to our standard categories
                label_map = {
                    'genuine positive review': 'genuine_positive',
                    'genuine negative review': 'genuine_negative',
                    'spam': 'spam',
                    'advertisement': 'advertisement',
                    'irrelevant content': 'irrelevant',
                    'fake rant': 'fake_rant',
                    'inappropriate content': 'inappropriate'
                }
                
                mapped_label = label_map.get(top_label, 'spam')
                
                results.append({
                    'bart_classification': mapped_label,
                    'bart_confidence': confidence,
                    'bart_raw_scores': dict(zip(result['labels'], result['scores']))
                })
                
            except Exception as e:
                logger.error(f"Error classifying review with BART: {str(e)}")
                results.append({
                    'bart_classification': 'spam',
                    'bart_confidence': 0.0,
                    'bart_raw_scores': {}
                })
        
        return results
    
    def evaluate_labeled_dataset(self, csv_file_path: str, sample_size: int = None) -> pd.DataFrame:
        """Evaluate LLM-labeled dataset using BART classifier"""
        
        logger.info(f"Loading labeled dataset from {csv_file_path}")
        
        # Load labeled dataset
        df = pd.read_csv(csv_file_path)
        
        # Ensure required columns exist
        required_cols = ['text', 'llm_classification', 'llm_confidence']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter valid reviews
        df = df[df['text'].notna()]
        df = df[df['text'].astype(str).str.strip().str.len() > 0]
        
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        logger.info(f"Evaluating {len(df)} reviews with BART classifier")
        
        # Process reviews in batches for efficiency
        batch_size = 16  # Larger batches for evaluation
        review_texts = df['text'].astype(str).tolist()
        
        all_bart_results = []
        total_batches = (len(review_texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(review_texts))
            
            batch_texts = review_texts[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")
            
            # Classify with BART
            batch_results = self.classify_with_bart(batch_texts)
            all_bart_results.extend(batch_results)
        
        # Add BART results to DataFrame - create new columns properly
        bart_classifications = [r['bart_classification'] for r in all_bart_results]
        bart_confidences = [r['bart_confidence'] for r in all_bart_results]
        bart_raw_scores = [r['bart_raw_scores'] for r in all_bart_results]
        
        # Add new columns to DataFrame
        df = df.copy()  # Ensure we can modify the DataFrame
        df['bart_classification'] = bart_classifications
        df['bart_confidence'] = bart_confidences
        df['bart_raw_scores'] = bart_raw_scores
        
        logger.info("BART evaluation completed")
        
        return df
    
    def analyze_agreement(self, df: pd.DataFrame) -> Dict:
        """Analyze agreement between LLM and BART classifications"""
        
        # Calculate agreement metrics
        llm_preds = df['llm_classification'].tolist()
        bart_preds = df['bart_classification'].tolist()
        
        # Overall agreement
        agreement = np.mean([llm == bart for llm, bart in zip(llm_preds, bart_preds)])
        
        # Agreement by category
        categories = df['llm_classification'].unique()
        category_agreement = {}
        
        for category in categories:
            mask = df['llm_classification'] == category
            if mask.sum() > 0:
                cat_llm = df[mask]['llm_classification'].tolist()
                cat_bart = df[mask]['bart_classification'].tolist()
                cat_agreement = np.mean([llm == bart for llm, bart in zip(cat_llm, cat_bart)])
                category_agreement[category] = cat_agreement
        
        # Confidence correlation
        confidence_corr = df['llm_confidence'].corr(df['bart_confidence'])
        
        # Classification distribution comparison
        llm_dist = df['llm_classification'].value_counts(normalize=True).sort_index()
        bart_dist = df['bart_classification'].value_counts(normalize=True).sort_index()
        
        analysis = {
            'overall_agreement': agreement,
            'category_agreement': category_agreement,
            'confidence_correlation': confidence_corr,
            'llm_distribution': llm_dist.to_dict(),
            'bart_distribution': bart_dist.to_dict(),
            'total_reviews': len(df)
        }
        
        return analysis
    
    def generate_evaluation_report(self, df: pd.DataFrame, output_file: str = None) -> Dict:
        """Generate comprehensive evaluation report"""
        
        logger.info("Generating evaluation report...")
        
        # Perform analysis
        analysis = self.analyze_agreement(df)
        
        # Print report
        print("\\n" + "="*80)
        print("GOOGLE REVIEWS EVALUATION REPORT")
        print("LLM vs BART Classifier Comparison")
        print("="*80)
        
        print(f"\\n📊 OVERALL METRICS:")
        print(f"  Total Reviews Evaluated: {analysis['total_reviews']:,}")
        print(f"  Overall Agreement: {analysis['overall_agreement']:.3f} ({analysis['overall_agreement']*100:.1f}%)")
        print(f"  Confidence Correlation: {analysis['confidence_correlation']:.3f}")
        
        print(f"\\n🎯 CATEGORY-WISE AGREEMENT:")
        for category, agreement in sorted(analysis['category_agreement'].items()):
            count = (df['llm_classification'] == category).sum()
            print(f"  {category:<20}: {agreement:.3f} ({agreement*100:.1f}%) - {count} reviews")
        
        print(f"\\n📈 CLASSIFICATION DISTRIBUTION COMPARISON:")
        print(f"{'Category':<20} {'LLM %':<10} {'BART %':<10} {'Difference':<12}")
        print("-" * 52)
        
        all_categories = set(list(analysis['llm_distribution'].keys()) + list(analysis['bart_distribution'].keys()))
        for category in sorted(all_categories):
            llm_pct = analysis['llm_distribution'].get(category, 0) * 100
            bart_pct = analysis['bart_distribution'].get(category, 0) * 100
            diff = abs(llm_pct - bart_pct)
            print(f"{category:<20} {llm_pct:<10.1f} {bart_pct:<10.1f} {diff:<12.1f}")
        
        # Disagreement analysis
        print(f"\\n🔍 DISAGREEMENT ANALYSIS:")
        disagreements = df[df['llm_classification'] != df['bart_classification']]
        if len(disagreements) > 0:
            print(f"  Total Disagreements: {len(disagreements)} ({len(disagreements)/len(df)*100:.1f}%)")
            
            # Show top disagreement patterns
            disagreement_patterns = disagreements.groupby(['llm_classification', 'bart_classification']).size().sort_values(ascending=False)
            print(f"  Top Disagreement Patterns:")
            for (llm, bart), count in disagreement_patterns.head(5).items():
                pct = count / len(disagreements) * 100
                print(f"    LLM: {llm:<15} → BART: {bart:<15} ({count} cases, {pct:.1f}%)")
            
            # Show sample disagreements
            print(f"\\n📝 SAMPLE DISAGREEMENTS:")
            sample_disagreements = disagreements.sample(n=min(3, len(disagreements)), random_state=42)
            for i, (_, row) in enumerate(sample_disagreements.iterrows()):
                text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
                print(f"    {i+1}. Text: \"{text_preview}\"")
                print(f"       LLM: {row['llm_classification']} (conf: {row['llm_confidence']:.3f})")
                print(f"       BART: {row['bart_classification']} (conf: {row['bart_confidence']:.3f})")
                print()
        
        # Save detailed results
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f'google_reviews_evaluation_{timestamp}.csv'
        
        df.to_csv(output_file, index=False)
        print(f"\\n💾 Detailed results saved to: {output_file}")
        
        return analysis

def main():
    """Main evaluation workflow"""
    
    print("🔬 GOOGLE REVIEWS EVALUATION SYSTEM")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = GoogleReviewsEvaluator()
    
    # Get labeled dataset file
    labeled_file = input("Enter path to labeled CSV file (or press Enter for latest): ").strip()
    if not labeled_file:
        # Look for latest labeled file
        import glob
        files = glob.glob("google_reviews_labeled_*.csv")
        if files:
            labeled_file = max(files)  # Get most recent
            print(f"Using latest file: {labeled_file}")
        else:
            print("❌ No labeled files found. Please run google_reviews_labeler.py first.")
            return
    
    # Get sample size
    try:
        sample_input = input("Enter number of reviews to evaluate (or press Enter for 500): ").strip()
        sample_size = int(sample_input) if sample_input else 500
    except ValueError:
        sample_size = 500
    
    print(f"\\nEvaluating up to {sample_size} reviews with BART classifier...")
    
    try:
        # Evaluate dataset
        evaluated_df = evaluator.evaluate_labeled_dataset(labeled_file, sample_size)
        
        # Generate report
        analysis = evaluator.generate_evaluation_report(evaluated_df)
        
        print(f"\\n✅ Evaluation completed!")
        print(f"📊 Agreement rate: {analysis['overall_agreement']*100:.1f}%")
        
        return evaluated_df, analysis
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find file {labeled_file}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
