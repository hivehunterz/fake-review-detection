#!/usr/bin/env python3
"""
Example Usage Script for Fake Review Detection System
This script demonstrates how to use the main components of the system.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Set up logging for the example script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('example_usage.log')
        ]
    )

def example_data_labeling():
    """Example: Label reviews using the multi-LLM system"""
    print("\n🏷️  EXAMPLE: Data Labeling with Multi-LLM System (Stage 0)")
    print("=" * 60)
    
    try:
        # Import from the organized structure
        sys.path.insert(0, str(project_root / "data_preprocessing"))
        from google_reviews_labeler import GoogleReviewsLabeler
        
        # Example reviews to label
        sample_reviews = [
            "This product is amazing! Best purchase I've ever made.",
            "Buy now! Limited time offer! Click here for discount!",
            "Good product but delivery was slow.",
            "Terrible quality, completely useless, don't waste your money!",
            "Nice place for shopping"
        ]
        
        print("📝 Sample reviews to label:")
        for i, review in enumerate(sample_reviews, 1):
            print(f"   {i}. {review}")
        
        print("\n💡 Note: Configure your API keys in config.ini before running the labeler.")
        print("   The labeler will use Groq Llama 3.3-70B for fast, accurate labeling.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure to run this from the project root directory.")

def example_bart_finetuning():
    """Example: Fine-tune BART model on labeled data"""
    print("\n🤖 EXAMPLE: BART Model Fine-tuning (Stage 1)")
    print("=" * 60)
    
    try:
        # Import from stage 1 folder
        sys.path.insert(0, str(project_root / "stage_1_bart_finetuning"))
        
        print("📊 BART Fine-tuning Process:")
        print("   1. Load labeled dataset from data/ folder")
        print("   2. Prepare training/validation splits (80/20)")
        print("   3. Fine-tune facebook/bart-large-mnli model")
        print("   4. Achieve 83% accuracy on 7-class classification")
        
        print("\n🎯 Classification Categories:")
        categories = [
            "genuine_positive (45.5%)",
            "genuine_negative (17.0%)",
            "spam (14.3%)",
            "advertisement (4.3%)",
            "irrelevant (10.6%)",
            "fake_rant (4.8%)",
            "inappropriate (3.6%)"
        ]
        for cat in categories:
            print(f"   • {cat}")
        
        print("\n💡 To run: cd stage_1_bart_finetuning && python enhanced_bart_finetune.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_model_evaluation():
    """Example: Evaluate fine-tuned models"""
    print("\n📈 EXAMPLE: Model Evaluation (Stage 1)")
    print("=" * 60)
    
    print("🔍 Evaluation Metrics:")
    print("   • Overall Accuracy: 83.0%")
    print("   • Macro F1-Score: 69.5%")
    print("   • Weighted F1-Score: 81.1%")
    print("   • Improvement over Zero-shot: +23.0%")
    
    print("\n📊 Class-wise Performance:")
    performance = [
        ("Spam Detection", "93.8% F1"),
        ("Advertisement", "90.7% F1"),
        ("Inappropriate", "68.8% F1"),
        ("Genuine Reviews", "81.9% F1")
    ]
    for metric, score in performance:
        print(f"   • {metric}: {score}")
    
    print("\n💡 To run: cd stage_1_bart_finetuning && python evaluate_fine_tuned_bart.py")

def example_relevancy_check():
    """Example: Secondary relevancy analysis"""
    print("\n🎯 EXAMPLE: Relevancy Check (Stage 3)")
    print("=" * 60)
    
    try:
        # Import from stage 3 folder
        sys.path.insert(0, str(project_root / "stage_3_relevancy_check"))
        
        print("🔄 Secondary Analysis Process:")
        print("   1. Takes BART predictions as input")
        print("   2. Performs relevancy scoring using layer2.py")
        print("   3. Validates context against business/product info")
        print("   4. Provides final quality scores")
        
        print("\n🎯 Relevancy Features:")
        features = [
            "Context validation against product/service",
            "Review-business relevance scoring", 
            "Secondary classification layer",
            "Final quality assessment"
        ]
        for feature in features:
            print(f"   • {feature}")
        
        print("\n💡 To run: cd stage_3_relevancy_check && python layer2.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_metadata_anomaly():
    """Example: Metadata anomaly detection"""
    print("\n🔍 EXAMPLE: Metadata Anomaly Detection (Stage 2)")
    print("=" * 60)
    
    print("🚧 Status: Under Development")
    print("\n📋 Planned Features:")
    features = [
        "Temporal pattern analysis",
        "User behavior anomaly detection",
        "Geographic irregularity detection",
        "Rating distribution analysis",
        "Device/platform pattern analysis"
    ]
    for feature in features:
        print(f"   • {feature}")
    
    print("\n💡 Future files:")
    files = [
        "metadata_analyzer.py",
        "temporal_anomaly_detector.py", 
        "user_behavior_analyzer.py",
        "rating_distribution_analyzer.py"
    ]
    for file in files:
        print(f"   • {file}")

def main():
    """Main function to run all examples"""
    setup_logging()
    
    print("🛡️  Advanced Fake Review Detection System - Example Usage")
    print("=" * 70)
    print("This script demonstrates the multi-stage detection pipeline.")
    
    # Run all examples
    example_data_labeling()
    example_bart_finetuning()
    example_model_evaluation()
    example_metadata_anomaly()
    example_relevancy_check()
    
    print("\n" + "=" * 70)
    print("🎉 All examples completed!")
    print("💡 Check the individual stage folders for detailed documentation.")
    print("📁 Project structure now organized into logical stages.")

if __name__ == "__main__":
    main()