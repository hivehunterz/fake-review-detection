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
    print("\n🏷️  EXAMPLE: Data Labeling with Multi-LLM System")
    print("=" * 60)
    
    try:
        from google_reviews_labeler import GoogleReviewsLabeler
        
        # Example reviews to label
        sample_reviews = [
            "This product is amazing! Best purchase I've ever made.",
            "Buy now! Limited time offer! Click here for discount!",
            "Good product but delivery was slow.",
            "Terrible quality, completely useless, don't waste your money!",
            "Nice place for shopping"
        ]
        
        print("Sample reviews to classify:")
        for i, review in enumerate(sample_reviews, 1):
            print(f"{i}. {review}")
        
        print("\n⚠️  Note: To run actual labeling, you need:")
        print("   - Valid API keys in environment variables")
        print("   - JSON file with review data")
        print("   - Run: python google_reviews_labeler.py")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

def example_model_training():
    """Example: Train a BART model for review classification"""
    print("\n🎯 EXAMPLE: BART Model Training")
    print("=" * 60)
    
    try:
        from enhanced_bart_finetune import EnhancedBARTFineTuner
        
        print("Training configuration:")
        print("  - Model: facebook/bart-large-mnli")
        print("  - Epochs: 3")
        print("  - Batch size: 8")
        print("  - Learning rate: 1e-5")
        
        print("\n⚠️  Note: To run actual training, you need:")
        print("   - Labeled dataset CSV file")
        print("   - GPU recommended for faster training")
        print("   - Run: python enhanced_bart_finetune.py")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")

def example_model_evaluation():
    """Example: Evaluate a trained model"""
    print("\n📊 EXAMPLE: Model Evaluation")
    print("=" * 60)
    
    try:
        from evaluate_fine_tuned_bart import FineTunedBARTEvaluator
        
        print("Evaluation features:")
        print("  - Compare fine-tuned vs zero-shot performance")
        print("  - Detailed classification metrics")
        print("  - Per-class performance analysis")
        print("  - Model agreement statistics")
        
        print("\n⚠️  Note: To run actual evaluation, you need:")
        print("   - Trained model directory")
        print("   - Test dataset CSV file")
        print("   - Run: python evaluate_fine_tuned_bart.py")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")

def example_interactive_prediction():
    """Example: Interactive prediction with a trained model"""
    print("\n🔮 EXAMPLE: Interactive Prediction")
    print("=" * 60)
    
    sample_predictions = [
        {
            "text": "Amazing product, highly recommend!",
            "predicted_class": "genuine_positive",
            "confidence": 0.92
        },
        {
            "text": "Click here for amazing deals!!!",
            "predicted_class": "advertisement",
            "confidence": 0.88
        },
        {
            "text": "Product is okay, nothing special",
            "predicted_class": "genuine_negative",
            "confidence": 0.75
        }
    ]
    
    print("Sample predictions:")
    for pred in sample_predictions:
        print(f"Text: '{pred['text']}'")
        print(f"  → Class: {pred['predicted_class']}")
        print(f"  → Confidence: {pred['confidence']:.2f}")
        print()

def check_environment():
    """Check if the environment is properly set up"""
    print("\n🔧 ENVIRONMENT CHECK")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("⚠️  Warning: Python 3.8+ is recommended")
    else:
        print("✅ Python version is compatible")
    
    # Check key dependencies
    required_packages = [
        'torch', 'transformers', 'pandas', 'sklearn', 
        'numpy', 'groq', 'openai'
    ]
    
    print("\nDependency check:")
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Run: pip install {package}")
    
    # Check for API keys
    print("\nAPI Keys check:")
    api_keys = ['GROQ_API_KEY', 'OPENAI_API_KEY']
    for key in api_keys:
        if os.getenv(key):
            print(f"✅ {key} is set")
        else:
            print(f"⚠️  {key} not found in environment")

def main():
    """Main example function"""
    setup_logging()
    
    print("🛡️  FAKE REVIEW DETECTION SYSTEM")
    print("Advanced Example Usage Guide")
    print("=" * 60)
    
    # Check environment
    check_environment()
    
    # Run examples
    example_data_labeling()
    example_model_training()
    example_model_evaluation()
    example_interactive_prediction()
    
    print("\n🚀 GETTING STARTED")
    print("=" * 60)
    print("1. Set up environment: conda create -n review_classifier python=3.12")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Set API keys in environment variables")
    print("4. Prepare your review data in JSON format")
    print("5. Run labeling: python google_reviews_labeler.py")
    print("6. Train model: python enhanced_bart_finetune.py")
    print("7. Evaluate: python evaluate_fine_tuned_bart.py")
    
    print("\n📚 For more information, see:")
    print("   - README.md for detailed setup instructions")
    print("   - docs/BART_FINETUNING_GUIDE.md for training details")
    print("   - CONTRIBUTING.md for development guidelines")

if __name__ == "__main__":
    main()
