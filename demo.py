"""
🎯 DEMO SCRIPT - Review Quality Detection System
Demonstrates how to use the complete ML pipeline

Usage examples:
1. Train all models: python scripts/training/train_all_models.py
2. Predict batch: python scripts/prediction/predict_review_quality.py --input data/data_all_test.csv
3. Predict single: python scripts/prediction/predict_review_quality.py --text "Great service and food!"
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

def demo_training():
    """Demo: Train all models"""
    print("🚀 DEMO: Training All Models")
    print("=" * 50)
    print("Command: python scripts/training/train_all_models.py")
    print()
    print("This will train:")
    print("1. 🤖 BART fine-tuned model (7-class text classification)")
    print("2. 📊 Enhanced metadata analyzer (ML anomaly detection)")
    print("3. 🔮 Advanced fusion model (gradient boosting)")
    print()
    print("Models will be saved in: models/")
    print()

def demo_batch_prediction():
    """Demo: Batch prediction"""
    print("🔍 DEMO: Batch Review Prediction")
    print("=" * 50)
    print("Command: python scripts/prediction/predict_review_quality.py --input data/data_all_test.csv")
    print()
    print("This will:")
    print("1. Load 242 reviews from test data")
    print("2. Run complete ML pipeline (no heuristics)")
    print("3. Generate quality predictions")
    print("4. Save results to: output/prediction_results.csv")
    print()
    print("Output includes:")
    print("- BART predictions and confidences")
    print("- P_BAD risk scores")
    print("- Metadata anomaly scores")
    print("- Final quality predictions")
    print("- Routing decisions")
    print()

def demo_single_prediction():
    """Demo: Single review prediction"""
    print("💎 DEMO: Single Review Prediction")
    print("=" * 50)
    print('Command: python scripts/prediction/predict_review_quality.py --text "Amazing food and excellent service!"')
    print()
    print("This will:")
    print("1. Process single review through ML pipeline")
    print("2. Show detailed prediction breakdown")
    print("3. Display confidence scores and routing decision")
    print()

def show_project_structure():
    """Show clean project structure"""
    print("📁 PROJECT STRUCTURE")
    print("=" * 50)
    print("""
fake-review-detection/
├── data/                          # Training and test data
│   ├── data_all_training.csv     # Training dataset
│   └── data_all_test.csv         # Test dataset
├── core/                          # Core ML modules
│   ├── stage1_bart/              # BART text classification
│   ├── stage2_metadata/          # Metadata anomaly detection
│   └── fusion/                   # Advanced fusion model
├── scripts/                       # Executable scripts
│   ├── training/                 # Model training scripts
│   │   └── train_all_models.py   # Complete training pipeline
│   └── prediction/               # Prediction scripts
│       └── predict_review_quality.py  # Quality prediction pipeline
├── models/                        # Trained models (created after training)
│   ├── bart_classifier/          # Fine-tuned BART model
│   ├── metadata_analyzer.pkl     # Metadata analyzer
│   └── fusion_model.pkl          # Fusion model
├── output/                        # Results and logs
│   ├── prediction_results.csv    # Prediction outputs
│   ├── training.log              # Training logs
│   └── prediction.log            # Prediction logs
└── docs/                          # Documentation
    """)

def main():
    """Main demo interface"""
    print("🎯 REVIEW QUALITY DETECTION SYSTEM - DEMO")
    print("=" * 80)
    print("Pure Machine Learning Approach - No Heuristics")
    print("=" * 80)
    
    show_project_structure()
    print()
    
    demo_training()
    demo_batch_prediction()
    demo_single_prediction()
    
    print("🎉 QUICK START GUIDE")
    print("=" * 50)
    print("1. Ensure training data exists: data/data_all_training.csv")
    print("2. Train models: python scripts/training/train_all_models.py")
    print("3. Predict quality: python scripts/prediction/predict_review_quality.py --input data/data_all_test.csv")
    print()
    print("✨ Features:")
    print("- 🤖 BART fine-tuned 7-class text classification")
    print("- 📊 ML-based metadata anomaly detection")
    print("- 🔮 Advanced gradient boosting fusion")
    print("- 📈 93.6% cross-validation accuracy")
    print("- 🚀 GPU-accelerated inference")
    print("- 💾 Persistent model storage")
    print("- 📋 Detailed logging and results")
    print()
    print("🔗 For more details, see: docs/")

if __name__ == "__main__":
    main()
