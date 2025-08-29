# 🛡️ Advanced Fake Review Detection System

A comprehensive machine learning system for detecting fake reviews, spam, and inappropriate content in Google Reviews using state-of-the-art NLP models.

## 🎯 Project Overview

This project implements a sophisticated fake review detection system that combines:
- **Multi-LLM labeling** with Groq Llama 3.3-70B for high-quality dataset creation
- **Fine-tuned BART models** achieving 83% accuracy
- **Comprehensive evaluation** against zero-shot baselines
- **Production-ready classification** for 7 review categories

## 📊 Performance Results

| Model | Accuracy | F1-Macro | F1-Weighted |
|-------|----------|----------|-------------|
| **Fine-tuned BART** | **83.0%** | **69.5%** | **81.1%** |
| Zero-shot BART | 60.0% | 28.7% | 50.9% |
| **Improvement** | **+23.0%** | **+40.8%** | **+30.2%** |

## 🏗️ System Architecture

```
Multi-Stage Detection Pipeline:

Stage 0: Data Preprocessing
├── Raw Google Reviews (JSON)
├── Multi-LLM Labeling (Groq Llama 3.3-70B)
└── Quality-Controlled Dataset

Stage 1: BART Fine-tuning
├── Enhanced Dataset → BART Model Training
├── 7-Category Classification
└── 83% Accuracy Achievement

Stage 2: Metadata Anomaly Detection (Planned)
├── Temporal Pattern Analysis
├── User Behavior Analysis
└── Geographic Anomaly Detection

Stage 3: Relevancy Check
├── Secondary Layer Analysis (layer2.py)
├── Context Validation
└── Final Quality Scoring

Classification Categories:
├── genuine_positive    (45.5%)
├── genuine_negative    (17.0%)
├── spam               (14.3%)
├── advertisement      (4.3%)
├── irrelevant         (10.6%)
├── fake_rant          (4.8%)
└── inappropriate      (3.6%)
```

## 🚀 Quick Start

### Prerequisites
```bash
conda create -n review_classifier python=3.12
conda activate review_classifier
pip install torch transformers pandas scikit-learn groq openai accelerate
```

### 1. Data Labeling (Stage 0)
```bash
# Label new reviews using multi-LLM system
cd data_preprocessing
python google_reviews_labeler.py
```

### 2. Model Training (Stage 1)
```bash
# Fine-tune BART model on labeled data
cd stage_1_bart_finetuning
python enhanced_bart_finetune.py
```

### 3. Model Evaluation (Stage 1)
```bash
# Evaluate model performance
cd stage_1_bart_finetuning
python evaluate_fine_tuned_bart.py
```

### 4. Relevancy Check (Stage 3)
```bash
# Secondary relevancy analysis
cd stage_3_relevancy_check
python layer2.py
```

## 📁 Project Structure

```
├── data_preprocessing/                  # Data labeling and preprocessing
│   ├── google_reviews_labeler.py      #   Multi-LLM labeling system
│   ├── google_reviews_labeler_fixed.py #   Fixed version with model updates
│   ├── google_reviews_evaluator.py    #   Data quality evaluation
│   └── README.md                      #   Stage documentation
├── stage_1_bart_finetuning/            # BART model training
│   ├── enhanced_bart_finetune.py      #   Enhanced BART fine-tuning
│   ├── evaluate_fine_tuned_bart.py    #   Model evaluation and comparison
│   ├── working_bart_finetune.py       #   Working fine-tuning pipeline
│   └── README.md                      #   Stage documentation
├── stage_2_metadata_anomaly/           # Metadata anomaly detection
│   └── README.md                      #   Stage documentation (TBD)
├── stage_3_relevancy_check/            # Review relevancy analysis
│   ├── layer2.py                      #   Secondary relevancy analysis
│   └── README.md                      #   Stage documentation
├── data/                              # Raw and processed datasets
│   ├── dataset (1).json              #   Raw Google Reviews data
│   └── google_reviews_labeled_*.csv  #   Labeled datasets
├── models/                            # Trained model artifacts
├── results/                           # Evaluation outputs
├── docs/                              # Documentation
│   ├── BART_FINETUNING_GUIDE.md     #   Detailed training guide
│   └── FINETUNING_GUIDE.md          #   General fine-tuning guide
└── example_usage.py                   # Usage examples
```

## 🔧 Key Features

### Multi-LLM Labeling System
- **Groq Llama 3.3-70B** prioritized for speed and logic
- **Automatic model switching** on rate limits
- **Progress saving** with resume capability
- **Batch processing** for efficiency

### Enhanced BART Fine-tuning
- **facebook/bart-large-mnli** base model
- **7-class classification** for comprehensive review analysis
- **Optimized training** with learning rate scheduling
- **Robust evaluation** with detailed metrics

### Production Features
- **GPU acceleration** support
- **Batch prediction** capabilities
- **Comprehensive logging** and monitoring
- **Error handling** and recovery

## 📈 Technical Details

### Model Architecture
- **Base Model**: facebook/bart-large-mnli
- **Fine-tuning**: Sequence classification head
- **Input Length**: 512 tokens
- **Training**: 3 epochs, 1e-5 learning rate

### Dataset Statistics
- **Total Reviews**: 3,671
- **Training Split**: 80%
- **Validation Split**: 20%
- **Label Quality**: 83% LLM agreement

## 🎯 Use Cases

1. **E-commerce Platforms**: Automatic fake review detection
2. **Content Moderation**: Spam and inappropriate content filtering
3. **Market Research**: Genuine sentiment analysis
4. **Quality Assurance**: Review authenticity verification

## 🛠️ Advanced Usage

### Custom Model Training
```python
from enhanced_bart_finetune import EnhancedBARTFineTuner

# Initialize fine-tuner
fine_tuner = EnhancedBARTFineTuner()

# Custom training
model_path = fine_tuner.fine_tune(
    csv_file="your_labeled_data.csv",
    epochs=3,
    batch_size=8,
    learning_rate=1e-5
)
```

### Batch Prediction
```python
from evaluate_fine_tuned_bart import FineTunedBARTEvaluator

# Load model
evaluator = FineTunedBARTEvaluator("path/to/model")

# Predict reviews
reviews = ["Great product!", "Spam content here"]
predictions = evaluator.predict_reviews(reviews)
```

## 📊 Evaluation Metrics

### Class-wise Performance
- **Spam Detection**: 93.8% F1
- **Advertisement**: 90.7% F1
- **Inappropriate**: 68.8% F1
- **Genuine Reviews**: 81.9% F1 (combined)

### Model Comparison
- **83% accuracy** vs 60% zero-shot
- **Perfect precision** on multiple classes
- **Robust recall** across all categories

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Groq** for high-speed LLM inference
- **Hugging Face** for transformer models
- **Facebook AI** for BART architecture

## 📧 Contact

For questions, issues, or collaboration opportunities, please open an issue or contact the maintainer.

---

⭐ **Star this repository if you find it useful!**
