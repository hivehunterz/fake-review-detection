# 🛡️ Fake Review Detection System - Clean Version

**Complete 3-Stage Fake Review Detection Pipeline**

## 📁 Project Structure

### **Stage 1: BART Fine-tuning** (`stage_1_bart_finetuning/`)
- ✅ **Trained Model**: `enhanced_bart_review_classifier_20250830_173055/`
- ✅ **Evaluation Results**: `evaluation_results_20250830_175519/`
- ✅ **Training Code**: `train_both_models.py`
- ✅ **Evaluation Code**: `comprehensive_model_evaluation.py`

**Performance**: 83.7% AUC, 76.9% Accuracy

### **Stage 2: Metadata Ensemble** (`stage_2_metadata_anomaly/`)
- ✅ **Training Script**: `FINAL_train_ensemble.py`
- ✅ **Evaluation Script**: `FINAL_evaluate_ensemble.py`
- ✅ **Documentation**: `FINAL_README.md`

**Performance**: 87.6% AUC, 82.0% Accuracy

### **Stage 3: Relevancy Check** (`stage_3_relevancy_check/`)
- ✅ **Working Code**: `fixed_enhanced_relevance.py`
- ✅ **Analysis Results**: `enhanced_relevancy_analysis_results.csv`
- ✅ **Documentation**: `README.md`

**Results**: 89.2% Irrelevant filtered, Enhanced TF-IDF + Fuzzy matching

## 🚀 Quick Start

### Stage 1: Text Classification
```bash
cd stage_1_bart_finetuning
python comprehensive_model_evaluation.py  # Use existing model
```

### Stage 2: Metadata Analysis
```bash
cd stage_2_metadata_anomaly
python FINAL_train_ensemble.py     # Train model
python FINAL_evaluate_ensemble.py  # Make predictions
```

### Stage 3: Relevancy Filtering
```bash
cd stage_3_relevancy_check
python fixed_enhanced_relevance.py  # Run relevancy analysis
```

## 📊 Combined Pipeline Performance

| Stage | Function | Performance | Status |
|-------|----------|-------------|---------|
| **1** | Text Analysis | 83.7% AUC | ✅ Complete |
| **2** | Metadata Patterns | 87.6% AUC | ✅ Complete |  
| **3** | Relevancy Filter | 89.2% Filtered | ✅ Complete |

## 💡 Usage

Each stage can be used independently or combined for comprehensive fake review detection:

1. **Stage 1**: Analyzes review text content using fine-tuned BART
2. **Stage 2**: Analyzes metadata patterns using ensemble ML
3. **Stage 3**: Filters reviews by business relevancy

## 📁 Data

- **Training Data**: `data/data_all_training.csv` (967 reviews)
- **Original Models**: Saved in respective stage folders
- **Results**: Evaluation metrics and analysis files included

All unnecessary files have been removed. Only essential working code and trained models remain.
