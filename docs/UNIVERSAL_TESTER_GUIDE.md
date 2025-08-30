# 🔬 Universal Review Tester - Usage Guide

## Overview
The Universal Review Tester is a comprehensive script that can analyze any CSV file containing reviews using the complete fake review detection pipeline. It provides detailed scores for each stage of the process.

## Features

### 🎯 Multi-Stage Analysis
- **Stage 1**: BART text classification (genuine vs non-genuine)
- **Stage 2**: Metadata anomaly detection
- **Stage 3**: Fusion analysis combining all stages

### 📊 Comprehensive Scoring
- Individual stage scores and confidence levels
- Combined risk assessment
- Detailed performance metrics
- Visual analysis charts

### 🔧 Flexible Input
- Works with any CSV file format
- Configurable text column name
- Optional metadata columns
- Handles missing data gracefully

## Quick Start

### 1. Basic Usage
```bash
cd "c:\Deep\ACTUAL PROJECT"
python scripts/evaluation/universal_review_tester.py --input your_file.csv --text_column review_text
```

### 2. Custom Output Directory
```bash
python scripts/evaluation/universal_review_tester.py --input data.csv --text_column content --output my_results
```

### 3. Demo with Sample Data
```bash
python scripts/evaluation/demo_universal_tester.py
```

## CSV File Requirements

### Required Columns
- **Text Column**: Contains the review text content
  - Common names: `review_text`, `content`, `text`, `comment`
  - Must be specified with `--text_column` parameter

### Optional Columns (Enhances Analysis)
- **Rating**: `rating`, `stars`, `score` (1-5 scale)
- **Date**: `date`, `timestamp`, `created_at`
- **Metadata**: `helpful_count`, `reviewer_name`, etc.

### Example CSV Structure
```csv
review_text,rating,date,reviewer_name
"Great product, highly recommend!",5,2025-08-15,John D.
"Terrible quality, waste of money",1,2025-08-16,Sarah M.
"Buy now! Best deals here!",5,2025-08-17,PromoBot123
```

## Output Files

### 📋 Detailed Results CSV
- `detailed_results_YYYYMMDD_HHMMSS.csv`
- Contains all original data plus analysis results:
  - `bart_classification`: 7-class BART prediction
  - `bart_binary`: Genuine vs non-genuine
  - `bart_confidence`: Confidence score (0-1)
  - `bart_quality_risk`: Risk score (0-1)
  - `metadata_anomaly_score`: Metadata anomaly score
  - `fusion_risk_score`: Final combined risk score
  - `fusion_prediction`: Final category prediction

### 📊 Analysis Report JSON
- `analysis_report_YYYYMMDD_HHMMSS.json`
- Comprehensive analysis summary:
  - Stage-by-stage performance metrics
  - Overall analysis statistics
  - System recommendations

### 📈 Stage Scores CSV
- `stage_scores_YYYYMMDD_HHMMSS.csv`
- Performance metrics for each pipeline stage

### 📈 Visualizations
- `pipeline_analysis.png`
- Charts showing:
  - Classification distributions
  - Risk score histograms
  - Stage comparisons
  - Final predictions

## Command Line Options

```bash
python universal_review_tester.py [OPTIONS]

Required:
  --input, -i          Input CSV file path

Optional:
  --text_column, -t    Name of text column (default: 'review_text')
  --output, -o         Output directory (default: 'universal_test_results')
  --verbose, -v        Enable verbose logging
  --help, -h           Show help message
```

## Scoring System

### Risk Score Interpretation
- **0.0 - 0.25**: Low risk (likely genuine)
- **0.25 - 0.50**: Medium-low risk 
- **0.50 - 0.75**: Medium-high risk
- **0.75 - 1.0**: High risk (likely fake/spam)

### Confidence Levels
- **> 0.8**: High confidence
- **0.6 - 0.8**: Medium confidence
- **< 0.6**: Low confidence

### Final Categories
- **genuine**: High-quality legitimate review
- **low_risk**: Likely genuine with minor concerns
- **medium_risk**: Requires manual review
- **high_risk**: Likely fake/spam/low-quality

## Advanced Usage

### 1. Programmatic Usage
```python
from scripts.evaluation.universal_review_tester import UniversalReviewTester

# Initialize tester
tester = UniversalReviewTester("my_output_dir")

# Run analysis
report = tester.run_complete_analysis(
    input_file="my_data.csv",
    text_column="review_content",
    output_dir="results"
)

# Access results
print(f"Average risk: {report['overall_analysis']['avg_risk_score']}")
```

### 2. Batch Processing Multiple Files
```python
import glob
from pathlib import Path

tester = UniversalReviewTester()
csv_files = glob.glob("data/*.csv")

for csv_file in csv_files:
    output_dir = f"results/{Path(csv_file).stem}"
    report = tester.run_complete_analysis(csv_file, "review_text", output_dir)
    print(f"Processed {csv_file}: {report['test_info']['total_reviews']} reviews")
```

## Model Requirements

### Available Models (Auto-detected)
- **BART Classifier**: `models/bart_classifier/` (fine-tuned) or zero-shot fallback
- **Metadata Analyzer**: Enhanced anomaly detection
- **Fusion Model**: `models/fusion_model.pkl` or weighted scoring fallback

### Graceful Degradation
- Script works even if some models are missing
- Falls back to simplified analysis methods
- Always provides meaningful results

## Example Output Summary

```
🎯 UNIVERSAL REVIEW TESTING SUMMARY
============================================================
📁 Input File: sample_reviews.csv
📊 Total Reviews: 100
⚡ Average Risk Score: 0.347
🔴 High Risk: 15.0%
🟢 Low Risk: 45.0%
📂 Results saved to: universal_test_results/
============================================================
```

## Troubleshooting

### Common Issues

1. **"Text column not found"**
   - Check column names in your CSV
   - Use `--text_column` to specify correct column name

2. **"BART classifier not available"**
   - Models may not be loaded, but script will use fallback methods
   - Check if `models/bart_classifier/` exists

3. **Memory issues with large files**
   - Process files in chunks
   - Use smaller batch sizes

### Getting Help
- Use `--verbose` flag for detailed logging
- Check the demo script for working examples
- Refer to error messages for specific guidance

## Performance Metrics

The system has been validated with:
- **Accuracy**: 76.9% on binary classification
- **ROC AUC**: 77.4%
- **PR AUC**: 67.7%
- **Processing Speed**: ~1-5 reviews/second depending on models loaded

## Integration Examples

### 1. E-commerce Platform
```bash
# Analyze product reviews
python universal_review_tester.py \
  --input product_reviews_2025.csv \
  --text_column review_content \
  --output ecommerce_analysis
```

### 2. App Store Reviews
```bash
# Analyze app reviews
python universal_review_tester.py \
  --input app_store_reviews.csv \
  --text_column review_text \
  --output app_analysis
```

### 3. Research Dataset
```bash
# Analyze research dataset
python universal_review_tester.py \
  --input research_reviews.csv \
  --text_column text \
  --output research_results \
  --verbose
```
