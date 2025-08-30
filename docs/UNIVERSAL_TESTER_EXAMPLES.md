# 🔬 Universal Review Tester - Quick Start Examples

## Example Usage Commands

### 1. Basic Analysis
```bash
# Navigate to project directory
cd "c:\Deep\ACTUAL PROJECT"

# Analyze reviews in a CSV file
python scripts/evaluation/universal_review_tester.py --input your_reviews.csv --text_column review_text

# Output will be saved to 'universal_test_results/' directory
```

### 2. Custom Column Name and Output Directory
```bash
# If your text column has a different name
python scripts/evaluation/universal_review_tester.py --input data.csv --text_column content --output my_analysis

# This works with any column name: content, comment, review, text, etc.
```

### 3. Verbose Mode for Detailed Logging
```bash
# Get detailed progress information
python scripts/evaluation/universal_review_tester.py --input reviews.csv --text_column review_text --verbose
```

### 4. Demo with Sample Data
```bash
# Run demo to see how it works
python scripts/evaluation/demo_universal_tester.py

# Creates sample data and shows complete analysis pipeline
```

## CSV File Examples

### Simple Format
```csv
review_text,rating
"Great product, highly recommend!",5
"Terrible quality, don't buy",1
"Average product, okay for the price",3
```

### Rich Format (Better Analysis)
```csv
review_text,rating,date,reviewer_name,helpful_count
"Excellent quality and fast shipping!",5,2025-08-15,John D.,10
"Buy now! Best deals ever! Click here!",5,2025-08-30,PromoBot,0
"Good product but expensive",3,2025-08-20,Sarah M.,5
"Completely broken on arrival",1,2025-08-25,Mike R.,8
```

## Output Files You'll Get

### 1. Detailed Results CSV
File: `detailed_results_YYYYMMDD_HHMMSS.csv`

Contains all your original data plus analysis columns:
- `fusion_risk_score`: Overall risk (0-1, higher = more suspicious)
- `fusion_prediction`: Category (genuine, low_risk, medium_risk, high_risk)
- `fusion_confidence`: How confident the system is
- `metadata_anomaly_score`: Metadata-based anomaly detection
- Plus BART scores if models are available

### 2. Analysis Report JSON
File: `analysis_report_YYYYMMDD_HHMMSS.json`

Summary of findings:
- Total reviews analyzed
- Risk distribution percentages
- Stage-by-stage performance
- System recommendations

### 3. Visualizations
File: `pipeline_analysis.png`

Charts showing:
- Risk score distributions
- Classification breakdowns
- Stage comparisons

## Real-World Examples

### E-commerce Product Reviews
```bash
python scripts/evaluation/universal_review_tester.py \
  --input product_reviews_2025.csv \
  --text_column review_content \
  --output ecommerce_analysis
```

### App Store Reviews
```bash
python scripts/evaluation/universal_review_tester.py \
  --input app_reviews.csv \
  --text_column review_text \
  --output app_analysis \
  --verbose
```

### Social Media Comments
```bash
python scripts/evaluation/universal_review_tester.py \
  --input social_comments.csv \
  --text_column comment_text \
  --output social_analysis
```

## Interpreting Results

### Risk Scores (0-1 scale)
- **0.0-0.25**: Very likely genuine
- **0.25-0.50**: Probably genuine
- **0.50-0.75**: Suspicious, needs review
- **0.75-1.0**: Very likely fake/spam

### Confidence Levels
- **High (>0.8)**: Very reliable prediction
- **Medium (0.5-0.8)**: Reasonably reliable
- **Low (<0.5)**: Uncertain, manual review recommended

### Categories
- **genuine**: High-quality legitimate review
- **low_risk**: Minor concerns but likely genuine
- **medium_risk**: Significant concerns, manual review needed
- **high_risk**: Very likely fake, spam, or low-quality

## Troubleshooting

### Common Error: "Text column not found"
```bash
# Check your CSV column names first
python -c "import pandas as pd; print(pd.read_csv('your_file.csv').columns.tolist())"

# Then use the correct column name
python scripts/evaluation/universal_review_tester.py --input your_file.csv --text_column actual_column_name
```

### Large Files
For files with thousands of reviews, the analysis may take several minutes. Use `--verbose` to see progress.

### Missing Models
The script works even without all models installed - it uses fallback methods to provide meaningful analysis.

## Performance Expectations

- **Speed**: 1-5 reviews per second (depending on available models)
- **Accuracy**: 76.9% binary classification accuracy when full models are loaded
- **Memory**: Requires ~2-4GB RAM for large datasets
- **Output Size**: Results files are typically 2-3x the size of input CSV

## Integration Tips

### Batch Processing
```python
import glob
from scripts.evaluation.universal_review_tester import UniversalReviewTester

tester = UniversalReviewTester()
for file in glob.glob("data/*.csv"):
    tester.run_complete_analysis(file, "review_text", f"results/{file}")
```

### API Integration
```python
from scripts.evaluation.universal_review_tester import UniversalReviewTester
import pandas as pd

def analyze_reviews(reviews_list):
    df = pd.DataFrame({'review_text': reviews_list})
    df.to_csv('temp_reviews.csv', index=False)
    
    tester = UniversalReviewTester('api_results')
    report = tester.run_complete_analysis('temp_reviews.csv', 'review_text')
    
    return report
```
