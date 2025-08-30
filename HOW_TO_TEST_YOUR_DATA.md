# 📝 How to Test Your Own Reviews

## 📂 Demo Folder

All testing files are now in the `demo/` folder:
- **`demo/demo_reviews.csv`** - Sample reviews ready for testing
- **`demo/README.md`** - Detailed instructions for testing

## 🚀 Quick Testing Guide

### Option 1: Test with Demo Data (Immediate)
```bash
python scripts/evaluation/universal_review_tester.py --input demo_reviews.csv --text_column text
```

### Option 2: Test with Your Own Data
1. **Prepare Your CSV File**:
   - Go to the `demo/` folder
   - Replace content in `demo_reviews.csv` with your own reviews, OR
   - Create a new CSV file (e.g., `my_reviews.csv`) using the same format

2. **Required CSV Format**:
   ```csv
   text,rating,business_name,category,reviewer_name,date,helpful_votes,total_votes,verified_purchase,location
   "Your review text here",5,Business Name,Category,Reviewer,2024-08-31,5,10,True,Singapore
   ```

3. **Run Analysis**:
   ```bash
   python scripts/evaluation/universal_review_tester.py --input demo/my_reviews.csv --text_column text
   ```

## 📋 CSV Format Requirements

### Required Columns
- **`text`** - Review content (must be present)
- **`rating`** - Star rating 1-5 (recommended)
- **`business_name`** - Business being reviewed (recommended)
- **`category`** - Business category (recommended)

### Optional Columns (Improve Detection Accuracy)
- **`reviewer_name`** - Reviewer username/name
- **`date`** - Review date (YYYY-MM-DD format)
- **`helpful_votes`** - Number of helpful votes
- **`total_votes`** - Total votes received
- **`verified_purchase`** - True/False for verified purchases
- **`location`** - Reviewer location

## 🎯 What Fields Help Detect Fakes?

### High Impact Fields
- **`reviewer_name`**: Generic names (like "User123") often indicate fake reviews
- **`verified_purchase`**: Unverified purchases are more likely to be fake
- **`helpful_votes` ratio**: Low helpful/total vote ratios suggest fake content

### Moderate Impact Fields
- **`date`**: Unusual posting patterns (clusters of reviews)
- **`location`**: Mismatched reviewer/business locations
- **`rating`**: Extreme ratings (1 or 5 stars) more often fake

## 📊 Analysis Output

The system provides:
- **Classification**: genuine_positive, genuine_negative, spam, inappropriate, advertisement, irrelevant
- **Confidence**: Prediction confidence (0.0-1.0)
- **Binary Result**: Genuine vs Non-Genuine
- **Stage Details**: Predictions from each model stage

## 🔧 Advanced Usage

### Custom Thresholds
Adjust classification sensitivity:
```bash
python scripts/evaluation/universal_review_tester.py --input demo/my_reviews.csv --text_column text --verbose
```

### Batch Processing
Process multiple files:
```bash
python scripts/evaluation/universal_review_tester.py --input demo/*.csv --text_column text
```

### Export Results
Save detailed results:
```bash
python scripts/evaluation/universal_review_tester.py --input demo/my_reviews.csv --text_column text --output results/
```

## 📈 Example Results

```
Review Analysis Results
======================
File: demo/my_reviews.csv
Total Reviews: 50

GENUINE Reviews: 35 (70.0%)
NON-GENUINE Reviews: 15 (30.0%)

Binary Classification Accuracy: 76.9%
ROC AUC Score: 77.4%

Classification Breakdown:
- genuine_positive: 20 reviews (40.0%)
- genuine_negative: 15 reviews (30.0%)
- spam: 8 reviews (16.0%)
- inappropriate: 4 reviews (8.0%)
- advertisement: 2 reviews (4.0%)
- irrelevant: 1 review (2.0%)

Top Predictions (with confidence):
1. "Amazing product!!!" → spam (98.5% confidence)
2. "Terrible service experience" → genuine_negative (94.2% confidence)
3. "Great food and atmosphere" → genuine_positive (91.8% confidence)
```

2. **Prepare Your CSV File**:
   - Replace the sample text with your actual reviews
   - Keep the same column structure
   - Add as many rows as you have reviews

3. **Run Analysis**:
```bash
python scripts/evaluation/universal_review_tester.py --input my_reviews.csv --text_column text --output my_results
```

## 📋 CSV Format Requirements

### Required Columns
- **`text`**: The review content (REQUIRED)

### Optional Columns (Help Improve Analysis)
- **`rating`**: Star rating (1-5)
- **`business_name`**: Name of business being reviewed
- **`category`**: Business category (Restaurant, Electronics, Hotel, etc.)
- **`reviewer_name`**: Name of reviewer
- **`date`**: Review date (YYYY-MM-DD format)
- **`helpful_votes`**: Number of helpful votes received
- **`total_votes`**: Total votes received
- **`verified_purchase`**: Whether purchase was verified (true/false)
- **`location`**: Reviewer location

### Example CSV Structure
```csv
text,rating,business_name,category,reviewer_name,date,helpful_votes,total_votes,verified_purchase,location
"Great food and excellent service! The pasta was perfectly cooked and the staff was very attentive.",5,Mario's Restaurant,Restaurant,John Smith,2024-01-15,10,12,true,New York
"Product arrived damaged. Customer service was unhelpful and took weeks for replacement.",1,TechStore,Electronics,Sarah Johnson,2024-02-20,5,8,true,California
"AMAZING PRODUCT!!! Changed my life completely! Buy now with 90% discount! Limited time only!",5,MegaFit,Health & Wellness,Amazing_User123,2024-03-10,2,15,false,Unknown
```

### Column Guidelines
- **Text**: Can be any length, put longer reviews in quotes
- **Rating**: Numbers 1-5, helps identify rating-text mismatches
- **Business Name**: Any text, helps with context analysis
- **Category**: Restaurant, Electronics, Hotel, etc. - helps with domain-specific patterns
- **Reviewer Name**: Helps identify fake reviewer patterns (e.g., "Amazing_User123")
- **Date**: YYYY-MM-DD format, helps identify review bombing or timing patterns
- **Helpful/Total Votes**: Numbers, low helpful ratios can indicate fake reviews
- **Verified Purchase**: true/false, unverified purchases are more suspicious
- **Location**: Helps identify geographic review patterns

### Fields That Improve Fake Detection
- **Reviewer Name**: Generic/promotional names (e.g., "BestProduct2024") are suspicious
- **Verified Purchase**: false values increase suspicion scores
- **Helpful Votes Ratio**: Low helpful/total ratios suggest poor quality
- **Date Patterns**: Many reviews on same date suggest review bombing
- **Location**: "Unknown" or missing locations are more suspicious

## 🎛️ Advanced Testing Options

### Different Text Columns
If your CSV has different column names:
```bash
# If your review text is in a column called "review_content"
python scripts/evaluation/universal_review_tester.py --input your_file.csv --text_column review_content --output results

# If your review text is in a column called "comment"
python scripts/evaluation/universal_review_tester.py --input your_file.csv --text_column comment --output results
```

### Large Dataset Testing
```bash
# For files with thousands of reviews
python scripts/evaluation/universal_review_tester.py --input large_dataset.csv --text_column text --output large_results
```

### Batch Testing Multiple Files
```bash
# Test multiple files
python scripts/evaluation/universal_review_tester.py --input file1.csv --text_column text --output results1
python scripts/evaluation/universal_review_tester.py --input file2.csv --text_column text --output results2
```

## 📊 Understanding Results

### Output Files Generated
- **`detailed_results_[timestamp].csv`**: Complete analysis for each review
- **`analysis_report_[timestamp].json`**: Summary statistics
- **`stage_scores_[timestamp].csv`**: Breakdown by analysis stage
- **`pipeline_analysis.png`**: Visual distribution chart

### Result Categories
- ✅ **Genuine**: High-quality, authentic reviews (70% typical)
- 🟡 **Low-Risk**: Likely genuine, minor concerns (20% typical)
- 🟠 **Medium-Risk**: Some suspicious patterns (10% typical)
- 🔴 **High-Risk**: Likely fake/spam (0% in demo)

## 🔧 Troubleshooting

### Common Issues

#### "File not found"
```bash
# Make sure you're in the right directory
cd "C:\Deep\ACTUAL PROJECT"
ls *.csv  # Should show your CSV files
```

#### "Column 'text' not found"
```bash
# Check your column names
python -c "import pandas as pd; df = pd.read_csv('your_file.csv'); print(df.columns.tolist())"

# Use the correct column name
python scripts/evaluation/universal_review_tester.py --input your_file.csv --text_column your_column_name --output results
```

#### "Empty results"
- Check that your text column contains actual review text
- Ensure text is not empty or just numbers
- Verify CSV format is correct

## 📁 File Organization

```
Your Project/
├── demo_reviews.csv          # Sample data (ready to test)
├── input_template.csv        # Template for your data
├── my_reviews.csv           # Your actual review data
├── results/                 # Output directory
│   ├── detailed_results_*.csv
│   ├── analysis_report_*.json
│   └── pipeline_analysis.png
└── scripts/
    └── evaluation/
        └── universal_review_tester.py
```

## 🎯 Testing Workflow

### 1. Quick Start (2 minutes)
```bash
# Test the system works
python scripts/evaluation/universal_review_tester.py --input demo_reviews.csv --text_column text --output test_run
```

### 2. Prepare Your Data (5 minutes)
- Copy `input_template.csv` to `my_data.csv`
- Replace sample text with your reviews
- Save the file

### 3. Run Your Analysis (2-10 minutes depending on size)
```bash
python scripts/evaluation/universal_review_tester.py --input my_data.csv --text_column text --output my_analysis
```

### 4. Review Results (5 minutes)
- Open `my_analysis/detailed_results_*.csv` to see individual review analysis
- Check `pipeline_analysis.png` for visual distribution
- Review `analysis_report_*.json` for summary statistics

## 💡 Pro Tips

### For Best Results
- Include rating and business context when possible
- Use actual review text (not summaries)
- Test with diverse review types (positive, negative, neutral)
- Include both genuine and suspicious reviews if available

### Performance Optimization
- Remove empty rows before testing
- Clean text of special characters if needed
- For very large files (>10,000 reviews), consider testing a sample first

### Data Privacy
- The system processes text locally
- No data is sent to external services
- Results are saved only to your local machine

---

**Need help?** Check the main README.md or LIMITATIONS.md for more detailed information about the system capabilities and constraints.
