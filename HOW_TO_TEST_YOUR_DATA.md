# 📝 How to Test Your Own Reviews

## 📂 Files Available

### 1. **Demo Data** (Ready to Use)
- **File**: `demo_reviews.csv`
- **Purpose**: 10 sample reviews showing different types (genuine, spam, fake, etc.)
- **Usage**: Test the system immediately without any setup

### 2. **Input Template** (For Your Data)
- **File**: `input_template.csv`
- **Purpose**: Template for testing your own review data
- **Usage**: Replace with your actual reviews

## 🚀 Quick Testing Guide

### Option 1: Test with Demo Data (Immediate)
```bash
python scripts/evaluation/universal_review_tester.py --input demo_reviews.csv --text_column text --output demo_results
```

### Option 2: Test with Your Own Data
1. **Prepare Your CSV File**:
   - Copy `input_template.csv` to `my_reviews.csv`
   - Replace the sample text with your actual reviews
   - Keep the same column structure

2. **Run Analysis**:
```bash
python scripts/evaluation/universal_review_tester.py --input my_reviews.csv --text_column text --output my_results
```

## 📋 CSV Format Requirements

### Required Columns
- **`text`**: The review content (REQUIRED)
- **`rating`**: Star rating (1-5) (OPTIONAL)
- **`business_name`**: Name of business being reviewed (OPTIONAL)
- **`category`**: Business category (OPTIONAL)

### Example CSV Structure
```csv
text,rating,business_name,category
"Great food and excellent service!",5,Mario's Restaurant,Restaurant
"Product arrived damaged and customer service was unhelpful",1,TechStore,Electronics
"Average experience, nothing special but not bad either",3,Coffee Shop,Food & Beverage
```

### Column Guidelines
- **Text**: Can be any length, put longer reviews in quotes
- **Rating**: Numbers 1-5, or leave blank if unknown
- **Business Name**: Any text, helps with context
- **Category**: Restaurant, Electronics, Hotel, etc.

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
