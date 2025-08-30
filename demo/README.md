# Demo Folder - Test Your Own Reviews

This folder contains everything you need to test the fake review detection system with your own data.

## Quick Start

1. **Demo File**: `demo_reviews.csv` contains sample reviews to test the system
2. **Your Data**: Replace the content of `demo_reviews.csv` with your own reviews, or create a new CSV file with the same format

## CSV Format Required

Your CSV file must have these columns (in any order):

### Required Columns
- `text` - The review content (required)
- `rating` - Star rating (1-5, optional but recommended)
- `business_name` - Name of business being reviewed (optional but recommended)
- `category` - Business category (optional but recommended)

### Optional Columns (improve detection accuracy)
- `reviewer_name` - Name/username of reviewer
- `date` - Review date (YYYY-MM-DD format)
- `helpful_votes` - Number of helpful votes received
- `total_votes` - Total votes received
- `verified_purchase` - True/False if purchase was verified
- `location` - Reviewer location

## How to Use

### Option 1: Use Demo Data
```bash
cd "C:\Deep\ACTUAL PROJECT"
python scripts/evaluation/universal_review_tester.py --input demo/demo_reviews.csv --text_column text
```

### Option 2: Test Your Own Data
1. **Prepare your CSV**: Replace content in `demo_reviews.csv` or create `my_reviews.csv`
2. **Keep the format**: Use the same column names as shown in demo_reviews.csv
3. **Run analysis**:
```bash
python scripts/evaluation/universal_review_tester.py --input demo/my_reviews.csv --text_column text
```

## Results

The system will analyze each review and provide:
- **Classification**: genuine_positive, genuine_negative, spam, inappropriate, advertisement, irrelevant
- **Confidence**: Prediction confidence (0.0-1.0)
- **Binary Classification**: Genuine vs Non-Genuine
- **Detailed Analysis**: Stage-by-stage predictions

## Tips for Better Results

- **More fields = better accuracy**: Include reviewer_name, verified_purchase, and vote counts when available
- **Realistic data**: The system was finetuned with Singapore reviews, so local context helps
- **English reviews**: System works best with English text
- **Complete information**: More metadata improves fake detection capabilities

## Troubleshooting

- **File not found**: Make sure your CSV is in the `demo/` folder
- **Column errors**: Check that required columns (text, rating, business_name, category) exist
- **Encoding issues**: Save CSV files with UTF-8 encoding
- **Large files**: For files with 1000+ reviews, the analysis may take several minutes

## Example Analysis Output

```
Review Analysis Results
======================
File: demo/demo_reviews.csv
Total Reviews: 10

GENUINE Reviews: 7 (70.0%)
NON-GENUINE Reviews: 3 (30.0%)

Classification Breakdown:
- genuine_positive: 4 reviews
- genuine_negative: 3 reviews  
- spam: 2 reviews
- inappropriate: 1 review
```
