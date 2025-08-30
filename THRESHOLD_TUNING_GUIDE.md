# 🎯 Classification Threshold Configuration

## Current Problem
Your distribution shows too many "Suspicious" reviews (47%), which require manual verification.
This creates a bottleneck and reduces automation efficiency.

## Target Distribution (Recommended)
- ✅ Genuine: 40-50% (increase from 30%)
- 🟡 Suspicious: 20-25% (decrease from 47%)
- ⚠️ Low-Quality: 10-15% (increase from 5%)
- 🚫 High-Confidence-Spam: 20-25% (increase from 18%)

## Threshold Adjustments

### 1. Fusion Model Thresholds (core/fusion/fusion_model.py)
```python
# In predict_fusion_with_thresholds() and batch_predict_with_thresholds()

# CURRENT RECOMMENDED VALUES:
genuine_threshold = 0.20    # Lower = more genuine (was 0.25)
spam_threshold = 0.30       # Lower = more spam detection (was 0.35)
low_quality_threshold = 0.18 # Higher = more low-quality (was 0.15)

# AGGRESSIVE VALUES (to maximize automation):
genuine_threshold = 0.15    # Very liberal genuine classification
spam_threshold = 0.25       # More aggressive spam detection
low_quality_threshold = 0.20 # More low-quality instead of suspicious
```

### 2. Universal Tester Thresholds (scripts/evaluation/universal_review_tester.py)
```python
# In _score_to_category()

# CURRENT RECOMMENDED VALUES:
if score < 0.35:     # More genuine (was 0.25)
    return 'genuine'
elif score < 0.55:   # More low_risk (was 0.5)
    return 'low_risk'
elif score < 0.8:    # Less medium_risk/suspicious (was 0.75)
    return 'medium_risk'
else:
    return 'high_risk'

# AGGRESSIVE VALUES:
if score < 0.45:     # Very liberal genuine
    return 'genuine'
elif score < 0.65:   # More low_risk
    return 'low_risk'
elif score < 0.85:   # Minimal suspicious
    return 'medium_risk'
else:
    return 'high_risk'
```

## Implementation Steps

### Quick Test (5 minutes):
1. Update thresholds in universal_review_tester.py (already done)
2. Run: `python scripts/evaluation/universal_review_tester.py --input demo_reviews.csv --text_column text --output test_results`
3. Check new distribution in results

### Full Implementation (15 minutes):
1. Update fusion model with threshold functions (already done)
2. Update prediction pipeline to use new thresholds
3. Test on full dataset
4. Fine-tune thresholds based on results

### Monitoring:
1. Track false positive/negative rates
2. Monitor manual verification workload
3. Adjust thresholds based on business needs

## Expected Results

### After Threshold Adjustment:
- ✅ Genuine: 35-45% (↑ from 30%)
- 🟡 Suspicious: 25-30% (↓ from 47%)
- ⚠️ Low-Quality: 8-12% (↑ from 5%)
- 🚫 High-Confidence-Spam: 20-25% (↑ from 18%)

### Benefits:
- 📈 40% reduction in manual verification
- 📈 Increased automation (65-70% vs current 48%)
- 📈 Better user experience (more approvals)
- 📊 Maintained spam detection accuracy

## Advanced Strategies

### 1. Data Rebalancing
```python
# During training, oversample genuine and spam classes
# Undersample suspicious class
from imblearn.over_sampling import SMOTE
```

### 2. Cost-Sensitive Learning
```python
# Add class weights to favor genuine and spam over suspicious
class_weights = {
    'genuine': 1.2,
    'suspicious': 0.8,
    'low-quality': 1.1,
    'high-confidence-spam': 1.2
}
```

### 3. Ensemble Voting
```python
# Combine multiple models with voting
# If 2/3 models say genuine → genuine
# If 2/3 models say spam → spam
# Only suspicious if models disagree
```

## Testing Commands

```bash
# Test current settings
python scripts/evaluation/universal_review_tester.py --input demo_reviews.csv --text_column text --output current_test

# Test with data
python scripts/evaluation/universal_review_tester.py --input data/data_all_test.csv --text_column text --output full_test

# Compare results
python -c "
import pandas as pd
current = pd.read_csv('current_test/detailed_results_*.csv')
print('Distribution:')
print(current['fusion_prediction'].value_counts(normalize=True))
"
```
