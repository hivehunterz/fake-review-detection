# Stage 1: BART Fine-tuning

This folder contains all scripts related to fine-tuning BART models for fake review detection.

## Files

- **`enhanced_bart_finetune.py`** - Main enhanced BART fine-tuning script with advanced metrics
- **`evaluate_fine_tuned_bart.py`** - Comprehensive evaluation of fine-tuned models
- **`working_bart_finetune.py`** - Working version of BART fine-tuning pipeline

## Results

The enhanced BART model achieves:
- **83% accuracy** on fake review detection
- **69.5% macro F1-score**
- **81.1% weighted F1-score**
- 23% improvement over zero-shot BART baseline

## Usage

1. Ensure labeled data is available in the `data/` folder
2. Run `enhanced_bart_finetune.py` to train the model
3. Use `evaluate_fine_tuned_bart.py` to evaluate performance

## Model Categories

- `genuine_positive` - Authentic positive reviews
- `genuine_negative` - Authentic negative reviews  
- `spam` - Spam/promotional content
- `advertisement` - Advertisement reviews
- `irrelevant` - Off-topic content
- `fake_rant` - Fabricated negative reviews
- `inappropriate` - Inappropriate content
