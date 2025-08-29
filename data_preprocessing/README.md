# Data Preprocessing

This folder contains scripts for labeling and preprocessing review data using multiple LLM models.

## Files

- **`google_reviews_labeler.py`** - Main labeling script using Groq Llama 3.3-70B and other LLM models
- **`google_reviews_labeler_fixed.py`** - Fixed version with deprecated model handling  
- **`google_reviews_evaluator.py`** - Script for evaluating labeled data quality

## Usage

1. Configure your API keys in `config.ini`
2. Run `google_reviews_labeler.py` to label raw review data
3. Use `google_reviews_evaluator.py` to validate labeling quality

## Features

- Multi-LLM support with automatic model switching on rate limits
- Batch processing for efficiency
- Progress saving and resumption
- Quality evaluation metrics
