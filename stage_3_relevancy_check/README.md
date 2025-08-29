# Stage 3: Relevancy Check

This folder contains scripts for checking review relevancy and performing secondary analysis.

## Files

- **`layer2.py`** - Secondary layer analysis for review relevancy checking

## Purpose

Stage 3 focuses on:
- **Relevancy Analysis** - Determining if reviews are relevant to the product/service
- **Context Validation** - Checking if review content matches the business context
- **Secondary Classification** - Additional layers of review analysis
- **Quality Scoring** - Assigning relevancy scores to reviews

## Usage

1. Run after Stage 1 BART classification
2. Use `layer2.py` for secondary relevancy analysis
3. Combine results with BART predictions for final classification

## Integration

This stage works in conjunction with:
- Stage 1 BART predictions
- Original review metadata
- Business/product context information
