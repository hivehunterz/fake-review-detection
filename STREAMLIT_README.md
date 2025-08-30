# 🛡️ Fake Review Detection System - Streamlit Web Interface

## Quick Start for Hackathon Demo

### 1. Install Dependencies
```bash
pip install streamlit plotly pandas numpy
```

### 2. Run the Web Interface
```bash
streamlit run streamlit_app.py
```

The interface will open at `http://localhost:8501`

## Features Overview

### 📊 Main Dashboard
- Real-time statistics and metrics
- Quick navigation buttons
- Activity trends visualization
- **No policy violation breakdown** (removed as requested)

### 🔍 Single Review Analysis
- **Business category dropdown** with 9 predefined categories
- Comprehensive review form with optional metadata
- Real-time ML analysis with confidence scores
- Risk assessment and key indicators

### 📝 Batch Analysis  
- CSV file upload with drag-and-drop
- **Multi-category filters dropdown** (no confidence threshold)
- Sample data loading for demonstration
- Results visualization and export

### ⚠️ Violation List
- **Combined filters and recent violations** in single section
- Advanced filtering: Date, Severity, Type, Status
- Real-time violation metrics
- Action buttons: Export, Escalate, Resolve

## Backend Integration

The interface supports three modes:

1. **Full ML Backend** - When trained models are available
2. **BART Fallback** - When only BART classifier is available  
3. **Enhanced Demo Mode** - Sophisticated heuristics for testing

The system gracefully degrades and provides clear status indicators.

## Demo Mode Features

- Realistic review classification using enhanced heuristics
- Proper confidence scoring and risk assessment
- Sample violation data generation
- Professional visualization and metrics

## Hackathon Ready ✅

- All requested features implemented
- Professional dark theme UI
- Comprehensive error handling
- Immediate functionality without trained models
- Export and action capabilities
- Mobile-responsive design

## Usage

1. **Single Review**: Enter review text, select category, get instant analysis
2. **Batch Analysis**: Upload CSV or load sample data, filter results, export
3. **Violations**: View filtered violations, take actions, export reports

Perfect for demonstrating fake review detection capabilities!