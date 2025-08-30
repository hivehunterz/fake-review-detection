# 🛡️ Fake Review Detection System - Web Interface

## Overview

This Streamlit-based web interface provides a comprehensive dashboard for the fake review detection system with advanced filtering, analysis capabilities, and dark mode support.

## Features

### 🎨 UI/UX Improvements
- **Dark Mode**: Fully implemented dark theme with proper color scheme for all text elements including small text
- **Responsive Design**: Mobile-friendly layout with proper spacing and typography
- **Navigation**: Easy-to-use sidebar navigation with quick links

### 📊 Main Dashboard
- **Overview Metrics**: Real-time statistics for total reviews, genuine reviews, suspicious reviews, and spam/fake reviews
- **Quick Actions**: Direct access buttons to key functionality
- **Activity Charts**: Visual trends and patterns (policy violation breakdown chart removed as requested)
- **Navigation Links**: Easy access to all sections

### 🔍 Single Review Analysis
- **Business Category Dropdown**: Fixed options with top 8 most common categories plus "Other"
  - Restaurant, Hotel, Retail Store, Service Business, Healthcare, Entertainment, Automotive, Technology, Other
- **Comprehensive Form**: Business name, review text, rating, date, and reviewer information
- **Real-time Analysis**: ML-powered classification with confidence scores and risk assessment
- **Detailed Results**: Classification, confidence percentage, risk score, and key indicators

### 📝 Batch Analysis
- **CSV Upload**: Support for drag-and-drop file upload with validation
- **Sample Data**: Pre-loaded examples for testing
- **Multi-Category Filters**: Checkbox filters for all review categories (confidence threshold removed as requested)
  - Genuine Positive, Genuine Negative, Spam, Low Quality, Suspicious, Advertisement, Fake Positive, Irrelevant
- **Bulk Processing**: Analyze multiple reviews simultaneously
- **Export Functionality**: Download results as CSV with timestamp

### ⚠️ Violation List
- **Combined Interface**: Filters and recent violations in unified "Violation List" section (as requested)
- **Advanced Filtering**: 
  - Date range selection
  - Severity levels (High, Medium, Low)
  - Violation types (Spam Detection, Fake Reviews, Policy Violations, Content Violations, User Behavior Issues)
  - Status filtering (Open, In Review, Resolved, Escalated)
- **Violation Management**: Export reports, escalate issues, mark as resolved
- **Real-time Metrics**: Total violations, high severity count, pending review count

## Installation

1. **Install Dependencies**:
   ```bash
   pip install streamlit plotly pandas numpy
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Access the Interface**:
   - Local: http://localhost:8501
   - Network: http://your-ip:8501

## Configuration

### Dark Mode
The interface defaults to dark mode with the following color scheme:
- Background: #0E1117
- Secondary Background: #262730
- Text Primary: #FAFAFA
- Text Secondary: #CCCCCC
- Text Muted: #999999
- Accent: #FF6B6B
- Success: #4ECDC4
- Warning: #FFE66D

### Business Categories
The system includes 8 primary business categories plus "Other":
```python
BUSINESS_CATEGORIES = [
    "Restaurant", "Hotel", "Retail Store", "Service Business",
    "Healthcare", "Entertainment", "Automotive", "Technology", "Other"
]
```

### Review Categories
All review classification types are supported:
```python
REVIEW_CATEGORIES = [
    "genuine_positive", "genuine_negative", "spam", "low_quality",
    "suspicious", "advertisement", "fake_positive", "irrelevant"
]
```

## Integration with ML Backend

The web interface connects to the existing ML pipeline:
- **Stage 1**: BART text classification
- **Stage 2**: Metadata anomaly detection (confidence thresholds removed from batch analysis)
- **Stage 3**: Relevancy analysis
- **Fusion Model**: Combined predictions with risk scoring

## Usage Examples

### Single Review Analysis
1. Select business category from dropdown
2. Enter business name (optional)
3. Paste review text
4. Add optional metadata (rating, date, reviewer)
5. Click "Analyze Review"
6. View classification results and risk assessment

### Batch Analysis
1. Upload CSV file or load sample data
2. Select text column for analysis
3. Choose category filters (no confidence threshold)
4. Start batch processing
5. Review results with visualizations
6. Export processed data

### Violation Management
1. Set date range and filter criteria
2. View violation summary metrics
3. Browse violation table with details
4. Take actions (export, escalate, resolve)
5. Monitor status changes

## Security & Performance

- **Input Validation**: All user inputs are validated and sanitized
- **File Upload Limits**: 200MB maximum file size for CSV uploads
- **Session Management**: Streamlit handles session state automatically
- **Error Handling**: Comprehensive error messages and fallback options

## Development

### File Structure
```
app.py                 # Main Streamlit application
.streamlit/
  config.toml         # Streamlit configuration
requirements.txt      # Updated dependencies
core/
  stage2_metadata/
    config.py         # Updated configuration (confidence threshold removed)
```

### Adding New Features
1. Create new page function in `app.py`
2. Add navigation option in sidebar
3. Update routing in `main()` function
4. Test thoroughly with dark mode

## Troubleshooting

### Common Issues
- **Port Already in Use**: Change port with `--server.port 8502`
- **Dark Mode Not Applied**: Clear browser cache and refresh
- **File Upload Errors**: Check file size and format (CSV only)
- **Analysis Timeout**: Increase timeout for large batch processing

### Performance Optimization
- Use caching for ML model loading
- Implement pagination for large datasets
- Optimize chart rendering with Plotly
- Consider async processing for batch analysis

## Screenshots

- **Main Dashboard**: Overview with metrics and quick actions
- **Single Review Analysis**: Business category dropdown and analysis results
- **Batch Analysis**: Multi-checkbox filters without confidence threshold
- **Violation List**: Combined filters and violation management

## License

This web interface is part of the fake review detection system and follows the same licensing terms as the main project.