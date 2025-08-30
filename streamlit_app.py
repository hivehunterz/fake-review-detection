"""
🛡️ Fake Review Detection System - Web Interface
Main Streamlit application for fake review detection with dashboard capabilities.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
from datetime import datetime, timedelta
import numpy as np

# Add project paths for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "scripts" / "prediction"))
sys.path.append(str(PROJECT_ROOT / "core"))

# Page configuration
st.set_page_config(
    page_title="Fake Review Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode and styling
def load_css():
    st.markdown("""
    <style>
    .main {
        background-color: #1e1e1e;
        color: #FAFAFA;
    }
    .stApp {
        background-color: #1e1e1e;
    }
    .metric-card {
        background-color: #2d2d2d;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    .status-genuine {
        color: #4CAF50;
    }
    .status-suspicious {
        color: #FF9800;
    }
    .status-spam {
        color: #F44336;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stSelectbox > div > div {
        background-color: #2d2d2d;
        color: #FAFAFA;
    }
    .stTextInput > div > div {
        background-color: #2d2d2d;
        color: #FAFAFA;
    }
    .stTextArea > div > div {
        background-color: #2d2d2d;
        color: #FAFAFA;
    }
    div[data-testid="metric-container"] {
        background-color: #2d2d2d;
        border: 1px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Business categories for dropdown (top 8 most common + Other)
BUSINESS_CATEGORIES = [
    "Restaurant",
    "Hotel", 
    "Retail Store",
    "Service Business",
    "Healthcare",
    "Entertainment",
    "Automotive",
    "Technology",
    "Other"
]

# Review categories for filtering
REVIEW_CATEGORIES = [
    "genuine_positive",
    "genuine_negative", 
    "spam",
    "low_quality",
    "suspicious",
    "advertisement",
    "fake_positive",
    "irrelevant"
]

def main():
    load_css()
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "Main Dashboard"
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
        
    # Initialize ML backend
    if st.session_state.predictor is None:
        try:
            initialize_ml_backend()
        except Exception as e:
            st.error(f"Warning: ML backend not fully loaded. Using demo mode. Error: {e}")
    
    # Sidebar navigation
    st.sidebar.title("🛡️ Review Guardian")
    st.sidebar.markdown("---")
    
    pages = [
        "📊 Main Dashboard",
        "🔍 Single Review Analysis", 
        "📝 Batch Analysis",
        "⚠️ Violation List"
    ]
    
    # Remove emoji for comparison
    page_names = [p.split(" ", 1)[1] for p in pages]
    
    selected_page = st.sidebar.selectbox("Navigate to:", pages, index=0)
    st.session_state.page = selected_page.split(" ", 1)[1]
    
    # Display selected page
    if st.session_state.page == "Main Dashboard":
        show_main_dashboard()
    elif st.session_state.page == "Single Review Analysis":
        show_single_review_analysis()
    elif st.session_state.page == "Batch Analysis":
        show_batch_analysis()
    elif st.session_state.page == "Violation List":
        show_violation_list()

def initialize_ml_backend():
    """Initialize the ML prediction pipeline"""
    try:
        # Create required directories
        output_dir = PROJECT_ROOT / "output"
        models_dir = PROJECT_ROOT / "models"
        output_dir.mkdir(exist_ok=True)
        models_dir.mkdir(exist_ok=True)
        
        from predict_review_quality import ReviewQualityPredictor
        st.session_state.predictor = ReviewQualityPredictor()
        
        # Try to load models, but don't fail if they don't exist
        try:
            if st.session_state.predictor.load_models():
                st.sidebar.success("✅ Full ML Backend Loaded")
                return True
            else:
                st.sidebar.warning("⚠️ Using Demo Mode - Models Not Found")
                # Keep the predictor object for interface consistency
                return False
        except Exception as model_error:
            st.sidebar.warning(f"⚠️ Demo Mode Active - {str(model_error)[:30]}...")
            return False
            
    except Exception as e:
        st.sidebar.error(f"❌ Backend Error: {str(e)[:30]}...")
        st.session_state.predictor = None
        return False

def show_main_dashboard():
    """Main dashboard page with overview and links to other sections"""
    st.title("📊 Fake Review Detection Dashboard")
    st.markdown("### Welcome to the Smart Review Guardian System")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", "2,847", "+127")
    with col2:
        st.metric("Genuine Reviews", "1,854", "+85")
    with col3:
        st.metric("Suspicious Reviews", "693", "+28")
    with col4:
        st.metric("Spam/Fake Reviews", "300", "+14")
    
    st.markdown("---")
    
    # Quick action buttons
    st.markdown("### Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Analyze Single Review", use_container_width=True):
            st.session_state.page = "Single Review Analysis"
            st.rerun()
    
    with col2:
        if st.button("📝 Batch Analysis", use_container_width=True):
            st.session_state.page = "Batch Analysis"
            st.rerun()
    
    with col3:
        if st.button("⚠️ View Violations", use_container_width=True):
            st.session_state.page = "Violation List"
            st.rerun()
    
    st.markdown("---")
    
    # Recent activity summary (no policy violation breakdown chart as requested)
    st.markdown("### Recent Activity Summary")
    
    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        'Date': pd.date_range(start='2025-01-01', periods=30, freq='D'),
        'Genuine': np.random.randint(20, 40, 30),
        'Suspicious': np.random.randint(5, 15, 30),
        'Spam': np.random.randint(2, 8, 30)
    })
    
    fig = px.line(sample_data, x='Date', y=['Genuine', 'Suspicious', 'Spam'],
                  title="Daily Review Classification Trends",
                  labels={'value': 'Number of Reviews', 'variable': 'Classification'})
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#FAFAFA'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_single_review_analysis():
    """Single review analysis page with business category dropdown"""
    st.title("🔍 Single Review Analysis")
    st.markdown("### Analyze Individual Reviews")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Review Details")
        
        # Business category dropdown with fixed options (top 8 + Other)
        business_category = st.selectbox(
            "Business Category",
            BUSINESS_CATEGORIES,
            help="Select the business category for better analysis context"
        )
        
        business_name = st.text_input("Business Name (Optional)", placeholder="e.g., Mario's Italian Restaurant")
        
        review_text = st.text_area(
            "Review Text",
            placeholder="Enter the review text to analyze...",
            height=150
        )
        
        # Additional optional fields
        st.markdown("#### Additional Information (Optional)")
        
        col_rating, col_date = st.columns(2)
        with col_rating:
            rating = st.selectbox("Rating", ["Select...", "1", "2", "3", "4", "5"])
        with col_date:
            review_date = st.date_input("Review Date", value=None)
        
        reviewer_name = st.text_input("Reviewer Name", placeholder="John Smith")
        
        # Analysis button
        if st.button("🔍 Analyze Review", type="primary", use_container_width=True):
            if review_text.strip():
                with st.spinner("Analyzing review..."):
                    result = analyze_single_review(
                        review_text, business_category, business_name, 
                        rating if rating != "Select..." else None, reviewer_name
                    )
                    st.session_state.analysis_result = result
            else:
                st.error("Please enter review text to analyze.")
    
    with col2:
        st.markdown("#### Analysis Results")
        
        if 'analysis_result' in st.session_state:
            result = st.session_state.analysis_result
            
            # Classification result
            classification = result.get('final_prediction', 'unknown')
            confidence = result.get('final_confidence', 0)
            
            if classification in ['genuine_positive', 'genuine_negative']:
                status_class = 'status-genuine'
                icon = '✅'
            elif classification in ['suspicious', 'low_quality']:
                status_class = 'status-suspicious'
                icon = '⚠️'
            else:
                status_class = 'status-spam'
                icon = '🚫'
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{icon} Classification</h3>
                <h4 class="{status_class}">{classification.replace('_', ' ').title()}</h4>
                <p>Confidence: {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk score
            risk_score = result.get('risk_score', 0)
            risk_color = 'status-genuine' if risk_score < 0.3 else 'status-suspicious' if risk_score < 0.7 else 'status-spam'
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Risk Score</h3>
                <h4 class="{risk_color}">{risk_score:.3f}</h4>
                <small>Lower is better</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Backend mode indicator
            backend_mode = result.get('backend_mode', 'demo')
            if backend_mode == 'full_ml':
                st.markdown("🤖 **Status**: Full ML Backend Active", help="Using trained BART, metadata, and fusion models")
            elif backend_mode == 'bart_fallback':
                st.markdown("🔄 **Status**: BART Fallback Mode", help="Using BART classifier only")
            else:
                st.markdown("🎭 **Status**: Demo Mode Active", help="Using enhanced heuristic analysis")
            
            # Key indicators
            st.markdown("#### Key Indicators")
            indicators = result.get('indicators', [])
            for indicator in indicators:
                st.markdown(f"• {indicator}")
                
            # Show fallback reason if applicable
            if 'fallback_reason' in result:
                st.markdown(f"**Fallback reason**: {result['fallback_reason']}")
        
        else:
            st.info("Enter review details and click 'Analyze Review' to see results here.")

def analyze_single_review(review_text, business_category, business_name, rating, reviewer_name):
    """Analyze a single review using the ML backend"""
    try:
        if st.session_state.predictor and st.session_state.predictor.models_loaded:
            # Use full ML backend
            metadata = {}
            if rating:
                metadata['rating'] = float(rating)
            if business_name:
                metadata['business_name'] = business_name
            if business_category:
                metadata['category'] = business_category
            if reviewer_name:
                metadata['reviewer_name'] = reviewer_name
                
            # Call ML backend
            result = st.session_state.predictor.predict_single_review(review_text, metadata)
            
            # Add indicators based on analysis
            indicators = []
            if result.get('final_confidence', 0) > 0.8:
                indicators.append("High confidence prediction")
            if result.get('risk_score', 0) < 0.2:
                indicators.append("Low risk review")
            elif result.get('risk_score', 0) > 0.7:
                indicators.append("High risk review - requires attention")
                
            result['indicators'] = indicators
            result['backend_mode'] = 'full_ml'
            return result
            
        elif st.session_state.predictor:
            # Try BART fallback if available
            try:
                import sys
                sys.path.append(str(PROJECT_ROOT / "core" / "stage1_bart"))
                from enhanced_bart_review_classifier import BARTReviewClassifier
                
                bart_classifier = BARTReviewClassifier(model_path=None, use_gpu=False)
                bart_result = bart_classifier.predict_single(review_text)
                
                # Convert BART result to app format
                result = convert_bart_to_app_format(bart_result, review_text, business_category)
                result['backend_mode'] = 'bart_fallback'
                return result
                
            except Exception as bart_error:
                # Fall back to demo mode
                result = generate_demo_result(review_text, business_category)
                result['backend_mode'] = 'demo'
                result['fallback_reason'] = f"BART error: {str(bart_error)[:50]}"
                return result
        else:
            # Pure demo mode
            result = generate_demo_result(review_text, business_category)
            result['backend_mode'] = 'demo'
            result['fallback_reason'] = "No ML backend available"
            return result
            
    except Exception as e:
        # Error fallback
        result = generate_demo_result(review_text, business_category)
        result['backend_mode'] = 'demo'
        result['fallback_reason'] = f"Error: {str(e)[:50]}"
        return result

def convert_bart_to_app_format(bart_result, text, business_category):
    """Convert BART result to app format"""
    return {
        'final_prediction': bart_result.get('classification', 'unknown'),
        'final_confidence': bart_result.get('confidence', 0.5),
        'risk_score': 1.0 - bart_result.get('confidence', 0.5),  # Inverse of confidence
        'indicators': [
            'BART classifier active',
            f'Text length: {len(text)} characters',
            f'Business category: {business_category or "Not specified"}'
        ]
    }

def generate_demo_result(review_text, business_category=None):
    """Generate demo result when ML backend is not available"""
    # Enhanced heuristic-based demo with more realistic logic
    text_lower = review_text.lower()
    text_length = len(review_text)
    
    # Sentiment indicators
    positive_words = ['excellent', 'amazing', 'love', 'great', 'perfect', 'fantastic', 'wonderful', 'awesome']
    negative_words = ['terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'pathetic']
    spam_words = ['buy', 'discount', 'cheap', 'sale', 'click', 'free', 'offer', 'deal']
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    spam_count = sum(1 for word in spam_words if word in text_lower)
    
    # Classification logic
    if spam_count >= 2:
        classification = 'advertisement'
        confidence = 0.85 + (spam_count * 0.05)
        risk_score = 0.80 + (spam_count * 0.05)
    elif text_length < 20:
        classification = 'low_quality'
        confidence = 0.75
        risk_score = 0.60
    elif positive_count >= 3 and text_length > 50:
        classification = 'genuine_positive'
        confidence = 0.80 + min(positive_count * 0.05, 0.15)
        risk_score = 0.15 - min(positive_count * 0.02, 0.10)
    elif negative_count >= 2 and text_length > 30:
        classification = 'genuine_negative'
        confidence = 0.75 + min(negative_count * 0.05, 0.15)
        risk_score = 0.25 - min(negative_count * 0.02, 0.10)
    elif positive_count >= 2 and text_length < 50:
        classification = 'suspicious'
        confidence = 0.65
        risk_score = 0.55
    else:
        classification = 'suspicious'
        confidence = 0.60
        risk_score = 0.45
        
    # Ensure values are in valid ranges
    confidence = max(0.0, min(1.0, confidence))
    risk_score = max(0.0, min(1.0, risk_score))
        
    indicators = [
        'Demo mode active - enhanced heuristics',
        f'Text length: {text_length} characters',
        f'Positive indicators: {positive_count}',
        f'Negative indicators: {negative_count}',
        f'Spam indicators: {spam_count}',
        f'Business category: {business_category or "Not specified"}'
    ]
    
    return {
        'final_prediction': classification,
        'final_confidence': confidence,
        'risk_score': risk_score,
        'demo_mode': True,
        'indicators': indicators
    }

def show_batch_analysis():
    """Batch analysis page without confidence threshold, with multi-category filters"""
    st.title("📝 Batch Analysis")
    st.markdown("### Analyze Multiple Reviews")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Upload Data")
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload a CSV file with review data. Required column: 'text' or 'review_text'"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Found {len(df)} reviews.")
                
                # Show preview
                st.markdown("#### Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Column selection
                text_columns = [col for col in df.columns if 'text' in col.lower() or 'review' in col.lower() or 'content' in col.lower()]
                if text_columns:
                    text_column = st.selectbox("Select text column", text_columns, index=0)
                else:
                    text_column = st.selectbox("Select text column", df.columns, index=0)
                
                # Analysis button
                if st.button("🚀 Start Batch Analysis", type="primary"):
                    with st.spinner("Processing reviews..."):
                        results = run_batch_analysis(df, text_column)
                        st.session_state.batch_results = results
                        
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        # Sample data option
        st.markdown("---")
        if st.button("📄 Load Sample Data"):
            sample_data = load_sample_batch_data()
            st.session_state.batch_results = sample_data
            st.success("Sample data loaded!")
    
    with col2:
        st.markdown("#### Filters")
        
        # Category filters (dropdown with all categories as requested)
        selected_categories = st.multiselect(
            "Filter by Review Categories",
            REVIEW_CATEGORIES,
            default=[],
            help="Select categories to filter results. Leave empty to show all."
        )
        
        if selected_categories:
            st.info(f"Filtering by: {', '.join(selected_categories)}")
        
        # Note: Confidence threshold removed as requested
        st.markdown("---")
        st.markdown("#### Analysis Options")
        
        show_details = st.checkbox("Show detailed analysis", value=True)
        export_results = st.checkbox("Export results", value=True)
        
        # Display current filters
        if 'batch_results' in st.session_state:
            st.markdown("#### Quick Stats")
            results = st.session_state.batch_results
            
            if selected_categories:
                filtered_results = results[results['classification'].isin(selected_categories)]
            else:
                filtered_results = results
            
            st.metric("Total Reviews", len(filtered_results))
            st.metric("Genuine", len(filtered_results[filtered_results['classification'].str.contains('genuine', na=False)]))
            st.metric("Suspicious", len(filtered_results[filtered_results['classification'] == 'suspicious']))
            st.metric("Spam/Fake", len(filtered_results[filtered_results['classification'].isin(['spam', 'fake_positive'])]))
    
    # Display results
    if 'batch_results' in st.session_state:
        st.markdown("---")
        st.markdown("#### Analysis Results")
        
        results = st.session_state.batch_results
        
        # Apply filters
        if selected_categories:
            filtered_results = results[results['classification'].isin(selected_categories)]
        else:
            filtered_results = results
            
        # Results visualization
        if not filtered_results.empty:
            fig = px.histogram(filtered_results, x='classification', 
                             title="Review Classification Distribution",
                             labels={'count': 'Number of Reviews'})
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#FAFAFA'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.dataframe(filtered_results, use_container_width=True)
            
            # Download option
            if export_results:
                csv = filtered_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"batch_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def run_batch_analysis(df, text_column):
    """Run batch analysis on uploaded data"""
    try:
        results = []
        progress_bar = st.progress(0)
        
        for idx, row in df.iterrows():
            text = str(row[text_column])
            
            if st.session_state.predictor and st.session_state.predictor.models_loaded:
                # Use full ML backend
                metadata = {}
                for col in ['rating', 'business_name', 'category', 'reviewer_name']:
                    if col in row and pd.notna(row[col]):
                        metadata[col] = row[col]
                        
                result = st.session_state.predictor.predict_single_review(text, metadata)
                
                results.append({
                    'text': text,
                    'classification': result.get('final_prediction', 'unknown'),
                    'confidence': result.get('final_confidence', 0),
                    'risk_score': result.get('risk_score', 0),
                    'backend_mode': 'full_ml'
                })
            else:
                # Use enhanced demo mode
                demo_result = generate_demo_result(text)
                results.append({
                    'text': text,
                    'classification': demo_result['final_prediction'],
                    'confidence': demo_result['final_confidence'],
                    'risk_score': demo_result['risk_score'],
                    'backend_mode': 'demo'
                })
                
            progress_bar.progress((idx + 1) / len(df))
        
        progress_bar.empty()
        
        # Add summary info
        result_df = pd.DataFrame(results)
        backend_mode = results[0]['backend_mode'] if results else 'demo'
        
        if backend_mode == 'demo':
            st.info("📝 Analysis completed using enhanced demo mode. For production accuracy, train ML models.")
        else:
            st.success("✅ Analysis completed using full ML backend.")
            
        return result_df
        
    except Exception as e:
        st.error(f"Error during batch analysis: {str(e)}")
        return pd.DataFrame()

def load_sample_batch_data():
    """Load sample data for demonstration"""
    sample_reviews = [
        {"text": "Amazing food and great service! Will definitely come back.", "classification": "genuine_positive", "confidence": 0.92, "risk_score": 0.08},
        {"text": "Terrible experience, food was cold and staff was rude.", "classification": "genuine_negative", "confidence": 0.87, "risk_score": 0.13},
        {"text": "Best pizza ever! Click here for discount codes!", "classification": "advertisement", "confidence": 0.95, "risk_score": 0.90},
        {"text": "Good", "classification": "low_quality", "confidence": 0.78, "risk_score": 0.45},
        {"text": "This place has the most incredible atmosphere and the food is absolutely phenomenal!", "classification": "suspicious", "confidence": 0.65, "risk_score": 0.55},
    ]
    
    return pd.DataFrame(sample_reviews)

def show_violation_list():
    """Violation List page - combining filters with recent violations as requested"""
    st.title("⚠️ Violation List")
    st.markdown("### Policy Violations and Recent Issues")
    
    # Filter options (moved here as requested)
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("#### Filters")
        
        # Date range filter
        date_range = st.date_input(
            "Date Range",
            value=[datetime.now() - timedelta(days=7), datetime.now()],
            max_value=datetime.now()
        )
        
        # Severity filter
        severity_filter = st.multiselect(
            "Severity Level",
            ["High", "Medium", "Low"],
            default=["High", "Medium"]
        )
        
        # Violation type filter
        violation_types = [
            "Spam Detection",
            "Fake Reviews", 
            "Policy Violations",
            "Content Violations",
            "User Behavior Issues"
        ]
        
        selected_violations = st.multiselect(
            "Violation Types",
            violation_types,
            default=violation_types
        )
        
        # Status filter
        status_filter = st.multiselect(
            "Status",
            ["Open", "In Review", "Resolved", "Escalated"],
            default=["Open", "In Review"]
        )
    
    with col2:
        st.markdown("#### Recent Violations")
        
        # Generate sample violation data
        sample_violations = generate_sample_violations()
        
        # Apply filters
        filtered_violations = apply_violation_filters(
            sample_violations, severity_filter, selected_violations, status_filter
        )
        
        if not filtered_violations.empty:
            # Display violation summary
            col_total, col_high, col_pending = st.columns(3)
            
            with col_total:
                st.metric("Total Violations", len(filtered_violations))
            
            with col_high:
                high_severity = len(filtered_violations[filtered_violations['Severity'] == 'High'])
                st.metric("High Severity", high_severity)
            
            with col_pending:
                pending = len(filtered_violations[filtered_violations['Status'].isin(['Open', 'In Review'])])
                st.metric("Pending Review", pending)
            
            st.markdown("---")
            
            # Violations table
            st.dataframe(
                filtered_violations[['Date', 'Type', 'Description', 'Severity', 'Status', 'Business']],
                use_container_width=True
            )
            
            # Action buttons
            st.markdown("#### Actions")
            col_export, col_escalate, col_resolve = st.columns(3)
            
            with col_export:
                if st.button("📊 Export Report"):
                    csv = filtered_violations.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"violations_report_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv"
                    )
            
            with col_escalate:
                if st.button("⚠️ Escalate Selected"):
                    st.info("Escalation feature would be implemented here")
            
            with col_resolve:
                if st.button("✅ Mark as Resolved"):
                    st.success("Selected violations marked as resolved")
        
        else:
            st.info("No violations found matching the current filters.")

def generate_sample_violations():
    """Generate sample violation data for demonstration"""
    violations = []
    violation_types = ["Spam Detection", "Fake Reviews", "Policy Violations", "Content Violations", "User Behavior Issues"]
    severities = ["High", "Medium", "Low"]
    statuses = ["Open", "In Review", "Resolved", "Escalated"]
    businesses = ["Pizza Palace", "Tech Store", "Hotel Paradise", "Fashion Boutique", "Car Dealership"]
    
    for i in range(50):
        violation = {
            'ID': f"V{1000 + i}",
            'Date': datetime.now() - timedelta(days=np.random.randint(0, 30)),
            'Type': np.random.choice(violation_types),
            'Description': f"Violation detected in review #{np.random.randint(1000, 9999)}",
            'Severity': np.random.choice(severities, p=[0.3, 0.5, 0.2]),
            'Status': np.random.choice(statuses, p=[0.4, 0.3, 0.2, 0.1]),
            'Business': np.random.choice(businesses),
            'Reviewer': f"User{np.random.randint(100, 999)}"
        }
        violations.append(violation)
    
    return pd.DataFrame(violations)

def apply_violation_filters(violations_df, severity_filter, violation_types, status_filter):
    """Apply filters to violations data"""
    filtered = violations_df.copy()
    
    if severity_filter:
        filtered = filtered[filtered['Severity'].isin(severity_filter)]
    
    if violation_types:
        filtered = filtered[filtered['Type'].isin(violation_types)]
    
    if status_filter:
        filtered = filtered[filtered['Status'].isin(status_filter)]
    
    return filtered

if __name__ == "__main__":
    main()