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
sys.path.append(str(PROJECT_ROOT / "scripts" / "evaluation"))
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
    /* Dark mode variables */
    :root {
        --bg-color: #0E1117;
        --secondary-bg: #262730;
        --text-color: #FAFAFA;
        --text-secondary: #CCCCCC;
        --text-muted: #999999;
        --accent-color: #FF6B6B;
        --success-color: #4ECDC4;
        --warning-color: #FFE66D;
        --border-color: #333333;
    }
    
    /* Apply dark mode colors to all text elements including small text */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }
    
    /* Ensure small text elements are properly colored in dark mode */
    .stApp p, .stApp span, .stApp div, .stApp label, .stApp small {
        color: var(--text-color) !important;
    }
    
    .stApp .stMarkdown small, .stApp .caption, .stApp .help {
        color: var(--text-muted) !important;
    }
    
    /* Card styling */
    .metric-card {
        background-color: var(--secondary-bg);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        margin: 0.5rem 0;
    }
    
    /* Navigation styling */
    .nav-link {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        background-color: var(--secondary-bg);
        border-radius: 4px;
        text-decoration: none;
        color: var(--text-color);
        border: 1px solid var(--border-color);
    }
    
    .nav-link:hover {
        background-color: var(--accent-color);
        color: white;
    }
    
    /* Status indicators */
    .status-genuine { color: var(--success-color); }
    .status-suspicious { color: var(--warning-color); }
    .status-spam { color: var(--accent-color); }
    
    /* Fix sidebar text in dark mode */
    .css-1d391kg, .css-1d391kg p, .css-1d391kg span {
        color: var(--text-color) !important;
    }
    
    /* Fix selectbox and input text */
    .stSelectbox label, .stTextInput label, .stTextArea label {
        color: var(--text-color) !important;
    }
    
    /* Fix multiselect labels and help text */
    .stMultiSelect label, .stMultiSelect .help {
        color: var(--text-color) !important;
    }
    
    /* Fix checkbox labels */
    .stCheckbox label {
        color: var(--text-color) !important;
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
    
    # Sidebar navigation
    st.sidebar.title("🛡️ Navigation")
    
    page = st.sidebar.selectbox(
        "Select Page",
        ["Main Dashboard", "Single Review Analysis", "Batch Analysis", "Violation List"]
    )
    
    # Dark mode toggle (for future enhancement)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Settings")
    dark_mode = st.sidebar.checkbox("Dark Mode", value=True)
    
    # Navigation links
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Links")
    st.sidebar.markdown("""
    - [📊 Main Dashboard](#main-dashboard)
    - [🔍 Single Review Analysis](#single-review-analysis)  
    - [📝 Batch Analysis](#batch-analysis)
    - [⚠️ Violation List](#violation-list)
    """, unsafe_allow_html=True)
    
    # Route to appropriate page
    if page == "Main Dashboard":
        show_main_dashboard()
    elif page == "Single Review Analysis":
        show_single_review_analysis()
    elif page == "Batch Analysis":
        show_batch_analysis()
    elif page == "Violation List":
        show_violation_list()

def show_main_dashboard():
    """Main dashboard page with overview and links to other sections"""
    st.title("🛡️ Fake Review Detection System")
    st.markdown("### Main Dashboard")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Total Reviews</h3>
            <h2 class="status-genuine">1,247</h2>
            <small>Last 30 days</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>✅ Genuine</h3>
            <h2 class="status-genuine">892 (71.5%)</h2>
            <small>High confidence reviews</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚠️ Suspicious</h3>
            <h2 class="status-suspicious">245 (19.6%)</h2>
            <small>Require manual review</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🚫 Spam/Fake</h3>
            <h2 class="status-spam">110 (8.9%)</h2>
            <small>Automatically flagged</small>
        </div>
        """, unsafe_allow_html=True)
    
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
            rating = st.selectbox("Rating", [None, 1, 2, 3, 4, 5], index=0)
        with col_date:
            review_date = st.date_input("Review Date", value=None)
        
        reviewer_name = st.text_input("Reviewer Name (Optional)")
        
        if st.button("🔍 Analyze Review", type="primary", use_container_width=True):
            if review_text.strip():
                analyze_single_review(review_text, business_category, business_name, rating, reviewer_name)
            else:
                st.error("Please enter review text to analyze.")
    
    with col2:
        st.markdown("#### Analysis Results")
        
        if 'single_analysis_result' in st.session_state:
            result = st.session_state.single_analysis_result
            
            # Display classification result
            classification = result.get('classification', 'Unknown')
            confidence = result.get('confidence', 0)
            
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
            
            # Key indicators
            st.markdown("#### Key Indicators")
            indicators = result.get('indicators', [])
            for indicator in indicators:
                st.markdown(f"• {indicator}")
        
        else:
            st.info("Enter review details and click 'Analyze Review' to see results here.")

def analyze_single_review(review_text, business_category, business_name, rating, reviewer_name):
    """Analyze a single review using the ML backend"""
    try:
        # Mock analysis for demonstration - in real implementation, this would call the ML models
        import random
        
        # Simulate analysis delay
        with st.spinner("Analyzing review..."):
            import time
            time.sleep(2)
        
        # Mock classification result
        classifications = ['genuine_positive', 'genuine_negative', 'suspicious', 'spam', 'low_quality']
        classification = random.choice(classifications)
        confidence = random.uniform(0.6, 0.95)
        risk_score = random.uniform(0.1, 0.8)
        
        # Mock indicators based on classification
        if classification in ['genuine_positive', 'genuine_negative']:
            indicators = [
                "Natural language patterns detected",
                "Appropriate length and detail",
                f"Content matches {business_category.lower()} context"
            ]
        elif classification == 'suspicious':
            indicators = [
                "Mixed signals in content analysis",
                "Unusual posting patterns detected",
                "Requires manual verification"
            ]
        else:
            indicators = [
                "Suspicious language patterns",
                "Potential promotional content",
                "Low content quality signals"
            ]
        
        result = {
            'classification': classification,
            'confidence': confidence,
            'risk_score': risk_score,
            'indicators': indicators
        }
        
        st.session_state.single_analysis_result = result
        st.success("Analysis complete!")
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")

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
                    selected_column = st.selectbox("Select text column", text_columns)
                else:
                    selected_column = st.selectbox("Select text column", df.columns.tolist())
                
                # Analysis button
                if st.button("🚀 Start Batch Analysis", type="primary", use_container_width=True):
                    run_batch_analysis(df, selected_column)
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        else:
            # Sample data option
            st.markdown("#### Or Try Sample Data")
            if st.button("📊 Load Sample Dataset"):
                load_sample_batch_data()
    
    with col2:
        st.markdown("#### Filter Options")
        
        # Multi-checkbox filters for all categories (as requested)
        st.markdown("**Review Categories**")
        selected_categories = []
        
        for category in REVIEW_CATEGORIES:
            if st.checkbox(category.replace('_', ' ').title(), key=f"filter_{category}"):
                selected_categories.append(category)
        
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

def run_batch_analysis(df, text_column):
    """Run batch analysis on uploaded data"""
    try:
        with st.spinner("Processing batch analysis..."):
            import time
            time.sleep(3)  # Simulate processing
            
            # Mock batch analysis results
            results = []
            for idx, row in df.iterrows():
                text = row[text_column]
                
                # Mock classification
                import random
                classification = random.choice(REVIEW_CATEGORIES)
                confidence = random.uniform(0.5, 0.95)
                risk_score = random.uniform(0.1, 0.9)
                
                results.append({
                    'index': idx,
                    'text': text,
                    'classification': classification,
                    'confidence': confidence,
                    'risk_score': risk_score
                })
            
            results_df = pd.DataFrame(results)
            st.session_state.batch_results = results_df
            
            st.success(f"✅ Batch analysis complete! Processed {len(results)} reviews.")
            
            # Display results
            st.markdown("#### Analysis Results")
            
            # Summary charts
            fig = px.histogram(results_df, x='classification', 
                             title="Review Classification Distribution",
                             labels={'count': 'Number of Reviews'})
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#FAFAFA'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.dataframe(results_df, use_container_width=True)
            
            # Download option
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name=f"batch_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"Error during batch analysis: {str(e)}")

def load_sample_batch_data():
    """Load sample data for demonstration"""
    sample_data = pd.DataFrame({
        'review_text': [
            "Great product, fast shipping, highly recommend!",
            "Buy now! Amazing deals! Click here for discount codes!",
            "Terrible quality, completely broken on arrival",
            "Decent product for the price, nothing special",
            "BEST EVER!!!! Everyone should buy this NOW!!!",
            "Good customer service, resolved my issue quickly",
            "Random spam content not related to business",
            "Professional service, clean facilities, will return"
        ],
        'rating': [5, 5, 1, 3, 5, 4, 1, 4],
        'reviewer': ['John D.', 'PromoBot', 'Sarah M.', 'Mike R.', 'FakeUser123', 'Anna L.', 'SpamBot', 'Professional User']
    })
    
    st.session_state.sample_data = sample_data
    st.success("Sample data loaded! Click 'Start Batch Analysis' to process.")
    st.dataframe(sample_data, use_container_width=True)

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
                if st.button("⚡ Escalate Selected"):
                    st.info("Escalation feature would be implemented here")
            
            with col_resolve:
                if st.button("✅ Mark Resolved"):
                    st.info("Resolution feature would be implemented here")
        
        else:
            st.info("No violations found matching the selected filters.")

def generate_sample_violations():
    """Generate sample violation data for demonstration"""
    sample_data = {
        'Date': [
            datetime.now() - timedelta(days=i) for i in range(1, 11)
        ],
        'Type': [
            'Spam Detection', 'Fake Reviews', 'Policy Violations', 'Content Violations',
            'User Behavior Issues', 'Spam Detection', 'Fake Reviews', 'Policy Violations',
            'Content Violations', 'User Behavior Issues'
        ],
        'Description': [
            'Multiple spam reviews detected from same IP',
            'Coordinated fake positive reviews',
            'Review content violates community guidelines',
            'Inappropriate language in review',
            'Suspicious reviewer account activity',
            'Promotional content in reviews',
            'Review bombing detected',
            'Off-topic review content',
            'Harassment in review comments',
            'Bot-like posting patterns'
        ],
        'Severity': ['High', 'High', 'Medium', 'Low', 'Medium', 'High', 'High', 'Low', 'Medium', 'Medium'],
        'Status': ['Open', 'In Review', 'Open', 'Resolved', 'In Review', 'Open', 'Escalated', 'Resolved', 'Open', 'In Review'],
        'Business': [
            'Restaurant ABC', 'Hotel XYZ', 'Tech Store', 'Cafe Luna', 'Auto Shop',
            'Restaurant ABC', 'Shopping Mall', 'Service Co', 'Healthcare Plus', 'Tech Store'
        ]
    }
    
    return pd.DataFrame(sample_data)

def apply_violation_filters(df, severity_filter, violation_types, status_filter):
    """Apply filters to violation data"""
    filtered = df.copy()
    
    if severity_filter:
        filtered = filtered[filtered['Severity'].isin(severity_filter)]
    
    if violation_types:
        filtered = filtered[filtered['Type'].isin(violation_types)]
    
    if status_filter:
        filtered = filtered[filtered['Status'].isin(status_filter)]
    
    return filtered

if __name__ == "__main__":
    main()