"""
Configuration for Metadata Anomaly Detection
"""
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = Path(__file__).parent / 'outputs'

# Input files
INPUT_CSV = DATA_DIR / 'data_all_training.csv'

# Output files
ANOMALY_REPORT = OUTPUT_DIR / 'anomaly_report.json'
FEATURES_CSV = OUTPUT_DIR / 'metadata_features.csv'

# Anomaly detection parameters
ANOMALY_THRESHOLDS = {
    'temporal': 0.6,      # Lowered from 0.85
    'user_behavior': 0.6, # Lowered from 0.8
    'content': 0.6,       # Lowered from 0.75
    'geographic': 0.6,   # Lowered from 0.9
    'business': 0.6       # Lowered from 0.8
}

# Feature configuration - updated to match actual features in metadata_analyzer.py
FEATURE_CONFIG = {
    'temporal_features': [
        'review_day_of_week',
        'review_hour_of_day',
        'is_weekend',
        'days_since_last_review',
        'rolling_7d_stars_mean',
        'rolling_7d_stars_count',
        'rolling_30d_stars_mean',
        'rolling_30d_stars_count'
    ],
    'user_behavior_features': [
        'user_review_count',
        'user_avg_rating',
        'user_rating_std',
        'user_min_rating',
        'user_max_rating',
        'user_unique_text_ratio',
        'user_rating_deviation'
    ],
    'content_features': [
        'has_image',
        'image_count',
        'text_length',
        'language_match'
    ],
    'business_features': [
        'business_avg_rating',
        'business_rating_std',
        'business_review_count',
        'business_rating_deviation'
    ]
}


# Rolling Window Parameters
TIME_WINDOWS = {
    'short': 7,    # 7 days
    'medium': 30,  # 30 days
    'long': 90     # 90 days
}

# BART Integration Configuration
BART_CONFIG = {
    'model_path': '../stage_1_bart_finetuning',  # Look for trained models here
    'enable_integration': True,
    'confidence_threshold': 0.7,  # Minimum confidence for high-quality features
    'risk_threshold': 0.5,  # Maximum risk for high-quality content
    'class_weights': {
        'genuine_positive': 1.0,
        'genuine_negative': 1.0,
        'spam': -2.0,           # High penalty for spam
        'advertisement': -1.5,   # Medium penalty for ads
        'irrelevant': -1.0,     # Lower penalty for irrelevant
        'fake_rant': -2.0,      # High penalty for fake rants
        'inappropriate': -2.0    # High penalty for inappropriate content
    }
}

# ML Model Parameters
ML_PARAMS = {
    'n_estimators': 200,      # Increased from 100
    'contamination': 0.1,     # Increased from 0.05
    'random_state': 42,
    'n_jobs': -1,
    'max_samples': 'auto',
    'bootstrap': False
}
