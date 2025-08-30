"""
Enhanced Metadata Analyzer with Stage 1 BART Integration
Incorporates BART text quality outputs into the metadata anomaly detection pipeline
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

# Import the existing Stage 1 BART model
import sys
sys.path.append('../stage_1_bart_finetuning')

# Try to import         # Add BART summary if BART features are available
        if 'bart_classification' in self.df.columns:
            bart_counts = self.df['bart_classification'].value_counts()
            bart_summary = {
                'classification_distribution': bart_counts.to_dict(),
            }
            
            # Add confidence metrics if available
            if 'bart_confidence' in self.df.columns:
                bart_summary['avg_confidence'] = float(self.df['bart_confidence'].mean())
            
            # Add quality risk metrics if available
            if 'bart_low_quality_risk' in self.df.columns:
                bart_summary['avg_quality_risk'] = float(self.df['bart_low_quality_risk'].mean())
                bart_summary['high_risk_reviews'] = int((self.df['bart_low_quality_risk'] > 0.7).sum())
            
            report['bart_summary'] = bart_summaryl evaluation - handle import gracefully
try:
    from comprehensive_model_evaluation import ComprehensiveModelEvaluator
    BART_AVAILABLE = True
except ImportError:
    print("Warning: BART model evaluation not available")
    ComprehensiveModelEvaluator = None
    BART_AVAILABLE = False

from config import (
    INPUT_CSV, OUTPUT_DIR, ANOMALY_REPORT, FEATURES_CSV,
    ANOMALY_THRESHOLDS, FEATURE_CONFIG, TIME_WINDOWS
)

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(OUTPUT_DIR) / 'enhanced_metadata_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedMetadataAnalyzer:
    """Enhanced metadata analyzer that incorporates Stage 1 BART text quality features."""
    
    def __init__(self, bart_model_path=None):
        """Initialize the analyzer with BART integration."""
        self.df = None
        self.features = {}
        self.anomaly_scores = {}
        self.output_dir = Path(OUTPUT_DIR)
        
        # Initialize BART model for Stage 1 integration
        self.bart_model = None
        if bart_model_path:
            self._load_bart_model(bart_model_path)
    
    def _load_bart_model(self, model_path):
        """Load the trained BART model from Stage 1."""
        if not BART_AVAILABLE:
            logger.warning("BART model evaluation not available - skipping BART integration")
            return
            
        try:
            # Look for trained BART model
            test_data_path = '../data/data_all_training.csv'
            if Path(test_data_path).exists() and ComprehensiveModelEvaluator:
                self.bart_model = ComprehensiveModelEvaluator(model_path, test_data_path)
                self.bart_model.load_models()
                logger.info(f"✅ Loaded BART model from {model_path}")
            else:
                logger.warning("❌ Test data not found for BART initialization or ComprehensiveModelEvaluator not available")
        except Exception as e:
            logger.error(f"❌ Failed to load BART model: {e}")
            self.bart_model = None
    
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the review data."""
        logger.info(f"Loading data from {INPUT_CSV}")
        self.df = pd.read_csv(INPUT_CSV, low_memory=False)
        
        # Convert date columns to datetime
        date_columns = ['publishAt', 'publishedAtDate']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        logger.info(f"Loaded {len(self.df)} reviews")
        return self.df
    
    def extract_bart_features(self) -> Dict:
        """Extract Stage 1 BART text quality features."""
        logger.info("🤖 Extracting BART text quality features...")
        features = {}
        
        if self.bart_model is None:
            logger.warning("BART model not available, skipping BART features")
            return features
        
        if 'text' not in self.df.columns:
            logger.warning("'text' column not found, cannot extract BART features")
            return features
        
        try:
            # Get BART predictions for all texts
            texts = self.df['text'].fillna('').astype(str).tolist()
            logger.info(f"Processing {len(texts)} texts with BART...")
            
            # Get predictions and confidence scores
            predictions, confidences = self.bart_model.predict_fine_tuned(texts)
            
            # Get full probability distributions
            import torch
            self.bart_model.fine_tuned_model.eval()
            all_probabilities = []
            
            with torch.no_grad():
                for text in texts:
                    inputs = self.bart_model.fine_tuned_tokenizer(
                        text, 
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512
                    ).to(self.bart_model.device)
                    
                    outputs = self.bart_model.fine_tuned_model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    prob_list = probs.squeeze().cpu().numpy().tolist()
                    all_probabilities.append(prob_list)
            
            # Convert probabilities to features
            probs = np.array(all_probabilities)
            labels = ['genuine_positive', 'genuine_negative', 'spam', 'advertisement', 
                     'irrelevant', 'fake_rant', 'inappropriate']
            
            # Add individual class probabilities
            for i, label in enumerate(labels):
                self.df[f'bart_prob_{label}'] = probs[:, i]
            
            # Calculate aggregated features
            self.df['bart_classification'] = predictions
            self.df['bart_confidence'] = confidences
            
            # Low quality risk score (sum of problematic classes)
            bad_labels = ['spam', 'advertisement', 'irrelevant', 'fake_rant', 'inappropriate']
            bad_mask = np.isin(labels, bad_labels)
            self.df['bart_low_quality_risk'] = probs[:, bad_mask].sum(axis=1)
            
            # High quality score (genuine classes)
            good_mask = np.isin(labels, ['genuine_positive', 'genuine_negative'])
            self.df['bart_high_quality_score'] = probs[:, good_mask].sum(axis=1)
            
            # Text quality indicators
            self.df['bart_spam_prob'] = probs[:, labels.index('spam')]
            self.df['bart_ad_prob'] = probs[:, labels.index('advertisement')]
            self.df['bart_irrelevant_prob'] = probs[:, labels.index('irrelevant')]
            
            # Confidence-weighted quality scores
            self.df['bart_weighted_quality'] = self.df['bart_high_quality_score'] * self.df['bart_confidence']
            self.df['bart_weighted_risk'] = self.df['bart_low_quality_risk'] * self.df['bart_confidence']
            
            logger.info(f"✅ Added {len([c for c in self.df.columns if c.startswith('bart_')])} BART features")
            
            # Log distribution of classifications
            class_counts = pd.Series(predictions).value_counts()
            logger.info(f"📊 BART Classification Distribution:")
            for class_name, count in class_counts.items():
                pct = (count / len(predictions)) * 100
                logger.info(f"   {class_name}: {count} ({pct:.1f}%)")
                
        except Exception as e:
            logger.error(f"❌ Error extracting BART features: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return features
    
    def add_time_based_features(self):
        """Add basic calendar/time-based features."""
        if 'publishedAtDate' not in self.df.columns:
            logger.warning("'publishedAtDate' column not found for time-based features.")
            return
        
        # Ensure datetime type
        self.df['publishedAtDate'] = pd.to_datetime(self.df['publishedAtDate'], errors='coerce')
    
        # Drop NaT rows if necessary
        valid_dates = self.df['publishedAtDate'].notna()
        if not valid_dates.any():
            logger.warning("No valid dates available for time-based features.")
            return
    
        self.df.loc[valid_dates, 'review_day_of_week'] = self.df.loc[valid_dates, 'publishedAtDate'].dt.dayofweek
        self.df.loc[valid_dates, 'review_hour_of_day'] = self.df.loc[valid_dates, 'publishedAtDate'].dt.hour
        self.df.loc[valid_dates, 'is_weekend'] = self.df.loc[valid_dates, 'publishedAtDate'].dt.dayofweek.isin([5, 6]).astype(int)
    
        # Days since previous review (per user if possible)
        if 'reviewerId' in self.df.columns:
            self.df = self.df.sort_values(['reviewerId', 'publishedAtDate'])
            self.df['days_since_last_review'] = self.df.groupby('reviewerId')['publishedAtDate'].diff().dt.days
        else:
            self.df = self.df.sort_values('publishedAtDate')
        self.df['days_since_last_review'] = self.df['publishedAtDate'].diff().dt.days
    
        logger.info("Added time-based features: day_of_week, hour_of_day, is_weekend, days_since_last_review")

    def extract_temporal_features(self) -> Dict:
        """Extract temporal features from review data using proper time-based windows."""
        logger.info("📅 Extracting temporal features")
        features = {}
        
        if 'publishedAtDate' not in self.df.columns:
            logger.warning("'publishedAtDate' column not found. Cannot extract temporal features.")
            return features
        
        try:
            # Ensure the column is in datetime format with timezone handling
            if not pd.api.types.is_datetime64_any_dtype(self.df['publishedAtDate']):
                self.df['publishedAtDate'] = pd.to_datetime(
                    self.df['publishedAtDate'], 
                    errors='coerce',
                    utc=True
                )
            
            # Only proceed if we have valid datetime values
            if self.df['publishedAtDate'].isna().all():
                logger.warning("No valid datetime values found in 'publishedAtDate' column")
                return features
            
            # Add base time features
            self.add_time_based_features()
            
            # Add rolling window features
            self._add_rolling_window_features()
            
            # Log the temporal features we've added
            temporal_features = [col for col in self.df.columns if any(
                x in col for x in ['_day_', '_hour_', 'weekend', 'days_since', 'rolling_']
            )]
            logger.info(f"Added {len(temporal_features)} temporal features")
            
        except Exception as e:
            logger.error(f"Error extracting temporal features: {str(e)}")
            logger.exception(e)
        
        return features
    
    def _add_rolling_window_features(self):
        """Add rolling window features while preventing data leakage."""
        if 'reviewerId' not in self.df.columns or 'publishedAtDate' not in self.df.columns:
            return
            
        # Sort by user and date
        self.df = self.df.sort_values(['reviewerId', 'publishedAtDate'])
        
        # Define the windows we want to use (in days)
        windows = [7, 30]  # 7-day and 30-day windows
        
        for window in windows:
            # For each user, calculate rolling statistics using only past data
            def rolling_stats(group):
                # Ensure we're working with a DataFrame
                if not isinstance(group, pd.DataFrame):
                    return pd.DataFrame()
                    
                # Make sure we have a datetime index
                group = group.set_index('publishedAtDate').sort_index()
                
                # Calculate rolling statistics
                rolling = group.rolling(
                    window=f'{window}D',
                    closed='left'  # Only use past data
                ).agg({
                    'stars': ['mean', 'count', 'std'],
                    'reviewId': 'count'
                })
                
                # Reset index to maintain the original structure
                return rolling.reset_index()
            
            try:
                # Apply rolling stats per user
                rolling = self.df.groupby('reviewerId', group_keys=False).apply(
                    lambda x: rolling_stats(x[['publishedAtDate', 'stars', 'reviewId']])
                )
                
                # Rename columns to be more descriptive
                if not rolling.empty:
                    rolling.columns = [
                        'publishedAtDate',
                        f'rolling_{window}d_stars_mean',
                        f'rolling_{window}d_stars_count',
                        f'rolling_{window}d_stars_std',
                        f'rolling_{window}d_review_count'
                    ]
                    
                    # Merge back with original data
                    self.df = self.df.merge(
                        rolling,
                        on=['reviewerId', 'publishedAtDate'],
                        how='left'
                    )
                    
            except Exception as e:
                logger.error(f"Error calculating {window}-day rolling features: {str(e)}")
                continue
    
    def extract_user_behavior_features(self) -> Dict:
        """Extract user behavior features."""
        logger.info("👤 Extracting user behavior features")
        features = {}
        
        if 'reviewerId' in self.df.columns:
            # Ensure we have the required columns
            required_cols = ['reviewId', 'stars', 'text']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns for user behavior analysis: {', '.join(missing_cols)}")
                logger.warning("Some features will not be generated.")
            
            try:
                # Calculate basic user statistics
                user_stats = self.df.groupby('reviewerId').agg({
                    'reviewId': 'count',
                    'stars': ['mean', 'std', 'min', 'max'],
                    'text': lambda x: x.nunique() / len(x) if len(x) > 0 else 0  # Uniqueness ratio
                }).reset_index()
                
                # Flatten multi-level column index
                user_stats.columns = [
                    'reviewerId', 
                    'user_review_count', 
                    'user_avg_rating',
                    'user_rating_std',
                    'user_min_rating',
                    'user_max_rating',
                    'user_unique_text_ratio'
                ]
                
                # Merge user stats back to the main dataframe
                self.df = self.df.merge(user_stats, on='reviewerId', how='left')
                
                # Calculate rating deviation from user's average
                if 'stars' in self.df.columns and 'user_avg_rating' in self.df.columns:
                    self.df['user_rating_deviation'] = abs(
                        self.df['stars'] - self.df['user_avg_rating']
                    )
                
                # Add BART-based user behavior features if available
                if 'bart_low_quality_risk' in self.df.columns:
                    user_bart_stats = self.df.groupby('reviewerId').agg({
                        'bart_low_quality_risk': ['mean', 'max', 'std'],
                        'bart_confidence': ['mean', 'min'],
                        'bart_spam_prob': 'mean'
                    }).reset_index()
                    
                    # Flatten columns
                    user_bart_stats.columns = [
                        'reviewerId',
                        'user_avg_quality_risk', 'user_max_quality_risk', 'user_std_quality_risk',
                        'user_avg_bart_confidence', 'user_min_bart_confidence',
                        'user_avg_spam_prob'
                    ]
                    
                    # Merge BART user stats
                    self.df = self.df.merge(user_bart_stats, on='reviewerId', how='left')
                    
                    logger.info("Added BART-enhanced user behavior features")
                
                logger.info(f"Extracted user behavior features for {len(self.df)} reviews")
                
            except Exception as e:
                logger.error(f"Error extracting user behavior features: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
        else:
            logger.warning("'reviewerId' column not found. Cannot extract user behavior features.")
        
        return features
    
    def extract_content_features(self) -> Dict:
        """Extract content-based features enhanced with BART."""
        logger.info("📝 Extracting enhanced content features")
        features = {}
        
        # Image features
        if 'reviewImageUrls' in self.df.columns:
            self.df['has_image'] = ~self.df['reviewImageUrls'].isna().astype(int)
            self.df['image_count'] = self.df['reviewImageUrls'].apply(
                lambda x: len(eval(x)) if isinstance(x, str) and x.startswith('[') else 0
            )
        
        # Text length features
        if 'text' in self.df.columns:
            self.df['text_length'] = self.df['text'].str.len()
            self.df['word_count'] = self.df['text'].str.split().str.len()
            self.df['avg_word_length'] = self.df['text'].apply(
                lambda x: np.mean([len(word) for word in str(x).split()]) if pd.notna(x) and str(x).split() else 0
            )
            
            # Language consistency (if available)
            if 'language' in self.df.columns and 'originalLanguage' in self.df.columns:
                self.df['language_match'] = (
                    self.df['language'] == self.df['originalLanguage']
                ).astype(int)
        
        # BART-enhanced content features
        if 'bart_low_quality_risk' in self.df.columns:
            # Content quality indicators based on BART + metadata
            self.df['content_quality_score'] = (
                (1 - self.df['bart_low_quality_risk']) * self.df['bart_confidence']
            )
            
            # Text complexity vs BART confidence (low complexity + high confidence = suspicious)
            if 'word_count' in self.df.columns:
                self.df['simplicity_confidence_ratio'] = (
                    self.df['bart_confidence'] / (self.df['word_count'] + 1)
                )
            
            # Spam probability vs text length (short spam is suspicious)
            if 'text_length' in self.df.columns:
                self.df['spam_length_ratio'] = (
                    self.df['bart_spam_prob'] / (self.df['text_length'] + 1)
                )
        
        return features
    
    def extract_business_features(self) -> Dict:
        """Extract business-related features."""
        logger.info("🏢 Extracting business features")
        features = {}
        
        if 'placeId' in self.df.columns and 'stars' in self.df.columns:
            # Business rating statistics
            business_stats = self.df.groupby('placeId').agg({
                'stars': ['mean', 'std', 'count']
            }).reset_index()
            
            business_stats.columns = [
                'placeId', 'business_avg_rating',
                'business_rating_std', 'business_review_count'
            ]
            
            self.df = self.df.merge(business_stats, on='placeId', how='left')
            
            # Rating deviation from business average
            self.df['business_rating_deviation'] = abs(
                self.df['stars'] - self.df['business_avg_rating']
            )
            
            # BART-enhanced business features
            if 'bart_low_quality_risk' in self.df.columns:
                business_bart_stats = self.df.groupby('placeId').agg({
                    'bart_low_quality_risk': ['mean', 'max'],
                    'bart_spam_prob': 'mean'
                }).reset_index()
                
                business_bart_stats.columns = [
                    'placeId', 'business_avg_quality_risk', 
                    'business_max_quality_risk', 'business_avg_spam_prob'
                ]
                
                self.df = self.df.merge(business_bart_stats, on='placeId', how='left')
                
                # Quality risk deviation from business average
                self.df['business_quality_risk_deviation'] = abs(
                    self.df['bart_low_quality_risk'] - self.df['business_avg_quality_risk']
                )
        
        return features
    
    def detect_enhanced_anomalies(self) -> Dict:
        """Detect anomalies using enhanced features including BART."""
        logger.info("🔍 Detecting enhanced anomalies")
        anomalies = {}
        
        # Enhanced feature categories including BART features
        enhanced_feature_config = {
            'temporal_features': FEATURE_CONFIG['temporal_features'],
            'user_behavior_features': FEATURE_CONFIG['user_behavior_features'] + [
                'user_avg_quality_risk', 'user_max_quality_risk', 'user_avg_spam_prob'
            ],
            'content_features': FEATURE_CONFIG['content_features'] + [
                'bart_low_quality_risk', 'bart_spam_prob', 'bart_ad_prob',
                'content_quality_score', 'simplicity_confidence_ratio', 'spam_length_ratio'
            ],
            'business_features': FEATURE_CONFIG['business_features'] + [
                'business_avg_quality_risk', 'business_quality_risk_deviation'
            ],
            'bart_features': [
                'bart_confidence', 'bart_low_quality_risk', 'bart_high_quality_score',
                'bart_weighted_quality', 'bart_weighted_risk'
            ]
        }
        
        # Calculate anomaly scores for each category
        for category, features in enhanced_feature_config.items():
            if not features:
                continue
                
            for feature in features:
                if feature in self.df.columns:
                    # Simple z-score based anomaly detection
                    mean = self.df[feature].mean()
                    std = self.df[feature].std()
                    self.df[f'{feature}_anomaly'] = (
                        abs(self.df[feature] - mean) / (std + 1e-6)  # Avoid division by zero
                    )
        
        return anomalies
    
    def detect_anomalies_ml_enhanced(self):
        """Enhanced ML-based anomaly detection including BART features."""
        logger.info("🤖 Running enhanced ML-based anomaly detection")
        
        # Select numerical features including BART features
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Remove columns that are already anomaly flags
        numerical_cols = [col for col in numerical_cols if not col.endswith('_anomaly')]
        
        # Prioritize BART features
        bart_features = [col for col in numerical_cols if col.startswith('bart_')]
        other_features = [col for col in numerical_cols if not col.startswith('bart_')]
        
        # Combine with BART features prioritized
        selected_features = bart_features + other_features[:30]  # Limit to avoid curse of dimensionality
        
        if not selected_features:
            logger.warning("No numerical features available for enhanced ML anomaly detection")
            return
            
        try:
            # Prepare the feature matrix
            X = self.df[selected_features].fillna(0)
            
            # Initialize and fit the model with adjusted parameters for BART integration
            iso_forest = IsolationForest(
                n_estimators=200,  # Increased for better performance
                contamination=0.08,  # Slightly higher with BART features
                random_state=42,
                n_jobs=-1,
                max_features=min(len(selected_features), 15)  # Feature subsampling
            )
            
            # Get anomaly predictions (-1 for anomalies, 1 for normal)
            predictions = iso_forest.fit_predict(X)
            
            # Convert to binary (1 for anomaly, 0 for normal)
            self.df['enhanced_ml_anomaly'] = (predictions == -1).astype(int)
            
            # Get anomaly scores (the lower, the more anomalous)
            self.df['enhanced_ml_anomaly_score'] = iso_forest.decision_function(X) * -1  # Invert so higher = more anomalous
            
            logger.info(f"Detected {self.df['enhanced_ml_anomaly'].sum()} anomalies using enhanced ML")
            
            # Feature importance analysis
            feature_importance = {}
            for i, feature in enumerate(selected_features):
                feature_scores = self.df[self.df['enhanced_ml_anomaly'] == 1][feature]
                if len(feature_scores) > 0:
                    feature_importance[feature] = abs(feature_scores.mean())
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            logger.info("Top features contributing to enhanced anomalies:")
            for feat, importance in sorted_features[:10]:
                logger.info(f"  - {feat}: {importance:.4f}")
                
            return {
                'enhanced_ml_anomaly_count': int(self.df['enhanced_ml_anomaly'].sum()),
                'enhanced_ml_anomaly_rate': float(self.df['enhanced_ml_anomaly'].mean()),
                'top_features': dict(sorted_features[:10]),
                'bart_feature_count': len(bart_features),
                'total_feature_count': len(selected_features)
            }
        
        except Exception as e:
            logger.error(f"Error in enhanced ML anomaly detection: {str(e)}")
            return None
    
    def save_enhanced_results(self):
        """Save the enhanced analysis results."""
        logger.info("💾 Saving enhanced results")
        
        # Convert NumPy types to Python native types before saving to CSV
        for col in self.df.columns:
            if self.df[col].dtype == 'int64':
                self.df[col] = self.df[col].astype(int).fillna(0).astype(int)
            elif self.df[col].dtype == 'float64':
                self.df[col] = self.df[col].astype(float)
        
        # Save enhanced features to CSV
        enhanced_features_csv = self.output_dir / 'enhanced_metadata_features.csv'
        self.df.to_csv(enhanced_features_csv, index=False)
        
        # Generate and save enhanced anomaly report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_reviews': int(len(self.df)),
            'analysis_type': 'enhanced_with_bart',
            'bart_integration': True,
            'anomaly_summary': {},
            'top_anomalies': {},
            'bart_summary': {}
        }

        # Add BART summary if available
        if 'bart_classification' in self.df.columns:
            bart_counts = self.df['bart_classification'].value_counts()
            report['bart_summary'] = {
                'classification_distribution': bart_counts.to_dict(),
                'avg_confidence': float(self.df['bart_confidence'].mean()),
                'avg_quality_risk': float(self.df['bart_low_quality_risk'].mean()),
                'high_risk_reviews': int((self.df['bart_low_quality_risk'] > 0.7).sum())
            }

        # Add enhanced ML results to report if available
        if 'enhanced_ml_anomaly' in self.df.columns:
            # Determine available columns for top anomalies
            top_anomaly_columns = ['reviewId', 'reviewerId', 'enhanced_ml_anomaly_score']
            if 'bart_low_quality_risk' in self.df.columns:
                top_anomaly_columns.append('bart_low_quality_risk')
            
            report['enhanced_ml_anomaly_summary'] = {
                'anomaly_count': int(self.df['enhanced_ml_anomaly'].sum()),
                'anomaly_rate': float(self.df['enhanced_ml_anomaly'].mean()),
                'top_anomalies': self.df.nlargest(5, 'enhanced_ml_anomaly_score')[
                    top_anomaly_columns
                ].to_dict('records')
            }
        
        # Enhanced anomaly summary including BART features
        enhanced_feature_config = {
            'bart_features': [col for col in self.df.columns if col.startswith('bart_') and col.endswith('_anomaly')],
            'temporal_features': [f'{f}_anomaly' for f in FEATURE_CONFIG['temporal_features'] 
                                if f'{f}_anomaly' in self.df.columns],
            'user_behavior_features': [f'{f}_anomaly' for f in FEATURE_CONFIG['user_behavior_features'] 
                                     if f'{f}_anomaly' in self.df.columns],
            'content_features': [f'{f}_anomaly' for f in FEATURE_CONFIG['content_features'] 
                               if f'{f}_anomaly' in self.df.columns],
            'business_features': [f'{f}_anomaly' for f in FEATURE_CONFIG['business_features'] 
                                if f'{f}_anomaly' in self.df.columns]
        }
        
        for category, anomaly_cols in enhanced_feature_config.items():
            if anomaly_cols:
                report['anomaly_summary'][category] = {
                    'anomaly_score': float(self.df[anomaly_cols].max(axis=1).mean()),
                    'anomaly_count': int((self.df[anomaly_cols].max(axis=1) > 3).sum()),
                    'features': anomaly_cols
                }
        
        # Find top anomalies for each feature
        for col in [c for c in self.df.columns if '_anomaly' in c]:
            top_idx = self.df[col].idxmax()
            if pd.notna(top_idx):
                report['top_anomalies'][col] = {
                    'max_score': float(self.df.loc[top_idx, col]),
                    'review_id': str(self.df.loc[top_idx, 'reviewId']),
                    'reviewer_id': str(self.df.loc[top_idx, 'reviewerId']),
                    'place_id': str(self.df.loc[top_idx, 'placeId']),
                    'bart_quality_risk': float(self.df.loc[top_idx, 'bart_low_quality_risk']) if 'bart_low_quality_risk' in self.df.columns else None
                }
        
        # Save the enhanced report
        enhanced_report_path = self.output_dir / 'enhanced_anomaly_report.json'
        with open(enhanced_report_path, 'w') as f:
            # Convert all NumPy types to Python native types
            def convert(o):
                if isinstance(o, (np.integer, np.floating)):
                    return int(o) if isinstance(o, np.integer) else float(o)
                raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
                
            json.dump(report, f, indent=2, default=convert)
        
        logger.info(f"Enhanced results saved to {self.output_dir}")
        logger.info(f"Enhanced features CSV: {enhanced_features_csv}")
        logger.info(f"Enhanced report JSON: {enhanced_report_path}")
        
        return report
    
    def run_enhanced_analysis(self, bart_model_path=None):
        """Run the complete enhanced metadata analysis pipeline with BART integration."""
        try:
            # Load BART model if path provided
            if bart_model_path and self.bart_model is None:
                self._load_bart_model(bart_model_path)
            
            # Load and preprocess data
            self.load_data()
            
            # Extract BART features first (if available)
            if self.bart_model:
                logger.info("🤖 Extracting BART features...")
                self.extract_bart_features()
            else:
                logger.warning("⚠️ BART model not available, proceeding without BART features")
            
            # Extract other features
            logger.info("📊 Extracting metadata features...")
            self.extract_temporal_features()
            self.extract_user_behavior_features()
            self.extract_content_features()
            self.extract_business_features()
            
            # Detect anomalies using enhanced approach
            logger.info("🔍 Running enhanced anomaly detection...")
            rule_based_results = self.detect_enhanced_anomalies()
            
            # Detect anomalies using enhanced ML
            logger.info("🤖 Running enhanced ML-based anomaly detection...")
            enhanced_ml_results = self.detect_anomalies_ml_enhanced()
            
            # Save enhanced results
            results = self.save_enhanced_results()
            
            # Combine results
            if enhanced_ml_results:
                results['enhanced_ml_results'] = enhanced_ml_results
                
            logger.info("✅ Enhanced analysis completed successfully")
            
            # Log summary
            total_features = len([c for c in self.df.columns if not c.endswith('_anomaly')])
            bart_features = len([c for c in self.df.columns if c.startswith('bart_')])
            anomaly_features = len([c for c in self.df.columns if c.endswith('_anomaly')])
            
            logger.info(f"📈 Analysis Summary:")
            logger.info(f"   Total Features: {total_features}")
            logger.info(f"   BART Features: {bart_features}")
            logger.info(f"   Anomaly Scores: {anomaly_features}")
            
            if enhanced_ml_results:
                logger.info(f"   Anomalies Detected: {enhanced_ml_results['enhanced_ml_anomaly_count']}")
                logger.info(f"   Anomaly Rate: {enhanced_ml_results['enhanced_ml_anomaly_rate']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced analysis pipeline: {str(e)}")
            raise

def main():
    """Main entry point for the enhanced script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced metadata analysis with BART integration.')
    parser.add_argument('--bart-model', type=str, 
                       help='Path to trained BART model directory')
    
    args = parser.parse_args()
    
    # Look for BART model if not specified
    bart_model_path = args.bart_model
    if not bart_model_path:
        # Try to find trained BART model
        bart_dir = Path('../stage_1_bart_finetuning')
        model_dirs = [d for d in bart_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('enhanced_bart_review_classifier')]
        if model_dirs:
            bart_model_path = str(model_dirs[0])
            logger.info(f"🔍 Found BART model: {bart_model_path}")
    
    analyzer = EnhancedMetadataAnalyzer(bart_model_path)
    analyzer.run_enhanced_analysis()

if __name__ == "__main__":
    main()
