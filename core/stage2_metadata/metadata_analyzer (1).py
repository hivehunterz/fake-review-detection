"""
Metadata Anomaly Detection for Fake Review Detection

This script analyzes review metadata to detect potential fake reviews based on anomalies
in temporal patterns, user behavior, content, geographic data, and business context.
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

from .config import (
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
        logging.FileHandler(Path(OUTPUT_DIR) / 'metadata_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MetadataAnalyzer:
    """Main class for detecting anomalies in review metadata."""
    
    def __init__(self):
        """Initialize the analyzer with empty data structures."""
        self.df = None
        self.features = {}
        self.anomaly_scores = {}
        self.output_dir = Path(OUTPUT_DIR)
    
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
        logger.info("Extracting temporal features")
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
        logger.info("Extracting user behavior features")
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
                
                logger.info(f"Extracted user behavior features for {len(self.df)} reviews")
                
            except Exception as e:
                logger.error(f"Error extracting user behavior features: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
        else:
            logger.warning("'reviewerId' column not found. Cannot extract user behavior features.")
        
        return features
    
    def extract_content_features(self) -> Dict:
        """Extract content-based features."""
        logger.info("Extracting content features")
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
            
            # Language consistency (if available)
            if 'language' in self.df.columns and 'originalLanguage' in self.df.columns:
                self.df['language_match'] = (
                    self.df['language'] == self.df['originalLanguage']
                ).astype(int)
        
        return features
    
    def extract_geographic_features(self) -> Dict:
        """Extract geographic features."""
        logger.info("Extracting geographic features")
        features = {}
        
        # Basic location features
        if 'location' in self.df.columns:
            # Extract lat/lng from location string if needed
            pass
            
        return features
    
    def extract_business_features(self) -> Dict:
        """Extract business-related features."""
        logger.info("Extracting business features")
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
        
        return features
    
    def detect_anomalies(self) -> Dict:
        """Detect anomalies across all feature categories."""
        logger.info("Detecting anomalies")
        anomalies = {}
        
        # Calculate anomaly scores for each category
        for category in FEATURE_CONFIG.keys():
            features = FEATURE_CONFIG[category]
            if not features:
                continue
                
            # Simple threshold-based anomaly detection
            # In a real implementation, you might use more sophisticated methods
            # like Isolation Forest, One-Class SVM, or autoencoders
            for feature in features:
                if feature in self.df.columns:
                    # Simple z-score based anomaly detection
                    mean = self.df[feature].mean()
                    std = self.df[feature].std()
                    self.df[f'{feature}_anomaly'] = (
                        abs(self.df[feature] - mean) / (std + 1e-6)  # Avoid division by zero
                    )
        
        return anomalies

    def detect_anomalies_ml(self):
        logger.info("Running ML-based anomaly detection")
        
        # Select only numerical features for ML
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Remove columns that are already anomaly flags
        numerical_cols = [col for col in numerical_cols if not col.endswith('_anomaly')]
        
        if not numerical_cols:
            logger.warning("No numerical features available for ML anomaly detection")
            return
            
        try:
            # Prepare the feature matrix
            X = self.df[numerical_cols].fillna(0)
            
            # Initialize and fit the model
            iso_forest = IsolationForest(
                n_estimators=100,
                contamination=0.05,  # Expected proportion of anomalies
                random_state=42,
                n_jobs=-1
            )
            
            # Get anomaly predictions (-1 for anomalies, 1 for normal)
            predictions = iso_forest.fit_predict(X)
            
            # Convert to binary (1 for anomaly, 0 for normal)
            self.df['ml_anomaly'] = (predictions == -1).astype(int)
            
            # Get anomaly scores (the lower, the more anomalous)
            self.df['ml_anomaly_score'] = iso_forest.decision_function(X) * -1  # Invert so higher = more anomalous
            
            logger.info(f"Detected {self.df['ml_anomaly'].sum()} anomalies using ML")
            
            # Get feature importance
            importances = pd.Series(
                iso_forest.feature_importances_ if hasattr(iso_forest, 'feature_importances_') 
                else iso_forest.score_samples(X).mean(axis=0),
                index=numerical_cols
            ).sort_values(ascending=False)
            
            logger.info("Top features contributing to anomalies:")
            for feat, importance in importances.head(5).items():
                logger.info(f"  - {feat}: {importance:.4f}")
                
            return {
                'ml_anomaly_count': int(self.df['ml_anomaly'].sum()),
                'ml_anomaly_rate': float(self.df['ml_anomaly'].mean()),
                'top_features': importances.head(5).to_dict()
            }
        
        except Exception as e:
            logger.error(f"Error in ML anomaly detection: {str(e)}")
            return None
    
    def save_results(self):
        """Save the analysis results."""
        logger.info("Saving results")
        
        # Convert NumPy types to Python native types before saving to CSV
        for col in self.df.columns:
            if self.df[col].dtype == 'int64':
                self.df[col] = self.df[col].astype(int).fillna(0).astype(int)
            elif self.df[col].dtype == 'float64':
                self.df[col] = self.df[col].astype(float)
        
        # Save features to CSV
        self.df.to_csv(FEATURES_CSV, index=False)
        
        # Generate and save anomaly report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_reviews': int(len(self.df)),
            'anomaly_summary': {},
            'top_anomalies': {}
        }

        # Add ML results to report if available
        if 'ml_anomaly' in self.df.columns:
            report['ml_anomaly_summary'] = {
                'anomaly_count': int(self.df['ml_anomaly'].sum()),
                'anomaly_rate': float(self.df['ml_anomaly'].mean()),
                'top_anomalies': self.df.nlargest(5, 'ml_anomaly_score')[
                    ['reviewId', 'reviewerId', 'ml_anomaly_score']
            ].to_dict('records')
        }
        
        # Add anomaly summary
        for category in FEATURE_CONFIG.keys():
            anomaly_cols = [f'{f}_anomaly' for f in FEATURE_CONFIG[category] 
                          if f'{f}_anomaly' in self.df.columns]
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
                    'place_id': str(self.df.loc[top_idx, 'placeId'])
                }
        
        # Save the report with proper type conversion
        with open(ANOMALY_REPORT, 'w') as f:
            # Convert all NumPy types to Python native types
            def convert(o):
                if isinstance(o, (np.integer, np.floating)):
                    return int(o) if isinstance(o, np.integer) else float(o)
                raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
                
            json.dump(report, f, indent=2, default=convert)
        
        logger.info(f"Results saved to {self.output_dir}")
        return report
    
    def run(self):
        """Run the complete metadata analysis pipeline."""
        try:
            # Load and preprocess data
            self.load_data()
            
            # Extract features
            logger.info("Extracting features...")
            self.extract_temporal_features()
            self._add_rolling_window_features()
            self.extract_user_behavior_features()
            self.extract_content_features()
            self.extract_geographic_features()
            self.extract_business_features()
            
            # Detect anomalies using rule-based approach
            logger.info("Running rule-based anomaly detection...")
            rule_based_results = self.detect_anomalies()
            
            # Detect anomalies using ML
            logger.info("Running ML-based anomaly detection...")
            ml_results = self.detect_anomalies_ml()
            
            # Save results
            results = self.save_results()
            
            # Combine results
            if ml_results:
                results['ml_results'] = ml_results
                
            logger.info("Analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {str(e)}")
            raise

def main():
    """Main entry point for the script."""
    analyzer = MetadataAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()
