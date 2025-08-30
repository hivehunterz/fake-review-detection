#!/usr/bin/env python3
"""
Simple Stage 2 Metadata Analyzer Training
Creates a minimal working metadata analyzer model
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_simple_metadata_analyzer():
    """Create a simple metadata analyzer model"""
    
    # Paths
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "data" / "data_all_training.csv"
    models_path = base_path / "models"
    model_path = models_path / "metadata_analyzer.pkl"
    
    # Create models directory if it doesn't exist
    models_path.mkdir(exist_ok=True)
    
    logger.info("📊 Loading training data...")
    df = pd.read_csv(data_path)
    logger.info(f"📂 Loaded {len(df)} training samples")
    
    # Extract basic metadata features
    logger.info("🔍 Extracting basic metadata features...")
    
    # Convert timestamp to datetime for temporal features
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Text length features
    if 'reviewText' in df.columns:
        df['text_length'] = df['reviewText'].fillna('').astype(str).apply(len)
        df['word_count'] = df['reviewText'].fillna('').astype(str).apply(lambda x: len(x.split()))
    
    # Rating features
    rating_col = None
    if 'rating' in df.columns:
        df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')
        rating_col = 'rating_numeric'
    elif 'llm_rating' in df.columns:
        df['rating_numeric'] = pd.to_numeric(df['llm_rating'], errors='coerce')
        rating_col = 'rating_numeric'
    elif 'stars' in df.columns:
        # Handle string ratings like "4.0"
        df['rating_numeric'] = pd.to_numeric(df['stars'], errors='coerce')
        rating_col = 'rating_numeric'
    
    if rating_col is None:
        df['rating_numeric'] = 3.0  # Default rating
        rating_col = 'rating_numeric'
        
    # User behavior features (if available)
    if 'reviewerId' in df.columns and rating_col:
        user_stats = df.groupby('reviewerId').agg({
            rating_col: ['count', 'mean', 'std']
        }).fillna(0)
        user_stats.columns = ['user_review_count', 'user_avg_rating', 'user_rating_std']
        
        # Add text length to user stats if available
        if 'text_length' in df.columns:
            user_text_stats = df.groupby('reviewerId')['text_length'].mean().fillna(0)
            user_stats['user_avg_text_length'] = user_text_stats
        
        df = df.merge(user_stats, left_on='reviewerId', right_index=True, how='left')
    
    # Business features (if available)
    if 'placeId' in df.columns and rating_col:
        business_stats = df.groupby('placeId').agg({
            rating_col: ['count', 'mean', 'std']
        }).fillna(0)
        business_stats.columns = ['business_review_count', 'business_avg_rating', 'business_rating_std']
        df = df.merge(business_stats, left_on='placeId', right_index=True, how='left')
    
    # Select numeric features for training
    numeric_features = df.select_dtypes(include=[np.number]).fillna(0)
    
    # Remove any infinite values
    numeric_features = numeric_features.replace([np.inf, -np.inf], 0)
    
    logger.info(f"🔧 Training with {len(numeric_features.columns)} features on {len(numeric_features)} samples")
    
    # Train anomaly detection model
    logger.info("🤖 Training anomaly detection model...")
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(numeric_features)
    
    # Train isolation forest
    model = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_estimators=100,
        n_jobs=-1
    )
    model.fit(features_scaled)
    
    # Test predictions
    anomaly_scores = model.decision_function(features_scaled)
    anomalies = model.predict(features_scaled)
    n_anomalies = (anomalies == -1).sum()
    
    logger.info(f"📈 Detected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.1f}%)")
    
    # Save model and metadata
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': numeric_features.columns.tolist(),
        'n_samples_trained': len(df),
        'n_features': len(numeric_features.columns),
        'contamination': 0.1,
        'model_type': 'IsolationForest'
    }
    
    joblib.dump(model_data, model_path)
    
    logger.info(f"✅ Model saved to: {model_path}")
    logger.info(f"💾 Model size: {model_path.stat().st_size / 1024:.1f} KB")
    
    return True

if __name__ == "__main__":
    success = create_simple_metadata_analyzer()
    if success:
        print("✅ Simple metadata analyzer training completed successfully!")
    else:
        print("❌ Training failed!")
