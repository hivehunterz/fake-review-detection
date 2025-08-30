"""
Advanced Fusion Model for Multi-Stage Spam Detection Pipeline
Combines BART, Enhanced Metadata, and Relevancy scores using machine learning
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import logging

class AdvancedFusionModel:
    """
    Advanced machine learning model for fusing multi-stage outputs
    """
    
    def __init__(self, model_type='gradient_boosting', save_path='fusion_model.pkl'):
        self.model_type = model_type
        self.save_path = save_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'p_bad', 'enhanced_prob', 'relevancy_score', 'is_relevant',
            'bart_confidence', 'enhanced_confidence', 'class_spam', 'class_advertisement',
            'class_irrelevant', 'class_fake_rant', 'class_inappropriate',
            'p_bad_squared', 'enhanced_prob_squared', 'relevancy_squared',
            'p_bad_enhanced_interaction', 'p_bad_relevancy_interaction',
            'enhanced_relevancy_interaction', 'confidence_weighted_p_bad',
            'confidence_weighted_enhanced', 'irrelevance_penalty'
        ]
        self.target_mapping = {
            'genuine': 0,
            'suspicious': 1, 
            'low-quality': 2,
            'high-confidence-spam': 3
        }
        self.reverse_mapping = {v: k for k, v in self.target_mapping.items()}
        
        # Initialize model based on type
        self._initialize_model()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_model(self):
        """Initialize the fusion model based on type"""
        if self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'neural_network':
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                multi_class='multinomial'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def engineer_features(self, p_bad_scores, enhanced_probs, relevancy_scores, 
                         is_relevant, bart_confidences, all_probabilities):
        """
        Engineer advanced features for fusion model
        """
        features = []
        
        for i in range(len(p_bad_scores)):
            p_bad = p_bad_scores[i]
            enhanced_prob = enhanced_probs[i]
            relevancy = relevancy_scores[i]
            relevant = int(is_relevant[i])
            bart_conf = bart_confidences[i]
            probs = all_probabilities[i]
            
            # Basic features
            row = [
                p_bad,
                enhanced_prob, 
                relevancy,
                relevant,
                bart_conf,
                enhanced_prob,  # enhanced_confidence placeholder
            ]
            
            # Individual class probabilities (7-class BART)
            labels = ['genuine_positive', 'genuine_negative', 'spam', 'advertisement', 
                     'irrelevant', 'fake_rant', 'inappropriate']
            
            # Extract specific bad class probabilities
            spam_prob = probs[2] if len(probs) > 2 else 0
            ad_prob = probs[3] if len(probs) > 3 else 0
            irrelevant_prob = probs[4] if len(probs) > 4 else 0
            fake_rant_prob = probs[5] if len(probs) > 5 else 0
            inappropriate_prob = probs[6] if len(probs) > 6 else 0
            
            row.extend([spam_prob, ad_prob, irrelevant_prob, fake_rant_prob, inappropriate_prob])
            
            # Polynomial features
            row.extend([
                p_bad ** 2,
                enhanced_prob ** 2,
                relevancy ** 2
            ])
            
            # Interaction features
            row.extend([
                p_bad * enhanced_prob,
                p_bad * relevancy,
                enhanced_prob * relevancy,
                p_bad * bart_conf,
                enhanced_prob * bart_conf,
                (1 - relevancy) * 2 if not relevant else 0  # irrelevance penalty
            ])
            
            features.append(row)
        
        return np.array(features)
    
    def create_training_labels(self, fusion_results):
        """
        Convert fusion results to training labels
        """
        labels = []
        for result in fusion_results:
            prediction = result['prediction']
            labels.append(self.target_mapping.get(prediction, 0))
        return np.array(labels)
    
    def train(self, training_data, validation_split=0.2):
        """
        Train the fusion model on historical data
        """
        self.logger.info(f"🚀 Training {self.model_type} fusion model...")
        
        # Prepare features
        X = self.engineer_features(
            training_data['p_bad_scores'],
            training_data['enhanced_probs'],
            training_data['relevancy_scores'],
            training_data['is_relevant'],
            training_data['bart_confidences'],
            training_data['all_probabilities']
        )
        
        # Prepare labels
        y = self.create_training_labels(training_data['fusion_results'])
        
        self.logger.info(f"📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"📊 Label distribution: {np.bincount(y)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Validate
        val_predictions = self.model.predict(X_val)
        val_probs = self.model.predict_proba(X_val)
        
        self.logger.info(f"✅ Model trained successfully!")
        self.logger.info(f"📊 Validation accuracy: {np.mean(val_predictions == y_val):.3f}")
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            top_features = np.argsort(importance)[-10:][::-1]
            
            self.logger.info("🔍 Top 10 most important features:")
            for i, feat_idx in enumerate(top_features):
                feat_name = self.feature_names[feat_idx] if feat_idx < len(self.feature_names) else f"feature_{feat_idx}"
                self.logger.info(f"  {i+1}. {feat_name}: {importance[feat_idx]:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        self.logger.info(f"📊 Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return {
            'train_accuracy': np.mean(self.model.predict(X_train) == y_train),
            'val_accuracy': np.mean(val_predictions == y_val),
            'cv_accuracy': cv_scores.mean(),
            'feature_importance': importance if hasattr(self.model, 'feature_importances_') else None
        }
    
    def predict_fusion(self, p_bad, enhanced_prob, relevancy_score, is_relevant, 
                      bart_confidence, all_probabilities):
        """
        Predict fusion result using trained model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Engineer features for single prediction
        features = self.engineer_features(
            [p_bad], [enhanced_prob], [relevancy_score], [is_relevant],
            [bart_confidence], [all_probabilities]
        )
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Convert to readable format
        prediction_name = self.reverse_mapping[prediction]
        confidence = probabilities.max()
        
        # Determine routing
        routing_map = {
            'genuine': 'automatic-approval',
            'suspicious': 'requires-manual-verification', 
            'low-quality': 'requires-manual-verification',
            'high-confidence-spam': 'automatic-rejection'
        }
        
        return {
            'prediction': prediction_name,
            'confidence': confidence,
            'routing': routing_map[prediction_name],
            'fusion_score': probabilities[3],  # Score for high-confidence-spam
            'all_probabilities': probabilities.tolist(),
            'model_type': self.model_type
        }
    
    def batch_predict(self, p_bad_scores, enhanced_probs, relevancy_scores, 
                     is_relevant, bart_confidences, all_probabilities):
        """
        Batch prediction for multiple samples
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Engineer features
        features = self.engineer_features(
            p_bad_scores, enhanced_probs, relevancy_scores, is_relevant,
            bart_confidences, all_probabilities
        )
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # Convert to readable format
        results = []
        for i in range(len(predictions)):
            pred = predictions[i]
            probs = probabilities[i]
            
            prediction_name = self.reverse_mapping[pred]
            confidence = probs.max()
            
            routing_map = {
                'genuine': 'automatic-approval',
                'suspicious': 'requires-manual-verification', 
                'low-quality': 'requires-manual-verification',
                'high-confidence-spam': 'automatic-rejection'
            }
            
            results.append({
                'prediction': prediction_name,
                'confidence': confidence,
                'routing': routing_map[prediction_name],
                'fusion_score': probs[3],  # Score for high-confidence-spam
                'all_probabilities': probs.tolist(),
                'model_type': self.model_type
            })
        
        return results
    
    def save_model(self, path=None):
        """Save the trained model"""
        save_path = path or self.save_path
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'target_mapping': self.target_mapping,
            'reverse_mapping': self.reverse_mapping
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"💾 Model saved to: {save_path}")
    
    def load_model(self, path=None):
        """Load a trained model"""
        load_path = path or self.save_path
        
        if not os.path.exists(load_path):
            self.logger.warning(f"❌ Model file not found: {load_path}")
            return False
        
        try:
            with open(load_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_type = model_data['model_type']
            self.feature_names = model_data['feature_names']
            self.target_mapping = model_data['target_mapping']
            self.reverse_mapping = model_data['reverse_mapping']
            
            self.logger.info(f"✅ Model loaded from: {load_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            return False
    
    def hyperparameter_tuning(self, training_data):
        """
        Perform hyperparameter tuning
        """
        self.logger.info("🔧 Starting hyperparameter tuning...")
        
        # Prepare data
        X = self.engineer_features(
            training_data['p_bad_scores'],
            training_data['enhanced_probs'],
            training_data['relevancy_scores'],
            training_data['is_relevant'],
            training_data['bart_confidences'],
            training_data['all_probabilities']
        )
        y = self.create_training_labels(training_data['fusion_results'])
        X_scaled = self.scaler.fit_transform(X)
        
        # Define parameter grids
        param_grids = {
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [4, 6, 8]
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 10, 14],
                'min_samples_split': [2, 5, 10]
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 75)],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
        }
        
        if self.model_type in param_grids:
            param_grid = param_grids[self.model_type]
            
            grid_search = GridSearchCV(
                self.model,
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_scaled, y)
            
            self.model = grid_search.best_estimator_
            self.logger.info(f"🎯 Best parameters: {grid_search.best_params_}")
            self.logger.info(f"🎯 Best CV score: {grid_search.best_score_:.3f}")
            
            return grid_search.best_params_, grid_search.best_score_
        
        return None, None

def create_training_data_from_pipeline_results(pipeline_results_csv):
    """
    Create training data from existing pipeline results
    """
    df = pd.read_csv(pipeline_results_csv)
    
    training_data = {
        'p_bad_scores': df['stage1_p_bad'].tolist(),
        'enhanced_probs': df['stage2_enhanced_prob'].tolist(),
        'relevancy_scores': df['stage3_relevancy'].tolist(),
        'is_relevant': df['stage3_relevant'].tolist(),
        'bart_confidences': df['stage1_confidence'].tolist(),
        'all_probabilities': [[0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.01]] * len(df),  # Placeholder
        'fusion_results': [
            {'prediction': pred} for pred in df['final_prediction'].tolist()
        ]
    }
    
    return training_data

if __name__ == "__main__":
    # Example usage
    print("🚀 Advanced Fusion Model for Multi-Stage Pipeline")
    print("=" * 60)
    
    # Create sample training data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'p_bad_scores': np.random.random(n_samples),
        'enhanced_probs': np.random.random(n_samples),
        'relevancy_scores': np.random.random(n_samples),
        'is_relevant': np.random.choice([True, False], n_samples),
        'bart_confidences': np.random.random(n_samples),
        'all_probabilities': [np.random.dirichlet([1]*7).tolist() for _ in range(n_samples)],
        'fusion_results': [
            {'prediction': np.random.choice(['genuine', 'suspicious', 'low-quality', 'high-confidence-spam'])}
            for _ in range(n_samples)
        ]
    }
    
    # Train model
    fusion_model = AdvancedFusionModel(model_type='gradient_boosting')
    training_results = fusion_model.train(sample_data)
    
    print(f"Training completed with accuracy: {training_results['val_accuracy']:.3f}")
    
    # Test prediction
    test_result = fusion_model.predict_fusion(
        p_bad=0.7,
        enhanced_prob=0.6,
        relevancy_score=0.3,
        is_relevant=False,
        bart_confidence=0.8,
        all_probabilities=[0.1, 0.2, 0.4, 0.1, 0.1, 0.05, 0.05]
    )
    
    print(f"Test prediction: {test_result}")
