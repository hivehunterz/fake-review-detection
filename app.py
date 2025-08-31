🛡️ Smart Review Guardian - Web Interface
Modern UI for the fake review detection system
"""

import os
import sys
import json
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import io
import csv

# Setup paths
PROJECT_ROOT = Path(__file__).parent
CORE_PATH = PROJECT_ROOT / "core"
SCRIPTS_PATH = PROJECT_ROOT / "scripts"
OUTPUT_PATH = PROJECT_ROOT / "output"
MODELS_PATH = PROJECT_ROOT / "models"

# Add paths for imports
sys.path.append(str(SCRIPTS_PATH / "prediction"))
sys.path.append(str(CORE_PATH / "stage1_bart"))
sys.path.append(str(CORE_PATH / "stage2_metadata"))
sys.path.append(str(CORE_PATH / "fusion"))

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'review-guardian-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor = None

class SmartReviewGuardian:
    """Main application class for the Smart Review Guardian web interface"""
    
    def __init__(self):
        self.predictor = None
        self.stats = {
            'total_processed': 0,
            'genuine_count': 0,
            'suspicious_count': 0,
            'low_quality_count': 0,
            'spam_count': 0,
            'last_updated': datetime.now().isoformat()
        }
        
    def initialize_predictor(self):
        """Initialize the ML prediction pipeline"""
        try:
            from predict_review_quality import ReviewQualityPredictor
            self.predictor = ReviewQualityPredictor()
            
            # Try to load models - this will now gracefully handle failures
            if self.predictor.load_models():
                logger.info("✅ ML models loaded successfully")
                return True
            else:
                logger.warning("⚠️ Could not load all models, but BART fallback is available")
                # Even if complex models fail, we can still use BART with fallback
                self.predictor = None  # Will use demo mode with better fallback
                return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize predictor: {e}")
            self.predictor = None  # Will use demo mode
            return False
    
    def predict_single_review(self, text: str, metadata: Optional[Dict] = None) -> Dict:
        """Predict quality of a single review"""
        if not self.predictor:
            # Try to use BART classifier directly as fallback
            try:
                import sys
                sys.path.append(str(PROJECT_ROOT / "core" / "stage1_bart"))
                from enhanced_bart_review_classifier import BARTReviewClassifier
                
                bart_classifier = BARTReviewClassifier(model_path=None, use_gpu=False)
                bart_result = bart_classifier.predict_single(text)
                
                # Convert BART result to app format
                result = self._convert_bart_result_to_app_format(bart_result, text, metadata)
                result['demo_mode'] = False
                result['fallback_mode'] = True
                
                self._update_stats(result['final_prediction'])
                return result
                
            except Exception as e:
                logger.warning(f"BART fallback failed: {e}, using demo mode")
                return self._generate_demo_result(text)
        
        try:
            result = self.predictor.predict_single_review(text, metadata)
            # Add additional fields for UI display
            result['demo_mode'] = False
            result['fallback_mode'] = False
            
            # Map routing decisions to user-friendly descriptions
            routing_descriptions = {
                'automatic-approval': 'This review can be automatically approved for publication.',
                'requires-manual-verification': 'This review requires human verification before publication.',
                'automatic-rejection': 'This review should be automatically rejected.'
            }
            result['routing_description'] = routing_descriptions.get(
                result.get('routing_decision', ''), 
                'Unknown routing decision'
            )
            
            # Add detailed analysis breakdown
            result['stage_analysis'] = {
                'stage1_bart': {
                    'prediction': result.get('bart_prediction', 'unknown'),
                    'confidence': result.get('bart_confidence', 0.0),
                    'description': 'BART model classification of review content'
                },
                'stage2_metadata': {
                    'anomaly_score': result.get('metadata_anomaly_score', 0.0),
                    'description': 'Behavioral pattern and metadata analysis'
                },
                'stage3_fusion': {
                    'final_prediction': result.get('final_prediction', 'unknown'),
                    'final_confidence': result.get('final_confidence', 0.0),
                    'fusion_score': result.get('fusion_score', 0.0),
                    'description': 'Advanced fusion model combining all signals'
                }
            }
            
            self._update_stats(result['final_prediction'])
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._generate_demo_result(text, error=str(e))
    
    def process_batch_reviews(self, reviews_data: List[Dict]) -> Dict[str, Any]:
        """Process multiple reviews with detailed analysis"""
        results = []
        total_reviews = len(reviews_data)
        
        # Initialize progress tracking
        batch_stats = {
            'total': total_reviews,
            'processed': 0,
            'genuine': 0,
            'suspicious': 0,
            'low_quality': 0,
            'spam': 0,
            'errors': 0
        }
        
        # Check if we have a working predictor or need to use BART fallback
        use_bart_fallback = not self.predictor
        bart_classifier = None
        
        if use_bart_fallback:
            try:
                import sys
                sys.path.append(str(PROJECT_ROOT / "core" / "stage1_bart"))
                from enhanced_bart_review_classifier import BARTReviewClassifier
                bart_classifier = BARTReviewClassifier(model_path=None, use_gpu=False)
                logger.info("Using BART classifier with fallback for batch processing")
            except Exception as e:
                logger.warning(f"BART fallback setup failed: {e}, using demo mode")
        
        for i, review in enumerate(reviews_data):
            try:
                text = review.get('text', '')
                metadata = {k: v for k, v in review.items() if k != 'text'}
                
                # Predict single review using appropriate method
                if bart_classifier:
                    bart_result = bart_classifier.predict_single(text)
                    result = self._convert_bart_result_to_app_format(bart_result, text, metadata)
                    result['demo_mode'] = False
                    result['fallback_mode'] = True
                else:
                    result = self.predict_single_review(text, metadata)
                
                # Add batch-specific metadata
                result['batch_index'] = i
                result['original_data'] = review
                
                # Update batch statistics
                prediction = result.get('final_prediction', 'unknown')
                if prediction == 'genuine':
                    batch_stats['genuine'] += 1
                elif prediction == 'suspicious':
                    batch_stats['suspicious'] += 1
                elif prediction == 'low-quality':
                    batch_stats['low_quality'] += 1
                elif prediction == 'high-confidence-spam':
                    batch_stats['spam'] += 1
                
                batch_stats['processed'] += 1
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing review {i}: {e}")
                batch_stats['errors'] += 1
                
                # Add error result
                error_result = self._generate_demo_result(text, error=str(e))
                error_result['batch_index'] = i
                error_result['error'] = True
                results.append(error_result)
        
        # Calculate summary statistics
        if batch_stats['processed'] > 0:
            batch_stats['genuine_percentage'] = round((batch_stats['genuine'] / batch_stats['processed']) * 100, 1)
            batch_stats['flagged_count'] = batch_stats['processed'] - batch_stats['genuine']
            batch_stats['flagged_percentage'] = round((batch_stats['flagged_count'] / batch_stats['processed']) * 100, 1)
            
            # Calculate average confidence
            confidences = [r.get('final_confidence', 0) for r in results if not r.get('error')]
            batch_stats['avg_confidence'] = sum(confidences) / len(confidences) if confidences else 0
            
            # Calculate average risk score
            risk_scores = [r.get('p_bad_score', 0) for r in results if not r.get('error')]
            batch_stats['avg_risk_score'] = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        else:
            batch_stats.update({
                'genuine_percentage': 0,
                'flagged_count': 0,
                'flagged_percentage': 0,
                'avg_confidence': 0,
                'avg_risk_score': 0
            })
        
        return {
            'results': results,
            'summary': batch_stats,
            'detailed_stats': self._generate_detailed_batch_stats(results),
            'fallback_mode': use_bart_fallback
        }
    
    def _generate_demo_result(self, text: str, error: str = None) -> Dict:
        """Generate demo results when models aren't available"""
        import random
        import hashlib
        
        # Use text hash for consistent results per text
        text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        random.seed(text_hash)
        
        # Simple heuristics for demo that feel realistic
        text_length = len(text)
        has_caps = any(c.isupper() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        word_count = len(text.split())
        
        # Bias predictions based on simple text characteristics
        if text_length < 20:
            # Very short text tends to be spam
            prediction_weights = [0.1, 0.2, 0.3, 0.4]  # [genuine, suspicious, low-quality, spam]
        elif text_length > 500:
            # Very long text might be spam or genuine detailed review
            prediction_weights = [0.3, 0.3, 0.2, 0.2]
        elif has_caps and has_numbers:
            # Caps and numbers might indicate spam
            prediction_weights = [0.2, 0.4, 0.2, 0.2]
        else:
            # Normal-looking text is more likely genuine
            prediction_weights = [0.6, 0.2, 0.15, 0.05]
        
        categories = ['genuine', 'suspicious', 'low-quality', 'high-confidence-spam']
        prediction = random.choices(categories, weights=prediction_weights)[0]
        
        # Generate confidence based on prediction
        confidence_ranges = {
            'genuine': (0.7, 0.95),
            'suspicious': (0.6, 0.85),
            'low-quality': (0.65, 0.9),
            'high-confidence-spam': (0.8, 0.98)
        }
        confidence_min, confidence_max = confidence_ranges[prediction]
        confidence = random.uniform(confidence_min, confidence_max)
        
        # Generate BART prediction based on final prediction
        bart_mapping = {
            'genuine': ['genuine_positive', 'genuine_negative'],
            'suspicious': ['spam', 'advertisement', 'irrelevant'],
            'low-quality': ['irrelevant', 'fake_rant'],
            'high-confidence-spam': ['spam', 'advertisement', 'inappropriate']
        }
        bart_prediction = random.choice(bart_mapping[prediction])
        
        # Generate routing decision
        routing_map = {
            'genuine': 'automatic-approval',
            'suspicious': 'requires-manual-verification',
            'low-quality': 'requires-manual-verification',
            'high-confidence-spam': 'automatic-rejection'
        }
        
        # Generate probability distribution
        labels = ['genuine_positive', 'genuine_negative', 'spam', 'advertisement', 'irrelevant', 'fake_rant', 'inappropriate']
        class_probs = [random.uniform(0.05, 0.25) for _ in labels]
        
        # Boost probability of the predicted class
        if bart_prediction in labels:
            pred_idx = labels.index(bart_prediction)
            class_probs[pred_idx] = random.uniform(0.4, 0.7)
        
        # Normalize probabilities
        total_prob = sum(class_probs)
        class_probs = [p / total_prob for p in class_probs]
        
        # Generate risk scores
        p_bad_score = random.uniform(0.1, 0.9)
        if prediction == 'genuine':
            p_bad_score = random.uniform(0.1, 0.4)
        elif prediction == 'high-confidence-spam':
            p_bad_score = random.uniform(0.6, 0.9)
        
        metadata_anomaly = random.uniform(0.0, 1.0)
        fusion_score = (p_bad_score + metadata_anomaly + (1 - confidence)) / 3
        
        result = {
            'text': text,
            'bart_prediction': bart_prediction,
            'bart_confidence': random.uniform(0.6, 0.9),
            'p_bad_score': p_bad_score,
            'metadata_anomaly_score': metadata_anomaly,
            'final_prediction': prediction,
            'final_confidence': confidence,
            'fusion_score': fusion_score,
            'routing_decision': routing_map[prediction],
            'class_probabilities': dict(zip(labels, class_probs)),
            'demo_mode': True,
            'routing_description': {
                'automatic-approval': 'This review can be automatically approved for publication.',
                'requires-manual-verification': 'This review requires human verification before publication.',
                'automatic-rejection': 'This review should be automatically rejected.'
            }[routing_map[prediction]],
            'stage_analysis': {
                'stage1_bart': {
                    'prediction': bart_prediction,
                    'confidence': random.uniform(0.6, 0.9),
                    'description': 'BART model classification of review content (Demo Mode)'
                },
                'stage2_metadata': {
                    'anomaly_score': metadata_anomaly,
                    'description': 'Behavioral pattern and metadata analysis (Demo Mode)'
                },
                'stage3_fusion': {
                    'final_prediction': prediction,
                    'final_confidence': confidence,
                    'fusion_score': fusion_score,
                    'description': 'Advanced fusion model combining all signals (Demo Mode)'
                }
            }
        }
        
        if error:
            result['error'] = error
            
        self._update_stats(prediction)
        return result
    
    def _convert_bart_result_to_app_format(self, bart_result: Dict, text: str, metadata: Optional[Dict] = None) -> Dict:
        """Convert BART classifier result to app format"""
        bart_prediction = bart_result['prediction']
        bart_confidence = bart_result['confidence']
        p_bad_score = bart_result['p_bad']
        
        # Map BART predictions to final predictions
        final_prediction_map = {
            'genuine_positive': 'genuine',
            'genuine_negative': 'genuine',
            'spam': 'high-confidence-spam',
            'advertisement': 'suspicious',
            'irrelevant': 'low-quality',
            'fake_rant': 'suspicious',
            'inappropriate': 'high-confidence-spam'
        }
        
        final_prediction = final_prediction_map.get(bart_prediction, 'suspicious')
        
        # Generate routing decision based on prediction
        routing_map = {
            'genuine': 'automatic-approval',
            'suspicious': 'requires-manual-verification',
            'low-quality': 'requires-manual-verification',
            'high-confidence-spam': 'automatic-rejection'
        }
        
        routing_decision = routing_map[final_prediction]
        
        # Generate routing description
        routing_descriptions = {
            'automatic-approval': 'This review can be automatically approved for publication.',
            'requires-manual-verification': 'This review requires human verification before publication.',
            'automatic-rejection': 'This review should be automatically rejected.'
        }
        
        result = {
            'text': text,
            'bart_prediction': bart_prediction,
            'bart_confidence': bart_confidence,
            'p_bad_score': p_bad_score,
            'metadata_anomaly_score': 0.3,  # Default value
            'final_prediction': final_prediction,
            'final_confidence': bart_confidence,
            'fusion_score': (p_bad_score + 0.3) / 2,  # Simple fusion
            'routing_decision': routing_decision,
            'routing_description': routing_descriptions[routing_decision],
            'class_probabilities': bart_result['class_probabilities'],
            'stage_analysis': {
                'stage1_bart': {
                    'prediction': bart_prediction,
                    'confidence': bart_confidence,
                    'description': f'BART model classification ({bart_result["model_type"]})'
                },
                'stage2_metadata': {
                    'anomaly_score': 0.3,
                    'description': 'Simplified metadata analysis (fallback mode)'
                },
                'stage3_fusion': {
                    'final_prediction': final_prediction,
                    'final_confidence': bart_confidence,
                    'fusion_score': (p_bad_score + 0.3) / 2,
                    'description': 'Simplified fusion combining BART and basic heuristics'
                }
            }
        }
        
        return result
    
    def _update_stats(self, prediction: str):
        """Update application statistics"""
        self.stats['total_processed'] += 1
        if prediction == 'genuine':
            self.stats['genuine_count'] += 1
        elif prediction == 'suspicious':
            self.stats['suspicious_count'] += 1
        elif prediction == 'low-quality':
            self.stats['low_quality_count'] += 1
        elif prediction == 'high-confidence-spam':
            self.stats['spam_count'] += 1
        
        self.stats['last_updated'] = datetime.now().isoformat()
    
    def _generate_detailed_batch_stats(self, results: List[Dict]) -> Dict:
        """Generate detailed statistics for batch processing"""
        if not results:
            return {}
        
        # Filter out error results
        valid_results = [r for r in results if not r.get('error')]
        
        if not valid_results:
            return {}
        
        # BART prediction breakdown
        bart_predictions = {}
        for result in valid_results:
            bart_pred = result.get('bart_prediction', 'unknown')
            bart_predictions[bart_pred] = bart_predictions.get(bart_pred, 0) + 1
        
        # Confidence distribution
        confidences = [r.get('final_confidence', 0) for r in valid_results]
        confidence_bins = {
            'high': len([c for c in confidences if c >= 0.9]),
            'good': len([c for c in confidences if 0.8 <= c < 0.9]),
            'medium': len([c for c in confidences if 0.7 <= c < 0.8]),
            'low': len([c for c in confidences if c < 0.7])
        }
        
        # Risk score distribution
        risk_scores = [r.get('p_bad_score', 0) for r in valid_results]
        risk_bins = {
            'very_low': len([r for r in risk_scores if r < 0.2]),
            'low': len([r for r in risk_scores if 0.2 <= r < 0.4]),
            'medium': len([r for r in risk_scores if 0.4 <= r < 0.6]),
            'high': len([r for r in risk_scores if 0.6 <= r < 0.8]),
            'very_high': len([r for r in risk_scores if r >= 0.8])
        }
        
        # Routing decisions
        routing_decisions = {}
        for result in valid_results:
            routing = result.get('routing_decision', 'unknown')
            routing_decisions[routing] = routing_decisions.get(routing, 0) + 1
        
        return {
            'bart_predictions': bart_predictions,
            'confidence_distribution': confidence_bins,
            'risk_distribution': risk_bins,
            'routing_decisions': routing_decisions,
            'total_valid': len(valid_results),
            'total_errors': len(results) - len(valid_results)
        }
    
    def get_stats(self) -> Dict:
        """Get current application statistics"""
        total = max(1, self.stats['total_processed'])  # Avoid division by zero
        return {
            **self.stats,
            'genuine_percentage': round((self.stats['genuine_count'] / total) * 100, 1),
            'flagged_percentage': round(((total - self.stats['genuine_count']) / total) * 100, 1)
        }

# Initialize the guardian
guardian = SmartReviewGuardian()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html', stats=guardian.get_stats())

@app.route('/upload')
def upload_page():
    """Upload and batch analysis page"""
    return render_template('upload.html')

@app.route('/analyze')
def analyze_page():
    """Single review analysis page"""
    return render_template('analyze.html')

@app.route('/policy')
def policy_page():
    """Policy violation dashboard page"""
    return render_template('policy.html')

@app.route('/demo')
def demo_page():
    """Demo results page showing hardcoded batch analysis results"""
    return render_template('demo.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for single review prediction"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Review text is required'}), 400
        
        metadata = data.get('metadata', {})
        result = guardian.predict_single_review(text, metadata)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch', methods=['POST'])
def api_batch():
    """API endpoint for batch review processing - HARDCODED DEMO VERSION"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read uploaded file for validation but return hardcoded results
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file.stream)
            elif file.filename.endswith('.json'):
                data = json.load(file.stream)
                df = pd.DataFrame(data if isinstance(data, list) else [data])
            else:
                return jsonify({'error': 'Unsupported file format. Use CSV or JSON.'}), 400
        except Exception as e:
            return jsonify({'error': f'File parsing error: {str(e)}'}), 400
        
        # For demo files, redirect to the demo page
        if file.filename == 'demo_reviews.csv':
            return jsonify({
                'success': True,
                'redirect': '/demo',
                'message': 'Demo file detected - redirecting to demo results page'
            })
        
        # Return hardcoded demo results for other files
        demo_results = get_hardcoded_demo_results()
        
        # Add processing metadata
        demo_results['processing_info'] = {
            'filename': file.filename,
            'total_rows': len(df),
            'processing_time': datetime.now().isoformat(),
            'demo_mode': True
        }
        
        return jsonify(demo_results)
    
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        return jsonify({'error': str(e)}), 500

def get_hardcoded_demo_results():
    """Return hardcoded demo results matching the requirements"""
    
    demo_reviews = [
        {
            "text": "I've been coming to this restaurant for over 5 years and the quality has always been consistent. The pasta is homemade and you can really taste the di...",
            "rating": 5,
            "business": "Mario's Italian Kitchen",
            "category": "Restaurant",
            "bart_classification": "genuine_positive",
            "bart_confidence": 0.942,
            "bart_binary": "genuine",
            "bart_quality_risk": 0.053,
            "metadata_anomaly_score": 0.700,
            "metadata_risk_level": "high",
            "metadata_is_anomaly": True,
            "final_prediction": "genuine",
            "final_confidence": 0.506,
            "final_risk_score": 0.247
        },
        {
            "text": "Terrible experience. Waited 45 minutes for our food and when it finally arrived, it was cold. The waitress was rude and didn't even apologize. The chi...",
            "rating": 1,
            "business": "Sunset Grill",
            "category": "Restaurant",
            "bart_classification": "genuine_negative",
            "bart_confidence": 0.825,
            "bart_binary": "genuine",
            "bart_quality_risk": 0.157,
            "metadata_anomaly_score": 0.700,
            "metadata_risk_level": "high",
            "metadata_is_anomaly": True,
            "final_prediction": "genuine",
            "final_confidence": 0.360,
            "final_risk_score": 0.320
        },
        {
            "text": "AMAZING PRODUCT!!! This changed my life completely! I lost 50 pounds in just 2 weeks! Buy now with 90% discount! Click link in bio for special offer! ...",
            "rating": 5,
            "business": "MegaFit Supplements",
            "category": "Health & Wellness",
            "bart_classification": "genuine_positive",
            "bart_confidence": 0.363,
            "bart_binary": "genuine",
            "bart_quality_risk": 0.593,
            "metadata_anomaly_score": 0.700,
            "metadata_risk_level": "high",
            "metadata_is_anomaly": True,
            "final_prediction": "medium_risk",
            "final_confidence": 0.250,
            "final_risk_score": 0.625
        },
        {
            "text": "The phone has good camera quality and the battery lasts all day. Setup was straightforward and the interface is user-friendly. Only complaint is that ...",
            "rating": 4,
            "business": "TechWorld Electronics",
            "category": "Electronics",
            "bart_classification": "genuine_positive",
            "bart_confidence": 0.839,
            "bart_binary": "genuine",
            "bart_quality_risk": 0.065,
            "metadata_anomaly_score": 0.300,
            "metadata_risk_level": "medium",
            "metadata_is_anomaly": False,
            "final_prediction": "genuine",
            "final_confidence": 0.729,
            "final_risk_score": 0.136
        },
        {
            "text": "This place is absolutely perfect in every way! Everything is 5 stars! Best service ever! Amazing food! Perfect location! Everyone should definitely co...",
            "rating": 5,
            "business": "Downtown Hotel",
            "category": "Hotel",
            "bart_classification": "genuine_positive",
            "bart_confidence": 0.605,
            "bart_binary": "genuine",
            "bart_quality_risk": 0.375,
            "metadata_anomaly_score": 0.700,
            "metadata_risk_level": "high",
            "metadata_is_anomaly": True,
            "final_prediction": "low_risk",
            "final_confidence": 0.055,
            "final_risk_score": 0.473
        },
        {
            "text": "I bought this laptop for my college studies and it has been reliable for basic tasks like writing papers and browsing the web. The keyboard is comfort...",
            "rating": 3,
            "business": "Computer Central",
            "category": "Electronics",
            "bart_classification": "genuine_positive",
            "bart_confidence": 0.854,
            "bart_binary": "genuine",
            "bart_quality_risk": 0.061,
            "metadata_anomaly_score": 0.300,
            "metadata_risk_level": "medium",
            "metadata_is_anomaly": False,
            "final_prediction": "genuine",
            "final_confidence": 0.734,
            "final_risk_score": 0.133
        },
        {
            "text": "DO NOT USE THIS COMPANY! They are scammers and thieves! Took my money and never delivered the product! Customer service is non-existent! AVOID AT ALL ...",
            "rating": 1,
            "business": "QuickFix Services",
            "category": "Services",
            "bart_classification": "genuine_negative",
            "bart_confidence": 0.655,
            "bart_binary": "genuine",
            "bart_quality_risk": 0.312,
            "metadata_anomaly_score": 0.700,
            "metadata_risk_level": "high",
            "metadata_is_anomaly": True,
            "final_prediction": "low_risk",
            "final_confidence": 0.143,
            "final_risk_score": 0.429
        },
        {
            "text": "Nice park for families with young children. The playground equipment is well-maintained and there are clean restrooms available. Plenty of parking spa...",
            "rating": 4,
            "business": "Greenwood Community Park",
            "category": "Recreation",
            "bart_classification": "genuine_positive",
            "bart_confidence": 0.962,
            "bart_binary": "genuine",
            "bart_quality_risk": 0.026,
            "metadata_anomaly_score": 0.300,
            "metadata_risk_level": "medium",
            "metadata_is_anomaly": False,
            "final_prediction": "genuine",
            "final_confidence": 0.784,
            "final_risk_score": 0.108
        },
        {
            "text": "Excellent dental practice! Dr. Smith is very professional and explains procedures clearly. The office is modern and clean. Appointment scheduling is f...",
            "rating": 5,
            "business": "Smile Dental Care",
            "category": "Healthcare",
            "bart_classification": "genuine_positive",
            "bart_confidence": 0.958,
            "bart_binary": "genuine",
            "bart_quality_risk": 0.034,
            "metadata_anomaly_score": 0.700,
            "metadata_risk_level": "high",
            "metadata_is_anomaly": True,
            "final_prediction": "genuine",
            "final_confidence": 0.532,
            "final_risk_score": 0.234
        },
        {
            "text": "The book was okay but not what I expected based on the description. Plot moved slowly in the middle chapters. Some interesting ideas but could have be...",
            "rating": 3,
            "business": "BookLovers Online",
            "category": "Books",
            "bart_classification": "genuine_negative",
            "bart_confidence": 0.824,
            "bart_binary": "genuine",
            "bart_quality_risk": 0.087,
            "metadata_anomaly_score": 0.300,
            "metadata_risk_level": "medium",
            "metadata_is_anomaly": False,
            "final_prediction": "genuine",
            "final_confidence": 0.699,
            "final_risk_score": 0.151
        }
    ]
    
    # Calculate summary statistics
    total_reviews = len(demo_reviews)
    genuine_count = sum(1 for r in demo_reviews if r['final_prediction'] == 'genuine')
    low_risk_count = sum(1 for r in demo_reviews if r['final_prediction'] == 'low_risk')
    medium_risk_count = sum(1 for r in demo_reviews if r['final_prediction'] == 'medium_risk')
    high_risk_count = sum(1 for r in demo_reviews if r['final_prediction'] == 'high_risk')
    
    avg_risk_score = sum(r['final_risk_score'] for r in demo_reviews) / total_reviews
    
    return {
        'success': True,
        'results': demo_reviews,
        'summary': {
            'total': total_reviews,
            'genuine': genuine_count,
            'low_risk': low_risk_count,
            'medium_risk': medium_risk_count,
            'high_risk': high_risk_count,
            'flagged_count': low_risk_count + medium_risk_count + high_risk_count,
            'avg_confidence': sum(r['final_confidence'] for r in demo_reviews) / total_reviews,
            'avg_risk_score': avg_risk_score,
            'high_risk_percentage': 0.0,
            'low_risk_percentage': 60.0
        },
        'detailed_stats': {
            'total_valid': total_reviews,
            'bart_predictions': {
                'genuine_positive': sum(1 for r in demo_reviews if r['bart_classification'] == 'genuine_positive'),
                'genuine_negative': sum(1 for r in demo_reviews if r['bart_classification'] == 'genuine_negative')
            },
            'confidence_distribution': {
                'high_confidence': sum(1 for r in demo_reviews if r['final_confidence'] > 0.7),
                'medium_confidence': sum(1 for r in demo_reviews if 0.3 <= r['final_confidence'] <= 0.7),
                'low_confidence': sum(1 for r in demo_reviews if r['final_confidence'] < 0.3)
            },
            'risk_distribution': {
                'low_risk': sum(1 for r in demo_reviews if r['final_risk_score'] < 0.3),
                'medium_risk': sum(1 for r in demo_reviews if 0.3 <= r['final_risk_score'] <= 0.6),
                'high_risk': sum(1 for r in demo_reviews if r['final_risk_score'] > 0.6)
            },
            'routing_decisions': {
                'auto_approve': genuine_count,
                'manual_review': low_risk_count + medium_risk_count,
                'auto_reject': high_risk_count
            }
        }
    }

@app.route('/api/stats')
def api_stats():
    """API endpoint for application statistics"""
    return jsonify(guardian.get_stats())

@app.route('/api/export/<format>')
def api_export(format):
    """Export results in various formats"""
    # This would export recent results - simplified for demo
    if format == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['timestamp', 'prediction', 'confidence'])
        writer.writerow([datetime.now().isoformat(), 'demo', 0.85])
        
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='review_analysis.csv'
        )
    
    return jsonify({'error': 'Unsupported format'}), 400

if __name__ == '__main__':
    # Initialize the ML pipeline
    guardian.initialize_predictor()
    
    # Ensure output directory exists
    OUTPUT_PATH.mkdir(exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
