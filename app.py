"""
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
            
            # Try to load models
            if self.predictor.load_models():
                logger.info("✅ ML models loaded successfully")
                return True
            else:
                logger.warning("⚠️ Could not load all models, using demo mode")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize predictor: {e}")
            return False
    
    def predict_single_review(self, text: str, metadata: Optional[Dict] = None) -> Dict:
        """Predict quality of a single review"""
        if not self.predictor:
            # Demo mode - return mock results
            return self._generate_demo_result(text)
        
        try:
            result = self.predictor.predict_single_review(text, metadata)
            self._update_stats(result['final_prediction'])
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._generate_demo_result(text, error=str(e))
    
    def process_batch_reviews(self, reviews_data: List[Dict]) -> List[Dict]:
        """Process multiple reviews"""
        results = []
        for review in reviews_data:
            text = review.get('text', '')
            metadata = {k: v for k, v in review.items() if k != 'text'}
            result = self.predict_single_review(text, metadata)
            results.append(result)
        return results
    
    def _generate_demo_result(self, text: str, error: str = None) -> Dict:
        """Generate demo results when models aren't available"""
        import random
        
        # Simple heuristics for demo
        categories = ['genuine', 'suspicious', 'low-quality', 'high-confidence-spam']
        weights = [0.4, 0.3, 0.2, 0.1]  # Bias towards genuine for demo
        
        prediction = np.random.choice(categories, p=weights)
        confidence = random.uniform(0.6, 0.95)
        
        routing_map = {
            'genuine': 'automatic-approval',
            'suspicious': 'requires-manual-verification',
            'low-quality': 'requires-manual-verification',
            'high-confidence-spam': 'automatic-rejection'
        }
        
        result = {
            'text': text,
            'bart_prediction': random.choice(['genuine_positive', 'genuine_negative', 'spam', 'advertisement']),
            'bart_confidence': random.uniform(0.6, 0.9),
            'p_bad_score': random.uniform(0.1, 0.8),
            'metadata_anomaly_score': random.uniform(0.0, 1.0),
            'final_prediction': prediction,
            'final_confidence': confidence,
            'fusion_score': random.uniform(0.0, 1.0),
            'routing_decision': routing_map[prediction],
            'class_probabilities': {
                'genuine_positive': random.uniform(0.1, 0.5),
                'genuine_negative': random.uniform(0.1, 0.3),
                'spam': random.uniform(0.0, 0.3),
                'advertisement': random.uniform(0.0, 0.2),
                'irrelevant': random.uniform(0.0, 0.2),
                'fake_rant': random.uniform(0.0, 0.1),
                'inappropriate': random.uniform(0.0, 0.1)
            },
            'demo_mode': True
        }
        
        if error:
            result['error'] = error
            
        self._update_stats(prediction)
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

@app.route('/insights')
def insights_page():
    """Insights and metrics page"""
    return render_template('insights.html')

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
    """API endpoint for batch review processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read uploaded file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.stream)
        elif file.filename.endswith('.json'):
            data = json.load(file.stream)
            df = pd.DataFrame(data if isinstance(data, list) else [data])
        else:
            return jsonify({'error': 'Unsupported file format. Use CSV or JSON.'}), 400
        
        # Ensure required columns
        if 'text' not in df.columns:
            return jsonify({'error': 'File must contain a "text" column'}), 400
        
        # Process reviews
        reviews_data = df.to_dict('records')
        results = guardian.process_batch_reviews(reviews_data)
        
        return jsonify({
            'results': results,
            'summary': {
                'total': len(results),
                'genuine': len([r for r in results if r['final_prediction'] == 'genuine']),
                'flagged': len([r for r in results if r['final_prediction'] != 'genuine'])
            }
        })
    
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        return jsonify({'error': str(e)}), 500

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