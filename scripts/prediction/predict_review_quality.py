"""
🔍 REVIEW QUALITY PREDICTION PIPELINE
Uses trained ML models to predict review quality without heuristics:
1. BART fine-tuned model for text classification
2. Enhanced metadata analyzer for anomaly detection
3. Advanced fusion model for final quality assessment

Pure machine learning approach - no hardcoded rules
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up to main project directory
CORE_PATH = PROJECT_ROOT / "core"
DATA_PATH = PROJECT_ROOT / "data"
MODELS_PATH = PROJECT_ROOT / "models"
OUTPUT_PATH = PROJECT_ROOT / "output"

# Add core modules to path
sys.path.append(str(CORE_PATH / "stage1_bart"))
sys.path.append(str(CORE_PATH / "stage2_metadata"))
sys.path.append(str(CORE_PATH / "fusion"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_PATH / "prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ReviewQualityPredictor:
    """Complete ML-based review quality prediction pipeline"""
    
    def __init__(self):
        self.bart_model = None
        self.metadata_analyzer = None
        self.fusion_model = None
        self.models_loaded = False
        
    def load_models(self) -> bool:
        """Load all trained models"""
        logger.info("📦 Loading trained models...")
        
        try:
            # Load BART model
            bart_model_path = MODELS_PATH / "bart_classifier"
            if bart_model_path.exists():
                from comprehensive_model_evaluation import ComprehensiveModelEvaluator
                test_data_path = str(DATA_PATH / "data_all_test.csv")
                self.bart_model = ComprehensiveModelEvaluator(str(bart_model_path), test_data_path)
                self.bart_model.load_models()
                logger.info("✅ BART model loaded")
            else:
                logger.error(f"❌ BART model not found: {bart_model_path}")
                return False
            
            # Load metadata analyzer
            analyzer_path = MODELS_PATH / "metadata_analyzer.pkl"
            if analyzer_path.exists():
                from enhanced_metadata_analyzer import EnhancedMetadataAnalyzer
                self.metadata_analyzer = EnhancedMetadataAnalyzer.load_model(analyzer_path)
                logger.info("✅ Metadata analyzer loaded")
            else:
                logger.warning(f"⚠️ Metadata analyzer not found: {analyzer_path}")
                # Create fallback analyzer
                self.metadata_analyzer = None
            
            # Load fusion model
            fusion_path = MODELS_PATH / "fusion_model.pkl"
            if fusion_path.exists():
                from fusion_model import AdvancedFusionModel
                self.fusion_model = AdvancedFusionModel(save_path=str(fusion_path))
                if self.fusion_model.load_model():
                    logger.info("✅ Fusion model loaded")
                else:
                    logger.error("❌ Failed to load fusion model")
                    return False
            else:
                logger.error(f"❌ Fusion model not found: {fusion_path}")
                return False
            
            self.models_loaded = True
            logger.info("🎉 All models loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def compute_p_bad_from_bart_probs(self, all_probabilities: List[List[float]], 
                                     labels: List[str]) -> List[float]:
        """Compute p_bad scores from BART probability distributions"""
        probs = np.array(all_probabilities)
        
        # Define bad classes
        bad_classes = ['spam', 'advertisement', 'irrelevant', 'fake_rant', 'inappropriate']
        bad_indices = [labels.index(label) for label in bad_classes if label in labels]
        
        # Sum probabilities of bad classes
        p_bad_scores = []
        for prob_array in probs:
            p_bad = sum(prob_array[i] for i in bad_indices)
            p_bad_scores.append(p_bad)
        
        return p_bad_scores
    
    def predict_bart_stage(self, texts: List[str]) -> Dict[str, Any]:
        """Stage 1: BART text classification"""
        logger.info("🤖 Stage 1: BART text classification")
        
        if not self.bart_model:
            raise ValueError("BART model not loaded")
        
        # Get predictions and confidences
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
        
        # Compute p_bad scores
        labels = ['genuine_positive', 'genuine_negative', 'spam', 'advertisement', 
                 'irrelevant', 'fake_rant', 'inappropriate']
        p_bad_scores = self.compute_p_bad_from_bart_probs(all_probabilities, labels)
        
        return {
            'predictions': predictions,
            'confidences': confidences,
            'p_bad_scores': p_bad_scores,
            'all_probabilities': all_probabilities,
            'labels': labels
        }
    
    def predict_metadata_stage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Stage 2: Enhanced metadata analysis"""
        logger.info("📊 Stage 2: Enhanced metadata analysis")
        
        if self.metadata_analyzer:
            try:
                # Prepare data for analyzer
                enhanced_df = df.copy()
                
                # Map required columns
                if 'rating' in enhanced_df.columns:
                    enhanced_df['stars'] = enhanced_df['rating']
                elif 'llm_rating' in enhanced_df.columns:
                    enhanced_df['stars'] = enhanced_df['llm_rating']
                else:
                    enhanced_df['stars'] = 3.0
                
                enhanced_df['placeId'] = enhanced_df.get('business_name', 'unknown_business')
                enhanced_df['reviewerId'] = [f'user_{i}' for i in range(len(enhanced_df))]
                enhanced_df['publishedAtDate'] = '2024-01-01T00:00:00Z'
                enhanced_df['reviewerNumberOfReviews'] = 10
                enhanced_df['isLocalGuide'] = False
                enhanced_df['likesCount'] = 0
                enhanced_df['reviewId'] = [f'review_{i}' for i in range(len(enhanced_df))]
                
                # Run analysis
                self.metadata_analyzer.df = enhanced_df
                anomaly_scores = self.metadata_analyzer.detect_enhanced_anomalies()
                
                # Normalize scores to probabilities
                enhanced_probabilities = [(score + 1) / 2 for score in anomaly_scores]
                
                return {
                    'enhanced_probabilities': enhanced_probabilities,
                    'anomaly_scores': anomaly_scores
                }
                
            except Exception as e:
                logger.warning(f"⚠️ Metadata analysis error: {e}")
                # Fallback to neutral scores
                return {
                    'enhanced_probabilities': [0.5] * len(df),
                    'anomaly_scores': [0.0] * len(df)
                }
        else:
            logger.warning("⚠️ Metadata analyzer not available, using neutral scores")
            return {
                'enhanced_probabilities': [0.5] * len(df),
                'anomaly_scores': [0.0] * len(df)
            }
    
    def predict_fusion_stage(self, stage1_results: Dict, stage2_results: Dict) -> List[Dict]:
        """Stage 3: Fusion model predictions"""
        logger.info("🔮 Stage 3: Fusion model predictions")
        
        if not self.fusion_model:
            raise ValueError("Fusion model not loaded")
        
        # Use fusion model for batch prediction
        fusion_results = self.fusion_model.batch_predict(
            stage1_results['p_bad_scores'],
            stage2_results['enhanced_probabilities'],
            [0.5] * len(stage1_results['p_bad_scores']),  # Neutral relevancy (no heuristics)
            [True] * len(stage1_results['p_bad_scores']),  # Assume relevant (no heuristics)
            stage1_results['confidences'],
            stage1_results['all_probabilities']
        )
        
        return fusion_results
    
    def predict_reviews(self, input_data: str, output_file: Optional[str] = None) -> pd.DataFrame:
        """Complete prediction pipeline for reviews"""
        logger.info("🔍 STARTING REVIEW QUALITY PREDICTION")
        logger.info("=" * 60)
        
        if not self.models_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Load input data
        if input_data.endswith('.csv'):
            df = pd.read_csv(input_data)
            logger.info(f"📂 Loaded {len(df)} reviews from CSV")
        elif input_data.endswith('.json'):
            with open(input_data, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            logger.info(f"📂 Loaded {len(df)} reviews from JSON")
        else:
            raise ValueError("Input must be CSV or JSON file")
        
        # Ensure required columns
        if 'text' not in df.columns:
            raise ValueError("Input data must have 'text' column")
        
        texts = df['text'].tolist()
        
        # Stage 1: BART predictions
        stage1_results = self.predict_bart_stage(texts)
        logger.info(f"✅ BART processed {len(texts)} reviews")
        
        # Stage 2: Metadata analysis
        stage2_results = self.predict_metadata_stage(df)
        logger.info("✅ Metadata analysis completed")
        
        # Stage 3: Fusion predictions
        fusion_results = self.predict_fusion_stage(stage1_results, stage2_results)
        logger.info("✅ Fusion predictions completed")
        
        # Combine results
        results_df = df.copy()
        results_df['bart_prediction'] = stage1_results['predictions']
        results_df['bart_confidence'] = stage1_results['confidences']
        results_df['p_bad_score'] = stage1_results['p_bad_scores']
        results_df['metadata_anomaly_score'] = stage2_results['enhanced_probabilities']
        results_df['final_prediction'] = [r['prediction'] for r in fusion_results]
        results_df['final_confidence'] = [r['confidence'] for r in fusion_results]
        results_df['fusion_score'] = [r['fusion_score'] for r in fusion_results]
        results_df['routing_decision'] = [r['routing'] for r in fusion_results]
        
        # Add probability details
        for i, probs in enumerate(stage1_results['all_probabilities']):
            labels = stage1_results['labels']
            for j, label in enumerate(labels):
                results_df.loc[i, f'prob_{label}'] = probs[j]
        
        # Save results
        if output_file:
            output_path = OUTPUT_PATH / output_file
        else:
            output_path = OUTPUT_PATH / "prediction_results.csv"
        
        results_df.to_csv(output_path, index=False)
        logger.info(f"💾 Results saved to: {output_path}")
        
        # Summary statistics
        pred_counts = results_df['final_prediction'].value_counts()
        logger.info("\n📊 PREDICTION SUMMARY")
        logger.info("=" * 40)
        for pred, count in pred_counts.items():
            percentage = (count / len(results_df)) * 100
            emoji = {
                "genuine": "✅",
                "suspicious": "🟡", 
                "low-quality": "⚠️",
                "high-confidence-spam": "🚫"
            }.get(pred, "❓")
            logger.info(f"  {emoji} {pred.upper()}: {count} ({percentage:.1f}%)")
        
        # Quality metrics
        avg_confidence = results_df['final_confidence'].mean()
        avg_p_bad = results_df['p_bad_score'].mean()
        logger.info(f"\n📈 QUALITY METRICS")
        logger.info(f"  Average prediction confidence: {avg_confidence:.3f}")
        logger.info(f"  Average p_bad risk score: {avg_p_bad:.3f}")
        
        return results_df
    
    def predict_single_review(self, text: str, metadata: Optional[Dict] = None) -> Dict:
        """Predict quality of a single review"""
        if not self.models_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Create DataFrame for single review
        review_data = {'text': [text]}
        if metadata:
            review_data.update({k: [v] for k, v in metadata.items()})
        
        df = pd.DataFrame(review_data)
        
        # Run prediction pipeline
        stage1_results = self.predict_bart_stage([text])
        stage2_results = self.predict_metadata_stage(df)
        fusion_results = self.predict_fusion_stage(stage1_results, stage2_results)
        
        # Return single result
        result = {
            'text': text,
            'bart_prediction': stage1_results['predictions'][0],
            'bart_confidence': stage1_results['confidences'][0],
            'p_bad_score': stage1_results['p_bad_scores'][0],
            'metadata_anomaly_score': stage2_results['enhanced_probabilities'][0],
            'final_prediction': fusion_results[0]['prediction'],
            'final_confidence': fusion_results[0]['confidence'],
            'fusion_score': fusion_results[0]['fusion_score'],
            'routing_decision': fusion_results[0]['routing'],
            'class_probabilities': dict(zip(stage1_results['labels'], 
                                          stage1_results['all_probabilities'][0]))
        }
        
        return result

def main():
    """Main prediction interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Review Quality Prediction Pipeline")
    parser.add_argument("--input", "-i", required=True, 
                       help="Input file (CSV or JSON) containing reviews")
    parser.add_argument("--output", "-o", 
                       help="Output CSV file name (default: prediction_results.csv)")
    parser.add_argument("--text", "-t",
                       help="Single review text to predict")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    OUTPUT_PATH.mkdir(exist_ok=True)
    
    # Initialize predictor
    predictor = ReviewQualityPredictor()
    
    # Load models
    if not predictor.load_models():
        logger.error("❌ Failed to load models. Please run training first.")
        return False
    
    try:
        if args.text:
            # Single review prediction
            result = predictor.predict_single_review(args.text)
            logger.info("\n🔍 SINGLE REVIEW PREDICTION")
            logger.info("=" * 50)
            logger.info(f"Text: {result['text']}")
            logger.info(f"BART: {result['bart_prediction']} (conf: {result['bart_confidence']:.3f})")
            logger.info(f"P_BAD: {result['p_bad_score']:.3f}")
            logger.info(f"Final: {result['final_prediction']} (conf: {result['final_confidence']:.3f})")
            logger.info(f"Routing: {result['routing_decision']}")
            
        else:
            # Batch prediction
            results_df = predictor.predict_reviews(args.input, args.output)
            
            # Show high-quality reviews
            genuine_reviews = results_df[results_df['final_prediction'] == 'genuine']
            if len(genuine_reviews) > 0:
                logger.info(f"\n✨ HIGH QUALITY REVIEWS FOUND: {len(genuine_reviews)}")
                logger.info("=" * 60)
                
                for i, (_, row) in enumerate(genuine_reviews.head(5).iterrows()):
                    logger.info(f"\n📄 GENUINE REVIEW #{i + 1}")
                    logger.info(f"Text: {row['text'][:200]}...")
                    logger.info(f"Confidence: {row['final_confidence']:.3f}")
                    logger.info(f"P_BAD: {row['p_bad_score']:.3f}")
        
        logger.info("\n🎉 PREDICTION COMPLETED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
