#!/usr/bin/env python3
"""
🔬 UNIVERSAL REVIEW TESTER
Universal script to test any CSV file with the complete fake review detection pipeline.
Provides scores for each stage: BART, Metadata, and Fusion analysis.

Usage:
    python universal_review_tester.py --input your_file.csv --text_column review_text --output results/
    
CSV Requirements:
    - Must have a text column containing review content
    - Optional: metadata columns (rating, date, etc.)
    - Optional: ground truth labels for evaluation
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CORE_PATH = PROJECT_ROOT / "core"
DATA_PATH = PROJECT_ROOT / "data"
MODELS_PATH = PROJECT_ROOT / "models"
OUTPUT_PATH = PROJECT_ROOT / "output"

# Add core modules to path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CORE_PATH))
sys.path.insert(0, str(CORE_PATH / "stage1_bart"))
sys.path.insert(0, str(CORE_PATH / "stage2_metadata"))
sys.path.insert(0, str(CORE_PATH / "fusion"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class UniversalReviewTester:
    """
    Universal tester for the complete fake review detection pipeline.
    Tests any CSV file and provides stage-by-stage analysis.
    """
    
    def __init__(self, output_dir: str = "universal_test_results"):
        """Initialize the universal tester"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.bart_classifier = None
        self.metadata_analyzer = None
        self.fusion_model = None
        
        # Results storage
        self.results = {}
        self.stage_scores = {}
        
        logger.info(f"🔬 Universal Review Tester initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
    def load_models(self):
        """Load all available models for the pipeline"""
        logger.info("🔄 Loading pipeline models...")
        
        # Load BART Classifier
        try:
            from core.stage1_bart.enhanced_bart_review_classifier import BARTReviewClassifier
            model_path = MODELS_PATH / "bart_classifier"
            if model_path.exists():
                self.bart_classifier = BARTReviewClassifier(str(model_path))
                logger.info("✅ BART classifier loaded (fine-tuned)")
            else:
                self.bart_classifier = BARTReviewClassifier()
                logger.info("✅ BART classifier loaded (zero-shot)")
        except Exception as e:
            logger.warning(f"⚠️  BART classifier not available: {e}")
            
        # Load Metadata Analyzer
        try:
            from core.stage2_metadata.enhanced_metadata_analyzer import EnhancedMetadataAnalyzer
            self.metadata_analyzer = EnhancedMetadataAnalyzer()
            logger.info("✅ Metadata analyzer loaded")
        except Exception as e:
            logger.warning(f"⚠️  Metadata analyzer not available: {e}")
            
        # Load Fusion Model
        try:
            from core.fusion.fusion_model import AdvancedFusionModel
            fusion_path = MODELS_PATH / "fusion_model.pkl"
            if fusion_path.exists():
                self.fusion_model = AdvancedFusionModel()
                self.fusion_model.load_model(str(fusion_path))
                logger.info("✅ Fusion model loaded")
            else:
                logger.warning("⚠️  Fusion model not found, will use weighted scoring")
        except Exception as e:
            logger.warning(f"⚠️  Fusion model not available: {e}")
    
    def validate_csv(self, file_path: str, text_column: str) -> pd.DataFrame:
        """Validate and load CSV file"""
        logger.info(f"📋 Loading CSV file: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"📊 Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Validate text column
            if text_column not in df.columns:
                available_cols = ', '.join(df.columns.tolist()[:10])
                raise ValueError(f"Text column '{text_column}' not found. Available columns: {available_cols}")
            
            # Check for missing text
            missing_text = df[text_column].isna().sum()
            if missing_text > 0:
                logger.warning(f"⚠️  {missing_text} rows have missing text content")
                df = df.dropna(subset=[text_column])
                logger.info(f"📊 After removing missing text: {len(df)} rows")
            
            # Show column info
            logger.info(f"📝 Text column: '{text_column}'")
            logger.info(f"📋 Available columns: {', '.join(df.columns.tolist())}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading CSV: {e}")
            raise
    
    def stage1_bart_analysis(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Stage 1: BART text classification analysis"""
        logger.info("🔍 Stage 1: BART Text Classification")
        
        if self.bart_classifier is None:
            logger.warning("⚠️  BART classifier not available, skipping Stage 1")
            return df
        
        results = []
        texts = df[text_column].tolist()
        
        logger.info(f"🔄 Analyzing {len(texts)} reviews with BART...")
        
        for i, text in enumerate(tqdm(texts, desc="BART Analysis")):
            try:
                # Get BART prediction (returns list of dicts)
                predictions = self.bart_classifier.predict([text])
                prediction_dict = predictions[0]  # Get first (and only) result
                
                result = {
                    'bart_classification': prediction_dict['prediction'],
                    'bart_confidence': prediction_dict['confidence'],
                    'bart_probabilities': prediction_dict['class_probabilities']
                }
                
                # Binary classification (genuine vs non-genuine)
                genuine_classes = ['genuine_positive', 'genuine_negative']
                is_genuine = prediction_dict['prediction'] in genuine_classes
                result['bart_binary'] = 'genuine' if is_genuine else 'non_genuine'
                result['bart_binary_confidence'] = prediction_dict['confidence'] if is_genuine else 1 - prediction_dict['confidence']
                
                # Quality risk score (use p_bad if available, otherwise calculate)
                if 'p_bad' in prediction_dict:
                    result['bart_quality_risk'] = prediction_dict['p_bad']
                else:
                    non_genuine_prob = sum([
                        prediction_dict['class_probabilities'].get(label, 0) 
                        for label in ['spam', 'advertisement', 'irrelevant', 'fake_rant', 'inappropriate']
                    ])
                    result['bart_quality_risk'] = non_genuine_prob
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"⚠️  Error processing review {i}: {e}")
                results.append({
                    'bart_classification': 'error',
                    'bart_confidence': 0.0,
                    'bart_binary': 'unknown',
                    'bart_binary_confidence': 0.0,
                    'bart_quality_risk': 0.5
                })
        
        # Add BART results to dataframe
        bart_df = pd.DataFrame(results)
        df_with_bart = pd.concat([df.reset_index(drop=True), bart_df], axis=1)
        
        # Stage 1 Summary
        stage1_summary = {
            'total_reviews': len(df_with_bart),
            'classification_distribution': df_with_bart['bart_classification'].value_counts().to_dict(),
            'binary_distribution': df_with_bart['bart_binary'].value_counts().to_dict(),
            'avg_confidence': df_with_bart['bart_confidence'].mean(),
            'avg_quality_risk': df_with_bart['bart_quality_risk'].mean(),
            'high_risk_count': (df_with_bart['bart_quality_risk'] > 0.7).sum()
        }
        
        self.stage_scores['stage1_bart'] = stage1_summary
        logger.info(f"✅ Stage 1 complete: {stage1_summary['binary_distribution']}")
        
        return df_with_bart
    
    def stage2_metadata_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 2: Metadata anomaly detection analysis"""
        logger.info("🔍 Stage 2: Metadata Anomaly Detection")
        
        if self.metadata_analyzer is None:
            logger.warning("⚠️  Metadata analyzer not available, using simplified analysis")
            return self._simplified_metadata_analysis(df)
        
        try:
            # Prepare metadata features
            metadata_features = self._extract_metadata_features(df)
            
            if metadata_features.empty:
                logger.warning("⚠️  No metadata features found, using simplified analysis")
                return self._simplified_metadata_analysis(df)
            
            # The metadata analyzer expects to work with its own data loading
            # So we'll use simplified analysis for now
            logger.info("🔄 Using simplified metadata analysis (analyzer expects specific data format)")
            return self._simplified_metadata_analysis(df)
            
        except Exception as e:
            logger.warning(f"⚠️  Metadata analysis error: {e}, using simplified analysis")
            df = self._simplified_metadata_analysis(df)
        
        return df
    
    def _extract_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract metadata features from the dataframe"""
        features = pd.DataFrame()
        
        # Common metadata columns to look for
        metadata_mappings = {
            'rating': ['rating', 'score', 'stars', 'star_rating'],
            'date': ['date', 'timestamp', 'created_at', 'review_date'],
            'helpful_count': ['helpful', 'helpful_count', 'thumbs_up', 'likes'],
            'length': ['length', 'text_length', 'character_count']
        }
        
        for feature, possible_cols in metadata_mappings.items():
            for col in possible_cols:
                if col in df.columns:
                    features[feature] = pd.to_numeric(df[col], errors='coerce')
                    break
        
        # Calculate text length if not available
        if 'length' not in features.columns and len(df.columns) > 0:
            text_col = df.select_dtypes(include=['object']).columns[0]
            features['length'] = df[text_col].astype(str).str.len()
        
        return features
    
    def _simplified_metadata_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simplified metadata analysis when full analyzer is not available"""
        # Basic anomaly detection based on available features
        df['metadata_anomaly_score'] = 0.3  # Default neutral score
        df['metadata_is_anomaly'] = False
        df['metadata_risk_level'] = 'medium'
        
        # If we have rating, detect extreme ratings as potential anomalies
        rating_cols = [col for col in df.columns if 'rating' in col.lower() or 'star' in col.lower()]
        if rating_cols:
            rating_col = rating_cols[0]
            ratings = pd.to_numeric(df[rating_col], errors='coerce')
            extreme_ratings = (ratings <= 1) | (ratings >= 5)
            df.loc[extreme_ratings, 'metadata_anomaly_score'] = 0.7
            df.loc[extreme_ratings, 'metadata_is_anomaly'] = True
            df.loc[extreme_ratings, 'metadata_risk_level'] = 'high'
        
        return df
    
    def _categorize_risk(self, score: float) -> str:
        """Categorize risk level based on anomaly score"""
        if score < 0.3:
            return 'low'
        elif score < 0.7:
            return 'medium'
        else:
            return 'high'
    
    def stage3_fusion_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 3: Fusion model analysis combining all stages"""
        logger.info("🔍 Stage 3: Fusion Analysis")
        
        if self.fusion_model is None:
            logger.info("🔄 Using weighted scoring fusion method")
            return self._weighted_fusion_analysis(df)
        
        try:
            # Prepare fusion features
            fusion_features = self._prepare_fusion_features(df)
            
            # For now, use simplified analysis since fusion model expects specific format
            logger.info("🔄 Using weighted scoring fusion method (model expects specific format)")
            return self._weighted_fusion_analysis(df)
            
        except Exception as e:
            logger.warning(f"⚠️  Fusion model error: {e}, using weighted scoring")
            df = self._weighted_fusion_analysis(df)
        
        return df
    
    def _weighted_fusion_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Weighted fusion analysis when ML model is not available"""
        # Combine BART and metadata scores with weights
        bart_weight = 0.7
        metadata_weight = 0.3
        
        # Calculate combined risk score
        bart_risk = df.get('bart_quality_risk', 0.5)
        metadata_risk = df.get('metadata_anomaly_score', 0.3)
        
        df['fusion_risk_score'] = (bart_risk * bart_weight + metadata_risk * metadata_weight)
        
        # Generate fusion predictions based on combined score
        df['fusion_prediction'] = df['fusion_risk_score'].apply(self._score_to_category)
        df['fusion_confidence'] = np.abs(df['fusion_risk_score'] - 0.5) * 2  # Distance from neutral
        
        # Stage 3 Summary
        stage3_summary = {
            'total_reviews': len(df),
            'prediction_distribution': df['fusion_prediction'].value_counts().to_dict(),
            'avg_risk_score': df['fusion_risk_score'].mean(),
            'avg_confidence': df['fusion_confidence'].mean(),
            'high_risk_count': (df['fusion_risk_score'] > 0.7).sum()
        }
        
        self.stage_scores['stage3_fusion'] = stage3_summary
        logger.info(f"✅ Stage 3 complete: {stage3_summary['prediction_distribution']}")
        
        return df
    
    def _score_to_category(self, score: float) -> str:
        """Convert risk score to category"""
        if score < 0.25:
            return 'genuine'
        elif score < 0.5:
            return 'low_risk'
        elif score < 0.75:
            return 'medium_risk'
        else:
            return 'high_risk'
    
    def _prepare_fusion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for fusion model"""
        features = pd.DataFrame()
        
        # BART features
        features['bart_quality_risk'] = df.get('bart_quality_risk', 0.5)
        features['bart_confidence'] = df.get('bart_confidence', 0.5)
        
        # Metadata features
        features['metadata_anomaly_score'] = df.get('metadata_anomaly_score', 0.3)
        
        # Fill missing values
        features = features.fillna(0.5)
        
        return features
    
    def generate_comprehensive_report(self, df: pd.DataFrame, input_file: str) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        logger.info("📊 Generating comprehensive report...")
        
        report = {
            'test_info': {
                'input_file': input_file,
                'timestamp': datetime.now().isoformat(),
                'total_reviews': len(df),
                'pipeline_version': '2.0'
            },
            'stage_scores': self.stage_scores,
            'overall_analysis': {},
            'recommendations': []
        }
        
        # Overall analysis
        if 'fusion_risk_score' in df.columns:
            overall = {
                'avg_risk_score': float(df['fusion_risk_score'].mean()),
                'high_risk_percentage': float((df['fusion_risk_score'] > 0.7).sum() / len(df) * 100),
                'low_risk_percentage': float((df['fusion_risk_score'] < 0.3).sum() / len(df) * 100),
                'risk_distribution': df['fusion_prediction'].value_counts().to_dict() if 'fusion_prediction' in df.columns else {}
            }
            report['overall_analysis'] = overall
            
            # Recommendations
            if overall['high_risk_percentage'] > 30:
                report['recommendations'].append("⚠️  High proportion of risky reviews detected - manual review recommended")
            if overall['avg_risk_score'] > 0.6:
                report['recommendations'].append("🔍 Overall high risk score - consider additional validation")
            if overall['low_risk_percentage'] > 70:
                report['recommendations'].append("✅ Mostly genuine reviews - system performance looks good")
        
        return report
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create visualization plots"""
        logger.info("📈 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Universal Review Testing - Pipeline Analysis', fontsize=16, fontweight='bold')
        
        # BART Classification Distribution
        if 'bart_classification' in df.columns:
            bart_counts = df['bart_classification'].value_counts()
            axes[0, 0].pie(bart_counts.values, labels=bart_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Stage 1: BART Classification')
        
        # BART Risk Score Distribution
        if 'bart_quality_risk' in df.columns:
            axes[0, 1].hist(df['bart_quality_risk'], bins=20, alpha=0.7, color='skyblue')
            axes[0, 1].set_title('Stage 1: Quality Risk Distribution')
            axes[0, 1].set_xlabel('Risk Score')
            axes[0, 1].set_ylabel('Count')
        
        # Metadata Anomaly Scores
        if 'metadata_anomaly_score' in df.columns:
            axes[0, 2].hist(df['metadata_anomaly_score'], bins=20, alpha=0.7, color='lightcoral')
            axes[0, 2].set_title('Stage 2: Metadata Anomaly Scores')
            axes[0, 2].set_xlabel('Anomaly Score')
            axes[0, 2].set_ylabel('Count')
        
        # Fusion Risk Scores
        if 'fusion_risk_score' in df.columns:
            axes[1, 0].hist(df['fusion_risk_score'], bins=20, alpha=0.7, color='lightgreen')
            axes[1, 0].set_title('Stage 3: Fusion Risk Scores')
            axes[1, 0].set_xlabel('Risk Score')
            axes[1, 0].set_ylabel('Count')
        
        # Final Predictions
        if 'fusion_prediction' in df.columns:
            pred_counts = df['fusion_prediction'].value_counts()
            axes[1, 1].bar(pred_counts.index, pred_counts.values, color='gold')
            axes[1, 1].set_title('Final Predictions')
            axes[1, 1].set_xlabel('Category')
            axes[1, 1].set_ylabel('Count')
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # Stage Comparison
        if all(col in df.columns for col in ['bart_quality_risk', 'metadata_anomaly_score', 'fusion_risk_score']):
            risk_comparison = df[['bart_quality_risk', 'metadata_anomaly_score', 'fusion_risk_score']].mean()
            axes[1, 2].bar(risk_comparison.index, risk_comparison.values, color=['skyblue', 'lightcoral', 'lightgreen'])
            axes[1, 2].set_title('Average Risk Scores by Stage')
            axes[1, 2].set_ylabel('Average Risk Score')
            plt.setp(axes[1, 2].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        viz_path = self.output_dir / "pipeline_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"📈 Visualizations saved: {viz_path}")
        plt.close()
    
    def save_results(self, df: pd.DataFrame, report: Dict[str, Any], input_file: str):
        """Save all results and reports"""
        logger.info("💾 Saving results...")
        
        # Save detailed results CSV
        results_path = self.output_dir / f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(results_path, index=False)
        logger.info(f"📊 Detailed results saved: {results_path}")
        
        # Save summary report
        report_path = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"📋 Analysis report saved: {report_path}")
        
        # Save stage scores CSV
        if self.stage_scores:
            scores_df = pd.DataFrame([
                {'stage': stage, **scores} 
                for stage, scores in self.stage_scores.items()
            ])
            scores_path = self.output_dir / f"stage_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            scores_df.to_csv(scores_path, index=False)
            logger.info(f"📈 Stage scores saved: {scores_path}")
    
    def run_complete_analysis(self, input_file: str, text_column: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Run complete pipeline analysis on CSV file"""
        logger.info("🚀 Starting Universal Review Testing Pipeline")
        
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)
        
        # Load models
        self.load_models()
        
        # Validate and load data
        df = self.validate_csv(input_file, text_column)
        
        # Stage 1: BART Analysis
        df = self.stage1_bart_analysis(df, text_column)
        
        # Stage 2: Metadata Analysis
        df = self.stage2_metadata_analysis(df)
        
        # Stage 3: Fusion Analysis
        df = self.stage3_fusion_analysis(df)
        
        # Generate report
        report = self.generate_comprehensive_report(df, input_file)
        
        # Create visualizations
        self.create_visualizations(df)
        
        # Save results
        self.save_results(df, report, input_file)
        
        logger.info("✅ Universal Review Testing Pipeline Complete!")
        
        # Print top 10 reviews with predictions
        self.print_top_reviews_with_predictions(df, input_file)
        
        return report
    
    def print_top_reviews_with_predictions(self, df: pd.DataFrame, input_file: str, top_n: int = 10):
        """Print top N reviews with all predictions in a readable format"""
        print("\n" + "🔍 TOP 10 REVIEWS WITH DETAILED PREDICTIONS")
        print("=" * 80)
        print(f"📁 Source: {input_file}")
        print(f"📊 Showing first {min(top_n, len(df))} reviews out of {len(df)} total")
        print("=" * 80)
        
        for i, (idx, row) in enumerate(df.head(top_n).iterrows()):
            print(f"\n📄 REVIEW #{i+1}")
            print("-" * 50)
            
            # Review text (truncated)
            text = row.get('text', 'N/A')
            if len(text) > 150:
                text = text[:150] + "..."
            print(f"📝 Text: {text}")
            
            # Basic info
            if 'rating' in row and not pd.isna(row['rating']):
                print(f"⭐ Rating: {row['rating']}")
            if 'business_name' in row and not pd.isna(row['business_name']):
                print(f"🏢 Business: {row['business_name']}")
            if 'category' in row and not pd.isna(row['category']):
                print(f"🏷️ Category: {row['category']}")
            
            print(f"\n🤖 STAGE-BY-STAGE PREDICTIONS:")
            
            # Stage 1: BART
            if 'bart_classification' in row:
                bart_class = row['bart_classification']
                bart_conf = row.get('bart_confidence', 0)
                bart_binary = row.get('bart_binary', 'unknown')
                bart_risk = row.get('bart_quality_risk', 0)
                
                print(f"  📊 Stage 1 (BART):")
                print(f"    • Classification: {bart_class} ({bart_conf:.3f} confidence)")
                print(f"    • Binary: {bart_binary}")
                print(f"    • Quality Risk: {bart_risk:.3f}")
            
            # Stage 2: Metadata
            if 'metadata_anomaly_score' in row:
                meta_score = row['metadata_anomaly_score']
                meta_risk = row.get('metadata_risk_level', 'unknown')
                meta_anomaly = row.get('metadata_is_anomaly', False)
                
                print(f"  📈 Stage 2 (Metadata):")
                print(f"    • Anomaly Score: {meta_score:.3f}")
                print(f"    • Risk Level: {meta_risk}")
                print(f"    • Is Anomaly: {meta_anomaly}")
            
            # Stage 3: Fusion (Final)
            if 'fusion_prediction' in row:
                fusion_pred = row['fusion_prediction']
                fusion_conf = row.get('fusion_confidence', 0)
                fusion_risk = row.get('fusion_risk_score', 0)
                
                # Choose emoji based on prediction
                pred_emoji = {
                    'genuine': '✅',
                    'low_risk': '🟡',
                    'medium_risk': '🟠',
                    'high_risk': '🔴'
                }.get(fusion_pred, '❓')
                
                print(f"  🎯 Stage 3 (Final Fusion):")
                print(f"    • {pred_emoji} FINAL PREDICTION: {fusion_pred.upper()}")
                print(f"    • Confidence: {fusion_conf:.3f}")
                print(f"    • Risk Score: {fusion_risk:.3f}")
        
        print("\n" + "=" * 80)
        print("🎯 PREDICTION LEGEND:")
        print("✅ genuine = Authentic review")
        print("🟡 low_risk = Likely genuine, minor concerns")
        print("🟠 medium_risk = Some suspicious patterns")
        print("🔴 high_risk = Likely fake/spam")
        print("=" * 80)

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Universal Review Testing Pipeline')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--text_column', '-t', default='review_text', help='Name of the text column (default: review_text)')
    parser.add_argument('--output', '-o', default='universal_test_results', help='Output directory (default: universal_test_results)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize tester
    tester = UniversalReviewTester(args.output)
    
    # Run analysis
    try:
        report = tester.run_complete_analysis(args.input, args.text_column, args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 UNIVERSAL REVIEW TESTING SUMMARY")
        print("="*60)
        print(f"📁 Input File: {args.input}")
        print(f"📊 Total Reviews: {report['test_info']['total_reviews']}")
        
        if 'overall_analysis' in report and report['overall_analysis']:
            overall = report['overall_analysis']
            print(f"⚡ Average Risk Score: {overall['avg_risk_score']:.3f}")
            print(f"🔴 High Risk: {overall['high_risk_percentage']:.1f}%")
            print(f"🟢 Low Risk: {overall['low_risk_percentage']:.1f}%")
        
        print(f"📂 Results saved to: {args.output}/")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
