"""
Enhanced Pipeline Runner - Integrates Stage 1 BART with Stage 2 Metadata Analysis
Demonstrates how to run the complete enhanced pipeline with BART integration
"""

import logging
import sys
from pathlib import Path

# Add paths for imports
sys.path.append('..')
sys.path.append('../stage_1_bart_finetuning')

from enhanced_metadata_analyzer import EnhancedMetadataAnalyzer
from config import BART_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_bart_model():
    """Find the most recent BART model in the stage_1_bart_finetuning directory."""
    bart_dir = Path('../stage_1_bart_finetuning')
    
    if not bart_dir.exists():
        logger.warning(f"BART directory not found: {bart_dir}")
        return None
    
    # Look for directories with BART model names
    model_patterns = [
        'enhanced_bart_review_classifier*',
        'bart_review_classifier*',
        'fine_tuned_model*'
    ]
    
    model_dirs = []
    for pattern in model_patterns:
        model_dirs.extend(bart_dir.glob(pattern))
    
    if model_dirs:
        # Sort by modification time and get the most recent
        latest_model = max(model_dirs, key=lambda p: p.stat().st_mtime)
        logger.info(f"🤖 Found BART model: {latest_model}")
        return str(latest_model)
    
    logger.warning("No BART models found in stage_1_bart_finetuning directory")
    return None

def run_enhanced_pipeline():
    """Run the complete enhanced pipeline with Stage 1 BART integration."""
    logger.info("🚀 Starting Enhanced Pipeline with BART Integration")
    logger.info("=" * 60)
    
    try:
        # Find BART model
        bart_model_path = find_bart_model()
        
        if not bart_model_path and BART_CONFIG['enable_integration']:
            logger.warning("⚠️  BART integration enabled but no model found")
            logger.warning("⚠️  Proceeding without BART features")
        
        # Initialize enhanced analyzer
        logger.info("📊 Initializing Enhanced Metadata Analyzer...")
        analyzer = EnhancedMetadataAnalyzer(bart_model_path)
        
        # Run the complete analysis
        logger.info("🔄 Running enhanced analysis pipeline...")
        results = analyzer.run_enhanced_analysis()
        
        # Display results summary
        logger.info("✅ Enhanced Pipeline Completed Successfully!")
        logger.info("=" * 60)
        
        logger.info(f"📈 Results Summary:")
        logger.info(f"   Total Reviews Analyzed: {results.get('total_reviews', 'N/A')}")
        logger.info(f"   BART Integration: {'✅ Enabled' if results.get('bart_integration') else '❌ Disabled'}")
        
        if 'bart_summary' in results:
            bart_summary = results['bart_summary']
            logger.info(f"   Average BART Confidence: {bart_summary.get('avg_confidence', 0):.3f}")
            logger.info(f"   Average Quality Risk: {bart_summary.get('avg_quality_risk', 0):.3f}")
            logger.info(f"   High Risk Reviews: {bart_summary.get('high_risk_reviews', 0)}")
        
        if 'enhanced_ml_results' in results:
            ml_results = results['enhanced_ml_results']
            logger.info(f"   ML Anomalies Detected: {ml_results.get('enhanced_ml_anomaly_count', 0)}")
            logger.info(f"   Anomaly Rate: {ml_results.get('enhanced_ml_anomaly_rate', 0):.3f}")
            logger.info(f"   BART Features Used: {ml_results.get('bart_feature_count', 0)}")
        
        # Log output files
        output_dir = Path('outputs')
        if output_dir.exists():
            logger.info(f"📁 Output Files Generated:")
            output_files = list(output_dir.glob('*'))
            for file in output_files:
                if file.is_file():
                    logger.info(f"   📄 {file.name}")
        
        logger.info("🎉 Pipeline completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        raise

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run enhanced pipeline with BART integration')
    parser.add_argument('--bart-model', type=str, 
                       help='Override BART model path')
    parser.add_argument('--disable-bart', action='store_true',
                       help='Disable BART integration')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.disable_bart:
        BART_CONFIG['enable_integration'] = False
        logger.info("🔧 BART integration disabled via command line")
    
    if args.bart_model:
        logger.info(f"🔧 Using custom BART model: {args.bart_model}")
    
    # Run the pipeline
    results = run_enhanced_pipeline()
    
    return results

if __name__ == "__main__":
    main()
