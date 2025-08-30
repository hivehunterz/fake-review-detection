"""
🎛️ CONFIGURABLE REVIEW QUALITY PREDICTION
Allows real-time threshold adjustment to optimize classification distribution
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

# Add project paths
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "scripts" / "evaluation"))

# Import after path setup
import sys
sys.path.append(str(PROJECT_ROOT / "scripts" / "evaluation"))
from universal_review_tester import UniversalReviewTester

class ConfigurablePredictor:
    """Prediction with configurable thresholds"""
    
    def __init__(self):
        self.tester = UniversalReviewTester()
    
    def predict_with_config(self, input_file, output_dir, 
                           genuine_threshold=0.35, low_risk_threshold=0.55, 
                           medium_risk_threshold=0.8, text_column='text'):
        """
        Run prediction with custom thresholds
        
        Args:
            genuine_threshold: Lower = more genuine classifications
            low_risk_threshold: Higher = more low-risk, less suspicious
            medium_risk_threshold: Higher = less suspicious classifications
        """
        
        # Temporarily update the threshold function
        original_function = self.tester._score_to_category
        
        def custom_score_to_category(score):
            if score < genuine_threshold:
                return 'genuine'
            elif score < low_risk_threshold:
                return 'low_risk'
            elif score < medium_risk_threshold:
                return 'medium_risk'
            else:
                return 'high_risk'
        
        # Replace the function
        self.tester._score_to_category = custom_score_to_category
        
        try:
            # Run analysis
            results = self.tester.run_universal_test(
                input_file=input_file,
                text_column=text_column,
                output_dir=output_dir
            )
            
            # Print threshold info
            print(f"\n🎛️ THRESHOLD CONFIGURATION:")
            print(f"  Genuine: < {genuine_threshold}")
            print(f"  Low-Risk: < {low_risk_threshold}")
            print(f"  Medium-Risk: < {medium_risk_threshold}")
            print(f"  High-Risk: >= {medium_risk_threshold}")
            
            return results
            
        finally:
            # Restore original function
            self.tester._score_to_category = original_function

def main():
    parser = argparse.ArgumentParser(description='Configurable Review Quality Prediction')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--text-column', default='text', help='Text column name')
    
    # Threshold arguments
    parser.add_argument('--genuine-threshold', type=float, default=0.35, 
                       help='Genuine threshold (lower = more genuine)')
    parser.add_argument('--low-risk-threshold', type=float, default=0.55,
                       help='Low-risk threshold (higher = more low-risk)')
    parser.add_argument('--medium-risk-threshold', type=float, default=0.8,
                       help='Medium-risk threshold (higher = less suspicious)')
    
    # Preset configurations
    parser.add_argument('--preset', choices=['conservative', 'balanced', 'aggressive'],
                       help='Use preset threshold configuration')
    
    args = parser.parse_args()
    
    # Apply presets
    if args.preset == 'conservative':
        # Conservative: More manual verification
        genuine_threshold, low_risk_threshold, medium_risk_threshold = 0.25, 0.45, 0.7
    elif args.preset == 'balanced':
        # Balanced: Default improved settings
        genuine_threshold, low_risk_threshold, medium_risk_threshold = 0.35, 0.55, 0.8
    elif args.preset == 'aggressive':
        # Aggressive: Maximum automation
        genuine_threshold, low_risk_threshold, medium_risk_threshold = 0.45, 0.65, 0.85
    else:
        # Custom thresholds
        genuine_threshold = args.genuine_threshold
        low_risk_threshold = args.low_risk_threshold
        medium_risk_threshold = args.medium_risk_threshold
    
    # Run prediction
    predictor = ConfigurablePredictor()
    results = predictor.predict_with_config(
        input_file=args.input,
        output_dir=args.output,
        genuine_threshold=genuine_threshold,
        low_risk_threshold=low_risk_threshold,
        medium_risk_threshold=medium_risk_threshold,
        text_column=args.text_column
    )
    
    print(f"\n✅ Prediction complete! Results saved to: {args.output}")

if __name__ == "__main__":
    main()
