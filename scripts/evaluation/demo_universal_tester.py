#!/usr/bin/env python3
"""
🔬 UNIVERSAL TESTER - DEMO SCRIPT
Example script showing how to use the universal review tester.
"""

import pandas as pd
from pathlib import Path
import sys

# Add the evaluation scripts to path
sys.path.append(str(Path(__file__).parent))

from universal_review_tester import UniversalReviewTester

def create_sample_data():
    """Create sample review data for testing"""
    sample_reviews = [
        {
            'review_text': 'This product is absolutely amazing! Great quality and fast shipping.',
            'rating': 5,
            'reviewer_name': 'John D.',
            'date': '2025-08-15'
        },
        {
            'review_text': 'Buy now! Best price ever! Click here for discount!',
            'rating': 5,
            'reviewer_name': 'PromoBot123',
            'date': '2025-08-30'
        },
        {
            'review_text': 'The product arrived quickly but the quality was not as expected. Would not recommend.',
            'rating': 2,
            'reviewer_name': 'Sarah M.',
            'date': '2025-08-20'
        },
        {
            'review_text': 'Excellent customer service and product quality. Highly recommend!',
            'rating': 5,
            'reviewer_name': 'Mike R.',
            'date': '2025-08-25'
        },
        {
            'review_text': 'asdfghjkl random text here nothing useful spam content',
            'rating': 1,
            'reviewer_name': 'RandomUser',
            'date': '2025-08-29'
        },
        {
            'review_text': 'Good product for the price. Delivery was on time and packaging was secure.',
            'rating': 4,
            'reviewer_name': 'Lisa K.',
            'date': '2025-08-18'
        }
    ]
    
    df = pd.DataFrame(sample_reviews)
    sample_file = "sample_reviews.csv"
    df.to_csv(sample_file, index=False)
    print(f"📋 Sample data created: {sample_file}")
    return sample_file

def demo_universal_tester():
    """Demonstrate the universal review tester"""
    print("🔬 UNIVERSAL REVIEW TESTER DEMO")
    print("=" * 50)
    
    # Create sample data
    sample_file = create_sample_data()
    
    # Initialize tester
    tester = UniversalReviewTester("demo_results")
    
    # Run complete analysis
    print("\n🚀 Running complete analysis...")
    try:
        report = tester.run_complete_analysis(
            input_file=sample_file,
            text_column='review_text',
            output_dir='demo_results'
        )
        
        print("\n✅ Analysis complete!")
        print("\n📊 RESULTS SUMMARY:")
        print("-" * 30)
        
        # Print stage results
        for stage, scores in report['stage_scores'].items():
            print(f"\n{stage.upper()}:")
            for key, value in scores.items():
                if isinstance(value, dict):
                    print(f"  {key}: {value}")
                elif isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        # Print overall analysis
        if 'overall_analysis' in report:
            overall = report['overall_analysis']
            print(f"\nOVERALL ANALYSIS:")
            print(f"  Average Risk Score: {overall.get('avg_risk_score', 0):.3f}")
            print(f"  High Risk Reviews: {overall.get('high_risk_percentage', 0):.1f}%")
            print(f"  Low Risk Reviews: {overall.get('low_risk_percentage', 0):.1f}%")
        
        # Print recommendations
        if report['recommendations']:
            print(f"\nRECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  {rec}")
        
        print(f"\n📁 Detailed results saved to: demo_results/")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure all required models are available in the models/ directory")

if __name__ == "__main__":
    demo_universal_tester()
