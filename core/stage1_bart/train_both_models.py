#!/usr/bin/env python3
"""
Model Comparison Script
Compare Standard BART vs Weighted BART performance
"""

import subprocess
import time
import pandas as pd
from datetime import datetime

def run_training_script(script_name, model_type):
    """Run a training script and capture the output"""
    print(f"\n{'='*60}")
    print(f"🚀 STARTING {model_type.upper()} TRAINING")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the training script
        result = subprocess.run(
            ["python", script_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"✅ {model_type} training completed successfully!")
        print(f"⏱️  Training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        
        # Extract model directory from output
        output_lines = result.stdout.split('\n')
        model_dir = None
        for line in output_lines:
            if "Model saved to:" in line:
                model_dir = line.split("Model saved to:")[-1].strip()
                break
        
        return {
            'success': True,
            'model_dir': model_dir,
            'training_time': training_time,
            'output': result.stdout
        }
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"❌ {model_type} training failed!")
        print(f"⏱️  Failed after: {training_time:.2f} seconds")
        print(f"Error: {e.stderr}")
        
        return {
            'success': False,
            'model_dir': None,
            'training_time': training_time,
            'error': e.stderr
        }

def main():
    """Run both training scripts and compare results"""
    
    print("🎯 BART MODEL COMPARISON TRAINING")
    print("Training both Standard and Weighted BART models")
    print("=" * 70)
    
    # Record start time
    overall_start = time.time()
    
    # Results storage
    results = {}
    
    # Train Standard BART (without weighted loss)
    print("\n🔵 Phase 1: Training Standard BART (baseline)")
    results['standard'] = run_training_script("standard_bart_finetune.py", "Standard BART")
    
    # Train Weighted BART (with weighted loss)
    print("\n🟡 Phase 2: Training Weighted BART (class-balanced)")
    results['weighted'] = run_training_script("weighted_bart_finetune.py", "Weighted BART")
    
    # Calculate total time
    overall_end = time.time()
    total_time = overall_end - overall_start
    
    # Generate comparison report
    print(f"\n{'='*70}")
    print("📊 TRAINING COMPARISON REPORT")
    print(f"{'='*70}")
    
    print(f"\\n⏱️  TIMING COMPARISON:")
    if results['standard']['success']:
        print(f"  Standard BART: {results['standard']['training_time']:.2f}s ({results['standard']['training_time']/60:.1f}m)")
    else:
        print(f"  Standard BART: FAILED")
        
    if results['weighted']['success']:
        print(f"  Weighted BART: {results['weighted']['training_time']:.2f}s ({results['weighted']['training_time']/60:.1f}m)")
    else:
        print(f"  Weighted BART: FAILED")
        
    print(f"  Total Time: {total_time:.2f}s ({total_time/60:.1f}m)")
    
    print(f"\\n📁 MODEL DIRECTORIES:")
    if results['standard']['success']:
        print(f"  Standard BART: {results['standard']['model_dir']}")
    if results['weighted']['success']:
        print(f"  Weighted BART: {results['weighted']['model_dir']}")
    
    print(f"\\n🎯 SUCCESS STATUS:")
    standard_status = "✅ SUCCESS" if results['standard']['success'] else "❌ FAILED"
    weighted_status = "✅ SUCCESS" if results['weighted']['success'] else "❌ FAILED"
    print(f"  Standard BART: {standard_status}")
    print(f"  Weighted BART: {weighted_status}")
    
    # Save results to CSV
    comparison_data = []
    for model_type, result in results.items():
        comparison_data.append({
            'model_type': model_type,
            'success': result['success'],
            'training_time_seconds': result['training_time'],
            'training_time_minutes': result['training_time'] / 60,
            'model_directory': result.get('model_dir', 'N/A'),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = f"model_comparison_results_{timestamp}.csv"
    comparison_df.to_csv(comparison_file, index=False)
    
    print(f"\\n💾 Comparison results saved to: {comparison_file}")
    
    # Next steps
    if results['standard']['success'] and results['weighted']['success']:
        print(f"\\n🎉 BOTH MODELS TRAINED SUCCESSFULLY!")
        print(f"\\n📋 NEXT STEPS:")
        print(f"  1. Run evaluation comparison between the two models")
        print(f"  2. Analyze performance differences on minority classes")
        print(f"  3. Compare zero-shot vs fine-tuned vs weighted performance")
        print(f"\\n💡 To evaluate both models, update the evaluation script with both model paths:")
        print(f"     - Standard: {results['standard']['model_dir']}")
        print(f"     - Weighted: {results['weighted']['model_dir']}")
    else:
        print(f"\\n⚠️  Some models failed to train. Check the error messages above.")

if __name__ == "__main__":
    main()
