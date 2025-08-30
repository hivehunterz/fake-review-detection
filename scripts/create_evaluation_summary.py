"""
📊 STAGE EVALUATION RESULTS SUMMARY
Generated from comprehensive stage evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_summary_report():
    """Create a clean summary report from the detailed CSV"""
    
    # Load the detailed results
    csv_path = Path("output/comprehensive_stage_evaluation.csv")
    df = pd.read_csv(csv_path)
    
    # Create clean summary DataFrame
    summary_data = []
    
    # Stage 1 - BART
    stage1 = df[df['stage'] == 'Stage 1 - BART'].iloc[0]
    summary_data.append({
        'Stage': 'Stage 1 - BART Classifier',
        'Accuracy': f"{stage1['accuracy']:.3f}",
        'Macro_F1': f"{stage1['macro_f1']:.3f}",
        'Weighted_F1': f"{stage1['weighted_f1']:.3f}",
        'PR_AUC_Macro': f"{stage1['pr_auc_macro']:.3f}",
        'Precision': f"{stage1['macro_precision']:.3f}",
        'Recall': f"{stage1['macro_recall']:.3f}",
        'Classes': int(stage1['num_classes']),
        'Samples': int(stage1['total_samples']),
        'Additional_Info': f"Avg Confidence: {stage1['avg_confidence']:.3f}"
    })
    
    # Stage 2 - Metadata
    stage2 = df[df['stage'] == 'Stage 2 - Metadata Analyzer'].iloc[0]
    summary_data.append({
        'Stage': 'Stage 2 - Metadata Analyzer',
        'Accuracy': f"{stage2['accuracy']:.3f}",
        'Macro_F1': f"{stage2['f1_score']:.3f}",
        'Weighted_F1': 'N/A (Binary)',
        'PR_AUC_Macro': f"{stage2['pr_auc']:.3f}",
        'Precision': f"{stage2['precision']:.3f}",
        'Recall': f"{stage2['recall']:.3f}",
        'Classes': 2,
        'Samples': int(stage2['total_samples']),
        'Additional_Info': f"Anomaly Rate: {stage2['anomaly_rate']:.1%}, Features: {int(stage2['features_count'])}"
    })
    
    # Stage 3 - Fusion
    stage3 = df[df['stage'] == 'Stage 3 - Fusion'].iloc[0]
    summary_data.append({
        'Stage': 'Stage 3 - Fusion Model',
        'Accuracy': f"{stage3['accuracy']:.3f}",
        'Macro_F1': f"{stage3['macro_f1']:.3f}",
        'Weighted_F1': f"{stage3['weighted_f1']:.3f}",
        'PR_AUC_Macro': f"{stage3['pr_auc_macro']:.3f}",
        'Precision': f"{stage3['macro_precision']:.3f}",
        'Recall': f"{stage3['macro_recall']:.3f}",
        'Classes': int(stage3['num_classes']),
        'Samples': int(stage3['total_samples']),
        'Additional_Info': f"Meta Categories: {int(stage3['fusion_categories'])}"
    })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_path = Path("output/stage_evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Create per-class performance table for BART
    bart_classes = ['genuine_positive', 'genuine_negative', 'spam', 'advertisement', 
                   'irrelevant', 'fake_rant', 'inappropriate']
    
    bart_performance = []
    for cls in bart_classes:
        precision = stage1.get(f'{cls}_precision', 0)
        recall = stage1.get(f'{cls}_recall', 0)
        f1 = stage1.get(f'{cls}_f1', 0)
        support = stage1.get(f'{cls}_support', 0)
        pr_auc = stage1.get(f'{cls}_pr_auc', 0)
        
        bart_performance.append({
            'Class': cls.replace('_', ' ').title(),
            'Precision': f"{precision:.3f}",
            'Recall': f"{recall:.3f}",
            'F1_Score': f"{f1:.3f}",
            'PR_AUC': f"{pr_auc:.3f}",
            'Support': int(support) if not pd.isna(support) else 0
        })
    
    bart_df = pd.DataFrame(bart_performance)
    bart_path = Path("output/bart_per_class_performance.csv")
    bart_df.to_csv(bart_path, index=False)
    
    # Create fusion performance table
    fusion_classes = ['genuine', 'suspicious', 'low-quality', 'high-confidence-spam']
    fusion_performance = []
    
    for cls in fusion_classes:
        precision = stage3.get(f'{cls}_precision', 0)
        recall = stage3.get(f'{cls}_recall', 0)
        f1 = stage3.get(f'{cls}_f1', 0)
        support = stage3.get(f'{cls}_support', 0)
        
        fusion_performance.append({
            'Fusion_Category': cls.replace('-', ' ').replace('_', ' ').title(),
            'Precision': f"{precision:.3f}",
            'Recall': f"{recall:.3f}",
            'F1_Score': f"{f1:.3f}",
            'Support': int(support) if not pd.isna(support) else 0
        })
    
    fusion_df = pd.DataFrame(fusion_performance)
    fusion_path = Path("output/fusion_per_class_performance.csv")
    fusion_df.to_csv(fusion_path, index=False)
    
    print("📊 COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    
    print("\n🤖 BART CLASSIFIER - PER CLASS PERFORMANCE")
    print("=" * 60)
    print(bart_df.to_string(index=False))
    
    print("\n🔮 FUSION MODEL - PER CATEGORY PERFORMANCE")
    print("=" * 60)
    print(fusion_df.to_string(index=False))
    
    print(f"\n📄 Files created:")
    print(f"   Summary: {summary_path}")
    print(f"   BART Details: {bart_path}")
    print(f"   Fusion Details: {fusion_path}")
    print(f"   Full Details: {csv_path}")

if __name__ == "__main__":
    create_summary_report()
