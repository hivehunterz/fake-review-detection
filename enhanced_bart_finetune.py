#!/usr/bin/env python3
"""
Enhanced BART Fine-tuning for Google Reviews Classification
Fine-tune facebook/bart-large-mnli on the enhanced labeled dataset
"""

import pandas as pd
import numpy as np
import torch
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedBARTFineTuner:
    """
    Enhanced BART Fine-tuner for Google Reviews with improved dataset
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_to_id = {}
        self.id_to_label = {}
        
        # Labels based on the enhanced dataset
        self.labels = [
            'genuine_positive',
            'genuine_negative', 
            'spam',
            'advertisement',
            'irrelevant',
            'fake_rant',
            'inappropriate'
        ]
        
        # Create label mappings
        for i, label in enumerate(self.labels):
            self.label_to_id[label] = i
            self.id_to_label[i] = label
        
        logger.info(f"Label mappings: {self.label_to_id}")
    
    def load_and_prepare_data(self, csv_path: str) -> Tuple[Dataset, Dataset]:
        """Load and prepare the enhanced dataset"""
        logger.info(f"Loading enhanced data from {csv_path}")
        
        # Read the CSV
        df = pd.read_csv(csv_path)
        
        # Filter out rows with missing text or labels
        df = df.dropna(subset=['text', 'llm_classification'])
        
        # Filter to only include our defined labels
        df = df[df['llm_classification'].isin(self.labels)]
        
        logger.info(f"Loaded {len(df)} valid reviews")
        
        # Log label distribution
        logger.info("Enhanced label distribution:")
        for label in self.labels:
            count = len(df[df['llm_classification'] == label])
            percentage = (count / len(df)) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        
        # Convert labels to numeric
        df['label'] = df['llm_classification'].map(self.label_to_id)
        
        # Split into train/eval
        train_df, eval_df = train_test_split(
            df[['text', 'label']], 
            test_size=0.2, 
            random_state=42,
            stratify=df['label']
        )
        
        logger.info(f"Train size: {len(train_df)}, Eval size: {len(eval_df)}")
        
        # Convert to datasets
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)
        
        return train_dataset, eval_dataset
    
    def initialize_model(self):
        """Initialize tokenizer and model"""
        logger.info(f"Initializing enhanced BART model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model with correct number of labels and ignore size mismatches
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.labels),
            id2label=self.id_to_label,
            label2id=self.label_to_id,
            ignore_mismatched_sizes=True  # Allow different classifier head sizes
        )
        
        logger.info(f"Enhanced BART model initialized with {len(self.labels)} labels")
    
    def tokenize_function(self, examples):
        """Tokenize text examples with proper padding and truncation"""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',  # Use fixed padding for consistency
            max_length=512,
            return_tensors=None  # Let the data collator handle tensor conversion
        )
    
    def compute_metrics(self, eval_pred):
        """Compute enhanced evaluation metrics"""
        predictions, labels = eval_pred
        
        # Debug: Check the shape and structure of predictions
        print(f"Predictions type: {type(predictions)}")
        print(f"Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else 'No shape attribute'}")
        
        # Handle different prediction formats
        if isinstance(predictions, tuple):
            # If predictions is a tuple, take the first element (logits)
            predictions = predictions[0]
        
        # Ensure predictions is a proper numpy array
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        
        # If predictions has more than 2 dimensions, we need to handle it differently
        if len(predictions.shape) > 2:
            # For sequence classification, we typically want the last dimension
            # Reshape to (batch_size, num_classes)
            predictions = predictions.reshape(-1, predictions.shape[-1])
        
        # Now safely apply argmax
        predictions = np.argmax(predictions, axis=1)
        
        # Debug: Check labels
        print(f"Labels type: {type(labels)}")
        print(f"Labels shape: {labels.shape if hasattr(labels, 'shape') else 'No shape attribute'}")
        
        # Ensure labels is a proper numpy array
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        
        # Flatten labels if needed
        if len(labels.shape) > 1:
            labels = labels.flatten()
        
        # Convert back to label names
        try:
            pred_labels = [self.id_to_label[pred] for pred in predictions]
            true_labels = [self.id_to_label[label] for label in labels]
        except KeyError as e:
            print(f"KeyError in label conversion: {e}")
            print(f"Available label IDs: {list(self.id_to_label.keys())}")
            print(f"Prediction IDs: {set(predictions)}")
            print(f"Label IDs: {set(labels)}")
            # Use safe conversion with fallback
            pred_labels = [self.id_to_label.get(pred, 'unknown') for pred in predictions]
            true_labels = [self.id_to_label.get(label, 'unknown') for label in labels]
        
        # Calculate accuracy
        accuracy = accuracy_score(labels, predictions)
        
        # Get detailed classification report
        report = classification_report(
            true_labels, pred_labels, 
            target_names=self.labels,
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'f1_macro': report['macro avg']['f1-score'],
            'f1_weighted': report['weighted avg']['f1-score']
        }
    
    def fine_tune(self, train_dataset: Dataset, eval_dataset: Dataset, 
                  output_dir: str, epochs: int = 3, batch_size: int = 8, 
                  learning_rate: float = 1e-5) -> str:
        """Fine-tune the enhanced BART model"""
        
        # Tokenize datasets
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(self.tokenize_function, batched=True)
        
        # Set format for PyTorch
        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        
        # Data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            learning_rate=learning_rate,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_strategy="steps",  # Fixed parameter name
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            remove_unused_columns=True,
            push_to_hub=False,
            report_to=None
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        logger.info("Starting enhanced BART training...")
        
        try:
            # Train the model
            trainer.train()
            
            # Save the final model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            # Save label mappings
            with open(f"{output_dir}/label_mappings.json", "w") as f:
                json.dump({
                    "label_to_id": self.label_to_id,
                    "id_to_label": self.id_to_label,
                    "labels": self.labels
                }, f, indent=2)
            
            logger.info(f"Enhanced BART model saved to {output_dir}")
            
            # Final evaluation
            final_metrics = trainer.evaluate()
            logger.info("Final evaluation metrics:")
            for key, value in final_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
            
            return output_dir
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

def main():
    """Main training function"""
    print("🎯 ENHANCED BART FINE-TUNING FOR GOOGLE REVIEWS")
    print("Fine-tuning facebook/bart-large-mnli on enhanced labeled dataset")
    print("=" * 65)
    
    # Fixed configuration for enhanced dataset
    csv_path = "google_reviews_labeled_combined_with_json.csv"
    epochs = 3
    batch_size = 8
    learning_rate = 1e-5
    
    print(f"\n🎯 ENHANCED TRAINING CONFIGURATION:")
    print(f"  Model: facebook/bart-large-mnli")
    print(f"  Dataset: {csv_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Enhanced dataset: 1,209 reviews with improved inappropriate/advertisement content")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"enhanced_bart_review_classifier_{timestamp}"
    
    try:
        # Initialize fine-tuner
        fine_tuner = EnhancedBARTFineTuner()
        
        # Load and prepare data
        train_dataset, eval_dataset = fine_tuner.load_and_prepare_data(csv_path)
        
        # Initialize model
        fine_tuner.initialize_model()
        
        # Fine-tune
        logger.info(f"Starting enhanced BART fine-tuning for {epochs} epochs")
        logger.info(f"Output directory: {output_dir}")
        
        model_path = fine_tuner.fine_tune(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        print(f"\n✅ Enhanced BART fine-tuning completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        print(f"🎯 Ready for enhanced evaluation and deployment!")
        
    except Exception as e:
        print(f"\n❌ Enhanced fine-tuning failed: {str(e)}")
        logger.error(f"Enhanced fine-tuning failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
