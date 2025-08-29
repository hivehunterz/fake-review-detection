#!/usr/bin/env python3
"""
Working BART Fine-tuning for Enhanced Google Reviews Dataset
Properly handles tokenization and dataset formatting
"""

import pandas as pd
import numpy as np
import torch
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkingBARTFineTuner:
    """
    Working BART Fine-tuner that properly handles the enhanced dataset
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.labels = [
            'genuine_positive',
            'genuine_negative', 
            'spam',
            'advertisement',
            'irrelevant',
            'fake_rant',
            'inappropriate'
        ]
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        self.id_to_label = {i: label for i, label in enumerate(self.labels)}
        
    def prepare_dataset(self, csv_path: str):
        """Prepare dataset for training"""
        logger.info(f"Loading enhanced dataset from {csv_path}")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['text', 'llm_classification'])
        df = df[df['llm_classification'].isin(self.labels)]
        
        logger.info(f"Loaded {len(df)} valid reviews")
        
        # Log distribution
        for label in self.labels:
            count = len(df[df['llm_classification'] == label])
            percentage = (count / len(df)) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        
        # Prepare data
        texts = df['text'].tolist()
        labels = [self.label_to_id[label] for label in df['llm_classification']]
        
        # Split data
        train_texts, eval_texts, train_labels, eval_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        logger.info(f"Train size: {len(train_texts)}, Eval size: {len(eval_texts)}")
        
        return train_texts, eval_texts, train_labels, eval_labels
    
    def initialize_model(self):
        """Initialize tokenizer and model"""
        logger.info(f"Initializing BART model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model with ignore_mismatched_sizes to handle different number of labels
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.labels),
            id2label=self.id_to_label,
            label2id=self.label_to_id,
            ignore_mismatched_sizes=True  # This allows different output layer sizes
        )
        
        logger.info(f"Model initialized with {len(self.labels)} labels")
    
    def tokenize_texts(self, texts, labels):
        """Tokenize texts properly"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item
            
            def __len__(self):
                return len(self.labels)
        
        return SimpleDataset(encodings, labels)
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        
        pred_labels = [self.id_to_label[pred] for pred in predictions]
        true_labels = [self.id_to_label[label] for label in labels]
        
        report = classification_report(
            true_labels, pred_labels, 
            target_names=self.labels,
            output_dict=True,
            zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'f1_macro': report['macro avg']['f1-score'],
            'f1_weighted': report['weighted avg']['f1-score']
        }
    
    def fine_tune(self, train_texts, eval_texts, train_labels, eval_labels, 
                  output_dir: str, epochs: int = 3, batch_size: int = 8, 
                  learning_rate: float = 1e-5):
        """Fine-tune the model"""
        
        # Tokenize datasets
        train_dataset = self.tokenize_texts(train_texts, train_labels)
        eval_dataset = self.tokenize_texts(eval_texts, eval_labels)
        
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
            logging_steps=20,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )
        
        logger.info("Starting training...")
        
        try:
            # Train
            trainer.train()
            
            # Save
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            # Save label mappings
            with open(f"{output_dir}/label_mappings.json", "w") as f:
                json.dump({
                    "label_to_id": self.label_to_id,
                    "id_to_label": self.id_to_label,
                    "labels": self.labels
                }, f, indent=2)
            
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
    """Main function"""
    print("🎯 WORKING BART FINE-TUNING FOR ENHANCED GOOGLE REVIEWS")
    print("Fine-tuning facebook/bart-large-mnli on enhanced dataset (1,209 reviews)")
    print("=" * 70)
    
    # Configuration
    csv_path = "google_reviews_labeled_combined.csv"
    epochs = 3
    batch_size = 8
    learning_rate = 1e-5
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"working_bart_enhanced_{timestamp}"
    
    print(f"\n🎯 CONFIGURATION:")
    print(f"  Dataset: {csv_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Output: {output_dir}")
    
    try:
        # Initialize
        fine_tuner = WorkingBARTFineTuner()
        
        # Prepare data
        train_texts, eval_texts, train_labels, eval_labels = fine_tuner.prepare_dataset(csv_path)
        
        # Initialize model
        fine_tuner.initialize_model()
        
        # Fine-tune
        logger.info("Starting fine-tuning process")
        model_path = fine_tuner.fine_tune(
            train_texts, eval_texts, train_labels, eval_labels,
            output_dir, epochs, batch_size, learning_rate
        )
        
        print(f"\n✅ Enhanced BART fine-tuning completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        print(f"🎯 Ready for evaluation with improved inappropriate/advertisement detection!")
        
    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {str(e)}")
        logger.error(f"Fine-tuning failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
