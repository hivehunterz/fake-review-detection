#!/usr/bin/env python3
"""
Enhanced BART Review Classifier
Production-ready BART classifier for review quality detection.
"""

import torch
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    pipeline
)
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class BARTReviewClassifier:
    """
    Enhanced BART classifier for review quality detection.
    Supports both fine-tuned and zero-shot classification.
    """
    
    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize BART classifier.
        
        Args:
            model_path: Path to fine-tuned model, or None for zero-shot
            use_gpu: Whether to use GPU acceleration
        """
        self.model_path = model_path
        self.device = 0 if (use_gpu and torch.cuda.is_available()) else -1
        self.use_fine_tuned = model_path is not None
        
        # Standard 7-class labels
        self.labels = [
            'genuine_positive',
            'genuine_negative', 
            'spam',
            'advertisement',
            'irrelevant',
            'fake_rant',
            'inappropriate'
        ]
        
        # Initialize model
        self._load_model()
        
        logger.info(f"BARTReviewClassifier initialized")
        logger.info(f"Model type: {'Fine-tuned' if self.use_fine_tuned else 'Zero-shot'}")
        logger.info(f"Device: {'GPU' if self.device >= 0 else 'CPU'}")
    
    def _load_model(self):
        """Load BART model (fine-tuned or zero-shot)"""
        if self.use_fine_tuned:
            self._load_fine_tuned_model()
        else:
            self._load_zero_shot_model()
    
    def _load_fine_tuned_model(self):
        """Load fine-tuned BART model"""
        logger.info(f"Loading fine-tuned BART model from {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            
            # Load label mappings if available
            mappings_path = os.path.join(self.model_path, "label_mappings.json")
            if os.path.exists(mappings_path):
                with open(mappings_path, 'r') as f:
                    mappings = json.load(f)
                    self.id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}
                    self.label_to_id = mappings['label_to_id']
                    self.labels = mappings['labels']
            else:
                # Fallback to default mappings
                self.label_to_id = {label: idx for idx, label in enumerate(self.labels)}
                self.id_to_label = {idx: label for idx, label in enumerate(self.labels)}
            
            # Move model to device
            if self.device >= 0:
                self.model = self.model.cuda()
            
            self.model.eval()
            logger.info(f"Fine-tuned model loaded with {len(self.labels)} labels")
            
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            logger.info("Falling back to zero-shot model")
            self.use_fine_tuned = False
            self._load_zero_shot_model()
    
    def _load_zero_shot_model(self):
        """Load zero-shot BART model"""
        logger.info("Loading zero-shot BART model")
        
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device
        )
        
        # Categories for zero-shot classification
        self.zero_shot_categories = [
            "genuine positive review",
            "genuine negative review", 
            "spam",
            "advertisement",
            "irrelevant content",
            "fake rant",
            "inappropriate content"
        ]
        
        # Mapping from zero-shot labels to standard labels
        self.zero_shot_label_map = {
            'genuine positive review': 'genuine_positive',
            'genuine negative review': 'genuine_negative',
            'spam': 'spam',
            'advertisement': 'advertisement',
            'irrelevant content': 'irrelevant',
            'fake rant': 'fake_rant',
            'inappropriate content': 'inappropriate'
        }
        
        logger.info("Zero-shot model loaded")
    
    def predict(self, texts: Union[str, List[str]], batch_size: int = 16) -> List[Dict]:
        """
        Predict review quality for text(s).
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.use_fine_tuned:
            return self._predict_fine_tuned(texts, batch_size)
        else:
            return self._predict_zero_shot(texts, batch_size)
    
    def _predict_fine_tuned(self, texts: List[str], batch_size: int) -> List[Dict]:
        """Predict using fine-tuned model"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move to device
            if self.device >= 0:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            # Process batch results
            for j, text in enumerate(batch_texts):
                prob_tensor = probs[j]
                
                # Get top prediction
                top_idx = torch.argmax(prob_tensor).item()
                confidence = prob_tensor[top_idx].item()
                prediction = self.id_to_label[top_idx]
                
                # Get all class probabilities
                class_probs = {}
                for idx, prob in enumerate(prob_tensor):
                    label = self.id_to_label[idx]
                    class_probs[label] = prob.item()
                
                # Calculate p_bad score (spam-related probability)
                spam_labels = ['spam', 'advertisement', 'irrelevant', 'fake_rant', 'inappropriate']
                p_bad = sum([class_probs.get(label, 0.0) for label in spam_labels])
                
                results.append({
                    'text': text,
                    'prediction': prediction,
                    'confidence': confidence,
                    'class_probabilities': class_probs,
                    'p_bad': p_bad,
                    'model_type': 'fine_tuned'
                })
        
        return results
    
    def _predict_zero_shot(self, texts: List[str], batch_size: int) -> List[Dict]:
        """Predict using zero-shot model"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            for text in batch_texts:
                try:
                    # Zero-shot classification
                    result = self.classifier(text, self.zero_shot_categories)
                    
                    # Get top prediction
                    top_label = result['labels'][0]
                    confidence = result['scores'][0]
                    prediction = self.zero_shot_label_map.get(top_label, 'spam')
                    
                    # Build class probabilities
                    class_probs = {}
                    for label, score in zip(result['labels'], result['scores']):
                        mapped_label = self.zero_shot_label_map.get(label, 'spam')
                        class_probs[mapped_label] = score
                    
                    # Ensure all labels have probabilities
                    for label in self.labels:
                        if label not in class_probs:
                            class_probs[label] = 0.0
                    
                    # Calculate p_bad score
                    spam_labels = ['spam', 'advertisement', 'irrelevant', 'fake_rant', 'inappropriate']
                    p_bad = sum([class_probs.get(label, 0.0) for label in spam_labels])
                    
                    results.append({
                        'text': text,
                        'prediction': prediction,
                        'confidence': confidence,
                        'class_probabilities': class_probs,
                        'p_bad': p_bad,
                        'model_type': 'zero_shot'
                    })
                    
                except Exception as e:
                    logger.error(f"Error in zero-shot prediction: {e}")
                    # Fallback result
                    results.append({
                        'text': text,
                        'prediction': 'spam',
                        'confidence': 0.0,
                        'class_probabilities': {label: 0.0 for label in self.labels},
                        'p_bad': 1.0,
                        'model_type': 'zero_shot_fallback'
                    })
        
        return results
    
    def predict_single(self, text: str) -> Dict:
        """
        Predict quality for a single review.
        
        Args:
            text: Review text
            
        Returns:
            Prediction dictionary
        """
        results = self.predict([text])
        return results[0]
    
    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        """
        Predict quality for multiple reviews.
        
        Args:
            texts: List of review texts
            batch_size: Processing batch size
            
        Returns:
            List of prediction dictionaries
        """
        return self.predict(texts, batch_size)
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_type': 'fine_tuned' if self.use_fine_tuned else 'zero_shot',
            'model_path': self.model_path,
            'device': 'GPU' if self.device >= 0 else 'CPU',
            'labels': self.labels,
            'num_labels': len(self.labels)
        }

def main():
    """Demo usage"""
    print("🤖 BART Review Classifier Demo")
    print("="*50)
    
    # Initialize classifier
    classifier = BARTReviewClassifier()
    
    # Test texts
    test_texts = [
        "Amazing food and excellent service! Highly recommend.",
        "Terrible experience, food was cold and service was awful.",
        "Best restaurant ever! Everyone should eat here!!! 5 stars!!!"
    ]
    
    # Predict
    results = classifier.predict(test_texts)
    
    # Display results
    for i, result in enumerate(results):
        print(f"\n📝 Text {i+1}: {result['text'][:50]}...")
        print(f"🎯 Prediction: {result['prediction']}")
        print(f"🔥 Confidence: {result['confidence']:.3f}")
        print(f"⚠️ P_BAD Score: {result['p_bad']:.3f}")
        print(f"📊 Top 3 Classes:")
        sorted_probs = sorted(result['class_probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        for label, prob in sorted_probs:
            print(f"   {label}: {prob:.3f}")

if __name__ == "__main__":
    main()
