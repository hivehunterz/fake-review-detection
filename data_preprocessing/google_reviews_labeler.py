#!/usr/bin/env python3
"""
Google Reviews Multi-LLM Labeling System
Uses multiple LLM providers (OpenAI GPT-4, Claude, Groq) to classify Google Reviews
Optimized for fast model switching on rate limits
"""

import json
import pandas as pd
import time
import logging
import random
from datetime import datetime
from typing import Dict, List, Optional
import requests
import os
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LLMProvider:
    """Configuration for an LLM provider"""
    name: str
    api_key: str
    base_url: str
    model: str
    max_tokens: int
    rate_limit_delay: float
    max_retries: int

class GoogleReviewsLabeler:
    """
    Multi-LLM Google Reviews classifier with fast model switching
    """
    
    def __init__(self):
        """Initialize with multiple LLM providers for redundancy and speed"""
        
        # Define LLM providers with fast switching configuration - Groq first for speed
        self.providers = [
            LLMProvider(
                name="Groq Llama",
                api_key=os.getenv('GROQ_API_KEY', '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'),
                base_url="https://api.groq.com/openai/v1/chat/completions",
                model="llama-3.3-70b-versatile",  # Fast logical model first
                max_tokens=1000,
                rate_limit_delay=0.4,  # Very fast
                max_retries=2
            ),
            LLMProvider(
                name="Groq Mixtral",
                api_key=os.getenv('GROQ_API_KEY', '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'),
                base_url="https://api.groq.com/openai/v1/chat/completions",
                model="llama3-groq-70b-8192-tool-use-preview",  # Alternative model (replaced decommissioned mixtral)
                max_tokens=1000,
                rate_limit_delay=0.3,  # Very fast
                max_retries=2
            ),
            LLMProvider(
                name="OpenAI GPT-4",
                api_key=os.getenv('OPENAI_API_KEY', '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'),
                base_url="https://api.openai.com/v1/chat/completions",
                model="gpt-4",  # Full GPT-4 as fallback only
                max_tokens=1500,
                rate_limit_delay=2.0,  # Slower fallback
                max_retries=1
            ),
            LLMProvider(
                name="OpenAI GPT-4-Turbo",
                api_key=os.getenv('OPENAI_API_KEY', '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'),
                base_url="https://api.openai.com/v1/chat/completions",
                model="gpt-4-turbo",  # GPT-4 Turbo as final fallback
                max_tokens=1500,
                rate_limit_delay=2.0,
                max_retries=1
            )
        ]
        
        # Filter to only active providers (those with API keys)
        self.active_providers = [p for p in self.providers if p.api_key and p.api_key != 'your_api_key_here']
        
        if not self.active_providers:
            raise ValueError("No valid API keys found for any LLM provider")
        
        logger.info(f"Initialized with {len(self.active_providers)} LLM providers: {[p.name for p in self.active_providers]}")
    
    def create_labeling_prompt(self, reviews_batch: List[str]) -> str:
        """Create a comprehensive prompt for review classification"""
        
        prompt = """You are an expert at identifying genuine vs fake/problematic Google reviews. 

CLASSIFICATION CATEGORIES:
1. genuine_positive - Authentic positive experiences
2. genuine_negative - Authentic negative experiences with specific issues
3. spam - Generic, repetitive, or clearly fake content
4. advertisement - Promotional content disguised as reviews
5. irrelevant - Off-topic or unrelated content
6. fake_rant - Emotionally extreme, unbalanced negative reviews
7. inappropriate - Offensive, discriminatory, or harmful content

ANALYSIS GUIDELINES:
- Look for specific, detailed experiences vs generic language
- Check for balanced perspectives vs extreme emotions
- Identify promotional language or obvious marketing
- Consider review length, specificity, and authenticity markers
- Flag reviews with inappropriate content or discrimination

For each review, provide:
- classification: (one of the 7 categories above)
- confidence: (0.0-1.0)
- reasoning: (brief explanation in under 100 chars)

REVIEWS TO CLASSIFY:
"""
        
        for i, review in enumerate(reviews_batch, 1):
            prompt += f"\n{i}. \"{review[:1000]}{'...' if len(review) > 1000 else ''}\"\n"
        
        prompt += """
RESPOND IN THIS EXACT JSON FORMAT:
[
    {"classification": "category", "confidence": 0.85, "reasoning": "brief explanation"},
    {"classification": "category", "confidence": 0.90, "reasoning": "brief explanation"}
]"""
        
        return prompt
    
    def call_openai_api(self, provider: LLMProvider, prompt: str) -> Optional[Dict]:
        """Call OpenAI-compatible API"""
        
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": provider.model,
            "messages": [
                {"role": "system", "content": "You are an expert review classifier. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": provider.max_tokens,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(provider.base_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                return {"content": data["choices"][0]["message"]["content"]}
            elif response.status_code == 429:
                error_data = response.json() if response.content else {"error": {"message": "Rate limit exceeded"}}
                logger.error(f"{provider.name} API error {response.status_code}: {error_data}")
                raise Exception(f"rate_limit_exceeded: {error_data}")
            elif response.status_code == 401:
                error_data = response.json() if response.content else {"error": {"message": "Invalid API key"}}
                logger.error(f"{provider.name} API error {response.status_code}: {error_data}")
                raise Exception(f"invalid_api_key: {error_data}")
            else:
                error_data = response.json() if response.content else {"error": {"message": f"HTTP {response.status_code}"}}
                logger.error(f"{provider.name} API error {response.status_code}: {error_data}")
                raise Exception(f"API error {response.status_code}: {error_data}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"{provider.name} request error: {str(e)}")
            raise Exception(f"Request error: {str(e)}")
    
    def call_llm_api(self, provider: LLMProvider, prompt: str) -> Optional[Dict]:
        """Generic LLM API caller"""
        return self.call_openai_api(provider, prompt)
    
    def parse_llm_response(self, response_text: str, expected_count: int) -> List[Dict]:
        """Parse LLM response and extract classifications"""
        
        try:
            # Clean up the response
            response_text = response_text.strip()
            
            # Look for JSON array in the response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                results = json.loads(json_str)
                
                if isinstance(results, list) and len(results) == expected_count:
                    # Validate and clean results
                    valid_categories = ['genuine_positive', 'genuine_negative', 'spam', 'advertisement', 'irrelevant', 'fake_rant', 'inappropriate']
                    
                    for result in results:
                        if 'classification' not in result or result['classification'] not in valid_categories:
                            result['classification'] = 'spam'  # Default fallback
                        if 'confidence' not in result or not isinstance(result['confidence'], (int, float)):
                            result['confidence'] = 0.5
                        if 'reasoning' not in result:
                            result['reasoning'] = 'Automatic classification'
                    
                    return results
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
        
        # Return default classifications if parsing fails
        return [
            {
                'classification': 'spam',
                'confidence': 0.0,
                'reasoning': 'Parse error - default classification'
            }
            for _ in range(expected_count)
        ]
    
    def label_reviews_batch(self, reviews_batch: List[str]) -> List[Dict]:
        """Label a batch of reviews using available LLM providers with fast switching on rate limits"""
        
        prompt = self.create_labeling_prompt(reviews_batch)
        
        # Try each provider in order, switching immediately on rate limits
        for provider in self.active_providers:
            logger.info(f"Trying {provider.name} for batch of {len(reviews_batch)} reviews...")
            
            try:
                response = self.call_llm_api(provider, prompt)
                
                if response:
                    results = self.parse_llm_response(response['content'], len(reviews_batch))
                    
                    if results and len(results) == len(reviews_batch):
                        logger.info(f"Successfully labeled batch with {provider.name}")
                        
                        # Add provider info to results
                        for r in results:
                            r['llm_provider'] = provider.name
                        
                        return results
                
            except Exception as e:
                error_str = str(e).lower()
                if any(rate_term in error_str for rate_term in ['rate limit', 'rate_limit_exceeded', '429']):
                    logger.warning(f"{provider.name} hit rate limit, switching to next provider immediately")
                    continue  # Switch immediately on rate limit
                elif any(auth_term in error_str for auth_term in ['401', 'invalid_api_key', 'unauthorized']):
                    logger.warning(f"{provider.name} has authentication issue, switching to next provider")
                    continue  # Switch immediately on auth issues
                else:
                    logger.error(f"Error with {provider.name}: {str(e)}")
                    # For other errors, do one retry with delay
                    time.sleep(provider.rate_limit_delay)
                    try:
                        response = self.call_llm_api(provider, prompt)
                        if response:
                            results = self.parse_llm_response(response['content'], len(reviews_batch))
                            if results and len(results) == len(reviews_batch):
                                logger.info(f"Successfully labeled batch with {provider.name} on retry")
                                for r in results:
                                    r['llm_provider'] = provider.name
                                return results
                    except Exception as retry_e:
                        logger.error(f"Retry failed for {provider.name}: {str(retry_e)}")
            
            logger.warning(f"{provider.name} failed, trying next provider")
        
        # If all providers fail, return default labels
        logger.error("All LLM providers failed, using default labels")
        return [{'classification': 'spam', 'confidence': 0.0, 'reasoning': 'All providers failed', 'llm_provider': 'none'} for _ in reviews_batch]
    
    def process_google_reviews(self, json_file_path: str, output_file: str = None, max_reviews: int = None) -> pd.DataFrame:
        """Process Google Reviews JSON file and label reviews"""
        
        logger.info(f"Loading Google Reviews from {json_file_path}")
        
        # Load JSON data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            reviews_data = json.load(f)
        
        logger.info(f"Loaded {len(reviews_data)} reviews from Google dataset")
        
        # Convert to DataFrame and extract text
        df = pd.DataFrame(reviews_data)
        
        # Filter reviews with valid text content
        df = df[df['text'].notna()]
        df = df[df['text'].astype(str).str.strip().str.len() > 0]
        
        if max_reviews and len(df) > max_reviews:
            df = df.head(max_reviews)
        
        review_texts = df['text'].astype(str).tolist()
        
        logger.info(f"Processing {len(review_texts)} reviews with valid text content")
        
        # Process in batches optimized for accuracy
        batch_size = 8  # Smaller batches for better accuracy
        total_batches = (len(review_texts) + batch_size - 1) // batch_size
        
        logger.info(f"Processing {len(review_texts)} reviews in {total_batches} batches of {batch_size}")
        
        all_results = []
        start_time = time.time()
        
        # Progress file for saving intermediate results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        progress_file = f'google_reviews_progress_{timestamp}.csv'
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(review_texts))
            
            batch_texts = review_texts[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches}: reviews {batch_start}-{batch_end-1}")
            
            # Label the batch
            batch_results = self.label_reviews_batch(batch_texts)
            all_results.extend(batch_results)
            
            # Progress tracking
            processed = len(all_results)
            elapsed_time = time.time() - start_time
            rate = (processed / elapsed_time) * 60 if elapsed_time > 0 else 0
            eta = ((len(review_texts) - processed) / rate) if rate > 0 else 0
            
            logger.info(f"Progress: {processed}/{len(review_texts)} ({processed/len(review_texts)*100:.1f}%) | "
                       f"Rate: {rate:.1f}/min | ETA: {eta:.1f}min")
            
            # Save progress every 25 batches (200 reviews) for premium processing
            if (batch_idx + 1) % 25 == 0:
                # Create progress DataFrame with only processed results
                temp_results = all_results[:processed]
                temp_df = pd.DataFrame({
                    'text': review_texts[:processed],
                    'llm_classification': [r['classification'] for r in temp_results],
                    'llm_confidence': [r['confidence'] for r in temp_results],
                    'llm_reasoning': [r['reasoning'] for r in temp_results],
                    'llm_provider': [r['llm_provider'] for r in temp_results]
                })
                fake_categories = ['spam', 'advertisement', 'irrelevant', 'fake_rant', 'inappropriate']
                temp_df['fake_review'] = temp_df['llm_classification'].apply(lambda x: 1 if x in fake_categories else 0)
                temp_df.to_csv(progress_file, index=False)
                logger.info(f"Progress saved to {progress_file}")
            
            # Small delay between batches to be respectful to APIs
            time.sleep(0.2)  # Reduced delay for faster processing
        
        # Create final DataFrame with processed results
        results_df = pd.DataFrame({
            'text': review_texts,
            'llm_classification': [r['classification'] for r in all_results],
            'llm_confidence': [r['confidence'] for r in all_results],
            'llm_reasoning': [r['reasoning'] for r in all_results],
            'llm_provider': [r['llm_provider'] for r in all_results]
        })
        
        # Add binary fake indicator (1 = suspicious/fake, 0 = genuine)
        fake_categories = ['spam', 'advertisement', 'irrelevant', 'fake_rant', 'inappropriate']
        results_df['fake_review'] = results_df['llm_classification'].apply(lambda x: 1 if x in fake_categories else 0)
        
        total_time = time.time() - start_time
        logger.info(f"Completed labeling {len(results_df)} reviews in {total_time/60:.1f} minutes")
        
        # Show classification distribution
        class_counts = results_df['llm_classification'].value_counts()
        logger.info("Classification Distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / len(results_df)) * 100
            logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Show provider distribution
        provider_counts = results_df['llm_provider'].value_counts()
        logger.info("Provider Usage:")
        for provider_name, count in provider_counts.items():
            percentage = (count / len(results_df)) * 100
            logger.info(f"  {provider_name}: {count} ({percentage:.1f}%)")
        
        # Save final results
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f'google_reviews_labeled_{timestamp}.csv'
        
        results_df.to_csv(output_file, index=False)
        logger.info(f"Final results saved to {output_file}")
        
        return results_df

def main():
    """Main execution function"""
    
    print("🤖 MULTI-LLM GOOGLE REVIEWS LABELING SYSTEM")
    print("=" * 60)
    
    try:
        # Initialize labeler
        labeler = GoogleReviewsLabeler()
        
        # Configuration
        json_file = "dataset (1).json"
        
        # Get user input for max reviews
        max_input = input("Enter max number of reviews to process (or press Enter for ALL): ").strip()
        max_reviews = int(max_input) if max_input else None
        
        print(f"\\nProcessing up to {max_reviews if max_reviews else 'ALL'} reviews from {json_file}...")
        print(f"Active LLM providers: {[p.name for p in labeler.active_providers]}")
        
        # Process reviews
        labeled_df = labeler.process_google_reviews(json_file, max_reviews=max_reviews)
        
        print(f"\\n✅ Successfully labeled {len(labeled_df)} reviews!")
        print(f"📊 Results saved to CSV file")
        
        return labeled_df
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Main execution failed: {str(e)}")

if __name__ == "__main__":
    main()
