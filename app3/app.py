#!/usr/bin/env python3
"""
Basic Large Language Model Implementation for Financial Applications
Optimized for MacBook Pro M1 (Apple Silicon)

This implementation includes:
- Transformer-based architecture
- Financial text preprocessing
- Model training and inference
- M1 optimization using MPS backend
- Financial sentiment analysis
- Risk assessment capabilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import re
import json
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for M1 optimization
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using Apple M1 GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("Using CUDA GPU")
else:
    device = torch.device("cpu")
    logger.info("Using CPU")

@dataclass
class ModelConfig:
    """Configuration for the financial LLM"""
    vocab_size: int = 50000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 1024
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10

class FinancialTokenizer:
    """Custom tokenizer for financial text"""
    
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<MASK>': 4
        }
        self.financial_patterns = {
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?': '<MONEY>',
            r'\d+(?:\.\d+)?%': '<PERCENT>',
            r'\b\d{4}-\d{2}-\d{2}\b': '<DATE>',
            r'\b[A-Z]{2,5}\b': '<TICKER>',
            r'\b\d+(?:,\d{3})*(?:\.\d+)?\b': '<NUMBER>'
        }
        
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """Build vocabulary from financial texts"""
        word_freq = {}
        
        for text in texts:
            # Apply financial pattern replacements
            processed_text = self._preprocess_financial_text(text)
            tokens = self._tokenize(processed_text)
            
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Add special tokens
        self.vocab.update(self.special_tokens)
        
        # Add frequent words
        vocab_id = len(self.special_tokens)
        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.vocab:
                self.vocab[word] = vocab_id
                vocab_id += 1
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        logger.info(f"Built vocabulary with {len(self.vocab)} tokens")
    
    def _preprocess_financial_text(self, text: str) -> str:
        """Preprocess financial text with pattern recognition"""
        for pattern, replacement in self.financial_patterns.items():
            text = re.sub(pattern, replacement, text)
        return text.lower()
    
    def _tokenize(self, text: str) -> List[str]:
        """Basic tokenization"""
        return re.findall(r'\b\w+\b|[^\w\s]', text)
    
    def encode(self, text: str, max_length: int = None) -> List[int]:
        """Encode text to token IDs"""
        processed_text = self._preprocess_financial_text(text)
        tokens = self._tokenize(processed_text)
        
        if max_length:
            tokens = tokens[:max_length-2]  # Reserve space for BOS/EOS
        
        # Add special tokens
        token_ids = [self.special_tokens['<BOS>']]
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.special_tokens['<UNK>']))
        token_ids.append(self.special_tokens['<EOS>'])
        
        # Pad if necessary
        if max_length and len(token_ids) < max_length:
            token_ids.extend([self.special_tokens['<PAD>']] * (max_length - len(token_ids)))
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in [self.special_tokens['<PAD>'], self.special_tokens['<BOS>'], self.special_tokens['<EOS>']]:
                continue
            tokens.append(self.reverse_vocab.get(token_id, '<UNK>'))
        return ' '.join(tokens)

class FinancialDataset(Dataset):
    """Dataset for financial text data"""
    
    def __init__(self, texts: List[str], tokenizer: FinancialTokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        token_ids = self.tokenizer.encode(text, self.max_length)
        
        # For language modeling, input and target are the same with offset
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        
        return input_ids, target_ids

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(context)

class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward layers"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class FinancialLLM(nn.Module):
    """Financial Large Language Model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            # Calculate loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        
        return logits

class FinancialLLMTrainer:
    """Trainer for the financial LLM"""
    
    def __init__(self, model: FinancialLLM, tokenizer: FinancialTokenizer, config: ModelConfig):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.num_epochs)
        
    def train(self, train_dataset: FinancialDataset, val_dataset: FinancialDataset = None):
        """Train the model"""
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        self.model.train()
        total_loss = 0
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            start_time = time.time()
            
            for batch_idx, (input_ids, targets) in enumerate(train_loader):
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                
                self.optimizer.zero_grad()
                
                logits, loss = self.model(input_ids, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(f'Epoch {epoch+1}/{self.config.num_epochs}, '
                              f'Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            self.scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            epoch_time = time.time() - start_time
            
            logger.info(f'Epoch {epoch+1} completed in {epoch_time:.2f}s, '
                       f'Average Loss: {avg_loss:.4f}')
            
            # Validation
            if val_dataset:
                val_loss = self.validate(val_dataset)
                logger.info(f'Validation Loss: {val_loss:.4f}')
    
    def validate(self, val_dataset: FinancialDataset) -> float:
        """Validate the model"""
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for input_ids, targets in val_loader:
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                
                logits, loss = self.model(input_ids, targets)
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(val_loader)
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.8) -> str:
        """Generate text from a prompt"""
        self.model.eval()
        
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        generated = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                if len(generated) >= self.config.max_seq_len:
                    break
                
                # Get model predictions
                logits = self.model(input_tensor)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Stop if EOS token
                if next_token == self.tokenizer.special_tokens['<EOS>']:
                    break
                
                generated.append(next_token)
                input_tensor = torch.tensor([generated], dtype=torch.long).to(device)
        
        return self.tokenizer.decode(generated)

class FinancialAnalyzer:
    """Financial analysis utilities"""
    
    def __init__(self, model: FinancialLLM, tokenizer: FinancialTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze financial sentiment"""
        # This is a simplified version - in practice, you'd fine-tune on financial sentiment data
        prompt = f"Analyze the financial sentiment of this text: {text}\nSentiment:"
        generated = self.generate_response(prompt)
        
        # Simple keyword-based sentiment (replace with trained model)
        positive_words = ['bullish', 'growth', 'profit', 'gains', 'positive', 'strong']
        negative_words = ['bearish', 'loss', 'decline', 'negative', 'weak', 'drop']
        
        pos_count = sum(1 for word in positive_words if word in generated.lower())
        neg_count = sum(1 for word in negative_words if word in generated.lower())
        
        total = pos_count + neg_count
        if total == 0:
            return {'positive': 0.5, 'negative': 0.5, 'neutral': 0.0}
        
        return {
            'positive': pos_count / total,
            'negative': neg_count / total,
            'neutral': 1 - (pos_count + neg_count) / total
        }
    
    def assess_risk(self, financial_text: str) -> Dict[str, any]:
        """Assess financial risk from text"""
        risk_keywords = {
            'high': ['volatile', 'risky', 'uncertain', 'speculation', 'bubble'],
            'medium': ['moderate', 'cautious', 'watchful', 'mixed'],
            'low': ['stable', 'secure', 'conservative', 'safe', 'steady']
        }
        
        text_lower = financial_text.lower()
        risk_scores = {}
        
        for level, keywords in risk_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            risk_scores[level] = score
        
        total_score = sum(risk_scores.values())
        if total_score == 0:
            return {'risk_level': 'unknown', 'confidence': 0.0}
        
        max_level = max(risk_scores, key=risk_scores.get)
        confidence = risk_scores[max_level] / total_score
        
        return {
            'risk_level': max_level,
            'confidence': confidence,
            'scores': risk_scores
        }
    
    def generate_response(self, prompt: str, max_length: int = 50) -> str:
        """Generate response to financial query"""
        trainer = FinancialLLMTrainer(self.model, self.tokenizer, ModelConfig())
        return trainer.generate(prompt, max_length)

def create_sample_financial_data() -> List[str]:
    """Create sample financial text data for demonstration"""
    return [
        "Apple Inc. reported strong quarterly earnings with revenue of $97.3 billion, beating analyst expectations.",
        "The Federal Reserve announced a 0.25% interest rate hike to combat inflation concerns.",
        "Tesla stock surged 15% after announcing record vehicle deliveries for Q3 2024.",
        "Market volatility increased as investors react to geopolitical tensions in Eastern Europe.",
        "Banking sector shows resilience with JPMorgan Chase reporting 12% profit growth year-over-year.",
        "Cryptocurrency market experiences significant correction with Bitcoin dropping below $40,000.",
        "Technology stocks lead market recovery as investors rotate back into growth names.",
        "Oil prices climb to $85 per barrel amid supply chain disruptions in the Middle East.",
        "Housing market shows signs of cooling as mortgage rates reach 7.5% nationally.",
        "Small-cap stocks outperform large-cap amid rotation to value-oriented investments."
    ]

def main():
    """Main execution function"""
    logger.info("Initializing Financial LLM for Apple M1...")
    
    # Configuration
    config = ModelConfig(
        vocab_size=10000,  # Smaller for demo
        d_model=256,       # Smaller for M1 efficiency
        n_heads=8,
        n_layers=4,        # Fewer layers for faster training
        max_seq_len=256,   # Shorter sequences
        batch_size=16,     # Smaller batch size for M1
        num_epochs=3       # Fewer epochs for demo
    )
    
    # Create sample data
    sample_texts = create_sample_financial_data()
    
    # Initialize tokenizer and build vocabulary
    tokenizer = FinancialTokenizer()
    tokenizer.build_vocab(sample_texts)
    
    # Update config with actual vocab size
    config.vocab_size = len(tokenizer.vocab)
    
    # Create dataset
    dataset = FinancialDataset(sample_texts, tokenizer, config.max_seq_len)
    
    # Initialize model
    model = FinancialLLM(config)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize trainer
    trainer = FinancialLLMTrainer(model, tokenizer, config)
    
    # Train model
    logger.info("Starting training...")
    trainer.train(dataset)
    
    # Test generation
    logger.info("Testing text generation...")
    test_prompts = [
        "The stock market today",
        "Federal Reserve policy",
        "Apple earnings report"
    ]
    
    for prompt in test_prompts:
        generated_text = trainer.generate(prompt, max_length=30)
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated: {generated_text}")
        logger.info("-" * 50)
    
    # Initialize financial analyzer
    analyzer = FinancialAnalyzer(model, tokenizer)
    
    # Test sentiment analysis
    test_text = "Apple stock surged 15% after beating earnings expectations with strong iPhone sales."
    sentiment = analyzer.analyze_sentiment(test_text)
    logger.info(f"Sentiment Analysis: {sentiment}")
    
    # Test risk assessment
    risk_assessment = analyzer.assess_risk(test_text)
    logger.info(f"Risk Assessment: {risk_assessment}")
    
    logger.info("Financial LLM demonstration completed!")

if __name__ == "__main__":
    main()