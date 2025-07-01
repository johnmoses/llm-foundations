import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import random
from typing import Optional, Tuple, List
from collections import Counter
import time

class SimpleTokenizer:
    """Simple word-level tokenizer with basic preprocessing"""
    
    def __init__(self, vocab_size: int = 2000):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        # Convert to lowercase and clean
        text = text.lower()
        # Add spaces around punctuation
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        all_words = []
        for text in texts:
            processed = self.preprocess_text(text)
            words = processed.split()
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Create vocabulary with special tokens
        vocab_words = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # Add most common words
        most_common = word_counts.most_common(self.vocab_size - len(vocab_words))
        vocab_words.extend([word for word, _ in most_common])
        
        # Create mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        print(f"Built vocabulary with {len(self.word_to_idx)} words")
        print(f"Most common words: {[word for word, _ in most_common[:10]]}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token ids"""
        processed = self.preprocess_text(text)
        words = processed.split()
        
        tokens = []
        if add_special_tokens:
            tokens.append(self.word_to_idx[self.bos_token])
        
        for word in words:
            if word in self.word_to_idx:
                tokens.append(self.word_to_idx[word])
            else:
                tokens.append(self.word_to_idx[self.unk_token])
        
        if add_special_tokens:
            tokens.append(self.word_to_idx[self.eos_token])
        
        return tokens
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids to text"""
        words = []
        special_tokens = {self.word_to_idx[self.pad_token], self.word_to_idx[self.bos_token], 
                         self.word_to_idx[self.eos_token]} if skip_special_tokens else set()
        
        for token in tokens:
            if token in special_tokens:
                continue
            if token in self.idx_to_word:
                word = self.idx_to_word[token]
                if word != self.unk_token:
                    words.append(word)
        
        return ' '.join(words)

class TextDataset:
    """Simple dataset for text sequences"""
    
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, seq_length: int = 64):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.sequences = []
        
        # Create sequences from texts
        for text in texts:
            tokens = tokenizer.encode(text)
            # Create overlapping sequences
            for i in range(0, len(tokens) - seq_length, seq_length // 2):
                sequence = tokens[i:i + seq_length]
                if len(sequence) == seq_length:
                    self.sequences.append(sequence)
        
        print(f"Created {len(self.sequences)} sequences of length {seq_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)
    
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True):
        """Create a simple dataloader"""
        indices = list(range(len(self)))
        if shuffle:
            random.shuffle(indices)
        
        batches = []
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = torch.stack([self[idx] for idx in batch_indices])
            batches.append(batch)
        
        return batches

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Generate Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Single transformer decoder block"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class BasicLLM(nn.Module):
    """Basic Large Language Model using Transformer architecture"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
        max_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between embedding and output projection
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask to prevent attention to future tokens"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        # Token embeddings + positional encoding
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.norm(x)
        
        # Output projection to vocabulary
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Shift targets for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Generate text using the model"""
        
        self.eval()
        
        for _ in range(max_new_tokens):
            # Truncate sequence if it exceeds max length
            if input_ids.size(1) >= self.max_len:
                input_ids = input_ids[:, -self.max_len+1:]
            
            # Forward pass
            logits, _ = self.forward(input_ids)
            
            # Get logits for the last token
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids

def calculate_perplexity(model, data_loader, device):
    """Calculate perplexity on a dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            logits, loss = model(batch, batch)
            
            if loss is not None:
                total_loss += loss.item() * batch.numel()
                total_tokens += batch.numel()
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 10))  # Cap to prevent overflow
    return perplexity

def train_model(model, train_loader, val_loader, device, epochs=5, lr=1e-3):
    """Train the model"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.to(device)
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            
            # Forward pass
            logits, loss = model(batch, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        # Calculate metrics
        avg_loss = epoch_loss / max(len(train_loader), 1)
        
        # Calculate perplexities with error handling
        try:
            train_perplexity = calculate_perplexity(model, train_loader[:5], device)  # Sample for speed
        except:
            train_perplexity = float('inf')
            
        try:
            val_perplexity = calculate_perplexity(model, val_loader[:3], device)
        except:
            val_perplexity = float('inf')
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Train Perplexity: {train_perplexity:.2f}")
        print(f"Validation Perplexity: {val_perplexity:.2f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 50)

def create_sample_dataset():
    """Create a minimal sample dataset for testing"""
    texts = [
        "The quick brown fox jumps over the lazy dog. This is a classic sentence used for testing.",
        "Machine learning is a powerful tool for solving complex problems. Neural networks can learn patterns from data.",
        "Natural language processing enables computers to understand human language. Transformers are state-of-the-art models.",
        "Deep learning has revolutionized artificial intelligence. Models can now generate human-like text.",
        "Python is a popular programming language for data science. It has many useful libraries and frameworks.",
        "The weather today is sunny and warm. Perfect for a walk in the park with friends and family.",
        "Books are a great source of knowledge and entertainment. Reading helps expand vocabulary and understanding.",
        "Technology continues to advance at a rapid pace. New innovations emerge every day in various fields.",
        "Education is important for personal and professional development. Learning never stops throughout life.",
        "Music brings joy and emotion to people around the world. Different genres appeal to different tastes.",
        "Cooking is both an art and a science. Good food requires quality ingredients and proper techniques.",
        "Travel broadens the mind and creates lasting memories. Exploring new places teaches us about different cultures.",
        "Exercise is essential for maintaining good health. Regular physical activity strengthens the body and mind.",
        "Friendship is one of life's greatest treasures. Good friends provide support and companionship through all seasons.",
        "Science helps us understand the world around us. Research and experimentation lead to new discoveries.",
        "Art expresses human creativity and emotion. Different forms of artistic expression have existed throughout history.",
        "Work provides purpose and financial stability. Finding a career that aligns with interests brings satisfaction.",
        "Family relationships form the foundation of society. Strong family bonds create security and belonging.",
        "Communication is key to successful relationships. Clear expression of thoughts and feelings prevents misunderstandings.",
        "Time is a precious resource that cannot be recovered. Using time wisely leads to a more fulfilling life."
    ]
    
    return texts

def main():
    """Main training and evaluation pipeline"""
    print("Starting Basic LLM Training Pipeline")
    print("=" * 50)
    
    # Set device (MPS for M1 Mac, CUDA for GPU, CPU as fallback)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Create sample dataset
    texts = create_sample_dataset()
    
    # Initialize tokenizer and build vocabulary
    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokenizer.build_vocab(texts)
    
    # Split data
    split_idx = int(0.8 * len(texts))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    # Create datasets
    seq_length = 32  # Shorter sequences for faster training
    train_dataset = TextDataset(train_texts, tokenizer, seq_length)
    val_dataset = TextDataset(val_texts, tokenizer, seq_length)
    
    # Check if datasets have enough data
    if len(train_dataset) == 0:
        print("Error: Training dataset is empty. Creating longer sequences...")
        seq_length = 16
        train_dataset = TextDataset(train_texts, tokenizer, seq_length)
        val_dataset = TextDataset(val_texts, tokenizer, seq_length)
    
    if len(train_dataset) == 0:
        print("Error: Still no training sequences. Using single sentences...")
        # Create sequences from individual sentences
        train_seqs = []
        for text in train_texts:
            tokens = tokenizer.encode(text)
            if len(tokens) >= 8:  # Minimum sequence length
                train_seqs.append(tokens[:seq_length])
        
        val_seqs = []
        for text in val_texts:
            tokens = tokenizer.encode(text)
            if len(tokens) >= 8:
                val_seqs.append(tokens[:seq_length])
        
        # Manually create dataset sequences
        train_dataset.sequences = train_seqs
        val_dataset.sequences = val_seqs
    
    print(f"Final dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("Error: No training data available. Exiting.")
        return
    
    # Create data loaders
    batch_size = 16  # Smaller batch size for M1 Mac
    train_loader = train_dataset.get_dataloader(batch_size, shuffle=True)
    val_loader = val_dataset.get_dataloader(batch_size, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model (smaller for M1 Mac)
    model = BasicLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        max_len=seq_length,
        dropout=0.1
    )
    
    # Train model
    train_model(model, train_loader, val_loader, device, epochs=3, lr=1e-3)
    
    # Test generation
    print("\n" + "=" * 50)
    print("TESTING TEXT GENERATION")
    print("=" * 50)
    
    model.eval()
    test_prompts = [
        "The quick brown",
        "Machine learning is",
        "Python is a",
        "The weather today"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Encode prompt
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)]).to(device)
        
        # Generate with different settings
        for temp, name in [(0.7, "Conservative"), (1.0, "Balanced"), (1.3, "Creative")]:
            generated = model.generate(
                input_ids, 
                max_new_tokens=15, 
                temperature=temp, 
                top_k=20, 
                top_p=0.9
            )
            
            generated_text = tokenizer.decode(generated[0].cpu().tolist())
            print(f"{name} (temp={temp}): {generated_text}")
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)
    
    try:
        final_train_perplexity = calculate_perplexity(model, train_loader[:10], device)
    except:
        final_train_perplexity = float('inf')
        
    try:
        final_val_perplexity = calculate_perplexity(model, val_loader, device)
    except:
        final_val_perplexity = float('inf')
    
    print(f"Final Training Perplexity: {final_train_perplexity:.2f}")
    print(f"Final Validation Perplexity: {final_val_perplexity:.2f}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()