import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import re
import random
import os
import csv
import requests
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

def load_dataset_from_file(file_path: str) -> List[str]:
    """Load dataset from various file formats"""
    texts = []
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        return texts
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Split by paragraphs or sentences
                if '\n\n' in content:  # Paragraph-based splitting
                    texts = [p.strip() for p in content.split('\n\n') if p.strip()]
                else:  # Sentence-based splitting
                    sentences = re.split(r'[.!?]+', content)
                    texts = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        elif file_extension == '.csv':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                csv_reader = csv.reader(f)
                headers = next(csv_reader, None)
                
                # Try to find text column
                text_columns = []
                if headers:
                    for i, header in enumerate(headers):
                        if any(keyword in header.lower() for keyword in ['text', 'content', 'message', 'review', 'comment', 'description']):
                            text_columns.append(i)
                
                # If no obvious text column, use all columns
                if not text_columns:
                    text_columns = list(range(len(headers))) if headers else [0]
                
                for row in csv_reader:
                    if row:  # Skip empty rows
                        for col_idx in text_columns:
                            if col_idx < len(row) and len(row[col_idx].strip()) > 20:
                                texts.append(row[col_idx].strip())
        
        else:
            print(f"Unsupported file format: {file_extension}")
            
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
    
    print(f"Loaded {len(texts)} text samples from {file_path}")
    return texts

def download_sample_dataset(url: str, filename: str) -> bool:
    """Download a sample dataset if local file doesn't exist"""
    try:
        print(f"Downloading dataset from {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Dataset downloaded successfully as {filename}")
        return True
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return False

def create_sample_datasets():
    """Create sample datasets and save them as files"""
    
    # Sample 1: General knowledge text file
    general_text = """
    Artificial intelligence is transforming the way we interact with technology. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions. Deep learning, a subset of machine learning, uses neural networks with multiple layers to model complex relationships in data.

    Natural language processing enables computers to understand and generate human language. This technology powers chatbots, translation services, and text analysis tools. Recent advances in transformer models have significantly improved the quality of language understanding and generation.

    Computer vision allows machines to interpret and analyze visual information from the world. Applications include facial recognition, autonomous vehicles, medical image analysis, and quality control in manufacturing. Convolutional neural networks are particularly effective for image processing tasks.

    The field of robotics combines artificial intelligence with mechanical engineering to create machines that can perform tasks autonomously. Modern robots are used in manufacturing, healthcare, exploration, and service industries. They can adapt to changing environments and learn from experience.

    Data science involves extracting insights from large datasets using statistical methods and machine learning techniques. Data scientists work with structured and unstructured data to solve business problems and drive decision-making. The role requires skills in programming, mathematics, and domain expertise.

    Cybersecurity is becoming increasingly important as digital threats evolve. Organizations must protect their data and systems from hackers, malware, and other cyber attacks. Security professionals use various tools and techniques to monitor, detect, and respond to threats.

    Cloud computing provides on-demand access to computing resources over the internet. This technology enables businesses to scale their operations without investing in physical infrastructure. Popular cloud services include data storage, web hosting, and software applications.

    The Internet of Things connects everyday objects to the internet, enabling them to collect and exchange data. Smart homes, wearable devices, and industrial sensors are examples of IoT applications. This connectivity creates new opportunities for automation and efficiency.

    Blockchain technology provides a secure and transparent way to record transactions. Originally developed for cryptocurrency, blockchain has applications in supply chain management, voting systems, and digital identity verification. The technology uses cryptographic principles to ensure data integrity.

    Quantum computing leverages quantum mechanical phenomena to perform calculations that would be impossible for classical computers. While still in early stages, quantum computers show promise for solving complex problems in cryptography, optimization, and scientific simulation.
    """
    
    # Sample 2: Conversational CSV
    conversations = [
        ["user", "Hello, how are you today?", "greeting"],
        ["assistant", "I'm doing well, thank you for asking! How can I help you?", "greeting"],
        ["user", "Can you explain machine learning?", "question"],
        ["assistant", "Machine learning is a method of data analysis that automates analytical model building. It uses algorithms that iteratively learn from data.", "explanation"],
        ["user", "What are the types of machine learning?", "question"],
        ["assistant", "There are three main types: supervised learning, unsupervised learning, and reinforcement learning. Each has different use cases.", "explanation"],
        ["user", "How does deep learning work?", "question"],
        ["assistant", "Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data.", "explanation"],
        ["user", "What is natural language processing?", "question"],
        ["assistant", "NLP is a branch of AI that helps computers understand, interpret and manipulate human language in a meaningful way.", "explanation"],
        ["user", "Can you give examples of AI applications?", "question"],
        ["assistant", "AI is used in recommendation systems, image recognition, speech recognition, autonomous vehicles, and medical diagnosis.", "explanation"],
        ["user", "What programming languages are used for AI?", "question"],
        ["assistant", "Python is the most popular, followed by R, Java, C++, and JavaScript. Python has excellent libraries like TensorFlow and PyTorch.", "explanation"],
        ["user", "How do I start learning AI?", "question"],
        ["assistant", "Start with mathematics and statistics, learn Python programming, take online courses, and practice with real datasets and projects.", "advice"],
        ["user", "What are neural networks?", "question"],
        ["assistant", "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information.", "explanation"],
        ["user", "What is the difference between AI and ML?", "question"],
        ["assistant", "AI is the broader concept of machines being able to carry out tasks smartly. ML is a subset of AI that focuses on learning from data.", "explanation"],
        ["user", "Thank you for the explanations!", "gratitude"],
        ["assistant", "You're welcome! I'm glad I could help explain these concepts. Feel free to ask if you have more questions.", "gratitude"]
    ]
    
    # Create directories if they don't exist
    os.makedirs('datasets', exist_ok=True)
    
    # Save general text file
    with open('datasets/general_knowledge.txt', 'w', encoding='utf-8') as f:
        f.write(general_text.strip())
    
    # Save conversational CSV
    with open('datasets/conversations.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['speaker', 'text', 'category'])  # Headers
        writer.writerows(conversations)
    
    # Create a larger mixed dataset
    mixed_texts = [
        "The history of computing dates back to ancient civilizations that used mechanical devices for calculations.",
        "Programming languages have evolved from simple assembly languages to high-level languages like Python and JavaScript.",
        "Software engineering involves designing, developing, testing, and maintaining software applications and systems.",
        "Database management systems store and organize data efficiently, allowing for quick retrieval and manipulation.",
        "Web development combines front-end technologies like HTML, CSS, and JavaScript with back-end technologies.",
        "Mobile app development has grown rapidly with the popularity of smartphones and tablets.",
        "Game development requires skills in programming, art, sound design, and project management.",
        "User experience design focuses on creating products that provide meaningful and relevant experiences to users.",
        "Information security protects digital information from unauthorized access, use, disclosure, or destruction.",
        "Network administration involves managing and maintaining computer networks and systems.",
        "Digital marketing uses online channels to promote products and services to potential customers.",
        "E-commerce platforms enable businesses to sell products and services over the internet.",
        "Social media has transformed how people communicate, share information, and connect with others.",
        "Content management systems help organizations create, manage, and publish digital content.",
        "Business intelligence tools analyze data to help organizations make informed decisions.",
        "Project management methodologies like Agile and Scrum help teams deliver projects efficiently.",
        "Quality assurance ensures that software products meet specified requirements and standards.",
        "Technical writing involves creating clear and concise documentation for technical products and processes.",
        "Innovation drives technological advancement and creates new opportunities for businesses.",
        "Digital transformation helps organizations adapt to changing market conditions and customer expectations."
    ]
    
    with open('datasets/mixed_topics.txt', 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(mixed_texts))
    
    print("Sample datasets created in 'datasets/' directory:")
    print("- general_knowledge.txt: Technical knowledge text")
    print("- conversations.csv: Conversational data")
    print("- mixed_topics.txt: Mixed topic paragraphs")

def load_or_create_dataset(file_paths: List[str] = None) -> List[str]:
    """Load dataset from files or create sample datasets"""
    
    if file_paths is None:
        file_paths = [
            'datasets/general_knowledge.txt',
            'datasets/conversations.csv',
            'datasets/mixed_topics.txt'
        ]
    
    all_texts = []
    
    # Check if any of the default files exist
    existing_files = [f for f in file_paths if os.path.exists(f)]
    
    if not existing_files:
        print("No dataset files found. Creating sample datasets...")
        create_sample_datasets()
        existing_files = [f for f in file_paths if os.path.exists(f)]
    
    # Load from existing files
    for file_path in existing_files:
        texts = load_dataset_from_file(file_path)
        all_texts.extend(texts)
        print(f"Loaded {len(texts)} samples from {file_path}")
    
    # If still no data and internet is available, try to download a sample
    if not all_texts:
        print("No local data found. Attempting to download sample dataset...")
        
        # Try to download a simple dataset (this is a fallback)
        sample_urls = [
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        ]
        
        for url in sample_urls:
            filename = f"datasets/downloaded_{url.split('/')[-1]}"
            if download_sample_dataset(url, filename):
                texts = load_dataset_from_file(filename)
                all_texts.extend(texts)
                break
    
    # Final fallback to original sample data
    if not all_texts:
        print("Using built-in sample dataset as fallback...")
        all_texts = create_sample_datasets()
    
    print(f"Total texts loaded: {len(all_texts)}")
    return all_texts

def main():
    """Main training and evaluation pipeline"""
    print("Starting Advanced LLM Training Pipeline")
    print("=" * 50)
    
    # Set device (MPS for M1 Mac, CUDA for GPU, CPU as fallback)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load dataset from files
    print("\nLoading dataset...")
    
    # You can specify custom file paths here
    custom_files = [
        # Add your own file paths here, e.g.:
        # 'path/to/your/dataset.txt',
        # 'path/to/your/data.csv',
    ]
    
    if custom_files and all(os.path.exists(f) for f in custom_files):
        print("Using custom dataset files...")
        texts = load_or_create_dataset(custom_files)
    else:
        print("Using default/sample datasets...")
        texts = load_or_create_dataset()
    
    if not texts:
        print("Error: No text data loaded. Exiting.")
        return
    
    # Show dataset statistics
    total_chars = sum(len(text) for text in texts)
    avg_length = total_chars / len(texts) if texts else 0
    
    print(f"\nDataset Statistics:")
    print(f"- Total samples: {len(texts)}")
    print(f"- Total characters: {total_chars:,}")
    print(f"- Average text length: {avg_length:.1f} characters")
    print(f"- Sample text: '{texts[0][:100]}...'")
    
    # Initialize tokenizer and build vocabulary
    vocab_size = min(3000, len(set(' '.join(texts).split())))  # Adaptive vocab size
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(texts)
    
    # Split data
    split_idx = int(0.8 * len(texts))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    print(f"\nData split: {len(train_texts)} train, {len(val_texts)} validation")
    
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