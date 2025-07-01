# Large Language Models (LLM) basics

We are here exploring the basics of the Large Language Model (LLM) going from very simple architechecture to real-world applications especially in the area of finance.

## Core Architecture

Multi-Head Attention: Implements the self-attention mechanism that allows the model to focus on different parts of the input sequence
Feed-Forward Networks: Position-wise fully connected layers with GELU activation
Transformer Blocks: Combines attention and feed-forward layers with residual connections and layer normalization
Positional Encoding: Adds positional information using sinusoidal encodings

## Key Features

Causal Masking: Prevents the model from attending to future tokens during training
Weight Tying: Shares weights between input embeddings and output projection
Text Generation: Includes methods for autoregressive text generation with temperature, top-k, and top-p sampling
Gradient Clipping: Prevents exploding gradients during training

Enhanced Tokenization:

Word-level tokenizer with vocabulary building
Text preprocessing (lowercase, punctuation handling)
Special tokens (PAD, UNK, BOS, EOS)
Frequency-based vocabulary selection

Dataset & Data Processing:

TextDataset class for sequence creation
Overlapping sequences for better data utilization
Simple dataloader implementation
Train/validation split

Training Infrastructure:

Complete training loop with progress tracking
AdamW optimizer with cosine annealing scheduler
Gradient clipping for stability
Loss tracking and epoch timing

Evaluation Metrics:

Perplexity calculation for model evaluation
Training and validation metrics
Parameter counting

Text Generation & Inference:

Multiple generation strategies (temperature, top-k, top-p)
Different sampling configurations for comparison
Prompt-based generation testing

## Model Configuration

Configurable number of layers, attention heads, and hidden dimensions
Dropout for regularization
Proper weight initialization

How to Use

```python
python basic1.py
```

## App 2

Multi-Format File Support

.txt files: Automatically splits by paragraphs or sentences
.csv files: Intelligently detects text columns (looks for 'text', 'content', 'message', 'review', etc.)
Error handling: Graceful handling of encoding issues and malformed files

Automatic Dataset Creation
The system creates three sample datasets in a datasets/ folder:

general_knowledge.txt: Technical AI/ML content (paragraphs)
conversations.csv: Question-answer pairs with categories
mixed_topics.txt: Diverse topics for better generalization

Advanced Data Loading

Smart text extraction: Handles different CSV structures automatically
Content filtering: Minimum length requirements to ensure quality
Multiple file support: Can load from multiple files simultaneously
Download capability: Can fetch datasets from URLs as fallback

Enhanced Dataset Statistics

Shows total samples, characters, and average lengths
Adaptive vocabulary sizing based on content
Better data split reporting

How to use

Option 1: Use Built-in Datasets (Default)
Just run the script - it will create sample datasets automatically.
Option 2: Use Your Own Files

Create a datasets/ folder
Add your .txt or .csv files
The system will automatically detect and use them

Option 3: Custom File Paths
Modify the custom_files list in main():

```python
custom_files = [
    'path/to/your/dataset.txt',
    'path/to/your/data.csv',
]
```

## App 3

In this section we will write a simple real-world financial application featuring financial sentiment analysis and risk assessment
