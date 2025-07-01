import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
)
from peft import LoraConfig, get_peft_model
# from trl import PPOTrainer  # Uncomment if using RLHF PPO training

def load_text_dataset_from_csv(csv_path, text_columns, delimiter="\n"):
    """
    Load and combine text columns from CSV into a single text field.
    """
    df = pd.read_csv(csv_path)
    # Combine specified columns into one text column separated by delimiter
    df['text'] = df[text_columns].astype(str).agg(delimiter.join, axis=1)
    return Dataset.from_pandas(df[['text']])

def load_text_dataset_from_txt(txt_path):
    """
    Load dataset from TXT file where each line is a text example.
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return Dataset.from_list([{'text': line} for line in lines])

def basic_fine_tuning(model_name, dataset_path, is_csv=True, instruction_col='instruction', response_col='response', output_dir='./basic_finetune_results'):
    """
    Basic fine-tuning: full model fine-tuning on instruction-response pairs.
    """
    print("Starting Basic Fine-Tuning...")

    # Load dataset
    if is_csv:
        dataset = load_text_dataset_from_csv(dataset_path, [instruction_col, response_col], delimiter="\nAssistant: ")
        # Prepend "Human: " to instruction and "\nAssistant: " to response for clarity
        dataset = dataset.map(lambda x: {'text': f"Human: {x['text']}"})
    else:
        dataset = load_text_dataset_from_txt(dataset_path)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenization
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="no",
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    print(f"Basic fine-tuning completed. Model saved to {output_dir}")

def peft_fine_tuning(model_name, dataset_path, is_csv=True, text_col='text', output_dir='./peft_finetune_results'):
    """
    PEFT fine-tuning using LoRA adapters.
    """
    print("Starting PEFT Fine-Tuning with LoRA...")

    # Load dataset
    if is_csv:
        dataset = load_text_dataset_from_csv(dataset_path, [text_col])
    else:
        dataset = load_text_dataset_from_txt(dataset_path)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenization
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # LoRA configuration
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    peft_model = get_peft_model(model, peft_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="no",
        save_total_limit=2,
        learning_rate=3e-4,
        fp16=True,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    print(f"PEFT fine-tuning completed. Model saved to {output_dir}")

def rlhf_reward_model_training(model_name, preference_csv_path, output_dir='./reward_model_results'):
    """
    Train a reward model for RLHF from human preference data.
    Expected CSV columns: ['prompt', 'chosen_response', 'rejected_response']
    """
    print("Starting RLHF Reward Model Training...")

    df = pd.read_csv(preference_csv_path)

    # Flatten preference pairs into labeled dataset
    def flatten_preferences(df):
        examples = []
        for _, row in df.iterrows():
            examples.append({"text": row["prompt"] + " " + row["chosen_response"], "label": 0})
            examples.append({"text": row["prompt"] + " " + row["rejected_response"], "label": 1})
        return examples

    reward_data = flatten_preferences(df)
    reward_dataset = Dataset.from_list(reward_data)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    reward_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    tokenized_dataset = reward_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="no",
        save_total_limit=2,
        learning_rate=5e-5,
        fp16=True,
    )

    trainer = Trainer(
        model=reward_model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    print(f"RLHF reward model training completed. Model saved to {output_dir}")

# Optional: Add PPO RLHF fine-tuning function here if you want to integrate TRL PPOTrainer

if __name__ == "__main__":
    # Example usage:
    MODEL_NAME = "meta-llama/Llama-3-8b"

    # Basic fine-tuning
    basic_fine_tuning(
        model_name=MODEL_NAME,
        dataset_path="basic_finetune_data.csv",
        is_csv=True,
        instruction_col="instruction",
        response_col="response",
        output_dir="./basic_finetune_results"
    )

    # PEFT fine-tuning
    peft_fine_tuning(
        model_name=MODEL_NAME,
        dataset_path="peft_finetune_data.csv",
        is_csv=True,
        text_col="text",
        output_dir="./peft_finetune_results"
    )

    # RLHF reward model training
    rlhf_reward_model_training(
        model_name=MODEL_NAME,
        preference_csv_path="preference_data.csv",
        output_dir="./reward_model_results"
    )
