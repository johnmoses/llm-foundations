from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
)
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_metric
import pandas as pd
from data import load_text_dataset_from_csv

def basic_fine_tuning(model_name, dataset_path, output_dir='./basic_finetune_results'):
    dataset = load_text_dataset_from_csv(dataset_path, ['instruction', 'response'], delimiter="\nAssistant: ")
    dataset = dataset.map(lambda x: {'text': f"Human: {x['text']}"})

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

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
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    print(f"Basic fine-tuning completed. Model saved to {output_dir}")

def peft_fine_tuning(model_name, dataset_path, output_dir='./peft_finetune_results'):
    dataset = load_text_dataset_from_csv(dataset_path, ['text'])

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    peft_model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="no",
        save_total_limit=2,
        learning_rate=3e-4,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    print(f"PEFT fine-tuning completed. Model saved to {output_dir}")

def rlhf_reward_model_training(model_name, preference_csv_path, output_dir='./reward_model_results'):
    df = pd.read_csv(preference_csv_path)

    def flatten_preferences(df):
        examples = []
        for _, row in df.iterrows():
            examples.append({"text": row["prompt"] + " " + row["chosen_response"], "label": 0})
            examples.append({"text": row["prompt"] + " " + row["rejected_response"], "label": 1})
        return examples

    reward_data = flatten_preferences(df)
    from datasets import Dataset
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
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=reward_model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    print(f"RLHF reward model training completed. Model saved to {output_dir}")

def compute_rouge(predictions, references):
    rouge = load_metric("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    print("ROUGE scores:", results)
    return results
