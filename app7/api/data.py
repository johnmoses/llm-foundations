import os
import csv
from datasets import Dataset
import pandas as pd

# Dataset file paths
BASIC_FINETUNE_PATH = os.path.join("data", "basic_finetune.csv")
PEFT_FINETUNE_PATH = os.path.join("data", "peft_finetune.csv")
RLHF_PREFERENCES_PATH = os.path.join("data", "rlhf_preferences.csv")


def generate_finance_datasets():
    os.makedirs("data", exist_ok=True)

    basic_data = [
        (
            "What is a stock?",
            "A stock represents ownership in a company and a claim on part of its assets and earnings.",
        ),
        (
            "Explain compound interest.",
            "Compound interest is interest calculated on the initial principal and also on accumulated interest.",
        ),
        (
            "What is a bond?",
            "A bond is a fixed income instrument representing a loan made by an investor to a borrower.",
        ),
        (
            "How does inflation affect investments?",
            "Inflation reduces the purchasing power of money, impacting investment returns.",
        ),
        (
            "What is diversification?",
            "Diversification is an investment strategy to reduce risk by allocating investments across various assets.",
        ),
    ]
    with open(BASIC_FINETUNE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["instruction", "response"])
        writer.writerows(basic_data)

    peft_data = [
        ["Stocks are traded on exchanges such as NYSE and NASDAQ."],
        ["The Federal Reserve controls monetary policy in the US."],
        ["Diversification helps reduce unsystematic risk in portfolios."],
        ["ETFs are investment funds traded on stock exchanges."],
        ["Credit ratings assess the creditworthiness of borrowers."],
    ]
    with open(PEFT_FINETUNE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text"])
        writer.writerows(peft_data)

    preference_data = [
        (
            "What is a stock?",
            "A stock represents ownership in a company and a claim on its assets and earnings.",
            "A stock is a type of bond.",
        ),
        (
            "Explain compound interest.",
            "Compound interest is interest on principal and accumulated interest.",
            "Compound interest is interest only on the principal.",
        ),
        (
            "What is diversification?",
            "Diversification spreads investments to reduce risk.",
            "Diversification means investing all in one asset.",
        ),
    ]
    with open(RLHF_PREFERENCES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "chosen_response", "rejected_response"])
        writer.writerows(preference_data)

    print(f"Datasets generated in 'data/' folder.")


def load_text_dataset_from_csv(csv_path, text_columns, delimiter="\n"):
    df = pd.read_csv(csv_path)
    df["text"] = df[text_columns].astype(str).agg(delimiter.join, axis=1)
    return Dataset.from_pandas(df[["text"]])


def load_text_dataset_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return Dataset.from_list([{"text": line} for line in lines])
