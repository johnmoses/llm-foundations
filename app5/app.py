import sqlite3
import time
import pandas as pd
import requests
import ta
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random  # For simulated data

# --- SQLite setup ---


def init_db(db_path="finance_chatbot.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_input TEXT,
            bot_response TEXT
        )
    """
    )
    conn.commit()
    return conn


def save_conversation(conn, user_input, bot_response):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (timestamp, user_input, bot_response) VALUES (?, ?, ?)",
        (time.strftime("%Y-%m-%d %H:%M:%S"), user_input, bot_response),
    )
    conn.commit()


# --- Technical indicators ---


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()
    df["sma_20"] = ta.trend.SMAIndicator(close=df["close"], window=20).sma_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(close=df["close"], window=50).sma_indicator()
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    return df.dropna()


def handle_technical_indicator_command():
    data = {"close": [100 + i * 0.5 for i in range(60)]}  # Mock price data
    df = pd.DataFrame(data)
    indicators_df = calculate_technical_indicators(df)
    latest = indicators_df.iloc[-1]
    return (
        f"RSI: {latest['rsi']:.2f}, SMA20: {latest['sma_20']:.2f}, "
        f"SMA50: {latest['sma_50']:.2f}, BB Upper: {latest['bb_upper']:.2f}, "
        f"BB Lower: {latest['bb_lower']:.2f}"
    )


# --- Crypto price fetching ---


def get_crypto_price(symbol: str) -> float:
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return float(data["price"])
    except requests.RequestException:
        return None


def handle_crypto_price_command(symbol: str):
    price = get_crypto_price(symbol)
    if price:
        return f"The current price of {symbol.upper()} is {price:.2f} USD."
    else:
        return "Sorry, couldn't fetch the crypto price at the moment. Please check the symbol."


# --- NEW: Account Management & Balance Inquiry (Simulated) ---
def handle_account_balance_command(account_type: str = "checking"):
    """Simulates checking account balance."""
    if account_type.lower() == "checking":
        balance = f"{random.uniform(1500.00, 5000.00):.2f}"
        return f"Your checking account balance is: ${balance}."
    elif account_type.lower() == "savings":
        balance = f"{random.uniform(5000.00, 20000.00):.2f}"
        return f"Your savings account balance is: ${balance}."
    else:
        return "I can only check checking or savings account balances at the moment."


# --- NEW: Spending Analysis & Budgeting (Simulated) ---
def handle_spending_analysis_command():
    """Simulates providing spending insights."""
    categories = {
        "Groceries": random.uniform(300, 600),
        "Utilities": random.uniform(100, 250),
        "Dining Out": random.uniform(150, 400),
        "Transportation": random.uniform(50, 200),
    }
    insights = "Here's a summary of your spending for the last month:\n"
    for category, amount in categories.items():
        insights += f"- {category}: ${amount:.2f}\n"
    insights += (
        "Consider setting a budget of $500 for dining out next month to save more!"
    )
    return insights


# --- NEW: Financial Product Information ---
def handle_product_info_command(product_type: str):
    """Provides information on various financial products."""
    product_info = {
        "savings account": "A savings account is an interest-bearing deposit account held at a bank or other financial institution. It's a secure place to store money and earn a modest return.",
        "checking account": "A checking account is a deposit account that allows for frequent withdrawals and deposits, commonly used for everyday transactions like bill payments and purchases.",
        "investment options": "Our investment options include mutual funds, exchange-traded funds (ETFs), stocks, and bonds. We can help you choose based on your risk tolerance and financial goals.",
        "loan": "We offer various types of loans including personal loans, auto loans, and mortgages. Each comes with different terms, interest rates, and eligibility criteria. What kind of loan are you interested in?",
        "credit card": "Our credit cards offer various benefits like rewards points, cashback, and low interest rates. Eligibility depends on your credit history. Do you want to know more about a specific card?",
        "mortgage": "A mortgage is a loan used to buy a house or other real estate. It's secured by the property itself. We offer fixed-rate and adjustable-rate mortgages with competitive rates.",
    }
    info = product_info.get(
        product_type.lower(),
        "I don't have information on that specific financial product. I can tell you about savings accounts, checking accounts, investment options, loans, credit cards, or mortgages.",
    )
    return info


# --- NEW: Loan/Credit Application Assistance (Simulated) ---
def handle_loan_application_command(loan_type: str = "personal"):
    """Simulates guiding through a loan application process."""
    if loan_type.lower() == "personal":
        return "To apply for a personal loan, we typically need your income details, credit score, and desired loan amount. Would you like to proceed with pre-qualification?"
    elif loan_type.lower() == "mortgage":
        return "For a mortgage application, we require detailed financial statements, employment history, and property information. Our system can guide you through the documents needed."
    else:
        return "I can assist with personal loan or mortgage applications. Which one are you interested in?"


# --- LLM setup and inference ---

MODEL_NAME = "gpt2"  # Change to your preferred Hugging Face model (e.g., "distilgpt2", "microsoft/DialoGPT-small")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Fix padding token issue
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def build_prompt_from_history(history, max_tokens=1024):
    prompt = "You are a helpful financial expert chatbot. You provide accurate and concise information. If a query is about specific data (like balance or price), use the tools. For general questions, provide a helpful answer.\n"
    current_prompt_tokens = tokenizer.encode(prompt)

    # Build history from most recent, keeping within max_tokens
    context_turns = []
    for user_text, bot_text in reversed(
        history[:-1]
    ):  # Exclude the current user turn for initial prompt build
        turn = f"User: {user_text}\nBot: {bot_text}\n"
        tokens = tokenizer.encode(turn)
        if len(current_prompt_tokens) + len(tokens) > max_tokens:
            break
        context_turns.insert(
            0, turn
        )  # Insert at beginning to maintain chronological order
        current_prompt_tokens.extend(tokens)

    full_context = "".join(context_turns) + f"User: {history[-1][0]}\nBot:"
    return prompt + full_context


def llm_respond_with_context(history, max_length=200, max_tokens=1024):
    prompt = build_prompt_from_history(history, max_tokens)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Ensure max_length doesn't exceed model's maximum position embeddings
    # And leave room for the input prompt itself
    actual_max_length = min(max_length + inputs.shape[1], tokenizer.model_max_length)

    outputs = model.generate(
        inputs,
        max_length=actual_max_length,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Decode only the newly generated text
    response = tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)
    return response.strip()


# --- Main chatbot CLI loop ---


def chatbot_cli():
    conn = init_db()
    print("Welcome to your Finance Chatbot CLI!")
    print("Type 'help' for available commands, or 'exit' to quit.")

    conversation_history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        elif user_input.lower() == "help":
            print("\nAvailable commands:")
            print("- `rsi`: Get mock technical indicators.")
            print(
                "- `price <symbol>`: Get current crypto price (e.g., `price BTCUSDT`)."
            )
            print(
                "- `balance [account_type]`: Check simulated account balance (e.g., `balance checking`, `balance savings`)."
            )
            print("- `spending`: Get simulated spending analysis.")
            print(
                "- `product info <type>`: Get info on financial products (e.g., `product info loan`, `product info savings account`)."
            )
            print(
                "- `loan application [type]`: Simulate loan application inquiry (e.g., `loan application personal`)."
            )
            print("- Any other query will be answered by the LLM.")
            print("---")
            continue  # Skip saving to history for help command

        bot_response = ""
        # Command handling
        if user_input.lower() == "rsi":
            bot_response = handle_technical_indicator_command()
        elif user_input.lower().startswith("price "):
            _, symbol = user_input.split(maxsplit=1)
            bot_response = handle_crypto_price_command(symbol)
        elif user_input.lower().startswith("balance"):
            parts = user_input.lower().split(maxsplit=1)
            account_type = parts[1] if len(parts) > 1 else ""
            bot_response = handle_account_balance_command(account_type)
        elif user_input.lower() == "spending":
            bot_response = handle_spending_analysis_command()
        elif user_input.lower().startswith("product info "):
            _, _, product_type = user_input.lower().split(maxsplit=2)
            bot_response = handle_product_info_command(product_type)
        elif user_input.lower().startswith("loan application"):
            parts = user_input.lower().split(maxsplit=2)
            loan_type = parts[2] if len(parts) > 2 else ""
            bot_response = handle_loan_application_command(loan_type)
        else:
            # If no command matched, send to LLM
            # Append current user input with empty bot response to history for prompt
            conversation_history.append((user_input, ""))
            bot_response = llm_respond_with_context(conversation_history)
            # Update last bot response in history
            conversation_history[-1] = (user_input, bot_response)

        print(f"Chatbot: {bot_response}")
        save_conversation(conn, user_input, bot_response)

    conn.close()


if __name__ == "__main__":
    chatbot_cli()

