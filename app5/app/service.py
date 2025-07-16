from llama_cpp import Llama
from .stock_api import get_stock_price
import re

# Load the model once globally (only once per app start)
MODEL_PATH = "/Users/johnmoses/.cache/lm-studio/models/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
llm = Llama(model_path=MODEL_PATH)

# Prompt templates keyed by intent
PROMPT_TEMPLATES = {
    "get_stock_price": (
        "You are a financial assistant. Use the stock data provided and answer concisely."
    ),
    "compare_stocks": (
        "You are a financial analyst. Compare the provided stock data in bullet points."
    ),
    "customer_service": (
        "You are a helpful customer service agent. Respond clearly and politely."
    ),
    "default": (
        "You are a knowledgeable financial assistant ready to help."
    )
}

def detect_intent(user_message):
    text = user_message.lower()
    if any(keyword in text for keyword in ["stock price", "price of", "quote for"]):
        return "get_stock_price"
    if any(keyword in text for keyword in ["compare", "vs", "versus", "difference between"]):
        return "compare_stocks"
    if any(keyword in text for keyword in ["how do i", "help with", "account", "support"]):
        return "customer_service"
    return "default"

def extract_symbols(text):
    """
    Very simple heuristic to extract stock ticker symbols:
    Assume uppercase words (1 to 5 letters) as symbols.
    """
    return re.findall(r'\b[A-Z]{1,5}\b', text)

def generate_response(chat_history):
    """
    Generate a response using your local LLaMA model,
    augment prompts dynamically based on intent and live stock data.
    """
    if not chat_history:
        return "Hello! How can I assist you today?"

    user_message = chat_history[-1]['content']
    intent = detect_intent(user_message)

    base_prompt = PROMPT_TEMPLATES.get(intent, PROMPT_TEMPLATES["default"]) + "\n\n"

    if intent == "get_stock_price":
        symbols = extract_symbols(user_message)
        if not symbols:
            stock_info_text = "No stock symbol detected in the query."
        else:
            stock_infos = []
            for sym in symbols:
                info = get_stock_price(sym)
                if info:
                    stock_infos.append(
                        f"{info['name']} ({info['symbol']}): ${info['price']} "
                        f"({info['change_percent']}%)"
                    )
                else:
                    stock_infos.append(f"Sorry, no data found for symbol: {sym}")
            stock_info_text = "Stock prices:\n" + "\n".join(stock_infos)

        prompt = base_prompt + stock_info_text + f"\n\nUser: {user_message}\nAssistant:"

    elif intent == "compare_stocks":
        symbols = extract_symbols(user_message)
        if len(symbols) < 2:
            comp_text = "I need at least two stock symbols to perform a comparison."
        else:
            comp_text = "Comparison \n"
            for sym in symbols:
                info = get_stock_price(sym)
                if info:
                    comp_text += (
                        f"{info['name']} ({info['symbol']}): Price ${info['price']}, "
                        f"Change {info['change_percent']}%\n"
                    )
                else:
                    comp_text += f"Data not found for symbol: {sym}\n"
        prompt = base_prompt + comp_text + f"\nUser: {user_message}\nAssistant:"

    else:
        # For customer service or general/default intents: include full chat history in prompt
        prompt = base_prompt
        for message in chat_history:
            role = "User" if message['role'] == 'user' else "Assistant"
            prompt += f"{role}: {message['content']}\n"
        prompt += "Assistant:"

    # Generate response using LLaMA model
    output = llm(
        prompt,
        max_tokens=200,
        temperature=0.7,
        stop=["User:", "Assistant:"]
    )

    response = output.get('choices', [{}])[0].get('text', '').strip()
    return response
