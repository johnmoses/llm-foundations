import sqlite3
import requests
import pandas as pd
import spacy
from pyti import relative_strength_index, moving_average_convergence_divergence
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForCausalLM, pipeline
)
import streamlit as st

# Constants and API keys
DB_PATH = "financial_data.db"
ALPHA_VANTAGE_API_KEY = "M9R6KH0JUPRIKWE8"
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# Initialize spaCy model for NER (small English model)
nlp = spacy.load("en_core_web_sm")

class FinancialChatbot:
    def __init__(self):
        # FinBERT for sentiment analysis
        self.finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        self.finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=self.finbert_model,
            tokenizer=self.finbert_tokenizer
        )

        # GPT-style causal LM for text generation
        self.gen_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.gen_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        self.text_generator = pipeline(
            "text-generation",
            model=self.gen_model,
            tokenizer=self.gen_tokenizer
        )

        # Setup SQLite DB
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            UNIQUE(ticker, date)
        );
        """
        self.conn.execute(create_table_sql)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker_date ON stock_prices(ticker, date);")
        self.conn.commit()

    def insert_historical_data(self, ticker):
        """Fetch daily historical data from Alpha Vantage and store in DB"""
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "outputsize": "compact"
        }
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        data = response.json()
        time_series = data.get("Time Series (Daily)", {})
        if not time_series:
            return False  # No data found or API limit reached

        with self.conn:
            for date_str, daily_data in time_series.items():
                self.conn.execute("""
                    INSERT OR IGNORE INTO stock_prices (ticker, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    date_str,
                    float(daily_data["1. open"]),
                    float(daily_data["2. high"]),
                    float(daily_data["3. low"]),
                    float(daily_data["4. close"]),
                    int(daily_data["6. volume"])
                ))
        return True

    def fetch_historical_data(self, ticker, days=60):
        query = """
        SELECT date, open, high, low, close, volume FROM stock_prices
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT ?
        """
        df = pd.read_sql_query(query, self.conn, params=(ticker, days))
        df = df.sort_values('date')
        return df

    def calculate_indicators(self, df):
        closes = df['close'].tolist()
        rsi = relative_strength_index(closes, period=14)[-1] if len(closes) >= 14 else None
        macd = moving_average_convergence_divergence(closes)[-1] if len(closes) >= 26 else None
        return {"RSI": rsi, "MACD": macd}

    def fetch_realtime_data(self, ticker):
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        data = response.json()
        quote = data.get("Global Quote", {})
        if not quote:
            return None
        return {
            "price": float(quote.get("05. price", 0)),
            "volume": int(quote.get("06. volume", 0))
        }

    def get_blockchain_data(self, assets=["BTC", "ETH", "USDT"]):
        """Fetch multiple crypto prices from Alpha Vantage"""
        prices = {}
        for asset in assets:
            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": asset,
                "to_currency": "USD",
                "apikey": ALPHA_VANTAGE_API_KEY
            }
            response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
            data = response.json()
            rate_info = data.get("Realtime Currency Exchange Rate", {})
            price = rate_info.get("5. Exchange Rate", None)
            prices[asset] = float(price) if price else None
        return prices

    def get_defi_data(self):
        """Example: Fetch DeFi TVL from public API (DefiLlama)"""
        url = "https://api.llama.fi/tvl"
        try:
            response = requests.get(url)
            data = response.json()
            # Summarize total TVL or pick some popular protocols
            total_tvl = sum([protocol.get("tvl", 0) for protocol in data])
            return total_tvl
        except Exception:
            return None

    def analyze_sentiment(self, text):
        result = self.sentiment_analyzer(text)[0]
        return result['label'], result['score']

    def generate_response(self, user_query, context):
        sentiment_label, sentiment_score = self.analyze_sentiment(user_query)
        prompt = (
            f"Financial Context:\n{context}\n\n"
            f"User Query Sentiment: {sentiment_label} (score: {sentiment_score:.2f})\n"
            f"User Query: {user_query}\n"
            f"Assistant:"
        )
        response = self.text_generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
        return response[len(prompt):].strip()

    def extract_tickers(self, text):
        """Extract ticker symbols using spaCy NER and heuristics"""
        doc = nlp(text)
        # Extract entities labeled as ORG or PRODUCT as candidates
        candidates = [ent.text.upper() for ent in doc.ents if ent.label_ in ("ORG", "PRODUCT")]
        # Simple heuristic: filter candidates that are 1-5 uppercase letters (typical ticker format)
        tickers = [c for c in candidates if c.isalpha() and 1 <= len(c) <= 5]
        # Fallback: if no entities, try to find uppercase words manually
        if not tickers:
            tickers = [word for word in text.split() if word.isupper() and 1 <= len(word) <= 5]
        return list(set(tickers)) if tickers else ["AAPL"]  # Default ticker if none found

    def process_query(self, user_query):
        tickers = self.extract_tickers(user_query)
        ticker = tickers[0]  # For simplicity, use first detected ticker

        # Update historical data, skip if API limit or error
        self.insert_historical_data(ticker)

        historical_df = self.fetch_historical_data(ticker)
        indicators = self.calculate_indicators(historical_df) if not historical_df.empty else {"RSI": None, "MACD": None}
        realtime = self.fetch_realtime_data(ticker) or {"price": "N/A", "volume": "N/A"}
        blockchain_prices = self.get_blockchain_data()
        defi_tvl = self.get_defi_data()

        context = (
            f"{ticker} Current Price: ${realtime['price']} | "
            f"RSI: {indicators['RSI'] if indicators['RSI'] else 'N/A'}, "
            f"MACD: {indicators['MACD'] if indicators['MACD'] else 'N/A'}\n"
            f"Crypto Prices (USD): " + ", ".join(f"{k}: ${v:.2f}" if v else f"{k}: N/A" for k,v in blockchain_prices.items()) + "\n"
            f"DeFi Total Value Locked (TVL): ${defi_tvl:.2f}" if defi_tvl else "DeFi TVL: N/A"
        )

        return self.generate_response(user_query, context)

# Streamlit UI for user-friendly interaction
def run_streamlit_app():
    st.title("AI-Powered Financial Chatbot")
    st.write("Ask me about stocks, crypto, DeFi, and more!")

    chatbot = FinancialChatbot()

    user_input = st.text_input("You:", "")
    if st.button("Send") and user_input.strip():
        with st.spinner("Generating response..."):
            response = chatbot.process_query(user_input)
        st.text_area("Assistant:", value=response, height=200)

if __name__ == "__main__":
    # To run the chatbot as a console app, uncomment below:
    chatbot = FinancialChatbot()
    print("Financial LLM Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer = chatbot.process_query(user_input)
        print(f"Assistant: {answer}")

    # Run Streamlit app (recommended for better UX)
    # run_streamlit_app()
