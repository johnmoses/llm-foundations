import os
import json
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
import ta
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from web3 import Web3
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
import sqlite3
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FinancialData:
    """Data class for financial information"""
    symbol: str
    price: float
    change: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    indicators: Optional[Dict] = None
    timestamp: datetime = None

class FinancialAPIManager:
    """Manages various financial API integrations"""
    
    def __init__(self):
        # Initialize API keys (replace with actual keys)
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY', 'demo')
        self.polygon_key = os.getenv('POLYGON_KEY', 'demo')
        self.coinbase_api_key = os.getenv('COINBASE_API_KEY', 'demo')
        
    def get_stock_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Fetch stock data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_crypto_data(self, symbol: str) -> Dict:
        """Fetch cryptocurrency data"""
        try:
            url = f"https://api.coinbase.com/v2/exchange-rates?currency={symbol}"
            response = requests.get(url)
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching crypto data: {e}")
            return {}
    
    def get_economic_indicators(self) -> Dict:
        """Fetch economic indicators from Alpha Vantage"""
        try:
            indicators = {}
            # GDP data
            url = f"https://www.alphavantage.co/query?function=REAL_GDP&interval=annual&apikey={self.alpha_vantage_key}"
            response = requests.get(url)
            indicators['gdp'] = response.json()
            
            # Unemployment rate
            url = f"https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey={self.alpha_vantage_key}"
            response = requests.get(url)
            indicators['unemployment'] = response.json()
            
            return indicators
        except Exception as e:
            logger.error(f"Error fetching economic indicators: {e}")
            return {}

class TechnicalIndicatorEngine:
    """Calculates various technical indicators"""
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> Dict:
        """Calculate technical indicators for given price data"""
        indicators = {}
        
        if df.empty or len(df) < 20:
            return indicators
        
        try:
            # Moving averages
            indicators['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20).iloc[-1]
            indicators['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50).iloc[-1] if len(df) >= 50 else None
            indicators['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12).iloc[-1]
            indicators['ema_26'] = ta.trend.ema_indicator(df['Close'], window=26).iloc[-1]
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            indicators['macd'] = macd.macd().iloc[-1]
            indicators['macd_signal'] = macd.macd_signal().iloc[-1]
            indicators['macd_histogram'] = macd.macd_diff().iloc[-1]
            
            # RSI
            indicators['rsi'] = ta.momentum.rsi(df['Close'], window=14).iloc[-1]
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['Close'])
            indicators['bb_upper'] = bb.bollinger_hband().iloc[-1]
            indicators['bb_middle'] = bb.bollinger_mavg().iloc[-1]
            indicators['bb_lower'] = bb.bollinger_lband().iloc[-1]
            
            # Volume indicators
            indicators['volume_sma'] = ta.volume.volume_sma(df['Close'], df['Volume']).iloc[-1]
            
            # Volatility
            indicators['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close']).iloc[-1]
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            indicators['stoch_k'] = stoch.stoch().iloc[-1]
            indicators['stoch_d'] = stoch.stoch_signal().iloc[-1]
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return indicators

class BlockchainIntegration:
    """Handles blockchain data and DeFi protocols"""
    
    def __init__(self):
        # Initialize Web3 connection (replace with actual RPC URL)
        self.w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'))
        self.uniswap_factory = "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"
        
    def get_eth_price(self) -> float:
        """Get current ETH price from on-chain data"""
        try:
            # This is a simplified example - in practice, you'd query a DEX
            # or use a price oracle contract
            response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd')
            return response.json()['ethereum']['usd']
        except Exception as e:
            logger.error(f"Error fetching ETH price: {e}")
            return 0.0
    
    def get_defi_tvl(self, protocol: str) -> Dict:
        """Get Total Value Locked for DeFi protocols"""
        try:
            url = f"https://api.llama.fi/protocol/{protocol}"
            response = requests.get(url)
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching DeFi TVL: {e}")
            return {}
    
    def analyze_token_metrics(self, token_address: str) -> Dict:
        """Analyze on-chain token metrics"""
        try:
            # This would involve querying various on-chain metrics
            # Simplified example using external API
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            response = requests.get(url)
            return response.json()
        except Exception as e:
            logger.error(f"Error analyzing token metrics: {e}")
            return {}

class FinancialLLM:
    """Core LLM for financial analysis and recommendations"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        
    def load_model(self):
        """Load the language model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model.to(self.device)
            logger.info(f"Model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to a simple text generation pipeline
            self.pipeline = pipeline("text-generation", model="gpt2", max_length=100)
    
    def generate_analysis(self, context: str, max_length: int = 200) -> str:
        """Generate financial analysis based on context"""
        try:
            if hasattr(self, 'pipeline'):
                # Use pipeline fallback
                result = self.pipeline(context, max_length=max_length, num_return_sequences=1)
                return result[0]['generated_text']
            
            # Prepare prompt for financial analysis
            prompt = f"Financial Analysis: {context}\n\nBased on the provided data, the analysis suggests:"
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text.split("Based on the provided data, the analysis suggests:")[-1].strip()
            
        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            return "Unable to generate analysis at this time."
    
    def sentiment_analysis(self, text: str) -> Dict:
        """Perform sentiment analysis on financial text"""
        try:
            # Simple keyword-based sentiment analysis
            positive_words = ['bullish', 'growth', 'profit', 'gain', 'strong', 'buy', 'outperform']
            negative_words = ['bearish', 'loss', 'decline', 'weak', 'sell', 'underperform', 'risk']
            
            text_lower = text.lower()
            pos_count = sum(word in text_lower for word in positive_words)
            neg_count = sum(word in text_lower for word in negative_words)
            
            if pos_count > neg_count:
                sentiment = "positive"
                confidence = pos_count / (pos_count + neg_count + 1)
            elif neg_count > pos_count:
                sentiment = "negative"
                confidence = neg_count / (pos_count + neg_count + 1)
            else:
                sentiment = "neutral"
                confidence = 0.5
                
            return {"sentiment": sentiment, "confidence": confidence}
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {"sentiment": "neutral", "confidence": 0.5}

class DatabaseManager:
    """Manages financial data storage"""
    
    def __init__(self, db_name: str = "financial_data.db"):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                price REAL,
                change_percent REAL,
                volume INTEGER,
                timestamp DATETIME,
                indicators TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                analysis_text TEXT,
                sentiment TEXT,
                confidence REAL,
                timestamp DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_stock_data(self, data: FinancialData):
        """Store stock data in database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO stock_data (symbol, price, change_percent, volume, timestamp, indicators)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (data.symbol, data.price, data.change, data.volume, 
              datetime.now(), json.dumps(data.indicators)))
        
        conn.commit()
        conn.close()
    
    def get_historical_data(self, symbol: str, days: int = 30) -> List[Dict]:
        """Retrieve historical data for a symbol"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM stock_data 
            WHERE symbol = ? AND timestamp > datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days), (symbol,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [dict(zip([col[0] for col in cursor.description], row)) for row in results]

class FinancialLLMSystem:
    """Main system orchestrating all components"""
    
    def __init__(self):
        self.api_manager = FinancialAPIManager()
        self.indicator_engine = TechnicalIndicatorEngine()
        self.blockchain = BlockchainIntegration()
        self.llm = FinancialLLM()
        self.db_manager = DatabaseManager()
        
    def analyze_asset(self, symbol: str, asset_type: str = "stock") -> Dict:
        """Comprehensive asset analysis"""
        try:
            analysis_result = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "asset_type": asset_type,
                "price_data": {},
                "technical_indicators": {},
                "fundamental_data": {},
                "blockchain_data": {},
                "llm_analysis": "",
                "sentiment": {},
                "recommendation": ""
            }
            
            # Get price data
            if asset_type == "stock":
                df = self.api_manager.get_stock_data(symbol)
                if not df.empty:
                    current_price = df['Close'].iloc[-1]
                    price_change = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                    
                    analysis_result["price_data"] = {
                        "current_price": current_price,
                        "price_change": price_change,
                        "volume": df['Volume'].iloc[-1],
                        "high_52w": df['High'].max(),
                        "low_52w": df['Low'].min()
                    }
                    
                    # Calculate technical indicators
                    indicators = self.indicator_engine.calculate_indicators(df)
                    analysis_result["technical_indicators"] = indicators
                    
            elif asset_type == "crypto":
                crypto_data = self.api_manager.get_crypto_data(symbol)
                if crypto_data:
                    analysis_result["price_data"] = crypto_data
                    
                    # Get blockchain data
                    if symbol.upper() == "ETH":
                        eth_price = self.blockchain.get_eth_price()
                        analysis_result["blockchain_data"]["eth_price"] = eth_price
            
            # Generate LLM analysis
            context = f"Asset: {symbol} ({asset_type})\n"
            context += f"Price Data: {analysis_result['price_data']}\n"
            context += f"Technical Indicators: {analysis_result['technical_indicators']}\n"
            
            llm_analysis = self.llm.generate_analysis(context)
            analysis_result["llm_analysis"] = llm_analysis
            
            # Sentiment analysis
            sentiment = self.llm.sentiment_analysis(llm_analysis)
            analysis_result["sentiment"] = sentiment
            
            # Generate recommendation
            recommendation = self.generate_recommendation(analysis_result)
            analysis_result["recommendation"] = recommendation
            
            # Store in database
            financial_data = FinancialData(
                symbol=symbol,
                price=analysis_result["price_data"].get("current_price", 0),
                change=analysis_result["price_data"].get("price_change", 0),
                volume=analysis_result["price_data"].get("volume", 0),
                indicators=analysis_result["technical_indicators"],
                timestamp=datetime.now()
            )
            self.db_manager.store_stock_data(financial_data)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in asset analysis: {e}")
            return {"error": str(e)}
    
    def generate_recommendation(self, analysis: Dict) -> str:
        """Generate investment recommendation based on analysis"""
        try:
            indicators = analysis.get("technical_indicators", {})
            sentiment = analysis.get("sentiment", {})
            price_data = analysis.get("price_data", {})
            
            # Simple rule-based recommendation system
            score = 0
            reasons = []
            
            # Technical analysis scoring
            if indicators.get("rsi"):
                rsi = indicators["rsi"]
                if rsi < 30:
                    score += 2
                    reasons.append("RSI indicates oversold condition")
                elif rsi > 70:
                    score -= 2
                    reasons.append("RSI indicates overbought condition")
            
            # MACD analysis
            if indicators.get("macd") and indicators.get("macd_signal"):
                if indicators["macd"] > indicators["macd_signal"]:
                    score += 1
                    reasons.append("MACD showing bullish signal")
                else:
                    score -= 1
                    reasons.append("MACD showing bearish signal")
            
            # Moving average analysis
            if indicators.get("sma_20") and indicators.get("sma_50"):
                if indicators["sma_20"] > indicators["sma_50"]:
                    score += 1
                    reasons.append("Short-term MA above long-term MA")
                else:
                    score -= 1
                    reasons.append("Short-term MA below long-term MA")
            
            # Sentiment scoring
            if sentiment.get("sentiment") == "positive":
                score += sentiment.get("confidence", 0) * 2
                reasons.append("Positive sentiment detected")
            elif sentiment.get("sentiment") == "negative":
                score -= sentiment.get("confidence", 0) * 2
                reasons.append("Negative sentiment detected")
            
            # Generate recommendation
            if score >= 3:
                recommendation = "BUY"
            elif score <= -3:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            return f"{recommendation} - Score: {score:.1f} - Reasons: {'; '.join(reasons)}"
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return "HOLD - Unable to generate recommendation"
    
    def portfolio_analysis(self, symbols: List[str]) -> Dict:
        """Analyze a portfolio of assets"""
        portfolio_result = {
            "portfolio_summary": {},
            "individual_analyses": {},
            "correlation_matrix": None,
            "risk_metrics": {},
            "recommendations": []
        }
        
        try:
            # Analyze each asset
            for symbol in symbols:
                analysis = self.analyze_asset(symbol)
                portfolio_result["individual_analyses"][symbol] = analysis
            
            # Calculate portfolio metrics (simplified)
            total_value = 0
            total_change = 0
            
            for symbol, analysis in portfolio_result["individual_analyses"].items():
                if "price_data" in analysis:
                    price = analysis["price_data"].get("current_price", 0)
                    change = analysis["price_data"].get("price_change", 0)
                    total_value += price
                    total_change += change
            
            portfolio_result["portfolio_summary"] = {
                "total_assets": len(symbols),
                "avg_change": total_change / len(symbols) if symbols else 0,
                "timestamp": datetime.now()
            }
            
            return portfolio_result
            
        except Exception as e:
            logger.error(f"Error in portfolio analysis: {e}")
            return {"error": str(e)}
    
    def get_market_overview(self) -> Dict:
        """Get overall market overview"""
        try:
            # Major indices
            indices = ["^GSPC", "^DJI", "^IXIC", "^RUT"]  # S&P 500, Dow, NASDAQ, Russell 2000
            market_data = {}
            
            for index in indices:
                df = self.api_manager.get_stock_data(index, period="5d")
                if not df.empty:
                    current = df['Close'].iloc[-1]
                    previous = df['Close'].iloc[-2]
                    change = ((current - previous) / previous) * 100
                    
                    market_data[index] = {
                        "price": current,
                        "change": change,
                        "volume": df['Volume'].iloc[-1]
                    }
            
            # Economic indicators
            economic_data = self.api_manager.get_economic_indicators()
            
            # Crypto market cap
            crypto_data = {
                "bitcoin": self.api_manager.get_crypto_data("BTC"),
                "ethereum": self.api_manager.get_crypto_data("ETH")
            }
            
            # Generate market analysis
            context = f"Market Overview:\nIndices: {market_data}\nCrypto: {crypto_data}"
            market_analysis = self.llm.generate_analysis(context)
            
            return {
                "indices": market_data,
                "economic_indicators": economic_data,
                "crypto_market": crypto_data,
                "market_analysis": market_analysis,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {"error": str(e)}

# Usage example and testing
def main():
    """Main function to demonstrate the system"""
    print("Initializing Financial LLM System...")
    system = FinancialLLMSystem()
    
    # Test single asset analysis
    print("\n=== Single Asset Analysis ===")
    analysis = system.analyze_asset("AAPL", "stock")
    print(f"Analysis for AAPL:")
    print(f"Price: ${analysis.get('price_data', {}).get('current_price', 'N/A')}")
    print(f"Change: {analysis.get('price_data', {}).get('price_change', 'N/A')}%")
    print(f"RSI: {analysis.get('technical_indicators', {}).get('rsi', 'N/A')}")
    print(f"Recommendation: {analysis.get('recommendation', 'N/A')}")
    print(f"LLM Analysis: {analysis.get('llm_analysis', 'N/A')[:100]}...")
    
    # Test portfolio analysis
    print("\n=== Portfolio Analysis ===")
    portfolio = system.portfolio_analysis(["AAPL", "GOOGL", "MSFT"])
    print(f"Portfolio Summary: {portfolio.get('portfolio_summary', {})}")
    
    # Test market overview
    print("\n=== Market Overview ===")
    market = system.get_market_overview()
    print(f"Market Analysis: {market.get('market_analysis', 'N/A')[:100]}...")
    
    print("\nSystem demonstration completed!")

if __name__ == "__main__":
    main()