# Large Language Model (LLM) for Finance

## Introduction

Integrating LLMs with financial systems bring the power of AI into personald and corporate finance. Core features that we will be looking at are as follows.

1. Database Integration (SQLite)

    Price data storage with timestamps
    Technical indicators tracking
    DeFi protocol data
    Chat history persistence

2. API Integrations

    CoinGecko API: Free cryptocurrency data
    AlphaVantage: Stock market data (configurable)
    Web3 Providers: Ethereum and BSC blockchain connectivity

3. Technical Analysis

    RSI (Relative Strength Index)
    MACD (Moving Average Convergence Divergence)
    SMA (Simple Moving Average)
    Bollinger Bands
    Uses TA-Lib for reliable calculations

4. Blockchain & Crypto Features

    Balance checking for Ethereum addresses
    Transaction count tracking
    Gas price monitoring
    Multi-chain support (Ethereum, BSC)

5. DeFi Analysis

    Protocol landscape analysis
    Market cap tracking
    Yield farming opportunities
    Top protocols ranking

6. LLM Chatbot

    Uses Microsoft's DialoGPT (open-source)
    Context-aware responses
    Financial data integration
    Fallback responses when LLM unavailable

7. CLI Interface

    Interactive commands for all features
    Rich terminal UI with tables and panels
    Real-time data fetching with progress indicators

8. Account Management

    Account Management & Balance Inquiry To demonstrate checking balances and recent transactions
    Spending Analysis & Budgeting for Offering insights into spending patterns
    Financial Product Information for providing details on different financial products
    Loan/Credit Application Assistance for Guiding users through a simplified application process

## Setup

```bash
# Install dependencies
pip install flask yfinance llama-cpp-python bert-score nltk chart.js

# Run specific commands
python app.py price -s bitcoin
python app.py analyze -s ethereum
python app.py defi
python app.py balance -a 0x742d35Cc6634C0532925a3b8D3A8d00FaBb8B06C
python app.py trending
python app.py chat  # Interactive chatbot

# Or run the CLI help
python app.py --help
```

Sample queries

Technical Indicators:
	•	`rsi`
	•	Crypto Price:
	•	`price BTCUSDT`
	•	`price ETHUSDT`
	•	Account Management & Balance Inquiry:
	•	`balance` (for checking account)
	•	`balance checking`
	•	`balance savings`
	•	Spending Analysis:
	•	`spending`
	•	Financial Product Information:
	•	`product info savings account`
	•	`product info investment options`
	•	`product info loan`
	•	`product info credit card`
	•	`product info mortgage`
	•	Loan/Credit Application Assistance:
    •	`loan application` (for personal loan)
	•	`loan application personal`
	•	`loan application mortgage`
	•	General LLM Queries (handled by the LLM):
	•	`What is inflation?`
	•	`Tell me about stock market trends.`
	•	`How can I save money effectively?`
	•	`What is the current economic outlook?`