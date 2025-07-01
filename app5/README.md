# App 5

Integrating LLMs with financial systems. Core features include

- APIs
- Technical indicators
- Blockchain data
- Named entity recognition (NER) for ticker extraction
- Multi-asset blockchain data fetching from BTC, ETH, and USDT prices from Alpha Vantage’s currency exchange endpoint
- DeFi data integration that Pulls Total Value Locked (TVL) data from DefiLlama public API
- Chatbot interface with command line as well as Streamlit

## Requirements

```bash
pip install transformers huggingface_hub pyti pandas requests spacy streamlit
python -m spacy download en_core_web_sm
```
