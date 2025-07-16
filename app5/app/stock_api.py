import yfinance as yf

def get_stock_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        price = info.get('regularMarketPrice')
        change = info.get('regularMarketChangePercent')
        name = info.get('shortName', symbol)
        if not price:
            return None
        return {
            'name': name,
            'symbol': symbol.upper(),
            'price': price,
            'change_percent': round(change, 2) if change else None
        }
    except Exception:
        return None
