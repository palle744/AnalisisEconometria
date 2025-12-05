import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict

from datetime import datetime, timedelta

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_technical(ticker: str, start_date: str, end_date: str, indicators: List[str]) -> Dict:
    """
    Real implementation using yfinance data.
    """
    # Adjust end_date to be inclusive (yfinance end is exclusive)
    if isinstance(end_date, str):
        try:
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
            adj_end_date = end_date_obj + timedelta(days=1)
        except ValueError:
            # Fallback if date format is wrong, though it should be validated by Pydantic
            adj_end_date = end_date
    else:
        adj_end_date = end_date

    # Fetch data
    df = yf.download(ticker, start=start_date, end=adj_end_date)
    
    if df.empty:
        return {"error": "No data found for ticker"}
        
    # yfinance returns MultiIndex columns if multiple tickers, but here we ask for one.
    # However, sometimes it returns 'Adj Close' and 'Close'. We prefer 'Adj Close'.
    if 'Adj Close' in df.columns:
        prices = df['Adj Close']
    elif 'Close' in df.columns:
        prices = df['Close']
    else:
        # Fallback if structure is different (e.g. single level)
        prices = df.iloc[:, 0] # Assume first column is close-like if structure is weird

    # Ensure prices is a Series
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    result = {
        "ticker": ticker,
        "dates": prices.index.strftime('%Y-%m-%d').tolist(),
        "price": prices.tolist(),
        "indicators": {}
    }
    
    if "SMA_50" in indicators:
        result["indicators"]["SMA_50"] = prices.rolling(window=50).mean().fillna(0).tolist()
        
    if "RSI_14" in indicators:
        result["indicators"]["RSI_14"] = calculate_rsi(prices, 14).fillna(50).tolist()
        
    if "Bollinger" in indicators:
        sma = prices.rolling(window=20).mean()
        std = prices.rolling(window=20).std()
        result["indicators"]["BB_Upper"] = (sma + 2 * std).fillna(0).tolist()
        result["indicators"]["BB_Lower"] = (sma - 2 * std).fillna(0).tolist()

    return result
