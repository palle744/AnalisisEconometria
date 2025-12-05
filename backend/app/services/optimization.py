import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

import yfinance as yf

def get_real_data(tickers: List[str], start_date, end_date) -> pd.DataFrame:
    """
    Fetches real price data from yfinance.
    """
    # Adjust end_date to be inclusive (yfinance end is exclusive)
    if isinstance(end_date, str):
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
    else:
        end_date_obj = end_date
        
    adj_end_date = end_date_obj + timedelta(days=1)

    # Download data
    data = yf.download(tickers, start=start_date, end=adj_end_date, progress=False, auto_adjust=False)
    
    # yfinance returns MultiIndex columns if multiple tickers.
    # We want 'Adj Close' or 'Close'.
    if isinstance(data.columns, pd.MultiIndex):
        # Try to find Adj Close level
        if 'Adj Close' in data.columns.get_level_values(0):
             prices = data['Adj Close']
        elif 'Close' in data.columns.get_level_values(0):
             prices = data['Close']
        else:
             # Fallback: take the first level whatever it is
             prices = data.iloc[:, 0] # This might be wrong if multi-level, but better than crash
    else:
        # Single level
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
        else:
            prices = data
            
    # Handle single ticker case (returns Series instead of DataFrame)
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
        
    # Handle missing data
    prices = prices.dropna()
    
    return prices

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe = (returns - risk_free_rate) / std
    return returns, std, sharpe

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def portfolio_volatility(weights, mean_returns, cov_matrix, risk_free_rate):
    return portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[1]

def calculate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, num_assets):
    frontier_points = []
    min_ret = mean_returns.min() * 252
    max_ret = mean_returns.max() * 252
    if min_ret >= max_ret:
        min_ret = max_ret - 0.1
        
    target_returns = np.linspace(min_ret, max_ret, 20)
    args = (mean_returns, cov_matrix, risk_free_rate)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets,]
    
    for target in target_returns:
        cons = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x, t=target: portfolio_performance(x, mean_returns, cov_matrix, risk_free_rate)[0] - t}
        )
        try:
            res = minimize(portfolio_volatility, init_guess, args=args,
                           method='SLSQP', bounds=bounds, constraints=cons)
            if res.success:
                r, s, sh = portfolio_performance(res.x, mean_returns, cov_matrix, risk_free_rate)
                frontier_points.append({
                    "return": float(r),
                    "std_dev": float(s),
                    "sharpe": float(sh)
                })
        except:
            continue
    return frontier_points

def run_optimization(
    tickers: List[str],
    start_date,
    end_date,
    risk_free_rate: float,
    allow_short: bool,
    constraints: Dict[str, List[float]] = None
) -> Dict:
    
    # 1. Get Data
    prices = get_real_data(tickers, start_date, end_date)
    if prices.empty:
        raise ValueError("No data found for the specified tickers and date range.")
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    num_assets = len(tickers)
    args = (mean_returns, cov_matrix, risk_free_rate)
    
    # Constraints
    # Sum of weights = 1
    constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Bounds
    if allow_short:
        bounds = tuple((-1.0, 1.0) for _ in range(num_assets))
    else:
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        
    # Initial guess
    init_guess = num_assets * [1. / num_assets,]
    
    # Optimize for Max Sharpe
    result = minimize(neg_sharpe_ratio, init_guess, args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints_list)
    
    weights = result.x
    perf = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    
    cleaned_weights = {ticker: float(round(weight, 4)) for ticker, weight in zip(tickers, weights)}
    
    # Efficient Frontier
    frontier_points = calculate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, num_assets)
            
    # Get latest prices for allocation calculation
    latest_prices = {ticker: float(prices[ticker].iloc[-1]) for ticker in tickers}
    start_prices = {ticker: float(prices[ticker].iloc[0]) for ticker in tickers}
    
    # Historical Data
    historical_dates = prices.index.strftime('%Y-%m-%d').tolist()
    historical_prices = {ticker: [float(p) for p in prices[ticker].tolist()] for ticker in tickers}

    # News
    news = {}
    try:
        for ticker in tickers:
            t = yf.Ticker(ticker)
            raw_news = t.news
            normalized_news = []
            for item in raw_news:
                if 'content' in item:
                    # New structure
                    c = item['content']
                    pub_date_str = c.get('pubDate', '')
                    timestamp = 0
                    try:
                        if pub_date_str:
                            dt = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                            timestamp = int(dt.timestamp())
                    except:
                        pass
                        
                    normalized_news.append({
                        'uuid': item.get('id'),
                        'title': c.get('title'),
                        'link': c.get('clickThroughUrl') and c.get('clickThroughUrl').get('url'),
                        'publisher': c.get('provider', {}).get('displayName'),
                        'providerPublishTime': timestamp,
                        'thumbnail': c.get('thumbnail')
                    })
                else:
                    # Old structure (fallback)
                    normalized_news.append(item)
            
            # Sort by providerPublishTime descending
            normalized_news.sort(key=lambda x: x.get('providerPublishTime', 0), reverse=True)
            news[ticker] = normalized_news
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        pass

    # Advanced Analysis: Cumulative Performance & Correlation
    cumulative_performance = None
    correlation_matrix = None
    
    try:
        # 1. Correlation Matrix
        returns = prices.pct_change().dropna()
        if not returns.empty:
            correlation_matrix = {
                "index": tickers,
                "data": returns.corr().values.tolist()
            }

        # 2. Cumulative Performance vs Benchmark (SPY)
        # Portfolio Cumulative Return
        portfolio_daily_ret = returns[tickers].dot(list(cleaned_weights.values()))
        portfolio_cum_ret = (1 + portfolio_daily_ret).cumprod()
        
        # Benchmark (SPY)
        # Fix: auto_adjust=False to ensure we get Adj Close
        spy_data = yf.download("SPY", start=prices.index[0], end=prices.index[-1], progress=False, auto_adjust=False)['Adj Close']
        
        # Handle yfinance returning DataFrame with MultiIndex or Series
        if isinstance(spy_data, pd.DataFrame):
            if not spy_data.empty:
                spy_data = spy_data.iloc[:, 0] # Take first column if multiple
            else:
                spy_data = pd.Series() # Empty series
        
        if not spy_data.empty:
            # Align dates
            spy_data = spy_data.reindex(prices.index, method='ffill').dropna()
            spy_ret = spy_data.pct_change().dropna()
            spy_cum_ret = (1 + spy_ret).cumprod()
            
            # Align lengths
            common_index = portfolio_cum_ret.index.intersection(spy_cum_ret.index)
            portfolio_cum_ret = portfolio_cum_ret.loc[common_index]
            spy_cum_ret = spy_cum_ret.loc[common_index]
            
            cumulative_performance = {
                "dates": common_index.strftime('%Y-%m-%d').tolist(),
                "portfolio": [0 if np.isnan(x) else x for x in ((portfolio_cum_ret - 1) * 100).tolist()],
                "benchmark": [0 if np.isnan(x) else x for x in ((spy_cum_ret - 1) * 100).tolist()]
            }
        else:
             # Fallback if SPY fails but we have portfolio data
             cumulative_performance = {
                "dates": portfolio_cum_ret.index.strftime('%Y-%m-%d').tolist(),
                "portfolio": [0 if np.isnan(x) else x for x in ((portfolio_cum_ret - 1) * 100).tolist()],
                "benchmark": []
            }
            
    except Exception as e:
        print(f"Error in advanced analysis: {e}")
        # Ensure we don't crash, just return None for these fields
        pass

    # Sanitize inputs to avoid JSON serialization errors (NaN/Inf)
    def sanitize(obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return 0.0
            return obj
        if isinstance(obj, list):
            return [sanitize(x) for x in obj]
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        return obj

    result = {
        "weights": cleaned_weights,
        "expected_return": float(perf[0]),
        "portfolio_std": float(perf[1]),
        "sharpe": float(perf[2]),
        "frontier": frontier_points,
        "latest_prices": latest_prices,
        "start_prices": start_prices,
        "historical_dates": historical_dates,
        "historical_prices": historical_prices,
        "news": news,
        "cumulative_performance": cumulative_performance,
        "correlation_matrix": correlation_matrix
    }
    return sanitize(result)

def analyze_portfolio(
    tickers: List[str],
    weights: List[float],
    start_date,
    end_date,
    risk_free_rate: float
) -> Dict:
    # 1. Get Data
    prices = get_real_data(tickers, start_date, end_date)
    if prices.empty:
        raise ValueError("No data found.")
        
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # 2. Calculate Performance
    w = np.array(weights)
    # Normalize weights just in case
    w = w / np.sum(w)
    
    perf = portfolio_performance(w, mean_returns, cov_matrix, risk_free_rate)
    
    # 3. Get latest prices
    latest_prices = {ticker: float(prices[ticker].iloc[-1]) for ticker in tickers}
    start_prices = {ticker: float(prices[ticker].iloc[0]) for ticker in tickers}
    
    # Historical Data
    historical_dates = prices.index.strftime('%Y-%m-%d').tolist()
    historical_prices = {ticker: [float(p) for p in prices[ticker].tolist()] for ticker in tickers}
    
    # News
    news = {}
    try:
        for ticker in tickers:
            t = yf.Ticker(ticker)
            raw_news = t.news
            normalized_news = []
            for item in raw_news:
                if 'content' in item:
                    # New structure
                    c = item['content']
                    pub_date_str = c.get('pubDate', '')
                    timestamp = 0
                    try:
                        if pub_date_str:
                            dt = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                            timestamp = int(dt.timestamp())
                    except:
                        pass
                        
                    normalized_news.append({
                        'uuid': item.get('id'),
                        'title': c.get('title'),
                        'link': c.get('clickThroughUrl') and c.get('clickThroughUrl').get('url'),
                        'publisher': c.get('provider', {}).get('displayName'),
                        'providerPublishTime': timestamp,
                        'thumbnail': c.get('thumbnail')
                    })
                else:
                    # Old structure (fallback)
                    normalized_news.append(item)
            
            # Sort by providerPublishTime descending
            normalized_news.sort(key=lambda x: x.get('providerPublishTime', 0), reverse=True)
            news[ticker] = normalized_news
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        pass

    cleaned_weights = {ticker: float(round(weight, 4)) for ticker, weight in zip(tickers, w)}
    
    # 4. Calculate Frontier
    frontier_points = calculate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, len(tickers))

    # Advanced Analysis: Cumulative Performance & Correlation
    cumulative_performance = None
    correlation_matrix = None
    
    try:
        # 1. Correlation Matrix
        returns = prices.pct_change().dropna()
        if not returns.empty:
            correlation_matrix = {
                "index": tickers,
                "data": returns.corr().values.tolist()
            }

        # 2. Cumulative Performance vs Benchmark (SPY)
        # Portfolio Cumulative Return
        portfolio_daily_ret = returns[tickers].dot(list(cleaned_weights.values()))
        portfolio_cum_ret = (1 + portfolio_daily_ret).cumprod()
        
        # Benchmark (SPY)
        # Fix: auto_adjust=False to ensure we get Adj Close
        spy_data = yf.download("SPY", start=prices.index[0], end=prices.index[-1], progress=False, auto_adjust=False)['Adj Close']
        
        # Handle yfinance returning DataFrame with MultiIndex or Series
        if isinstance(spy_data, pd.DataFrame):
            if not spy_data.empty:
                spy_data = spy_data.iloc[:, 0] # Take first column if multiple
            else:
                spy_data = pd.Series()
        
        if not spy_data.empty:
            # Align dates
            spy_data = spy_data.reindex(prices.index, method='ffill').dropna()
            spy_ret = spy_data.pct_change().dropna()
            spy_cum_ret = (1 + spy_ret).cumprod()
            
            # Align lengths
            common_index = portfolio_cum_ret.index.intersection(spy_cum_ret.index)
            portfolio_cum_ret = portfolio_cum_ret.loc[common_index]
            spy_cum_ret = spy_cum_ret.loc[common_index]
            
            cumulative_performance = {
                "dates": common_index.strftime('%Y-%m-%d').tolist(),
                "portfolio": [0 if np.isnan(x) else x for x in ((portfolio_cum_ret - 1) * 100).tolist()], # Percentage growth
                "benchmark": [0 if np.isnan(x) else x for x in ((spy_cum_ret - 1) * 100).tolist()]
            }
        else:
             # Fallback if SPY fails but we have portfolio data
             cumulative_performance = {
                "dates": portfolio_cum_ret.index.strftime('%Y-%m-%d').tolist(),
                "portfolio": [0 if np.isnan(x) else x for x in ((portfolio_cum_ret - 1) * 100).tolist()],
                "benchmark": []
            }

    except Exception as e:
        print(f"Error in advanced analysis: {e}")
        pass

    # Sanitize inputs to avoid JSON serialization errors (NaN/Inf)
    def sanitize(obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return 0.0
            return obj
        if isinstance(obj, list):
            return [sanitize(x) for x in obj]
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        return obj

    result = {
        "weights": cleaned_weights,
        "expected_return": float(perf[0]),
        "portfolio_std": float(perf[1]),
        "sharpe": float(perf[2]),
        "frontier": frontier_points,
        "latest_prices": latest_prices,
        "start_prices": start_prices,
        "historical_dates": historical_dates,
        "historical_prices": historical_prices,
        "news": news,
        "cumulative_performance": cumulative_performance,
        "correlation_matrix": correlation_matrix
    }
    return sanitize(result)
