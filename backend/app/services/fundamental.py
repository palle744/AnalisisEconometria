import yfinance as yf
from typing import Dict, List

def analyze_fundamental(ticker_symbol: str, years: List[int]) -> Dict:
    """
    Real implementation using yfinance.
    Note: yfinance financials are often limited to the last 4 years.
    """
    ticker = yf.Ticker(ticker_symbol)
    
    # Get financials (balance sheet, income statement, cash flow)
    # These return pandas DataFrames
    balance_sheet = ticker.balance_sheet
    income_stmt = ticker.income_stmt
    
    results = {}
    
    # yfinance columns are dates. We need to map requested years to the closest available date.
    available_dates = income_stmt.columns
    
    for year in years:
        # Find column for the year
        col_date = None
        for date in available_dates:
            if date.year == year:
                col_date = date
                break
        
        if col_date is None:
            results[year] = {"error": "Data not available for this year"}
            continue
            
        try:
            # Extract data safely (using .get to avoid errors if row missing)
            revenue = income_stmt.loc['Total Revenue', col_date] if 'Total Revenue' in income_stmt.index else 0
            net_income = income_stmt.loc['Net Income', col_date] if 'Net Income' in income_stmt.index else 0
            total_assets = balance_sheet.loc['Total Assets', col_date] if 'Total Assets' in balance_sheet.index else 0
            total_equity = balance_sheet.loc['Stockholders Equity', col_date] if 'Stockholders Equity' in balance_sheet.index else 0
            
            # Calculate ratios
            net_margin = net_income / revenue if revenue else 0
            roe = net_income / total_equity if total_equity else 0
            roa = net_income / total_assets if total_assets else 0
            asset_turnover = revenue / total_assets if total_assets else 0
            
            results[year] = {
                "revenue": float(revenue),
                "net_income": float(net_income),
                "total_assets": float(total_assets),
                "total_equity": float(total_equity),
                "ratios": {
                    "net_margin": round(net_margin, 4),
                    "roe": round(roe, 4),
                    "roa": round(roa, 4),
                    "asset_turnover": round(asset_turnover, 4)
                }
            }
        except Exception as e:
            results[year] = {"error": str(e)}

    return {
        "ticker": ticker_symbol,
        "financials": results,
        "source": "Yahoo Finance"
    }
