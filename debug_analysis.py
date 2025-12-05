
import sys
import os
import pandas as pd
import numpy as np

# Add app directory to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

try:
    from app.services.optimization import analyze_portfolio
    print("Successfully imported analyze_portfolio")
except ImportError as e:
    print(f"Error importing: {e}")
    sys.exit(1)

# Mock data
tickers = ['AAPL', 'MSFT', 'GOOG']
weights = [0.33, 0.33, 0.34]
start_date = '2023-01-01'
end_date = '2023-01-31'
risk_free_rate = 0.02

print("Running analysis...")
try:
    result = analyze_portfolio(tickers, weights, start_date, end_date, risk_free_rate)
    print("Analysis successful!")
    # Check for NaNs in result
    import json
    try:
        json.dumps(result)
        print("JSON serialization successful")
    except Exception as e:
        print(f"JSON serialization FAILED: {e}")
        
except Exception as e:
    print("Analysis FAILED")
    import traceback
    traceback.print_exc()
