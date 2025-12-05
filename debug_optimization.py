
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add app directory to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

try:
    from app.services.optimization import run_optimization
    print("Successfully imported run_optimization")
except ImportError as e:
    print(f"Error importing: {e}")
    sys.exit(1)

# Mock data
tickers = ['META', 'AVGO', 'GLD']
start_date = '2023-01-01'
end_date = '2023-01-31'
risk_free_rate = 0.02

print("Running optimization...")
try:
    result = run_optimization(tickers, start_date, end_date, risk_free_rate, allow_short=False)
    print("Optimization successful!")
    print(result.keys())
except Exception as e:
    print("Optimization FAILED")
    import traceback
    traceback.print_exc()
