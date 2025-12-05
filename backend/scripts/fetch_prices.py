import argparse
import yfinance as yf
import pandas as pd
import os

def fetch_prices(tickers, start_date, output_file):
    print(f"Fetching data for {tickers} from {start_date}...")
    
    # Download data
    data = yf.download(tickers, start=start_date, group_by='ticker', auto_adjust=False)
    
    # Reshape data to long format
    dfs = []
    for ticker in tickers:
        df = data[ticker].copy()
        df['ticker'] = ticker
        df = df.reset_index()
        # Ensure columns are lowercase
        df.columns = [c.lower() for c in df.columns]
        dfs.append(df)
        
    final_df = pd.concat(dfs)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch historical prices.')
    parser.add_argument('--tickers', type=str, required=True, help='Comma separated tickers (e.g. META,AVGO,GLD)')
    parser.add_argument('--from_date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='../seed_data/prices.csv', help='Output CSV file')
    
    args = parser.parse_args()
    tickers_list = args.tickers.split(',')
    
    fetch_prices(tickers_list, args.from_date, args.output)
