import yfinance as yf
from argparse import ArgumentParser

def main(args):
    tickers = ["MSFT", "AAPL", "NVDA", "GOOG", "AMZN",
               "META", "BRK-B", "LLY", "AVGO", "JPM",
               "TSLA", "V", "WMT", "XOM", "UNH", 
               "MA", "PG", "COST", "JNJ", "ORCL"]
    for ticker in tickers:
        data = yf.download(ticker, interval="1d", start=args.start, end=args.end)
        data.to_csv(f'data/{ticker}_{args.start}_{args.end}.csv')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--start', type=str, default='2000-01-01')
    parser.add_argument('--end', type=str, default='2023-12-31')
    args = parser.parse_args()
    main(args)