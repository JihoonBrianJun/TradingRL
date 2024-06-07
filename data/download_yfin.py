import os
import yfinance as yf
from argparse import ArgumentParser

def main(args):
    save_dir = os.path.join(args.save_dir, args.country)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if args.country == 'US':
        tickers = ["MSFT", "AAPL", "NVDA", "GOOG", "AMZN",
                   "META", "BRK-B", "LLY", "AVGO", "JPM",
                   "TSLA", "V", "WMT", "XOM", "UNH",
                   "MA", "PG", "COST", "JNJ", "ORCL"]
    elif args.country == 'KR':
        tickers = ["005930.KS", "000660.KS", "005380.KS", "000270.KS", "068270.KS",
                   "005490.KS", "105560.KS", "035420.KS", "006400.KS", "051910.KS",
                   "028260.KS", "055550.KS", "012330.KS", "086790.KS", "032830.KS",
                   "066570.KS", "000810.KS", "042700.KS", "011200.KS", "034730.KS"]
    
    for ticker in tickers:
        data = yf.download(ticker, interval="1d", start=args.start, end=args.end)
        data.to_csv(os.path.join(save_dir, f'{ticker}_{args.start}_{args.end}.csv'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='data')
    parser.add_argument('--country', type=str, default='US')
    parser.add_argument('--start', type=str, default='2000-01-01')
    parser.add_argument('--end', type=str, default='2023-12-31')
    args = parser.parse_args()
    main(args)