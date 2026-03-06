#!/usr/bin/env python3
import os
import sys
import pickle
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from alpaca.data.historical.crypto import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:
    print("ERROR: alpaca-py not installed!")
    print("Install with: pip install alpaca-py")
    sys.exit(1)


ALL_SYMBOLS = [
    'AAVE', 'AVAX', 'BAT', 'BCH', 'BTC', 'CRV', 'DOGE', 'DOT', 'ETH',
    'GRT', 'LINK', 'LTC', 'PEPE', 'SHIB', 'SOL', 'SUSHI', 'TRUMP',
    'UNI', 'XRP', 'XTZ', 'YFI'
]

DEFAULT_DAYS_BACK = 365 * 6

TIMEFRAMES = {
    '1min': '1min',
    '5min': '5min',
    '15min': '15min',
    '30min': '30min',
    '1h': '1h',
    '2h': '2h',
    '4h': '4h',
    '6h': '6h',
    '1d': '1D',
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data_cache')


def get_cache_path(symbol: str, timeframe: str, days_back: int) -> str:
    symbol_clean = symbol.replace('/', '_')
    return os.path.join(CACHE_DIR, f"{symbol_clean}_{timeframe}_{days_back}d.pkl")


def is_cached(symbol: str, days_back: int) -> bool:
    symbol_usd = f"{symbol}/USD" if '/' not in symbol else symbol
    cache_path = get_cache_path(symbol_usd, '1min', days_back)
    return os.path.exists(cache_path)


def get_cached_info(symbol: str, days_back: int) -> Optional[dict]:
    symbol_usd = f"{symbol}/USD" if '/' not in symbol else symbol
    cache_path = get_cache_path(symbol_usd, '1min', days_back)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            df = pickle.load(f)
        
        return {
            'bars': len(df),
            'first_date': df.index[0].strftime('%Y-%m-%d'),
            'last_date': df.index[-1].strftime('%Y-%m-%d'),
            'size_mb': os.path.getsize(cache_path) / 1024 / 1024
        }
    except Exception as e:
        logger.warning(f"Failed to load cache info for {symbol}: {e}")
        return None


def apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df[df['volume'] > 0]
    df = df[
        (df['high'] >= df['low']) &
        (df['high'] >= df['open']) &
        (df['high'] >= df['close']) &
        (df['low'] <= df['open']) &
        (df['low'] <= df['close'])
    ]
    return df


def download_symbol(symbol: str, days_back: int) -> Optional[pd.DataFrame]:
    symbol_usd = f"{symbol}/USD" if '/' not in symbol else symbol
    
    print(f"\n{'='*70}")
    print(f"Downloading {symbol_usd}")
    print(f"{'='*70}")
    
    try:
        client = CryptoHistoricalDataClient()
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        print(f"  Requesting {days_back} days ({start_time.date()} to {end_time.date()})...")
        
        request = CryptoBarsRequest(
            symbol_or_symbols=symbol_usd,
            timeframe=TimeFrame.Minute,
            start=start_time,
            end=end_time
        )
        
        bars = client.get_crypto_bars(request)
        df = bars.df
        
        if df.empty:
            print(f"  [FAIL] No data returned")
            return None
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol_usd, level='symbol')
        df.columns = df.columns.str.lower()
        df.index = pd.to_datetime(df.index)
        df = apply_quality_filters(df)
        size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        first_date = df.index[0].strftime('%Y-%m-%d')
        last_date = df.index[-1].strftime('%Y-%m-%d')
        
        print(f"  [OK] {len(df):,} bars ({first_date} to {last_date})")
        print(f"  [OK] Size: {size_mb:.1f} MB")
        
        return df
        
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return None


def resample_and_save(df_1min: pd.DataFrame, symbol: str, days_back: int) -> int:
    symbol_usd = f"{symbol}/USD" if '/' not in symbol else symbol
    saved_count = 0
    
    print(f"  Saving timeframes...")
    
    for timeframe, resample_rule in TIMEFRAMES.items():
        try:
            if timeframe == '1min':
                df = df_1min
            else:
                df = df_1min.resample(resample_rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            
            if df.empty:
                continue
            
            cache_path = get_cache_path(symbol_usd, timeframe, days_back)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            
            saved_count += 1
            
        except Exception as e:
            print(f"    [FAIL] {timeframe}: {e}")
    
    print(f"  [OK] Saved {saved_count} timeframes")
    return saved_count


def show_status(symbols: List[str], days_back: int):
    print(f"\n{'='*70}")
    print("CACHE STATUS")
    print(f"{'='*70}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Days back: {days_back}")
    print(f"{'='*70}\n")
    
    cached = []
    missing = []
    
    for symbol in symbols:
        info = get_cached_info(symbol, days_back)
        if info:
            cached.append(symbol)
            print(f"  [CACHED] {symbol}/USD: {info['bars']:,} bars ({info['first_date']} to {info['last_date']})")
        else:
            missing.append(symbol)
            print(f"  [MISSING] {symbol}/USD")
    
    print(f"\n{'='*70}")
    print(f"Cached: {len(cached)}/{len(symbols)}")
    print(f"Missing: {len(missing)}/{len(symbols)}")
    if missing:
        print(f"\nTo download missing: python download_data.py")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Crypto Data Downloader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_data.py                    # Download all missing
    python download_data.py --symbols BTC ETH  # Specific symbols only
    python download_data.py --force            # Re-download everything
    python download_data.py --check            # Show cache status
        """
    )
    
    parser.add_argument('--symbols', nargs='+', type=str,
                        help='Download specific symbols (e.g., BTC ETH SOL)')
    parser.add_argument('--days', type=int, default=DEFAULT_DAYS_BACK,
                        help=f'Days of history (default: {DEFAULT_DAYS_BACK})')
    parser.add_argument('--force', action='store_true',
                        help='Re-download even if already cached')
    parser.add_argument('--check', action='store_true',
                        help='Show cache status without downloading')
    
    args = parser.parse_args()
    
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = ALL_SYMBOLS
    if args.check:
        show_status(symbols, args.days)
        return
    if args.force:
        to_download = symbols
    else:
        to_download = [s for s in symbols if not is_cached(s, args.days)]
    
    if not to_download:
        print(f"\n{'='*70}")
        print("ALL SYMBOLS ALREADY CACHED")
        print(f"{'='*70}")
        print(f"Use --force to re-download")
        print(f"Use --check to see cache details")
        return
    print(f"\n{'='*70}")
    print("CRYPTO DATA DOWNLOADER")
    print(f"{'='*70}")
    print(f"To download: {len(to_download)} symbols")
    print(f"Symbols: {', '.join(to_download)}")
    print(f"Already cached: {len(symbols) - len(to_download)}")
    print(f"Days: {args.days} ({args.days/365:.1f} years)")
    print(f"Timeframes: {', '.join(TIMEFRAMES.keys())}")
    print(f"{'='*70}")
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    start_time = datetime.now()
    successful = []
    failed = []
    
    for symbol in to_download:
        df = download_symbol(symbol, args.days)
        if df is not None and not df.empty:
            resample_and_save(df, symbol, args.days)
            successful.append(symbol)
        else:
            failed.append(symbol)
    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"Successful: {len(successful)}/{len(to_download)}")
    if successful:
        print(f"  Downloaded: {', '.join(successful)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"Duration: {duration:.0f}s ({duration/60:.1f} min)")
    total_size = 0
    if os.path.exists(CACHE_DIR):
        for f in os.listdir(CACHE_DIR):
            if f.endswith('.pkl'):
                total_size += os.path.getsize(os.path.join(CACHE_DIR, f))
    
    print(f"Total cache size: {total_size / 1024 / 1024:.1f} MB")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()