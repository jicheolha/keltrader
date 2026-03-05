#!/usr/bin/env python3
"""
January 2026 Backtest - Shows all trades the bot would have made.

Usage:
    python january_backtest.py
"""
import os
import sys
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from technical import BBSqueezeAnalyzer
from signal_generator import BBSqueezeSignalGenerator
from backtester import BBSqueezeBacktester

try:
    from alpaca.data.historical.crypto import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:
    print("ERROR: alpaca-py not installed!")
    print("Install with: pip install alpaca-py")
    sys.exit(1)


# =============================================================================
# CONFIGURATION - Same as run_live_multi_asset.py
# =============================================================================

SYMBOLS = ['DOGE/USD', 'BTC/USD','ETH/USD','SOL/USD', 'XRP/USD']

# Timeframes
SIGNAL_TIMEFRAME = '4h'
ATR_TIMEFRAME = '1h'

# Bollinger Bands
BB_PERIOD = 19
BB_STD = 2.47

# Keltner Channels
KC_PERIOD = 17
KC_ATR_MULT = 2.38

# Momentum
MOMENTUM_PERIOD = 15

# RSI
RSI_PERIOD = 21
RSI_OVERBOUGHT = 68
RSI_OVERSOLD = 25

# Squeeze
MIN_SQUEEZE_BARS = 2

# Volume
VOLUME_PERIOD = 45
MIN_VOLUME_RATIO = 1.02

# Stops
ATR_PERIOD = 16
ATR_STOP_MULT = 3.45
ATR_TARGET_MULT = 4.0

# Setup
SETUP_VALIDITY_BARS = 8

# Position Sizing
BASE_POSITION = 0.25
MIN_POSITION = 0.10
MAX_POSITION = 0.50

# Risk
MAX_POSITIONS = 2
MAX_DAILY_LOSS = 0.03
MAX_HOLD_DAYS = 7
LONG_ONLY = False

# Leverage
INITIAL_CAPITAL = 800
MAINTENANCE_MARGIN = 0.666


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_january_data():
    """Fetch January 2026 data from Alpaca."""
    print("Fetching January 2026 data from Alpaca...")
    
    client = CryptoHistoricalDataClient()
    
    # January 2026 + buffer for indicators
    start = datetime(2025, 12, 1)  # Buffer for indicator warmup
    end = datetime(2026, 2, 1)
    
    data = {}
    
    for symbol in SYMBOLS:
        print(f"  Fetching {symbol}...")
        
        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end
            )
            
            bars = client.get_crypto_bars(request)
            df = bars.df
            
            if df.empty:
                print(f"    No data for {symbol}")
                continue
            
            # Handle multi-index
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level='symbol')
            
            df.columns = df.columns.str.lower()
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Filter quality
            df = df[df['volume'] > 0]
            
            print(f"    Got {len(df):,} 1-min bars")
            
            # Resample to needed timeframes
            data[symbol] = {
                'trade': df,  # 1min for execution
                'signal': resample(df, '4H'),
                'atr': resample(df, '1H'),
            }
            
            print(f"    Signal (4h): {len(data[symbol]['signal'])} bars")
            print(f"    ATR (1h): {len(data[symbol]['atr'])} bars")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
    
    return data


def resample(df, rule):
    """
    Resample OHLCV data with Coinbase-aligned boundaries.
    
    Coinbase 4h candles: 0:00, 4:00, 8:00, 12:00, 16:00, 20:00 UTC
    Coinbase 1h candles: 0:00, 1:00, 2:00, ... UTC
    
    Using origin='start_day' ensures candles align to midnight UTC,
    which matches Coinbase's candle boundaries.
    """
    return df.resample(rule, origin='start_day').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()


def filter_january(data):
    """Filter data to January 2026 only for backtest period."""
    jan_start = pd.Timestamp('2026-01-01', tz='UTC')
    jan_end = pd.Timestamp('2026-02-01', tz='UTC')
    
    filtered = {}
    for symbol in data:
        filtered[symbol] = {}
        for tf, df in data[symbol].items():
            # Keep all data for indicators, backtest will use January
            filtered[symbol][tf] = df
    
    return filtered


def is_4h_boundary(ts) -> bool:
    """Check if timestamp is on a 4h candle boundary (0, 4, 8, 12, 16, 20 UTC)."""
    return ts.hour % 4 == 0 and ts.minute == 0


def filter_to_4h_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter 1-min trade data to only include 4h boundary times.
    This makes the backtester only check for entries at candle closes.
    """
    mask = df.index.map(is_4h_boundary)
    return df[mask]


# =============================================================================
# BACKTEST
# =============================================================================

def run_january_backtest(data):
    """Run backtest on January 2026 data."""
    
    print("\n" + "="*70)
    print("JANUARY 2026 BACKTEST")
    print("="*70)
    print(f"Capital: ${INITIAL_CAPITAL:,}")
    print(f"Leverage: ENABLED (maintenance margin: {MAINTENANCE_MARGIN:.1%})")
    print(f"Max Positions: {MAX_POSITIONS}")
    print(f"Position Size: {BASE_POSITION:.0%} base, {MIN_POSITION:.0%}-{MAX_POSITION:.0%} range")
    print("="*70 + "\n")
    
    # Initialize analyzer
    analyzer = BBSqueezeAnalyzer(
        bb_period=BB_PERIOD,
        bb_std=BB_STD,
        kc_period=KC_PERIOD,
        kc_atr_mult=KC_ATR_MULT,
        momentum_period=MOMENTUM_PERIOD,
        rsi_period=RSI_PERIOD,
        volume_period=VOLUME_PERIOD,
        atr_period=ATR_PERIOD
    )
    
    # Initialize signal generator
    signal_gen = BBSqueezeSignalGenerator(
        analyzer=analyzer,
        min_squeeze_bars=MIN_SQUEEZE_BARS,
        min_volume_ratio=MIN_VOLUME_RATIO,
        rsi_overbought=RSI_OVERBOUGHT,
        rsi_oversold=RSI_OVERSOLD,
        atr_stop_mult=ATR_STOP_MULT,
        atr_target_mult=ATR_TARGET_MULT,
        base_position=BASE_POSITION,
        min_position=MIN_POSITION,
        max_position=MAX_POSITION,
        setup_validity_bars=SETUP_VALIDITY_BARS,
        signal_timeframe_minutes=240,  # 4h
    )
    
    # Set signal and ATR data
    signal_data = {s: data[s]['signal'] for s in data}
    atr_data = {s: data[s]['atr'] for s in data}
    signal_gen.set_signal_data(signal_data)
    signal_gen.set_atr_data(atr_data)
    
    # Initialize backtester
    backtester = BBSqueezeBacktester(
        initial_capital=INITIAL_CAPITAL,
        commission=0.0005,
        slippage_pct=0.0002,
        max_positions=MAX_POSITIONS,
        max_daily_loss_pct=MAX_DAILY_LOSS,
        max_hold_days=MAX_HOLD_DAYS,
        long_only=LONG_ONLY,
        verbose=True,
        leverage=True,
        maintenance_margin_pct=MAINTENANCE_MARGIN,
        use_intrabar_timeframes=True,
    )
    
    backtester.analyzer = analyzer
    backtester.signal_generator = signal_gen
    
    # Prepare data for backtest - NOW MATCHES run_backtest.py BEHAVIOR
    # Uses 1-min data for BOTH entry and exit checking
    # This allows entries at any time when signal conditions are met
    jan_start = pd.Timestamp('2026-01-01', tz='UTC')
    jan_end = pd.Timestamp('2026-02-01', tz='UTC')
    
    trade_data = {}
    for symbol in data:
        # Use full 1-min resolution for January (same as run_backtest.py)
        df_1min = data[symbol]['trade']
        df_jan_1min = df_1min[(df_1min.index >= jan_start) & (df_1min.index < jan_end)]
        trade_data[symbol] = df_jan_1min
        print(f"{symbol}: {len(trade_data[symbol]):,} 1-min bars in January")
    
    print()
    
    # Run backtest with 1-min data (matching run_backtest.py)
    results = backtester.run(trade_data)
    
    # Print detailed trade list
    print("\n" + "="*70)
    print("TRADE LIST")
    print("="*70)
    
    if not results.trades:
        print("No trades in January 2026")
        return results
    
    print(f"\n{'#':<3} {'Symbol':<10} {'Dir':<6} {'Entry Time':<18} {'Exit Time':<18} {'Entry$':>10} {'Exit$':>10} {'P&L':>10} {'Exit Reason'}")
    print("-"*120)
    
    for i, t in enumerate(results.trades, 1):
        entry_time = t.entry_time.strftime('%Y-%m-%d %H:%M') if t.entry_time else 'N/A'
        exit_time = t.exit_time.strftime('%Y-%m-%d %H:%M') if t.exit_time else 'OPEN'
        
        print(f"{i:<3} {t.symbol:<10} {t.direction.upper():<6} {entry_time:<18} {exit_time:<18} "
              f"${t.entry_price:>9,.2f} ${t.exit_price:>9,.2f} ${t.pnl:>+9,.2f} {t.exit_reason or ''}")
    
    print("-"*120)
    
    # Summary by symbol
    print("\n" + "="*70)
    print("SUMMARY BY SYMBOL")
    print("="*70)
    
    by_symbol = {}
    for t in results.trades:
        if t.symbol not in by_symbol:
            by_symbol[t.symbol] = {'trades': 0, 'wins': 0, 'pnl': 0}
        by_symbol[t.symbol]['trades'] += 1
        by_symbol[t.symbol]['pnl'] += t.pnl
        if t.pnl > 0:
            by_symbol[t.symbol]['wins'] += 1
    
    print(f"\n{'Symbol':<12} {'Trades':>8} {'Wins':>8} {'Win%':>8} {'P&L':>12}")
    print("-"*50)
    for symbol, stats in sorted(by_symbol.items()):
        wr = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
        print(f"{symbol:<12} {stats['trades']:>8} {stats['wins']:>8} {wr:>7.1f}% ${stats['pnl']:>+11,.2f}")
    print("-"*50)
    print(f"{'TOTAL':<12} {len(results.trades):>8} {sum(s['wins'] for s in by_symbol.values()):>8} "
          f"{results.statistics['win_rate']:>7.1f}% ${results.statistics['total_pnl']:>+11,.2f}")
    
    # Overall stats
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    stats = results.statistics
    print(f"Total Trades:     {stats['total_trades']}")
    print(f"Win Rate:         {stats['win_rate']:.1f}%")
    print(f"Profit Factor:    {stats['profit_factor']:.2f}")
    print(f"Total P&L:        ${stats['total_pnl']:+,.2f}")
    print(f"Return:           {stats['return_pct']:+.1f}%")
    print(f"Max Drawdown:     {stats['max_drawdown']:.1f}%")
    print(f"Sharpe Ratio:     {stats['sharpe']:.2f}")
    print(f"Avg Leverage:     {stats['avg_leverage']:.1f}x")
    print(f"Liquidations:     {stats['liquidations']}")
    print(f"Final Equity:     ${stats['final_equity']:,.2f}")
    print("="*70)
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Fetch data
    data = fetch_january_data()
    
    if not data:
        print("ERROR: No data fetched")
        return
    
    # Filter to January
    data = filter_january(data)
    
    # Run backtest
    results = run_january_backtest(data)
    
    print("\nDone!")


if __name__ == "__main__":
    main()