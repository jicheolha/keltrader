# Keltrader

A Bollinger Band Squeeze momentum trading system for crypto. Deployed 24/7 on a Digital Ocean VPS. Trades perpetual futures on Coinbase with leverage set by the exchange. Entry signals are derived from the native (spot) price series of each asset.

Some files are redacted from this repository to protect the trading edge.

## Strategy

Signal logic is based on a BB Squeeze breakout:

1. Bollinger Bands contract inside Keltner Channels (squeeze = price compression)
2. Momentum accelerates as the squeeze releases
3. Volume confirms the breakout direction
4. RSI filters entries at overbought/oversold extremes
5. ATR-based stop loss and take profit levels are set at entry

Each asset runs independently optimized parameters covering BB period/std, KC period/ATR multiplier, RSI thresholds, minimum squeeze bar count, volume ratio, and ATR stop/target multipliers.

## Project Structure

```
Keltrader/
├── technical.py          # BB, Keltner Channel, squeeze detection, RSI, volume
├── signal_generator.py   # Entry/exit signal logic
├── backtester.py         # Event-driven backtester (spot and leveraged futures)
├── optimize_lib.py       # Walk-forward split logic and shared optimizer utilities
├── data_utils.py         # Multi-timeframe OHLCV data loading and caching
├── download_data.py      # Alpaca data downloader
├── debug_coinbase_pnl.py # Reconcile Coinbase P&L vs internal trade journal
├── diagnostics.py        # Strategy diagnostics
└── utils.py              # Shared utilities
```

> Execution, optimization, live trading, and parameter files are redacted.

## Backtesting

```bash
# Single asset
python run_backtest.py --symbols BTC/USD

# Multiple assets
python run_backtest.py --symbols BTC/USD,XRP/USD

# Leveraged futures mode
python run_backtest.py --symbols BTC/USD,XRP/USD --leverage

# Custom maintenance margin
python run_backtest.py --leverage --maintenance-margin 0.3
```

Backtests run per-asset with separate capital pools. When multiple assets are specified, equity curves are combined by summing PnL deltas across assets on a shared timeline.

## Optimization

Uses Optuna with walk-forward cross-validation. Scoring is designed to penalize overfitting:

- Bell curve penalties on win rate and profit factor
- tanh-saturated Sharpe (diminishing returns above 1.5)
- Neighborhood stability penalty: params perturbed +/-10%, narrow performance peaks penalized up to 30%
- Fold aggregation via `mean - 0.5 * std` to penalize high fold variance
- R:R hard cap at 3.5
- Minimum 10 trades per training fold, 6 per test fold

## Live Trading

Runs on Coinbase Advanced Trade API. Requires credentials as environment variables:

```bash
export COINBASE_API_KEY="your_key"
export COINBASE_API_SECRET="your_secret"
```

The live engine syncs position state from Coinbase on startup, tracks P&L and drawdown, monitors drift vs backtest expectations, and sends Telegram notifications on trade events.

## Setup

```bash
pip install -r requirements.txt
```

OHLCV data is sourced from [Alpaca Markets](https://alpaca.markets). Set credentials:

```bash
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"
```

## Backtest Results

### BTC/USD — Leveraged

![BTC/USD equity curve](equity_curve_BTCUSD_leverage.png)

### XRP/USD — Leveraged

![XRP/USD equity curve](equity_curve_XRPUSD_leverage.png)

### BTC/USD + XRP/USD — Leveraged (Combined)

![BTC+XRP equity curve](equity_curve_BTCUSD_XRPUSD_leverage.png)

## Disclaimer

For informational purposes only. Cryptocurrency trading carries substantial risk of loss. The authors are not responsible for any financial losses incurred through the use of this software.

## License

MIT — see [LICENSE](LICENSE)
