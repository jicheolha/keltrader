# Keltrader

A Bollinger Band Squeeze momentum trading system for crypto. Deployed 24/7 on a Digital Ocean VPS. Trades perpetual futures on Coinbase with leverage set by the exchange. Entry signals are derived from the native (spot) price series of each asset.

Backtest 2021-01-01 to 2025-12-28 (BTC/USD + XRP/USD, leveraged): +2,286% return | 89% CAGR | 2.07 Sharpe | 25.3% max drawdown | 117 trades | 73.5% win rate.

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

| File | Description |
|---|---|
| `technical.py` | Computes Bollinger Bands, Keltner Channels, squeeze state, momentum, RSI, and volume ratio from OHLCV data |
| `signal_generator.py` | Converts indicator state into directional trade signals with entry price, stop loss, take profit, and position size |
| `backtester.py` | Event-driven backtester supporting spot and leveraged futures. Tracks capital, margin, unrealized P&L, and computes equity curve, drawdown, Sharpe, and trade statistics |
| `optimize_lib.py` | Walk-forward data splitting, shared scoring logic, and optimizer base classes used by the optimization pipeline |
| `data_utils.py` | Fetches and caches multi-timeframe OHLCV data from Alpaca. Serves trade, signal, and ATR timeframes independently |
| `download_data.py` | Downloads 1-minute bars from Alpaca and resamples to all required timeframes. Skips already-cached symbols |
| `utils.py` | Terminal color formatting and shared price formatting utilities |
| `run_backtest.py` | Backtest runner. Accepts symbol list, leverage flag, and date range. Outputs per-asset and combined equity curves. **Redacted** |
| `run_live_multi_asset.py` | Live trading loop. Polls signals on each bar close, manages positions across multiple assets simultaneously. **Redacted** |
| `coinbase_live_trader.py` | Live trading engine. Handles order placement, position sync from Coinbase, P&L tracking, drawdown monitoring, drift detection, and Telegram alerts. **Redacted** |
| `optimizer.py` | Single-asset Optuna optimizer with walk-forward cross-validation. Supports TPE, random, and CMA-ES samplers. Trials persist to SQLite so runs can be stopped and resumed. **Redacted** |
| `permutation_test.py` | Monte Carlo permutation test for statistical significance. **Redacted** |
| `check_trade.py` | Queries open and closed positions from the Coinbase API and prints a summary. **Redacted** |

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

Optimization trials persist to a SQLite database, so runs can be stopped and resumed without losing progress. Each new run builds on prior trial history, allowing TPE to improve its surrogate model over multiple sessions.

### Statistical Validation

After optimization, parameters are validated using a Monte Carlo permutation test. The strategy is run on real data, then re-run on N shuffled versions of the same data where temporal structure is destroyed but the return distribution is preserved. The p-value is the fraction of shuffled runs that match or exceed the real result.

Two shuffle methods are available: bar-level return shuffling (destroys all autocorrelation, strictest test) and block shuffling (shuffles weekly chunks, preserves short-term intrabar structure while destroying longer patterns like squeezes). A low p-value indicates the parameters are capturing a real market pattern rather than noise.

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
