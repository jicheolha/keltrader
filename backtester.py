"""
Fast Backtester - Signal Timeframe Only

Loops on signal timeframe (e.g., 4h) instead of 1-minute bars.
Entry and exit checked on same timeframe for speed.

Key simplifications:
- No 1-min bar looping (1440x faster)
- No setup_validity_bars - enter immediately on signal or not at all
- No incremental resampling - uses pre-built candles
- Stops/targets checked using bar high/low
"""
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from technical import BBSqueezeAnalyzer
from signal_generator import BBSqueezeSignalGenerator, TradeSignal
from utils import fmt_price


logger = logging.getLogger(__name__)


# =============================================================================
# LEVERAGE CONFIGURATION - CONSERVATIVE OVERNIGHT RATES ONLY
# =============================================================================

LEVERAGE_RATES_LONG = {
    'BTC': 0.246,   # 4.1x leverage
    'ETH': 0.249,   # 4.0x leverage
    'SOL': 0.366,   # 2.7x leverage
    'XRP': 0.389,   # 2.6x leverage
    'DOGE': 0.499,  # 2.0x leverage
}

LEVERAGE_RATES_SHORT = {
    'BTC': 0.302,   # 3.3x leverage
    'ETH': 0.347,   # 2.9x leverage
    'SOL': 0.549,   # 1.8x leverage
    'XRP': 0.615,   # 1.6x leverage
    'DOGE': 0.998,  # 1.0x leverage (nearly no leverage for shorts)
}

DEFAULT_MARGIN_RATE = 0.25  # 4x leverage default


@dataclass
class Position:
    """Open position."""
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    quantity: float
    capital_used: float
    stop_loss: float
    take_profit: float
    atr: float
    reasons: List[str]
    entry_equity: float = 0.0
    position_size_pct: float = 0.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    r_multiple: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    is_open: bool = True
    # Leverage fields
    leverage: float = 1.0
    margin_used: float = 0.0
    notional_value: float = 0.0
    liquidation_price: Optional[float] = None


@dataclass
class BacktestResults:
    """Backtest results container."""
    trades: List[Position]
    equity_curve: pd.Series
    statistics: Dict


class BBSqueezeBacktester:
    """
    Fast backtester that loops on signal timeframe only.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.001,
        slippage_pct: float = 0.0005,
        max_positions: int = 3,
        max_hold_days: float = None,
        verbose: bool = False,
        leverage: bool = False,
        maintenance_margin_pct: float = 0.5,
        signal_generator: BBSqueezeSignalGenerator = None,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.slippage_pct = slippage_pct
        self.max_positions = max_positions
        self.max_hold_days = max_hold_days
        self.verbose = verbose
        self.leverage_enabled = leverage
        self.maintenance_margin_pct = maintenance_margin_pct
        self.signal_generator = signal_generator
        
        # State
        self.positions: Dict[str, Position] = {}
        self.trades: List[Position] = []
        self.equity_history: List[tuple] = []
        self.liquidation_count = 0
    
    def _extract_base_currency(self, symbol: str) -> str:
        """Extract base currency from symbol."""
        if '/' in symbol:
            return symbol.split('/')[0]
        elif '-' in symbol:
            return symbol.split('-')[0]
        return symbol
    
    def _get_margin_rate(self, symbol: str, direction: str = 'long') -> float:
        """Get margin rate for symbol."""
        base = self._extract_base_currency(symbol)
        if direction == 'long':
            return LEVERAGE_RATES_LONG.get(base, DEFAULT_MARGIN_RATE)
        else:
            return LEVERAGE_RATES_SHORT.get(base, DEFAULT_MARGIN_RATE)
    
    def _calculate_liquidation_price(
        self, 
        entry_price: float, 
        direction: str, 
        margin_rate: float,
        maintenance_pct: float
    ) -> float:
        """Calculate liquidation price."""
        max_loss_pct = margin_rate * (1 - maintenance_pct)
        if direction == 'long':
            return entry_price * (1 - max_loss_pct)
        else:
            return entry_price * (1 + max_loss_pct)
    
    def run(
        self, 
        signal_data: Dict[str, pd.DataFrame],
        atr_data: Dict[str, pd.DataFrame] = None,
        exit_data: Dict[str, pd.DataFrame] = None
    ) -> BacktestResults:
        """
        Run backtest on signal timeframe data.
        
        Args:
            signal_data: {symbol: DataFrame} with signal timeframe OHLCV (e.g., 4h)
            atr_data: {symbol: DataFrame} optional ATR timeframe data
            exit_data: {symbol: DataFrame} optional 1-min data for precise exit timing
        """
        self._reset()
        
        if atr_data is None:
            atr_data = signal_data
        
        # Store exit data for precise stop/TP checking
        self.exit_data = exit_data
        
        # Set data on signal generator
        self.signal_generator.signal_data = signal_data
        self.signal_generator.atr_data = atr_data
        
        # Get all symbols
        symbols = list(signal_data.keys())
        
        # Get all unique timestamps across all symbols
        all_times = set()
        for df in signal_data.values():
            all_times.update(df.index.tolist())
        all_times = sorted(all_times)
        
        if self.verbose:
            print(f"\nBacktesting {len(all_times):,} bars across {len(symbols)} symbols")
            print(f"Period: {all_times[0].strftime('%Y-%m-%d')} to {all_times[-1].strftime('%Y-%m-%d')}")
            if self.leverage_enabled:
                print(f"Leverage: ENABLED")
            if exit_data:
                print(f"Exit precision: 1-min bars")
            print()
        
        # Pre-calculate indicators for all symbols
        indicator_cache = {}
        for symbol, df in signal_data.items():
            indicator_cache[symbol] = self.signal_generator.analyzer.calculate_indicators(df)
        
        # Pre-calculate ATR indicators from ATR timeframe data
        atr_indicator_cache = {}
        for symbol, df in atr_data.items():
            atr_indicator_cache[symbol] = self.signal_generator.analyzer.calculate_indicators(df)
        
        # Main loop - iterate through signal timeframe bars
        for ts in all_times:
            equity = self._calc_equity(signal_data, ts)
            
            # Process each symbol
            for symbol in symbols:
                df = signal_data[symbol]
                if ts not in df.index:
                    continue
                
                idx = df.index.get_loc(ts)
                if idx < 50:  # Need enough history for indicators
                    continue
                
                indicators = indicator_cache[symbol]
                if ts not in indicators.index:
                    continue
                
                ind_idx = indicators.index.get_loc(ts)
                current = indicators.iloc[ind_idx]
                bar = df.iloc[idx]
                high = float(bar['high'])
                low = float(bar['low'])
                close = float(bar['close'])
                
                # Get ATR from ATR timeframe (find most recent ATR bar <= current timestamp)
                atr_value = self._get_atr_at_time(symbol, ts, atr_indicator_cache)
                
                # Check exits using 1-min data for precision (if available)
                if symbol in self.positions:
                    if self.exit_data and symbol in self.exit_data:
                        self._check_exit_precise(symbol, ts, signal_data)
                    else:
                        self._check_exit(symbol, close, high, low, ts)
                
                # Check for new entries (only if no position and room for more)
                if symbol not in self.positions and len(self.positions) < self.max_positions:
                    # Need at least 2 bars for breakout detection
                    if ind_idx >= 1:
                        prev = indicators.iloc[ind_idx - 1]
                        self._check_entry(symbol, current, prev, close, ts, equity, atr_value)
            
            # Record equity
            equity = self._calc_equity(signal_data, ts)
            self.equity_history.append((ts, equity))
        
        # Close remaining positions at end
        for symbol in list(self.positions.keys()):
            if symbol in signal_data:
                final_price = float(signal_data[symbol].iloc[-1]['close'])
                self._close(symbol, final_price, all_times[-1], "End of backtest")
        
        return self._compile_results()
    
    def _reset(self):
        """Reset state for new backtest."""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_history = []
        self.liquidation_count = 0
        if self.signal_generator:
            self.signal_generator.consecutive_losses = 0
    
    def _get_atr_at_time(self, symbol: str, ts: datetime, atr_cache: Dict) -> Optional[float]:
        """Get ATR value from ATR timeframe at or before given timestamp."""
        if symbol not in atr_cache:
            return None
        
        atr_indicators = atr_cache[symbol]
        
        # Find most recent ATR bar <= current timestamp
        available = atr_indicators[atr_indicators.index <= ts]
        if len(available) < 1:
            return None
        
        atr_value = available['ATR'].iloc[-1]
        if pd.isna(atr_value) or atr_value <= 0:
            return None
        
        return float(atr_value)
    
    def _check_entry(
        self,
        symbol: str,
        current: pd.Series,
        prev: pd.Series,
        price: float,
        ts: datetime,
        equity: float,
        atr_from_atr_tf: float = None
    ):
        """Check for entry signal and enter if valid."""

        # Breakout detection: was in squeeze, now released
        squeeze_released = prev['Squeeze'] and not current['Squeeze']
        if not squeeze_released:
            return

        # Check squeeze duration (from prev candle - how long it was in squeeze)
        squeeze_duration = int(prev['Squeeze_Duration'])
        if squeeze_duration < self.signal_generator.min_squeeze_bars:
            return

        # Check volume on current candle (the breakout candle)
        volume_ratio = current['Volume_Ratio']
        if pd.isna(volume_ratio) or volume_ratio < self.signal_generator.min_volume_ratio:
            return

        # Determine direction
        if current['close'] > current['BB_Upper']:
            direction = 'long'
        elif current['close'] < current['BB_Lower']:
            direction = 'short'
        elif current['Momentum'] > 0:
            direction = 'long'
        elif current['Momentum'] < 0:
            direction = 'short'
        else:
            return

        # RSI filter
        rsi = current['RSI']
        if pd.isna(rsi):
            return
        if direction == 'long' and rsi > self.signal_generator.rsi_overbought:
            return
        if direction == 'short' and rsi < self.signal_generator.rsi_oversold:
            return

        # ATR for stops
        if atr_from_atr_tf is not None:
            atr = atr_from_atr_tf
        else:
            atr = current['ATR']
            if pd.isna(atr) or atr <= 0:
                return

        # Normalized momentum for sizing
        momentum = current['Momentum_Norm']
        if pd.isna(momentum):
            momentum = 0.0

        reasons = [
            f"SQ{squeeze_duration}",
            f"V{volume_ratio:.1f}",
            f"M{abs(momentum):.1f}",
            f"RSI{int(rsi)}"
        ]

        if direction == 'long':
            stop_loss   = price - (atr * self.signal_generator.atr_stop_mult)
            take_profit = price + (atr * self.signal_generator.atr_target_mult)
            entry_price = price * (1 + self.slippage_pct)
        else:
            stop_loss   = price + (atr * self.signal_generator.atr_stop_mult)
            take_profit = price - (atr * self.signal_generator.atr_target_mult)
            entry_price = price * (1 - self.slippage_pct)

        breakout_dict = {
            'squeeze_bars': squeeze_duration,
            'volume_ratio': volume_ratio,
            'momentum': momentum,
        }
        position_size     = self.signal_generator._calculate_position_size(breakout_dict)
        position_size_pct = position_size * 100
        position_value    = equity * position_size

        if self.leverage_enabled:
            margin_rate       = self._get_margin_rate(symbol, direction)
            leverage          = 1.0 / margin_rate
            margin_used       = position_value
            notional_value    = position_value * leverage
            quantity          = notional_value / entry_price
            liquidation_price = self._calculate_liquidation_price(
                entry_price, direction, margin_rate, self.maintenance_margin_pct
            )
            capital_used = margin_used
        else:
            leverage          = 1.0
            margin_used       = position_value
            notional_value    = position_value
            quantity          = position_value / entry_price
            liquidation_price = None
            capital_used      = position_value

        commission_cost = quantity * entry_price * self.commission
        if margin_used + commission_cost > self.capital:
            return

        risk_per_unit = abs(entry_price - stop_loss)
        risk_amount   = risk_per_unit * quantity
        risk_pct      = (risk_amount / equity) * 100 if equity > 0 else 0

        self.capital -= (margin_used + commission_cost)

        pos = Position(
            symbol=symbol,
            direction=direction,
            entry_time=ts,
            entry_price=entry_price,
            quantity=quantity,
            capital_used=capital_used,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr,
            reasons=reasons,
            entry_equity=equity,
            position_size_pct=position_size_pct,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            leverage=leverage,
            margin_used=margin_used,
            notional_value=notional_value,
            liquidation_price=liquidation_price
        )

        self.positions[symbol] = pos

        if self.verbose:
            lev_str = f" ({leverage:.1f}x)" if self.leverage_enabled else ""
            print(f"{ts.strftime('%Y-%m-%d %H:%M')}  ENTRY{lev_str}  {direction.upper():5}  "
                  f"${fmt_price(entry_price):>12}  qty={quantity:.4f}  ${margin_used:>10,.0f}")

    def _check_exit(self, symbol: str, price: float, high: float, low: float, ts: datetime):
        """Check exit conditions."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        
        # Check liquidation first (leverage only)
        if self.leverage_enabled and pos.liquidation_price is not None:
            if pos.direction == 'long' and low <= pos.liquidation_price:
                self.liquidation_count += 1
                self._close(symbol, pos.liquidation_price, ts, "LIQUIDATED", exact_price=True)
                return
            elif pos.direction == 'short' and high >= pos.liquidation_price:
                self.liquidation_count += 1
                self._close(symbol, pos.liquidation_price, ts, "LIQUIDATED", exact_price=True)
                return
        
        # Check stop loss
        if pos.direction == 'long' and low <= pos.stop_loss:
            self._close(symbol, pos.stop_loss, ts, "Stop loss hit", exact_price=True)
            return
        elif pos.direction == 'short' and high >= pos.stop_loss:
            self._close(symbol, pos.stop_loss, ts, "Stop loss hit", exact_price=True)
            return
        
        # Check take profit
        if pos.direction == 'long' and high >= pos.take_profit:
            self._close(symbol, pos.take_profit, ts, "Take profit hit", exact_price=True)
            return
        elif pos.direction == 'short' and low <= pos.take_profit:
            self._close(symbol, pos.take_profit, ts, "Take profit hit", exact_price=True)
            return
        
        # Check max hold time
        if self.max_hold_days is not None:
            hold_time = (ts - pos.entry_time).total_seconds() / 86400
            if hold_time >= self.max_hold_days:
                self._close(symbol, price, ts, "Max hold time")
                return
    
    def _check_exit_precise(self, symbol: str, signal_bar_ts: datetime, signal_data: Dict[str, pd.DataFrame]):
        """
        Check exit conditions using 1-min bars for precise timing.
        
        Loops through 1-min bars within the current signal timeframe bar
        to find exact exit time when stop/TP is hit.
        """
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        exit_df = self.exit_data[symbol]
        
        # Get the next signal bar timestamp to know the window
        signal_df = signal_data[symbol]
        try:
            sig_idx = signal_df.index.get_loc(signal_bar_ts)
            if sig_idx + 1 < len(signal_df):
                next_signal_ts = signal_df.index[sig_idx + 1]
            else:
                next_signal_ts = signal_bar_ts + timedelta(hours=4)  # Assume 4h
        except (KeyError, IndexError):
            next_signal_ts = signal_bar_ts + timedelta(hours=4)
        
        # Get 1-min bars within this window
        mask = (exit_df.index >= signal_bar_ts) & (exit_df.index < next_signal_ts)
        window_bars = exit_df[mask]
        
        if window_bars.empty:
            # Fallback to signal bar high/low
            bar = signal_df.loc[signal_bar_ts]
            self._check_exit(symbol, float(bar['close']), float(bar['high']), float(bar['low']), signal_bar_ts)
            return
        
        # Loop through 1-min bars to find exact exit
        for bar_ts, bar in window_bars.iterrows():
            high = float(bar['high'])
            low = float(bar['low'])
            close = float(bar['close'])
            
            # Check liquidation first (leverage only)
            if self.leverage_enabled and pos.liquidation_price is not None:
                if pos.direction == 'long' and low <= pos.liquidation_price:
                    self.liquidation_count += 1
                    self._close(symbol, pos.liquidation_price, bar_ts, "LIQUIDATED", exact_price=True)
                    return
                elif pos.direction == 'short' and high >= pos.liquidation_price:
                    self.liquidation_count += 1
                    self._close(symbol, pos.liquidation_price, bar_ts, "LIQUIDATED", exact_price=True)
                    return
            
            # Check stop loss
            if pos.direction == 'long' and low <= pos.stop_loss:
                self._close(symbol, pos.stop_loss, bar_ts, "Stop loss hit", exact_price=True)
                return
            elif pos.direction == 'short' and high >= pos.stop_loss:
                self._close(symbol, pos.stop_loss, bar_ts, "Stop loss hit", exact_price=True)
                return
            
            # Check take profit
            if pos.direction == 'long' and high >= pos.take_profit:
                self._close(symbol, pos.take_profit, bar_ts, "Take profit hit", exact_price=True)
                return
            elif pos.direction == 'short' and low <= pos.take_profit:
                self._close(symbol, pos.take_profit, bar_ts, "Take profit hit", exact_price=True)
                return
            
            # Check max hold time
            if self.max_hold_days is not None:
                hold_time = (bar_ts - pos.entry_time).total_seconds() / 86400
                if hold_time >= self.max_hold_days:
                    self._close(symbol, close, bar_ts, "Max hold time")
                    return

    def _close(self, symbol: str, price: float, ts: datetime, reason: str, exact_price: bool = False):
        """Close position."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        pos.exit_time = ts
        
        # Apply slippage only for market orders
        if exact_price or reason == "LIQUIDATED":
            exit_price = price
        elif pos.direction == 'long':
            exit_price = price * (1 - self.slippage_pct)
        else:
            exit_price = price * (1 + self.slippage_pct)
        
        pos.exit_price = exit_price
        pos.exit_reason = reason
        pos.is_open = False
        
        # Calculate P&L
        if pos.direction == 'long':
            gross = (exit_price - pos.entry_price) * pos.quantity
        else:
            gross = (pos.entry_price - exit_price) * pos.quantity
        
        commission_cost = pos.quantity * exit_price * self.commission
        net = gross - commission_cost
        pos.pnl = net
        pos.pnl_pct = (net / pos.margin_used) * 100 if pos.margin_used > 0 else 0
        
        # R-multiple
        if pos.risk_amount > 0:
            pos.r_multiple = net / pos.risk_amount
        
        # Update capital
        self.capital += pos.margin_used + net
        
        # Record result
        self.signal_generator.record_trade_result(net)
        
        self.trades.append(pos)
        del self.positions[symbol]
        
        if self.verbose:
            lev_str = f" ({pos.leverage:.1f}x)" if self.leverage_enabled else ""
            print(f"{ts.strftime('%Y-%m-%d %H:%M')}  EXIT{lev_str}   ${fmt_price(exit_price):>12}  "
                  f"${net:>+9,.2f}  {reason}")
    
    def _calc_equity(self, data: Dict[str, pd.DataFrame], ts: datetime) -> float:
        """Calculate current equity."""
        equity = self.capital
        
        for symbol, pos in self.positions.items():
            if symbol not in data:
                equity += pos.margin_used
                continue
            
            df = data[symbol]
            
            if ts in df.index:
                price = float(df.loc[ts, 'close'])
            else:
                prior = df.index[df.index <= ts]
                if len(prior) > 0:
                    price = float(df.loc[prior[-1], 'close'])
                else:
                    price = pos.entry_price
            
            if pos.direction == 'long':
                unrealized = (price - pos.entry_price) * pos.quantity
            else:
                unrealized = (pos.entry_price - price) * pos.quantity
            
            equity += pos.margin_used + unrealized
        
        return equity
    
    def _compile_results(self) -> BacktestResults:
        """Compile backtest results."""
        if not self.trades:
            return BacktestResults(
                trades=[],
                equity_curve=pd.Series([self.initial_capital]),
                statistics=self._empty_stats()
            )
        
        eq_df = pd.DataFrame(self.equity_history, columns=['time', 'equity'])
        eq_df.set_index('time', inplace=True)
        
        stats = self._calculate_stats(eq_df['equity'])
        
        if self.verbose:
            self._print_stats(stats)
        
        return BacktestResults(
            trades=self.trades,
            equity_curve=eq_df['equity'],
            statistics=stats
        )
    
    def _calculate_stats(self, equity: pd.Series) -> Dict:
        """Calculate performance statistics."""
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        
        # Drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        max_dd = abs(drawdown.min())
        
        # Sharpe
        try:
            daily_equity = equity.resample('D').last().dropna()
            daily_returns = daily_equity.pct_change().dropna()
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365)
            else:
                sharpe = 0
        except Exception:
            sharpe = 0
        
        # R-multiples
        r_multiples = [t.r_multiple for t in self.trades]
        avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0
        avg_r_win = sum(t.r_multiple for t in wins) / len(wins) if wins else 0
        avg_r_loss = sum(t.r_multiple for t in losses) / len(losses) if losses else 0
        
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        avg_win = gross_profit / len(wins) if wins else 0
        avg_loss = gross_loss / len(losses) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        expectancy = total_pnl / len(self.trades) if self.trades else 0
        
        # Kelly
        if avg_loss > 0 and avg_win > 0:
            win_loss_ratio = avg_win / avg_loss
            kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        else:
            kelly = 0
        kelly_pct = max(0, min(kelly * 100, 100))
        
        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_consec = 0
        last_was_win = None
        
        for t in self.trades:
            is_win = t.pnl > 0
            if is_win == last_was_win:
                current_consec += 1
            else:
                current_consec = 1
                last_was_win = is_win
            
            if is_win:
                max_consec_wins = max(max_consec_wins, current_consec)
            else:
                max_consec_losses = max(max_consec_losses, current_consec)
        
        # Hold time
        hold_times = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in self.trades if t.exit_time]
        avg_hold_hours = sum(hold_times) / len(hold_times) if hold_times else 0
        
        # Leverage stats
        if self.leverage_enabled:
            avg_leverage = sum(t.leverage for t in self.trades) / len(self.trades) if self.trades else 1.0
            liquidations = sum(1 for t in self.trades if t.exit_reason == "LIQUIDATED")
        else:
            avg_leverage = 1.0
            liquidations = 0
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate * 100,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'final_equity': float(equity.iloc[-1]),
            'return_pct': (float(equity.iloc[-1]) - self.initial_capital) / self.initial_capital * 100,
            'avg_r_multiple': avg_r,
            'avg_r_win': avg_r_win,
            'avg_r_loss': avg_r_loss,
            'expectancy': expectancy,
            'kelly_pct': kelly_pct,
            'max_consec_wins': max_consec_wins,
            'max_consec_losses': max_consec_losses,
            'avg_hold_hours': avg_hold_hours,
            # Leverage stats
            'leverage_enabled': self.leverage_enabled,
            'avg_leverage': avg_leverage,
            'liquidations': liquidations,
            # Backwards compatibility
            'initial_capital': self.initial_capital,
            'wins': len(wins),
            'losses': len(losses),
            'avg_r': avg_r,
        }
    
    def _empty_stats(self) -> Dict:
        """Return empty stats."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown': 0,
            'sharpe': 0,
            'final_equity': self.initial_capital,
            'return_pct': 0,
            'avg_r_multiple': 0,
            'avg_r_win': 0,
            'avg_r_loss': 0,
            'expectancy': 0,
            'kelly_pct': 0,
            'max_consec_wins': 0,
            'max_consec_losses': 0,
            'avg_hold_hours': 0,
            'leverage_enabled': self.leverage_enabled,
            'avg_leverage': 1.0,
            'liquidations': 0,
            # Backwards compatibility
            'initial_capital': self.initial_capital,
            'wins': 0,
            'losses': 0,
            'avg_r': 0,
        }
    
    def _print_stats(self, stats: Dict):
        """Print statistics."""
        print(f"\n{'='*50}")
        print("BACKTEST RESULTS")
        print(f"{'='*50}")
        print(f"Total P&L:      ${stats['total_pnl']:,.2f}")
        print(f"Return:         {stats['return_pct']:.2f}%")
        print(f"Trades:         {stats['total_trades']}")
        print(f"Win Rate:       {stats['win_rate']:.1f}%")
        print(f"Profit Factor:  {stats['profit_factor']:.2f}")
        print(f"Sharpe Ratio:   {stats['sharpe']:.2f}")
        print(f"Max Drawdown:   {stats['max_drawdown']:.1f}%")
        if self.leverage_enabled:
            print(f"Liquidations:   {stats['liquidations']}")
        print(f"{'='*50}\n")