"""
Signal Generator for Crypto Trading Bot.

Entry Logic:
1. Detect consolidation pattern
2. Wait for breakout confirmation
3. Confirm with volume
4. Enter in direction of breakout
5. RSI filter to avoid extremes

Exit Logic:
- ATR-based stops
- Take profit targets
- Time-based exit
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import pytz

from technical import BBSqueezeAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Trade signal with entry/exit levels."""
    timestamp: datetime
    symbol: str
    direction: str  # 'long', 'short', 'neutral'
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    score: float
    reasons: List[str] = field(default_factory=list)
    atr: float = 0.0


@dataclass
class PendingSetup:
    """Setup waiting for entry."""
    timestamp: datetime
    symbol: str
    direction: str
    signal_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    score: float
    reasons: List[str]
    atr: float


class BBSqueezeSignalGenerator:
    """
    Generate trading signals based on volatility breakout patterns.
    """
    
    def __init__(
        self,
        analyzer: BBSqueezeAnalyzer,
        # Squeeze params
        min_squeeze_bars: int = 3,
        # Volume
        min_volume_ratio: float = 1.2,
        # RSI filter
        rsi_overbought: float = 75,
        rsi_oversold: float = 25,
        # Stops (ATR multiples)
        atr_stop_mult: float = 2.0,
        atr_target_mult: float = 3.0,
        # Position sizing
        base_position: float = 0.10,
        min_position: float = 0.05,
        max_position: float = 0.30,
        signal_timeframe_minutes: int = 60,
    ):
        self.analyzer = analyzer
        self.min_squeeze_bars = min_squeeze_bars
        self.min_volume_ratio = min_volume_ratio
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.base_position = base_position
        self.min_position = min_position
        self.max_position = max_position
        self.signal_timeframe_minutes = signal_timeframe_minutes
        
        # State
        self.consecutive_losses = 0
        
        # Data containers - populated externally each cycle via direct assignment
        # e.g., generator.signal_data[symbol] = df
        self.signal_data: Dict[str, pd.DataFrame] = {}
        self.atr_data: Dict[str, pd.DataFrame] = {}
    
    def set_signal_data(self, data: Dict[str, pd.DataFrame]):
        """Set signal timeframe data."""
        self.signal_data = data
    
    def set_atr_data(self, data: Dict[str, pd.DataFrame]):
        """Set ATR timeframe data."""
        self.atr_data = data
    
    def _normalize_time(self, dt: datetime, reference_index=None) -> datetime:
        """Normalize datetime to match reference timezone."""
        if dt is None:
            return None
        
        # Get reference timezone
        ref_tz = None
        if reference_index is not None and hasattr(reference_index, 'tz') and reference_index.tz is not None:
            ref_tz = reference_index.tz
        
        # If dt is naive and we have a reference tz, localize it
        if dt.tzinfo is None:
            if ref_tz is not None:
                return ref_tz.localize(dt) if hasattr(ref_tz, 'localize') else dt.replace(tzinfo=ref_tz)
            return dt.replace(tzinfo=pytz.UTC)
        
        # If dt has tz and reference has tz, convert
        if ref_tz is not None:
            return dt.astimezone(ref_tz)
        
        return dt
    
    def generate_signal(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        current_time: datetime,
        current_price: float = None
    ) -> TradeSignal:
        """
        Generate trading signal - enters immediately on breakout detection.
        
        Args:
            df: DataFrame with OHLCV data (used for setup detection)
            symbol: Trading symbol
            current_time: Current timestamp
            current_price: Optional real-time price override. If provided, this price
                          is used for entry instead of df.iloc[-1]['close']. This allows
                          live trading to use real-time prices while backtest uses bar close.
        """
        
        # Check for breakout setup and enter immediately
        if symbol in self.signal_data:
            setup = self._check_for_setup(symbol, current_time)
            if setup is not None:
                signal = self._check_entry(df, setup, current_time, current_price)
                if signal.direction != 'neutral':
                    return signal
        
        return self._neutral(symbol, current_time)
    
    def _check_for_setup(self, symbol: str, current_time: datetime) -> Optional[PendingSetup]:
        """Check signal timeframe for breakout setup."""
        if symbol not in self.signal_data:
            return None
        
        sig_df = self.signal_data[symbol]
        
        # Normalize time for filtering
        normalized_time = self._normalize_time(current_time, sig_df.index)
        
        # Filter data up to current time
        try:
            available = sig_df[sig_df.index <= normalized_time]
        except TypeError:
            # Fallback if timezone comparison fails
            available = sig_df
        
        if len(available) < 50:
            return None
        
        # Calculate indicators
        df = self.analyzer.calculate_indicators(available.tail(200))
        
        # Detect breakout
        breakout = self.analyzer.detect_breakout(
            df,
            min_squeeze_bars=self.min_squeeze_bars,
            min_volume_ratio=self.min_volume_ratio
        )
        
        if breakout is None:
            return None
        
        direction = breakout['direction']
        
        # RSI filter (only place where RSI filtering happens)
        rsi = breakout['rsi']
        if pd.isna(rsi):
            return None
        if direction == 'long' and rsi > self.rsi_overbought:
            return None
        if direction == 'short' and rsi < self.rsi_oversold:
            return None
        
        # Get ATR for stops
        atr = self._get_atr(symbol, current_time)
        if atr is None or atr <= 0:
            atr = breakout['atr']
        
        # Calculate stops
        price = breakout['price']
        if direction == 'long':
            stop_loss = price - (atr * self.atr_stop_mult)
            take_profit = price + (atr * self.atr_target_mult)
        else:
            stop_loss = price + (atr * self.atr_stop_mult)
            take_profit = price - (atr * self.atr_target_mult)
        
        # Position sizing based on signal quality
        size = self._calculate_position_size(breakout)
        
        # Score
        score = self._calculate_score(breakout)
        
        # Reasons
        reasons = [
            f"SQ{int(breakout['squeeze_bars'])}",
            f"V{breakout['volume_ratio']:.1f}",
            f"M{breakout['momentum']:.1f}",
            f"RSI{rsi:.0f}"
        ]
        
        return PendingSetup(
            timestamp=current_time,
            symbol=symbol,
            direction=direction,
            signal_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=size,
            score=score,
            reasons=reasons,
            atr=atr
        )
    
    def _check_entry(
        self, 
        df: pd.DataFrame, 
        setup: PendingSetup, 
        current_time: datetime,
        current_price: float = None
    ) -> TradeSignal:
        """
        Check if entry conditions are met.
        
        Args:
            df: DataFrame with OHLCV data
            setup: Pending setup to check
            current_time: Current timestamp
            current_price: Optional real-time price override. If provided, this price
                          is used for entry instead of df.iloc[-1]['close'].
        """
        if len(df) < 2:
            return self._neutral(setup.symbol, current_time)
        
        # Use provided current_price if available, otherwise use df close
        price = current_price if current_price is not None else df.iloc[-1]['close']
        
        # Check if price has already hit stop (invalidates setup)
        if setup.direction == 'long':
            if price <= setup.stop_loss:
                return self._neutral(setup.symbol, current_time)
        else:
            if price >= setup.stop_loss:
                return self._neutral(setup.symbol, current_time)
        
        # Enter immediately if setup is valid (no pullback filter)
        # Recalculate stops from actual entry price
        atr = setup.atr
        if setup.direction == 'long':
            sl = price - (atr * self.atr_stop_mult)
            tp = price + (atr * self.atr_target_mult)
        else:
            sl = price + (atr * self.atr_stop_mult)
            tp = price - (atr * self.atr_target_mult)
        
        return TradeSignal(
            timestamp=current_time,
            symbol=setup.symbol,
            direction=setup.direction,
            entry_price=price,
            stop_loss=sl,
            take_profit=tp,
            position_size=setup.position_size,
            score=setup.score,
            reasons=setup.reasons,
            atr=atr
        )
    
    def _get_atr(self, symbol: str, current_time: datetime) -> Optional[float]:
        """Get ATR from ATR timeframe data."""
        if symbol not in self.atr_data:
            return None
        
        atr_df = self.atr_data[symbol]
        
        # Normalize time for filtering
        normalized_time = self._normalize_time(current_time, atr_df.index)
        
        try:
            available = atr_df[atr_df.index <= normalized_time]
        except TypeError:
            available = atr_df
        
        if len(available) < 20:
            return None
        
        df = self.analyzer.calculate_indicators(available.tail(100))
        if 'ATR' not in df.columns:
            return None
        
        return float(df['ATR'].iloc[-1])
    
    def _signal_quality(self, breakout: Dict) -> float:
        """
        Compute signal quality score in [0, 1] using exponential diminishing returns.

        Each factor is normalized so that a signal just barely passing the minimum
        filter threshold scores near 0, and scores grow with diminishing returns above it.

        Weights:
          - Squeeze bars (40%): longer consolidation = stronger coiled energy
          - Volume ratio (35%): volume confirms conviction on breakout
          - Momentum (25%): normalized ATR momentum shows breakout force

        Tau (decay) constants control how quickly each factor saturates:
          - squeeze: tau = min_squeeze_bars  → at threshold score ~63% of factor max
          - volume:  tau = min_volume_ratio  → at threshold score ~63% of factor max
          - momentum: tau = 0.5             → fixed, momentum has no per-asset min
        """
        import math

        # Excess above the per-asset minimum threshold — below threshold = 0
        squeeze_excess  = max(0.0, breakout['squeeze_bars'] - self.min_squeeze_bars)
        volume_excess   = max(0.0, breakout['volume_ratio'] - self.min_volume_ratio)
        momentum_excess = max(0.0, abs(breakout['momentum']))

        # Tau = minimum threshold so threshold itself maps to 1 - e^-1 ≈ 63%
        squeeze_tau  = max(self.min_squeeze_bars, 1)
        volume_tau   = max(self.min_volume_ratio, 0.1)
        momentum_tau = 0.5

        squeeze_factor  = 1 - math.exp(-squeeze_excess  / squeeze_tau)
        volume_factor   = 1 - math.exp(-volume_excess   / volume_tau)
        momentum_factor = 1 - math.exp(-momentum_excess / momentum_tau)

        quality = 0.40 * squeeze_factor + 0.35 * volume_factor + 0.25 * momentum_factor
        return max(0.0, min(quality, 1.0))

    def _calculate_position_size(self, breakout: Dict) -> float:
        """
        Position size scales continuously with signal quality.
        
        size = base + (max - base) * quality
        
        Weakest possible signal → base_position
        Strongest possible signal → max_position
        Consecutive losses apply a smooth reduction (not a hard cut).
        """
        quality = self._signal_quality(breakout)
        size = self.base_position + (self.max_position - self.base_position) * quality
        
        # Smooth consecutive loss reduction: each loss reduces by 15%, compounding
        if self.consecutive_losses > 0:
            loss_factor = 0.85 ** self.consecutive_losses
            size *= loss_factor
        
        return max(self.min_position, min(size, self.max_position))
    
    def _calculate_score(self, breakout: Dict) -> float:
        """Signal quality score [0, 1] — same as sizing quality for consistency."""
        return self._signal_quality(breakout)
    
    def _neutral(self, symbol: str, ts: datetime) -> TradeSignal:
        """Return neutral signal."""
        return TradeSignal(
            timestamp=ts,
            symbol=symbol,
            direction='neutral',
            entry_price=0,
            stop_loss=0,
            take_profit=0,
            position_size=0,
            score=0
        )
    
    def record_trade_result(self, pnl: float):
        """Record trade result for position sizing."""
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0