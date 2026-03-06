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
    def __init__(
        self,
        analyzer: BBSqueezeAnalyzer,
        min_squeeze_bars: int = 3,
        min_volume_ratio: float = 1.2,
        rsi_overbought: float = 75,
        rsi_oversold: float = 25,
        atr_stop_mult: float = 2.0,
        atr_target_mult: float = 3.0,
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
        self.consecutive_losses = 0
        self.signal_data: Dict[str, pd.DataFrame] = {}
        self.atr_data: Dict[str, pd.DataFrame] = {}
    
    def set_signal_data(self, data: Dict[str, pd.DataFrame]):
        self.signal_data = data

    def set_atr_data(self, data: Dict[str, pd.DataFrame]):
        self.atr_data = data
    
    def _normalize_time(self, dt: datetime, reference_index=None) -> datetime:
        if dt is None:
            return None
        ref_tz = None
        if reference_index is not None and hasattr(reference_index, 'tz') and reference_index.tz is not None:
            ref_tz = reference_index.tz
        if dt.tzinfo is None:
            if ref_tz is not None:
                return ref_tz.localize(dt) if hasattr(ref_tz, 'localize') else dt.replace(tzinfo=ref_tz)
            return dt.replace(tzinfo=pytz.UTC)
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
        if symbol in self.signal_data:
            setup = self._check_for_setup(symbol, current_time)
            if setup is not None:
                signal = self._check_entry(df, setup, current_time, current_price)
                if signal.direction != 'neutral':
                    return signal
        
        return self._neutral(symbol, current_time)
    
    def _check_for_setup(self, symbol: str, current_time: datetime) -> Optional[PendingSetup]:
        if symbol not in self.signal_data:
            return None
        sig_df = self.signal_data[symbol]
        normalized_time = self._normalize_time(current_time, sig_df.index)
        try:
            available = sig_df[sig_df.index <= normalized_time]
        except TypeError:
            available = sig_df
        if len(available) < 50:
            return None
        df = self.analyzer.calculate_indicators(available.tail(200))
        breakout = self.analyzer.detect_breakout(
            df,
            min_squeeze_bars=self.min_squeeze_bars,
            min_volume_ratio=self.min_volume_ratio
        )
        if breakout is None:
            return None
        direction = breakout['direction']
        rsi = breakout['rsi']
        if pd.isna(rsi):
            return None
        if direction == 'long' and rsi > self.rsi_overbought:
            return None
        if direction == 'short' and rsi < self.rsi_oversold:
            return None
        atr = self._get_atr(symbol, current_time)
        if atr is None or atr <= 0:
            atr = breakout['atr']
        price = breakout['price']
        if direction == 'long':
            stop_loss = price - (atr * self.atr_stop_mult)
            take_profit = price + (atr * self.atr_target_mult)
        else:
            stop_loss = price + (atr * self.atr_stop_mult)
            take_profit = price - (atr * self.atr_target_mult)
        size = self._calculate_position_size(breakout)
        score = self._calculate_score(breakout)
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
        if len(df) < 2:
            return self._neutral(setup.symbol, current_time)
        price = current_price if current_price is not None else df.iloc[-1]['close']
        if setup.direction == 'long':
            if price <= setup.stop_loss:
                return self._neutral(setup.symbol, current_time)
        else:
            if price >= setup.stop_loss:
                return self._neutral(setup.symbol, current_time)
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
        if symbol not in self.atr_data:
            return None
        atr_df = self.atr_data[symbol]
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
        quality = self._signal_quality(breakout)
        size = self.base_position + (self.max_position - self.base_position) * quality
        if self.consecutive_losses > 0:
            loss_factor = 0.85 ** self.consecutive_losses
            size *= loss_factor
        
        return max(self.min_position, min(size, self.max_position))
    
    def _calculate_score(self, breakout: Dict) -> float:
        return self._signal_quality(breakout)

    def _neutral(self, symbol: str, ts: datetime) -> TradeSignal:
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
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0