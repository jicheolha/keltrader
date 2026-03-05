"""
Shared utilities for Crypto Trading Bot.
"""
from datetime import datetime
from typing import Optional
import math
import pytz


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def colored(text: str, color: str) -> str:
    """Apply color to text."""
    return f"{color}{text}{Colors.RESET}"


def fmt_price(price: float, sigfigs: int = 5) -> str:
    """Format a price to a given number of significant figures with comma separators.

    Examples (5 sig figs):
        95000  -> "95,000"
        3000.1 -> "3,000.1"
        150.00 -> "150.00"
        2.5001 -> "2.5001"
        0.1500 -> "0.15000"
    """
    if price is None or (isinstance(price, float) and math.isnan(price)):
        return "N/A"
    if price == 0:
        return "0.00000"
    magnitude = math.floor(math.log10(abs(price)))
    decimal_places = max(0, (sigfigs - 1) - magnitude)
    return f"{price:,.{decimal_places}f}"


# Timeframe mappings (single source of truth)
TIMEFRAME_MINUTES = {
    '1min': 1, '1m': 1,
    '5min': 5, '5m': 5,
    '15min': 15, '15m': 15,
    '30min': 30, '30m': 30,
    '1h': 60,
    '2h': 120,
    '4h': 240,
    '6h': 360,
    '1d': 1440,
}


def get_tf_minutes(tf: str) -> int:
    """Get minutes for a timeframe string."""
    return TIMEFRAME_MINUTES.get(tf, 60)


def ensure_tz_aware(dt: datetime, reference_tz=None) -> datetime:
    """Ensure datetime is timezone-aware."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        if reference_tz is None:
            return dt.replace(tzinfo=pytz.UTC)
        return reference_tz.localize(dt)
    return dt


def infer_timeframe_from_index(index) -> str:
    """
    Infer timeframe from a pandas DatetimeIndex.
    
    Returns timeframe string like '1m', '1h', '1d'.
    """
    if len(index) < 2:
        return '1m'
    
    # Calculate median difference between timestamps
    diffs = index.to_series().diff().dropna()
    if len(diffs) == 0:
        return '1m'
    
    median_diff = diffs.median()
    minutes = median_diff.total_seconds() / 60
    
    # Map to nearest standard timeframe (shorter format)
    if minutes <= 1.5:
        return '1m'
    elif minutes <= 7:
        return '5m'
    elif minutes <= 20:
        return '15m'
    elif minutes <= 45:
        return '30m'
    elif minutes <= 90:
        return '1h'
    elif minutes <= 180:
        return '2h'
    elif minutes <= 300:
        return '4h'
    elif minutes <= 480:
        return '6h'
    else:
        return '1d'