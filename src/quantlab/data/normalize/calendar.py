"""
Trading calendar utilities
"""
import polars as pl
from datetime import datetime, timedelta
from typing import Literal

from ...common.logging import get_logger
from ...common.timeutils import to_utc

logger = get_logger(__name__)

CalendarType = Literal["CN_A", "US", "CRYPTO_24_7"]


def get_trading_sessions(
    start: datetime,
    end: datetime,
    calendar: CalendarType,
) -> pl.DataFrame:
    """
    Get trading session timestamps.

    Args:
        start: Start datetime (UTC)
        end: End datetime (UTC)
        calendar: Calendar type

    Returns:
        DataFrame with ts_utc column
    """
    if calendar == "CRYPTO_24_7":
        # Crypto markets are 24/7
        return _generate_continuous_ts(start, end)
    elif calendar == "US":
        return _generate_market_hours_ts(start, end, "US")
    elif calendar == "CN_A":
        return _generate_market_hours_ts(start, end, "CN")
    else:
        raise ValueError(f"Unknown calendar: {calendar}")


def _generate_continuous_ts(start: datetime, end: datetime) -> pl.DataFrame:
    """Generate continuous timestamps for 24/7 markets."""
    dates = pl.date_range(
        start=start,
        end=end,
        interval="1h",
        time_zone="UTC",
    )
    return pl.DataFrame({"ts_utc": dates})


def _generate_market_hours_ts(
    start: datetime,
    end: datetime,
    market: Literal["US", "CN"],
) -> pl.DataFrame:
    """
    Generate market hours timestamps.

    Args:
        start: Start datetime (UTC)
        end: End datetime (UTC)
        market: Market type

    Returns:
        DataFrame with ts_utc column
    """
    # This is a simplified version - in production, use pandas_market_calendars
    # or similar to get accurate trading days and hours

    if market == "US":
        # US market: 9:30 AM - 4:00 PM ET, Monday-Friday
        # ET is UTC-4 (DST) or UTC-5 (standard)
        # Simplified: assume UTC-4 for now
        tz_offset = timedelta(hours=4)

        # Generate hourly timestamps during market hours
        # This is a placeholder - proper implementation would use trading calendar
        dates = pl.date_range(
            start=start,
            end=end,
            interval="1h",
            time_zone="UTC",
        )

        # Filter to trading hours (simplified)
        # In production, use actual trading calendar
        df = pl.DataFrame({"ts_utc": dates})
        df = df.filter(
            (pl.col("ts_utc").dt.hour() >= 14) &  # 9:30 AM ET
            (pl.col("ts_utc").dt.hour() < 21) &   # 4:00 PM ET
            (pl.col("ts_utc").dt.weekday() < 5)   # Monday-Friday
        )

        return df

    elif market == "CN":
        # CN market: 9:30 AM - 11:30 AM, 1:00 PM - 3:00 PM CST (UTC+8)
        # CST is UTC+8

        # Generate hourly timestamps during market hours
        dates = pl.date_range(
            start=start,
            end=end,
            interval="1h",
            time_zone="UTC",
        )

        # Filter to trading hours (simplified)
        df = pl.DataFrame({"ts_utc": dates})
        df = df.filter(
            # Morning session: 1:30 AM - 3:30 AM UTC
            ((pl.col("ts_utc").dt.hour() >= 1) & (pl.col("ts_utc").dt.hour() < 4)) |
            # Afternoon session: 5:00 AM - 7:00 AM UTC
            ((pl.col("ts_utc").dt.hour() >= 5) & (pl.col("ts_utc").dt.hour() < 7))
        )
        df = df.filter(pl.col("ts_utc").dt.weekday() < 5)  # Monday-Friday

        return df

    else:
        raise ValueError(f"Unknown market: {market}")


def is_trading_day(dt: datetime, calendar: CalendarType) -> bool:
    """
    Check if a date is a trading day.

    Args:
        dt: Datetime to check
        calendar: Calendar type

    Returns:
        True if trading day
    """
    if calendar == "CRYPTO_24_7":
        return True
    elif calendar in ["US", "CN"]:
        # Check if weekday (simplified)
        return dt.weekday() < 5
    else:
        raise ValueError(f"Unknown calendar: {calendar}")
