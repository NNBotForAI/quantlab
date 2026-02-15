"""
UTC conversions and timezone utilities
"""
import pytz
from datetime import datetime, timezone
from typing import Optional, Union

import polars as pl


# Common timezones
UTC = pytz.UTC
SHANGHAI = pytz.timezone("Asia/Shanghai")
NEW_YORK = pytz.timezone("America/New_York")


def to_utc(dt: Union[datetime, str, int, float], tz: Optional[pytz.BaseTzInfo] = None) -> datetime:
    """
    Convert datetime to UTC.

    Args:
        dt: Input datetime (datetime, string, or timestamp)
        tz: Source timezone (if dt is naive)

    Returns:
        UTC datetime
    """
    if isinstance(dt, (int, float)):
        return datetime.fromtimestamp(dt, UTC)

    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))

    if dt.tzinfo is None:
        if tz:
            dt = tz.localize(dt)
        else:
            dt = UTC.localize(dt)

    return dt.astimezone(UTC)


def from_utc(dt: Union[datetime, int, float], tz: pytz.BaseTzInfo) -> datetime:
    """
    Convert UTC datetime to target timezone.

    Args:
        dt: UTC datetime or timestamp
        tz: Target timezone

    Returns:
        Localized datetime
    """
    if isinstance(dt, (int, float)):
        dt = datetime.fromtimestamp(dt, UTC)

    if dt.tzinfo is None:
        dt = UTC.localize(dt)

    return dt.astimezone(tz)


def now_utc() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(UTC)


def tz_name_to_tz(tz_name: str) -> pytz.BaseTzInfo:
    """
    Get timezone object from name.

    Args:
        tz_name: Timezone name (e.g., "Asia/Shanghai")

    Returns:
        Timezone object
    """
    return pytz.timezone(tz_name)


def add_utc_timestamp(df: pl.DataFrame, col_name: str = "ts_utc") -> pl.DataFrame:
    """
    Add UTC timestamp column to dataframe.

    Args:
        df: Input dataframe
        col_name: Column name for timestamp

    Returns:
        Dataframe with UTC timestamp column
    """
    if col_name not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Datetime(time_zone="UTC")).alias(col_name))
    return df


def ensure_utc(df: pl.DataFrame, col_name: str = "ts_utc") -> pl.DataFrame:
    """
    Ensure datetime column is in UTC timezone.

    Args:
        df: Input dataframe
        col_name: Datetime column name

    Returns:
        Dataframe with UTC timestamp column
    """
    if col_name not in df.columns:
        raise ValueError(f"Column {col_name} not found in dataframe")

    dtype = df.schema[col_name]
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone != "UTC":
            return df.with_columns(
                pl.col(col_name).dt.convert_time_zone("UTC").alias(col_name)
            )
    return df


def floor_to_freq(dt: datetime, freq: str) -> datetime:
    """
    Floor datetime to frequency.

    Args:
        dt: Input datetime
        freq: Frequency ("1D", "1H", "5m", etc.)

    Returns:
        Floored datetime
    """
    freq_map = {
        "1m": "minute",
        "5m": "5min",
        "15m": "15min",
        "1H": "hour",
        "1D": "day",
    }

    period = freq_map.get(freq, freq)
    if isinstance(dt, datetime):
        return dt.replace(microsecond=0)
    return dt
