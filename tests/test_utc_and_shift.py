"""
UTC and shift enforcement tests
"""
import numpy as np
import polars as pl
import pytest
from datetime import datetime, timezone

from src.quantlab.common.timeutils import to_utc, from_utc, UTC, SHANGHAI
from src.quantlab.features.signals import shift_enforce


def test_utc_conversion():
    """Test UTC datetime conversion."""
    # Test naive datetime conversion
    naive_dt = datetime(2024, 1, 1, 12, 0)
    utc_dt = to_utc(naive_dt)
    
    assert utc_dt.tzinfo is not None, "Converted datetime should be timezone-aware"
    assert utc_dt.tzinfo == UTC, "Should be in UTC timezone"

    # Test UTC datetime passthrough
    utc_input = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    utc_output = to_utc(utc_input)
    
    assert utc_output == utc_input, "UTC datetime should pass through unchanged"


def test_from_utc():
    """Test conversion from UTC to target timezone."""
    utc_dt = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    
    # Convert to Shanghai
    shanghai_dt = from_utc(utc_dt, SHANGHAI)
    assert shanghai_dt.tzinfo == SHANGHAI
    
    # Shanghai is UTC+8, so 12:00 UTC should be 20:00 CST
    assert shanghai_dt.hour == 20, f"Expected 20:00, got {shanghai_dt.hour}:00"


def test_shift_enforce():
    """Test that shift_enforce prevents lookahead bias."""
    df = pl.DataFrame({
        "ts_utc": [1, 2, 3, 4, 5],
        "close": [100, 101, 102, 103, 104],
        "signal_raw": [0.02, 0.03, 0.01, -0.01, -0.02],
    })

    # Apply shift
    df_shifted = shift_enforce(df, columns=["signal_raw"], periods=1)

    # First value should be null (shifted)
    assert df_shifted[0, "signal_raw"] is None, "First shifted value should be null"

    # Second value should be first original value
    assert df_shifted[1, "signal_raw"] == 0.02, "Shifted value should match original"


def test_shift_prevents_lookahead():
    """Test that shifting prevents using future information."""
    df = pl.DataFrame({
        "ts_utc": list(range(1, 11)),
        "close": [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
    })

    # Generate momentum without shift (lookahead)
    df_with_lookahead = df.with_columns(
        (pl.col("close") / pl.col("close").shift(2) - 1).alias("momentum_raw")
    )

    # Generate momentum with shift (no lookahead)
    df_no_lookahead = df.with_columns(
        (pl.col("close") / pl.col("close").shift(2) - 1).alias("momentum_raw")
    )
    df_no_lookahead = shift_enforce(df_no_lookahead, columns=["momentum_raw"], periods=2)

    # Check that values are different
    assert not df_with_lookahead["momentum_raw"].equals(
        df_no_lookahead["momentum_raw"]
    ), "Shifted and unshifted values should differ"


def test_multi_column_shift():
    """Test shifting multiple columns simultaneously."""
    df = pl.DataFrame({
        "ts_utc": list(range(1, 11)),
        "col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "col2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "col3": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    })

    df_shifted = shift_enforce(df, columns=["col1", "col2", "col3"], periods=1)

    # All columns should be shifted
    assert df_shifted[0, "col1"] is None
    assert df_shifted[0, "col2"] is None
    assert df_shifted[0, "col3"] is None

    # Check second row
    assert df_shifted[1, "col1"] == 1
    assert df_shifted[1, "col2"] == 10
    assert df_shifted[1, "col3"] == 5
