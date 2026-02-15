"""
Feature computation performance benchmarks
"""
import time
import numpy as np
import polars as pl
from datetime import datetime

from src.quantlab.features.indicators import (
    momentum,
    volatility,
    rsi,
    sma,
    ema,
    bollinger_bands,
)


def generate_synthetic_data(n_rows: int = 100000, n_symbols: int = 100) -> pl.DataFrame:
    """
    Generate synthetic OHLCV data for benchmarking.

    Args:
        n_rows: Number of rows per symbol
        n_symbols: Number of symbols

    Returns:
        Synthetic OHLCV dataframe
    """
    np.random.seed(42)

    data = []
    for symbol_id in range(n_symbols):
        # Generate random walk price series
        returns = np.random.normal(0, 0.01, n_rows)
        prices = 100 * np.exp(np.cumsum(returns))

        # Generate OHLC from close prices
        for i in range(n_rows):
            close = prices[i]
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_ = close if i == 0 else prices[i - 1]
            volume = abs(np.random.normal(1000000, 200000))

            data.append({
                "ts_utc": i,
                "symbol": f"SYM{symbol_id:03d}",
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            })

    return pl.DataFrame(data)


def benchmark_momentum(n_rows: int = 100000, n_symbols: int = 100):
    """Benchmark momentum indicator computation."""
    print(f"\n=== Benchmarking Momentum (n_rows={n_rows}, n_symbols={n_symbols}) ===")

    df = generate_synthetic_data(n_rows, n_symbols)
    period = 20

    start = time.perf_counter()
    result = momentum(df, period=period)
    elapsed = time.perf_counter() - start

    print(f"Rows processed: {df.height:,}")
    print(f"Time elapsed: {elapsed:.4f} seconds")
    print(f"Rows/second: {df.height / elapsed:,.0f}")
    print(f"Memory usage: {df.estimated_size('mb'):.2f} MB")


def benchmark_volatility(n_rows: int = 100000, n_symbols: int = 100):
    """Benchmark volatility indicator computation."""
    print(f"\n=== Benchmarking Volatility (n_rows={n_rows}, n_symbols={n_symbols}) ===")

    df = generate_synthetic_data(n_rows, n_symbols)
    period = 20

    start = time.perf_counter()
    result = volatility(df, period=period)
    elapsed = time.perf_counter() - start

    print(f"Rows processed: {df.height:,}")
    print(f"Time elapsed: {elapsed:.4f} seconds")
    print(f"Rows/second: {df.height / elapsed:,.0f}")


def benchmark_rsi(n_rows: int = 100000, n_symbols: int = 100):
    """Benchmark RSI indicator computation."""
    print(f"\n=== Benchmarking RSI (n_rows={n_rows}, n_symbols={n_symbols}) ===")

    df = generate_synthetic_data(n_rows, n_symbols)
    period = 14

    start = time.perf_counter()
    result = rsi(df, period=period, use_numba=True)
    elapsed = time.perf_counter() - start

    print(f"Rows processed: {df.height:,}")
    print(f"Time elapsed: {elapsed:.4f} seconds")
    print(f"Rows/second: {df.height / elapsed:,.0f}")


def benchmark_sma(n_rows: int = 100000, n_symbols: int = 100):
    """Benchmark SMA indicator computation."""
    print(f"\n=== Benchmarking SMA (n_rows={n_rows}, n_symbols={n_symbols}) ===")

    df = generate_synthetic_data(n_rows, n_symbols)
    period = 20

    start = time.perf_counter()
    result = sma(df, period=period)
    elapsed = time.perf_counter() - start

    print(f"Rows processed: {df.height:,}")
    print(f"Time elapsed: {elapsed:.4f} seconds")
    print(f"Rows/second: {df.height / elapsed:,.0f}")


def benchmark_ema(n_rows: int = 100000, n_symbols: int = 100):
    """Benchmark EMA indicator computation."""
    print(f"\n=== Benchmarking EMA (n_rows={n_rows}, n_symbols={n_symbols}) ===")

    df = generate_synthetic_data(n_rows, n_symbols)
    period = 20

    start = time.perf_counter()
    result = ema(df, period=period)
    elapsed = time.perf_counter() - start

    print(f"Rows processed: {df.height:,}")
    print(f"Time elapsed: {elapsed:.4f} seconds")
    print(f"Rows/second: {df.height / elapsed:,.0f}")


def benchmark_bollinger_bands(n_rows: int = 100000, n_symbols: int = 100):
    """Benchmark Bollinger Bands indicator computation."""
    print(f"\n=== Benchmarking Bollinger Bands (n_rows={n_rows}, n_symbols={n_symbols}) ===")

    df = generate_synthetic_data(n_rows, n_symbols)
    period = 20
    std_dev = 2.0

    start = time.perf_counter()
    result = bollinger_bands(df, period=period, std_dev=std_dev)
    elapsed = time.perf_counter() - start

    print(f"Rows processed: {df.height:,}")
    print(f"Time elapsed: {elapsed:.4f} seconds")
    print(f"Rows/second: {df.height / elapsed:,.0f}")


def run_all_benchmarks(n_rows: int = 100000, n_symbols: int = 100):
    """
    Run all feature computation benchmarks.

    Args:
        n_rows: Number of rows per symbol
        n_symbols: Number of symbols
    """
    print("\n" + "="*60)
    print("Feature Computation Benchmarks")
    print("="*60)
    print(f"Configuration: n_rows={n_rows:,}, n_symbols={n_symbols:,}, total_rows={n_rows*n_symbols:,}")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print("="*60)

    benchmark_momentum(n_rows, n_symbols)
    benchmark_volatility(n_rows, n_symbols)
    benchmark_rsi(n_rows, n_symbols)
    benchmark_sma(n_rows, n_symbols)
    benchmark_ema(n_rows, n_symbols)
    benchmark_bollinger_bands(n_rows, n_symbols)

    print("\n" + "="*60)
    print("Benchmarks Complete")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run feature computation benchmarks")
    parser.add_argument("--n-rows", type=int, default=100000, help="Number of rows per symbol")
    parser.add_argument("--n-symbols", type=int, default=100, help="Number of symbols")

    args = parser.parse_args()

    run_all_benchmarks(args.n_rows, args.n_symbols)
