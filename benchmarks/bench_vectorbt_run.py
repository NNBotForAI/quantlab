"""
VectorBT backtest performance benchmarks
"""
import time
import numpy as np
import polars as pl
from datetime import datetime

from src.quantlab.features.signals import momentum_signal
from src.quantlab.backtest.vectorbt_engine import VectorBTBacktestEngine


def generate_synthetic_backtest_data(
    n_rows: int = 2520,
    n_symbols: int = 100,
) -> tuple[pl.DataFrame, np.ndarray]:
    """
    Generate synthetic OHLCV and signals for backtesting.

    Args:
        n_rows: Number of rows per symbol (approx 10 years of daily data)
        n_symbols: Number of symbols

    Returns:
        Tuple of (OHLCV dataframe, signals array)
    """
    np.random.seed(42)

    data = []
    signals = []

    for symbol_id in range(n_symbols):
        # Generate random walk price series
        returns = np.random.normal(0, 0.01, n_rows)
        prices = 100 * np.exp(np.cumsum(returns))

        # Generate momentum signals
        momentum_raw = np.zeros(n_rows)
        for i in range(20, n_rows):
            momentum_raw[i] = prices[i] / prices[i - 20] - 1

        # Shift to prevent lookahead
        momentum = np.roll(momentum_raw, 1)
        momentum[:1] = 0

        # Generate signals based on momentum
        symbol_signals = np.zeros(n_rows)
        symbol_signals[momentum > 0.02] = 1  # Buy signal
        symbol_signals[momentum < -0.02] = -1  # Sell signal

        signals.extend(symbol_signals)

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

    df = pl.DataFrame(data)
    signals_array = np.array(signals).reshape(n_rows, n_symbols)

    return df, signals_array


def benchmark_vectorbt_run(
    n_rows: int = 2520,
    n_symbols: int = 100,
    chunk_size: int = 200,
):
    """
    Benchmark VectorBT backtest execution.

    Args:
        n_rows: Number of rows per symbol
        n_symbols: Number of symbols
        chunk_size: Symbols per chunk for memory control
    """
    print(f"\n=== Benchmarking VectorBT Backtest ===")
    print(f"Configuration: n_rows={n_rows:,}, n_symbols={n_symbols:,}, chunk_size={chunk_size}")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")

    # Generate data
    df, signals = generate_synthetic_backtest_data(n_rows, n_symbols)

    # Flatten signals for backtest
    flat_signals = signals.T.flatten()

    # Create spec
    spec = {
        "instrument": {
            "asset_type": "US_STOCK",
            "symbol": "test",
            "venue": "test",
            "quote_currency": "USD",
            "lot_size": 1,
            "allow_fractional": True,
            "shortable": True,
            "leverage": 1,
        },
        "data": {
            "frequency": "1D",
            "price_mode": "raw",
        },
        "backtest": {
            "initial_capital": 100000,
            "commission": 0.0001,
            "slippage": 0.0005,
        },
        "performance": {
            "parallel_backend": "joblib",
            "max_workers": 4,
            "cache_enabled": False,  # Disable cache for benchmarking
            "vectorbt_chunking": chunk_size,
        },
    }

    # Create engine
    engine = VectorBTBacktestEngine(spec)

    # Benchmark without chunking
    print(f"\n--- Without Chunking ---")
    start = time.perf_counter()
    results = engine.run_backtest(df, flat_signals, initial_capital=100000)
    elapsed_no_chunk = time.perf_counter() - start

    print(f"Rows processed: {df.height:,}")
    print(f"Time elapsed: {elapsed_no_chunk:.4f} seconds")
    print(f"Rows/second: {df.height / elapsed_no_chunk:,.0f}")
    print(f"Memory: ~{df.estimated_size('mb'):.2f} MB")
    print(f"Sharpe ratio: {results['metrics'].get('sharpe_ratio', 0):.4f}")

    # Benchmark with chunking
    print(f"\n--- With Chunking (chunk_size={chunk_size}) ---")
    start = time.perf_counter()
    results_chunked = engine.run_backtest_chunked(
        df, flat_signals, initial_capital=100000, chunk_size=chunk_size
    )
    elapsed_chunk = time.perf_counter() - start

    print(f"Rows processed: {df.height:,}")
    print(f"Time elapsed: {elapsed_chunk:.4f} seconds")
    print(f"Rows/second: {df.height / elapsed_chunk:,.0f}")
    print(f"Memory: (controlled by chunking)")
    print(f"Sharpe ratio: {results_chunked['metrics'].get('sharpe_ratio', 0):.4f}")

    # Comparison
    print(f"\n--- Comparison ---")
    print(f"Speedup from chunking: {elapsed_no_chunk / elapsed_chunk:.2f}x")
    print(f"Overhead: {((elapsed_chunk / elapsed_no_chunk - 1) * 100):.1f}%")


def benchmark_signal_generation(n_rows: int = 2520, n_symbols: int = 100):
    """Benchmark signal generation."""
    print(f"\n=== Benchmarking Signal Generation ===")
    print(f"Configuration: n_rows={n_rows:,}, n_symbols={n_symbols:,}")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")

    # Generate data
    df, _ = generate_synthetic_backtest_data(n_rows, n_symbols)

    # Benchmark signal generation
    start = time.perf_counter()
    signals_df = momentum_signal(
        df,
        period=20,
        long_threshold=0.02,
        short_threshold=-0.02,
    )
    elapsed = time.perf_counter() - start

    print(f"Rows processed: {df.height:,}")
    print(f"Time elapsed: {elapsed:.4f} seconds")
    print(f"Rows/second: {df.height / elapsed:,.0f}")
    print(f"Signals generated: {signals_df.filter(pl.col('signal') != 0).height:,}")


def benchmark_metrics_calculation():
    """Benchmark metrics calculation."""
    print(f"\n=== Benchmarking Metrics Calculation ===")

    # Generate synthetic equity curve
    np.random.seed(42)
    n_days = 2520  # ~10 years
    returns = np.random.normal(0.0001, 0.01, n_days)
    equity = 100000 * np.exp(np.cumsum(returns))

    equity_df = pl.DataFrame({"equity": equity})

    # Generate synthetic trades
    n_trades = 1000
    trade_pnl = np.random.normal(100, 500, n_trades)
    trades_df = pl.DataFrame({"pnl": trade_pnl})

    # Benchmark metrics calculation
    from src.quantlab.backtest.metrics import calculate_all_metrics

    start = time.perf_counter()
    metrics = calculate_all_metrics(equity_df, trades_df, initial_capital=100000)
    elapsed = time.perf_counter() - start

    print(f"Days processed: {n_days:,}")
    print(f"Trades processed: {n_trades:,}")
    print(f"Time elapsed: {elapsed:.4f} seconds")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Calmar ratio: {metrics['calmar_ratio']:.4f}")


def run_all_benchmarks(
    n_rows: int = 2520,
    n_symbols: int = 100,
    chunk_size: int = 200,
):
    """
    Run all backtest benchmarks.

    Args:
        n_rows: Number of rows per symbol
        n_symbols: Number of symbols
        chunk_size: Chunk size for VectorBT
    """
    print("\n" + "="*60)
    print("Backtest Benchmarks")
    print("="*60)
    print(f"Configuration: n_rows={n_rows:,}, n_symbols={n_symbols:,}, chunk_size={chunk_size}")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print("="*60)

    benchmark_signal_generation(n_rows, n_symbols)
    benchmark_vectorbt_run(n_rows, n_symbols, chunk_size)
    benchmark_metrics_calculation()

    print("\n" + "="*60)
    print("Benchmarks Complete")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run backtest benchmarks")
    parser.add_argument("--n-rows", type=int, default=2520, help="Number of rows per symbol")
    parser.add_argument("--n-symbols", type=int, default=100, help="Number of symbols")
    parser.add_argument("--chunk-size", type=int, default=200, help="Chunk size for VectorBT")

    args = parser.parse_args()

    run_all_benchmarks(args.n_rows, args.n_symbols, args.chunk_size)
