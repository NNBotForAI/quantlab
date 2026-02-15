"""
DuckDB query performance benchmarks
"""
import time
import numpy as np
import polars as pl
import duckdb
from datetime import datetime
from pathlib import Path


def generate_synthetic_data_file(
    path: Path,
    n_rows: int = 100000,
    n_symbols: int = 100,
) -> None:
    """
    Generate synthetic data and save as Parquet.

    Args:
        path: Path to save Parquet file
        n_rows: Number of rows per symbol
        n_symbols: Number of symbols
    """
    path.parent.mkdir(parents=True, exist_ok=True)

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
                "market": "US_STOCK",
                "venue": "test",
                "freq": "1D",
            })

    df = pl.DataFrame(data)
    df.write_parquet(path, use_pyarrow=True)


def benchmark_polars_full_load(parquet_path: Path):
    """Benchmark full Polars load vs DuckDB scan."""
    print(f"\n=== Polars Full Load vs DuckDB Scan ===")

    # Benchmark Polars full load
    print("\n--- Polars Full Load ---")
    start = time.perf_counter()
    df_polars = pl.read_parquet(parquet_path)
    elapsed_polars = time.perf_counter() - start

    print(f"Rows loaded: {df_polars.height:,}")
    print(f"Time elapsed: {elapsed_polars:.4f} seconds")
    print(f"Memory: {df_polars.estimated_size('mb'):.2f} MB")

    # Benchmark DuckDB scan (no full load)
    print("\n--- DuckDB Scan ---")
    start = time.perf_counter()
    df_duckdb = duckdb.query(
        "SELECT * FROM read_parquet(?) LIMIT 100000",
        [str(parquet_path)]
    ).pl()
    elapsed_duckdb = time.perf_counter() - start

    print(f"Rows loaded: {df_duckdb.height:,}")
    print(f"Time elapsed: {elapsed_duckdb:.4f} seconds")
    print(f"Memory: (streaming, minimal)")

    # Comparison
    print(f"\n--- Comparison ---")
    print(f"Speedup (DuckDB scan): {elapsed_polars / elapsed_duckdb:.2f}x")


def benchmark_duckdb_queries(parquet_path: Path):
    """Benchmark various DuckDB queries."""
    print(f"\n=== DuckDB Query Benchmarks ===")

    queries = [
        {
            "name": "Simple count",
            "sql": "SELECT COUNT(*) as count FROM read_parquet(?)",
        },
        {
            "name": "Filter by symbol",
            "sql": "SELECT symbol, COUNT(*) as count FROM read_parquet(?) WHERE symbol = 'SYM000' GROUP BY symbol",
        },
        {
            "name": "Aggregate by symbol",
            "sql": "SELECT symbol, AVG(close) as avg_close, COUNT(*) as count FROM read_parquet(?) GROUP BY symbol",
        },
        {
            "name": "Time series aggregation",
            "sql": "SELECT ts_utc, AVG(close) as avg_close FROM read_parquet(?) GROUP BY ts_utc ORDER BY ts_utc LIMIT 1000",
        },
        {
            "name": "Complex filter and join",
            "sql": """
                SELECT t1.symbol, t1.close
                FROM read_parquet(?) t1
                JOIN (SELECT symbol, AVG(close) as avg_close FROM read_parquet(?) GROUP BY symbol) t2
                ON t1.symbol = t2.symbol
                WHERE t1.close > t2.avg_close
                LIMIT 10000
            """,
        },
    ]

    for query_def in queries:
        print(f"\n--- {query_def['name']} ---")
        start = time.perf_counter()

        try:
            result = duckdb.query(query_def['sql'], [str(parquet_path), str(parquet_path)]).pl()
            elapsed = time.perf_counter() - start

            if not result.is_empty():
                print(f"Rows returned: {result.height:,}")
            print(f"Time elapsed: {elapsed:.4f} seconds")
        except Exception as e:
            print(f"Error: {str(e)}")


def benchmark_duckdb_window_functions(parquet_path: Path):
    """Benchmark DuckDB window functions."""
    print(f"\n=== DuckDB Window Function Benchmarks ===")

    queries = [
        {
            "name": "Moving average",
            "sql": """
                SELECT symbol, ts_utc, 
                       AVG(close) OVER (PARTITION BY symbol ORDER BY ts_utc ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as ma20
                FROM read_parquet(?)
                LIMIT 100000
            """,
        },
        {
            "name": "Percent change",
            "sql": """
                SELECT symbol, ts_utc,
                       (close - LAG(close, 1) OVER (PARTITION BY symbol ORDER BY ts_utc)) / LAG(close, 1) OVER (PARTITION BY symbol ORDER BY ts_utc) as pct_change
                FROM read_parquet(?)
                LIMIT 100000
            """,
        },
        {
            "name": "Rank by performance",
            "sql": """
                SELECT symbol, 
                       RANK() OVER (ORDER BY close DESC) as rank,
                       close
                FROM (SELECT symbol, close FROM read_parquet(?) LIMIT 1000)
            """,
        },
    ]

    for query_def in queries:
        print(f"\n--- {query_def['name']} ---")
        start = time.perf_counter()

        try:
            result = duckdb.query(query_def['sql'], [str(parquet_path)]).pl()
            elapsed = time.perf_counter() - start

            print(f"Rows returned: {result.height:,}")
            print(f"Time elapsed: {elapsed:.4f} seconds")
        except Exception as e:
            print(f"Error: {str(e)}")


def run_all_benchmarks(
    n_rows: int = 100000,
    n_symbols: int = 100,
):
    """
    Run all DuckDB benchmarks.

    Args:
        n_rows: Number of rows per symbol
        n_symbols: Number of symbols
    """
    parquet_path = Path("test_data.benchmark.parquet")

    # Generate test data
    print("\n" + "="*60)
    print("DuckDB Benchmarks")
    print("="*60)
    print(f"Configuration: n_rows={n_rows:,}, n_symbols={n_symbols:,}, total_rows={n_rows*n_symbols:,}")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print("="*60)

    print("\nGenerating test data...")
    generate_synthetic_data_file(parquet_path, n_rows, n_symbols)

    # Run benchmarks
    benchmark_polars_full_load(parquet_path)
    benchmark_duckdb_queries(parquet_path)
    benchmark_duckdb_window_functions(parquet_path)

    # Cleanup
    print(f"\nCleaning up test data...")
    parquet_path.unlink()

    print("\n" + "="*60)
    print("Benchmarks Complete")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DuckDB query benchmarks")
    parser.add_argument("--n-rows", type=int, default=100000, help="Number of rows per symbol")
    parser.add_argument("--n-symbols", type=int, default=100, help="Number of symbols")

    args = parser.parse_args()

    run_all_benchmarks(args.n_rows, args.n_symbols)
