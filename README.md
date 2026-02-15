# QuantLab Multi-Market

High-performance quantitative trading platform supporting:
- CN_STOCK (A股)
- US_STOCK (美股)
- CRYPTO_SPOT (加密货币现货)
- CRYPTO_PERP (加密货币永续合约)

## Performance Features

- **Polars + Arrow + DuckDB** - Fast data processing and querying
- **VectorBT + Numba** - Accelerated backtesting
- **DiskCache** - Intelligent feature caching
- **Parallel execution** - Joblib (default) / Ray (optional)
- **UTC storage** - Always store in UTC, enforce shift(1) for signals

## Quick Start

```bash
# Install
pip install -e .

# Update data
python -m quantlab.pipeline.run data:update

# Run backtest
python -m quantlab.pipeline.run backtest:run --config configs/examples/us_momentum.json

# Run optimization
python -m quantlab.pipeline.run optimize:run --config configs/examples/us_momentum.json

# Build report
python -m quantlab.pipeline.run report:build --run-id <run_id>
```

## Architecture

- `data/` - Multi-market data sources (yfinance, akshare, ccxt)
- `features/` - Vectorized feature computation with caching
- `execution/` - Market-specific execution constraints
- `backtest/` - Fast (VectorBT) and accurate (Backtrader) engines
- `optimize/` - Coarse-to-fine optimization with Optuna
- `robustness/` - Walk-forward, sensitivity, bootstrap, regime split
- `report/` - Lightweight HTML reports

## License

MIT
