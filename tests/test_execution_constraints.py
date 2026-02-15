"""
Execution constraints tests
"""
import numpy as np
import polars as pl
import pytest

from src.quantlab.execution.cn_equity import CNEquityExecutionModel
from src.quantlab.execution.us_equity import USEquityExecutionModel
from src.quantlab.execution.crypto_spot import CryptoSpotExecutionModel
from src.quantlab.execution.crypto_perp import CryptoPerpExecutionModel


@pytest.fixture
def cn_stock_spec():
    """CN stock strategy specification."""
    return {
        "instrument": {
            "asset_type": "CN_STOCK",
            "symbol": "000001.SZ",
            "venue": "SZSE",
            "quote_currency": "CNY",
            "lot_size": 100,
            "allow_fractional": False,
            "shortable": False,
            "leverage": 1,
        },
        "backtest": {
            "initial_capital": 1000000,
            "commission": 0.0003,
            "slippage": 0.001,
        },
    }


@pytest.fixture
def us_stock_spec():
    """US stock strategy specification."""
    return {
        "instrument": {
            "asset_type": "US_STOCK",
            "symbol": "AAPL",
            "venue": "NASDAQ",
            "quote_currency": "USD",
            "lot_size": 1,
            "allow_fractional": True,
            "shortable": True,
            "leverage": 1,
        },
        "backtest": {
            "initial_capital": 100000,
            "commission": 0.0001,
            "slippage": 0.0005,
        },
    }


@pytest.fixture
def crypto_spec():
    """Crypto spot strategy specification."""
    return {
        "instrument": {
            "asset_type": "CRYPTO_SPOT",
            "symbol": "BTC/USDT",
            "venue": "binance",
            "quote_currency": "USDT",
            "lot_size": 0.0001,
            "allow_fractional": True,
            "shortable": True,
            "leverage": 1,
        },
        "backtest": {
            "initial_capital": 10000,
            "commission": 0.001,
            "slippage": 0.0005,
        },
    }


def test_cn_equity_lot_sizing(cn_stock_spec):
    """Test CN equity lot sizing."""
    model = CNEquityExecutionModel(cn_stock_spec)

    # Test lot sizing
    quantities = np.array([150, 250, 350, 450, 550])
    sized = model.apply_lot_sizing(quantities)

    # Should be rounded to nearest 100
    expected = np.array([200, 300, 400, 500, 600])
    np.testing.assert_array_equal(sized, expected)


def test_cn_equity_shortability(cn_stock_spec):
    """Test CN equity shortability constraint."""
    model = CNEquityExecutionModel(cn_stock_spec)

    # Shorting should be disabled
    signals = np.array([1, -1, 0, -1, 1])
    adjusted = model.apply_shortability(signals)

    # All negative signals should be zero
    expected = np.array([1, 0, 0, 0, 1])
    np.testing.assert_array_equal(adjusted, expected)


def test_us_equity_fractional_shares(us_stock_spec):
    """Test US equity fractional shares."""
    model = USEquityExecutionModel(us_stock_spec)

    # Test fractional shares
    quantities = np.array([1.5, 2.7, 3.3])
    sized = model.apply_lot_sizing(quantities)

    # Should preserve fractional shares
    expected = np.round(quantities, 4)  # 4 decimal places
    np.testing.assert_array_almost_equal(sized, expected)


def test_us_equity_shortability_allowed(us_stock_spec):
    """Test US equity shortability allowed."""
    model = USEquityExecutionModel(us_stock_spec)

    # Shorting should be allowed
    signals = np.array([1, -1, 0, -1, 1])
    adjusted = model.apply_shortability(signals)

    # Should pass through unchanged
    np.testing.assert_array_equal(adjusted, signals)


def test_crypto_spot_fractional_trading(crypto_spec):
    """Test crypto spot fractional trading."""
    model = CryptoSpotExecutionModel(crypto_spec)

    # Test very small fractional quantities
    quantities = np.array([0.00005, 0.00015, 0.00125])
    sized = model.apply_lot_sizing(quantities)

    # Should preserve fractional values
    # Crypto typically allows 8 decimal places
    expected = np.round(quantities, 8)
    np.testing.assert_array_almost_equal(sized, expected)


def test_fee_calculation(crypto_spec):
    """Test fee calculation."""
    model = CryptoSpotExecutionModel(crypto_spec)

    prices = np.array([100, 200, 300])
    quantities = np.array([1, 2, 3])

    # Calculate buy fees (0.1% = 0.001)
    buy_fees = model.calculate_fees(prices, quantities, "buy")

    # Notional: [100, 400, 900]
    # Fees: [0.1, 0.4, 0.9]
    expected_fees = np.array([0.1, 0.4, 0.9])
    np.testing.assert_array_almost_equal(buy_fees, expected_fees)

    # Calculate sell fees (should be same)
    sell_fees = model.calculate_fees(prices, -quantities, "sell")
    np.testing.assert_array_almost_equal(sell_fees, expected_fees)


def test_cn_stamp_duty(cn_stock_spec):
    """Test CN equity stamp duty on sell."""
    model = CNEquityExecutionModel(cn_stock_spec)

    prices = np.array([10, 20, 30])
    quantities = np.array([100, 200, 300])

    # Buy fees: 0.03% commission only
    buy_fees = model.calculate_fees(prices, quantities, "buy")
    # Notional: [1000, 4000, 9000]
    # Fees: [0.3, 1.2, 2.7]
    expected_buy_fees = np.array([0.3, 1.2, 2.7])
    np.testing.assert_array_almost_equal(buy_fees, expected_buy_fees)

    # Sell fees: 0.03% commission + 0.1% stamp duty
    sell_fees = model.calculate_fees(prices, -quantities, "sell")
    # Notional: [1000, 4000, 9000]
    # Commission: [0.3, 1.2, 2.7]
    # Stamp duty: [1.0, 4.0, 9.0]
    # Total: [1.3, 5.2, 11.7]
    expected_sell_fees = np.array([1.3, 5.2, 11.7])
    np.testing.assert_array_almost_equal(sell_fees, expected_sell_fees)
