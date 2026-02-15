"""
Schema validation tests
"""
import json
import pytest
from pathlib import Path

from src.quantlab.common.hashing import hash_spec


def test_strategy_spec_schema():
    """Test strategy spec JSON schema validation."""
    schema_path = Path("configs/templates/strategy_spec.schema.json")

    assert schema_path.exists(), "Strategy spec schema file not found"

    with open(schema_path, 'r') as f:
        schema = json.load(f)

    # Validate schema structure
    assert "properties" in schema
    assert "instrument" in schema["properties"]
    assert "data" in schema["properties"]
    assert "backtest" in schema["properties"]
    assert "performance" in schema["properties"]


def test_pipeline_run_schema():
    """Test pipeline run JSON schema validation."""
    schema_path = Path("configs/templates/pipeline_run.schema.json")

    assert schema_path.exists(), "Pipeline run schema file not found"

    with open(schema_path, 'r') as f:
        schema = json.load(f)

    # Validate schema structure
    assert "properties" in schema
    assert "command" in schema["properties"]
    assert "strategy_config" in schema["properties"]


def test_hash_spec_stability():
    """Test that hash_spec produces stable results."""
    spec1 = {
        "strategy_name": "test",
        "instrument": {"asset_type": "US_STOCK", "symbol": "AAPL"},
        "data": {"frequency": "1D"},
        "backtest": {"initial_capital": 100000},
    }

    spec2 = spec1.copy()
    spec2["extra_field"] = "should not affect hash"

    hash1 = hash_spec(spec1)
    hash2 = hash_spec(spec2)

    assert hash1 != hash2, "Hash should change with spec"
    assert hash1 == hash_spec(spec1), "Hash should be deterministic"
