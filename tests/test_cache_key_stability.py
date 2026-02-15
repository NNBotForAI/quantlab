"""
Cache key stability tests
"""
import pytest
from pathlib import Path

from src.quantlab.common.cache import FeatureCache
from src.quantlab.common.hashing import hash_spec, create_run_id


def test_cache_key_includes_spec_hash():
    """Test that cache key includes spec hash."""
    cache = FeatureCache(Path("test_cache"), run_id="test")

    spec = {"strategy_name": "test", "data": {"frequency": "1D"}}
    spec_hash = hash_spec(spec)
    data_version = "abc123"
    code_version = "def456"
    stage = "features"

    # Create cache key
    key = cache._make_key(stage, spec_hash, data_version, code_version)

    # Check that spec_hash is in key
    assert spec_hash in key, "Spec hash should be in cache key"
    assert stage in key, "Stage name should be in cache key"
    assert data_version in key, "Data version should be in cache key"
    assert code_version in key, "Code version should be in cache key"


def test_cache_key_stability():
    """Test that cache keys are stable."""
    cache = FeatureCache(Path("test_cache"), run_id="test")

    spec = {
        "strategy_name": "test",
        "data": {"frequency": "1D"},
        "backtest": {"initial_capital": 100000}
    }
    spec_hash = hash_spec(spec)
    data_version = "abc123"
    code_version = "def456"
    stage = "features"

    # Create cache key twice
    key1 = cache._make_key(stage, spec_hash, data_version, code_version)
    key2 = cache._make_key(stage, spec_hash, data_version, code_version)

    # Keys should be identical
    assert key1 == key2, "Cache keys should be stable"


def test_cache_key_changes_with_spec():
    """Test that cache key changes when spec changes."""
    cache = FeatureCache(Path("test_cache"), run_id="test")

    spec1 = {"strategy_name": "test", "data": {"frequency": "1D"}}
    spec2 = {"strategy_name": "test", "data": {"frequency": "1H"}}

    spec_hash1 = hash_spec(spec1)
    spec_hash2 = hash_spec(spec2)

    data_version = "abc123"
    code_version = "def456"
    stage = "features"

    # Create cache keys
    key1 = cache._make_key(stage, spec_hash1, data_version, code_version)
    key2 = cache._make_key(stage, spec_hash2, data_version, code_version)

    # Keys should be different
    assert key1 != key2, "Cache keys should differ when spec changes"


def test_create_run_id():
    """Test run ID creation."""
    spec = {"strategy_name": "test"}
    data_version = "abc123"
    code_version = "def456"

    run_id = create_run_id(spec, data_version, code_version)

    # Run ID should be a string
    assert isinstance(run_id, str), "Run ID should be a string"

    # Run ID should be 8 characters (from hash truncation)
    assert len(run_id) == 8, f"Run ID should be 8 chars, got {len(run_id)}"


def test_cache_get_or_compute():
    """Test cache get_or_compute functionality."""
    cache_dir = Path("test_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = FeatureCache(cache_dir, run_id="test")

    spec = {"strategy_name": "test"}
    spec_hash = hash_spec(spec)
    data_version = "abc123"
    code_version = "def456"
    stage = "test_stage"

    # Define compute function
    call_count = 0

    def compute_fn():
        nonlocal call_count
        call_count += 1
        return {"result": call_count}

    # First call should compute
    result1 = cache.get_or_compute(
        spec_hash=spec_hash,
        data_version=data_version,
        code_version=code_version,
        stage=stage,
        compute_fn=compute_fn,
        use_cache=True,
    )

    assert call_count == 1, "Should have computed on first call"
    assert result1 == {"result": 1}

    # Second call with same parameters should use cache
    result2 = cache.get_or_compute(
        spec_hash=spec_hash,
        data_version=data_version,
        code_version=code_version,
        stage=stage,
        compute_fn=compute_fn,
        use_cache=True,
    )

    assert call_count == 1, "Should not recompute when cache is used"
    assert result2 == {"result": 1}

    # Third call without cache should recompute
    result3 = cache.get_or_compute(
        spec_hash=spec_hash,
        data_version=data_version,
        code_version=code_version,
        stage=stage,
        compute_fn=compute_fn,
        use_cache=False,
    )

    assert call_count == 2, "Should recompute when cache is not used"
    assert result3 == {"result": 2}
