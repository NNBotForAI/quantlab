"""
Stable hashing for configs, specs, and files
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def hash_dict(data: Dict[str, Any], sort_keys: bool = True) -> str:
    """
    Create stable hash from dictionary.

    Args:
        data: Dictionary to hash
        sort_keys: Sort keys for deterministic output

    Returns:
        SHA256 hex string
    """
    if sort_keys:
        json_str = json.dumps(data, sort_keys=True, default=str)
    else:
        json_str = json.dumps(data, default=str)

    return hashlib.sha256(json_str.encode()).hexdigest()


def hash_file(path: Path, chunk_size: int = 8192) -> str:
    """
    Create hash from file.

    Args:
        path: File path
        chunk_size: Read chunk size

    Returns:
        SHA256 hex string
    """
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)

    return sha256.hexdigest()


def hash_files(paths: list[Path]) -> str:
    """
    Create combined hash from multiple files.

    Args:
        paths: List of file paths

    Returns:
        SHA256 hex string
    """
    sha256 = hashlib.sha256()

    for path in sorted(paths):
        if path.exists():
            file_hash = hash_file(path)
            sha256.update(file_hash.encode())

    return sha256.hexdigest()


def hash_spec(spec: Dict[str, Any]) -> str:
    """
    Create hash from strategy specification.

    Args:
        spec: Strategy spec dictionary

    Returns:
        SHA256 hex string
    """
    # Remove runtime-specific fields that shouldn't affect cache
    cache_key = {
        k: v for k, v in spec.items()
        if k not in ["run_id", "data_version", "code_version"]
    }
    return hash_dict(cache_key)


def create_run_id(spec: Dict[str, Any], data_version: str, code_version: str) -> str:
    """
    Create unique run ID from spec, data version, and code version.

    Args:
        spec: Strategy spec dictionary
        data_version: Data version hash
        code_version: Code version hash

    Returns:
        Unique run ID (8 chars)
    """
    combined = hash_dict({
        "spec": hash_spec(spec),
        "data": data_version,
        "code": code_version,
    })
    return combined[:8]
