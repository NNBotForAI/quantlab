"""
Atomic writes and safe file operations
"""
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Union


def safe_path_join(base: Path, *parts: Union[str, Path]) -> Path:
    """
    Safely join paths without directory traversal.

    Args:
        base: Base directory
        *parts: Path components to join

    Returns:
        Safe joined path
    """
    result = base
    for part in parts:
        result = result / str(part)
        # Resolve parent directory references but stay within base
        result = result.resolve()
        try:
            result.relative_to(base.resolve())
        except ValueError:
            raise ValueError(f"Path traversal detected: {result}")
    return result


def atomic_write(path: Path, content: Union[str, bytes], mode: str = "w") -> None:
    """
    Atomic file write using temporary file.

    Args:
        path: Target file path
        content: Content to write
        mode: Write mode
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file first
    with tempfile.NamedTemporaryFile(
        mode=mode,
        dir=path.parent,
        prefix=f".tmp_{path.name}",
        delete=False,
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    # Atomic rename
    tmp_path.replace(path)


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """
    Atomic JSON write.

    Args:
        path: Target file path
        data: Data to serialize
        indent: JSON indentation
    """
    json_str = json.dumps(data, indent=indent, default=str, ensure_ascii=False)
    atomic_write(path, json_str, mode="w")


def read_json(path: Path) -> Any:
    """
    Read JSON file.

    Args:
        path: File path

    Returns:
        Parsed JSON data
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    """
    Ensure directory exists.

    Args:
        path: Directory path
    """
    path.mkdir(parents=True, exist_ok=True)


def safe_delete(path: Path) -> None:
    """
    Safely delete file or directory.

    Args:
        path: Path to delete
    """
    if not path.exists():
        return

    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def safe_move(src: Path, dst: Path) -> None:
    """
    Safely move file or directory.

    Args:
        src: Source path
        dst: Destination path
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
