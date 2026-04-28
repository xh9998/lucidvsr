import os
import subprocess
import sys
from contextlib import contextmanager
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional, Sequence


def _normalize_listing_url(url: str) -> str:
    normalized = url
    if normalized.startswith("s3://"):
        normalized = "conductor://" + normalized[len("s3://") :]
    return normalized


def _to_conductor_cli_url(url: str) -> Optional[str]:
    if url.startswith("s3://"):
        return url
    if url.startswith("conductor://"):
        return "s3://" + url[len("conductor://") :]
    return None


def _extract_bucket_from_cli_url(cli_url: str) -> Optional[str]:
    if not cli_url.startswith("s3://"):
        return None
    remainder = cli_url[len("s3://") :]
    if not remainder:
        return None
    return remainder.split("/", 1)[0]


def _from_conductor_cli_path(path: str, bucket: Optional[str]) -> str:
    normalized_path = path.lstrip("/")
    if normalized_path.startswith("s3://"):
        return "conductor://" + normalized_path[len("s3://") :]
    if bucket is None:
        return "conductor://" + normalized_path
    if normalized_path.startswith(f"{bucket}/"):
        return "conductor://" + normalized_path
    return f"conductor://{bucket}/{normalized_path}"


def _list_via_conductor_cli(url: str, recursive: bool = True) -> list[str]:
    cli_url = _to_conductor_cli_url(url)
    if cli_url is None:
        return []
    bucket = _extract_bucket_from_cli_url(cli_url)
    command = ["conductor", "s3", "ls", cli_url]
    if recursive:
        command.append("--recursive")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    urls: list[str] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("PRE "):
            continue
        parts = line.split()
        if not parts:
            continue
        path = parts[-1]
        if path.endswith("/"):
            continue
        if path.startswith("s3://"):
            urls.append("conductor://" + path[len("s3://") :])
        else:
            urls.append(_from_conductor_cli_path(path, bucket))
    return sorted(set(urls))


_INITIALIZED = False
_CLIENT_IMPORT_ERROR: Optional[Exception] = None
_CONDUCTOR_OPEN = None
_GET_CONDUCTOR_CLIENT = None
_APPLE_FSSPEC = None


def _candidate_ref_big_roots() -> list[str]:
    repo_root = Path(__file__).resolve().parents[5]
    code_root = repo_root.parent.parent
    candidates = [
        "/mnt/task_runtime/ref_big",
        "/Users/lixiaohui/Library/CloudStorage/Box-Box/code/ref_big",
        str(code_root / "ref_big"),
    ]
    unique: list[str] = []
    for item in candidates:
        if item not in unique and os.path.isdir(item):
            unique.append(item)
    return unique


def _ensure_ref_big_client() -> bool:
    global _INITIALIZED, _CLIENT_IMPORT_ERROR, _CONDUCTOR_OPEN, _GET_CONDUCTOR_CLIENT, _APPLE_FSSPEC
    if _INITIALIZED:
        return _CONDUCTOR_OPEN is not None and _GET_CONDUCTOR_CLIENT is not None and _APPLE_FSSPEC is not None
    _INITIALIZED = True

    for root in _candidate_ref_big_roots():
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            import apple_fsspec
            from data.data_utils.conductor_integration import enable_conductor_optimizations

            enable_conductor_optimizations()
            from data.data_utils.conductor_client import conductor_open, get_conductor_client

            _APPLE_FSSPEC = apple_fsspec
            _CONDUCTOR_OPEN = conductor_open
            _GET_CONDUCTOR_CLIENT = get_conductor_client
            return True
        except Exception as exc:
            _CLIENT_IMPORT_ERROR = exc
            continue
    return False


def has_ref_big_conductor() -> bool:
    return _ensure_ref_big_client()


@contextmanager
def open_conductor_stream(url: str, mode: str = "rb"):
    if not _ensure_ref_big_client():
        raise RuntimeError(f"ref_big conductor client unavailable: {_CLIENT_IMPORT_ERROR}")
    handle = _CONDUCTOR_OPEN(url, mode)
    try:
        yield handle
    finally:
        handle.close()


def get_conductor_local_path(url: str) -> Optional[str]:
    if not _ensure_ref_big_client():
        return None
    with open_conductor_stream(url, "rb") as handle:
        local_path = getattr(handle, "name", None)
        if isinstance(local_path, str) and os.path.exists(local_path):
            return local_path
    return None


def read_conductor_bytes(url: str) -> bytes:
    if not _ensure_ref_big_client():
        raise RuntimeError(f"ref_big conductor client unavailable: {_CLIENT_IMPORT_ERROR}")
    with open_conductor_stream(url, "rb") as handle:
        return handle.read()


def get_ref_big_conductor_error() -> Optional[Exception]:
    _ensure_ref_big_client()
    return _CLIENT_IMPORT_ERROR


def list_remote_files(url: str, recursive: bool = True, line_limit: Optional[int] = None) -> list[str]:
    normalized = _normalize_listing_url(url)
    if not (
        normalized.startswith("conductor://")
        or normalized.startswith("blobby://")
    ):
        raise ValueError(f"list_remote_files only supports conductor/s3/blobby urls, got: {url}")

    have_ref_big = _ensure_ref_big_client()

    base = normalized.rstrip("/")
    patterns = [f"{base}/**/*", f"{base}/**"] if recursive else [f"{base}/*"]
    urls: list[str] = []
    errors: list[Exception] = []
    if have_ref_big:
        for pattern in patterns:
            try:
                urls = list(_APPLE_FSSPEC.find_files(pattern))
            except Exception as exc:
                errors.append(exc)
                continue
            if urls:
                break
    if not urls and errors:
        cli_urls = _list_via_conductor_cli(url, recursive=recursive)
        if cli_urls:
            urls = cli_urls
        else:
            raise RuntimeError(f"failed to list {url} via ref_big/apple_fsspec: {errors[-1]}")
    elif not urls and not have_ref_big:
        cli_urls = _list_via_conductor_cli(url, recursive=recursive)
        if cli_urls:
            urls = cli_urls
        else:
            raise RuntimeError(f"ref_big conductor client unavailable: {_CLIENT_IMPORT_ERROR}")
    urls = [item for item in urls if not item.endswith("/")]
    urls = sorted(set(urls))
    if line_limit is not None:
        urls = urls[: int(line_limit)]
    return urls


def list_remote_files_with_suffixes(
    url: str,
    suffixes: Sequence[str],
    recursive: bool = True,
    line_limit: Optional[int] = None,
) -> list[str]:
    suffixes = tuple(suffixes)
    normalized = _normalize_listing_url(url)
    base = normalized.rstrip("/")
    recursive_patterns = [f"{base}/**/*{suffix}" for suffix in suffixes]
    flat_patterns = [f"{base}/*{suffix}" for suffix in suffixes]
    patterns = recursive_patterns if recursive else flat_patterns

    have_ref_big = _ensure_ref_big_client()

    urls: list[str] = []
    errors: list[Exception] = []
    if have_ref_big:
        for pattern in patterns:
            try:
                urls.extend(list(_APPLE_FSSPEC.find_files(pattern)))
            except Exception as exc:
                errors.append(exc)
    filtered_urls = [
        item for item in sorted(set(urls))
        if any(fnmatch(item, f"*{suffix}") for suffix in suffixes)
    ]
    if not filtered_urls:
        cli_urls = _list_via_conductor_cli(url, recursive=recursive)
        filtered_urls = [
            item for item in sorted(set(cli_urls))
            if any(fnmatch(item, f"*{suffix}") for suffix in suffixes)
    ]
    if not filtered_urls and errors:
        raise RuntimeError(f"failed to list {url} via ref_big/apple_fsspec: {errors[-1]}")
    if not filtered_urls and not have_ref_big and _to_conductor_cli_url(url) is None:
        raise RuntimeError(f"ref_big conductor client unavailable: {_CLIENT_IMPORT_ERROR}")
    if line_limit is not None:
        filtered_urls = filtered_urls[: int(line_limit)]
    return filtered_urls
