import os
import random
import subprocess
import time
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import fcntl


def _io_limiter_dir() -> Path:
    base_dir = os.environ.get("FLASHVSR_IO_NODE_LIMIT_DIR")
    if base_dir:
        return Path(base_dir)
    return Path(tempfile.gettempdir()) / "flashvsr_io_limiter"


def _io_limiter_parallelism() -> int:
    try:
        value = int(os.environ.get("FLASHVSR_IO_MAX_PARALLEL", "4"))
    except Exception:
        value = 4
    return max(1, value)


@contextmanager
def acquire_node_io_slot():
    limiter_dir = _io_limiter_dir()
    limiter_dir.mkdir(parents=True, exist_ok=True)
    slot_count = _io_limiter_parallelism()
    handles = []
    try:
        while True:
            for slot_index in range(slot_count):
                slot_path = limiter_dir / f"slot_{slot_index:02d}.lock"
                handle = open(slot_path, "a+", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    handles.append(handle)
                    yield slot_index
                    return
                except BlockingIOError:
                    handle.close()
            time.sleep(0.1 + random.uniform(0.0, 0.1))
    finally:
        for handle in handles:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            handle.close()


def run_subprocess_with_retry(
    command: Sequence[str],
    *,
    max_attempts: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    stdout=None,
    stderr=None,
    capture_output: bool = False,
    text: bool = False,
):
    last_error = None
    for attempt in range(max_attempts):
        try:
            with acquire_node_io_slot():
                if capture_output:
                    return subprocess.run(
                        list(command),
                        check=True,
                        capture_output=True,
                        text=text,
                    )
                subprocess.check_call(
                    list(command),
                    stdout=stdout,
                    stderr=stderr,
                )
                return None
        except Exception as error:
            last_error = error
            if attempt >= max_attempts - 1:
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            delay *= random.uniform(0.75, 1.25)
            time.sleep(delay)
    raise last_error
