from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import imageio
import math
from natsort import natsorted
import os
import threading
import time
import traceback
from typing import Any


def DIVUP(a: int | float, b: int | float) -> int:
    """Divide a by b and round up to the nearest integer."""
    return int(math.ceil( float(a) / float(b)))

class AtomicCounter:
    """A thread-safe counter that can be decremented and waited for zero."""
    def __init__(self, value: int, timeout=None) -> None:
        """
        Args:
            value: initial value for the counter
            timeout: optional timeout after which wait_for_zero will exit.
        """
        self._value = value
        self._mutex = threading.Lock()
        self._zero = threading.Event()
        self._zero_time = 0.
        self._timeout = timeout
        if value <= 0:
            self._zero_time: float = time.monotonic()
            self._zero.set()

    def dec(self) -> int:
        """Decrement the counter and return the new value."""
        with self._mutex:
            self._value -= 1
            if self._value <= 0:
                self._zero_time = time.monotonic()
                self._zero.set()
            return self._value

    def wait_for_zero(self, timeout=None) -> None:
        """Wait until the counter reaches zero."""
        if timeout is None:
            timeout = self._timeout
        self._zero.wait(timeout=timeout)

    def timestamp_zero(self) -> float:
        """Return the timestamp when the counter reached zero."""
        return self._zero_time

def _error_displaying_wrapper(f, *args, **kwargs) -> Any:
    """A wrapper to catch exceptions and print them."""
    try:
        return f(*args, **kwargs)
    except Exception:
        print(f"{f} failed with arguments {args}, {kwargs}")
        print(traceback.format_exc())

class DebugThreadPoolExecutor(ThreadPoolExecutor):
    """A ThreadPoolExecutor that catches exceptions and prints them."""
    def submit(self, fn, *args, **kwargs):
        """Submit a task to the executor with error handling."""
        return super().submit(_error_displaying_wrapper, fn, *args, **kwargs)

def find_all_images(path) -> Sequence[str]:
    """
    Report all files at path which
    we are supposed to be able to read
    """
    files = []
    if not(os.path.isdir(path)):
        _, extension = os.path.splitext(path)
        if extension.lower() in imageio.config.known_extensions.keys():
            return [path]
        else:
            return []
    for item in os.scandir(path):
        if item.is_dir():
            files += find_all_images(item)
        elif item.is_file():
            _, extension = os.path.splitext(item.path)
            if extension.lower() in imageio.config.known_extensions.keys():
                files.append(item.path)
    return files

def sort_all_files(files) -> Sequence[str]:
    """
    We do not just sort based on the string, as 
    we want prefix_2.ext to be before prefix_10.ext
    """
    return natsorted(files)