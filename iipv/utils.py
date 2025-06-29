import math
import numpy as np
import threading
import time
import traceback

from concurrent.futures import ThreadPoolExecutor

def DIVUP(a: int | float, b: int | float) -> int:
    """Divide a by b and round up to the nearest integer."""
    return int(math.ceil( float(a) / float(b)))

class AtomicCounter:
    """A thread-safe counter that can be decremented and waited for zero."""
    def __init__(self, value) -> None:
        self._value = value
        self._mutex = threading.Lock()
        self._zero = threading.Event()
        self._zero_time = 0.
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

    def wait_for_zero(self) -> None:
        """Wait until the counter reaches zero."""
        self._zero.wait()

    def timestamp_zero(self) -> float:
        """Return the timestamp when the counter reached zero."""
        return self._zero_time

def _error_displaying_wrapper(f, *args, **kwargs) -> None:
    """A wrapper to catch exceptions and print them."""
    try:
        f(*args, **kwargs)
    except Exception:
        print(f"{f} failed with arguments {args}, {kwargs}")
        print(traceback.format_exc())

class DebugThreadPoolExecutor(ThreadPoolExecutor):
    """A ThreadPoolExecutor that catches exceptions and prints them."""
    def submit(self, fn, *args, **kwargs):
        """Submit a task to the executor with error handling."""
        return super().submit(_error_displaying_wrapper, fn, *args, **kwargs)