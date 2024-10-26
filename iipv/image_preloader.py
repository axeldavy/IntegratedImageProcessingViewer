
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import imageio
import numpy as np
import os
import threading
import traceback

has_tiffile = False
try:
    import tifffile
    has_tiffile = True
except Exception:
    pass

has_lycon = False
try:
    import lycon
    has_lycon = True
except Exception:
    pass


def error_displaying_wrapper(obj, f, *args, **kwargs):
    try:
        f(*args, **kwargs)
    except Exception:
        print(f"{f} failed with arguments {args}, {kwargs}")
        print(traceback.format_exc())

class ThreadPoolExecutorWithErrorMessage(ThreadPoolExecutor):
    """A threadpool that doesn't fail silently"""
    def submit(f, *args, **kwargs):
        return super().submit(error_displaying_wrapper, f, *args, **kwargs)

class TiffZarrAsNumpyArray:
    def __init__(self, path):
        # Using zarr enables to only read the displayed
        # portion (best with tiffs encoded with tiles)
        reader = tifffile.TiffReader(path)
        shape = reader.shaped_metadata['shape']
        self.shape = shape
        self.zarr_array = reader.aszarr(level=0)
        self.one_pixel = np.asarray(self.zarr_array[0, 0])
        self.nbytes = self.one_pixel.nbytes * np.prod(self.shape)
        self.dtype = self.one_pixel.dtype

    def __getitem__(self, *args):
        return np.asarray(self.zarr_array.__getitem__(*args))

class ImagePreloader:
    def __init__(self, cache_size_GB=2.):
        self.max_cache_size_B = cache_size_GB * (1024 ** 3)
        self.cache_size = 0
        self.threadpool = ThreadPoolExecutorWithErrorMessage()
        self.mutex = threading.Lock()
        self.cache = {}
        self.cache_insertion_order = deque()

    def read_image(self, path):
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        #if ext in ['.tiff', '.tif'] and has_tiffile:
        #    return TiffZarrAsNumpyArray(path)
        #if ext in ['.png', '.jpg'] and has_lycon:
        #    reader = tifffile.TiffReader(path)
        #    return lycon.TODO
        return imageio.imread(path)

    def read_and_remember(self, path, callback):
        # Retrieve if in cache
        with self.mutex:
            image = self.cache.get(path, None)
        # Retrieve if not in cache
        if image is None:
            image = self.read_image(path)
        # Store in cache
        with self.mutex:
            is_cache = path in self.cache
            self.cache[path] = image
            if is_cache:
                self.cache_insertion_order.remove(path)
                self.cache_size += image.nbytes
            self.cache_insertion_order.append(path)
            # Free cache excess
            while self.cache_size > self.max_cache_size_B:
                other_path = self.cache_insertion_order.popleft()
                other_image = self.cache[other_path]
                del self.cache[other_path]
                self.cache_size -= other_image.nbytes

        return callback(image)

    def delayed_read(self, path, callback):
        self.threadpool.submit(self.read_and_remember, path, callback)