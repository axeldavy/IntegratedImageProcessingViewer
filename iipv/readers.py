from collections.abc import Sequence
import imageio
import math
from numpy.typing import NDArray

def read_image(path: str) -> NDArray:
    return imageio.v3.imread(path, index=0)

class SeriesReader(Sequence):
    """
    Reader that supports images,
    videos, and multi-page files
    """
    def __init__(self, path: str) -> None:
        self._path: str = path
        self._fp = imageio.v3.imopen(self._path, 'r')
        self._properties = self._fp.properties(index=...) # type: ignore
        if self._properties.n_images == math.inf: # type: ignore
            print("Warning: the number of frames is undetermined")
            self._num_images: int = 1000  # Default to 1000 if unknown
        else:
            self._num_images: int = self._properties.n_images if self._properties.n_images is not None else 1
        if self._num_images <= 0:
            raise ValueError(f"Invalid number of images: {self._num_images} in {self._path}")

    def __len__(self) -> int:
        return self._num_images

    def __getitem__(self, index: int) -> NDArray | None: # type: ignore
        try:
            return self._fp.read(index=index)
        except Exception:
            self.num_images = min(self.num_images, index-1)
            if index > 0:
                return self[index-1]
