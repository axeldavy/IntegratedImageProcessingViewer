import imageio
import math

def read_image(path):
    return imageio.v3.imread(path, index=0)

class SeriesReader:
    """
    Reader that supports images,
    videos, and multi-page files
    """
    def __init__(self, path):
        self.path = path
        self.fp = imageio.v3.imopen(self.path, 'r')
        self.properties = self.fp.properties(index=...)
        self.num_images = self.properties.n_images
        if self.num_images == math.inf:
            print("Warning: the number of frames is undetermined")
            self.num_images = 1e3
        self.num_images = int(self.num_images)

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        try:
            return self.fp.read(index=index)
        except Exception:
            self.num_images = min(self.num_images, index-1)
            if index > 0:
                return self[index-1]