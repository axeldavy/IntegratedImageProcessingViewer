import dearcygui as dcg
import os

from iipv.viewer import ViewerElement
from iipv.readers import read_image

from functools import lru_cache
import numpy as np

import cv2

@lru_cache(5)
def read_image_with_cache(path):
    return read_image(path)

C = dcg.Context()

noise_images = {}
noise_std = dcg.SharedFloat(C, 0.)
blur_width = dcg.SharedInt(C, 1)
blur_height = dcg.SharedInt(C, 1)
viewer = None

def read_image_and_add_process(path):
    global noise_std
    global noise_images
    print(path)
    image = read_image_with_cache(path) / 255.
    # Add noise
    noise_image = noise_images.get(path, None)
    if noise_image is None:
        noise_image = np.random.randn(*image.shape)
        noise_images[path] = noise_image
    image = image + (noise_std.value / 255) * noise_image
    # Add blur
    kernel = (blur_width.value, blur_height.value)
    image = cv2.blur(image, ksize=kernel)
    return image


path = os.path.join(os.path.dirname(__file__), "lenapashm.png")
with dcg.Window(C, primary=True):
    with dcg.ChildWindow(C, width=-200, height=0, no_newline=True, no_scrollbar=True):
        viewer = ViewerElement(C, [path], 1, read_image_and_add_process)
    with dcg.ChildWindow(C, width=0, height=0):
        dcg.Slider(C, label="Noise std.", shareable_value=noise_std, min_value=0, max_value=50, width=100, callback=viewer.refresh_image)
        dcg.Slider(C, label="Blur width", shareable_value=blur_width, min_value=1, format='int', max_value=10, width=100, callback=viewer.refresh_image)
        dcg.Slider(C, label="Blur height", shareable_value=blur_height, min_value=1, format='int', max_value=10, width=100, callback=viewer.refresh_image)

C.viewport.initialize(vsync=True,
                      wait_for_input=True,
                      title="Just a demo")

while C.running:
    C.viewport.render_frame()
