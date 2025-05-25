import argparse
import dearcygui as dcg
import imageio
from natsort import natsorted
import os
from .viewer import ViewerWindow
from .readers import SeriesReader
import math

def find_all_images(path):
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

def sort_all_files(files):
    """
    We do not just sort based on the string, as 
    we want prefix_2.ext to be before prefix_10.ext
    """
    return natsorted(files)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', help=('Input directory or file'))
    args = parser.parse_args()

    paths = sort_all_files(find_all_images(args.indir))

    C = dcg.Context()
    # vsync: limit to screen refresh rate and have no tearing
    # wait_for_input: Do not refresh until a change is detected (C.viewport.wake() to help)
    C.viewport.initialize(vsync=True,
                          wait_for_input=True,
                          title="Integrated Image Processing Viewer")
    # primary: use the whole window area
    # no_bring_to_front_on_focus: enables to have windows on top to
    # add your custom UI, and not have them hidden when clicking on the image.
    if len(paths) == 0:
        raise ValueError("No compatible file found")
    else:
        reader = SeriesReader
        num_images = len(paths)

    ViewerWindow(C, [paths], [num_images], [reader], primary=True, no_bring_to_front_on_focus=True)
    while C.running:
        C.viewport.render_frame()

if __name__ == '__main__':
    main()
    
