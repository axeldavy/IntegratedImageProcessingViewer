import argparse
import dearcygui as dcg
import imageio
from natsort import natsorted
import os
from viewer import ViewerWindow

def find_all_images(path):
    """
    Report all files at path which
    we are supposed to be able to read
    """
    files = []
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', help=('Input directory'))
    args = parser.parse_args()

    paths = sort_all_files(find_all_images(args.indir))

    C = dcg.Context()
    C.viewport.initialize(vsync=True, wait_for_input=True)
    ViewerWindow(C, paths, primary=True, no_bring_to_front_on_focus=True)
    #dcg.MetricsWindow(C)
    while C.running:
        C.viewport.render_frame(can_skip_presenting=True)
    