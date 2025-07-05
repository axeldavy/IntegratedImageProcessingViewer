import argparse
import dearcygui as dcg

from .readers import SeriesReader
from .utils import find_all_images, sort_all_files
from .viewer import ViewerWindow

def on_resize(sender, target: dcg.Viewport, data):
    pass # TODO
    #if target.maximized:
        #target.decorated = False
        # request fullscreen
        #target.fullscreen = True

def main() -> None:
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

    ViewerWindow(C, [paths], [num_images], [reader], primary=True, no_bring_to_front_on_focus=True, resize_callback=on_resize)
    while C.running:
        C.viewport.render_frame()

if __name__ == '__main__':
    main()
    
