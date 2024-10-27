import dearcygui as dcg
import imageio
import math
import numpy as np
import os
import threading
import traceback
from .image_preloader import ImagePreloader
import time

def DIVUP(a, b):
    return int(math.ceil( float(a) / float(b)))

class ViewerImage(dcg.DrawInPlot):
    """
    Instance representing the image displayed
    """
    def __init__(self, context, tile_size=[1024, 1024], margin=256, **kwargs):
        super().__init__(context, **kwargs)
        self.image = np.zeros([0, 0], dtype=np.int32)
        self._transform = lambda x: x
        # Cut the image into tiles for display,
        # as we might show bigger images than
        # a texture can support
        self.tile_size = tile_size
        self.margin = margin # spatial margin to load in advance
        self.should_fit = True
        # Use double buffering as uploading the image
        # to the gpu is not instantaneous
        self.front = dcg.DrawingList(context, parent=self)
        self.back = dcg.DrawingList(context, parent=self, show=False)
        self.up_to_date_tiles_front = set()
        self.up_to_date_tiles_back = set()
        self.tiles_front = dict()
        self.tiles_back = dict()
        # We finish uploading the previous image before
        # starting a new upload
        self.update_mutex = threading.RLock()

    @property
    def transform(self):
        """
        Function that is applied on tiles of the data
        before displaying.
        Output should be between 0 and 255,
        and can be R (in that case displayed as gray),
        RG (B is set to 0), RGB or RGBA.
        R: single channel or no channel
        RG: two channels
        RGB/RGBA: three or four channels.
        Expected output dtype is uint8 or float32.
        """
        return self._transform

    @transform.setter
    def transform(self, value):
        self._transform = value
        with self.update_mutex:
            self.up_to_date_tiles_front.clear()
            self.up_to_date_tiles_back.clear()
            self.update_image()

    def display(self, image):
        """Display an image, replacing any old one"""
        with self.update_mutex:
            if image is not self.image:
                self.up_to_date_tiles_front.clear()
                self.up_to_date_tiles_back.clear()
            self.image = image
            self.update_image()

    def update_image(self):
        """Update the image displayed if needed"""
        with self.update_mutex:
            h = self.image.shape[0]
            w = self.image.shape[1]
            if w == 0 or h == 0:
                return
            tiles_w = DIVUP(w, self.tile_size[1])
            tiles_h = DIVUP(h, self.tile_size[0])
            # TODO: configurable axes
            X = self.parent.X1
            Y = self.parent.Y1
            min_x = X.min - self.margin
            max_x = X.max + self.margin
            min_y = Y.min - self.margin
            max_y = Y.max + self.margin
            if self.should_fit:
                min_x = 0
                max_x = w
                min_y = 0
                max_y = h
            any_action = False
            for i_w in range(tiles_w):
                xm = i_w * self.tile_size[1]
                xM = min(xm + self.tile_size[1], w)
                if xM < min_x or xm > max_x:
                    continue
                for i_h in range(tiles_h):
                    ym = i_h * self.tile_size[0]
                    yM = min(ym + self.tile_size[0], h)
                    if yM < min_y or ym > max_y:
                        continue
                    if (i_h, i_w) in self.up_to_date_tiles_front:
                        continue
                    any_action = True
            if not(any_action):
                return
            switched_textures = {}
            for i_w in range(tiles_w):
                xm = i_w * self.tile_size[1]
                xM = min(xm + self.tile_size[1], w)
                if xM < min_x or xm > max_x:
                    continue
                for i_h in range(tiles_h):
                    ym = i_h * self.tile_size[0]
                    yM = min(ym + self.tile_size[0], h)
                    if yM < min_y or ym > max_y:
                        continue
                    if (i_h, i_w) in self.up_to_date_tiles_back:
                        continue
                    # Try to reuse existing textures if possible
                    prev_content = self.tiles_back.get((i_h, i_w), None)
                    if prev_content is None:
                        # Initialize the image and its texture
                        prev_content = dcg.DrawImage(self.context,
                                                     parent=self.back,
                                                     pmin=(xm, ym),
                                                     pmax=(xM, yM))
                        prev_content.texture = \
                            dcg.Texture(self.context,
                                        nearest_neighbor_upsampling=True)
                        self.tiles_back[(i_h, i_w)] = prev_content
                    else:
                        if prev_content.texture.width != (xM-xm) or \
                           prev_content.texture.height != (yM-ym):
                            # Right now texture resize in DCG is slow
                            # best to allocate a new one
                            prev_content.texture = \
                                dcg.Texture(self.context,
                                            nearest_neighbor_upsampling=True)
                    # Update max in case of change of size
                    prev_content.pmax = (xM, yM)
                    # Already have a texture with up to date content. Take it
                    if (i_h, i_w) in self.up_to_date_tiles_front:
                        switched_textures[(i_h, i_w)] = prev_content.texture
                        prev_content.texture = self.tiles_front[(i_h, i_w)].texture
                        self.up_to_date_tiles_back.add((i_h, i_w))
                        continue
                    tile = self.image[ym:yM, xm:xM, ...]
                    #print(tile.shape, prev_content.pmin, prev_content.pmax)
                    # We don't use self._transform, so that the user
                    # can subclass and replace transform
                    try:
                        processed_tile = self.transform(tile)
                        prev_content.texture.set_value(processed_tile)
                    except Exception:
                        print(traceback.format_exc())
                    self.up_to_date_tiles_back.add((i_h, i_w))
            # Free previous out of date tiles
            out_of_date = [key for key in self.tiles_back.keys() if key not in self.up_to_date_tiles_back]
            for key in out_of_date:
                self.tiles_back[key].detach_item()
                del self.tiles_back[key]
            # Switch back and front
            tmp = self.front
            self.front = self.back
            self.back = tmp
            self.front.show = True
            self.back.show = False
            # Once the new back is not shown anymore, we can replace
            # with the non-up to date textures.
            for ((i_h, i_w), texture) in switched_textures.items():
                self.tiles_front[(i_h, i_w)].texture = texture
                self.up_to_date_tiles_front.remove((i_h, i_w))
            tmp = self.tiles_front
            self.tiles_front = self.tiles_back
            self.tiles_back = tmp
            tmp = self.up_to_date_tiles_front
            self.up_to_date_tiles_front = self.up_to_date_tiles_back
            self.up_to_date_tiles_back = tmp
            # Fit if requested
            if self.should_fit:
                X.fit()
                Y.fit()
                self.should_fit = False
            # Indicate content has changed (wait_for_input)
            self.context.viewport.wake()


class ViewerElement(dcg.Plot):
    """
    Sub-window to visualize one sequence of data
    """
    def __init__(self, context, paths, index=0, reader=None, transform=None, **kwargs):
        super().__init__(context, **kwargs)
        self.paths = paths
        self._index = index
        self.image_loader = reader if reader is not None else ImagePreloader()
        self.image_viewer = ViewerImage(context, parent=self)
        if transform is not None:
            self.image_viewer.transform = transform
        # Disable all plot features we don't want
        self.X1.no_label = True
        self.X1.no_gridlines = True
        self.X1.no_tick_marks = True
        self.X1.no_tick_labels = True
        self.X1.no_menus = True
        self.X1.no_side_switch = True
        self.X1.no_highlight = True
        self.Y1.no_label = True
        self.Y1.no_gridlines = True
        self.Y1.no_tick_marks = True
        self.Y1.no_tick_labels = True
        self.Y1.no_menus = True
        self.Y1.no_side_switch = True
        self.Y1.no_highlight = True
        # invert Y
        self.Y1.invert = True
        self.fit_button = 4 # we don't want that, so set to an useless button
        self.no_title = True
        self.no_mouse_pos = True
        self.equal_aspects = True # We do really want that for images
        self.no_frame = True
        self.no_legend = True
        # fit whole size available
        self.width = -1
        self.height = -1
        # Set a handler to update the images when the plot min/max change
        self.handlers += [
            dcg.AxesResizeHandler(context, callback=self.on_resize)
        ]
        # Remove empty borders
        self.theme = dcg.ThemeStyleImPlot(self.context, PlotPadding=(0, 0))
        self.load_image()

    @property
    def transform(self):
        """
        Function that is applied on tiles of the data
        before displaying.
        Output should be between 0 and 255,
        and can be R (in that case displayed as gray),
        RG (B is set to 0), RGB or RGBA.
        R: single channel or no channel
        RG: two channels
        RGB/RGBA: three or four channels
        """
        return self.image_viewer.transform

    @property
    def num_images(self):
        return len(self.paths)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        value = max(0, min(self.num_images-1, value))
        if self._index == value:
            return
        self._index = value
        self.load_image()

    @transform.setter
    def transform(self, value):
        self.image_viewer.transform = value

    def on_resize(self, sender, target, data):
        self.image_viewer.update_image()

    def load_image(self):
        path = self.paths[self._index]
        if isinstance(self.image_loader, ImagePreloader):
            self.image_loader.delayed_read(path, lambda result: self.image_viewer.display(result))
        else:
            self.image_viewer.display(self.image_loader(path))

class ViewerWindow(dcg.Window):
    """
    Window instance with a menu to visualize one
    or multiple sequence of data.
    """
    def __init__(self, context, *paths_lists, **kwargs):
        super().__init__(context, **kwargs)
        self.seqs = []
        for paths in paths_lists:
            self.add_sequence(paths)
        self.no_scroll_with_mouse = True
        self.no_scrollbar = True
        # Make the window content use the whole size
        self.theme = \
            dcg.ThemeStyleImGui(context,
                                WindowPadding=(0, 0),
                                WindowBorderSize=0)
        self.handlers += [
            dcg.KeyPressHandler(context, key=dcg.constants.mvKey_Left, callback=self.index_down),
            dcg.KeyPressHandler(context, key=dcg.constants.mvKey_Right, callback=self.index_up)
        ]

    def add_sequence(self, paths):
        """Add a sequence represented by a list of paths"""
        # TODO: put in child window. Subplots
        self.seqs.append(ViewerElement(self.context, paths, parent=self))

    def index_down(self):
        cur_index = max([seq.index for seq in self.seqs])
        cur_index -= 1
        for seq in self.seqs:
            seq.index = cur_index

    def index_up(self):
        cur_index = max([seq.index for seq in self.seqs])
        cur_index += 1
        for seq in self.seqs:
            seq.index = cur_index
