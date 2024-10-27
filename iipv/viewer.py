import dearcygui as dcg
import math
import numpy as np
import threading
import traceback
from .image_preloader import ImagePreloader

def DIVUP(a, b):
    return int(math.ceil( float(a) / float(b)))

class ViewerImage(dcg.DrawInPlot):
    """
    Instance representing the image displayed
    """
    def __init__(self, context, tile_size=[1024, 1024], margin=256, **kwargs):
        super().__init__(context, **kwargs)
        self.image = np.zeros([0, 0], dtype=np.int32)
        self.scale = 1.
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

    def display(self, image, scale=1.):
        """Display an image, replacing any old one
        
        Scale can be used to scale the image in regard
        to the current axes coordinates.
        For instance this can be used to display
        a lower resolution image, which you will later
        replace with the full image.
        """
        with self.update_mutex:
            if image is not self.image:
                self.up_to_date_tiles_front.clear()
                self.up_to_date_tiles_back.clear()
            self.image = image
            self.scale = scale
            self.update_image()

    def update_image(self):
        """Update the image displayed if needed"""
        with self.update_mutex:
            h = self.image.shape[0]
            w = self.image.shape[1]
            scale = self.scale
            if w == 0 or h == 0:
                return
            tiles_w = DIVUP(w, self.tile_size[1])
            tiles_h = DIVUP(h, self.tile_size[0])
            # TODO: configurable axes
            X = self.parent.X1
            Y = self.parent.Y1
            min_x = (X.min - self.margin) / scale
            max_x = (X.max + self.margin) / scale
            min_y = (Y.min - self.margin) / scale
            max_y = (Y.max + self.margin) / scale
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
                                                     pmin=(xm*scale, ym*scale),
                                                     pmax=(xM*scale, yM*scale))
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
                        prev_content.pmin = (xm*scale, ym*scale)
                        prev_content.pmax = (xM*scale, yM*scale)
                    # Already have a texture with up to date content. Take it
                    if (i_h, i_w) in self.up_to_date_tiles_front:
                        switched_textures[(i_h, i_w)] = prev_content.texture
                        prev_content.texture = self.tiles_front[(i_h, i_w)].texture
                        self.up_to_date_tiles_back.add((i_h, i_w))
                        continue
                    tile = self.image[ym:yM, xm:xM, ...]
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
    def __init__(self, context, paths, num_paths, reader, index=0, sub_index=0, transform=None, **kwargs):
        """
        paths: iterable (for example a list) of parameters to pass to the reader function
        
        num_paths: number of paths (len(paths) if paths is a list)
        
        reader: function to read the images
        reader(paths[0]) reads image 0
        reader(paths[1]) reads image 1
        etc.
        The returned image must be an ndarray of at least 2 dimensions.
        Alternatively if paths[i] contains several sub-images, an
        array (list, array of objects, etc) can be passed.
        
        transform: if not None, transform will be called on tiles
        of the image.
        The image tile after transform must be one of the following formats:
        - 1, 2, 3 or 4 channels
        - uint8 or float32. If float32, the data must be normalized between 0 and 1.
        """
        super().__init__(context, **kwargs)
        self.paths = paths
        self.num_images = num_paths
        self._index = index
        self._sub_index = sub_index
        self.image_loader = reader
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
        self.update_thread = threading.Thread(target=self.background_update, args=(), daemon=True)
        self.update_request = threading.Event()
        self.update_mutex = threading.Lock()
        self.background_index = -1
        self.background_subindex = -1
        self.background_current_image = None
        self.should_refresh = False
        self.update_thread.start()
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
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        value = max(0, min(self.num_images-1, value))
        if self._index == value:
            return
        self._index = value
        self.load_image()

    @property
    def sub_index(self):
        return self._sub_index

    @sub_index.setter
    def sub_index(self, value):
        if self._sub_index == value:
            return
        self._sub_index = value
        self.load_image()

    @transform.setter
    def transform(self, value):
        self.image_viewer.transform = value

    def background_update(self):
        """
        Since reading the image and loading the texture can be slow,
        do it in a background thread.
        """
        while True:
            self.update_request.wait()
            with self.update_mutex:
                self.update_request.clear()
                requested_index = self._index
                requested_sub_index = self._sub_index
                refresh_requested = self.should_refresh
                self.should_refresh = False
            full_refresh = False
            image = None
            if requested_index != self.background_index:
                path = self.paths[requested_index]
                self.background_index = requested_index
                self.background_current_image = self.image_loader(path)
                full_refresh = True
            if isinstance(self.background_current_image, np.ndarray):
                requested_sub_index = 0
                image = self.background_current_image
            else:
                num_subimages = len(self.background_current_image)
                requested_sub_index = max(0, min(num_subimages-1, requested_sub_index))
                if requested_sub_index != self.background_subindex:
                    image = self.background_current_image[requested_sub_index]
                    full_refresh = True
            if full_refresh:
                self.image_viewer.display(image)
            elif refresh_requested:
                self.image_viewer.update_image()

    def on_resize(self, sender, target, data):
        with self.update_mutex:
            self.should_refresh = True
            self.update_request.set()


    def load_image(self):
        with self.update_mutex:
            self.update_request.set()
        #path = self.paths[self._index]
        #if isinstance(self.image_loader, ImagePreloader):
        #    self.image_loader.delayed_read(path, lambda result: self.image_viewer.display(result))
        #else:
        #    self.image_viewer.display(self.image_loader(path))

class ViewerWindow(dcg.Window):
    """
    Window instance with a menu to visualize one
    or multiple sequence of data.
    """
    def __init__(self, context, paths_lists, num_paths_per_item, readers, **kwargs):
        super().__init__(context, **kwargs)
        self.seqs = []
        for paths, num_paths, readers in zip(paths_lists, num_paths_per_item, readers):
            self.add_sequence(paths, num_paths, readers)
        self.no_scroll_with_mouse = True
        self.no_scrollbar = True
        # Make the window content use the whole size
        self.theme = \
            dcg.ThemeStyleImGui(context,
                                WindowPadding=(0, 0),
                                WindowBorderSize=0)
        self.handlers += [
            dcg.KeyPressHandler(context, key=dcg.constants.mvKey_Left, callback=self.index_down),
            dcg.KeyPressHandler(context, key=dcg.constants.mvKey_Right, callback=self.index_up),
            dcg.KeyPressHandler(context, key=dcg.constants.mvKey_Down, callback=self.sub_index_down),
            dcg.KeyPressHandler(context, key=dcg.constants.mvKey_Up, callback=self.sub_index_up)
        ]

    def add_sequence(self, paths, num_paths, reader):
        """Adds a sequence represented by a list of paths"""
        # TODO: put in child window. Subplots
        self.seqs.append(ViewerElement(self.context, paths, num_paths, reader, parent=self))

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

    def sub_index_down(self):
        cur_index = max([seq.sub_index for seq in self.seqs])
        cur_index -= 1
        for seq in self.seqs:
            seq.sub_index = cur_index

    def sub_index_up(self):
        cur_index = max([seq.sub_index for seq in self.seqs])
        cur_index += 1
        for seq in self.seqs:
            seq.sub_index = cur_index
