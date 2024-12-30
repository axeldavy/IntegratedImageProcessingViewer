import json
import os
from pathlib import Path
from appdirs import user_config_dir
import dearcygui as dcg
import math
import numpy as np
import threading
import time
import traceback
from .image_preloader import ImagePreloader

from concurrent.futures import ThreadPoolExecutor

def DIVUP(a, b):
    return int(math.ceil( float(a) / float(b)))

class AtomicCounter:
    def __init__(self, value):
        self._value = value
        self._mutex = threading.Lock()
        self._zero = threading.Event()
        self._zero_time = 0.
        if value <= 0:
            self._zero_time = time.monotonic()
            self._zero.set()

    def dec(self):
        with self._mutex:
            self._value -= 1
            if self._value <= 0:
                self._zero_time = time.monotonic()
                self._zero.set()
            return self._value

    def wait_for_zero(self):
        self._zero.wait()

    def timestamp_zero(self):
        return self._zero_time

def error_displaying_wrapper(f, *args, **kwargs):
    try:
        f(*args, **kwargs)
    except Exception:
        print(f"{f} failed with arguments {args}, {kwargs}")
        print(traceback.format_exc())

class ViewerImage(dcg.DrawingList):
    """
    Instance representing the image displayed
    """
    def __init__(self, context,
                 tile_size=[2048, 2048],
                 margin=256, **kwargs):
        super().__init__(context, **kwargs)
        self.image = None
        self.scale = 1.
        self._transform = lambda x: x
        # Cut the image into tiles for display,
        # as we might show bigger images than
        # a texture can support
        self._tile_size = tile_size
        self._margin = margin # spatial margin to load in advance
        self._use_pixelwise_transform = True
        self._global_mutex = threading.RLock()
        self._token_refresh = 0
        self._visible_bounds = [0, 0, 1e9, 1e9]
        # What is currently shown
        self._displayed = dcg.DrawingList(context, parent=self)
        self._up_to_date_tiles_displayed = set()
        self._tiles_displayed = dict()
        # What is being prepared
        self._back = dcg.DrawingList(context, parent=self, show=False)
        self._up_to_date_tiles_back = set()
        self._tiles_back = dict()
        self._image_back = np.zeros([0, 0], dtype=np.int32)
        self._transformed_image_back = None
        self._back_image_height = 0
        self._back_image_width = 0
        # We finish uploading the previous image before
        # starting a new upload
        self._update_queue = ThreadPoolExecutor(max_workers=1)

    @property
    def transform(self):
        """
        Function that is applied to the data before displaying.
        Expected output dtype is uint8 or float32.
        For uint8, the output should be between 0 and 255,
        and for float32 between 0 and 1.
        The dimensions can be HxW or HxWxC.
        The number of channels can be 1, 2, 3 or 4.
        When the number of channels is 1, the image is displayed
        as gray. When the number of channels is 2, the image is
        displayed as RG. When the number of channels is 3 or 4,
        the image is displayed as RGB or RGBA.

        Input of transform:
        If use_pixelwise_transform is False:
            - One input: the image field passed to display(). May be an index.
        If use_pixelwise_transform is True:
            - If the image field passed to display() is an ndarray: the tile to transform.
            - If the image field passed to display is not an array: the image field,
              (for example an index), xm, xM, ym, yM (the coordinates of the tile to transform).
        If the image field is not an array, the size of image is unknown,
        thus transform might be called for undefined regions of the image.
        In that case the corresponding crop of image may be returned, or None,
        if the region if fully out of bounds.
        """
        return self._transform

    def set_transform(self, value, counter=None):
        self._transform = value
        self.dirty()
        return self.refresh(counter=counter)

    @transform.setter
    def transform(self, value):
        self.set_transform(value)

    @property
    def use_pixelwise_transform(self):
        """Controls whether transforms are applied per-tile or globally.
        
        When True (default), transforms are applied to individual tiles, which is 
        more memory efficient but may cause artifacts at tile boundaries for some 
        transforms.
        
        When False, transforms are applied to the entire image at once before
        tiling. This may be slower and use more memory but ensures consistency
        across tile boundaries.
        """
        return self._use_pixelwise_transform

    def set_use_pixelwise_transform(self, value, counter=None):
        if self._use_pixelwise_transform != value:
            self._use_pixelwise_transform = value
            self.dirty()
            return self.refresh(counter=counter)

    @use_pixelwise_transform.setter
    def use_pixelwise_transform(self, value):
        self.set_use_pixelwise_transform(value)

    def set_visible_bounds(self, bounds, counter=None):
        """Set the visible bounds of the image (hint)
        
        bounds: [min_x, min_y, max_x, max_y]
        """
        with self._global_mutex:
            self._visible_bounds = bounds
        return self.refresh(counter=counter)

    def display(self, image, scale=1., bounds=None, counter=None):
        """Display an image, replacing any old one
        
        Scale can be used to scale the image in regard
        to the current axes coordinates.
        For instance this can be used to display
        a lower resolution image, which you will later
        replace with the full image.

        image doesn't have to be a valid ndarray. It can be
        an index used to convey to 'transform' the piece to read.
        """
        self.image = image
        self.scale = scale
        if bounds is not None:
            self._visible_bounds = bounds
        return self.refresh(counter=counter)

    def dirty(self):
        """Mark the image as dirty, to make it update completly"""
        with self._global_mutex:
            self._up_to_date_tiles_back.clear()
            self._up_to_date_tiles_displayed.clear()
            self._transformed_image_back = None
            self.max_w = 1e9 # Maximum estimated bound for the width
            self.max_h = 1e9 # Maximum estimated bound for the height

    def refresh(self, counter=None):
        """Trigger a refresh of the image
        counter: optional AtomicCounter instance to sync
        the refresh with other elements.
        """
        with self._global_mutex:
            self._token_refresh += 1
            return self._update_queue.submit(error_displaying_wrapper,
                                             self._background_update_image,
                                             self._token_refresh,
                                             self.image,
                                             self.scale,
                                             counter)

    def _background_update_image(self, token, image, scale, counter):
        """Update the image displayed if needed"""
        if image is None:
            return

        # Retrieve the parameters
        with self._global_mutex:
            if token != self._token_refresh:
                # Skip outdated refresh requests
                return
            if image is not self._image_back:
                # New image, clear all tile states
                self._up_to_date_tiles_back.clear()
                self._up_to_date_tiles_displayed.clear()
                self._transformed_image_back = None
                self._image_back = image
                self._back_image_height = 1e9 # Maximum estimated bound for the width
                self._back_image_width = 1e9
            transform = self._transform
            use_pixelwise_transform = self._use_pixelwise_transform
            transformed_image = self._transformed_image_back
            margin = self._margin
            tile_size = self._tile_size
            visible_bounds = self._visible_bounds
            min_x = (visible_bounds[0] - margin) / scale
            max_x = (visible_bounds[2] + margin) / scale
            min_y = (visible_bounds[1] - margin) / scale
            max_y = (visible_bounds[3] + margin) / scale

        # If using a global transform, process the entire image at once
        if not(use_pixelwise_transform) and transformed_image is None:
            try:
                transformed_image = transform(image)
                with self._global_mutex:
                    self._transformed_image_back = transformed_image
            except Exception:
                print("Error while running the image transform: ", traceback.format_exc())
            
        if transformed_image is not None:
            max_h, max_w = transformed_image.shape[:2]
        elif hasattr(image, 'shape'):
            # if transformed_image is defined, 
            # prefer to use it instead of the image data.
            max_h, max_w = image.shape[:2]
        else:
            max_w = self._back_image_width
            max_h = self._back_image_height
        if max_h == 0 or max_w == 0:
            return
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(max_w, max_x)
        max_y = min(max_h, max_y)

        num_tiles_w = DIVUP(max_x, tile_size[1])
        num_tiles_h = DIVUP(max_y, tile_size[0])
        if max_x > 1e9:
            # Check if the visible tiles are already up to date to skip computation
            any_action = any(
                (i_h, i_w) not in self._up_to_date_tiles_displayed
                for i_w in range(num_tiles_w)
                for i_h in range(num_tiles_h)
                if not (i_w * tile_size[1] > max_x or (i_w + 1) * tile_size[1] < min_x or
                        i_h * tile_size[0] > max_y or (i_h + 1) * tile_size[0] < min_y)
            )
            if not any_action:
                return

        for i_w in range(num_tiles_w):
            xm = i_w * tile_size[1]
            xM = min(xm + tile_size[1], max_w)
            if xM < min_x:
                # Only update visible parts
                continue
            if xm >= min(max_x, max_w):
                break
            for i_h in range(num_tiles_h):
                ym = i_h * tile_size[0]
                yM = min(ym + tile_size[0], max_h)
                if yM < min_y:
                    # Only update visible parts
                    continue
                if ym >= min(max_y, max_h):
                    break
                if (i_h, i_w) in self._up_to_date_tiles_back:
                    # back already up to date
                    continue
                # Try to reuse existing textures if possible
                draw_tile = self._tiles_back.get((i_h, i_w), None)
                if draw_tile is None or \
                    draw_tile.texture.width != (xM-xm) or \
                    draw_tile.texture.height != (yM-ym):
                    # Initialize the image and its texture
                    draw_tile = \
                        dcg.DrawImage(self.context,
                                      parent=self._back,
                                      texture=\
                                         dcg.Texture(self.context,
                                                     dynamic=True,
                                                     nearest_neighbor_upsampling=True))
                    if (i_h, i_w) in self._tiles_back:
                        self._tiles_back[(i_h, i_w)].detach_item()
                    self._tiles_back[(i_h, i_w)] = draw_tile

                # Configure the DrawImage and the Texture
                draw_tile.pmin = (xm*scale, ym*scale)
                draw_tile.pmax = (xM*scale, yM*scale)
                draw_tile.show = True
                # Already have a texture with up to date content. Take it
                with self._global_mutex:
                    if (i_h, i_w) in self._up_to_date_tiles_displayed:
                        visible_tile = self._tiles_displayed[(i_h, i_w)]
                        draw_tile.texture = visible_tile.texture
                        draw_tile.pmax = visible_tile.pmax
                        draw_tile.show = visible_tile.show
                        self._up_to_date_tiles_back.add((i_h, i_w))
                        continue
                # Dirty tile, update it
                if not(use_pixelwise_transform):
                    # global transform:
                    # Use slice from globally processed image
                    tile = transformed_image[ym:yM, xm:xM, ...]
                    draw_tile.texture.set_value(tile)
                else:
                    # tile processing
                    if hasattr(image, 'shape'):
                        # Process tile individually
                        tile = image[ym:yM, xm:xM, ...]
                        try:
                            processed_tile = transform(tile)
                            draw_tile.texture.set_value(processed_tile)
                        except Exception:
                            print("Error while running the image transform: ", traceback.format_exc())
                            draw_tile.texture.set_value(np.zeros((yM-ym, xM-xm, 4), dtype=np.uint8))
                    else:
                        # Fetch the tile
                        try:
                            tile = transform(image, xm, xM, ym, yM)
                            if tile is not None:
                                if tile.shape[1] < tile_size[1]:
                                    max_w = min(max_w, xm + tile.shape[1])
                                    draw_tile.pmax = (max_w*scale, draw_tile.pmax[1])
                                if tile.shape[0] < tile_size[0]:
                                    max_h = min(max_h, ym + tile.shape[0])
                                    draw_tile.pmax = (draw_tile.pmax[0], max_h*scale)
                                draw_tile.texture.set_value(tile)
                            else:
                                # Completly outside the image
                                max_h = min(max_h, ym)
                                max_w = min(max_w, xm)
                                # Set transparent content
                                #draw_tile.texture.set_value(np.zeros((yM-ym, xM-xm, 4), dtype=np.uint8))
                                draw_tile.show = False
                        except Exception:
                            print("Error while running the image transform: ", traceback.format_exc())
                            #draw_tile.texture.set_value(np.zeros((yM-ym, xM-xm, 4), dtype=np.uint8))
                            draw_tile.show = False
                self._up_to_date_tiles_back.add((i_h, i_w))
        self._back_image_width = max_w
        self._back_image_height = max_h
        # Hide out of bound tiles
        out_of_date = [key for key in self._tiles_back.keys() if key not in self._up_to_date_tiles_back]
        for key in out_of_date:
            self._tiles_back[key].show = False
        if isinstance(counter, AtomicCounter):
            counter.dec()
            # Wait all updates are done
            counter.wait_for_zero()
        self._swap_back()

    def _swap_back(self):
        """Swap the back and displayed images"""
        with self._global_mutex:
            tmp = self._displayed
            self._displayed = self._back
            self._back = tmp
            tmp = self._tiles_displayed
            self._tiles_displayed = self._tiles_back
            self._tiles_back = tmp
            tmp = self._up_to_date_tiles_displayed
            self._up_to_date_tiles_displayed = self._up_to_date_tiles_back
            self._up_to_date_tiles_back = tmp
            # Swap atomically
            with self.mutex:
                self._displayed.show = True
                self._back.show = False
        # Indicate content has changed (wait_for_input)
        self.context.viewport.wake()

class ViewerImageInPlot(dcg.DrawInPlot):
    """
    A viewer image that automatically update the
    image according to the target axes bounds.
    """
    def __init__(self, context, **kwargs):
        super().__init__(context, **kwargs)
        # Create the ViewerImage as a child
        self.viewer = ViewerImage(context, parent=self)
        # Add handler to update image when plot axes change
        self.parent.handlers += [
            dcg.AxesResizeHandler(context,
                                  axes = self.axes,
                                  callback=self.on_axes_resize)
        ]

    @property 
    def transform(self):
        return self.viewer.transform

    @transform.setter
    def transform(self, value):
        self.viewer.transform = value

    @property
    def use_pixelwise_transform(self):
        return self.viewer.use_pixelwise_transform

    @use_pixelwise_transform.setter
    def use_pixelwise_transform(self, value):
        self.viewer.use_pixelwise_transform = value

    def dirty(self):
        """Mark the image as dirty, to make it update completly"""
        return self.viewer.dirty()

    def refresh(self, counter=None):
        """Refresh the displayed image"""
        return self.viewer.refresh(counter=counter)

    def display(self, image, scale=1., counter=None):
        """Display a new image"""
        # Get current plot bounds
        return self.viewer.display(image, scale, None, counter)

    def on_axes_resize(self, sender, target, data):
        """Handler callback for axes resize events"""
        # Get current plot bounds from resize data
        ((xmin, xmax, _), (ymin, ymax, _)) = data
        self.viewer.set_visible_bounds([xmin, ymin, xmax, ymax])

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
        # Remove empty borders
        self.theme = dcg.ThemeStyleImPlot(self.context, PlotPadding=(0, 0))
        # Custom fields
        self.paths = paths
        self.num_images = num_paths
        self._index = index
        self._sub_index = sub_index
        self.image_loader = reader
        self._current_index = None
        self._current_sub_index = None
        self._current_image = None
        self._current_subimage = None
        self.image_viewer = ViewerImageInPlot(context, parent=self)
        self.transform = transform

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
        return self.image_transform

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        value = max(0, min(self.num_images-1, value))
        if self._index == value:
            return
        self._index = value
        self.image_viewer.display((self._index, self._sub_index))

    @property
    def sub_index(self):
        return self._sub_index

    @sub_index.setter
    def sub_index(self, value):
        if self._sub_index == value:
            return
        self._sub_index = value
        self.image_viewer.display((self._index, self._sub_index))

    def load_and_transform(self, index_pair, xm, xM, ym, yM):
        index, sub_index = index_pair
        if index != self._current_index:
            image = self.image_loader(self.paths[index])
            self._current_image = image
            self._current_index = index
            self._current_sub_index = None
        else:
            image = self._current_image
        if sub_index != self._current_sub_index:
            if isinstance(image, np.ndarray):
                sub_index = 0
            else:
                num_subimages = len(image)
                sub_index = max(0, min(num_subimages-1, sub_index))
                image = image[sub_index]
            self._current_subimage = image
            self._current_sub_index = sub_index
        image = self._current_subimage
        if xm >= image.shape[1] or ym >= image.shape[0]:
            return None
        xM = min(image.shape[1], xM)
        yM = min(image.shape[0], yM)
        image = image[ym:yM, xm:xM, ...]
        if self.image_transform is not None:
            return self.image_transform(image)
        return image

    @transform.setter
    def transform(self, value):
        self.image_transform = value
        self.image_viewer.transform = self.load_and_transform
        self.image_viewer.display((self._index, self._sub_index))


class ViewerWindow(dcg.Window):
    """
    Window instance with a menu to visualize one
    or multiple sequence of data.
    """
    seqs : list[ViewerElement]
    def __init__(self, context, paths_lists, num_paths_per_item, readers, **kwargs):
        super().__init__(context, **kwargs)
        self.seqs = []
        for paths, num_paths, readers in zip(paths_lists, num_paths_per_item, readers):
            self.add_sequence(paths, num_paths, readers)
        self.no_scroll_with_mouse = True
        self.no_scrollbar = True
        
        # Add menubar with transform editor
        self.bar = dcg.MenuBar(context, parent=self)
        with dcg.Menu(context, label="Edit", parent=self.bar):
            dcg.MenuItem(context, label="Transform Editor", 
                        callback=self.show_transform_editor)

        # Add transform editor popup
        self.transform_popup = dcg.Window(context, label="Transform Editor", 
                                        show=False, modal=True,
                                        no_open_over_existing_popup=False,
                                        min_size=(600, 400), autosize=True)

        # Add transform history
        self.transform_history = TransformHistory()
        
        # Add editor components with navigation
        with dcg.HorizontalLayout(context, parent=self.transform_popup):
            dcg.Button(context, arrow=True, direction=dcg.ButtonDirection.LEFT, callback=self.prev_transform)
            dcg.Button(context, arrow=True, direction=dcg.ButtonDirection.RIGHT, callback=self.next_transform)
            dcg.Button(context, label="Reset", callback=self.reset_transform)
        
        
        # Add editor components
        self.transform_editor = dcg.InputText(context, 
            parent=self.transform_popup,
            multiline=True,
            width=-1, height=320,
            value=self.transform_history.get_current() or """def transform(image):
    # Transform the image here
    # image: numpy array of shape (H,W) or (H,W,C)
    # return: uint8 array with 1-4 channels
    return image""")

        if not(self.transform_history.get_current()):
            self.transform_history.add_transform(self.transform_editor.value)
            
        self.pixelwise_transform = dcg.Checkbox(context,
            label="Pixelwise Transform",
            value=True,
            parent=self.transform_popup)
            
        with dcg.HorizontalLayout(context, parent=self.transform_popup):
            dcg.Button(context, label="Apply", callback=self.apply_transform)
            dcg.Button(context, label="Cancel", 
                      callback=lambda: setattr(self.transform_popup, 'show', False))

        # Theme and handlers setup
        self.theme = \
            dcg.ThemeStyleImGui(context,
                                WindowPadding=(0, 0),
                                WindowBorderSize=0)
        key_handlers = \
        [
            dcg.KeyPressHandler(context, key=dcg.Key.LEFTARROW, callback=self.index_down),
            dcg.KeyPressHandler(context, key=dcg.Key.RIGHTARROW, callback=self.index_up),
            dcg.KeyPressHandler(context, key=dcg.Key.DOWNARROW, callback=self.sub_index_down),
            dcg.KeyPressHandler(context, key=dcg.Key.UPARROW, callback=self.sub_index_up)
        ]
        # Only take keys if the window has focus
        self.handlers += [
            dcg.ConditionalHandler(context, children=[
                dcg.HandlerList(context, children=key_handlers),
                dcg.FocusHandler(context)])
        ]

    def show_transform_editor(self):
        """Show the transform editor popup"""
        self.transform_popup.show = True

    def prev_transform(self):
        """Load previous transform from history"""
        code = self.transform_history.go_prev()
        if code:
            self.transform_editor.value = code

    def next_transform(self):
        """Load next transform from history"""
        code = self.transform_history.go_next()
        if code:
            self.transform_editor.value = code

    def reset_transform(self):
        """Reset transform to default"""
        self.transform_editor.value = self.transform_history.reset()

    def apply_transform(self):
        """Validate and apply the transform function"""
        try:
            # Create a namespace to execute the code
            namespace = {}
            
            # Add required imports
            namespace.update({
                'np': np,
                'math': math
            })
            
            # Execute the transform code
            exec(self.transform_editor.value, namespace)
            
            # Validate the transform function exists
            if 'transform' not in namespace:
                raise ValueError("No transform function defined")
                
            # Try to apply it to validate it works
            test_arr = np.zeros((10,10), dtype=np.uint8)
            transform = namespace['transform']
            result = transform(test_arr)
            
            if not isinstance(result, np.ndarray):
                raise ValueError("Transform must return a numpy array")
            
            if result.dtype not in (np.uint8, np.float32, np.float64):
                raise ValueError("Transform must return uint8 or float32 array")
            if result.dtype in (np.float32, np.float64):
                def normalize(x, f=transform):
                    return (f(x) / 255).astype(np.float32)
                transform = normalize

            if len(result.shape) > 3 or (len(result.shape) == 3 and result.shape[2] > 4):
                raise ValueError("Transform must return 1-4 channel image")

            # If validation passed, save to history
            self.transform_history.add_transform(self.transform_editor.value)

            # Apply to all sequences
            for seq in self.seqs:
                seq.transform = transform
                seq.image_viewer.use_pixelwise_transform = self.pixelwise_transform.value
                seq.refresh_image()

            # Remove any previous error message
            if isinstance(self.transform_popup.children[-1], dcg.Text):
                self.transform_popup.children[-1].detach_item()
                
            # Close popup
            self.transform_popup.show = False
            
        except Exception as e:
            # Remove any previous error message
            if isinstance(self.transform_popup.children[-1], dcg.Text):
                self.transform_popup.children[-1].detach_item()
            with self.transform_popup:
                dcg.Text(self.context, value=f"{str(e)}, {traceback.format_exc()}")

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

class TransformHistory:
    """Manages a history of image transforms with file persistence"""
    
    def __init__(self, config_dir=None):
        if config_dir is None:
            config_dir = user_config_dir("iipv")
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "transforms.json"
        self.transforms = []
        self.current_index = -1
        self._load_transforms()

    def _load_transforms(self):
        """Load transforms from config file"""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            if self.config_file.exists():
                with open(self.config_file) as f:
                    self.transforms = json.load(f)
                self.current_index = len(self.transforms) - 1
        except Exception as e:
            print(f"Error loading transforms: {e}")
            self.transforms = []
            self.current_index = -1

    def _save_transforms(self):
        """Save transforms to config file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.transforms, f, indent=2)
        except Exception as e:
            print(f"Error saving transforms: {e}")

    def add_transform(self, transform_code):
        """Add a new transform to history"""
        # Don't add if same as current
        if self.transforms and transform_code == self.transforms[-1]:
            return
        # Remove any past duplicate
        if transform_code in self.transforms:
            self.transforms.remove(transform_code)

        self.transforms.append(transform_code)
        self.current_index = len(self.transforms) - 1
        self._save_transforms()

    def get_current(self):
        """Get current transform code"""
        if 0 <= self.current_index < len(self.transforms):
            return self.transforms[self.current_index]
        return None

    def go_prev(self):
        """Go to previous transform in history"""
        if self.current_index > 0:
            self.current_index -= 1
            return self.get_current()
        return None

    def go_next(self):
        """Go to next transform in history"""
        if self.current_index < len(self.transforms) - 1:
            self.current_index += 1
            return self.get_current()
        return None

    def reset(self):
        """Reset transform history to initial empty state"""
        self.transforms = []
        self.current_index = -1
        self._save_transforms()
        return """def transform(image):
    # Transform the image here
    # image: numpy array of shape (H,W) or (H,W,C)
    # return: uint8 array with 1-4 channels
    return image"""
