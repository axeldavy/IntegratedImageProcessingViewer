import json
import os
from pathlib import Path
from appdirs import user_config_dir
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
        self._use_pixelwise_transform = True

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
        
    @use_pixelwise_transform.setter 
    def use_pixelwise_transform(self, value):
        if self._use_pixelwise_transform != value:
            self._use_pixelwise_transform = value
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

            # If using global transform, process entire image at once
            global_processed = None
            if not(self._use_pixelwise_transform) and self.image is not None:
                try:
                    global_processed = self.transform(self.image)
                except Exception:
                    print(traceback.format_exc())

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

                    if global_processed is not None:
                        # Use slice from globally processed image
                        tile = global_processed[ym:yM, xm:xM, ...]
                        prev_content.texture.set_value(tile)
                    else:
                        # Process tile individually
                        tile = self.image[ym:yM, xm:xM, ...]
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
        self.full_refresh = False
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
                full_refresh = self.full_refresh
                self.full_refresh = False
            image = None
            if requested_index != self.background_index or full_refresh:
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

    def refresh_image(self):
        with self.update_mutex:
            self.full_refresh = True
            self.update_request.set()

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
