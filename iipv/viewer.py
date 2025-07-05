from collections.abc import Callable, Sequence
from concurrent.futures import Future
import dearcygui as dcg
import math
import numpy as np
from numpy.typing import NDArray
import os
from typing import Any, Self

from .readers import SeriesReader
from .transforms import TransformEditor
from .utils import AtomicCounter, DIVUP, find_all_images, sort_all_files
from .viewer_image import ViewerImage, TransformT

class ViewerImageInPlot(dcg.DrawInPlot):
    """
    A viewer image that automatically update the
    image according to the target axes bounds.
    """
    def __init__(self, context: dcg.Context, **kwargs) -> None:
        super().__init__(context, **kwargs)
        # Create the ViewerImage as a child
        self.viewer = ViewerImage(context, parent=self)
        # Add handler to update image when plot axes change
        assert isinstance(self.parent, dcg.Plot), \
            "ViewerImageInPlot must be used inside a Plot"
        self.parent.handlers += [
            dcg.AxesResizeHandler(context,
                                  axes = self.axes,
                                  callback=self.on_axes_resize)
        ]

    @property 
    def transform(self) -> TransformT:
        return self.viewer.transform

    @transform.setter
    def transform(self, value: TransformT) -> None:
        self.viewer.transform = value

    @property
    def use_pixelwise_transform(self) -> bool:
        return self.viewer.use_pixelwise_transform

    @use_pixelwise_transform.setter
    def use_pixelwise_transform(self, value: bool) -> None:
        self.viewer.use_pixelwise_transform = value

    def dirty(self) -> None:
        """Mark the image as dirty, to make it update completly"""
        return self.viewer.dirty()

    def refresh(self, counter: AtomicCounter | None = None) -> Future[None]:
        """Refresh the displayed image"""
        return self.viewer.refresh(counter=counter)

    def display(self, image: NDArray | Any, scale: float = 1., counter: AtomicCounter | None = None) -> Future[None]:
        """Display a new image"""
        # Get current plot bounds
        return self.viewer.display(image, scale, None, counter)

    def on_axes_resize(self,
                       sender: dcg.AxesResizeHandler,
                       target: Self,
                       data: tuple[tuple[float, float, float], tuple[float, float, float]]) -> None:
        """Handler callback for axes resize events"""
        # Get current plot bounds from resize data
        ((xmin, xmax, _), (ymin, ymax, _)) = data
        self.viewer.set_visible_bounds((xmin, ymin, xmax, ymax))

class ViewerElement(dcg.Plot):
    """
    Sub-window to visualize one sequence of data
    """
    def __init__(self,
                 context: dcg.Context,
                 paths: Sequence[str],
                 num_paths: int,
                 reader: Callable[[str], NDArray | Sequence[NDArray]],
                 index: int = 0,
                 sub_index: int = 0,
                 transform: Callable[[NDArray], NDArray] | None = None,
                 **kwargs) -> None:
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
        # Setup handlers before anything so users can erase them
        with dcg.ConditionalHandler(context) as self.handlers:
            with dcg.HandlerList(context):
                dcg.KeyPressHandler(context, key=dcg.Key.R, callback=self._fit, repeat=False)
                dcg.KeyPressHandler(context, key=dcg.Key.I, callback=self._zoom_in, repeat=False)
                dcg.KeyPressHandler(context, key=dcg.Key.O, callback=self._zoom_out, repeat=False)
            dcg.FocusHandler(context) # Only take keys if the plot has focus

        kwargs.setdefault("equal_aspects", True) # We do really want that for images
        kwargs.setdefault("fit_button", dcg.MouseButton.X1) # We don't want that, so set to an useless button
        kwargs.setdefault("no_frame", True)
        kwargs.setdefault("no_legend", True)
        kwargs.setdefault("no_menus", True)
        kwargs.setdefault("no_mouse_pos", True)
        kwargs.setdefault("no_title", True)
        kwargs.setdefault("width", "fillx") # fill the whole width
        kwargs.setdefault("height", "filly") # fill the whole height
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
        # backup axes config
        self._X1_backup = self.X1
        self._Y1_backup = self.Y1
        # Remove empty borders
        self.theme = dcg.ThemeStyleImPlot(self.context, plot_padding=(0, 0), plot_border_size=0)
        # Custom fields
        self.paths: Sequence[str] = paths
        self.num_images: int = num_paths
        self._index: int = index
        self._sub_index: int = sub_index
        self._image_loader: Callable[[str], NDArray | Sequence[NDArray]] = reader
        self._current_index: int | None = None
        self._current_sub_index: int | None = None
        self._current_image: NDArray | Sequence[NDArray] | None = None
        self._current_subimage: NDArray | None = None
        self._transform: Callable[[NDArray], NDArray] = lambda image: image
        self.image_viewer = ViewerImageInPlot(context, parent=self)
        self.transform = transform

    @property
    def index(self) -> int:
        """Current index of the image in the sequence"""
        return self._index

    @index.setter
    def index(self, value: int) -> None:
        value = max(0, min(self.num_images-1, value))
        if self._index == value:
            return
        self._index = value
        self.image_viewer.display((self._index, self._sub_index))

    def set_index(self, value: int, counter: AtomicCounter | None = None):
        value = max(0, min(self.num_images-1, value))
        if self._index == value:
            return
        self._index = value
        self.image_viewer.display((self._index, self._sub_index), counter=counter)

    @property
    def sub_index(self) -> int:
        """In case the image is a multi-array (video, high dimensional, etc), the index in that array"""
        return self._sub_index

    @sub_index.setter
    def sub_index(self, value: int) -> None:
        if self._sub_index == value:
            return
        self._sub_index = value
        self.image_viewer.display((self._index, self._sub_index))

    def set_sub_index(self, value: int, counter: AtomicCounter | None = None) -> None:
        if self._sub_index == value:
            return
        self._sub_index = value
        self.image_viewer.display((self._index, self._sub_index), counter=counter)

    @property
    def transform(self) -> Callable[[NDArray], NDArray]:
        """
        Function that is applied on tiles of the data before displaying.

        Output should be between 0 and 255,
        and can be R (in that case displayed as gray),
        RG (B is set to 0), RGB or RGBA.
        R: single channel or no channel
        RG: two channels
        RGB/RGBA: three or four channels
        """
        return self._transform

    @transform.setter
    def transform(self, value: Callable[[NDArray], NDArray] | None) -> None:
        if value is None:
            value = lambda image: image
        self._transform = value
        self.image_viewer.transform = self.load_and_transform
        self.image_viewer.display((self._index, self._sub_index))

    def restore_axes(self) -> None:
        """Restore the original axes config"""
        self.X1 = self._X1_backup
        self.Y1 = self._Y1_backup

    def _fit(self, sender: dcg.KeyPressHandler, target: Self, data: Any) -> None:
        """Fit the image to the axes"""
        self.X1.fit()
        self.Y1.fit()

    def _zoom_in(self, sender: dcg.KeyPressHandler, target: Self, data: Any) -> None:
        """Zoom in the image by a factor of 2"""
        center_x = (self.X1.min + self.X1.max) / 2
        center_y = (self.Y1.min + self.Y1.max) / 2
        spread_x = self.X1.max - self.X1.min
        spread_y = self.Y1.max - self.Y1.min
        # TODO clip to a power of two
        self.X1.min = center_x - spread_x / 4
        self.X1.max = center_x + spread_x / 4
        self.Y1.min = center_y - spread_y / 4
        self.Y1.max = center_y + spread_y / 4
        self.image_viewer.refresh()

    def _zoom_out(self, sender: dcg.KeyPressHandler, target: Self, data: Any) -> None:
        """Zoom out the image by a factor of 2"""
        center_x = (self.X1.min + self.X1.max) / 2
        center_y = (self.Y1.min + self.Y1.max) / 2
        spread_x = self.X1.max - self.X1.min
        spread_y = self.Y1.max - self.Y1.min
        self.X1.min = center_x - spread_x * 2
        self.X1.max = center_x + spread_x * 2
        self.Y1.min = center_y - spread_y * 2
        self.Y1.max = center_y + spread_y * 2
        self.image_viewer.refresh()

    def load_and_transform(self, index_pair: tuple[int, int],
                           xm: int, xM: int, ym: int, yM: int) -> NDArray | None:
        """
        Wrapper for the transform that loads dynamically the target image section
        """
        index, sub_index = index_pair
        image: NDArray | Sequence[NDArray]

        if index != self._current_index:
            image = self._image_loader(self.paths[index])
            self._current_image = image
            self._current_index = index
            self._current_sub_index = None
        elif self._current_image is not None:
            image = self._current_image
        else:
            return None

        if sub_index != self._current_sub_index:
            if isinstance(image, np.ndarray):
                sub_index = 0
            else:
                num_subimages: int = len(image)
                sub_index: int = max(0, min(num_subimages-1, sub_index))
                image = image[sub_index]
            self._current_subimage = image
            self._current_sub_index = sub_index

        assert self._current_subimage is not None
        subimage: NDArray = self._current_subimage
        if xm >= subimage.shape[1] or ym >= subimage.shape[0]:
            return None

        xM = min(subimage.shape[1], xM)
        yM = min(subimage.shape[0], yM)
        return self._transform(subimage[ym:yM, xm:xM, ...])


class ViewerWindow(dcg.Window):
    """
    Window instance with a menu to visualize one
    or multiple sequences of data.
    """
    seqs : list[ViewerElement]
    def __init__(self,
                 context: dcg.Context,
                 paths_lists: Sequence[Sequence[str]],
                 num_paths_per_item: Sequence[int],
                 readers_list: Sequence[Callable[[str], NDArray | Sequence[NDArray]]],
                 layout = "grid_col_major",
                 **kwargs) -> None:
        super().__init__(context, **kwargs)
        self.no_scroll_with_mouse = True
        self.no_scrollbar = True

        # layout
        if layout not in ("overlap", "grid_col_major", "grid_row_major", "col", "row"):
            raise ValueError(f"Invalid layout {layout}. Accepted are overlap, grid_row_major, grid_col_major, row and col")
        self._layout = layout
        self._overlap_index = 0 # index of the seq in overlap mode

        # linked axes
        self._axes_indices = []

        # linked indices
        self._linked_indices = []

        # Add menubar with transform editor
        self.bar = dcg.MenuBar(context, parent=self)
        with dcg.Menu(context, label="Edit", parent=self.bar):
            dcg.MenuItem(context, label="Transform Editor", 
                        callback=self.show_transform_editor)

        # Create transform editor
        self.transform_editor = TransformEditor(context, self.on_transform_change)

        # Theme and handlers setup
        self.theme = \
            dcg.ThemeStyleImGui(context,
                                window_padding=(0, 0),
                                window_border_size=0,
                                item_spacing=(1, 1))
        container_key_handlers = \
        [
            dcg.KeyPressHandler(context, key=dcg.Key.LEFTARROW, callback=self.index_down),
            dcg.KeyPressHandler(context, key=dcg.Key.RIGHTARROW, callback=self.index_up),
            dcg.KeyPressHandler(context, key=dcg.Key.DOWNARROW, callback=self.sub_index_down),
            dcg.KeyPressHandler(context, key=dcg.Key.UPARROW, callback=self.sub_index_up),
        ]
        # Only take keys if the child-window container has focus
        self._container_handlers = [
            dcg.ConditionalHandler(context, children=[
                dcg.HandlerList(context, children=container_key_handlers),
                dcg.FocusHandler(context)])
        ]
        # Add support for drag and drop
        # C + drag => copy transform
        with dcg.ConditionalHandler(context) as axes_source:
            dcg.DragDropSourceHandler(context, drag_type="transform")
            dcg.MouseClickHandler(context)
            dcg.MouseOverHandler(context)
            dcg.KeyDownHandler(context, key=dcg.Key.C)
        # V + drag => copy view
        with dcg.ConditionalHandler(context) as view_source:
            dcg.DragDropSourceHandler(context, drag_type="view")
            dcg.MouseClickHandler(context)
            dcg.MouseOverHandler(context)
            dcg.KeyDownHandler(context, key=dcg.Key.V)
        # X + drag => copy player
        with dcg.ConditionalHandler(context) as player_source:
            dcg.DragDropSourceHandler(context, drag_type="player")
            dcg.MouseClickHandler(context)
            dcg.MouseOverHandler(context)
            dcg.KeyDownHandler(context, key=dcg.Key.X)

        self._container_handlers += [
            axes_source,
            view_source,
            player_source,
            dcg.DragDropTargetHandler(context, accepted_types=["item_transform", "item_view", "item_player"], callback=self._handle_drop),
            dcg.DragDropTargetHandler(context, accepted_types=["text", "file"], callback=self._drop_new_seq) # TODO investigate why it doesn't work on the window
        ]

        with dcg.Tooltip(context, parent=self, target=self, condition_from_handler=dcg.DragDropActiveHandler(context, any_target=True, accepted_types="item_transform")):
            dcg.Text(context, value="Select target plot to share axes with")
        with dcg.Tooltip(context, parent=self, target=self, condition_from_handler=dcg.DragDropActiveHandler(context, any_target=True, accepted_types="item_view")):
            dcg.Text(context, value="Select target plot to share view with")
        with dcg.Tooltip(context, parent=self, target=self, condition_from_handler=dcg.DragDropActiveHandler(context, any_target=True, accepted_types="item_player")):
            dcg.Text(context, value="Select target plot to share indices with")

        with dcg.HandlerList(context) as key_handlers:
            # L
            dcg.KeyPressHandler(context, key=dcg.Key.L, callback=self.next_layout)
            # Tab
            dcg.KeyPressHandler(context, key=dcg.Key.TAB, callback=self.next_overlap)

        self.handlers += [
            dcg.ConditionalHandler(context, children=[
                dcg.HandlerList(context, children=[key_handlers]),
                dcg.FocusHandler(context)]),
            #dcg.DragDropTargetHandler(context, accepted_types=["text", "file"], callback=self._drop_new_seq)
        ]

        self._seqs = []
        for (paths, num_paths, readers) in zip(paths_lists, num_paths_per_item, readers_list):
            self.add_sequence(paths, num_paths, readers)

    def _configure_layout(self):
        """Configure the layout of the child-windows"""
        n_seqs = len(self._seqs)
        if n_seqs <= 1:
            self._layout = "overlap"

        if self._layout == "overlap":
            self._overlap_index = max(0, min(n_seqs-1, self._overlap_index))
            for container, _ in self._seqs:
                container.show = False
            if n_seqs >= 1:
                visible_container = self._seqs[self._overlap_index][0]
                visible_container.show = True
                visible_container.x = 0
                visible_container.y = 0
                visible_container.width = "fillx"
                visible_container.height = "filly"
        elif self._layout == "grid_col_major":
            n_rows = int(round(math.sqrt(n_seqs)))#int(math.ceil(math.sqrt(n_seqs)))
            n_cols_ideal = DIVUP(n_seqs, n_rows)
            n_cols_row = [n_cols_ideal for _ in range(n_rows)]
            n_cols_row[-1] = n_seqs - n_cols_ideal * (n_rows-1)
            for i, (container, _) in enumerate(self._seqs):
                row = int(i / n_cols_ideal)
                col = i % n_cols_ideal
                container.show = True
                container.x = f"parent.x1 + {float(col) / n_cols_row[row]} * (fullx - {n_cols_row[row]-1} * theme.item_spacing.x) + {col} * theme.item_spacing.x"
                container.y = f"parent.y1 + {float(row) / n_rows} * (fully - {n_rows-1} * theme.item_spacing.y) + {row} * theme.item_spacing.y"
                container.width = f"(fullx - {n_cols_row[row]-1} * theme.item_spacing.x) / {n_cols_row[row]}"
                container.height = f"(fully - {n_rows-1} * theme.item_spacing.y) / {n_rows}"
        elif self._layout == "grid_row_major":
            n_cols = int(round(math.sqrt(n_seqs))) #int(math.ceil(math.sqrt(n_seqs)))
            n_rows_ideal = DIVUP(n_seqs, n_cols)
            n_rows_col = [n_rows_ideal for _ in range(n_cols)]
            n_rows_col[-1] = n_seqs - n_rows_ideal * (n_cols-1)
            for i, (container, _) in enumerate(self._seqs):
                col = int(i / n_rows_ideal)
                row = i % n_rows_ideal
                container.show = True
                container.x = f"parent.x1 + {float(col) / n_cols} * (fullx - {n_cols-1} * theme.item_spacing.x) + {col} * theme.item_spacing.x"
                container.y = f"parent.y1 + {float(row) / n_rows_col[col]} * (fully - {n_rows_col[col]-1} * theme.item_spacing.y) + {row} * theme.item_spacing.y"
                container.width = f"(fullx - {n_cols-1} * theme.item_spacing.x) / {n_cols}"
                container.height = f"(fully - {n_rows_col[col]-1} * theme.item_spacing.y) / {n_rows_col[col]}"
        elif self._layout == "col":
            for i, (container, _) in enumerate(self._seqs):
                container.show = True
                container.x = f"parent.x1 + {float(i) / n_seqs} * (fullx - {n_seqs-1} * theme.item_spacing.x) + {i} * theme.item_spacing.x"
                container.y = 0
                container.width = f"(fullx - {n_seqs-1} * theme.item_spacing.x) / {n_seqs}"
                container.height = "filly"
                container.no_newline = True
        elif self._layout == "row":
            for i, (container, _) in enumerate(self._seqs):
                container.show = True
                container.x = 0
                container.y = f"parent.y1 + {float(i) / n_seqs} * (fully - {n_seqs-1} * theme.item_spacing.y) + {i} * theme.item_spacing.y"
                container.width = "fillx"
                container.height = f"(fully - {n_seqs-1} * theme.item_spacing.y) / {n_seqs}"
                container.no_newline = False

        # refresh soon
        self.context.viewport.wake(delay=0.016)

    def _validate_linked_axes(self) -> None:
        """Clean _axes_indices to have only self._axes_indices[i] <= i"""
        for i in range(len(self._axes_indices)):
            if self._axes_indices[i] > i:
                seen = set()
                j = self._axes_indices[i]
                while j not in seen:
                    seen.add(j)
                    j = self._axes_indices[j]
                j = min(seen)
                if j < i:
                    self._axes_indices[i] = j
                else:
                    self._axes_indices[i] = i
                    for j in seen:
                        self._axes_indices[j] = i

    def _configure_linked_axes(self) -> None:
        """Configure the linked axes of the plot elements"""
        self._validate_linked_axes()

        assert (len(self._axes_indices) == len(self._seqs))
        for i, j in enumerate(self._axes_indices):
            assert j <= i
            element = self._seqs[i][1]
            if j == i:
                element.restore_axes()
            else:
                target = self._seqs[j][1]
                element.X1 = target.X1
                element.Y1 = target.Y1

    def _get_linked_indices(self, start_index: int) -> set[int]:
        """Retrieve the set of linked indices containing this index"""
        seen = set()
        seen.add(start_index)
        to_treat = set()
        to_treat.add(start_index)
        while len(to_treat) > 0:
            i = to_treat.pop()
            j = self._linked_indices[i]
            # propagate both ways
            if j not in seen:
                to_treat.add(j)
                seen.add(j)
            for j in range(len(self._linked_indices)):
                if self._linked_indices[j] != i:
                    continue
                if j in seen:
                    continue
                to_treat.add(j)
                seen.add(j)
        return seen

    def _handle_drop(self, sender, target, data):
        """Handle a seq dropping on top of another"""
        (payload_type, payload, _, _) = data
        # find source and target containers
        target_index = [i for i, (container, _) in enumerate(self._seqs) if container is target]
        source_index = [i for i, (container, _) in enumerate(self._seqs) if container is payload]
        if len(target_index) == 0 or len(source_index) == 0:
            return # not from this window ?
        i = target_index[0]
        j = source_index[0]
        if payload_type == "item_transform":
            return # TODO
        elif payload_type == "item_view":
            # ensure i <= j
            if j < i:
                i, j = j, i
            # Keep previous links together
            for k in range(len(self._axes_indices)):
                if self._axes_indices[k] == j:
                    self._axes_indices[k] = self._axes_indices[j]
            # Note: i == j is accepted and will reset j
            self._axes_indices[j] = i
            self._configure_linked_axes()
        elif payload_type == "item_player":
            self._linked_indices[j] = i
            self._seqs[j][1].index = self._seqs[i][1].index
            self._seqs[j][1].sub_index = self._seqs[i][1].sub_index

    def _drop_new_seq(self, sender, target, data):
        """Handle an OS seq drop"""
        (payload_type, payload, _, _) = data
        paths = []
        if payload_type == "text":
            paths = payload
            if len(paths) == 1:
                paths = paths[0].split() # TODO handle "\ " for paths with spaces
        elif payload_type == "file":
            paths = payload
        image_paths = []
        for p in paths:
            if "." not in p and os.path.isdir(p): # the . check is to gain speed
                # If a directory, find all images in it
                image_paths.extend(sort_all_files(find_all_images(p)))
            else:
                #assert os.path.isfile(p), f"File {p} does not exist"
                image_paths.append(p)
        self.add_sequence(image_paths, len(image_paths), SeriesReader)

    def show_transform_editor(self) -> None:
        """Show the transform editor popup"""
        self.transform_editor.show()

    def on_transform_change(self, transform: Callable[[NDArray], NDArray]) -> None:
        """Called when the transform function changes"""
        for _, seq in self._seqs:
            seq.transform = transform
            seq.image_viewer.use_pixelwise_transform = self.transform_editor.pixelwise
            seq.image_viewer.refresh()

    def add_sequence(self, paths, num_paths, reader) -> None:
        """Adds a sequence represented by a list of paths"""
        container = dcg.ChildWindow(self.context,
                                    parent=self,
                                    handlers=self._container_handlers,
                                    no_scrollbar=True,
                                    no_scroll_with_mouse=True,
                                    flattened_navigation=True)
        self._seqs.append(
            (container,
             ViewerElement(self.context, paths, num_paths, reader, parent=container)))
        # By default copy the previous axis link
        self._axes_indices.append(0 if len(self._axes_indices) == 0 else self._axes_indices[-1])
        # and the previous index link
        self._linked_indices.append(0 if len(self._linked_indices) == 0 else self._linked_indices[-1])

        # Process the change
        self._configure_layout()
        self._configure_linked_axes()


    def index_down(self, sender, target) -> None:
        # Retrieve target sequence
        target_seq_index = [i for i, (container, _) in enumerate(self._seqs) if container is target]
        if len(target_seq_index) == 0:
            return # container has just been deleted ?
        assert len(target_seq_index) == 1

        # Compute target index
        target_seq = self._seqs[target_seq_index[0]][1]
        cur_index = target_seq.index
        cur_index -= 1

        # Retrieve target sequences
        seqs = self._get_linked_indices(target_seq_index[0])

        # Synchronize actions for better feel, as long it doesn't delay too much
        counter = AtomicCounter(len(seqs), timeout=0.1)

        # Apply index to linked seqs
        for i in seqs:
            seq = self._seqs[i][1]
            seq.set_index(cur_index, counter=counter) # clamping is done by seq

    def index_up(self, sender, target) -> None:
        # Retrieve target sequence
        target_seq_index = [i for i, (container, _) in enumerate(self._seqs) if container is target]
        if len(target_seq_index) == 0:
            return # container has just been deleted ?
        assert len(target_seq_index) == 1

        # Compute target index
        target_seq = self._seqs[target_seq_index[0]][1]
        cur_index = target_seq.index
        cur_index += 1

        # Retrieve target sequences
        seqs = self._get_linked_indices(target_seq_index[0])

        # Synchronize actions for better feel, as long it doesn't delay too much
        counter = AtomicCounter(len(seqs), timeout=0.1)

        # Apply index to linked seqs
        for i in seqs:
            seq = self._seqs[i][1]
            seq.set_index(cur_index, counter=counter) # clamping is done by seq

    def sub_index_down(self, sender, target) -> None:
        # Retrieve target sequence
        target_seq_index = [i for i, (container, _) in enumerate(self._seqs) if container is target]
        if len(target_seq_index) == 0:
            return # container has just been deleted ?
        assert len(target_seq_index) == 1

        # Compute target subindex
        target_seq = self._seqs[target_seq_index[0]][1]
        cur_index = target_seq.sub_index
        cur_index -= 1

        # Retrieve target sequences
        seqs = self._get_linked_indices(target_seq_index[0])

        # Synchronize actions for better feel, as long it doesn't delay too much
        counter = AtomicCounter(len(seqs), timeout=0.1)

        # Apply index to linked seqs
        for i in seqs:
            seq = self._seqs[i][1]
            seq.set_sub_index(cur_index, counter=counter) # clamping is done by seq

    def sub_index_up(self, sender, target) -> None:
        # Retrieve target sequence
        target_seq_index = [i for i, (container, _) in enumerate(self._seqs) if container is target]
        if len(target_seq_index) == 0:
            return # container has just been deleted ?
        assert len(target_seq_index) == 1

        # Compute target subindex
        target_seq = self._seqs[target_seq_index[0]][1]
        cur_index = target_seq.sub_index
        cur_index += 1

        # Retrieve target sequences
        seqs = self._get_linked_indices(target_seq_index[0])

        # Synchronize actions for better feel, as long it doesn't delay too much
        counter = AtomicCounter(len(seqs), timeout=0.1)

        # Apply index to linked seqs
        for i in seqs:
            seq = self._seqs[i][1]
            seq.set_sub_index(cur_index, counter=counter) # clamping is done by seq

    def next_layout(self) -> None:
        """Circle through available layouts"""
        if self._layout == "overlap":
            self._layout = "grid_col_major"
        elif self._layout == "grid_col_major":
            self._layout = "grid_row_major"
        elif self._layout == "grid_row_major":
            self._layout = "col"
        elif self._layout == "col":
            self._layout = "row"
        else:
            self._layout = "overlap"
        self._configure_layout()

    def next_overlap(self) -> None:
        """Circle through sequences in overlap mode"""
        if self._layout != "overlap":
            return
        n_seqs = len(self._seqs)
        if n_seqs > 1:
            self._overlap_index = (self._overlap_index + 1) % n_seqs
        self._configure_layout()

