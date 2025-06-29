from collections.abc import Callable, Sequence
from concurrent.futures import Future
import dearcygui as dcg
import numpy as np
from numpy.typing import NDArray
from typing import Any, Self

from .transforms import TransformEditor
from .utils import AtomicCounter
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
        self.fit_button = dcg.MouseButton.X1 # we don't want that, so set to an useless button
        self.no_title = True
        self.no_mouse_pos = True
        self.equal_aspects = True # We do really want that for images
        self.no_frame = True
        self.no_legend = True
        # fit whole size available
        self.width = -1
        self.height = -1
        # Remove empty borders
        self.theme = dcg.ThemeStyleImPlot(self.context, plot_padding=(0, 0))
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
        return self._index

    @index.setter
    def index(self, value: int) -> None:
        value = max(0, min(self.num_images-1, value))
        if self._index == value:
            return
        self._index = value
        self.image_viewer.display((self._index, self._sub_index))

    @property
    def sub_index(self) -> int:
        return self._sub_index

    @sub_index.setter
    def sub_index(self, value: int) -> None:
        if self._sub_index == value:
            return
        self._sub_index = value
        self.image_viewer.display((self._index, self._sub_index))

    @property
    def transform(self) -> Callable[[NDArray], NDArray]:
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
        return self._transform

    @transform.setter
    def transform(self, value: Callable[[NDArray], NDArray] | None) -> None:
        if value is None:
            value = lambda image: image
        self._transform = value
        self.image_viewer.transform = self.load_and_transform
        self.image_viewer.display((self._index, self._sub_index))

    def load_and_transform(self, index_pair: tuple[int, int],
                           xm: int, xM: int, ym: int, yM: int) -> NDArray | None:
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
    or multiple sequence of data.
    """
    seqs : list[ViewerElement]
    def __init__(self,
                 context: dcg.Context,
                 paths_lists: Sequence[Sequence[str]],
                 num_paths_per_item: Sequence[int],
                 readers_list: Sequence[Callable[[str], NDArray | Sequence[NDArray]]],
                 **kwargs) -> None:
        super().__init__(context, **kwargs)
        self.seqs = []
        for (paths, num_paths, readers) in zip(paths_lists, num_paths_per_item, readers_list):
            self.add_sequence(paths, num_paths, readers)
        self.no_scroll_with_mouse = True
        self.no_scrollbar = True
        
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
                                window_border_size=0)
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

    def show_transform_editor(self) -> None:
        """Show the transform editor popup"""
        self.transform_editor.show()

    def on_transform_change(self, transform: Callable[[NDArray], NDArray]) -> None:
        """Called when the transform function changes"""
        for seq in self.seqs:
            seq.transform = transform
            seq.image_viewer.use_pixelwise_transform = self.transform_editor.pixelwise
            seq.image_viewer.refresh()

    def add_sequence(self, paths, num_paths, reader) -> None:
        """Adds a sequence represented by a list of paths"""
        # TODO: put in child window. Subplots
        self.seqs.append(ViewerElement(self.context, paths, num_paths, reader, parent=self))

    def index_down(self) -> None:
        cur_index: int = max([seq.index for seq in self.seqs])
        cur_index -= 1
        for seq in self.seqs:
            seq.index = cur_index

    def index_up(self) -> None:
        cur_index: int = max([seq.index for seq in self.seqs])
        cur_index += 1
        for seq in self.seqs:
            seq.index = cur_index

    def sub_index_down(self) -> None:
        cur_index: int = max([seq.sub_index for seq in self.seqs])
        cur_index -= 1
        for seq in self.seqs:
            seq.sub_index = cur_index

    def sub_index_up(self) -> None:
        cur_index: int = max([seq.sub_index for seq in self.seqs])
        cur_index += 1
        for seq in self.seqs:
            seq.sub_index = cur_index

