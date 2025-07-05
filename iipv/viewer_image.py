from ast import Call
from collections.abc import Callable, Sequence
from concurrent.futures import Future
import dearcygui as dcg
import numpy as np
from numpy.typing import NDArray
import threading
import traceback
from typing import Any, cast, TypeAlias

from .utils import DIVUP, AtomicCounter, DebugThreadPoolExecutor

TransformT: TypeAlias = Callable[[NDArray], NDArray] | Callable[[Any, int, int, int, int], NDArray | None]

class ViewerImage(dcg.DrawingList):
    """
    Instance representing the image displayed
    """
    def __init__(self,
                 context: dcg.Context,
                 tile_size: tuple[int, int] = (2048, 2048),
                 margin: float = 256.,
                 **kwargs) -> None:
        super().__init__(context, **kwargs)
        self._image: NDArray | Any | None = None
        self._scale: float = 1.
        self._transform: TransformT = lambda x: x
        # Cut the image into tiles for display,
        # as we might show bigger images than
        # a texture can support
        self._tile_size: tuple[int, int] = tile_size
        self._margin: float = margin # spatial margin to load in advance
        self._use_pixelwise_transform: bool = True
        self._global_mutex = threading.RLock()
        self._token_refresh: int = 0
        self._visible_bounds: tuple[float, float, float, float] = (0., 0., 1e9, 1e9)
        # What is currently shown
        self._displayed = dcg.DrawingList(context, parent=self)
        self._up_to_date_tiles_displayed: set[tuple[int, int]] = set()
        self._tiles_displayed: dict[tuple[int, int], dcg.DrawImage] = dict()
        # What is being prepared
        self._back = dcg.DrawingList(context, parent=self, show=False)
        self._up_to_date_tiles_back: set[tuple[int, int]] = set()
        self._tiles_back: dict[tuple[int, int], dcg.DrawImage] = dict()
        self._image_back: NDArray = np.zeros([0, 0], dtype=np.int32)
        self._transformed_image_back: NDArray | None = None
        self._back_image_height: int = 0
        self._back_image_width: int = 0
        # We finish uploading the previous image before
        # starting a new upload
        self._update_queue = DebugThreadPoolExecutor(max_workers=1)

    @property
    def transform(self) -> TransformT:
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

    def set_transform(self, value: TransformT, counter: AtomicCounter | None = None) -> Future[None]:
        self._transform = value
        self.dirty()
        return self.refresh(counter=counter)

    @transform.setter
    def transform(self, value: TransformT) -> None:
        self.set_transform(value)

    @property
    def use_pixelwise_transform(self) -> bool:
        """Controls whether transforms are applied per-tile or globally.
        
        When True (default), transforms are applied to individual tiles, which is 
        more memory efficient but may cause artifacts at tile boundaries for some 
        transforms.
        
        When False, transforms are applied to the entire image at once before
        tiling. This may be slower and use more memory but ensures consistency
        across tile boundaries.
        """
        return self._use_pixelwise_transform

    def set_use_pixelwise_transform(self, value: bool, counter: AtomicCounter | None = None) -> Future[None] | None:
        if self._use_pixelwise_transform != value:
            self._use_pixelwise_transform = value
            self.dirty()
            return self.refresh(counter=counter)

    @use_pixelwise_transform.setter
    def use_pixelwise_transform(self, value) -> None:
        self.set_use_pixelwise_transform(value)

    def set_visible_bounds(self,
                           bounds: tuple[float, float, float, float],
                           counter: AtomicCounter | None = None) -> Future[None]:
        """Set the visible bounds of the image (hint)
        
        bounds: [min_x, min_y, max_x, max_y]
        """
        with self._global_mutex:
            self._visible_bounds = bounds
        return self.refresh(counter=counter)

    def display(self,
                image: NDArray | Any,
                scale: float = 1.,
                bounds: tuple[float, float, float, float] | None = None,
                counter: AtomicCounter | None = None) -> Future[None]:
        """Display an image, replacing any old one
        
        Scale can be used to scale the image in regard
        to the current axes coordinates.
        For instance this can be used to display
        a lower resolution image, which you will later
        replace with the full image.

        image doesn't have to be a valid ndarray. It can be
        an index used to convey to 'transform' the piece to read.
        """
        self._image = image
        self._scale = scale
        if bounds is not None:
            self._visible_bounds = bounds
        return self.refresh(counter=counter)

    def dirty(self) -> None:
        """Mark the image as dirty, to make it update completly"""
        with self._global_mutex:
            self._up_to_date_tiles_back.clear()
            self._up_to_date_tiles_displayed.clear()
            self._transformed_image_back = None
            self.max_w = 1e9 # Maximum estimated bound for the width
            self.max_h = 1e9 # Maximum estimated bound for the height

    def refresh(self, counter: AtomicCounter | None = None) -> Future[None]:
        """Trigger a refresh of the image
        counter: optional AtomicCounter instance to sync
        the refresh with other elements.
        """
        with self._global_mutex:
            self._token_refresh += 1
            return self._update_queue.submit(self._background_update_image,
                                             self._token_refresh,
                                             self._image,
                                             self._scale,
                                             counter)

    def _background_update_image(self,
                                 token: int,
                                 image: NDArray | None,
                                 scale: float,
                                 counter: AtomicCounter | None) -> None:
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
                self._back_image_height = 1_000_000_000 # Maximum estimated bound for the height
                self._back_image_width = 1_000_000_000 # Maximum estimated bound for the width
            transform = self._transform
            use_pixelwise_transform: bool = self._use_pixelwise_transform
            transformed_image: NDArray | None = self._transformed_image_back
            margin: float = self._margin
            tile_size: tuple[int, int] = self._tile_size
            visible_bounds: tuple[float, float, float, float] = self._visible_bounds
            min_x: float = (visible_bounds[0] - margin) / scale
            max_x: float = (visible_bounds[2] + margin) / scale
            min_y: float = (visible_bounds[1] - margin) / scale
            max_y: float = (visible_bounds[3] + margin) / scale

        # If using a global transform, process the entire image at once
        if not(use_pixelwise_transform) and transformed_image is None:
            try:
                if isinstance(image, np.ndarray):
                    image = cast(NDArray, image)
                    transform = cast(Callable[[NDArray], NDArray], transform)
                    transformed_image = transform(image)
                else:
                    # If image is not an ndarray, we assume it is an index
                    # and we call the transform with the image and the tile coordinates
                    transform = cast(Callable[[Any, int, int, int, int], NDArray | None], transform)
                    transformed_image = transform(image, 0, 1_000_000, 0, 1_000_000)
                with self._global_mutex:
                    self._transformed_image_back = transformed_image
            except Exception:
                print("Error while running the image transform: ", traceback.format_exc())

        max_h: int
        max_w: int
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
            any_action: bool = any(
                (i_h, i_w) not in self._up_to_date_tiles_displayed
                for i_w in range(num_tiles_w)
                for i_h in range(num_tiles_h)
                if not (i_w * tile_size[1] > max_x or (i_w + 1) * tile_size[1] < min_x or
                        i_h * tile_size[0] > max_y or (i_h + 1) * tile_size[0] < min_y)
            )
            if not any_action:
                return

        for i_w in range(num_tiles_w):
            xm: int = i_w * tile_size[1]
            xM: int = min(xm + tile_size[1], max_w)
            if xM < min_x:
                # Only update visible parts
                continue
            if xm >= min(max_x, max_w):
                break
            for i_h in range(num_tiles_h):
                ym: int = i_h * tile_size[0]
                yM: int = min(ym + tile_size[0], max_h)
                if yM < min_y:
                    # Only update visible parts
                    continue
                if ym >= min(max_y, max_h):
                    break
                if (i_h, i_w) in self._up_to_date_tiles_back:
                    # back already up to date
                    continue
                # Try to reuse existing textures if possible
                draw_tile: dcg.DrawImage | None = self._tiles_back.get((i_h, i_w), None)
                if (draw_tile is None
                    or draw_tile.texture.width != (xM-xm) # type:ignore
                    or draw_tile.texture.height != (yM-ym)): # type: ignore
                    # Initialize the image and its texture
                    draw_tile = \
                        dcg.DrawImage(self.context,
                                      parent=self._back,
                                      texture=\
                                         dcg.Texture(self.context,
                                                     hint_dynamic=True,
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
                        visible_tile: dcg.DrawImage = self._tiles_displayed[(i_h, i_w)]
                        draw_tile.texture = visible_tile.texture
                        draw_tile.pmax = visible_tile.pmax
                        draw_tile.show = visible_tile.show
                        self._up_to_date_tiles_back.add((i_h, i_w))
                        continue
                # Dirty tile, update it
                if not(use_pixelwise_transform):
                    # global transform:
                    # Use slice from globally processed image
                    assert transformed_image is not None
                    tile = transformed_image[ym:yM, xm:xM, ...]
                    draw_tile.texture.set_value(tile) # type: ignore
                else:
                    # tile processing
                    if hasattr(image, 'shape'):
                        # Process tile individually
                        tile = image[ym:yM, xm:xM, ...]
                        try:
                            transform = cast(Callable[[NDArray], NDArray], transform)
                            processed_tile = transform(tile)
                            draw_tile.texture.set_value(processed_tile) # type: ignore
                        except Exception:
                            print("Error while running the image transform: ", traceback.format_exc())
                            draw_tile.texture.set_value(np.zeros((yM-ym, xM-xm, 4), dtype=np.uint8)) # type: ignore
                    else:
                        # Fetch the tile
                        try:
                            transform = cast(Callable[[Any, int, int, int, int], NDArray | None], transform)
                            tile: NDArray | None = transform(image, xm, xM, ym, yM)
                            if tile is not None:
                                if tile.shape[1] < tile_size[1]:
                                    max_w = min(max_w, xm + tile.shape[1])
                                    draw_tile.pmax = (max_w*scale, draw_tile.pmax[1])
                                if tile.shape[0] < tile_size[0]:
                                    max_h = min(max_h, ym + tile.shape[0])
                                    draw_tile.pmax = (draw_tile.pmax[0], max_h*scale)
                                draw_tile.texture.set_value(tile) # type: ignore
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
        out_of_date: list[tuple[int, int]] = \
            [key for key in self._tiles_back.keys() if key not in self._up_to_date_tiles_back]
        for key in out_of_date:
            self._tiles_back[key].show = False
        if isinstance(counter, AtomicCounter):
            counter.dec()
            # Wait all updates are done
            counter.wait_for_zero()
        self._swap_back()

    def _swap_back(self) -> None:
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
