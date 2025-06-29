
from appdirs import user_config_dir
from collections.abc import Callable
import dearcygui as dcg
import json
import math
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import traceback


class TransformHistory:
    """Manages a history of image transforms with file persistence"""
    
    def __init__(self, config_dir: str | None = None) -> None:
        if config_dir is None:
            config_dir = user_config_dir("iipv")
        self._config_dir = Path(config_dir)
        self._config_file: Path = self._config_dir / "transforms.json"
        self._transforms: list[str] = []
        self._current_index: int = -1
        self._load_transforms()

    def _load_transforms(self) -> None:
        """Load transforms from config file"""
        try:
            self._config_dir.mkdir(parents=True, exist_ok=True)
            if self._config_file.exists():
                with open(self._config_file) as f:
                    self._transforms = json.load(f)
                self._current_index = len(self._transforms) - 1
        except Exception as e:
            print(f"Error loading transforms: {e}")
            self._transforms = []
            self._current_index = -1

    def _save_transforms(self) -> None:
        """Save transforms to config file"""
        try:
            with open(self._config_file, 'w') as f:
                json.dump(self._transforms, f, indent=2)
        except Exception as e:
            print(f"Error saving transforms: {e}")

    def add_transform(self, transform_code) -> None:
        """Add a new transform to history"""
        # Don't add if same as current
        if self._transforms and transform_code == self._transforms[-1]:
            return
        # Remove any past duplicate
        if transform_code in self._transforms:
            self._transforms.remove(transform_code)

        self._transforms.append(transform_code)
        self._current_index = len(self._transforms) - 1
        self._save_transforms()

    def get_current(self) -> str | None:
        """Get current transform code"""
        if 0 <= self._current_index < len(self._transforms):
            return self._transforms[self._current_index]
        return None

    def go_prev(self) -> str | None:
        """Go to previous transform in history"""
        if self._current_index > 0:
            self._current_index -= 1
            return self.get_current()
        return None

    def go_next(self) -> str | None:
        """Go to next transform in history"""
        if self._current_index < len(self._transforms) - 1:
            self._current_index += 1
            return self.get_current()
        return None

    def reset(self) -> str:
        """Reset transform history to initial empty state"""
        self._transforms = []
        self._current_index = -1
        self._save_transforms()
        return """def transform(image):
    # Transform the image here
    # image: numpy array of shape (H,W) or (H,W,C)
    # return: uint8 array with 1-4 channels
    return image"""


def _validate_transform_code(code: str) -> Callable[[NDArray], NDArray]:
    """Validate and compile the transform code"""
    # Create a namespace to execute the code
    namespace = {}

    # Add required imports
    namespace.update({
        'np': np,
        'math': math
    })
    
    # Execute the transform code
    exec(code, namespace)
    
    # Validate the transform function exists
    if 'transform' not in namespace:
        raise ValueError("No transform function defined")
        
    # Try to apply it to validate it works
    test_arr: NDArray = np.zeros((10,10), dtype=np.uint8)
    transform: Callable[[NDArray], NDArray] = namespace['transform']
    result: NDArray = transform(test_arr)
    
    if not isinstance(result, np.ndarray):
        raise ValueError("Transform must return a numpy array")
    
    if result.dtype not in (np.uint8, np.float32, np.float64):
        raise ValueError("Transform must return uint8 or float32 array")
    if result.dtype in (np.float32, np.float64):
        # wrap the transform to normalize the output
        # to [0, 1] range if it returns float
        def normalize(x, f=transform):
            return (f(x) / 255).astype(np.float32)
        transform = normalize

    if len(result.shape) > 3 or (len(result.shape) == 3 and result.shape[2] > 4):
        raise ValueError("Transform must return 1-4 channel image")

    return transform

class TransformEditor:
    def __init__(self, C: dcg.Context, on_change: Callable[[Callable[[NDArray], NDArray]], None]) -> None:
        self.context: dcg.Context = C
        self._on_change: Callable[[Callable[[NDArray], NDArray]], None] = on_change

        # Add transform history
        self._history = TransformHistory()

        # Initial transform:
        self._transform: Callable[[NDArray], NDArray] = lambda image: image  # Identity transform

        # Preinitialize the transform editor popup
        self._popup = \
            dcg.Window(C, label="Transform Editor", 
                       show=False, modal=True,
                       no_open_over_existing_popup=False,
                       min_size=(600, 400), autosize=True)
        
        # Add transform navigation
        with dcg.HorizontalLayout(C, parent=self._popup, no_newline=True):
            dcg.Button(C, arrow=dcg.ButtonDirection.LEFT, callback=self._prev)
            dcg.Button(C, arrow=dcg.ButtonDirection.RIGHT, callback=self._next)
        with dcg.HorizontalLayout(C, parent=self._popup, alignment_mode=dcg.Alignment.RIGHT):
            dcg.Button(C, label="Reset", callback=self._reset)
        
        # Add editor component
        self._transform_editor = dcg.InputText(C, 
            parent=self._popup,
            multiline=True,
            width=-1, height=320,
            value=self._history.get_current() or \
    """def transform(image):
    # Transform the image here
    # image: numpy array of shape (H,W) or (H,W,C)
    # return: uint8 array with 1-4 channels
    return image""")

        # Initialize history
        if not(self._history.get_current()):
            self._history.add_transform(self._transform_editor.value)

        self._pixelwise = dcg.Checkbox(C,
            label="Pixelwise Transform",
            value=True,
            parent=self._popup)

        with dcg.HorizontalLayout(C, parent=self._popup, alignment_mode=dcg.Alignment.JUSTIFIED):
            dcg.Button(C, label="Apply", callback=self._apply)
            dcg.Button(C, label="Cancel", 
                      callback=lambda: setattr(self._popup, 'show', False))

    @property
    def pixelwise(self) -> bool:
        """Check if the transform is pixelwise"""
        return self._pixelwise.value

    def show(self) -> None:
        """Show the transform editor popup"""
        self._popup.show = True

    @property
    def transform(self) -> Callable[[NDArray], NDArray]:
        """Get the current transform function"""
        return self._transform

    def _apply(self) -> None:
        """Validate and produce the transform function"""
        try:
            code: str = self._transform_editor.value

            transform: Callable[[NDArray], NDArray] = _validate_transform_code(code)

            # If validation passed, save to history
            self._history.add_transform(code)

            # Set the transform
            self._transform = transform

            # Remove any previous error message
            self._clear_error_message()

            # Close popup
            self._popup.show = False

            # Notify the change
            self._on_change(self._transform)

        except Exception as e:
            self._show_error_message(f"{str(e)}, {traceback.format_exc()}")

    def _clear_error_message(self) -> None:
        """Remove any existing error message from the popup."""
        if isinstance(self._popup.children[-1], dcg.Text):
            self._popup.children[-1].detach_item()

    def _next(self) -> None:
        """Load next transform from history"""
        code: str | None = self._history.go_next()
        if code:
            self._transform_editor.value = code

    def _prev(self) -> None:
        """Load previous transform from history"""
        code: str | None = self._history.go_prev()
        if code:
            self._transform_editor.value = code

    def _reset(self) -> None:
        """Reset transform to default"""
        self._transform_editor.value = self._history.reset()
        self._clear_error_message()
        self._transform = lambda image: image  # Reset to identity transform
        self._on_change(self._transform)

    def _show_error_message(self, message: str) -> None:
        """Show an error message in the popup."""
        self._clear_error_message()
        with self._popup:
            dcg.Text(self.context, value=f"Error: {message}", color=(255, 100, 100))

