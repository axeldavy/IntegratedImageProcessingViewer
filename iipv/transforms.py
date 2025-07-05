
from appdirs import user_config_dir
from collections.abc import Callable
import colorsys
import dearcygui as dcg
import json
import math
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import traceback

_PRESETS = {
    "Identity": """def transform(image):
    # No changes - returns the original image
    return image""",
    
    "Grayscale": """def transform(image):
    # Convert image to grayscale
    if len(image.shape) == 3 and image.shape[2] >= 3:
        return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return image""",

    "Optical flow": """def transform(image):
    # Ensure we're working with at least a 2-channel image
    if len(image.shape) < 3 or image.shape[2] < 2:
        # Can't visualize flow without x and y components
        return image
    
    # Extract x and y flow components
    flow_x = image[:,:,0].astype(np.float32)
    flow_y = image[:,:,1].astype(np.float32)
    
    # Calculate angle in degrees [0, 360]
    angles = np.mod(np.degrees(np.arctan2(-flow_x, flow_y)) + 180, 360)
    
    # Calculate magnitude (radius)
    magnitudes = np.sqrt(np.square(flow_x) + np.square(flow_y))
    scale_factor = 1.0
    magnitudes = np.clip(magnitudes * scale_factor, 0, 1)
    
    # Create HSV representation
    hsv = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
    hsv[:,:,0] = (angles / 2).astype(np.uint8)  # OpenCV uses 0-180 for hue
    hsv[:,:,1] = (magnitudes * 255).astype(np.uint8)  # 0-255 for saturation
    hsv[:,:,2] = (magnitudes * 255).astype(np.uint8)  # 0-255 for value
    
    # Convert HSV to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb""",

    "Jet": """def transform(image):
    # Extract single channel (using first channel or grayscale)
    if len(image.shape) == 3 and image.shape[2] >= 3:
        gray = image[:,:,0].astype(np.float32) / 255.0
    else:
        gray = image.astype(np.float32) / 255.0
    
    # Clip slightly outside [0, 1]
    gray = np.clip(gray, -0.05, 1.05)
    
    # Shrink
    gray = gray / 1.15 + 0.1
    
    # Create RGB channels based on the jet mapping
    rgb = np.zeros((*gray.shape, 3), dtype=np.float32)
    
    # Calculate the RGB components
    rgb[:,:,0] = 1.5 - np.abs(gray - 0.75) * 4.0  # Red
    rgb[:,:,1] = 1.5 - np.abs(gray - 0.50) * 4.0  # Green
    rgb[:,:,2] = 1.5 - np.abs(gray - 0.25) * 4.0  # Blue
    
    # Clip values to [0, 1] range
    rgb = np.clip(rgb, 0, 1)
    
    # Convert to 8-bit
    rgb = (rgb * 255).astype(np.uint8)
    
    return rgb""",

    "Sentinel 1": """def transform(image):
    # Create color mapping effect for Sentinel-1 images
    
    # Ensure we have an RGB image
    if len(image.shape) < 3 or image.shape[2] < 3:
        if len(image.shape) < 3:
            # Convert grayscale to RGB
            image = np.stack([image, image, image], axis=2)
        else:
            rgb = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
            for i in range(min(3, image.shape[2])):
                rgb[:,:,i] = image[:,:,i]
            image = rgb
    
    # Extract RGB components as float
    img_float = image[:,:,:3].astype(np.float32)
    
    # Calculate distance for each pixel (vectorized)
    distances = np.sqrt(np.sum(np.square(img_float), axis=2))
    
    # Calculate hue based on distance
    hues = np.minimum(np.power(distances / 100.0, 1.4) * 30.0, 300.0)
    
    # Apply distance-based conditions to hue
    mask1 = distances < 300.0
    mask2 = (distances >= 300.0) & (distances < 600.0)
    mask3 = (distances >= 600.0) & (distances < 900.0)
    mask4 = distances >= 900.0
    
    hues[mask1] = distances[mask1] / 300.0 * 90.0
    hues[mask2] = (distances[mask2] - 300.0) / 300.0 * 100.0 + 90.0
    hues[mask3] = (distances[mask3] - 600.0) / 300.0 * 50.0 + 190.0
    hues[mask4] = (distances[mask4] - 900.0) / 900.0 * 30.0 + 240.0
    
    hues = np.minimum(hues, 270.0)
    hues = (hues + 280.0) % 360.0
    
    # Calculate lightness/saturation value
    lightness = np.minimum(np.sqrt(distances) / np.sqrt(2000.0), 1.0)
    
    # Convert HSV to RGB using OpenCV (much faster)
    hsv = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
    hsv[:,:,0] = (hues / 2).astype(np.uint8)  # OpenCV uses 0-180 for hue
    hsv[:,:,1] = (lightness * 255).astype(np.uint8)  # 0-255 for saturation
    hsv[:,:,2] = (lightness * 255).astype(np.uint8)  # 0-255 for value
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb""",
    
    "Invert": """def transform(image):
    # Invert the colors of the image
    return 255 - image""",
    
    "Brightness +50": """def transform(image):
    # Increase brightness by 50
    return np.clip(image.astype(np.int16) + 50, 0, 255).astype(np.uint8)""",
    
    "Contrast Boost": """def transform(image):
    # Enhance contrast
    mean = np.mean(image)
    return np.clip((image.astype(np.float32) - mean) * 1.5 + mean, 0, 255).astype(np.uint8)""",
    
    "Gaussian Blur": """def transform(image):
    return cv2.GaussianBlur(image, (5, 5), 1.0)""",
            
    "Edge Detection": """def transform(image):
    if len(image.shape) == 3 and image.shape[2] >= 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Sobel operators using OpenCV
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Calculate magnitude
    edges = cv2.magnitude(grad_x, grad_y)
    return cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)""",
    
    "Threshold": """def transform(image):
    # Simple threshold operation
    if len(image.shape) == 3 and image.shape[2] >= 3:
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    else:
        gray = image
        
    threshold = 128
    return (gray > threshold) * 255.""",
    
    "Sepia": """def transform(image):
    # Apply sepia filter
    if len(image.shape) < 3 or image.shape[2] < 3:
        # Convert grayscale to RGB first
        if len(image.shape) < 3:
            rgb = np.stack((image,)*3, axis=-1)
        else:
            rgb = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
            rgb[:,:,0] = image[:,:,0]
            rgb[:,:,1] = image[:,:,0]
            rgb[:,:,2] = image[:,:,0]
    else:
        rgb = image[:,:,:3]
    
    # Sepia matrix
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    
    # Apply transformation
    sepia_img = np.zeros_like(rgb)
    for i in range(3):
        sepia_img[:,:,i] = np.clip(
            rgb[:,:,0] * sepia_matrix[i,0] +
            rgb[:,:,1] * sepia_matrix[i,1] +
            rgb[:,:,2] * sepia_matrix[i,2],
            0, 255
        ).astype(np.uint8)
    
    return sepia_img"""
        }

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
        'colorsys': colorsys,
        'np': np,
        'math': math
    })
    try:
        import cv2
        namespace['cv2'] = cv2
    except ImportError:
        pass
    
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

        # Keep track of whether current editor content is from a preset
        self._current_is_preset = False
        self._last_user_content = None  # Stores last non-preset content

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
            self._preset_combo = dcg.Combo(
                C,
                label="Presets",
                items=list(_PRESETS.keys()),
                callback=self._on_preset_selected,
                width=200
            )
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
        # Reset preset combo selection when showing the popup
        self._preset_combo.value = ""
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
            if not self._current_is_preset:
                self._history.add_transform(code)

            # After applying, it's no longer considered a preset
            self._current_is_preset = False

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
        # If current content is a preset, restore the last user content first
        if self._current_is_preset and self._last_user_content:
            self._transform_editor.value = self._last_user_content
            self._current_is_preset = False
            return

        code: str | None = self._history.go_next()
        if code:
            self._transform_editor.value = code
            self._current_is_preset = False

    def _on_preset_selected(self) -> None:
        """Handle preset selection from the combo box"""
        selected = self._preset_combo.value
        
        # Save current content if it's not a preset
        if not self._current_is_preset and self._transform_editor.value:
            self._last_user_content = self._transform_editor.value
            
        # Set the editor content to the selected preset
        if selected in _PRESETS:
            self._transform_editor.value = _PRESETS[selected]
            self._current_is_preset = True

    def _prev(self) -> None:
        """Load previous transform from history"""
        # If current content is a preset, restore the last user content first
        if self._current_is_preset and self._last_user_content:
            self._transform_editor.value = self._last_user_content
            self._current_is_preset = False
            return
            
        code: str | None = self._history.go_prev()
        if code:
            self._transform_editor.value = code
            self._current_is_preset = False

    def _reset(self) -> None:
        """Reset transform to default"""
        self._transform_editor.value = self._history.reset()
        self._current_is_preset = False
        self._clear_error_message()
        self._transform = lambda image: image  # Reset to identity transform
        self._on_change(self._transform)

    def _show_error_message(self, message: str) -> None:
        """Show an error message in the popup."""
        self._clear_error_message()
        with self._popup:
            dcg.Text(self.context, value=f"Error: {message}", color=(255, 100, 100))

