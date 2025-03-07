import numpy as np
from typing import Optional, Union, Tuple, Dict, Any
import torch

class MiDASWrapper:
    """
    A wrapper class for the MiDASDetector that provides convenient configuration and usage.
    """
    
    def __init__(
        self,
        model_path: str = "lllyasviel/ControlNet",
        model_type: str = "dpt_hybrid",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the MiDAS wrapper with default or specified parameters.
        
        Args:
            model_path: Path to the pretrained model or HuggingFace model identifier
            model_type: Type of the MiDAS model (e.g., "dpt_hybrid")
            device: Device to run the model on (e.g., "cuda", "cpu")
            params: Dictionary of parameters for the MiDASDetector
        """
        # Import here to avoid circular imports
        from controlnet_aux.midas import MiDaSInference, MidasDetector
        
        # Default parameters
        self.default_params = {
            "a": np.pi * 2.0,
            "bg_th": 0.1,
            "depth_and_normal": False,
            "detect_resolution": 512,
            "image_resolution": 512,
            "output_type": None
        }
        
        # Update with user-provided parameters if any
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)
        
        # Initialize the detector
        self.detector = MidasDetector.from_pretrained(
            model_path,
            model_type=model_type
        ).to(device)
        
        self.device = device
    
    def update_params(self, **kwargs) -> 'MiDASWrapper':
        """
        Update specific parameters without changing others.
        
        Args:
            **kwargs: Parameters to update
            
        Returns:
            self: For method chaining
        """
        self.params.update(kwargs)
        return self
    
    def reset_params(self) -> 'MiDASWrapper':
        """
        Reset parameters to default values.
        
        Returns:
            self: For method chaining
        """
        self.params = self.default_params.copy()
        return self
    
    def process_image(self, input_image) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Process an input image using the current parameters.
        
        Args:
            input_image: Input image (numpy array or PIL Image)
            
        Returns:
            Depth map or tuple of (depth map, normal map) if depth_and_normal is True
        """
        return self.detector(
            input_image,
            a=self.params["a"],
            bg_th=self.params["bg_th"],
            depth_and_normal=self.params["depth_and_normal"],
            detect_resolution=self.params["detect_resolution"],
            image_resolution=self.params["image_resolution"],
            output_type=self.params["output_type"]
        )
    
    def __call__(self, input_image) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make the wrapper callable, processing the image with current parameters.
        
        Args:
            input_image: Input image (numpy array or PIL Image)
            
        Returns:
            Result of process_image()
        """
        return self.process_image(input_image)
    
    def create_config(self, name: str, **params) -> Dict[str, Any]:
        """
        Create a named configuration that can be applied later.
        
        Args:
            name: Name of the configuration
            **params: Parameters for this configuration
            
        Returns:
            Configuration dictionary
        """
        config = self.params.copy()
        config.update(params)
        return {name: config}
    
    def apply_config(self, config: Dict[str, Dict[str, Any]], name: str) -> 'MiDASWrapper':
        """
        Apply a previously created configuration.
        
        Args:
            config: Configuration dictionary
            name: Name of the configuration to apply
            
        Returns:
            self: For method chaining
        """
        if name in config:
            self.params = config[name].copy()
        else:
            raise KeyError(f"Configuration '{name}' not found")
        return self
    
    @property
    def get_detector(self):
        """
        Get the underlying MiDASDetector instance.
        
        Returns:
            MiDASDetector: The detector instance
        """
        return self.detector
    
    def save_output(self, output, save_path, file_format="png"):
        """
        Save the output of the MiDAS detector to disk.
        
        Args:
            output: Output from process_image() (depth map or tuple of depth and normal map)
            save_path: Base path where to save the output (without extension)
            file_format: Format to save the image (default: png)
        """
        from PIL import Image
        import numpy as np
        
        # Check if output is a tuple (depth_map, normal_map)
        if isinstance(output, tuple) and len(output) == 2:
            depth_map, normal_map = output
            self._save_single_output(depth_map, f"{save_path}_depth.{file_format}")
            self._save_single_output(normal_map, f"{save_path}_normal.{file_format}")
        else:
            # Single output (just depth map)
            self._save_single_output(output, f"{save_path}.{file_format}")
    
    def _save_single_output(self, img, save_path):
        """
        Save a single image to disk.
        
        Args:
            img: Image to save (numpy array or PIL Image)
            save_path: Full path including filename and extension
        """
        from PIL import Image
        import numpy as np
        
        # If img is already a PIL Image
        if isinstance(img, Image.Image):
            img.save(save_path)
        # If img is a numpy array
        elif isinstance(img, np.ndarray):
            # Convert to PIL Image and save
            Image.fromarray(img).save(save_path)
        else:
            raise TypeError(f"Unsupported output type: {type(img)}")