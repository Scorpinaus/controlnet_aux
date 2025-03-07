import os
import numpy as np
import torch
from typing import Dict, Union, Optional, Any, Callable

from PIL import Image
from controlnet_aux.mlsd import MLSDdetector


class MLSDWrapper:
    """
    A wrapper class for MLSDdetector that provides parameter management,
    configuration profiles, and a callable interface.
    """
    
    DEFAULT_PARAMS = {
        "thr_v": 0.1,              # Value threshold
        "thr_d": 0.1,              # Distance threshold
        "detect_resolution": 512,  # Resolution for detection
        "image_resolution": 512,   # Resolution for output
        "output_type": "pil"       # Output type ("pil" or "np")
    }
    
    # Predefined configuration profiles
    CONFIGS = {
        "default": DEFAULT_PARAMS,
        "high_sensitivity": {"thr_v": 0.05, "thr_d": 0.05},
        "low_sensitivity": {"thr_v": 0.2, "thr_d": 0.2},
        "high_res": {"detect_resolution": 1024, "image_resolution": 1024},
    }
    
    def __init__(
        self,
        model_path: str = "lllyasviel/ControlNet",
        filename: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config: str = "default",
        **kwargs
    ):
        """
        Initialize the MLSDWrapper with the specified parameters.
        
        Args:
            model_path: Path to the model or HuggingFace repo ID
            filename: Specific model filename (optional)
            device: Device to use for inference ("cuda", "cpu", etc.)
            config: Configuration profile name or "default"
            **kwargs: Override parameters for the selected configuration
        """
        self.model_path = model_path
        self.filename = filename
        self.device = device
        self._detector = None
        
        # Start with default parameters
        self.params = self.DEFAULT_PARAMS.copy()
        
        # Apply configuration profile if specified
        if config in self.CONFIGS and config != "default":
            self.params.update(self.CONFIGS[config])
        
        # Override with any provided parameters
        self.params.update(kwargs)
    
    @property
    def detector(self) -> MLSDdetector:
        """
        Lazily initialize and return the detector.
        """
        if self._detector is None:
            self._detector = MLSDdetector.from_pretrained(
                self.model_path, 
                filename=self.filename,
                local_files_only=os.path.exists(self.model_path)
            ).to(self.device)
        return self._detector
    
    def update_params(self, **kwargs) -> None:
        """
        Update detector parameters.
        
        Args:
            **kwargs: Parameters to update
        """
        self.params.update(kwargs)
    
    def apply_config(self, config_name: str) -> None:
        """
        Apply a predefined configuration profile.
        
        Args:
            config_name: Name of the configuration profile
        """
        if config_name in self.CONFIGS:
            # Start with default parameters
            self.params = self.DEFAULT_PARAMS.copy()
            # Apply the specified configuration
            self.params.update(self.CONFIGS[config_name])
        else:
            raise ValueError(f"Unknown configuration profile: {config_name}")
    
    def process_image(
        self, 
        image: Union[np.ndarray, Image.Image],
        **override_params
    ) -> Union[np.ndarray, Image.Image]:
        """
        Process an image with the detector.
        
        Args:
            image: Input image (numpy array or PIL Image)
            **override_params: Parameters to override for this call only
            
        Returns:
            Processed image as numpy array or PIL Image based on output_type
        """
        # Combine current parameters with overrides
        params = {**self.params, **override_params}
        return self.detector(image, **params)
    
    def __call__(
        self, 
        image: Union[np.ndarray, Image.Image],
        **override_params
    ) -> Union[np.ndarray, Image.Image]:
        """
        Make the wrapper callable, delegating to process_image.
        
        Args:
            image: Input image (numpy array or PIL Image)
            **override_params: Parameters to override for this call only
            
        Returns:
            Processed image as numpy array or PIL Image based on output_type
        """
        return self.process_image(image, **override_params)
    
    @classmethod
    def create_config(cls, name: str, params: Dict[str, Any]) -> None:
        """
        Create a new configuration profile.
        
        Args:
            name: Name of the configuration profile
            params: Parameters for the configuration
        """
        cls.CONFIGS[name] = params

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