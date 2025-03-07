# anyline_wrapper.py
import os
from typing import Dict, Union, Optional, Any, Callable
import numpy as np
from PIL import Image

from controlnet_aux.anyline import AnylineDetector


class AnylineWrapper:
    """
    A wrapper class for AnylineDetector with enhanced configuration capabilities.
    
    This wrapper allows:
    - Easy parameter management
    - Reusable configurations
    - Callable behavior for integration with other components
    - Local model path support
    """
    
    DEFAULT_PARAMS = {
        "detect_resolution": 1280,
        "guassian_sigma": 2.0,
        "intensity_threshold": 3,
        "output_type": "pil"
    }
    
    def __init__(
        self, 
        model_path: str,
        model_filename: Optional[str] = None,
        subfolder: Optional[str] = None,
        device: str = "cpu",
        **params
    ):
        """
        Initialize the AnylineWrapper with a model and parameters.
        
        Args:
            model_path: Path to the model or HuggingFace repo id
            model_filename: Specific filename if using HuggingFace or a folder
            subfolder: Optional subfolder for model files
            device: Device to run the model on ("cpu", "cuda", etc.)
            **params: Additional parameters for the detector
        """
        self.model_path = model_path
        self.model_filename = model_filename
        self.subfolder = subfolder
        self.device = device
        
        # Initialize parameters with defaults, then update with provided values
        self.params = self.DEFAULT_PARAMS.copy()
        self.params.update(params)
        
        # Initialize the detector
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize or reinitialize the AnylineDetector with current settings."""
        self.detector = AnylineDetector.from_pretrained(
            self.model_path, 
            filename=self.model_filename,
            subfolder=self.subfolder
        ).to(self.device)
    
    def update_params(self, **params):
        """
        Update specific parameters without changing others.
        
        Args:
            **params: Parameters to update
            
        Returns:
            self: For method chaining
        """
        self.params.update(params)
        return self
    
    def set_configuration(self, config: Dict[str, Any]):
        """
        Set a complete configuration from a dictionary.
        
        Args:
            config: Dictionary containing parameter settings
            
        Returns:
            self: For method chaining
        """
        self.params = config.copy()
        return self
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Dict: Current parameters
        """
        return self.params.copy()
    
    def process(self, image: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
        """
        Process an image using the current parameters.
        
        Args:
            image: Input image as numpy array or PIL Image
            
        Returns:
            Processed image (type depends on output_type parameter)
        """
        return self.detector(image, **self.params)
    
    def create_preset(self, name: str, **params):
        """
        Create a new AnylineWrapper with the same model but different parameters.
        
        Args:
            name: Name for the preset (stored as an attribute)
            **params: Parameters for the new configuration (merged with current params)
            
        Returns:
            AnylineWrapper: New wrapper instance with the specified configuration
        """
        new_params = self.params.copy()
        new_params.update(params)
        
        preset = AnylineWrapper(
            model_path=self.model_path,
            model_filename=self.model_filename,
            subfolder=self.subfolder,
            device=self.device,
            **new_params
        )
        preset.preset_name = name
        return preset
    
    def __call__(self, image: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
        """
        Make the wrapper callable for easy integration with other components.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        return self.process(image)