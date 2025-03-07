from typing import Optional, Union, Any
from PIL import Image
import numpy as np


class CannyDetectorWrapper:
    """
    A wrapper class for CannyDetector that provides edge detection functionality.
    """
    
    def __init__(
        self, 
        low_threshold: int = 100, 
        high_threshold: int = 200, 
        resolution: Optional[int] = None
    ) -> None:
        """
        Initialize the CannyDetectorWrapper.
        
        Args:
            low_threshold: Lower threshold for edge detection (default: 100)
            high_threshold: Higher threshold for edge detection (default: 200)
            resolution: Optional resolution for both detection and output (default: None)
        """
        # Import here to avoid circular imports
        from controlnet_aux import CannyDetector
        
        self.detector = CannyDetector()
        self.params = {
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "resolution": resolution,
        }
    
    def detect(
        self, 
        input_image: Union[Image.Image, np.ndarray],
        **kwargs: Any
    ) -> Image.Image:
        """
        Process an image with Canny edge detection.
        
        Args:
            input_image: Input image as PIL Image or numpy array
            **kwargs: Additional parameters to override default values
            
        Returns:
            Processed image as PIL Image
        """
        # Combine default parameters with any overrides
        params = {**self.params, **kwargs}
        
        return self.detector(
            input_image=input_image,
            low_threshold=params["low_threshold"],
            high_threshold=params["high_threshold"],
            resolution=params["resolution"],
            output_type="pil"
        )
    
    def update_params(self, **kwargs: Any) -> None:
        """
        Update detector parameters.
        
        Args:
            **kwargs: Parameters to update (low_threshold, high_threshold, resolution)
        """
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
    
    def __call__(self, input_image: Union[Image.Image, np.ndarray], **kwargs: Any) -> Image.Image:
        """
        Allow the wrapper to be called directly.
        
        Args:
            input_image: Input image as PIL Image or numpy array
            **kwargs: Additional parameters to override default values
            
        Returns:
            Processed image as PIL Image
        """
        return self.detect(input_image, **kwargs)