import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Union, Tuple

# Import the ContentShuffleDetector from the module
from controlnet_aux.shuffle import ContentShuffleDetector

class ContentShuffleWrapper:
    """
    A wrapper class for ContentShuffleDetector that provides configuration management
    and a consistent interface.
    """
    
    def __init__(self, 
                 detect_resolution: int = 512,
                 image_resolution: int = 512,
                 output_type: str = "pil",
                 h: Optional[int] = None,
                 w: Optional[int] = None,
                 f: Optional[int] = None):
        """
        Initialize the ContentShuffleWrapper with configuration parameters.
        
        Args:
            detect_resolution: Resolution used for detection
            image_resolution: Resolution of the output image
            output_type: Type of output ("pil" for PIL Image or "np" for numpy array)
            h: Height parameter for noise disk (defaults to input image height)
            w: Width parameter for noise disk (defaults to input image width)
            f: Frequency parameter for noise disk (defaults to 256)
        """
        self.config = {
            "detect_resolution": detect_resolution,
            "image_resolution": image_resolution,
            "output_type": output_type,
            "h": h,
            "w": w,
            "f": f
        }
        self.detector = ContentShuffleDetector()
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self.config.update(kwargs)
    
    def process_image(self, 
                      input_image: Union[np.ndarray, Image.Image], 
                      **kwargs) -> Union[np.ndarray, Image.Image]:
        """
        Process an input image using the detector with current configuration.
        
        Args:
            input_image: Input image (PIL Image or numpy array)
            **kwargs: Optional override parameters
        
        Returns:
            Processed image according to the output_type configuration
        """
        # Combine stored config with any override parameters
        params = {**self.config, **kwargs}
        return self.detector(input_image, **params)
    
    def __call__(self, 
                input_image: Union[np.ndarray, Image.Image], 
                **kwargs) -> Union[np.ndarray, Image.Image]:
        """
        Make the wrapper callable, equivalent to process_image.
        
        Args:
            input_image: Input image (PIL Image or numpy array)
            **kwargs: Optional override parameters
        
        Returns:
            Processed image according to the output_type configuration
        """
        return self.process_image(input_image, **kwargs)
    
    def create_preset(self, preset_name: str, **kwargs) -> 'ContentShuffleWrapper':
        """
        Create a new wrapper instance with a preset configuration.
        
        Args:
            preset_name: Name for the preset configuration (for documentation)
            **kwargs: Configuration parameters for this preset
        
        Returns:
            A new ContentShuffleWrapper instance with the preset configuration
        """
        # Create a new instance with current config
        preset = ContentShuffleWrapper(**self.config)
        # Update with the preset-specific parameters
        preset.update_config(**kwargs)
        return preset