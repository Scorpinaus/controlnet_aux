from PIL import Image
from typing import Dict, Any

# Import the existing TilingDetector
from controlnet_aux.tiling import TilingDetector

class TilingDetectorWrapper:
    """
    A wrapper class for TilingDetector that allows configuration management and reuse.
    
    This wrapper provides methods to:
    - Initialize the detector with custom parameters
    - Process images using the current configuration
    - Update parameters without re-specifying all of them
    - Pass the detector as a callable to other components
    """
    def __init__(self, resolution: int = 1024, **kwargs):
        """
        Initialize the detector with configuration parameters.
        
        Args:
            resolution: The target resolution for image processing
            **kwargs: Additional configuration parameters
        """
        self.config: Dict[str, Any] = {"resolution": resolution, **kwargs}
        self.detector = TilingDetector()
        
    def process(self, image: Image.Image) -> Image.Image:
        """
        Process an image using the current configuration.
        
        Args:
            image: A PIL Image to process
            
        Returns:
            The processed PIL Image
        """
        return self.detector(image, **self.config)
        
    def update_config(self, **kwargs) -> 'TilingDetectorWrapper':
        """
        Update configuration parameters.
        
        Args:
            **kwargs: New configuration parameters to update
            
        Returns:
            Self for method chaining
        """
        self.config.update(kwargs)
        return self
        
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Make the wrapper callable like the original detector.
        
        Args:
            image: A PIL Image to process
            
        Returns:
            The processed PIL Image
        """
        return self.process(image)