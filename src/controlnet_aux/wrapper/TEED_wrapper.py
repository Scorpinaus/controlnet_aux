import os
import numpy as np
from PIL import Image
from typing import Union, Dict, Optional, Any, Callable

class TEEDWrapper:
    """
    A wrapper class for TEEDdetector that provides configuration management
    and simplified usage.
    """
    
    DEFAULT_CONFIG = {
        "detect_resolution": 512,
        "safe_steps": 2,
        "output_type": "pil",
        "device": "cpu"
    }
    
    def __init__(
        self,
        model_path: str,
        filename: Optional[str] = None,
        subfolder: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the TEEDWrapper with a model path and optional configuration.
        
        Args:
            model_path: Path to a local directory or HuggingFace model
            filename: Specific model filename (required for HuggingFace models)
            subfolder: Optional subfolder within the model path
            config: Optional configuration parameters
        """
        from controlnet_aux.teed import TED
        from controlnet_aux.teed import TEEDdetector
        
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
            
        # Initialize detector
        self.detector = TEEDdetector.from_pretrained(
            pretrained_model_or_path=model_path,
            filename=filename,
            subfolder=subfolder
        )
        
        # Move to specified device
        self.detector = self.detector.to(self.config["device"])
    
    def update_config(self, **kwargs) -> "TEEDWrapper":
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Returns:
            self: The updated wrapper instance for method chaining
        """
        self.config.update(kwargs)
        
        # Handle device updates
        if "device" in kwargs:
            self.detector = self.detector.to(self.config["device"])
            
        return self
    
    def process_image(
        self,
        image: Union[np.ndarray, Image.Image],
        **kwargs
    ) -> Union[np.ndarray, Image.Image]:
        """
        Process an input image using the TEEDdetector.
        
        Args:
            image: Input image as NumPy array or PIL Image
            **kwargs: Optional parameters that override the configuration
            
        Returns:
            Processed image as NumPy array or PIL Image
        """
        # Create processing parameters by updating config with any provided kwargs
        params = self.config.copy()
        params.update(kwargs)
        
        # Process the image
        result = self.detector(
            input_image=image,
            detect_resolution=params["detect_resolution"],
            safe_steps=params["safe_steps"],
            output_type=params["output_type"]
        )
        
        return result
    
    def __call__(
        self,
        image: Union[np.ndarray, Image.Image],
        **kwargs
    ) -> Union[np.ndarray, Image.Image]:
        """
        Make the wrapper callable, equivalent to process_image.
        
        Args:
            image: Input image as NumPy array or PIL Image
            **kwargs: Optional parameters that override the configuration
            
        Returns:
            Processed image as NumPy array or PIL Image
        """
        return self.process_image(image, **kwargs)
    
    @classmethod
    def create_config(cls, **kwargs) -> Dict[str, Any]:
        """
        Create a configuration dictionary with specified parameters.
        
        Args:
            **kwargs: Configuration parameters
            
        Returns:
            Dict containing the configuration
        """
        config = cls.DEFAULT_CONFIG.copy()
        config.update(kwargs)
        return config
    
class TEEDFactory:
    """
    Factory class for creating TEEDWrapper instances with different configurations.
    """
    
    def __init__(
        self,
        model_path: str,
        filename: Optional[str] = None,
        subfolder: Optional[str] = None,
        base_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the factory with model information and base configuration.
        
        Args:
            model_path: Path to a local directory or HuggingFace model
            filename: Specific model filename (required for HuggingFace models)
            subfolder: Optional subfolder within the model path
            base_config: Optional base configuration for all detectors
        """
        self.model_path = model_path
        self.filename = filename
        self.subfolder = subfolder
        self.base_config = base_config or TEEDWrapper.DEFAULT_CONFIG.copy()
    
    def create_detector(self, **config_overrides) -> TEEDWrapper:
        """
        Create a TEEDWrapper instance with specified configuration overrides.
        
        Args:
            **config_overrides: Configuration parameters to override from base_config
            
        Returns:
            Configured TEEDWrapper instance
        """
        config = self.base_config.copy()
        config.update(config_overrides)
        
        return TEEDWrapper(
            model_path=self.model_path,
            filename=self.filename,
            subfolder=self.subfolder,
            config=config
        )
    
    def create_preset(self, preset_name: str) -> TEEDWrapper:
        """
        Create a TEEDWrapper with a predefined preset configuration.
        
        Args:
            preset_name: Name of the preset configuration
            
        Returns:
            Configured TEEDWrapper instance
        """
        presets = {
            "high_resolution": {
                "detect_resolution": 1024,
                "safe_steps": 2,
                "output_type": "pil"
            },
            "fast": {
                "detect_resolution": 384,
                "safe_steps": 1,
                "output_type": "np"
            },
            "detailed": {
                "detect_resolution": 768,
                "safe_steps": 4,
                "output_type": "pil"
            }
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {list(presets.keys())}")
        
        return self.create_detector(**presets[preset_name])
    
from typing import List, Tuple, Union, Dict, Callable
import numpy as np
from PIL import Image

def batch_process_images(
    detector: Callable,
    images: List[Union[np.ndarray, Image.Image]],
    **kwargs
) -> List[Union[np.ndarray, Image.Image]]:
    """
    Process a batch of images using a configured detector.
    
    Args:
        detector: A callable TEEDWrapper instance
        images: List of input images
        **kwargs: Optional parameters that override the detector configuration
        
    Returns:
        List of processed images
    """
    return [detector(img, **kwargs) for img in images]

def compare_detector_configs(
    image: Union[np.ndarray, Image.Image],
    configs: List[Dict[str, Any]],
    model_path: str,
    filename: Optional[str] = None,
    subfolder: Optional[str] = None
) -> Dict[str, Union[np.ndarray, Image.Image]]:
    """
    Compare different detector configurations on the same image.
    
    Args:
        image: Input image to process
        configs: List of configuration dictionaries
        model_path: Path to the model
        filename: Specific model filename
        subfolder: Optional subfolder
        
    Returns:
        Dictionary mapping configuration name/id to processed image
    """
    results = {}
    
    for i, config in enumerate(configs):
        config_name = config.pop("name", f"config_{i}")
        detector = TEEDWrapper(model_path, filename, subfolder, config)
        results[config_name] = detector(image)
    
    return results