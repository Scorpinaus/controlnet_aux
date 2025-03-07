import os
import warnings
from typing import Dict, Any, Optional, Union

import numpy as np
from PIL import Image

from controlnet_aux.pidi import PidiNetDetector


class PidiNetDetectorWrapper:
    """
    A wrapper for PidiNetDetector that provides easy configuration management
    and a callable interface for edge detection.
    """

    def __init__(
        self, 
        model_path: Optional[str] = None,
        model_name: Optional[str] = "table5_pidinet.pth",
        device: str = "cpu",
        config_name: str = "default",
        **params
    ):
        """
        Initialize an EdgeDetector with the specified model and parameters.
        
        Args:
            model_path: Path to the model file or directory containing the model.
                        If None, the model will be downloaded from the Hugging Face Hub.
            model_name: Name of the model file if model_path is a directory.
            device: Device to run the model on ('cpu', 'cuda', etc.).
            config_name: Name for this configuration.
            **params: Additional parameters for edge detection.
        """
        self.device = device
        self.configs = {}
        
        # Initialize default parameters
        self.default_params = {
            "detect_resolution": 512,
            "image_resolution": 512,
            "safe": False,
            "output_type": "pil",
            "scribble": False,
            "apply_filter": False
        }
        
        # Initialize the detector
        if model_path is not None and os.path.exists(model_path):
            if os.path.isdir(model_path):
                self.detector = PidiNetDetector.from_pretrained(
                    model_path, 
                    filename=model_name,
                    local_files_only=True
                )
            else:
                # Assume model_path is a direct path to the model file
                model_dir = os.path.dirname(model_path)
                model_file = os.path.basename(model_path)
                self.detector = PidiNetDetector.from_pretrained(
                    model_dir, 
                    filename=model_file,
                    local_files_only=True
                )
        else:
            # Download from Hugging Face Hub
            self.detector = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        
        # Move to device
        self.detector.to(device)
        
        # Create the initial configuration
        self.add_config(config_name, **params)
        self.current_config = config_name

    def add_config(self, config_name: str, **params) -> None:
        """
        Add a new named configuration with the specified parameters.
        
        Args:
            config_name: Name for this configuration.
            **params: Parameters for edge detection with this configuration.
        """
        # Merge with default parameters
        config_params = self.default_params.copy()
        config_params.update(params)
        self.configs[config_name] = config_params
    
    def process(
        self, 
        image: Union[np.ndarray, Image.Image], 
        config_name: Optional[str] = None, 
        **override_params
    ) -> Union[np.ndarray, Image.Image]:
        """
        Process an image with the specified configuration.
        
        Args:
            image: The input image to process.
            config_name: Name of the configuration to use. If None, use the current configuration.
            **override_params: Parameters to override for this specific call.
            
        Returns:
            The detected edges as a PIL Image or numpy array.
        """
        # Determine which configuration to use
        if config_name is not None:
            if config_name not in self.configs:
                raise ValueError(f"Configuration '{config_name}' does not exist")
            params = self.configs[config_name].copy()
        else:
            params = self.configs[self.current_config].copy()
        
        # Apply any parameter overrides
        params.update(override_params)
        
        # Process the image
        return self.detector(image, **params)
    
    def update_params(self, config_name: Optional[str] = None, **params) -> None:
        """
        Update parameters for the specified configuration.
        
        Args:
            config_name: Name of the configuration to update. If None, update the current configuration.
            **params: Parameters to update.
        """
        target_config = config_name if config_name is not None else self.current_config
        if target_config not in self.configs:
            raise ValueError(f"Configuration '{target_config}' does not exist")
        
        self.configs[target_config].update(params)
    
    def set_current_config(self, config_name: str) -> None:
        """
        Set the current configuration.
        
        Args:
            config_name: Name of the configuration to set as current.
        """
        if config_name not in self.configs:
            raise ValueError(f"Configuration '{config_name}' does not exist")
        
        self.current_config = config_name

    def __call__(
        self, 
        image: Union[np.ndarray, Image.Image], 
        config_name: Optional[str] = None, 
        **override_params
    ) -> Union[np.ndarray, Image.Image]:
        """
        Process an image with the specified configuration.
        
        Args:
            image: The input image to process.
            config_name: Name of the configuration to use. If None, use the current configuration.
            **override_params: Parameters to override for this specific call.
            
        Returns:
            The detected edges as a PIL Image or numpy array.
        """
        return self.process(image, config_name, **override_params)
    
    def get_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all configurations.
        
        Returns:
            A dictionary mapping configuration names to parameter dictionaries.
        """
        return self.configs.copy()
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get the current configuration parameters.
        
        Returns:
            The current configuration parameters.
        """
        return self.configs[self.current_config].copy()