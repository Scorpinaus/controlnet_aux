import os
import numpy as np
from typing import Dict, Any, Optional, Union
from PIL import Image

class LeresDetectorWrapper:
   """
   A wrapper for the LeresDetector that provides parameter management and configuration presets.
   """
   
   DEFAULT_PARAMS = {
       "thr_a": 0,
       "thr_b": 0,
       "boost": False,
       "detect_resolution": 512,
       "image_resolution": 512,
       "output_type": "pil"
   }
   
   def __init__(
       self, 
       pretrained_model_or_path: str, 
       filename: Optional[str] = None,
       pix2pix_filename: Optional[str] = None,
       cache_dir: Optional[str] = None,
       local_files_only: bool = False,
       device: str = "cpu",
       **kwargs
   ):
       """
       Initialize the LeresDetector wrapper.
       
       Args:
           pretrained_model_or_path: Path to the model or HuggingFace model name
           filename: Main model filename (defaults to "res101.pth")
           pix2pix_filename: Pix2pix model filename (defaults to "latest_net_G.pth")
           cache_dir: Directory to cache downloaded models
           local_files_only: Whether to only use local files
           device: Device to load the model on ('cpu' or 'cuda')
           **kwargs: Initial parameter values that override the defaults
       """
       # Import here to avoid circular imports
       from controlnet_aux.leres import LeresDetector
       
       # Initialize the detector
       self.detector = LeresDetector.from_pretrained(
           pretrained_model_or_path=pretrained_model_or_path,
           filename=filename,
           pix2pix_filename=pix2pix_filename,
           cache_dir=cache_dir,
           local_files_only=local_files_only
       )
       
       # Move to specified device
       self.detector = self.detector.to(device)
       
       # Store current parameters
       self.params = self.DEFAULT_PARAMS.copy()
       self.update_params(**kwargs)
       
       # Store named configurations
       self.saved_configs = {}
   
   def update_params(self, **kwargs) -> None:
       """
       Update processing parameters.
       
       Args:
           **kwargs: Parameters to update
       """
       for key, value in kwargs.items():
           if key in self.params:
               self.params[key] = value
           else:
               raise ValueError(f"Unknown parameter: {key}")
   
   def save_config(self, name: str) -> None:
       """
       Save the current configuration under a name.
       
       Args:
           name: Name to save the configuration under
       """
       self.saved_configs[name] = self.params.copy()
   
   def load_config(self, name: str) -> None:
       """
       Load a saved configuration.
       
       Args:
           name: Name of the configuration to load
       """
       if name in self.saved_configs:
           self.params = self.saved_configs[name].copy()
       else:
           raise ValueError(f"No configuration saved under name: {name}")
   
   def process_image(self, image: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
       """
       Process an image with the current parameters.
       
       Args:
           image: Input image (numpy array or PIL Image)
           
       Returns:
           Processed depth map (format depends on output_type parameter)
       """
       return self.detector(
           input_image=image,
           **self.params
       )
   
   def __call__(self, image: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
       """
       Make the wrapper callable, processing images with current parameters.
       
       Args:
           image: Input image
           
       Returns:
           Processed depth map
       """
       return self.process_image(image)
   
   @property
   def available_configs(self) -> list:
       """
       Get a list of available configuration names.
       
       Returns:
           List of configuration names
       """
       return list(self.saved_configs.keys())
   
   def create_preset_configs(self) -> None:
       """
       Create some useful preset configurations.
       """
       # High detail depth map
       self.params = self.DEFAULT_PARAMS.copy()
       self.update_params(boost=True, detect_resolution=768)
       self.save_config("high_detail")
       
       # Faster processing
       self.params = self.DEFAULT_PARAMS.copy()
       self.update_params(detect_resolution=384, image_resolution=384)
       self.save_config("faster")
       
       # Remove background
       self.params = self.DEFAULT_PARAMS.copy()
       self.update_params(thr_a=10, thr_b=80)
       self.save_config("remove_background")
       
       # Reset to default
       self.params = self.DEFAULT_PARAMS.copy()