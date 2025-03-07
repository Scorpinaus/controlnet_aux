import os
import numpy as np
import torch
from PIL import Image
from typing import Dict, Any, Optional, Union, Tuple

class NormalBaeWrapper:
    """
    A wrapper for the NormalBaeDetector that allows for easy configuration and reuse.
    """
    def __init__(
        self,
        model_path: str = "lllyasviel/Annotators",
        model_filename: str = "scannet.pt",
        detect_resolution: int = 512,
        image_resolution: int = 512,
        output_type: str = "pil",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Initialize the NormalBaeWrapper with customizable parameters.
        
        Args:
            model_path: Path to the model directory or HuggingFace repo ID
            model_filename: Name of the model file
            detect_resolution: Resolution for detection
            image_resolution: Resolution for output image
            output_type: Type of output (pil, np)
            device: Device to run the model on (cuda, cpu)
            **kwargs: Additional parameters to pass to the detector
        """
        self.params = {
            "detect_resolution": detect_resolution,
            "image_resolution": image_resolution,
            "output_type": output_type,
            **kwargs
        }
        
        self.model_path = model_path
        self.model_filename = model_filename
        self.device = device
        self.detector = None
        
        # Initialize the detector
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize or reinitialize the detector with current parameters."""
        from controlnet_aux.normalbae import NormalBaeDetector
        
        self.detector = NormalBaeDetector.from_pretrained(
            self.model_path,
            filename=self.model_filename,
            local_files_only=os.path.isdir(self.model_path)
        )
        self.detector.to(self.device)
    
    def update_params(self, **kwargs):
        """
        Update processing parameters.
        
        Args:
            **kwargs: Parameters to update
        """
        self.params.update(kwargs)
        
    def update_model(self, model_path=None, model_filename=None, device=None):
        """
        Update model parameters and reinitialize the detector.
        
        Args:
            model_path: New model path
            model_filename: New model filename
            device: New device
        """
        if model_path is not None:
            self.model_path = model_path
        if model_filename is not None:
            self.model_filename = model_filename
        if device is not None:
            self.device = device
        
        self._initialize_detector()
    
    def __call__(
        self, 
        image: Union[np.ndarray, Image.Image], 
        **kwargs
    ) -> Union[np.ndarray, Image.Image]:
        """
        Process an input image with the current parameters.
        
        Args:
            image: Input image (numpy array or PIL Image)
            **kwargs: Optional parameters to override for this call
        
        Returns:
            Processed normal map image
        """
        # Combine stored parameters with any call-specific overrides
        call_params = {**self.params, **kwargs}
        
        # Call the detector with the parameters
        return self.detector(image, **call_params)
    
    @classmethod
    def with_preset(cls, preset_name: str, **kwargs):
        """
        Create a new wrapper with a predefined preset configuration.
        
        Args:
            preset_name: Name of the preset to use
            **kwargs: Additional parameters to override the preset
        
        Returns:
            Configured NormalBaeWrapper instance
        """
        presets = {
            "high_quality": {
                "detect_resolution": 1024,
                "image_resolution": 1024,
                "output_type": "pil"
            },
            "balanced": {
                "detect_resolution": 512,
                "image_resolution": 512,
                "output_type": "pil"
            },
            "fast": {
                "detect_resolution": 384,
                "image_resolution": 384,
                "output_type": "pil"
            }
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(presets.keys())}")
        
        # Combine preset with any additional parameters
        preset_params = {**presets[preset_name], **kwargs}
        
        return cls(**preset_params)
    
    def create_preset(self, preset_name: str) -> Dict[str, Any]:
        """
        Create a reusable configuration preset from the current parameters.
        
        Args:
            preset_name: Name to identify this preset
        
        Returns:
            Dictionary containing the current configuration
        """
        preset = {
            "model_path": self.model_path,
            "model_filename": self.model_filename,
            "device": self.device,
            **self.params
        }
        
        # Store in class-level dictionary for future use
        if not hasattr(NormalBaeWrapper, "_presets"):
            NormalBaeWrapper._presets = {}
        
        NormalBaeWrapper._presets[preset_name] = preset
        return preset

    @classmethod
    def from_preset(cls, preset_name: str, **kwargs):
        """
        Create a new wrapper instance from a previously saved preset.
        
        Args:
            preset_name: Name of the saved preset
            **kwargs: Additional parameters to override from the preset
        
        Returns:
            New NormalBaeWrapper instance with preset configuration
        """
        if not hasattr(cls, "_presets") or preset_name not in cls._presets:
            raise ValueError(f"Preset '{preset_name}' not found")
        
        # Combine preset with overrides
        config = {**cls._presets[preset_name], **kwargs}
        return cls(**config)

    def get_current_config(self) -> Dict[str, Any]:
        """Get the current configuration parameters."""
        return {
            "model_path": self.model_path,
            "model_filename": self.model_filename,
            "device": self.device,
            **self.params
        }

    def process_batch(self, images: list, **kwargs) -> list:
        """
        Process a batch of images with current parameters.
        
        Args:
            images: List of input images
            **kwargs: Optional parameters to override for this batch
        
        Returns:
            List of processed normal map images
        """
        return [self(img, **kwargs) for img in images]