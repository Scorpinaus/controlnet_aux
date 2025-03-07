import os
import torch
import numpy as np
from PIL import Image

class ZoeDetectorWrapper:
    """
    A wrapper for the ZoeDetector that provides configuration management
    and simplified interface.
    """
    def __init__(self, 
                 pretrained_model_or_path="zoedepth/ZoeD_M12_N",
                 model_type="zoedepth", 
                 filename=None, 
                 detect_resolution=512, 
                 image_resolution=512, 
                 gamma_corrected=False,
                 output_type="pil",
                 device=None,
                 cache_dir=None,
                 local_files_only=False):
        """
        Initialize the ZoeDetector wrapper with customizable parameters.
        
        Args:
            pretrained_model_or_path: Model path or HuggingFace model ID
            model_type: Type of model to use ('zoedepth' or 'zoedepth_nk')
            filename: Specific model filename to load
            detect_resolution: Resolution for detection
            image_resolution: Resolution for output image
            gamma_corrected: Whether to apply gamma correction
            output_type: Output format ('pil' or 'np')
            device: Device to run the model on ('cuda', 'cpu', etc.)
            cache_dir: Directory to cache downloaded models
            local_files_only: Whether to use only local files
        """
        self.config = {
            'pretrained_model_or_path': pretrained_model_or_path,
            'model_type': model_type,
            'filename': filename,
            'detect_resolution': detect_resolution,
            'image_resolution': image_resolution,
            'gamma_corrected': gamma_corrected,
            'output_type': output_type,
            'cache_dir': cache_dir,
            'local_files_only': local_files_only
        }
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = None
        self._initialize_detector()
        
    def _initialize_detector(self):
        """Initialize the ZoeDetector with current configuration."""
        from controlnet_aux import ZoeDetector
        
        self.detector = ZoeDetector.from_pretrained(
            pretrained_model_or_path=self.config['pretrained_model_or_path'],
            model_type=self.config['model_type'],
            filename=self.config['filename'],
            cache_dir=self.config['cache_dir'],
            local_files_only=self.config['local_files_only']
        )
        self.detector.to(self.device)
        
    def process_image(self, image):
        """
        Process an input image through the detector.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Depth map as PIL Image or numpy array
        """
        if self.detector is None:
            self._initialize_detector()
            
        return self.detector(
            input_image=image,
            detect_resolution=self.config['detect_resolution'],
            image_resolution=self.config['image_resolution'],
            output_type=self.config['output_type'],
            gamma_corrected=self.config['gamma_corrected']
        )
    
    def update_parameters(self, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            elif key == 'device':
                self.device = value
                if self.detector is not None:
                    self.detector.to(self.device)
    
    def __call__(self, image):
        """
        Make the wrapper callable, allowing it to be used as a function.
        
        Args:
            image: Input image
            
        Returns:
            Processed depth map
        """
        return self.process_image(image)